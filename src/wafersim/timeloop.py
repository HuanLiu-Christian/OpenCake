"""TimeLoop input generation and result-to-contract adaptation.

This module drives real ``timeloop-mapper``/``timeloop-model`` binaries
inside Docker. There is no synthetic/offline fallback: if Docker or either
binary is unavailable, or produces no usable output, this raises instead of
substituting fabricated results.

Two ways this module's output gets used (see cli.py's ``--mode``):
  - ``timeloop-mapper`` (run_timeloop_mapper): a single search over the
    constraints-pruned mapspace; TimeLoop itself picks tile_k by its own
    delay/energy metric, blind to inter-PE network cost.
  - ``timeloop-model`` (run_timeloop_model), called once per SRAM-feasible
    tile_k candidate by ``search.search_tile_k``: a real, deterministic,
    single-point evaluation, used so OpenCake can pick the tile_k that
    actually minimizes full-system (compute + network) cost -- see
    search.py's module docstring for the full rationale.

SCHEMA NOTE (learned from an actual failed run against
`timeloopaccelergy/accelergy-timeloop-infrastructure:latest`): the
`timeloop-mapper` binary in that image is the classic/native parser, NOT
the newer "timeloopfe" v0.4 front-end. It rejects the `nodes:` /
`!Component`/`!Container` YAML-tag architecture format documented at
https://timeloop.csail.mit.edu/v4/input-formats/design/architecture with
"'nodes' found as a sub-key in the architecture ... will not be recognized
by 'timeloop-...' commands". This module therefore uses the older native
`subtree`/`local` architecture format instead (confirmed against a real,
complete Eyeriss example at
https://timeloop.csail.mit.edu/previous_versions/timeloop-accelergy-v3/examples/full-design-examples/eyeriss),
and splits mapping constraints into `architecture_constraints.yaml` /
`mapspace_constraints.yaml` (also confirmed against a real working example)
rather than the unified v0.4 `constraints:` key.

Confirmed against real runs of `timeloop-mapper` inside
`timeloopaccelergy/accelergy-timeloop-infrastructure:latest`: the
problem/architecture/constraints/mapper YAML above are all accepted
(only non-fatal hyphenated-key deprecation warnings). Getting a non-empty
energy reference table out of accelergy additionally requires telling it
what actions our custom SRAM/DRAM/intmac classes support -- see
primitive_components.yaml/accelergy_config.yaml below (confirmed by
reading accelergy's own source: an undeclared class silently gets
`actions: []`, so its per-action energy loop just does nothing, no error).
If the mapper rejects something else, read `timeloop.log` under the
output directory and iterate on component classes/attributes.

Design choice: architecture_constraints.yaml pins only the spatial M/N
fanout onto the physical PE mesh (that part isn't a search question -- it's
determined by the mesh geometry). mapspace_constraints.yaml leaves the
temporal K dimension (the intra-PE tile_k / k_phases split) completely
open: timeloop-mapper genuinely searches it, bounded only by the per-PE
SRAM `depth` declared in architecture.yaml (see the halving comment in
generate_timeloop_inputs() for how that depth is derived from this
project's real double-buffering requirement). active_rows/active_cols/
tile_m/tile_n/tile_k are therefore parsed back out of the mapper's own real
`.map.yaml` search result (see `_extract_tiling_from_map_yaml`), NOT from
`mapping_plan.json` -- that file (and mapping.yaml) still exist only to
drive the optional single-point `run_timeloop_model` cross-check, and are
no longer consulted for the contract TimeLoop actually decides. Cycles/
Energy are parsed back out of the mapper's real `stats.txt`, from its
"Summary Stats" section.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import json
import re
import subprocess

from .contract import GemmMappingContract
from .specs import GemmSpec, WaferSpec

MAPPER_MODE = "docker-timeloop-mapper"
MODEL_MODE = "docker-timeloop-model"

# Root cause of the long-chased "empty ERT.yaml" failure (confirmed by
# reading accelergy's actual source, not the image): a class accelergy
# hasn't been told about (via a "primitive component class" declaration
# listing its actions) silently gets `actions: []`, so the per-action
# energy-table loop just does nothing for it -- no error, no log line,
# just an empty result. It has nothing to do with the
# `/usr/local/src/accelergy_default_config.yaml` "cp" error chased
# earlier (that string doesn't appear anywhere in accelergy's source; the
# stub-mount workaround for it was a red herring that happened to
# coincide with unrelated progress). The real fix is
# primitive_components.yaml below, declaring actions for our custom
# SRAM/DRAM/intmac classes that match what the CactiSRAM/CactiDRAM/
# Aladdin_table plugins actually support.

_ENERGY_UNIT_TO_UJ = {
    "pj": 1e-6,
    "nj": 1e-3,
    "uj": 1.0,
    "µj": 1.0,
    "mj": 1e3,
    "j": 1e6,
}

# CONFIRMED REAL FAILURE MODE (2026-08-23, against a real
# `outputs/area_budget_check` Docker run): every SRAM-class component this
# project declares to Accelergy (pe_scratchpad, fifo_buffer, router_buffer
# -- see architecture.yaml below) is estimated by the real
# accelergy-cacti-plug-in (cacti_wrapper.py's CactiMemory._interp_size),
# which computes a CACTI "cache_size" as:
#   scaled_width = max(ceil(width_bits/8)*8, 32)   # width_bits = word-bits*block-size
#   cache_size   = scaled_width * depth_words // 8
# Past some real cache_size (bytes) threshold, CACTI's own C/C++ integer
# handling breaks down and Accelergy silently returns a near-zero
# placeholder area for that component instead of erroring -- confirmed by
# comparing two real runs' actual accelergy.log/ART.yaml: fifo_buffer at
# fifo_capacity_packets=8 (cache_size=65536) correctly produced a real
# 59674.0 um^2, while fifo_buffer at fifo_capacity_packets=524288
# (cache_size=4_294_967_296, exactly 2**32) never even shows a matching
# "Calling CACTI with cache_size=..." line in the real accelergy log, and
# the final reported area is a nonsensical 1.0 um^2.
#
# SAFE_CACTI_CACHE_SIZE_BYTES is a REASONED, NOT independently confirmed at
# this exact value, conservative ceiling: 2**24 (16,777,216) -- 128x below
# the 2**31 boundary a 32-bit *signed* C `int` would overflow at (the most
# likely real cause, since cacti_wrapper.py writes cache_size as plain text
# into a CACTI config file for CACTI's own C++ binary to parse), and 256x
# below the confirmed-broken 2**32. If a future real Docker run confirms a
# larger cache_size still works, this can be raised with that run's
# evidence cited here -- until then, every SRAM-class component this
# project declares is validated against it BEFORE any real Docker call is
# made (see _validate_cacti_safe_sram below), so an unsafe declaration
# fails fast and loud in plain Python instead of silently producing a
# wrong-but-plausible-looking real Docker result. area_budget.py's own
# search ceilings (_SRAM_KIB_PER_PE_MAX/_FIFO_CAPACITY_PACKETS_MAX) are
# derived from this SAME constant, so both the automatic area-budget search
# and any direct --sram-kib/--fifo-packets CLI usage are covered by one
# single source of truth.
SAFE_CACTI_CACHE_SIZE_BYTES = 1 << 24


def _cacti_cache_size_bytes(word_bits: int, block_size_bytes: int, depth_words: int) -> int:
    """Mirrors accelergy-cacti-plug-in's cacti_wrapper.py
    (CactiMemory._interp_size) cache_size computation EXACTLY -- see
    SAFE_CACTI_CACHE_SIZE_BYTES above for the real data points this was
    verified against. Kept in lockstep with the word-bits=8/block-size
    convention each SRAM-class component below is actually declared with."""
    width_bits = word_bits * block_size_bytes
    scaled_width_bits = max(-(-width_bits // 8) * 8, 32)  # ceil(width_bits/8)*8, floored at 32
    return scaled_width_bits * depth_words // 8


def _validate_cacti_safe_sram(
    component_name: str, *, word_bits: int, block_size_bytes: int, depth_words: int
) -> None:
    """Raises ValueError BEFORE any real Docker call if `component_name`'s
    declared depth would imply a real CACTI cache_size past
    SAFE_CACTI_CACHE_SIZE_BYTES -- see that constant's docstring for why
    this is necessary: past that point, real CACTI/Accelergy does not
    error, it silently returns a wrong-but-plausible-looking near-zero
    area, which nothing downstream (contract.py's validate(), this
    project's binary searches) can otherwise detect on its own."""
    cache_size = _cacti_cache_size_bytes(word_bits, block_size_bytes, depth_words)
    if cache_size > SAFE_CACTI_CACHE_SIZE_BYTES:
        raise ValueError(
            f"{component_name}'s declared depth ({depth_words} words, "
            f"block-size {block_size_bytes} bytes, word-bits {word_bits}) implies "
            f"a real CACTI cache_size of {cache_size} bytes, above the "
            f"confirmed-safe ceiling of {SAFE_CACTI_CACHE_SIZE_BYTES} bytes (see "
            "timeloop.py's SAFE_CACTI_CACHE_SIZE_BYTES comment) -- a real Docker "
            "run confirmed CACTI silently returns a nonsensical near-zero area "
            "past this range instead of erroring, so this is rejected up front "
            "rather than risking a silently-wrong real result. Shrink the "
            f"configuration that grew {component_name}'s depth this large."
        )


@dataclass(slots=True)
class TimeLoopArtifacts:
    root: Path
    problem_yaml: Path
    architecture_yaml: Path
    architecture_constraints_yaml: Path
    mapspace_constraints_yaml: Path
    mapper_yaml: Path
    model_mapping_yaml: Path
    primitive_components_yaml: Path
    accelergy_config_yaml: Path
    mapping_plan_json: Path
    map_yaml: Path
    stats_txt: Path
    art_yaml: Path
    log: Path


def feasible_tile_k_candidates(
    gemm: GemmSpec, wafer: WaferSpec, tile_m: int, tile_n: int
) -> list[int]:
    """Every SRAM-feasible tile_k, largest (fewest K phases) first.

    A candidate must (a) evenly divide K -- so every K phase does identical
    work, matching the single fully-specified mapping.yaml this project
    hands to timeloop-model for that candidate (see generate_timeloop_
    inputs()) -- and (b) respect this project's real single-buffered per-PE
    SRAM budget: c_bytes + a_bytes + b_bytes <= wafer.sram_bytes_per_pe -
    wafer.fifo_router_bytes_per_pe (see GemmMappingContract.
    data_sram_bytes_per_pe -- single-buffered because the real Cerebras CSL
    task synchronization this project's simulate_gemm() now follows does
    not prefetch a phase ahead; see that module's docstring). The FIFO/
    router reservation is subtracted here because those buffers now really
    occupy part of the PE's declared SRAM (see specs.py's
    fifo_router_bytes_per_pe), not a separate budget. This is the candidate
    set search.search_tile_k() enumerates: one real timeloop-model Docker
    run per entry, not a Python-side performance estimate.
    """
    data_sram_budget = wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe
    candidates = []
    for tile_k in range(gemm.k, 0, -1):
        if gemm.k % tile_k:
            continue
        a_bytes = tile_m * tile_k * gemm.bytes_per_element
        b_bytes = tile_k * tile_n * gemm.bytes_per_element
        c_bytes = tile_m * tile_n * gemm.bytes_per_element
        if c_bytes + a_bytes + b_bytes <= data_sram_budget:
            candidates.append(tile_k)
    if not candidates:
        raise ValueError("no GEMM K tile fits the configured PE SRAM")
    return candidates


def _largest_feasible_k(
    gemm: GemmSpec, wafer: WaferSpec, tile_m: int, tile_n: int
) -> int:
    return feasible_tile_k_candidates(gemm, wafer, tile_m, tile_n)[0]


def _largest_divisor_leq(n: int, upper_bound: int) -> int:
    """The largest divisor of ``n`` that is ``<= upper_bound`` (and ``>= 1``).

    REAL BUG THIS FIXES (found and root-caused this session, via an actual
    Docker crash on Qwen3-8B's Q/O-projection GEMM, M=10 N=4096 K=4096,
    against a wafer mesh derived from a DIFFERENT, larger GEMM in the same
    model -- the MLP's M=10 N=12288 K=4096): the old code below computed
    ``active_cols = min(wafer.mesh_cols, gemm.n)`` and then
    ``tile_n = ceil(gemm.n / active_cols)``, with NO guarantee that
    ``active_cols * tile_n == gemm.n`` exactly. That product is exactly the
    real TimeLoop spatial+temporal N factor product architecture_
    constraints.yaml/mapping.yaml declare -- and real ``timeloop-model``
    requires it to equal the workload's N dimension EXACTLY, with no
    boundary/remainder tile (confirmed by a real Docker error this session:
    "product of all factors of dimension N is 4608, which is not equal to
    the dimension size of the workload 4096", from mesh_cols=1536 (a true
    divisor of the MLP's own N=12288, since a SEPARATE WaferSpec-derivation
    step -- see area_budget.py's ``_derive_mesh`` -- only ever guarantees
    divisibility against the ONE representative GEMM it was given) not
    dividing evenly into the Q/O-projection's N=4096).

    This is the SAME real constraint area_budget.py's own ``_derive_mesh``
    already documents and searches for (see its module docstring's stage 1
    and its own ``_divisors`` helper, which this function's algorithm
    mirrors) -- but ``_derive_mesh`` only ever runs ONCE, against a single
    representative GEMM, to pick mesh_rows/mesh_cols; the resulting
    WaferSpec is then reused across EVERY GEMM shape a full model
    composition (like Qwen3-8B's per-layer Q/K/V/O/gate/up/down
    projections) actually calls, and those other shapes were never
    guaranteed to share that one representative GEMM's divisors. So the
    SAME "search real divisors, pick the largest that fits" discipline
    has to be applied again here, per-GEMM, at the point where THIS
    specific gemm's active_rows/active_cols get pinned -- not just once,
    upstream, at wafer-derivation time.

    Real, honest hardware consequence, not swept under the rug: when
    ``upper_bound`` (the physical mesh_rows/mesh_cols) is not itself a
    divisor of ``n`` (the GEMM's M/N), this returns a SMALLER divisor,
    meaning fewer PEs are actually used along that dimension for this
    particular GEMM -- the remaining mesh rows/columns sit idle for this
    op's run (this project's fixed one-static-tile-per-PE execution model
    -- see generate_timeloop_inputs's own module-level docstring reference
    to wafer/simulator.py -- has no boundary-tile mechanism on the real
    Docker-facing path, only wafer/simulator.py's own pure-Python model
    does, via ``_extent()``, per area_budget.py's stage-1 docstring). This
    under-utilization is real and reported via active_pes/tile_m/tile_n
    exactly as computed below, not hidden.

    O(sqrt(n)) trial division, same as area_budget.py's ``_divisors`` (n is
    at most a few hundred thousand for any realistic GEMM dimension in this
    project, so this is cheap -- no factorization library needed).
    """
    if n <= 0 or upper_bound <= 0:
        raise ValueError(f"_largest_divisor_leq requires positive inputs, got n={n}, upper_bound={upper_bound}")
    cap = min(n, upper_bound)
    best = 1
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i <= cap:
                best = max(best, i)
            partner = n // i
            if partner <= cap:
                best = max(best, partner)
        i += 1
    return best


def generate_timeloop_inputs(
    gemm: GemmSpec, wafer: WaferSpec, out_dir: Path, *, tile_k: int | None = None
) -> TimeLoopArtifacts:
    """Materialize TimeLoop v4 problem/architecture/constraints/mapper YAML.

    ``tile_k``, when given, pins the intra-PE K tile used to build
    mapping.yaml/mapping_plan.json to that exact value instead of this
    project's own greedy "largest that fits" choice -- this is what lets
    search.search_tile_k() generate one full, independent input set per
    SRAM-feasible candidate (see feasible_tile_k_candidates()) and run a
    real timeloop-model evaluation on each. It must still evenly divide K
    and respect the single-buffered SRAM budget; violating either raises
    immediately rather than silently producing an inconsistent mapping.
    """
    gemm.validate()
    wafer.validate()
    root = out_dir / "timeloop"
    root.mkdir(parents=True, exist_ok=True)
    artifacts = TimeLoopArtifacts(
        root=root,
        problem_yaml=root / "problem.yaml",
        architecture_yaml=root / "architecture.yaml",
        architecture_constraints_yaml=root / "architecture_constraints.yaml",
        mapspace_constraints_yaml=root / "mapspace_constraints.yaml",
        mapper_yaml=root / "mapper.yaml",
        model_mapping_yaml=root / "mapping.yaml",
        primitive_components_yaml=root / "primitive_components.yaml",
        accelergy_config_yaml=root / "accelergy_config.yaml",
        mapping_plan_json=root / "mapping_plan.json",
        map_yaml=root / "resolved.map.yaml",
        stats_txt=root / "resolved.stats.txt",
        art_yaml=root / "resolved.ART.yaml",
        log=root / "timeloop.log",
    )

    # REAL FIX (this session, see _largest_divisor_leq's own docstring for
    # the full root-cause citation): active_rows/active_cols must be TRUE
    # divisors of gemm.m/gemm.n, not just <= them, or real timeloop-model
    # rejects the resulting mapping with a "product of all factors ...
    # not equal to the dimension size of the workload" error whenever the
    # wafer's mesh size doesn't happen to divide THIS gemm's shape evenly
    # (which a WaferSpec derived from a DIFFERENT, representative GEMM --
    # e.g. area_budget.py's derive_wafer_spec_from_area_budget -- cannot
    # guarantee for every GEMM shape a full model composition later calls).
    active_rows = _largest_divisor_leq(gemm.m, wafer.mesh_rows)
    active_cols = _largest_divisor_leq(gemm.n, wafer.mesh_cols)
    active_pes = active_rows * active_cols
    tile_m = gemm.m // active_rows  # exact: active_rows is a true divisor of gemm.m
    tile_n = gemm.n // active_cols  # exact: active_cols is a true divisor of gemm.n
    if tile_k is None:
        tile_k = _largest_feasible_k(gemm, wafer, tile_m, tile_n)
    else:
        if gemm.k % tile_k:
            raise ValueError(f"tile_k={tile_k} does not evenly divide K={gemm.k}")
        a_bytes = tile_m * tile_k * gemm.bytes_per_element
        b_bytes = tile_k * tile_n * gemm.bytes_per_element
        c_bytes = tile_m * tile_n * gemm.bytes_per_element
        # Budget is the raw per-PE SRAM minus what FIFO/router now really
        # reserve out of it (see feasible_tile_k_candidates' docstring and
        # specs.py's fifo_router_bytes_per_pe).
        data_sram_budget = wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe
        if c_bytes + a_bytes + b_bytes > data_sram_budget:
            raise ValueError(
                f"tile_k={tile_k} exceeds the single-buffered PE SRAM budget "
                f"after reserving {wafer.fifo_router_bytes_per_pe} bytes for "
                f"FIFO/router ({c_bytes + a_bytes + b_bytes} > {data_sram_budget} bytes)"
            )
    k_phases = ceil(gemm.k / tile_k)
    datawidth = gemm.bytes_per_element * 8
    dram_block_size = max(1, int(wafer.link_bandwidth_bytes_per_cycle))
    dram_width_bits = dram_block_size * 8
    # Aladdin_table (the intmac/mac energy estimator plugin) computes
    # latency as global_cycle_seconds * action_latency_cycles, defaulting
    # global_cycle_seconds to its own hardcoded 5ns if the component
    # doesn't declare it (see accelergy-aladdin-plug-in's aladdin_table.py,
    # query_csv_using_latency()). Pass this project's real configured
    # clock instead, so the mac energy number reflects wafer.clock_ghz
    # rather than an unrelated hardcoded value.
    global_cycle_seconds = 1.0 / (wafer.clock_ghz * 1e9)
    # TimeLoop's own mapspace search checks single-buffered occupancy at
    # each level (a_bytes + b_bytes + c_bytes <= depth). This project's
    # actual execution model (wafer/simulator.py's simulate_gemm) is now
    # ALSO single-buffered for A/B -- real Cerebras CSL GEMM task
    # synchronization does not prefetch a phase ahead (see that module's
    # docstring) -- so TimeLoop's own check already matches this project's
    # real SRAM requirement exactly; the physical SRAM depth can be declared
    # to TimeLoop directly, with no halving trick needed. FIFO/router
    # storage now really occupies part of this same physical SRAM budget
    # (wafer.fifo_router_bytes_per_pe -- see specs.py, per the project's
    # decision that these are not a separate/additional allocation), so
    # that many bytes are subtracted before converting the remainder to
    # GEMM-element words; WaferSpec.validate() (called above) already
    # guarantees this leaves a positive remainder.
    pe_scratchpad_depth_words = max(
        1,
        (wafer.sram_bytes_per_pe - wafer.fifo_router_bytes_per_pe)
        // gemm.bytes_per_element,
    )
    # fifo_buffer/router_buffer are declared as their own real SRAM-class
    # Accelergy components (see architecture.yaml below) rather than folded
    # into pe_scratchpad's own depth, so CactiSRAM gives each a real,
    # separately-attributable area number (see GemmMappingContract.
    # pe_internal_area_um2). Both use a byte-granular word convention
    # (block-size 1, word-bits 8, width 8) since they store raw fabric
    # bytes, not GEMM-typed elements -- unlike pe_scratchpad, whose depth is
    # in gemm.bytes_per_element-wide words.
    fifo_buffer_depth_words = max(1, wafer.fifo_bytes_per_pe)
    router_buffer_depth_words = max(1, wafer.router_bytes_per_pe)

    # Pre-flight check, BEFORE any real Docker call: reject any SRAM-class
    # declaration real CACTI cannot be trusted to model correctly (see
    # SAFE_CACTI_CACHE_SIZE_BYTES above) with a clear Python ValueError,
    # rather than risking a real Docker run that silently returns a
    # wrong-but-plausible-looking near-zero area. Covers this function's
    # own direct callers (including a plain --sram-kib/--fifo-packets CLI
    # run, not just area_budget.py's own search, which stays well under
    # this ceiling by construction but is not the only way to reach here).
    _validate_cacti_safe_sram(
        "pe_scratchpad",
        word_bits=8,
        block_size_bytes=gemm.bytes_per_element,
        depth_words=pe_scratchpad_depth_words,
    )
    _validate_cacti_safe_sram(
        "fifo_buffer", word_bits=8, block_size_bytes=1, depth_words=fifo_buffer_depth_words
    )
    _validate_cacti_safe_sram(
        "router_buffer", word_bits=8, block_size_bytes=1, depth_words=router_buffer_depth_words
    )

    # --- problem.yaml -----------------------------------------------------
    # C[M, N] += A[M, K] * B[K, N], dense (no sparsity modeled here).
    artifacts.problem_yaml.write_text(
        "problem:\n"
        "  shape:\n"
        "    name: GEMM\n"
        "    dimensions: [ M, N, K ]\n"
        "    data-spaces:\n"
        "      - name: A\n"
        "        projection: [ [ [M] ], [ [K] ] ]\n"
        "      - name: B\n"
        "        projection: [ [ [K] ], [ [N] ] ]\n"
        "      - name: C\n"
        "        projection: [ [ [M] ], [ [N] ] ]\n"
        "        read-write: True\n"
        "  instance:\n"
        f"    M: {gemm.m}\n"
        f"    N: {gemm.n}\n"
        f"    K: {gemm.k}\n",
        encoding="ascii",
    )

    # --- architecture.yaml --------------------------------------------------
    # Native subtree/local hierarchy (NOT the v0.4 nodes:/!Component format,
    # see module docstring): DRAM (off-chip) -> replicated PE[0..active_pes-1]
    # (meshX = active_cols encodes the 2D grid shape) -> per-PE scratchpad
    # (SRAM), fifo_buffer, router_buffer, and mac. fifo_buffer/router_buffer
    # model the real Cerebras placement of the fabric router and its FIFO
    # queues INSIDE each PE tile (cerebras.ai's architecture deep-dive: "each
    # core has a fabric router...5-port design"), each declared as its own
    # real SRAM-class component so CactiSRAM gives it a real (if
    # approximate, for router_buffer) area number, same as pe_scratchpad --
    # see specs.py's WaferSpec.fifo_bytes_per_pe/router_bytes_per_pe for
    # their sizing and the real citations behind it. Both are byte-granular
    # (block-size 1, word-bits 8, width 8), unlike pe_scratchpad's GEMM-
    # element-wide words, since they store raw fabric bytes.
    artifacts.architecture_yaml.write_text(
        "architecture:\n"
        "  version: 0.3\n"
        "  subtree:\n"
        "  - name: system\n"
        "    local:\n"
        "    - name: DRAM\n"
        "      class: DRAM\n"
        "      attributes:\n"
        f"        width: {dram_width_bits}\n"
        f"        block-size: {dram_block_size}\n"
        "        word-bits: 8\n"
        "    subtree:\n"
        "    - name: wafer\n"
        "      attributes:\n"
        f"        technology: \"{wafer.technology_node}\"\n"
        "      subtree:\n"
        f"      - name: PE[0..{active_pes - 1}]\n"
        "        local:\n"
        "        - name: pe_scratchpad\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {pe_scratchpad_depth_words}\n"
        f"            block-size: {gemm.bytes_per_element}\n"
        "            word-bits: 8\n"
        # CactiSRAM's real constructor signature is
        # (technology, width, depth, n_rw_ports=1, n_banks=1) -- width has
        # no default (see accelergy-cacti-plug-in's cacti_wrapper.py).
        # DRAM above already declares width and is confirmed to reach
        # CactiDRAM successfully; pe_scratchpad previously omitted it,
        # which is the leading suspect for why CactiSRAM was never
        # actually invoked (silent fallback to accelergy's dummy
        # estimator instead). width = block-size(bytes) * word-bits, same
        # convention DRAM already uses.
        f"            width: {datawidth}\n"
        f"            meshX: {active_cols}\n"
        "        - name: fifo_buffer\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {fifo_buffer_depth_words}\n"
        "            block-size: 1\n"
        "            word-bits: 8\n"
        "            width: 8\n"
        f"            meshX: {active_cols}\n"
        "        - name: router_buffer\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {router_buffer_depth_words}\n"
        "            block-size: 1\n"
        "            word-bits: 8\n"
        "            width: 8\n"
        f"            meshX: {active_cols}\n"
        "        - name: mac\n"
        "          class: intmac\n"
        "          attributes:\n"
        f"            datawidth: {datawidth}\n"
        f"            width: {datawidth}\n"
        f"            meshX: {active_cols}\n"
        f"            global_cycle_seconds: {global_cycle_seconds:.6e}\n",
        encoding="ascii",
    )

    # --- architecture_constraints.yaml --------------------------------------
    # Pin the spatial fanout: M rows and N columns unrolled across the PE
    # array (targeting DRAM, the level directly above the PE array, per
    # TimeLoop's convention that a spatial constraint on level X controls
    # how the loops below X are spread across X's children). Keep A/B/C
    # resident in the per-PE scratchpad (no bypass). fifo_buffer/
    # router_buffer never hold any of this problem's GEMM dataspaces (they
    # exist purely so Accelergy gives them a real area number -- see
    # architecture.yaml above); bypass all three there so timeloop-mapper
    # doesn't try to allocate A/B/C onto them.
    artifacts.architecture_constraints_yaml.write_text(
        "architecture_constraints:\n"
        "  targets:\n"
        "  - target: pe_scratchpad\n"
        "    type: dataspace\n"
        "    keep: [A, B, C]\n"
        "  - target: fifo_buffer\n"
        "    type: dataspace\n"
        "    bypass: [A, B, C]\n"
        "  - target: router_buffer\n"
        "    type: dataspace\n"
        "    bypass: [A, B, C]\n"
        "  - target: DRAM\n"
        "    type: spatial\n"
        "    permutation: NM\n"
        f"    factors: N={active_cols} M={active_rows}\n",
        encoding="ascii",
    )

    # --- mapspace_constraints.yaml -------------------------------------------
    # K is left completely unconstrained (no factors pin it anywhere) --
    # timeloop-mapper genuinely searches every valid K factorization across
    # pe_scratchpad and DRAM's temporal levels (DRAM's own spatial K factor
    # is structurally forced to 1: all 1024 declared PE instances are
    # already consumed by the M/N spatial fanout pinned below, so there is
    # no spare PE dimension for the mapper to fan K out across), bounded by
    # the halved SRAM `depth` in architecture.yaml -- i.e. TimeLoop itself
    # decides the intra-PE tile_k, not this project's own
    # `_largest_feasible_k` Python search (that function's output now only
    # seeds mapping_plan.json/mapping.yaml for the optional single-point
    # timeloop-model cross-check below).
    #
    # M/N, by contrast, DO need one explicit pin here: architecture_
    # constraints.yaml above only fixes DRAM's *spatial* M/N fanout (the
    # mesh mapping), leaving DRAM's *temporal* M/N free -- and an
    # unconstrained search will happily split the remaining per-PE M/N
    # across BOTH pe_scratchpad and an outer DRAM temporal loop (confirmed
    # against a real run: it produced `pe_scratchpad M8 N8` + `DRAM M4 N4`,
    # i.e. each PE looping over 16 separate 8x8 output tiles instead of
    # owning one static tile for the whole computation). This project's
    # wafer/simulator.py models exactly one static (tile_m, tile_n) output
    # block per PE for the entire run, with only K streamed in phases -- so
    # DRAM's temporal M/N is pinned to 1 here, forcing the mapper to keep
    # the *entire* per-PE M/N tile resident at pe_scratchpad (where
    # parse_mapping_contract() reads tile_m/tile_n from), matching that
    # execution model. This is a real architectural constraint of this
    # project's fixed PE-tiling scheme, not a stand-in for TimeLoop's own
    # K-tiling decision.
    artifacts.mapspace_constraints_yaml.write_text(
        "mapspace_constraints:\n"
        "  targets:\n"
        "  - target: DRAM\n"
        "    type: temporal\n"
        "    permutation: MN\n"
        "    factors: M=1 N=1\n",
        encoding="ascii",
    )

    # --- mapper.yaml ---------------------------------------------------------
    # K is now a genuinely open search dimension (see mapspace_constraints
    # above), so the mapspace is meaningfully larger than a single point.
    # A victory-condition of 1 (stop as soon as one non-improving candidate
    # is found after the last upgrade) is only adequate for a
    # near-single-point space; against real data it converged on a poor
    # local optimum (tile_k=8 instead of a much better tile_k=64) the
    # instant the search space opened up. Require many consecutive
    # non-improving candidates before conceding, so the random-pruned
    # search actually explores the K factorization space instead of
    # stopping at the first lucky/unlucky sample. Hyphenated keys match
    # this native-format binary (confirmed against a real working
    # example), not the v0.4 underscored spelling.
    artifacts.mapper_yaml.write_text(
        "mapper:\n"
        "  optimization-metrics: [ delay, energy ]\n"
        "  algorithm: random-pruned\n"
        "  num-threads: 4\n"
        "  timeout: 1000\n"
        "  victory-condition: 500\n",
        encoding="ascii",
    )

    # --- mapping.yaml ---------------------------------------------------------
    # A fully-specified mapping for ``timeloop-model`` (evaluates one exact
    # mapping, no search -- see run_timeloop_model()). This mirrors the
    # EXACT shape of a real, proven-working timeloop-mapper output
    # (outputs/gemm1024_v10/timeloop/resolved.map.yaml), not a hand-guessed
    # schema: `datatype`/keep+bypass entries at both storage levels, the
    # FULL per-PE (tile_m, tile_n, tile_k) temporal loop at pe_scratchpad
    # (TimeLoop's own mapper puts all three there together, not split
    # across an extra innermost level), and the DRAM spatial/temporal
    # entries for the mesh fanout and K streaming. There is deliberately
    # NO entry targeting "mac": TimeLoop's mapping schema only accepts
    # storage levels as mapping targets, not the arithmetic/compute unit --
    # an earlier version of this file targeted "mac" directly and a real
    # Docker run failed with "ERROR: target storage level not found: mac"
    # (confirmed against actual timeloop-model output, not guessed).
    artifacts.model_mapping_yaml.write_text(
        "mapping:\n"
        "  - target: pe_scratchpad\n"
        "    type: datatype\n"
        "    keep: [A, B, C]\n"
        "    bypass: []\n"
        "  - target: fifo_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [A, B, C]\n"
        "  - target: router_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [A, B, C]\n"
        "  - target: DRAM\n"
        "    type: datatype\n"
        "    keep: [A, B, C]\n"
        "    bypass: []\n"
        "  - target: pe_scratchpad\n"
        "    type: temporal\n"
        f"    factors: M={tile_m} N={tile_n} K={tile_k}\n"
        "    permutation: MNK\n"
        "  - target: DRAM\n"
        "    type: spatial\n"
        f"    factors: M={active_rows} N={active_cols} K=1\n"
        "    permutation: NMK\n"
        "    split: 1\n"
        "  - target: DRAM\n"
        "    type: temporal\n"
        f"    factors: M=1 N=1 K={k_phases}\n"
        "    permutation: KMN\n",
        encoding="ascii",
    )

    # --- primitive_components.yaml -------------------------------------------
    # Declares the action list for our custom SRAM/DRAM/intmac classes.
    # accelergy has no built-in notion of these classes' actions (its
    # shipped primitive_component_libs dir is empty by design -- plugins
    # only advertise what they *can* estimate if asked, they don't tell
    # accelergy what to ask for); without this file accelergy silently
    # treats every unrecognized class as `actions: []`, so the ERT's
    # per-action energy loop iterates zero times and produces an empty
    # table with no error. Action names below match what the bundled
    # plugins actually implement (confirmed by reading their source):
    # CactiSRAM/CactiDRAM support idle/read/update/write/leak for
    # ['SRAM','sram']/['DRAM','dram']. Aladdin_table's intmac estimator was
    # originally assumed (from reading its source only) to special-case
    # mac_gated/mac_reused/leak and only need action_latency_cycles for
    # mac_random -- a real Docker run's accelergy.log proved that wrong:
    # all four actions hit query_csv_using_latency() and the same
    # `KeyError: 'action_latency_cycles'` when it isn't supplied. Declaring
    # it only on mac_random happened to still produce a correct total
    # (this project's dense GEMM mapping never actually invokes
    # mac_gated/mac_reused with a nonzero count, so their dummy fallback
    # was multiplied by zero), but that's incidental, not a real fix --
    # all four actions get the argument now.
    artifacts.primitive_components_yaml.write_text(
        "classes:\n"
        "  - name: SRAM\n"
        "    attributes: {}\n"
        "    actions:\n"
        "      - name: read\n"
        "      - name: write\n"
        "      - name: update\n"
        "      - name: idle\n"
        "      - name: leak\n"
        "  - name: DRAM\n"
        "    attributes: {}\n"
        "    actions:\n"
        "      - name: read\n"
        "      - name: write\n"
        "      - name: update\n"
        "      - name: idle\n"
        "      - name: leak\n"
        "  - name: intmac\n"
        "    attributes: {}\n"
        "    actions:\n"
        # `1..1` is accelergy's range syntax for a fixed single value --
        # architecture.yaml above declares exactly one scalar `mac`
        # component per PE (no internal batching/vector width), so a
        # 1-cycle-per-action latency matches that model; TimeLoop's real
        # measured Cycles for this mapping is exactly this rate applied to
        # the mapping's MAC count (confirmed against a real Docker run:
        # 1,048,576 Cycles for 1,048,576 MACs/PE -- see wafer/simulator.py,
        # which now derives its own per-mapping cycles/MAC rate directly
        # from this real number instead of an assumed constant). All four
        # intmac actions need this now (see comment above).
        "      - name: mac_random\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n"
        "      - name: mac_gated\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n"
        "      - name: mac_reused\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n"
        "      - name: leak\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n",
        encoding="ascii",
    )

    # --- accelergy_config.yaml ------------------------------------------------
    # accelergy looks for accelergy_config.yaml in its cwd before falling
    # back to ~/.config/accelergy/ (see accelergy/utils/utils.py
    # get_config_file_path()) -- since timeloop-mapper/timeloop-model run
    # with cwd=/work (this directory), dropping our own config here is
    # enough to get primitive_components.yaml picked up, no extra Docker
    # mount required. estimator_plug_ins/table_plug_ins/the default (empty)
    # primitive_component_libs path mirror exactly what accelergy would
    # have auto-generated on its own (confirmed against a real run's log).
    artifacts.accelergy_config_yaml.write_text(
        "version: 0.4\n"
        "estimator_plug_ins:\n"
        "  - /usr/local/share/accelergy/estimation_plug_ins\n"
        "primitive_components:\n"
        "  - /usr/local/share/accelergy/primitive_component_libs\n"
        "  - primitive_components.yaml\n"
        "compound_components: []\n"
        "math_functions: []\n"
        "python_plug_ins: []\n"
        "table_plug_ins:\n"
        "  roots:\n"
        "    - /usr/local/share/accelergy/table_plug_ins\n",
        encoding="ascii",
    )

    # Reference-only: this project's own SRAM-capacity-constrained greedy
    # tile_k search, used solely to build the fully-specified mapping.yaml
    # for the optional single-point run_timeloop_model() cross-check.
    # parse_mapping_contract() does NOT read this file -- the real contract
    # comes from timeloop-mapper's own search result in .map.yaml (see
    # _extract_tiling_from_map_yaml()).
    artifacts.mapping_plan_json.write_text(
        json.dumps(
            {
                "active_rows": active_rows,
                "active_cols": active_cols,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return artifacts


def _latest_fresh_docker_output(root: Path, pattern: str, *, exclude: Path) -> Path | None:
    """Find the file matching `pattern` under `root` that the real Docker
    invocation which JUST RAN actually produced -- excluding `exclude`
    (always one of artifacts.map_yaml/stats_txt/art_yaml, this project's
    OWN canonicalized copy destination, never a genuine fresh Docker
    output) and, among what remains, picking by real filesystem mtime
    (most recent first) rather than filename.

    CONFIRMED REAL BUG (2026-08-23, against a real re-run into an already-
    used --out-dir): the previous version of every call site here was
    `sorted(root.glob(pattern))[0]`, which does NOT exclude the canonical
    copy -- and alphabetically "resolved.*" (this project's own
    canonicalized filename) sorts before "timeloop-mapper.*"/
    "timeloop-model.*" (the real Docker binaries' own output prefix). This
    project reuses output directories across separate runs (search.py's
    fixed tile_k_<N> subdirectories in particular -- a second CLI
    invocation with a DIFFERENT --sram-kib/--fifo-packets/area budget into
    the SAME --out-dir writes into the exact same tile_k_<N> folders as
    the first), so once a directory has been used once, `sorted(...)[0]`
    picked the STALE canonicalized copy from the FIRST run right back up
    on every subsequent run, silently ignoring that run's genuinely fresh
    Docker output sitting right next to it. Confirmed against real files:
    a re-run's `resolved.ART.yaml` still held fifo_buffer area=1.0 (a
    stale leftover from an earlier, unrelated fifo_capacity_packets
    config) while the SAME directory's fresh `timeloop-model.ART.yaml`
    (correct real mtime, correct real content for the actual re-run)
    reported fifo_buffer area=65197.7 -- and the parsed contract silently
    used the wrong, stale one. Because Cycles/tile_m/tile_n/tile_k happen
    to be independent of fifo_capacity_packets in this project's model,
    this corruption was invisible in the Cycles/energy columns and only
    showed up in the area numbers -- so a stale re-run can otherwise look
    completely normal.
    """
    candidates = [p for p in root.glob(pattern) if p != exclude]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_timeloop_mapper(
    artifacts: TimeLoopArtifacts,
    *,
    image: str = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest",
) -> None:
    """Run the real ``timeloop-mapper`` inside Docker.

    Raises ``RuntimeError`` if Docker fails or the mapper produces no
    ``.map.yaml``/``.stats.txt`` output; there is no fallback path.
    """
    command = [
        "docker", "run", "--rm",
        "-v", f"{artifacts.root.resolve()}:/work",
        "-v", f"{artifacts.accelergy_config_yaml.resolve()}:/root/.config/accelergy/accelergy_config.yaml:ro",
        "-w", "/work",
        image, "timeloop-mapper",
        "problem.yaml", "architecture.yaml",
        "architecture_constraints.yaml", "mapspace_constraints.yaml", "mapper.yaml",
    ]
    # NOTE: primitive_components.yaml is *not* passed as a positional
    # arg here -- accelergy only discovers primitive component class
    # declarations via its own config file's `primitive_components` list
    # (see accelergy_config.yaml above), never via a generically-typed
    # command-line input file. Since cwd=/work for this container,
    # accelergy_config.yaml is picked up automatically (accelergy checks
    # ./accelergy_config.yaml before ~/.config/accelergy/), no extra
    # mount or argument needed.
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    artifacts.log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"timeloop-mapper exited with status {completed.returncode}; see {artifacts.log}"
        )

    map_candidate = _latest_fresh_docker_output(artifacts.root, "*.map.yaml", exclude=artifacts.map_yaml)
    stats_candidate = _latest_fresh_docker_output(artifacts.root, "*.stats.txt", exclude=artifacts.stats_txt)
    if map_candidate is None or stats_candidate is None:
        raise RuntimeError(
            f"timeloop-mapper produced no .map.yaml/.stats.txt under {artifacts.root}; see {artifacts.log}"
        )
    artifacts.map_yaml.write_text(map_candidate.read_text(), encoding="utf-8")
    artifacts.stats_txt.write_text(stats_candidate.read_text(), encoding="utf-8")

    # Accelergy generates ERT.yaml/ART.yaml as a side effect of every real
    # run alongside stats.txt (confirmed against TimeLoop's own docs); no
    # fallback if it's missing -- that would mean this run's real PE-internal
    # area simply isn't available, not a case to paper over with a guess.
    art_candidate = _latest_fresh_docker_output(artifacts.root, "*.ART.yaml", exclude=artifacts.art_yaml)
    if art_candidate is None:
        raise RuntimeError(
            f"timeloop-mapper produced no .ART.yaml under {artifacts.root}; see {artifacts.log}"
        )
    artifacts.art_yaml.write_text(art_candidate.read_text(), encoding="utf-8")


def run_timeloop_model(
    artifacts: TimeLoopArtifacts,
    *,
    image: str = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest",
) -> None:
    """Run the real ``timeloop-model`` inside Docker against mapping.yaml.

    Unlike ``timeloop-mapper`` (a search over the constraints-pruned
    mapspace), ``timeloop-model`` evaluates exactly one fully-specified
    mapping -- see model_mapping_yaml in generate_timeloop_inputs(). Useful
    as a quick single-point check, and the one search.search_tile_k() calls
    once per candidate tile_k. run_timeloop_mapper() is the one that
    actually lets TimeLoop *decide* the intra-PE tiling via search.

    Raises ``RuntimeError`` if Docker fails or the model produces no
    ``.stats.txt`` output; there is no fallback path.
    """
    command = [
        "docker", "run", "--rm",
        "-v", f"{artifacts.root.resolve()}:/work",
        "-v", f"{artifacts.accelergy_config_yaml.resolve()}:/root/.config/accelergy/accelergy_config.yaml:ro",
        "-w", "/work",
        image, "timeloop-model",
        "architecture.yaml", "problem.yaml", "mapping.yaml",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    artifacts.log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"timeloop-model exited with status {completed.returncode}; see {artifacts.log}"
        )

    stats_candidate = _latest_fresh_docker_output(artifacts.root, "*.stats.txt", exclude=artifacts.stats_txt)
    if stats_candidate is None:
        raise RuntimeError(
            f"timeloop-model produced no .stats.txt under {artifacts.root}; see {artifacts.log}"
        )
    artifacts.stats_txt.write_text(stats_candidate.read_text(), encoding="utf-8")

    # Same real accelergy side-effect as run_timeloop_mapper() above -- no
    # fallback if it's missing.
    art_candidate = _latest_fresh_docker_output(artifacts.root, "*.ART.yaml", exclude=artifacts.art_yaml)
    if art_candidate is None:
        raise RuntimeError(
            f"timeloop-model produced no .ART.yaml under {artifacts.root}; see {artifacts.log}"
        )
    artifacts.art_yaml.write_text(art_candidate.read_text(), encoding="utf-8")

    # Confirmed against a real run: unlike timeloop-mapper, timeloop-model
    # does NOT re-emit a `*.map.yaml` (it only evaluates the mapping given
    # to it -- the real Docker output directory has `timeloop-model.map.txt`,
    # a different, human-readable format, no `.map.yaml` at all). There is
    # nothing to search for here, and nothing to guess: the exact mapping
    # actually fed to Docker is model_mapping_yaml, on disk and unchanged
    # since generate_timeloop_inputs() wrote it, in the same target/type/
    # factors schema `_extract_tiling_from_map_yaml` already parses -- so
    # record that verbatim as the resolved mapping. In this mode OpenCake
    # itself picked tile_k (see search.search_tile_k), not TimeLoop; what
    # TimeLoop's real Docker run contributes here is stats.txt's Cycles/
    # Energy for that exact, already-known mapping. (map_candidate here
    # should always be None in practice per the above -- kept as a guard,
    # via the same _latest_fresh_docker_output helper as every other site,
    # in case a future TimeLoop version ever does emit one; a STALE
    # leftover *.map.yaml from an earlier run into this same directory
    # must never be mistaken for one, which is exactly the real bug this
    # helper's `exclude`+mtime logic fixes -- see its docstring.)
    map_candidate = _latest_fresh_docker_output(artifacts.root, "*.map.yaml", exclude=artifacts.map_yaml)
    if map_candidate is not None:
        artifacts.map_yaml.write_text(map_candidate.read_text(), encoding="utf-8")
    else:
        artifacts.map_yaml.write_text(
            artifacts.model_mapping_yaml.read_text(encoding="utf-8"), encoding="utf-8"
        )


def _parse_summary_stats(stats_text: str, log_path: Path) -> tuple[int, float]:
    """Parse Cycles/Energy out of stats.txt's trailing 'Summary Stats' block.

    Raises with the actual block contents on failure instead of defaulting
    to 0 -- if TimeLoop's real output doesn't match this pattern, that
    needs to be visible and fixed, not silently swallowed.
    """
    summary_match = re.search(r"Summary\s+Stats\b[\s\S]*", stats_text)
    if not summary_match:
        raise RuntimeError(
            "could not find a 'Summary Stats' section in "
            f"{log_path.parent / 'timeloop-mapper.stats.txt'}; the real output format "
            "may not match what this parser expects -- open that file and check by hand."
        )
    block = summary_match.group(0)

    cycles_match = re.search(r"Cycles\s*:\s*([\d,]+)", block)
    energy_match = re.search(
        r"Energy\s*:\s*([\d,.]+)\s*(pJ|nJ|uJ|µJ|mJ|J)\b", block, re.IGNORECASE
    )
    if not cycles_match or not energy_match:
        raise RuntimeError(
            "found a 'Summary Stats' section but could not extract both Cycles and "
            f"Energy from it:\n{block[:2000]}"
        )
    cycles = int(cycles_match.group(1).replace(",", ""))
    raw_energy = float(energy_match.group(1).replace(",", ""))
    energy_uj = raw_energy * _ENERGY_UNIT_TO_UJ[energy_match.group(2).lower()]
    return cycles, energy_uj


_ART_ENTRY_RE = re.compile(
    r"-\s*name:\s*(?P<name>\S+)\s*\n\s*area:\s*(?P<area>[\d.eE+-]+)"
)


def _parse_art_table(art_text: str, art_path: Path) -> dict[str, float]:
    """Hand-parse a real ART.yaml's ``ART: tables:`` list, real-only (no
    PyYAML dependency this project has never needed -- same rationale as
    _parse_map_entries below).

    Real ART.yaml entries are named by their full architecture-tree path
    (e.g. ``system.wafer.PE[0..15].mac``); this keys the returned dict by
    just the LEAF component name (``mac``), since that's what this project's
    architecture.yaml declares and what callers here look up by.
    """
    entries = {
        match.group("name").rsplit(".", 1)[-1]: float(match.group("area"))
        for match in _ART_ENTRY_RE.finditer(art_text)
    }
    if not entries:
        raise RuntimeError(
            f"could not find any '- name: ...\\n  area: ...' entries in {art_path}; "
            "the real ART.yaml format may not match what this parser expects -- "
            "open that file and check by hand."
        )
    return entries


def _pe_internal_area_um2(art_text: str, art_path: Path) -> tuple[float, float, float, float]:
    """Real, per-single-instance mac/pe_scratchpad/fifo_buffer/router_buffer
    area out of a real ART.yaml.

    Raises if any leaf is missing -- e.g. because accelergy silently fell
    back to its dummy estimator for one of them (the exact failure mode this
    project hit and fixed for energy; see timeloop.py's module history) would
    still produce SOME area entry, just not a trustworthy one, so this alone
    can't catch that case -- but a genuinely absent entry (wrong component
    name, architecture.yaml change not reflected here) is caught rather than
    silently treated as zero area. fifo_buffer/router_buffer are new as of
    the FIFO/router-in-PE redesign (see architecture.yaml in generate_
    timeloop_inputs) -- if CACTI's SRAM estimator can't handle router_
    buffer's very small declared capacity (5 ports * 4 bytes = 20 bytes; see
    specs.py's router_bytes_per_pe), that will surface here as a missing
    entry rather than a silent zero.
    """
    areas = _parse_art_table(art_text, art_path)
    missing = [
        name
        for name in ("mac", "pe_scratchpad", "fifo_buffer", "router_buffer")
        if name not in areas
    ]
    if missing:
        raise RuntimeError(
            f"ART.yaml at {art_path} has no area entry for {missing} -- "
            f"found entries: {sorted(areas)}"
        )
    return areas["mac"], areas["pe_scratchpad"], areas["fifo_buffer"], areas["router_buffer"]


_MAP_TARGET_RE = re.compile(
    r"^[ \t]*-\s*target:\s*(?P<target>\S+)\s*\n"
    r"(?P<body>(?:(?![ \t]*-\s*target:)[^\n]*\n?)*)",
    re.MULTILINE,
)
_MAP_FACTOR_RE = re.compile(r"([A-Za-z])=?(\d+)")


def _parse_map_entries(map_yaml_text: str) -> list[dict]:
    """Hand-parse a real TimeLoop `.map.yaml`'s ``mapping:`` list.

    This is TimeLoop's native/classic format (see the module docstring), a
    flat list of ``- target/type/factors/permutation`` entries -- simple
    enough that a small regex walk is more robust here than adding a
    PyYAML dependency this project has never needed elsewhere.
    """
    entries = []
    for match in _MAP_TARGET_RE.finditer(map_yaml_text):
        body = match.group("body")
        type_match = re.search(r"type:\s*(\S+)", body)
        factors_match = re.search(r"factors:\s*([^\n]+)", body)
        factors = (
            {dim.upper(): int(val) for dim, val in _MAP_FACTOR_RE.findall(factors_match.group(1))}
            if factors_match
            else {}
        )
        entries.append({"target": match.group("target"), "type": type_match.group(1) if type_match else None, "factors": factors})
    return entries


def _extract_tiling_from_map_yaml(map_yaml_text: str, map_path: Path) -> dict[str, int]:
    """Extract TimeLoop's REAL tiling decision out of its own `.map.yaml`.

    Per the project requirement that TimeLoop itself decides the intra-PE
    tile partitioning (not this project's own Python/SimPy code):
    active_rows/active_cols come from the DRAM spatial entry (the M/N
    fanout onto the physical PE mesh -- pinned by architecture_constraints,
    but read back from the mapper's actual output rather than assumed);
    tile_m/tile_n/tile_k come from the pe_scratchpad temporal entry, whose
    K factor in particular is the one timeloop-mapper genuinely searched
    (mapspace_constraints.yaml leaves it unconstrained -- see
    generate_timeloop_inputs()).
    """
    entries = _parse_map_entries(map_yaml_text)

    dram_spatial = next(
        (e for e in entries if e["target"] == "DRAM" and e["type"] == "spatial"), None
    )
    pe_temporal = next(
        (e for e in entries if e["target"] == "pe_scratchpad" and e["type"] == "temporal"),
        None,
    )
    if dram_spatial is None or pe_temporal is None:
        raise RuntimeError(
            "could not find a 'DRAM spatial' and/or 'pe_scratchpad temporal' "
            f"entry in {map_path} -- its structure may not match what this "
            "parser expects; open the file by hand and check."
        )
    for label, entry, dims in (
        ("DRAM spatial", dram_spatial, ("M", "N")),
        ("pe_scratchpad temporal", pe_temporal, ("M", "N", "K")),
    ):
        missing = [d for d in dims if d not in entry["factors"]]
        if missing:
            raise RuntimeError(
                f"{label} entry in {map_path} is missing factor(s) {missing}: "
                f"{entry['factors']}"
            )
    return {
        "active_rows": dram_spatial["factors"]["M"],
        "active_cols": dram_spatial["factors"]["N"],
        "tile_m": pe_temporal["factors"]["M"],
        "tile_n": pe_temporal["factors"]["N"],
        "tile_k": pe_temporal["factors"]["K"],
    }


def parse_mapping_contract(
    artifacts: TimeLoopArtifacts,
    gemm: GemmSpec,
    wafer: WaferSpec,
    *,
    mode: str = MAPPER_MODE,
) -> GemmMappingContract:
    """Build the OpenCake handoff contract from TimeLoop's REAL decision.

    active_rows/active_cols/tile_m/tile_n/tile_k -- the intra-PE tiling
    decision, in particular tile_k -- are parsed directly out of
    TimeLoop's own real `.map.yaml` output (see
    `_extract_tiling_from_map_yaml`), NOT out of this project's own
    `mapping_plan.json` blocking search. Cycles/Energy are parsed back out
    of the real `stats.txt`.

    ``mode`` records which real TimeLoop binary produced these artifacts:
    ``MAPPER_MODE`` (default) for a `timeloop-mapper` search result (see
    `run_timeloop_mapper`), or ``MODEL_MODE`` for a single fully-specified
    `timeloop-model` evaluation of one candidate tile_k (see
    `run_timeloop_model` and `search.search_tile_k`). Either way the
    numbers are real Docker output -- this only labels their provenance.

    Raises if `artifacts.map_yaml`/`artifacts.stats_txt` don't exist --
    consistent with this module having no synthetic fallback: if the real
    Docker run didn't succeed, there is no real tiling decision to report.
    """
    if (
        not artifacts.map_yaml.exists()
        or not artifacts.stats_txt.exists()
        or not artifacts.art_yaml.exists()
    ):
        raise RuntimeError(
            f"no real TimeLoop output found (missing one of {artifacts.map_yaml}, "
            f"{artifacts.stats_txt}, {artifacts.art_yaml}); run_timeloop_mapper()/"
            "run_timeloop_model() must succeed before a contract can be "
            "built -- there is no fallback tiling/energy/area decision."
        )

    tiling = _extract_tiling_from_map_yaml(
        artifacts.map_yaml.read_text(encoding="utf-8"), artifacts.map_yaml
    )
    stats = artifacts.stats_txt.read_text(encoding="utf-8")
    local_compute_cycles, local_energy_uj = _parse_summary_stats(stats, artifacts.log)
    mac_area_um2, pe_scratchpad_area_um2, fifo_area_um2, router_area_um2 = _pe_internal_area_um2(
        artifacts.art_yaml.read_text(encoding="utf-8"), artifacts.art_yaml
    )

    contract = GemmMappingContract(
        gemm=gemm,
        wafer=wafer,
        active_rows=tiling["active_rows"],
        active_cols=tiling["active_cols"],
        tile_m=tiling["tile_m"],
        tile_n=tiling["tile_n"],
        tile_k=tiling["tile_k"],
        mapper_mode=mode,
        map_path=str(artifacts.map_yaml),
        stats_path=str(artifacts.stats_txt),
        local_compute_cycles=local_compute_cycles,
        local_energy_uj=local_energy_uj,
        mac_area_um2=mac_area_um2,
        pe_scratchpad_area_um2=pe_scratchpad_area_um2,
        fifo_area_um2=fifo_area_um2,
        router_area_um2=router_area_um2,
    )
    contract.validate()
    return contract
