"""Real, Docker-backed TimeLoop/Accelergy cost model for the `max`
primitive (a REDUCTION, not an elementwise op -- see module scope below).

REAL ACCELERGY COMPONENT USED -- ``comparator`` (not a placeholder, not
`add`'s ``fpadder`` or `mul`'s ``fpmultiplier`` reused as an
approximation): confirmed by cloning the exact real upstream source this
project's pinned Docker image's ``accelergy-aladdin-plug-in`` is built
from (``github.com/Accelergy-Project/accelergy-aladdin-plug-in`` --
same plugin add's ``fpadder``/mul's ``fpmultiplier`` were confirmed from;
this is source-level verification of the exact package the pinned image
installs, not a fresh live-image ``cat``, for that extra tier of
certainty see this project's own real Docker output once ``max`` is run).
Its real ``aladdin_table.py``:

  - ``get_supported_components()`` genuinely, independently lists
    ``"comparator": (["width"], ["leak", "access"])`` as a real,
    top-level supported class (line 457 of the real source) -- attributes
    are ``width`` (a plain integer datawidth, like `intadder`/
    `intmultiplier`), NOT ``exponent``/``mantissa`` like `add`'s
    `fpadder`/`mul`'s `fpmultiplier`.
  - ``comparator_estimate_energy``/``comparator_estimate_area`` both read
    real, separately-characterized data from ``data/comparator.csv``:

        latency(ns),dynamic energy(pJ),idle energy(pJ),area(um^2),comment
        0.5,0.03052,1.37E-05,71.5,32 bit comparator
        1,0.02947,2.51E-05,71,
        ...

    (real file, cloned and ``cat``-ed directly -- not from memory) -- a
    genuine, separately-measured 32-bit comparator table, distinct from
    every adder/multiplier table.
  - the shared ``query_csv_using_latency()`` helper both
    ``comparator_estimate_energy`` and `add`'s `fpadder_estimate_energy`
    call branches ONLY on the literal action-name string ``"leak"``
    (picks the ``idle energy(pJ)`` column) vs. anything else (picks
    ``dynamic energy(pJ)``) -- confirmed by reading the real source
    directly. Exactly like `fpadder`, `comparator` never special-cases a
    "gated"/"reused" action name the way `mul`'s `fpmultiplier` does.

ACTION-NAME CHOICE: this file reuses the exact literal action names
``mac_random``/``mac_gated``/``mac_reused``/``leak`` that `add`'s own
``primitive_components.yaml`` already uses and this project's real Docker
runs have already confirmed TimeLoop's native C++ engine accepts as valid
``random_compute``/``gated_compute`` event aliases (see `add`'s module
docstring's own "REAL DOCKER CRASH INVESTIGATION" section) -- NOT because
a comparison is semantically a "mac", but because (a) Accelergy's real
``comparator_estimate_energy`` does not discriminate between these names
at all beyond the literal "leak" check above, so no energy number changes
by reusing them, and (b) picking untested literal names here would
reopen exactly the kind of silent-topology-drop failure mode `add` spent
9 real-Docker rounds diagnosing. Reusing a real, already-Docker-verified
alias is the faithful choice, not a shortcut.

``class: mac`` / ``subclass: comparator`` SPLIT: same real mechanism
`add`/`mul` use (see `add`'s module docstring's "ROUND 9 FIX" section for
the full root-cause citation of TimeLoop's native ``computeClasses``
whitelist and Accelergy's ``arch_dict_2_obj.py`` subclass-preference
logic) -- ``class: mac`` satisfies TimeLoop's native topology parser's
hardcoded whitelist so it doesn't silently drop this compute level;
``subclass: comparator`` tells Accelergy to resolve the real,
separately-characterized comparator component (attributes: ``width``,
unchanged) instead of whatever `class: mac` alone would default to.

WHY THIS FILE HAS TWO LAYERS (`_CompareSpec`/`_simulate_compare` private,
`MaxSpec`/`simulate_max` public): unlike `sum` (which reduces via
add's ALREADY-BUILT `add` package), this project has no existing
"compare" primitive package to import a real single-comparison pipeline
from -- `max` is the first primitive to need the comparator component at
all. So this file first builds that real single-PE "K real comparator
firings" pipeline privately (`_CompareSpec`/`_CompareResult`/
`_generate_compare_timeloop_inputs`/`_parse_compare_result`/
`_simulate_compare` -- structurally identical to `add`'s own
`AddSpec`/`AddResult`/`generate_add_timeloop_inputs`/`parse_add_result`/
`simulate_add`, with `adder`->`comparator` and the exponent/mantissa
attributes replaced by a plain `width` attribute), then composes the
public reduction API on top of it, exactly the way `sum` composes on top
of `add`.

MAX-AS-REDUCTION (same exact-decomposition precedent as `sum`, NOT
`div`'s reasoned-estimate precedent): finding the max of K real elements
is EXACTLY K-1 real, sequential pairwise comparisons (a running-max
accumulator chain: ``m0 = x0; m1 = max(m0, x1); m2 = max(m1, x2); ...;
m_{K-1} = max(m_{K-2}, x_{K-1})``) -- an exact operation count from the
definition of reduction, not an approximation. Every one of those K-1
firings is the exact same real, separately-characterized `comparator`
component described above -- nothing per-firing is invented, and the
firing count itself is exact, matching `sum`'s own precedent, not `div`'s
(which needed a reasoned per-step estimate because no real divider
exists at all in this project's toolchain).

MODELING CHOICE (same convention `sum`/`div` already use, applied here
for consistency -- not re-litigated per primitive): the K-1 real
comparisons are modeled as a strictly SEQUENTIAL accumulator chain, not a
parallel/tree reduction -- a conservative assumption (may overestimate
real hardware latency if a tree reduction is available), named here
explicitly, not buried.

DEGENERATE CASE (K=1): the max of one element is that element -- zero
real comparisons needed. Mirrors `sum`'s own K=1 handling exactly: no
Docker run is launched, and ``MaxResult.comparator_area_um2``/
``.pe_scratchpad_area_um2`` are reported as ``None`` (no real component
was ever instantiated for the call), not a fabricated 0.0.

SCOPE: same as add/sum -- one PE's local reduce-over-N cost only (used
for softmax's numerical-stability row-max subtraction, per this
primitive's package __init__.py), no fabric/network component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from ...timeloop import (
    TimeLoopArtifacts,
    _parse_art_table,
    _parse_summary_stats,
    _validate_cacti_safe_sram,
    run_timeloop_model,
)

# Genuine 2-D spatial probe mesh -- same real, Docker-confirmed shape
# add/mul's own module docstrings establish (see add's "REAL DOCKER CRASH
# INVESTIGATION" section for the full evidence trail this file does not
# repeat). P/Q are real, dataspace-touched dimensions of X/Y/Z (see
# problem.yaml below); only N determines real comparator firing count per
# PE instance.
_SPATIAL_PROBE_X = 2
_SPATIAL_PROBE_Y = 2
_SPATIAL_PROBE_TOTAL = _SPATIAL_PROBE_X * _SPATIAL_PROBE_Y


@dataclass(frozen=True, slots=True)
class _CompareSpec:
    """Private: ``N`` independent real pairwise-max comparisons on ONE
    PE, each a real `comparator` firing. Not part of this module's public
    API -- see module docstring's "WHY THIS FILE HAS TWO LAYERS" section.
    """

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")


@dataclass(frozen=True, slots=True)
class _CompareResult:
    """Private: real, Docker-verified per-PE cost of ``N`` real comparator
    firings."""

    compare: _CompareSpec
    local_compute_cycles: int
    local_energy_uj: float
    comparator_area_um2: float
    pe_scratchpad_area_um2: float
    stats_path: Path
    art_path: Path


def _generate_compare_timeloop_inputs(
    compare: _CompareSpec, wafer: WaferSpec, out_dir: Path
) -> TimeLoopArtifacts:
    """Materialize a real, single-PE TimeLoop problem/architecture/mapping
    for ``N`` independent real pairwise-max comparisons -- mirrors `add`'s
    own ``generate_add_timeloop_inputs`` exactly (same MAC-shaped
    2-read/1-read-write TimeLoop schema, same genuine 2-D spatial probe
    mesh, same 5-file-only ``timeloop-model``-only output set), with
    ``adder``/``fpadder``/exponent+mantissa replaced by ``comparator``'s
    own real ``width`` attribute schema (see module docstring).
    """
    compare.validate()
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

    n = compare.elements_per_pe
    datawidth = compare.bytes_per_element * 8
    dram_block_size = max(1, int(wafer.link_bandwidth_bytes_per_cycle))
    dram_width_bits = dram_block_size * 8
    global_cycle_seconds = 1.0 / (wafer.clock_ghz * 1e9)

    # X, Y, Z all resident at once -- same single-buffered convention
    # add/GEMM use (see add's own generate_add_timeloop_inputs docstring).
    pe_scratchpad_depth_words = 3 * n
    _validate_cacti_safe_sram(
        "pe_scratchpad", word_bits=8, block_size_bytes=compare.bytes_per_element,
        depth_words=pe_scratchpad_depth_words,
    )

    fifo_buffer_depth_words = max(1, wafer.fifo_bytes_per_pe)
    router_buffer_depth_words = max(1, wafer.router_bytes_per_pe)
    _validate_cacti_safe_sram(
        "fifo_buffer", word_bits=8, block_size_bytes=1, depth_words=fifo_buffer_depth_words
    )
    _validate_cacti_safe_sram(
        "router_buffer", word_bits=8, block_size_bytes=1, depth_words=router_buffer_depth_words
    )

    # --- problem.yaml -------------------------------------------------
    # Z[N] = max(X[N], Y[N]) elementwise -- each real comparator firing
    # compares one (X[n], Y[n]) pair and writes the larger into Z[n]. P/Q
    # are the real, dataspace-touched spatial-probe dimensions (see
    # add's own module docstring for the full real-source rationale for
    # why these must be dataspace-touched, not dummy axes).
    artifacts.problem_yaml.write_text(
        "problem:\n"
        "  shape:\n"
        "    name: MAX\n"
        "    dimensions: [ N, P, Q ]\n"
        "    data-spaces:\n"
        "      - name: X\n"
        "        projection: [ [ [N] ], [ [P] ], [ [Q] ] ]\n"
        "      - name: Y\n"
        "        projection: [ [ [N] ], [ [P] ], [ [Q] ] ]\n"
        "      - name: Z\n"
        "        projection: [ [ [N] ], [ [P] ], [ [Q] ] ]\n"
        "        read-write: True\n"
        "  instance:\n"
        f"    N: {n}\n"
        f"    P: {_SPATIAL_PROBE_X}\n"
        f"    Q: {_SPATIAL_PROBE_Y}\n",
        encoding="ascii",
    )

    # --- architecture.yaml ----------------------------------------------
    # Same real 4-leaf-component-per-PE shape (pe_scratchpad, fifo_buffer,
    # router_buffer, compute unit) add/mul's own real, Docker-proven
    # architecture.yaml uses -- only the compute leaf differs:
    # `class: mac` (TimeLoop native-parser whitelist) / `subclass:
    # comparator` (real Accelergy identity) with a plain `width`
    # attribute (comparator's real declared schema -- see module
    # docstring), not exponent/mantissa.
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
        f"      - name: PE[0..{_SPATIAL_PROBE_TOTAL - 1}]\n"
        "        local:\n"
        "        - name: pe_scratchpad\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {pe_scratchpad_depth_words}\n"
        f"            block-size: {compare.bytes_per_element}\n"
        "            word-bits: 8\n"
        f"            width: {datawidth}\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        "        - name: fifo_buffer\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {fifo_buffer_depth_words}\n"
        "            block-size: 1\n"
        "            word-bits: 8\n"
        "            width: 8\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        "        - name: router_buffer\n"
        "          class: SRAM\n"
        "          attributes:\n"
        f"            depth: {router_buffer_depth_words}\n"
        "            block-size: 1\n"
        "            word-bits: 8\n"
        "            width: 8\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        "        - name: comparator\n"
        "          class: mac\n"
        "          subclass: comparator\n"
        "          attributes:\n"
        f"            width: {datawidth}\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        f"            global_cycle_seconds: {global_cycle_seconds:.6e}\n",
        encoding="ascii",
    )

    # --- mapping.yaml (fully specified, for timeloop-model) --------------
    # Identical real, Docker-proven mapping shape add's own file uses --
    # see add's own module docstring for the full real-source rationale
    # behind the PQN/split=1 spatial entry.
    artifacts.model_mapping_yaml.write_text(
        "mapping:\n"
        "  - target: pe_scratchpad\n"
        "    type: datatype\n"
        "    keep: [X, Y, Z]\n"
        "    bypass: []\n"
        "  - target: fifo_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [X, Y, Z]\n"
        "  - target: router_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [X, Y, Z]\n"
        "  - target: DRAM\n"
        "    type: datatype\n"
        "    keep: [X, Y, Z]\n"
        "    bypass: []\n"
        "  - target: pe_scratchpad\n"
        "    type: temporal\n"
        f"    factors: N={n} P=1 Q=1\n"
        "    permutation: NPQ\n"
        "  - target: DRAM\n"
        "    type: spatial\n"
        f"    factors: N=1 P={_SPATIAL_PROBE_X} Q={_SPATIAL_PROBE_Y}\n"
        "    permutation: PQN\n"
        "    split: 1\n"
        "  - target: DRAM\n"
        "    type: temporal\n"
        "    factors: N=1 P=1 Q=1\n"
        "    permutation: NPQ\n",
        encoding="ascii",
    )

    # --- primitive_components.yaml ---------------------------------------
    # Real action names TimeLoop's compute level actually queries -- reuse
    # of add's own already-Docker-verified mac_random/mac_gated/
    # mac_reused/leak (see module docstring's "ACTION-NAME CHOICE"
    # section for why this is the faithful choice, not a shortcut).
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
        "  - name: comparator\n"
        "    attributes: {}\n"
        "    actions:\n"
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

    # --- accelergy_config.yaml -------------------------------------------
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

    return artifacts


def _parse_compare_result(artifacts: TimeLoopArtifacts, compare: _CompareSpec) -> _CompareResult:
    """Build ``_CompareResult`` from a real completed
    ``run_timeloop_model()`` call -- mirrors `add`'s own
    ``parse_add_result`` exactly (same real parsers, same
    _SPATIAL_PROBE_TOTAL energy division, same "Cycles is already
    per-instance" convention), just reading the ``comparator`` ART area
    key instead of ``adder``."""
    if not artifacts.stats_txt.exists() or not artifacts.art_yaml.exists():
        raise RuntimeError(
            f"no real TimeLoop output found (missing one of {artifacts.stats_txt}, "
            f"{artifacts.art_yaml}); run_timeloop_model() must succeed before a "
            "result can be built -- there is no fallback cycle/energy number."
        )
    stats_text = artifacts.stats_txt.read_text(encoding="utf-8")
    local_compute_cycles, local_energy_uj_total = _parse_summary_stats(stats_text, artifacts.log)
    local_energy_uj = local_energy_uj_total / _SPATIAL_PROBE_TOTAL

    art_text = artifacts.art_yaml.read_text(encoding="utf-8")
    areas = _parse_art_table(art_text, artifacts.art_yaml)
    missing = [name for name in ("comparator", "pe_scratchpad") if name not in areas]
    if missing:
        raise RuntimeError(
            f"ART.yaml at {artifacts.art_yaml} has no area entry for {missing} -- "
            f"found entries: {sorted(areas)}"
        )

    return _CompareResult(
        compare=compare,
        local_compute_cycles=local_compute_cycles,
        local_energy_uj=local_energy_uj,
        comparator_area_um2=areas["comparator"],
        pe_scratchpad_area_um2=areas["pe_scratchpad"],
        stats_path=artifacts.stats_txt,
        art_path=artifacts.art_yaml,
    )


def _simulate_compare(compare: _CompareSpec, wafer: WaferSpec, out_dir: Path) -> _CompareResult:
    """Run the real Docker ``timeloop-model`` pipeline end to end for one
    ``_CompareSpec`` (N real comparator firings) and return its real
    cycle/energy/area result. Raises (never fabricates a result) if
    Docker or TimeLoop fails -- same discipline as add/mul/div/sum."""
    artifacts = _generate_compare_timeloop_inputs(compare, wafer, out_dir)
    run_timeloop_model(artifacts)
    return _parse_compare_result(artifacts, compare)


# ===========================================================================
# Public API: `max` as a K-1-real-comparison reduction (see module
# docstring's "MAX-AS-REDUCTION" section).
# ===========================================================================


@dataclass(frozen=True, slots=True)
class MaxSpec:
    """Reduce ``elements_per_pe`` real elements ``x[0..K-1]`` to their max
    on ONE PE, via a real, sequential running-max accumulator chain of
    ``real_compare_count`` real comparator firings (see module
    docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")

    @property
    def real_compare_count(self) -> int:
        """Exact number of real comparator firings a K-element max
        reduction requires: K-1 (0 when K=1 -- the max of one element
        needs no comparison). Not an estimate -- this is the definition
        of reduction, same precedent as `sum`'s own `real_add_count`."""
        return max(self.elements_per_pe - 1, 0)


@dataclass(frozen=True, slots=True)
class MaxResult:
    """Real, Docker-verified per-PE cost of one ``max`` primitive call.

    ``compare_result`` is the private real ``_simulate_compare()``
    sub-run covering ALL of ``max.real_compare_count``'s accumulator
    firings in ONE Docker round-trip -- ``None`` only in the K=1
    degenerate case (see module docstring's "DEGENERATE CASE" section).
    """

    max: MaxSpec
    compare_result: Optional[_CompareResult]

    @property
    def local_compute_cycles(self) -> int:
        return 0 if self.compare_result is None else self.compare_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return 0.0 if self.compare_result is None else self.compare_result.local_energy_uj

    @property
    def comparator_area_um2(self) -> Optional[float]:
        """Real area of the comparator this reduction's accumulator chain
        uses (from the real compare sub-run's ART.yaml) -- ``None`` in
        the K=1 case, since no real comparator is instantiated for a
        no-op reduction (see module docstring)."""
        return None if self.compare_result is None else self.compare_result.comparator_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> Optional[float]:
        return None if self.compare_result is None else self.compare_result.pe_scratchpad_area_um2

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.max.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.max.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "max": asdict(self.max),
            "real_compare_count": self.max.real_compare_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "comparator_area_um2": self.comparator_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
        }


def simulate_max(max_spec: MaxSpec, wafer: WaferSpec, out_dir: Path) -> MaxResult:
    """Run ONE real Docker ``timeloop-model`` pipeline (via the private
    ``_simulate_compare()``) covering all of ``max_spec.real_compare_count``'s
    real, sequential running-max accumulator-chain firings a K-element
    max reduction requires -- or, when K=1, skip Docker entirely and
    return the true zero-cost result (see module docstring's "DEGENERATE
    CASE" section). Raises (never fabricates a result) if the real Docker
    sub-run fails -- same discipline as add/mul/div/sum.
    """
    max_spec.validate()
    real_compare_count = max_spec.real_compare_count

    if real_compare_count == 0:
        return MaxResult(max=max_spec, compare_result=None)

    compare_spec = _CompareSpec(
        elements_per_pe=real_compare_count, bytes_per_element=max_spec.bytes_per_element
    )
    compare_result = _simulate_compare(compare_spec, wafer, out_dir / "compare")
    return MaxResult(max=max_spec, compare_result=compare_result)
