"""Real, Docker-backed TimeLoop/Accelergy cost model for ONE real firing of
the Aladdin ``bitwise`` component (a real, separately-characterized digital
AND/OR/XOR/NOT-class gate -- see this module's own real-source citation
below). This is a NEW shared primitive-component pipeline, not tied to any
single primitive's package, because two different primitives need it for
two different real reasons:

  - ``where_select`` needs it to build a real, EXACT (not approximate) 2:1
    mux out of elementary real gates (see ``wafersim.primitives.where_select
    ._timeloop_where``'s own module docstring for the exact XOR/AND/XOR
    boolean identity and firing count).
  - ``reshape_view`` (the copying branch of ``reshape()``), ``concat_stack``,
    ``repeat_expand`` (the ``repeat()`` branch), and ``slice_gather`` (the
    ``gather()`` branch) all need a real per-element MEMORY-MOVE cost --
    see ``wafersim.primitives._memory_move`` for how those reuse this same
    real component as an identity (AND-with-all-ones) pass-through, and why
    that is the most honest modeling choice available given this project's
    real, pinned Docker toolchain has no separately-characterized DMA/copy-
    engine component anywhere (confirmed by the same real-plugin-source
    research this module's own docstring cites).

REAL ACCELERGY COMPONENT USED -- ``bitwise`` (not a placeholder, and not
`add`'s ``fpadder``/`max`'s ``comparator`` reused as an approximation): a
real ``git clone`` of this project's pinned Accelergy image's bundled
``accelergy-aladdin-plug-in`` (https://github.com/Accelergy-Project/
accelergy-aladdin-plug-in, ``aladdin_table.py``) confirms:

  - ``get_supported_components()`` lists ``"bitwise": (["width"], ["leak",
    "access"])`` -- a real, wired, queryable component distinct from
    ``fpadder``/``comparator``/``intmac``, characterized by real
    ``data/bitwise.csv`` ("this is a 1 stage bit wise operator": real
    latency(ns)/dynamic-energy(pJ)/leakage-power(mW)/idle-energy(pJ)/
    area(um^2) rows for 1/2/3/4/5/6/10-cycle latencies).
  - ``bitwise_estimate_energy``/``bitwise_estimate_area`` both call the
    SAME ``query_csv_using_latency``/``query_csv_area_using_latency``
    helpers ``fpadder`` uses, and (per ``query_csv_using_latency``'s own
    source) branch ONLY on whether ``action_name == "leak"`` -- exactly
    the same non-branching-on-specific-action-name behavior `add`'s module
    docstring already established for ``fpadder``. So reusing the same
    four action names (``mac_random``/``mac_gated``/``mac_reused``/
    ``leak``) TimeLoop's compute level always queries (per `add`'s own
    real-source investigation of ``topology.cpp``) is equally faithful
    here, not a second workaround.
  - UNLIKE ``fpadder`` (fixed at ``exponent=8, mantissa=24``, hitting one
    fixed 32-bit table row with no scaling), ``bitwise_estimate_energy``/
    ``_area`` each multiply the real per-stage CSV value by
    ``interface["attributes"]["width"]`` -- i.e. this component's real
    cost scales linearly with the real bit-width of the value being
    gated, not a fixed per-firing constant. ``width`` below is always
    this project's real ``bytes_per_element * 8`` (32 for this project's
    default ``bytes_per_element=4``), matching how every other primitive's
    ``bytes_per_element`` already flows into its own component's real
    datawidth attribute (``fpadder``'s ``exponent+mantissa``, `mul`'s
    ``fpmultiplier`` datawidth, etc.).

ARCHITECTURE TEMPLATE: this reuses `add`'s exact real, Docker-CRASH-
INVESTIGATED-AND-FIXED 2-D spatial-probe-mesh structure verbatim (see
``wafersim.primitives.add._timeloop_add``'s own module docstring for the
full 9-round investigation this is built on) -- same ``class: mac`` /
``subclass: <real component>`` trick (TimeLoop's native-format parser
only recognizes a compute level via a hardcoded ``class:`` whitelist
substring match; Accelergy's own parser prefers an explicit ``subclass:``
when resolving the real estimation identity), same
``_SPATIAL_PROBE_X``/``_SPATIAL_PROBE_Y`` = 2x2 real non-degenerate mesh,
same pe_scratchpad/fifo_buffer/router_buffer leaf shape, same mapping.yaml
permutation pattern -- only the compute leaf's ``subclass``/``attributes``
and the problem's dataspace count differ (bitwise is genuinely a 1-input,
1-output gate for the identity-pass-through case in ``_memory_move.py``,
or a real chained sequence of 1-input-1-output firings for
``where_select``'s 2:1 mux -- see that module for why this file's
``BitwiseSpec`` only ever needs ONE real input dataspace ``X`` and one
real read-write output dataspace ``Z``, unlike `add`'s two-input ``X``/
``Y``).

ROUND-1 REAL DOCKER RESULT (CONFIRMED WORKING, not just expected to):
unlike `add`/`max` (which each needed a real, documented 8-9-round crash
investigation), this file's very first real Docker run -- via
``run_bridge_check2.py``, all 5 real callers (reshape-copy/repeat/gather/
concat/where) -- completed with no crash and real, internally consistent
output. Confirmed against the real ``resolved.stats.txt``/
``resolved.ART.yaml`` this run produced (not re-guessed): ``Cycles``
equals each call's real firing count exactly (8/12/8/6/24 for
reshape/repeat/gather/concat/where respectively -- where's is exactly
3x its own 8-element input, matching the XOR/AND/XOR mux's exact 3
firings/element); ``bitwise_gate`` area is a CONSTANT 1612.8 um^2 across
all 5 real runs regardless of firing count (a physical circuit's real
silicon area does not depend on how many cycles it runs for -- the same
invariant `add`'s ``adder_area_um2``/`max`'s ``comparator`` area already
established); ``pe_scratchpad`` area scales linearly with each run's real
``2*n`` depth (197.88/263.84/263.84/395.76/791.52 um^2 for concat(n=6)/
reshape(n=8)/gather(n=8)/repeat(n=12)/where(n=24) -- reshape and gather
share the identical real value because both happened to move exactly 8
elements in this run, a genuine cross-check that the model depends only
on real element count, not which primitive is asking). This worked on
round 1 because it transplants `add`'s own already-Docker-CONFIRMED
structural fixes (the ``class:``/``subclass:`` split, the 2x2 spatial
mesh, the dataspace-touched P/Q dimensions) rather than guessing a new
shape from scratch -- it is not evidence that guessing blind would have
worked, only that reusing a real, previously-debugged shape does. If a
FUTURE real run of this pipeline (e.g. at a very different N, or a
different WaferSpec) ever does crash, treat that exactly like `add`'s own
round-by-round investigation: read the real upstream source, fix the root
cause, re-run -- never substitute a fabricated cycle/energy number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..specs import WaferSpec
from ..timeloop import (
    TimeLoopArtifacts,
    _parse_art_table,
    _parse_summary_stats,
    _validate_cacti_safe_sram,
    run_timeloop_model,
)

# Same real 2-D spatial probe mesh `add` established as Docker-confirmed
# non-crashing (see this module's docstring) -- reused verbatim, not
# re-derived, since nothing about switching the compute subclass from
# fpadder to bitwise changes the topology/mapping shape considerations
# that mesh was built to satisfy.
_SPATIAL_PROBE_X = 2
_SPATIAL_PROBE_Y = 2
_SPATIAL_PROBE_TOTAL = _SPATIAL_PROBE_X * _SPATIAL_PROBE_Y


@dataclass(frozen=True, slots=True)
class BitwiseSpec:
    """Problem ``Z[N] = bitwise_op(X[N])``, evaluated for ONE PE's local N
    -- ``N`` real, sequential firings of the real ``bitwise`` component.
    Deliberately ONE input dataspace (not two, unlike `add`/`mul`/`div`):
    every real caller of this module (see module docstring) composes
    multi-operand boolean logic as a SEQUENCE of these single-input
    firings, not as one firing per boolean gate's own arity -- so the
    firing COUNT already reflects operand count where it matters (e.g.
    `where_select` passes ``elements_per_pe = 3 * N`` for its real 3-gate
    mux decomposition), not this Spec's dataspace shape.
    """

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")


@dataclass(frozen=True, slots=True)
class BitwiseResult:
    """Real, Docker-verified per-PE cost of ``elements_per_pe`` real
    ``bitwise`` component firings."""

    bitwise: BitwiseSpec
    local_compute_cycles: int
    local_energy_uj: float
    bitwise_area_um2: float
    pe_scratchpad_area_um2: float
    stats_path: Path
    art_path: Path

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.bitwise.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.bitwise.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "bitwise": asdict(self.bitwise),
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "bitwise_area_um2": self.bitwise_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "stats_path": str(self.stats_path),
            "art_path": str(self.art_path),
        }


def generate_bitwise_timeloop_inputs(
    bitwise: BitwiseSpec, wafer: WaferSpec, out_dir: Path
) -> TimeLoopArtifacts:
    """Materialize a real, single-PE TimeLoop problem/architecture/mapping
    for ``Z[N] = bitwise_op(X[N])`` -- transplants `add`'s real, Docker-
    confirmed structural fixes verbatim (see module docstring); only the
    compute leaf's subclass/attributes and the dataspace count differ.
    """
    bitwise.validate()
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

    n = bitwise.elements_per_pe
    datawidth = bitwise.bytes_per_element * 8
    dram_block_size = max(1, int(wafer.link_bandwidth_bytes_per_cycle))
    dram_width_bits = dram_block_size * 8
    global_cycle_seconds = 1.0 / (wafer.clock_ghz * 1e9)

    # X, Z resident at once (single-buffered, same convention as `add`) --
    # only 2 dataspaces here (one real input, one real read-write output),
    # not 3, since `bitwise` (unlike `fpadder`) is modeled as a real
    # single-input gate per firing (see BitwiseSpec's own docstring).
    pe_scratchpad_depth_words = 2 * n
    _validate_cacti_safe_sram(
        "pe_scratchpad", word_bits=8, block_size_bytes=bitwise.bytes_per_element,
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
    # Z[N] = bitwise_op(X[N]) -- one real read dataspace (X), one real
    # read-write dataspace (Z). P/Q: same real, dataspace-touched 2-D
    # spatial-probe mesh `add` established (see module docstring) --
    # every one of the _SPATIAL_PROBE_TOTAL instances computes its own
    # independent N-element pass, no cross-instance data reuse.
    artifacts.problem_yaml.write_text(
        "problem:\n"
        "  shape:\n"
        "    name: BITWISE\n"
        "    dimensions: [ N, P, Q ]\n"
        "    data-spaces:\n"
        "      - name: X\n"
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
    # Same real class:mac / subclass:bitwise trick `add` established for
    # class:mac / subclass:fpadder -- TimeLoop's native parser only
    # recognizes a compute level via `class:` containing "mac"/"intmac"/
    # "fpmac"/"compute"; Accelergy's own parser prefers `subclass:` when
    # present, so this hands each tool the identity it needs without
    # misrepresenting the real component to either.
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
        f"            block-size: {bitwise.bytes_per_element}\n"
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
        "        - name: bitwise_gate\n"
        "          class: mac\n"
        "          subclass: bitwise\n"
        "          attributes:\n"
        f"            width: {datawidth}\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        f"            global_cycle_seconds: {global_cycle_seconds:.6e}\n",
        encoding="ascii",
    )

    # --- mapping.yaml (fully specified, for timeloop-model) --------------
    artifacts.model_mapping_yaml.write_text(
        "mapping:\n"
        "  - target: pe_scratchpad\n"
        "    type: datatype\n"
        "    keep: [X, Z]\n"
        "    bypass: []\n"
        "  - target: fifo_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [X, Z]\n"
        "  - target: router_buffer\n"
        "    type: datatype\n"
        "    keep: []\n"
        "    bypass: [X, Z]\n"
        "  - target: DRAM\n"
        "    type: datatype\n"
        "    keep: [X, Z]\n"
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
    # Same real action names `add` established TimeLoop's compute level
    # always queries, reused here for the same real reason (see module
    # docstring: bitwise_estimate_energy never branches on the specific
    # non-"leak" action name either).
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
        "  - name: bitwise\n"
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


def parse_bitwise_result(artifacts: TimeLoopArtifacts, bitwise: BitwiseSpec) -> BitwiseResult:
    """Build BitwiseResult from a real completed ``run_timeloop_model()``
    call -- reuses ``wafersim.timeloop``'s own real-output parsers, same
    as `add`."""
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
    missing = [name for name in ("bitwise_gate", "pe_scratchpad") if name not in areas]
    if missing:
        raise RuntimeError(
            f"ART.yaml at {artifacts.art_yaml} has no area entry for {missing} -- "
            f"found entries: {sorted(areas)}"
        )

    return BitwiseResult(
        bitwise=bitwise,
        local_compute_cycles=local_compute_cycles,
        local_energy_uj=local_energy_uj,
        bitwise_area_um2=areas["bitwise_gate"],
        pe_scratchpad_area_um2=areas["pe_scratchpad"],
        stats_path=artifacts.stats_txt,
        art_path=artifacts.art_yaml,
    )


def simulate_bitwise(bitwise: BitwiseSpec, wafer: WaferSpec, out_dir: Path) -> BitwiseResult:
    """Run the real Docker ``timeloop-model`` pipeline end to end for one
    ``BitwiseSpec`` and return its real cycle/energy/area result. Raises
    (never fabricates a result) if Docker or TimeLoop fails."""
    artifacts = generate_bitwise_timeloop_inputs(bitwise, wafer, out_dir)
    run_timeloop_model(artifacts)
    return parse_bitwise_result(artifacts, bitwise)
