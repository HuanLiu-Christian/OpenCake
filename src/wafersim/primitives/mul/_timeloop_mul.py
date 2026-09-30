"""Real, Docker-backed TimeLoop/Accelergy cost model for the `mul` primitive.

Mirrors this project's `add` primitive (see ``wafersim.primitives.add.
_timeloop_add``) for a single elementwise-multiply compute level: same
"one PE's local N-element op" scope, same MAC-shaped 2-read/1-read-write
TimeLoop schema, same 2x2 dummy spatial-replication mesh (P/Q, each a
real dataspace-touched dimension of X/Y/Z -- see add's module docstring's
"REAL DOCKER CRASH INVESTIGATION" section for why), and the same
``class: mac`` / ``subclass: <real class>`` split add's round-9 fix
established as required: TimeLoop's native C++ topology parser
(``src/model/topology.cpp``'s ``ParseTreeSpecs``) recognizes a `local:`
component as the arithmetic/compute level ONLY if its literal `class:`
string contains "mac"/"intmac"/"fpmac"/"compute" as a substring -- a
hardcoded whitelist in ``include/model/topology.hpp``'s
``computeClasses`` -- and drops the component from the topology entirely
otherwise, regardless of mesh shape or dataspace projections. Accelergy's
own real parser (``accelergy/arch_dict_2_obj.py``) prefers an explicit
``subclass:`` over ``class:`` when resolving a component's real
estimation identity, so ``class: mac`` (satisfies TimeLoop) + a real
``subclass:`` (satisfies Accelergy) is what both files below use. This
file is built directly on that lesson instead of re-deriving it from
scratch -- there is no multi-round crash history here, unlike add's.

REAL ACCELERGY COMPONENT USED -- ``fpmultiplier`` (a real, separately-
characterized floating-point multiplier -- NOT GEMM's intmac/mac reused
as an approximation, and NOT add's fpadder repurposed either): confirmed
by reading Accelergy-Project/accelergy-aladdin-plug-in's real
``aladdin_table.py`` source directly. ``get_supported_components()``
lists ``fpmultiplier`` as its own real class with attributes
``exponent``/``mantissa``, distinct from ``fpadder``/``fpmac``/
``intmultiplier``/``intmac``/``intadder``. Its real
``fpmultiplier_estimate_energy``/``fpmultiplier_estimate_area`` select
``data/fp_sp_multiplier.csv`` -- a real "3 stage floating point single
precision multiplier" characterization (confirmed by reading that CSV's
real header/rows directly, e.g. its `latency(ns)=1` row: dynamic
energy 35.3 pJ, area 4.36e+03 um^2) -- whenever
``exponent + mantissa <= 32``, so ``exponent=8, mantissa=24`` (the same
fp32 split add uses) hits its 32-bit row exactly, no
``oneD_quadratic_interpolation`` scaling away from the table value.

REAL ACTION NAMES -- unlike ``fpadder_estimate_energy`` (which, per
add's own docstring, never branches on the action name beyond literally
"leak"), ``fpmultiplier_estimate_energy``'s real body DOES special-case
the literal strings ``"mult_gated"`` (remapped to idle/leak energy
before the CSV lookup) and ``"mult_reused"`` (a 0.85x discount on the
returned dynamic energy). TimeLoop's own C++ engine
(``include/model/topology.hpp``'s compute-event-to-action-name alias
map, read directly) maps its 3 internal compute events to whichever of
several accepted aliases a component actually declares:
``random_compute`` accepts any of
``{"mac_random","mult_random","mac","mult","compute"}``;
``gated_compute``/``skipped_compute`` accept
``{"mac_gated","mult_gated","compute_gated","mac","mult","compute",...}``.
So declaring ``mult_random``/``mult_gated``/``leak`` below (instead of
blindly reusing add's ``mac_random``/``mac_gated`` names) is both
structurally valid AND activates fpmultiplier's real gated-energy
discount, which ``mac_gated`` would not (a literal string compare that
would simply never match ``"mult_gated"``). ``mult_reused`` is declared
too for completeness even though TimeLoop's own ``arithmeticOperation
Types`` (``random_compute``/``skipped_compute``/``gated_compute`` --
no "reused" event at all) never actually queries it, the same situation
add's own unused ``mac_reused`` declaration was already in.

WHY THE SAME MAC-SHAPED SCHEMA FOR A PLAIN MULTIPLY: see add's module
docstring's parallel explanation in full -- TimeLoop's classic engine has
no notion of what the two read operands' mathematical relationship is;
it only counts how many times the compute-level component fired. With N
as the only real problem dimension, Z[n] = X[n] * Y[n] fires the real
fpmultiplier exactly once per element -- one real multiply per point,
not a relabeled add or GEMM run.

SCOPE: same as add -- this models ONE PE's local multiply-over-N cost
only, no fabric/network component wired up (RMSNorm scaling, SwiGLU
gating, and causal-mask multiply -- this primitive's real uses per this
package's own __init__.py -- are all PE-local elementwise ops on
already-co-located operands, matching add's own SCOPE note). Extending
to a full active-PE-count wafer run is the same documented follow-up
add's docstring defers, not done here either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ...specs import WaferSpec
from ...timeloop import (
    TimeLoopArtifacts,
    _parse_art_table,
    _parse_summary_stats,
    _validate_cacti_safe_sram,
    run_timeloop_model,
)

# IEEE-754 single-precision split matching this project's real
# MulSpec/AddSpec/GemmSpec default bytes_per_element=4 (32-bit datawidth):
# 8 stored exponent bits + 24 significand bits (23 stored + 1 implicit
# leading bit). exponent + mantissa == 32 exactly, so aladdin_table.py's
# fpmultiplier handler selects fp_sp_multiplier.csv's real 32-bit row
# with no interpolation scaling away from the table value (see this
# module's docstring).
_FP32_EXPONENT_BITS = 8
_FP32_MANTISSA_BITS = 24

# Same genuine 2-D spatial probe mesh as add's round-8/9 (now confirmed-
# working, see this module's docstring) -- _SPATIAL_PROBE_X PEs along
# meshX, _SPATIAL_PROBE_Y "rows" along the implied meshY. P and Q are
# real, dataspace-touched dimensions of X, Y, AND Z (see problem.yaml
# below): each of the _SPATIAL_PROBE_TOTAL spatial instances computes its
# own independent N-element multiply (no cross-instance data reuse).
# parse_mul_result() below divides the parsed local_energy_uj by
# _SPATIAL_PROBE_TOTAL to recover the true single-PE number; local_
# compute_cycles is NOT divided (Cycles is already per-instance -- same
# convention add/GemmMappingContract document).
_SPATIAL_PROBE_X = 2
_SPATIAL_PROBE_Y = 2
_SPATIAL_PROBE_TOTAL = _SPATIAL_PROBE_X * _SPATIAL_PROBE_Y


@dataclass(frozen=True, slots=True)
class MulSpec:
    """Problem ``Z[N] = X[N] * Y[N]``, evaluated for ONE PE's local N."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")


@dataclass(frozen=True, slots=True)
class MulResult:
    """Real, Docker-verified per-PE cost of one ``mul`` primitive call."""

    mul: MulSpec
    local_compute_cycles: int
    local_energy_uj: float
    multiplier_area_um2: float
    pe_scratchpad_area_um2: float
    stats_path: Path
    art_path: Path

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.mul.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.mul.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "mul": asdict(self.mul),
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "multiplier_area_um2": self.multiplier_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "stats_path": str(self.stats_path),
            "art_path": str(self.art_path),
        }


def generate_mul_timeloop_inputs(
    mul: MulSpec, wafer: WaferSpec, out_dir: Path
) -> TimeLoopArtifacts:
    """Materialize a real, single-PE TimeLoop problem/architecture/mapping
    for ``Z[N] = X[N] * Y[N]`` -- see this module's docstring for why the
    same MAC-shaped 2-read/1-read-write TimeLoop schema GEMM/add use is a
    faithful way to express a plain elementwise multiply here.

    Only ``timeloop-model`` inputs are written (single, fully-specified
    mapping -- no ``timeloop-mapper`` search makes sense for a single-PE,
    single-dimension problem with nothing to tile), same as add.
    """
    mul.validate()
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

    n = mul.elements_per_pe
    datawidth = mul.bytes_per_element * 8
    dram_block_size = max(1, int(wafer.link_bandwidth_bytes_per_cycle))
    dram_width_bits = dram_block_size * 8
    global_cycle_seconds = 1.0 / (wafer.clock_ghz * 1e9)

    # X, Y, Z all resident at once (single-buffered -- same real
    # execution model add/GEMM use; see wafersim.timeloop's
    # generate_timeloop_inputs docstring).
    pe_scratchpad_depth_words = 3 * n
    _validate_cacti_safe_sram(
        "pe_scratchpad", word_bits=8, block_size_bytes=mul.bytes_per_element,
        depth_words=pe_scratchpad_depth_words,
    )

    # fifo_buffer/router_buffer: same real WaferSpec-sized SRAM components
    # add/GEMM's architecture.yaml always declare alongside pe_scratchpad
    # and the compute unit -- exist purely so this PE's leaf-component
    # shape matches the one TimeLoop has actually been proven to accept
    # (see add's own module docstring for the real Docker crash history
    # behind that requirement), not to hold any of X/Y/Z's data.
    fifo_buffer_depth_words = max(1, wafer.fifo_bytes_per_pe)
    router_buffer_depth_words = max(1, wafer.router_bytes_per_pe)
    _validate_cacti_safe_sram(
        "fifo_buffer", word_bits=8, block_size_bytes=1, depth_words=fifo_buffer_depth_words
    )
    _validate_cacti_safe_sram(
        "router_buffer", word_bits=8, block_size_bytes=1, depth_words=router_buffer_depth_words
    )

    # --- problem.yaml -------------------------------------------------
    # Z[N] = X[N] * Y[N]. `P`/`Q` are the same real, dataspace-touched
    # 2x2 spatial-replication mesh add's round-8 fix established (each of
    # X, Y, Z is a genuine rank-3 [N, P, Q] data-space: every spatial
    # instance computes its own independent N-element multiply, no
    # cross-instance reuse) -- N alone determines the real fpmultiplier
    # firing count per PE instance.
    artifacts.problem_yaml.write_text(
        "problem:\n"
        "  shape:\n"
        "    name: MUL\n"
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
    # Same genuine 2-D PE mesh as add (meshX=_SPATIAL_PROBE_X implying
    # meshY=_SPATIAL_PROBE_Y via every leaf's meshX -- REQUIRED, see add's
    # docstring for the real crash this triggers if omitted).
    #
    # `multiplier`'s `class: mac` / `subclass: fpmultiplier` is add's
    # round-9 fix applied directly (not re-derived): `class: mac`
    # satisfies TimeLoop's native topology parser's hardcoded compute-
    # class whitelist; `subclass: fpmultiplier` is what Accelergy's real
    # parser (arch_dict_2_obj.py) actually resolves the component's
    # estimation identity from, keeping the real, separately-
    # characterized fp multiplier (exponent/mantissa attributes below are
    # untouched, unaffected by the class/subclass split).
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
        f"            block-size: {mul.bytes_per_element}\n"
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
        "        - name: multiplier\n"
        "          class: mac\n"
        "          subclass: fpmultiplier\n"
        "          attributes:\n"
        f"            exponent: {_FP32_EXPONENT_BITS}\n"
        f"            mantissa: {_FP32_MANTISSA_BITS}\n"
        f"            meshX: {_SPATIAL_PROBE_X}\n"
        f"            global_cycle_seconds: {global_cycle_seconds:.6e}\n",
        encoding="ascii",
    )

    # --- mapping.yaml (fully specified, for timeloop-model) --------------
    # Identical shape to add's real, Docker-verified mapping.yaml -- see
    # that module's docstring for the full split/permutation reasoning
    # (first `split` characters of `permutation` are TimeLoop's X-group,
    # product must equal meshX; the rest are the Y-group, product must
    # equal the implied meshY). No "multiplier" mapping entry, same real
    # constraint add/GEMM document: TimeLoop's mapping schema only
    # accepts storage levels as targets, never the arithmetic unit.
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
    # `mult_random`/`mult_gated`/`mult_reused`/`leak` -- NOT add's
    # `mac_random`/`mac_gated`/`mac_reused` names reused verbatim. See
    # this module's docstring's "REAL ACTION NAMES" section: TimeLoop's
    # own compute-event alias map accepts either mac_* or mult_* for its
    # random_compute/gated_compute events, but fpmultiplier_estimate_
    # energy's real body specifically special-cases the literal string
    # "mult_gated" (and "mult_reused") -- so mult_* is the more faithful
    # choice here, not just a cosmetic rename.
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
        "  - name: fpmultiplier\n"
        "    attributes: {}\n"
        "    actions:\n"
        "      - name: mult_random\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n"
        "      - name: mult_gated\n"
        "        arguments:\n"
        "          action_latency_cycles: 1..1\n"
        "      - name: mult_reused\n"
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


def parse_mul_result(artifacts: TimeLoopArtifacts, mul: MulSpec) -> MulResult:
    """Build MulResult from a real completed ``run_timeloop_model()`` call.

    Reuses ``wafersim.timeloop``'s own real-output parsers, same
    rationale as add's ``parse_add_result()``.
    """
    if not artifacts.stats_txt.exists() or not artifacts.art_yaml.exists():
        raise RuntimeError(
            f"no real TimeLoop output found (missing one of {artifacts.stats_txt}, "
            f"{artifacts.art_yaml}); run_timeloop_model() must succeed before a "
            "result can be built -- there is no fallback cycle/energy number."
        )
    stats_text = artifacts.stats_txt.read_text(encoding="utf-8")
    local_compute_cycles, local_energy_uj_total = _parse_summary_stats(stats_text, artifacts.log)
    # local_energy_uj_total is TimeLoop's real aggregate across all
    # _SPATIAL_PROBE_TOTAL active instances -- divide back down to the
    # true single-PE number MulResult promises. local_compute_cycles is
    # NOT divided: Cycles is already per-instance (same convention add/
    # GemmMappingContract document).
    local_energy_uj = local_energy_uj_total / _SPATIAL_PROBE_TOTAL

    art_text = artifacts.art_yaml.read_text(encoding="utf-8")
    areas = _parse_art_table(art_text, artifacts.art_yaml)
    missing = [name for name in ("multiplier", "pe_scratchpad") if name not in areas]
    if missing:
        raise RuntimeError(
            f"ART.yaml at {artifacts.art_yaml} has no area entry for {missing} -- "
            f"found entries: {sorted(areas)}"
        )

    return MulResult(
        mul=mul,
        local_compute_cycles=local_compute_cycles,
        local_energy_uj=local_energy_uj,
        multiplier_area_um2=areas["multiplier"],
        pe_scratchpad_area_um2=areas["pe_scratchpad"],
        stats_path=artifacts.stats_txt,
        art_path=artifacts.art_yaml,
    )


def simulate_mul(mul: MulSpec, wafer: WaferSpec, out_dir: Path) -> MulResult:
    """Run the real Docker ``timeloop-model`` pipeline end to end for one
    ``MulSpec`` and return its real cycle/energy/area result. Raises
    (never fabricates a result) if Docker or TimeLoop fails -- same
    discipline as add/``wafersim.timeloop``."""
    artifacts = generate_mul_timeloop_inputs(mul, wafer, out_dir)
    run_timeloop_model(artifacts)
    return parse_mul_result(artifacts, mul)
