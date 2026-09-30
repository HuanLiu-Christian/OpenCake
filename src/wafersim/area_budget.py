"""Derive a WaferSpec's mesh size, SRAM capacity, and FIFO capacity from a
physical die-area budget, instead of specifying mesh_rows/mesh_cols/
sram_kib_per_pe/fifo_capacity_packets directly.

This inverts the usual direction of this project's area accounting.
Everywhere else (contract.py/wafer/area.py), a full hardware config goes IN
and real Accelergy tells you how much area it costs. Here, a total physical
die-area budget and a compute/memory/network percentage split go IN, and
this module answers: what is the LARGEST hardware config (PE count, SRAM
depth, FIFO depth) that real Accelergy confirms fits inside each category's
share of that budget?

Nothing here is estimated locally. Every accept/reject decision is made
from a real per-instance area number parsed out of an actual
`timeloop-model` Docker run's ART.yaml (see timeloop.py) -- exactly the same
real-only discipline as search.py's tile_k search. Only the control flow
(which candidate to try next) is local Python; the area numbers themselves
always come from Docker. This makes the search expensive: one real Docker
invocation per probed candidate (see MAX_SEARCH_ITERATIONS below) -- a full
derivation typically costs dozens of real `timeloop-model` runs, not one.

Category classification matches contract.py's compute_area_um2/
memory_area_um2/network_area_um2 exactly, so the same five area terms
(mac, pe_scratchpad, fifo_buffer, router_buffer, link) are bucketed the same
way here as they are when reporting the area breakdown of an already-run
GEMM (see contract.py's "compute / memory / network area classification"
comment for the full rationale of what goes in which bucket).

Search strategy, in three stages:

  1. Compute (no search -- one cheap real probe): mac_area_um2 is a real,
     structurally-FIXED per-instance number. It does not depend on GEMM
     shape, mesh size, or tiling -- confirmed project-wide: every real
     Docker run so far has reported the identical mac_area_um2 for a given
     technology_node/clock_ghz/bytes_per_element, because Aladdin_table's
     intmac estimate is a flat CSV lookup keyed only by those, never by
     mapping/mesh geometry (see PIPELINE_SUMMARY.md's TimeLoop-internals
     section). So there is nothing to search here: one real Docker run
     against a tiny throwaway probe GEMM (decoupled from the real gemm/mesh
     this module ultimately derives) is enough to learn it. From there,
     max_pes = floor(compute_area_budget_um2 / mac_area_um2) is an upper
     bound, not the mesh size directly -- see _derive_mesh for why.

     mesh_rows/mesh_cols are NOT simply floor(sqrt(max_pes)): confirmed
     against a real Docker run that generate_timeloop_inputs's mapping.yaml
     requires mesh_rows to evenly divide gemm.m and mesh_cols to evenly
     divide gemm.n EXACTLY (TimeLoop's own mapping parser rejects anything
     else -- a real run with a non-dividing mesh_rows failed with "product
     of all factors of dimension M is 1026, which is not equal to the
     dimension size of the workload 1024"). wafer/simulator.py's own Python
     model DOES support a non-dividing mesh via boundary tiles (_extent()),
     but the real TimeLoop input this project hands to Docker does not --
     every prior real run in this project's history happened to use a mesh
     size that evenly divided the GEMM (mesh_rows=mesh_cols=32 against
     M=N=1024), so this constraint was never actually exercised before this
     module started deriving arbitrary mesh sizes. So _derive_mesh searches
     only mesh_rows in divisors(gemm.m) and mesh_cols in divisors(gemm.n),
     picking the pair with the largest PE count (rows*cols <= max_pes,
     both >=2 -- see _real_mac_area_um2's docstring for why >=2), breaking
     ties toward the closest-to-square shape. For a highly composite GEMM
     shape (few, coarse divisors -- e.g. a power-of-two M/N) this can leave
     much of the compute budget unused: the achievable mesh sizes jump in
     large discrete steps rather than smoothly tracking the budget.

  2. Memory (real binary search): pe_scratchpad_area_um2 genuinely IS a
     function of its declared SRAM depth (CactiSRAM actually uses the
     `technology`/depth/width attributes, unlike the flat Aladdin_table mac
     lookup -- see specs.py's technology_node comment), so this really does
     need search, not a formula. Binary search over sram_kib_per_pe (KiB,
     holding fifo/router config at base_wafer's values) for the largest
     value whose real per-instance pe_scratchpad_area_um2 fits the per-PE
     share of the memory budget.

  3. Network (real binary search): fifo_area_um2 is similarly meant to be a
     real function of declared FIFO depth. Binary search over
     fifo_capacity_packets (recombined with step 2's SRAM finding so
     pe_scratchpad keeps its already-approved byte allocation regardless of
     how big FIFO grows -- see _wafer_for_fifo_candidate) for the largest
     value whose real (fifo_area_um2 + router_area_um2 +
     wafer.link_area_um2_per_pe) fits the per-PE share of the network
     budget. router_area_um2 and link_area_um2_per_pe are NOT searched --
     router_ports_per_pe/router_port_bytes/link_area_um2_per_pe stay at
     base_wafer's real-cited values throughout (this project only exposes
     FIFO depth as a configurable "network size" knob; see specs.py).

     CONFIRMED REAL FAILURE MODE (2026-08-23, real Docker run at
     fifo_capacity_packets up to 1,048,576): fifo_buffer is declared to
     Accelergy with depth = wafer.fifo_bytes_per_pe = 8 * fifo_capacity_packets
     * packet_payload_bytes (raw bytes, block-size 1 / word-bits 8 -- see
     timeloop.py). The real accelergy-cacti-plug-in (cacti_wrapper.py's
     CactiMemory._interp_size) turns that into a CACTI "cache_size" of
     roughly 4 * depth bytes. At fifo_capacity_packets=524288 that is
     already ~4.3 GB (2^32) and at 1,048,576 it is ~8.6 GB (2^33) -- and a
     real run's timeloop-model.accelergy.log at that depth NEVER shows a
     "Calling CACTI with cache_size=..." line for fifo_buffer at all (the
     only cache_size values that appear are pe_scratchpad's and
     router_buffer's), while the final ART.yaml still reports a
     fifo_buffer area of 1.0 um^2 -- a value that cannot correspond to a
     real CACTI computation at that depth (cross-checked against a real
     run at fifo_capacity_packets=8, where cache_size DID scale correctly
     to 65536 and produced a sensible, real 59674.0 um^2). In other words:
     past some real, so-far-unmeasured cache_size threshold, this pipeline
     silently substitutes a near-zero placeholder area for fifo_buffer
     instead of erroring -- which breaks _binary_search_max_fit's
     non-decreasing-area assumption (area appears to STOP growing, or even
     drop back near zero, well before the budget is exhausted), so the
     search always "fits" and walks straight to _FIFO_CAPACITY_PACKETS_MAX
     regardless of the real per-PE network budget. A result whose
     fifo_capacity_packets lands exactly on _FIFO_CAPACITY_PACKETS_MAX (see
     AreaBudgetResult.fifo_search_saturated) must NOT be trusted as a real
     budget-driven optimum -- see that flag's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor
from pathlib import Path
from typing import Callable

from .contract import GemmMappingContract
from .specs import GemmSpec, WaferSpec
from .timeloop import (
    MODEL_MODE,
    SAFE_CACTI_CACHE_SIZE_BYTES,
    generate_timeloop_inputs,
    parse_mapping_contract,
    run_timeloop_model,
)

DEFAULT_IMAGE = "timeloopaccelergy/accelergy-timeloop-infrastructure:latest"

# Per-stage cap on real Docker invocations during binary search. 24 is
# unchanged from when _SRAM_KIB_PER_PE_MAX/_FIFO_CAPACITY_PACKETS_MAX were
# both 1,048,576 (needing ~20 halvings in the worst case -- confirmed
# against a real run where 10 iterations was not enough and produced a "no
# candidate fits" error even though a fitting sram_kib_per_pe genuinely
# existed). Both bounds are now far smaller (4096 KiB / 2048 packets --
# see their derivation above), needing only ~12 halvings worst case, so 24
# leaves comfortable headroom rather than being a tight requirement now.
# _search_max_sram_kib_per_pe/_search_max_fifo_capacity_packets still
# detect and clearly report the "iterations exhausted while the search
# range was still open" situation regardless, in case these bounds are
# ever widened again in the future without raising this to match.
MAX_SEARCH_ITERATIONS = 24

# Search bounds, in the candidate's own unit -- NOT real hardware limits,
# but derived from timeloop.py's SAFE_CACTI_CACHE_SIZE_BYTES, the single
# shared, real-Docker-confirmed ceiling on what the real
# accelergy-cacti-plug-in can be trusted to model correctly for any
# SRAM-class component this project declares (see that constant's
# docstring in timeloop.py for the full derivation and the two real data
# points -- CONFIRMED WORKING at cache_size=65536, CONFIRMED BROKEN at
# cache_size=2**32 -- that bracket it). generate_timeloop_inputs() itself
# also enforces this same ceiling directly (see its
# _validate_cacti_safe_sram calls) -- these two search bounds exist so the
# search here never even PROPOSES a candidate that pre-flight check would
# reject, not as the only place the ceiling is enforced.

# fifo_buffer: width_bits = word-bits(8) * block-size(1) = 8 -> scaled_width
# = max(8, 32) = 32. depth = wafer.fifo_bytes_per_pe = 8 * fifo_capacity_
# packets * packet_payload_bytes(256) = 2048 * fifo_capacity_packets.
# cache_size = 32 * depth // 8 = 8192 * fifo_capacity_packets (confirmed
# exactly against both real data points in timeloop.py's docstring:
# 8192*8=65536, 8192*524288=2**32). So the largest fifo_capacity_packets
# keeping cache_size <= SAFE_CACTI_CACHE_SIZE_BYTES is:
_FIFO_CAPACITY_PACKETS_MAX = SAFE_CACTI_CACHE_SIZE_BYTES // 8192  # = 2048

# pe_scratchpad: width_bits = word-bits(8) * block-size(gemm.bytes_per_
# element) -> scaled_width = 32 for bytes_per_element in {1, 2, 4} (every
# real dtype this project models -- int8/fp16/fp32), so cache_size =
# 32 * depth // 8 = 4 * depth regardless of dtype. depth is largest (most
# constraining) for the SMALLEST bytes_per_element (more words for the same
# byte total), so sizing against bytes_per_element=1 (int8) is the
# conservative choice: depth = sram_bytes_per_pe / 1, cache_size =
# 4 * sram_bytes_per_pe <= SAFE_CACTI_CACHE_SIZE_BYTES ->
# sram_bytes_per_pe <= SAFE_CACTI_CACHE_SIZE_BYTES / 4 -> convert to KiB:
_SRAM_KIB_PER_PE_MAX = (SAFE_CACTI_CACHE_SIZE_BYTES // 4) // 1024  # = 4096 KiB (4 MiB/PE)


@dataclass(frozen=True, slots=True)
class AreaBudget:
    """A physical die-area upper limit for the WHOLE mesh, split into
    non-overlapping compute/memory/network percentage shares.

    The three fractions must not sum to more than 1.0 -- they are shares of
    one physical die, not independent limits, so double-booking any area is
    a configuration error, not something to average away.
    """

    wafer_area_budget_um2: float
    compute_area_fraction: float
    memory_area_fraction: float
    network_area_fraction: float

    def validate(self) -> None:
        if self.wafer_area_budget_um2 <= 0:
            raise ValueError("wafer_area_budget_um2 must be positive")
        fractions = (self.compute_area_fraction, self.memory_area_fraction, self.network_area_fraction)
        if min(fractions) < 0:
            raise ValueError("area fractions cannot be negative")
        total = sum(fractions)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"compute_area_fraction + memory_area_fraction + network_area_fraction "
                f"({total:.6f}) exceeds 1.0 -- they are non-overlapping shares of one "
                "physical die and must not sum to more than 100%"
            )

    @property
    def compute_area_budget_um2(self) -> float:
        return self.wafer_area_budget_um2 * self.compute_area_fraction

    @property
    def memory_area_budget_um2(self) -> float:
        return self.wafer_area_budget_um2 * self.memory_area_fraction

    @property
    def network_area_budget_um2(self) -> float:
        return self.wafer_area_budget_um2 * self.network_area_fraction


@dataclass(frozen=True, slots=True)
class AreaBudgetResult:
    """Everything derived from one `derive_wafer_spec_from_area_budget`
    call, including the intermediate real-Docker-confirmed numbers that
    justify the final WaferSpec -- kept around so a caller can show its
    work, not just the final answer.
    """

    wafer: WaferSpec
    budget: AreaBudget
    mac_area_um2: float
    compute_pes: int
    compute_area_used_um2: float
    memory_contract: GemmMappingContract
    network_contract: GemmMappingContract
    sram_search_saturated: bool
    fifo_search_saturated: bool

    @property
    def memory_area_used_um2(self) -> float:
        return self.memory_contract.pe_scratchpad_area_um2

    @property
    def network_area_used_um2(self) -> float:
        return (
            self.network_contract.fifo_area_um2
            + self.network_contract.router_area_um2
            + self.wafer.link_area_um2_per_pe
        )

    @property
    def saturation_warning(self) -> str | None:
        """None if neither search stage's result landed exactly on its own
        artificial search ceiling (_SRAM_KIB_PER_PE_MAX /
        _FIFO_CAPACITY_PACKETS_MAX) -- the normal, trustworthy case, where
        the search converged on a real budget-driven optimum somewhere
        strictly inside its range. Otherwise, a human-readable caveat: a
        result sitting exactly on the ceiling is NOT proof the budget can
        actually support an even larger config -- see this module's
        docstring, stage 3, for a real, confirmed case (fifo_capacity_packets
        landing on 1,048,576) where the true cause was Accelergy/CACTI
        silently returning a near-zero placeholder area for an
        unrealistically large SRAM depth, not a genuine budget optimum.
        This is a data-integrity caveat, not a hard error, because a search
        landing on its ceiling COULD also mean the budget is simply huge
        enough to legitimately want more than this module allows searching
        for -- the flag cannot distinguish the two from the search's own
        results alone, which is exactly why it must be surfaced rather than
        silently trusted either way.
        """
        parts = []
        if self.sram_search_saturated:
            parts.append(
                f"sram_kib_per_pe landed exactly on the search's own ceiling "
                f"(_SRAM_KIB_PER_PE_MAX={_SRAM_KIB_PER_PE_MAX}) -- this is NOT "
                "confirmed to be a real budget-driven optimum"
            )
        if self.fifo_search_saturated:
            parts.append(
                f"fifo_capacity_packets landed exactly on the search's own "
                f"ceiling (_FIFO_CAPACITY_PACKETS_MAX={_FIFO_CAPACITY_PACKETS_MAX}) "
                "-- a real Docker run confirmed Accelergy/CACTI silently returns a "
                "near-zero placeholder fifo_buffer area for SRAM depths this large "
                "instead of erroring, so this result is NOT confirmed to be a real "
                "budget-driven optimum; see this module's docstring, stage 3"
            )
        if not parts:
            return None
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": {
                "wafer_area_budget_um2": self.budget.wafer_area_budget_um2,
                "compute_area_fraction": self.budget.compute_area_fraction,
                "memory_area_fraction": self.budget.memory_area_fraction,
                "network_area_fraction": self.budget.network_area_fraction,
                "compute_area_budget_um2": self.budget.compute_area_budget_um2,
                "memory_area_budget_um2": self.budget.memory_area_budget_um2,
                "network_area_budget_um2": self.budget.network_area_budget_um2,
            },
            "mac_area_um2": self.mac_area_um2,
            "compute_pes": self.compute_pes,
            "mesh_rows": self.wafer.mesh_rows,
            "mesh_cols": self.wafer.mesh_cols,
            "compute_area_used_um2": self.compute_area_used_um2,
            "sram_kib_per_pe": self.wafer.sram_kib_per_pe,
            "memory_area_used_um2": self.memory_area_used_um2,
            "fifo_capacity_packets": self.wafer.fifo_capacity_packets,
            "network_area_used_um2": self.network_area_used_um2,
            "sram_search_saturated": self.sram_search_saturated,
            "fifo_search_saturated": self.fifo_search_saturated,
            "saturation_warning": self.saturation_warning,
            "wafer": {
                "mesh_rows": self.wafer.mesh_rows,
                "mesh_cols": self.wafer.mesh_cols,
                "sram_kib_per_pe": self.wafer.sram_kib_per_pe,
                "fifo_capacity_packets": self.wafer.fifo_capacity_packets,
            },
        }


def _probe_contract(
    gemm: GemmSpec, wafer: WaferSpec, out_dir: Path, tag: str, *, image: str
) -> GemmMappingContract:
    """One real `timeloop-model` Docker run purely to read back real
    mac_area_um2/pe_scratchpad_area_um2/fifo_area_um2/router_area_um2 for
    `wafer` -- always tile_k=1.

    Area never depends on tile_k: pe_scratchpad_depth_words/
    fifo_buffer_depth_words/router_buffer_depth_words (see timeloop.py's
    generate_timeloop_inputs) are functions of wafer/gemm.bytes_per_element
    only, never of the tiling. tile_k=1 is simply the tiling with the
    smallest SRAM footprint, minimizing the chance this probe spuriously
    fails "tile_k exceeds SRAM budget" for a candidate that is otherwise a
    perfectly valid architecture to ask Accelergy about.

    Raises ValueError if tile_k=1 itself does not fit `wafer`'s declared
    SRAM for this GEMM/mesh (this specific candidate is structurally
    infeasible, not just "over the caller's area budget") -- callers here
    treat that the same as "too small, search a larger candidate". Raises
    RuntimeError if the real Docker run itself fails, which is NOT caught
    anywhere in this module and propagates straight to the caller, per this
    project's standing rule that a real tool failure is never silently
    treated as "candidate rejected".
    """
    probe_dir = out_dir / tag
    artifacts = generate_timeloop_inputs(gemm, wafer, probe_dir, tile_k=1)
    run_timeloop_model(artifacts, image=image)
    return parse_mapping_contract(artifacts, gemm, wafer, mode=MODEL_MODE)


def _real_mac_area_um2(gemm: GemmSpec, base_wafer: WaferSpec, out_dir: Path, *, image: str) -> float:
    """One real, cheap probe for the structurally-fixed mac_area_um2 -- see
    this module's docstring, stage 1. Uses a tiny throwaway 8x8x8 GEMM on a
    2x2 mesh, decoupled entirely from the real `gemm`/derived mesh this
    module ultimately produces, except for `gemm.bytes_per_element` (kept
    consistent with the real problem, since intmac's declared datawidth
    does come from it -- see timeloop.py's architecture.yaml generation).

    2x2, not 1x1: confirmed against a real Docker run that a 1x1 mesh makes
    TimeLoop itself fail with "ERROR: cannot find spatial tiling level
    associated with storage level DRAM. This is because the number of
    instances of the next-inner level (pe_scratchpad) is the same as this
    level" -- with exactly 1 PE, TimeLoop cannot distinguish "a spatial
    fanout of 1" from "no spatial level at all" and refuses to run. This is
    a real constraint of this project's architecture_constraints.yaml/
    mapping.yaml scheme (a DRAM-to-PE spatial fanout entry), not a probe
    sizing choice -- see _derive_mesh below, which enforces the same
    minimum on the mesh this module ultimately derives.
    """
    probe_gemm = GemmSpec(m=8, n=8, k=8, bytes_per_element=gemm.bytes_per_element)
    probe_wafer = replace(base_wafer, mesh_rows=2, mesh_cols=2)
    contract = _probe_contract(probe_gemm, probe_wafer, out_dir, "area_budget_probe_compute", image=image)
    return contract.mac_area_um2


def _divisors(n: int) -> list[int]:
    """All positive divisors of n, ascending. n is a GEMM dimension in this
    module's usage (at most a few million in any realistic problem), so the
    O(sqrt(n)) trial-division enumeration below is more than fast enough --
    no need for a factorization library."""
    divs: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


def _derive_mesh(
    compute_area_budget_um2: float, mac_area_um2: float, gemm_m: int, gemm_n: int
) -> tuple[int, int]:
    """The largest mesh_rows|gemm_m x mesh_cols|gemm_n (both true divisors)
    with mesh_rows*mesh_cols <= floor(compute_area_budget_um2/mac_area_um2)
    PEs, both >=2, breaking ties toward the closest-to-square shape -- see
    this module's docstring, stage 1, for why the divisor restriction is a
    real TimeLoop requirement (not a style choice) and why >=2 (a real
    Docker run confirmed TimeLoop itself rejects a 1-PE mesh -- see
    _real_mac_area_um2's docstring for the exact real error).
    """
    max_pes = floor(compute_area_budget_um2 / mac_area_um2)
    if max_pes < 1:
        raise ValueError(
            f"compute area budget ({compute_area_budget_um2:.2f} um^2) is smaller than "
            f"one real PE's mac area ({mac_area_um2:.2f} um^2) -- cannot fit even 1 PE; "
            "raise wafer_area_budget_um2 or compute_area_fraction"
        )
    row_divisors = [d for d in _divisors(gemm_m) if d >= 2]
    col_divisors = [d for d in _divisors(gemm_n) if d >= 2]
    best: tuple[int, int, int, int] | None = None  # (pes, -|rows-cols|, rows, cols), for ranking
    for rows in row_divisors:
        if rows > max_pes:
            break
        for cols in col_divisors:
            pes = rows * cols
            if pes > max_pes:
                continue
            rank = (pes, -abs(rows - cols))
            if best is None or rank > best[:2]:
                best = (*rank, rows, cols)
    if best is None:
        raise ValueError(
            f"no mesh_rows (a divisor of gemm.m={gemm_m}, >=2) x mesh_cols (a divisor "
            f"of gemm.n={gemm_n}, >=2) pair fits within {max_pes} PEs (compute area "
            f"budget {compute_area_budget_um2:.2f} um^2 / real mac_area_um2="
            f"{mac_area_um2:.2f}) -- raise wafer_area_budget_um2 or compute_area_fraction, "
            "or use a GEMM shape with smaller/more divisors"
        )
    _, _, rows, cols = best
    return rows, cols


def _binary_search_max_fit(
    lo: int,
    hi: int,
    max_iterations: int,
    probe: Callable[[int], tuple[float, object] | None],
    budget_um2: float,
) -> tuple[int | None, object | None, bool]:
    """Generic integer binary search for the largest candidate in [lo, hi]
    whose real area fits budget_um2, assuming area is non-decreasing in the
    candidate value (true for real CACTI SRAM area vs. declared depth --
    more capacity never yields less real area).

    `probe(candidate)` returns `None` for a candidate this project's own
    rules reject as structurally infeasible (too small to even form a valid
    mapping -- search moves up, exactly like "over budget" moves down would
    be wrong to conflate with), or `(area_um2, extra)` for a structurally
    valid candidate, where `extra` is whatever the caller wants carried
    through (here: the real GemmMappingContract that produced `area_um2`).
    Every call to `probe` is expected to hit real Docker -- this function
    itself does no area estimation, only control flow.

    Returns `(best_candidate_or_None, best_extra_or_None, exhausted)`.
    `exhausted` is True when `max_iterations` ran out while `lo <= hi` was
    still true -- i.e. the search space was NOT fully narrowed down, so a
    `None` result here can mean "genuinely no fit exists" OR "not enough
    iterations to find it yet" (with a [1, ~1e6] default range, that needs
    roughly 20 halvings in the worst case -- see MAX_SEARCH_ITERATIONS).
    Callers use this to give an accurate error message instead of always
    blaming the budget.
    """
    best: int | None = None
    best_extra: object | None = None
    for _ in range(max_iterations):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        result = probe(mid)
        if result is None:
            lo = mid + 1
            continue
        area_um2, extra = result
        if area_um2 <= budget_um2:
            best, best_extra = mid, extra
            lo = mid + 1
        else:
            hi = mid - 1
    exhausted = lo <= hi
    return best, best_extra, exhausted


def _search_max_sram_kib_per_pe(
    gemm: GemmSpec,
    base_wafer: WaferSpec,
    mesh_rows: int,
    mesh_cols: int,
    per_pe_memory_budget_um2: float,
    out_dir: Path,
    *,
    max_iterations: int,
    image: str,
) -> tuple[int, GemmMappingContract]:
    """Largest sram_kib_per_pe (integer KiB) whose real pe_scratchpad_area_um2
    fits per_pe_memory_budget_um2 -- see this module's docstring, stage 2.
    fifo/router config is held at base_wafer's values throughout."""
    mesh_wafer = replace(base_wafer, mesh_rows=mesh_rows, mesh_cols=mesh_cols)

    def probe(candidate_kib: int) -> tuple[float, GemmMappingContract] | None:
        candidate = replace(mesh_wafer, sram_kib_per_pe=candidate_kib)
        try:
            candidate.validate()
            contract = _probe_contract(
                gemm, candidate, out_dir, f"area_budget_mem_{candidate_kib}", image=image
            )
        except ValueError:
            return None
        return contract.pe_scratchpad_area_um2, contract

    best, best_contract, exhausted = _binary_search_max_fit(
        1, _SRAM_KIB_PER_PE_MAX, max_iterations, probe, per_pe_memory_budget_um2
    )
    if best is None:
        if exhausted:
            raise ValueError(
                f"ran out of search iterations (max_iterations={max_iterations}) before "
                f"narrowing sram_kib_per_pe down to a value that fits the per-PE memory "
                f"budget ({per_pe_memory_budget_um2:.2f} um^2) for this GEMM at a "
                f"{mesh_rows}x{mesh_cols} mesh -- a [1, {_SRAM_KIB_PER_PE_MAX}] KiB range "
                "needs roughly 20 halvings in the worst case; raise --area-search-iterations "
                "(or max_iterations) toward MAX_SEARCH_ITERATIONS instead of lowering it, "
                "this is NOT necessarily a real budget infeasibility"
            )
        raise ValueError(
            f"no sram_kib_per_pe in [1, {_SRAM_KIB_PER_PE_MAX}] KiB fits the per-PE "
            f"memory budget ({per_pe_memory_budget_um2:.2f} um^2) for this GEMM at a "
            f"{mesh_rows}x{mesh_cols} mesh -- raise wafer_area_budget_um2 or "
            "memory_area_fraction, or shrink the compute-derived mesh"
        )
    return best, best_contract  # type: ignore[return-value]


def _search_max_fifo_capacity_packets(
    gemm: GemmSpec,
    base_wafer: WaferSpec,
    mesh_rows: int,
    mesh_cols: int,
    sram_kib_per_pe_from_memory_search: int,
    per_pe_network_budget_um2: float,
    out_dir: Path,
    *,
    max_iterations: int,
    image: str,
) -> tuple[int, WaferSpec, GemmMappingContract]:
    """Largest fifo_capacity_packets whose real
    (fifo_area_um2 + router_area_um2 + link_area_um2_per_pe) fits
    per_pe_network_budget_um2 -- see this module's docstring, stage 3.

    Each candidate's sram_kib_per_pe is recomputed so pe_scratchpad keeps
    EXACTLY the byte allocation the memory search (stage 2) already
    approved, regardless of how big this stage grows the FIFO reservation
    -- growing FIFO must not silently shrink or grow the memory category's
    already-budgeted pe_scratchpad area (see _wafer_for_fifo_candidate).
    """
    mesh_wafer = replace(base_wafer, mesh_rows=mesh_rows, mesh_cols=mesh_cols)
    stage2_wafer = replace(mesh_wafer, sram_kib_per_pe=sram_kib_per_pe_from_memory_search)
    # The exact byte count stage 2 approved for pe_scratchpad alone, using
    # base_wafer's ORIGINAL fifo/router reservation (the one in effect
    # during the memory search) -- this is what must be preserved as FIFO
    # capacity is varied below.
    pe_scratchpad_bytes_target = (
        stage2_wafer.sram_bytes_per_pe - stage2_wafer.fifo_router_bytes_per_pe
    )

    def wafer_for(candidate_packets: int) -> WaferSpec:
        partial = replace(mesh_wafer, fifo_capacity_packets=candidate_packets)
        needed_sram_bytes = pe_scratchpad_bytes_target + partial.fifo_router_bytes_per_pe
        needed_sram_kib = ceil(needed_sram_bytes / 1024)
        return replace(partial, sram_kib_per_pe=needed_sram_kib)

    def probe(candidate_packets: int) -> tuple[float, tuple[WaferSpec, GemmMappingContract]] | None:
        candidate = wafer_for(candidate_packets)
        try:
            candidate.validate()
            contract = _probe_contract(
                gemm, candidate, out_dir, f"area_budget_net_{candidate_packets}", image=image
            )
        except ValueError:
            return None
        network_area_um2 = (
            contract.fifo_area_um2 + contract.router_area_um2 + candidate.link_area_um2_per_pe
        )
        return network_area_um2, (candidate, contract)

    best, best_extra, exhausted = _binary_search_max_fit(
        1, _FIFO_CAPACITY_PACKETS_MAX, max_iterations, probe, per_pe_network_budget_um2
    )
    if best is None:
        if exhausted:
            raise ValueError(
                f"ran out of search iterations (max_iterations={max_iterations}) before "
                f"narrowing fifo_capacity_packets down to a value that fits the per-PE "
                f"network budget ({per_pe_network_budget_um2:.2f} um^2) for this GEMM at a "
                f"{mesh_rows}x{mesh_cols} mesh -- a [1, {_FIFO_CAPACITY_PACKETS_MAX}] range "
                "needs roughly 20 halvings in the worst case; raise --area-search-iterations "
                "(or max_iterations) toward MAX_SEARCH_ITERATIONS instead of lowering it, "
                "this is NOT necessarily a real budget infeasibility"
            )
        raise ValueError(
            f"no fifo_capacity_packets in [1, {_FIFO_CAPACITY_PACKETS_MAX}] fits the "
            f"per-PE network budget ({per_pe_network_budget_um2:.2f} um^2) for this "
            f"GEMM at a {mesh_rows}x{mesh_cols} mesh -- raise wafer_area_budget_um2 or "
            "network_area_fraction, or shrink the compute-derived mesh"
        )
    best_wafer, best_contract = best_extra  # type: ignore[misc]
    return best, best_wafer, best_contract


def derive_wafer_spec_from_area_budget(
    budget: AreaBudget,
    gemm: GemmSpec,
    base_wafer: WaferSpec,
    out_dir: Path,
    *,
    max_iterations: int = MAX_SEARCH_ITERATIONS,
    image: str = DEFAULT_IMAGE,
) -> AreaBudgetResult:
    """Derive mesh_rows/mesh_cols/sram_kib_per_pe/fifo_capacity_packets from
    a physical die-area budget instead of taking them directly, per this
    module's three-stage real-Docker search (see module docstring).

    `base_wafer` supplies every OTHER field unchanged (technology_node,
    clock_ghz, router_ports_per_pe/router_port_bytes, link_area_um2_per_pe,
    link_bandwidth_bytes_per_cycle, etc.) -- only mesh_rows/mesh_cols/
    sram_kib_per_pe/fifo_capacity_packets are overwritten by this search.

    Raises ValueError (with a specific, actionable message -- see each
    stage's search function) if any stage's budget cannot fit even the
    smallest structurally-valid candidate. Raises RuntimeError if any real
    Docker/TimeLoop/Accelergy invocation itself fails -- there is no
    fallback to a locally-estimated config at any stage.
    """
    budget.validate()
    gemm.validate()

    mac_area_um2 = _real_mac_area_um2(gemm, base_wafer, out_dir, image=image)
    mesh_rows, mesh_cols = _derive_mesh(budget.compute_area_budget_um2, mac_area_um2, gemm.m, gemm.n)
    num_pes = mesh_rows * mesh_cols
    compute_area_used_um2 = num_pes * mac_area_um2

    per_pe_memory_budget_um2 = budget.memory_area_budget_um2 / num_pes
    sram_kib_per_pe, memory_contract = _search_max_sram_kib_per_pe(
        gemm,
        base_wafer,
        mesh_rows,
        mesh_cols,
        per_pe_memory_budget_um2,
        out_dir,
        max_iterations=max_iterations,
        image=image,
    )

    per_pe_network_budget_um2 = budget.network_area_budget_um2 / num_pes
    fifo_capacity_packets, final_wafer, network_contract = _search_max_fifo_capacity_packets(
        gemm,
        base_wafer,
        mesh_rows,
        mesh_cols,
        sram_kib_per_pe,
        per_pe_network_budget_um2,
        out_dir,
        max_iterations=max_iterations,
        image=image,
    )

    return AreaBudgetResult(
        wafer=final_wafer,
        budget=budget,
        mac_area_um2=mac_area_um2,
        compute_pes=num_pes,
        compute_area_used_um2=compute_area_used_um2,
        memory_contract=memory_contract,
        network_contract=network_contract,
        # Compare the stages' OWN returned search values, not fields read
        # back off final_wafer -- stage 3 recomputes sram_kib_per_pe to
        # accommodate whatever FIFO reservation it lands on (see
        # _search_max_fifo_capacity_packets's wafer_for()), so final_wafer's
        # sram_kib_per_pe generally will NOT equal _SRAM_KIB_PER_PE_MAX even
        # when stage 2 itself did saturate.
        sram_search_saturated=(sram_kib_per_pe == _SRAM_KIB_PER_PE_MAX),
        fifo_search_saturated=(fifo_capacity_packets == _FIFO_CAPACITY_PACKETS_MAX),
    )
