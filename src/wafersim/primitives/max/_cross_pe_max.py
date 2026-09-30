"""Real, Docker-backed + real-network-simulated cross-PE extension of the
`max` primitive: reduce ``total_elements`` real elements that are SHARDED
ACROSS ``num_pes`` distinct physical PEs (not all living on one PE), the
way a real tensor-parallel softmax numerical-stability row-max actually
has to when the reduced dimension has been split across the mesh.
Local-only ``MaxSpec``/``simulate_max`` (see ``_timeloop_max.py``) stays
correct and untouched for the case where all K elements already live on
one PE -- this module is additive, structurally a straight copy of this
package's own sibling ``sum._cross_pe_sum`` module with the reduction op
swapped (comparator instead of adder) -- see that module's own docstring
for the fuller citation trail; this docstring only covers what differs
for `max`.

WHY THIS WAS MISSING, AND THE REAL GAP THAT MAKES IT DIFFERENT FROM `sum`
(real research, already done for this project -- not re-asserted from
memory): Cerebras's own official CSL SDK docs (``collectives_2d`` library,
https://sdk.cerebras.net/csl/language/libraries) expose exactly ONE
row/column reduction primitive -- ``reduce_fadds(root, send_buf, recv_buf,
count, callback)``, an MPI_Sum equivalent. There is NO documented
``reduce_fmax``/``reduce_max``/min-reduction primitive anywhere in that
library -- a real, confirmed gap in Cerebras's own published collectives
API, not an oversight in this project's research. WaferLLM (arXiv
2502.04563) also only names allreduce-SUM-style GEMV solutions for
RMSNorm/Softmax's normalization; it does not separately describe a
cross-PE max mechanism either. So, unlike `sum`, there is no real, named
"this is how cross-PE max works on Cerebras hardware" mechanism to point
to -- only the REASONABLE INFERENCE that if a cross-PE sum-reduction
exists via a row/column relay (as `reduce_fadds` demonstrably does), a
cross-PE max-reduction over the same physical fabric would use the
identical relay topology, just swapping the per-hop combine operation
from a real add to a real comparison. This module makes that inference
explicit and named, not a silent assumption: it is LESS certain than
`sum`'s cross-PE module (which has a real, named official primitive
behind it), and this file says so plainly rather than borrowing `sum`'s
confidence level.

MECHANISM (identical relay topology to `sum._cross_pe_sum`, see that
module's docstring for the full evidence trail on why a SEQUENTIAL relay
chain -- not a parallel tree -- was chosen for consistency with this
project's other reductions): PE(0, P-1) sends its local partial max west
to PE(0, P-2), which real-compares it against its own local partial max
and relays the running max further west, ... down to PE(0, 0), which
ends up holding the grand max after P-1 real hops and P-1 real
comparator firings.

WHAT IS REAL DOCKER, WHAT IS REAL SIMULATION-BUT-NOT-DOCKER, AND WHAT IS
OPENCAKE'S OWN ANALYTICAL COEFFICIENT (identical three-way split to
`sum._cross_pe_sum` -- see that module's docstring for the full
breakdown; summarized here):
  1. REAL DOCKER: one real ``simulate_max()`` sub-run for the local
     per-PE reduction (representative of every one of the ``num_pes``
     PEs); one real single-firing ``_simulate_compare()`` sub-run
     (representative of each of the ``num_pes - 1`` real combine-step
     comparisons along the relay chain). ``_simulate_compare``/
     ``_CompareSpec`` are this package's own PRIVATE single-comparison
     pipeline (see ``_timeloop_max.py``'s module docstring's "WHY THIS
     FILE HAS TWO LAYERS" section) -- reused here directly (same-package
     import) exactly the way the public ``simulate_add`` is reused by
     `sum._cross_pe_sum`, since `max` has no public single-comparison API
     to import from another package the way `sum` reuses `add`.
  2. REAL SIMULATION, NOT DOCKER: the real hop-by-hop LINK/FIFO timing
     for the P-1 relay hops, via ``wafer.network.PacketNetwork._send_hop``
     -- identical reuse to `sum._cross_pe_sum` and GEMM's own broadcast.
  3. OPENCAKE'S OWN ANALYTICAL COEFFICIENT: the ENERGY of those P-1 real
     hops (fifo/router/wire terms) and static leakage -- identical reuse
     of ``wafer.energy.estimate_energy``/``wafer.area.estimate_area``.

AREA ACCOUNTING FIX (applies here AND was retroactively applied to
`sum._cross_pe_sum` -- named explicitly so it isn't silently
inconsistent between the two files): naively reporting
``pe_internal_area_um2`` from ONLY the local-reduction sub-run's real
component area (as `sum._cross_pe_sum`'s first version did) silently
reports a fabricated ZERO area whenever the LOCAL per-PE reduction is
itself degenerate (``elements_per_pe == 1``, i.e. ``total_elements ==
num_pes``) -- a real, reachable configuration (`validate()` allows it),
where a real comparator genuinely IS instantiated for the combine phase
even though the local phase needed none. Fixed here (and in `sum`) by
falling back to the COMBINE sub-run's own real component area whenever
the local phase produced none but a combine phase still ran -- see
``_pe_internal_area_um2()`` below.

SHARDING ASSUMPTION: ``total_elements`` must divide evenly by ``num_pes``
(this module raises otherwise rather than silently rounding) -- same
convention as `sum._cross_pe_sum`.

DEGENERATE CASE (num_pes=1): no cross-PE step exists -- returns the
existing local-only ``MaxSpec``/``simulate_max`` result directly with a
true zero-cost combine phase (no network, no combine-compare Docker
run), not a fabricated near-zero number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from ...wafer.area import AreaBreakdown, estimate_area
from ...wafer.energy import EnergyBreakdown, estimate_energy
from ...wafer.network import PacketNetwork
from ._timeloop_max import (
    MaxResult,
    MaxSpec,
    _CompareResult,
    _CompareSpec,
    _simulate_compare,
    simulate_max,
)


@dataclass(frozen=True, slots=True)
class MaxCrossPESpec:
    """Reduce ``total_elements`` real elements to their max, evenly
    sharded across ``num_pes`` distinct physical PEs arranged in one row
    of the mesh, down to one scalar on the leftmost PE -- see module
    docstring for the real mechanism (local reduce, then a real P-1-hop
    sequential relay chain of real comparisons)."""

    total_elements: int
    num_pes: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if self.num_pes <= 0:
            raise ValueError("num_pes must be positive")
        if min(self.total_elements, self.bytes_per_element) <= 0:
            raise ValueError("total_elements and bytes_per_element must be positive")
        if self.total_elements % self.num_pes != 0:
            raise ValueError(
                f"total_elements ({self.total_elements}) must divide evenly by "
                f"num_pes ({self.num_pes}) -- this module models an even shard "
                "only, see module docstring's SHARDING ASSUMPTION"
            )

    @property
    def elements_per_pe(self) -> int:
        """Real element count each of the num_pes PEs reduces locally
        before the cross-PE combine phase."""
        return self.total_elements // self.num_pes

    @property
    def real_combine_hop_count(self) -> int:
        """Exact number of real relay hops (and real combine-compare
        firings) the sequential chain requires: num_pes - 1 (0 when
        num_pes=1). Not an estimate -- this is the definition of
        reducing num_pes partial maxes to one."""
        return max(self.num_pes - 1, 0)


@dataclass(frozen=True, slots=True)
class MaxCrossPEResult:
    """Real, Docker-verified + real-network-simulated cost of one
    cross-PE ``max`` reduction (see module docstring for exactly which
    parts are Docker, which are real-but-not-Docker simulation, and which
    are OpenCake's own analytical coefficients)."""

    spec: MaxCrossPESpec
    local_result: MaxResult
    """Real per-PE local-reduction sub-run, representative of every one
    of ``spec.num_pes`` PEs (see module docstring)."""
    combine_compare_result: Optional[_CompareResult]
    """Real single-comparator-firing sub-run, representative of every one
    of ``spec.real_combine_hop_count`` combine steps -- ``None`` only
    when ``num_pes=1`` (nothing to combine, see module docstring's
    DEGENERATE CASE)."""
    sim_time_cycles: float
    """Real total wall-clock cycles: local phase (all PEs in parallel,
    one real per-PE local_compute_cycles) + the real, sequential P-1-hop
    combine chain (real network hop timing + real per-hop compare, back
    to back since each hop depends on the previous one's result)."""
    network_summary: dict[str, float | int]
    """Real ``PacketNetwork.summary()`` output for the P-1-hop relay
    chain -- empty-network zeros when num_pes=1."""
    energy: EnergyBreakdown
    """Real Docker PE-internal energy (local reductions + combine
    compares, summed across every real physical PE/firing) plus
    OpenCake's own analytical network/static energy terms."""
    area: AreaBreakdown
    """Real Docker per-PE area (comparator + scratchpad, from whichever
    real sub-run actually instantiated one -- see module docstring's
    "AREA ACCOUNTING FIX") plus OpenCake's own analytical inter-PE link
    area."""

    @property
    def cycles_per_element(self) -> float:
        return self.sim_time_cycles / self.spec.total_elements

    @property
    def energy_uj_per_element(self) -> float:
        return self.energy.total_uj / self.spec.total_elements

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": {
                "total_elements": self.spec.total_elements,
                "num_pes": self.spec.num_pes,
                "elements_per_pe": self.spec.elements_per_pe,
                "bytes_per_element": self.spec.bytes_per_element,
                "real_combine_hop_count": self.spec.real_combine_hop_count,
            },
            "sim_time_cycles": self.sim_time_cycles,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "local_result": self.local_result.to_dict(),
            "combine_compare_result": (
                None if self.combine_compare_result is None else asdict(self.combine_compare_result)
            ),
            "network_summary": self.network_summary,
            "energy": self.energy.to_dict(),
            "area": self.area.to_dict(),
        }


def _pe_internal_area_um2(
    local_component_area: Optional[float],
    local_scratchpad_area: Optional[float],
    combine_component_area: Optional[float],
    combine_scratchpad_area: Optional[float],
) -> float:
    """Real per-PE area, preferring the LOCAL sub-run's real component
    (the combine phase reuses that exact same physical comparator, so no
    double count) but falling back to the COMBINE sub-run's real
    component when the local phase was itself degenerate and
    instantiated nothing (see module docstring's "AREA ACCOUNTING FIX").
    Returns a true 0.0 only when neither sub-run ever ran (num_pes=1 AND
    elements_per_pe=1, i.e. total_elements=1 -- nothing real computed at
    all)."""
    if local_component_area is not None:
        return local_component_area + (local_scratchpad_area or 0.0)
    if combine_component_area is not None:
        return combine_component_area + (combine_scratchpad_area or 0.0)
    return 0.0


def simulate_max_cross_pe(
    spec: MaxCrossPESpec, wafer: WaferSpec, out_dir: Path
) -> MaxCrossPEResult:
    """Run the real local-reduction Docker sub-run, then (if num_pes > 1)
    the real single-comparator-firing Docker sub-run plus the real,
    sequential P-1-hop network relay simulation -- composing all of them
    into one real cross-PE `max` cost. Raises (never fabricates) if any
    real Docker sub-run fails, same discipline as every other primitive
    in this project.
    """
    spec.validate()

    local_spec = MaxSpec(
        elements_per_pe=spec.elements_per_pe, bytes_per_element=spec.bytes_per_element
    )
    local_result = simulate_max(local_spec, wafer, out_dir / "local")

    hop_count = spec.real_combine_hop_count
    empty_network_summary: dict[str, float | int] = {
        "links": 0,
        "packet_hops": 0,
        "link_bytes": 0,
        "router_byte_hops": 0,
        "wire_byte_hops": 0,
        "fifo_enqueued_bytes": 0,
        "send_fifo_wait_cycles": 0,
        "receive_fifo_wait_cycles": 0,
        "link_queue_wait_cycles": 0,
        "max_send_fifo_depth": 0,
        "max_receive_fifo_depth": 0,
        "max_link_utilization": 0.0,
    }

    if hop_count == 0:
        sim_time_cycles = float(local_result.local_compute_cycles)
        pe_internal_energy_uj = local_result.local_energy_uj
        energy = estimate_energy(
            wafer=wafer,
            active_pes=spec.num_pes,
            sim_time_cycles=sim_time_cycles,
            pe_internal_energy_uj=pe_internal_energy_uj,
            network=empty_network_summary,
        )
        pe_internal_area_um2 = _pe_internal_area_um2(
            local_result.comparator_area_um2, local_result.pe_scratchpad_area_um2, None, None
        )
        area = estimate_area(
            wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
        )
        return MaxCrossPEResult(
            spec=spec,
            local_result=local_result,
            combine_compare_result=None,
            sim_time_cycles=sim_time_cycles,
            network_summary=empty_network_summary,
            energy=energy,
            area=area,
        )

    # One real Docker sub-run for a single combine-compare firing, reused
    # for every one of the hop_count real combine steps (see module
    # docstring). Private same-package pipeline -- see module docstring's
    # point 1.
    combine_compare_spec = _CompareSpec(elements_per_pe=1, bytes_per_element=spec.bytes_per_element)
    combine_compare_result = _simulate_compare(combine_compare_spec, wafer, out_dir / "combine_compare")

    # Real, sequential P-1-hop relay chain along one row of the mesh:
    # PE(0, num_pes-1) -> PE(0, num_pes-2) -> ... -> PE(0, 0). Real
    # network timing via PacketNetwork._send_hop; real per-hop compare
    # cost from combine_compare_result.
    network = PacketNetwork(wafer, rows=1, cols=spec.num_pes)
    cycle = float(local_result.local_compute_cycles)
    for col in range(spec.num_pes - 1, 0, -1):
        hop_end = network._send_hop(
            ready_cycle=cycle,
            size_bytes=spec.bytes_per_element,
            source=(0, col),
            destination=(0, col - 1),
            direction="west",
        )
        cycle = hop_end + combine_compare_result.local_compute_cycles

    sim_time_cycles = max(cycle, network.latest_cycle)
    network_summary = network.summary(sim_time_cycles)

    pe_internal_energy_uj = (
        local_result.local_energy_uj * spec.num_pes
        + combine_compare_result.local_energy_uj * hop_count
    )
    energy = estimate_energy(
        wafer=wafer,
        active_pes=spec.num_pes,
        sim_time_cycles=sim_time_cycles,
        pe_internal_energy_uj=pe_internal_energy_uj,
        network=network_summary,
    )

    pe_internal_area_um2 = _pe_internal_area_um2(
        local_result.comparator_area_um2,
        local_result.pe_scratchpad_area_um2,
        combine_compare_result.comparator_area_um2,
        combine_compare_result.pe_scratchpad_area_um2,
    )
    area = estimate_area(
        wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
    )

    return MaxCrossPEResult(
        spec=spec,
        local_result=local_result,
        combine_compare_result=combine_compare_result,
        sim_time_cycles=sim_time_cycles,
        network_summary=network_summary,
        energy=energy,
        area=area,
    )
