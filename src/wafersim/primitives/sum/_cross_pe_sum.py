"""Real, Docker-backed + real-network-simulated cross-PE extension of the
`sum` primitive: reduce ``total_elements`` real elements that are SHARDED
ACROSS ``num_pes`` distinct physical PEs (not all living on one PE), the
way a real RMSNorm/softmax reduction along a dimension that a real
tensor-parallel/model-parallel layout has split across the mesh actually
has to. Local-only ``SumSpec``/``simulate_sum`` (see ``_timeloop_sum.py``)
stays correct and untouched for the case where all K elements already live
on one PE -- this module is additive, for the other real case.

WHY THIS WAS MISSING (real research, not assumed): Cerebras's own official
CSL SDK docs (``collectives_2d`` library, https://sdk.cerebras.net/csl/
language/libraries) expose exactly one row/column reduction primitive --
``reduce_fadds(root, send_buf, recv_buf, count, callback)``, described as
"an MPI_Sum for buffers of f32" -- confirming a real cross-PE sum-reduction
mechanism genuinely exists in Cerebras's own toolchain, along the same 2-D
mesh row/column dimension that this project's ``wafer/network.py`` already
models for GEMM. WaferLLM (arXiv 2502.04563, already cited by `add`'s own
docstring) independently states "Operations needing allreduce, such as
RMSNorm and Softmax, can leverage GEMV solutions" -- i.e. the same K-tree/
row-reduction machinery GEMM's decode-phase GEMV already needs. So `sum`
crossing PE boundaries is a real, named mechanism on real hardware, not a
hypothetical -- this module gives it a real, Docker + real-network-timed
cost model, composed entirely from already-Docker-verified components.

WHAT NEITHER SOURCE SAYS (also stated plainly, not glossed over): neither
Cerebras's own docs nor WaferLLM's paper publish the exact internal
topology `reduce_fadds` uses on real silicon (ring vs. tree vs. something
else) or its real hop count/latency. This module does NOT claim to
reproduce Cerebras's proprietary internal implementation. It makes one
explicit, named MODELING CHOICE instead, for consistency with a choice
this project already made and documented: a SEQUENTIAL relay chain along
one row of the mesh, i.e. PE(0, P-1) sends its local partial sum west to
PE(0, P-2), which real-adds it into its own partial sum and relays the
running total further west, ... down to PE(0, 0), which ends up holding
the grand total after P-1 real hops and P-1 real adds. This mirrors the
exact same "sequential, not tree, reduction" choice `_timeloop_sum.py`'s
own module docstring already made and justified for LOCAL sum (see that
file's "MODELING CHOICE" section) -- kept consistent here rather than
silently switching to a different (parallel-tree) assumption for the
cross-PE case. Like that choice, this one is conservative (a real tree/
broadcast reduction would need only ceil(log2(P)) sequential depth instead
of P-1), not fabricated, and named explicitly so it can be revisited if
this project ever adds a tree-reduction mapping for either the local or
cross-PE case.

WHAT IS REAL DOCKER, WHAT IS REAL SIMULATION-BUT-NOT-DOCKER, AND WHAT IS
OPENCAKE'S OWN ANALYTICAL COEFFICIENT -- named explicitly, not blended
together into one undifferentiated "cost" number:
  1. REAL DOCKER (TimeLoop/Accelergy, via already-verified components):
     - The local per-PE reduction of ``total_elements / num_pes`` elements
       -- ONE real ``simulate_sum()`` sub-run (reused as-is, unmodified),
       representative of every one of the ``num_pes`` PEs since they all
       do IDENTICAL local work (same real fpadder component, same real
       element count) -- matching this project's established convention
       of running one real, representative Docker sub-run for identical
       parallel PE work rather than re-running Docker num_pes times for
       an outcome already known to be identical (see e.g. GEMM's own
       ``contract.local_compute_cycles`` being a per-PE number, not
       per-PE-instance-times-N).
     - ONE real single-firing ``simulate_add()`` sub-run
       (``AddSpec(elements_per_pe=1, ...)``), representative of each of
       the ``num_pes - 1`` real combine-step adds along the relay chain
       (same real fpadder component, one firing) -- again one real,
       representative Docker sub-run, not ``num_pes - 1`` identical ones.
  2. REAL SIMULATION, NOT DOCKER (cycle-accurate but analytical): the
     real hop-by-hop LINK/FIFO timing for the P-1 relay hops, via
     ``wafer.network.PacketNetwork._send_hop`` -- the EXACT SAME real
     link-bandwidth/router-latency/FIFO-backpressure code GEMM's own
     cross-PE broadcast already uses (see ``wafer/network.py``), applied
     here to a 1xnum_pes row instead of GEMM's full mesh. This is a real,
     parameter-driven discrete-event simulation of the fabric, not a
     Docker run and not a made-up constant.
  3. OPENCAKE'S OWN ANALYTICAL COEFFICIENT (explicitly NOT Docker/
     Accelergy-derived -- same as GEMM's own network energy, see
     ``wafer/energy.py``'s own module docstring): the ENERGY of those P-1
     real hops (fifo/router/wire pJ-per-byte-hop terms from
     ``WaferSpec``) and the static leakage term. TimeLoop/Accelergy has no
     notion of inter-PE distance or routing, so it structurally cannot
     produce this number -- this module reuses the identical
     ``wafer.energy.estimate_energy``/``wafer.area.estimate_area``
     functions GEMM already uses, for the identical reason and with the
     identical honesty about what they are.

SHARDING ASSUMPTION: ``total_elements`` must divide evenly by ``num_pes``
(this module raises otherwise rather than silently rounding) -- an even
shard across the row is the case this module models; a real ragged/uneven
shard is not handled here.

DEGENERATE CASE (num_pes=1): no cross-PE step exists -- this is exactly
the existing local-only ``SumSpec``/``simulate_sum`` case, so this module
returns that result directly with a true zero-cost combine phase (no
network, no combine-add Docker run), not a fabricated near-zero number.

AREA ACCOUNTING FIX: ``pe_internal_area_um2`` prefers the LOCAL
reduction's real adder area (the combine phase reuses that same physical
adder) but falls back to the COMBINE sub-run's own real adder area when
the local phase is itself degenerate (``elements_per_pe == 1``, i.e.
``total_elements == num_pes`` -- a real, reachable configuration
``validate()`` allows) yet a combine phase still ran -- see
``_pe_internal_area_um2()`` below; naively using only the local sub-run's
area in that configuration would silently report a fabricated 0.0 despite
a real comparator/adder genuinely being instantiated for the combine
step. See `max`'s sibling ``_cross_pe_max.py`` for the identical fix
applied there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from ...wafer.area import AreaBreakdown, estimate_area
from ...wafer.energy import EnergyBreakdown, estimate_energy
from ...wafer.network import PacketNetwork
from ..add import AddResult, AddSpec, simulate_add
from ._timeloop_sum import SumResult, SumSpec, simulate_sum


@dataclass(frozen=True, slots=True)
class SumCrossPESpec:
    """Reduce ``total_elements`` real elements, evenly sharded across
    ``num_pes`` distinct physical PEs arranged in one row of the mesh,
    down to one scalar on the leftmost PE -- see module docstring for the
    real mechanism (local reduce, then a real P-1-hop sequential relay
    chain of real adds)."""

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
        """Exact number of real relay hops (and real combine-add firings)
        the sequential chain requires: num_pes - 1 (0 when num_pes=1 --
        nothing to combine across a single PE). Not an estimate -- this
        is the definition of reducing num_pes partial sums to one."""
        return max(self.num_pes - 1, 0)


@dataclass(frozen=True, slots=True)
class SumCrossPEResult:
    """Real, Docker-verified + real-network-simulated cost of one
    cross-PE ``sum`` reduction (see module docstring for exactly which
    parts are Docker, which are real-but-not-Docker simulation, and which
    are OpenCake's own analytical coefficients)."""

    spec: SumCrossPESpec
    local_result: SumResult
    """Real per-PE local-reduction sub-run, representative of every one
    of ``spec.num_pes`` PEs (see module docstring)."""
    combine_add_result: Optional[AddResult]
    """Real single-add-firing sub-run, representative of every one of
    ``spec.real_combine_hop_count`` combine steps -- ``None`` only when
    ``num_pes=1`` (nothing to combine, see module docstring's
    DEGENERATE CASE)."""
    sim_time_cycles: float
    """Real total wall-clock cycles: local phase (all PEs in parallel,
    one real per-PE local_compute_cycles) + the real, sequential P-1-hop
    combine chain (real network hop timing + real per-hop add, back to
    back since each hop depends on the previous one's result)."""
    network_summary: dict[str, float | int]
    """Real ``PacketNetwork.summary()`` output for the P-1-hop relay
    chain (packet/byte/FIFO/link statistics) -- empty-network zeros when
    num_pes=1."""
    energy: EnergyBreakdown
    """Real Docker PE-internal energy (local reductions + combine adds,
    summed across every real physical PE/firing) plus OpenCake's own
    analytical network/static energy terms -- see module docstring."""
    area: AreaBreakdown
    """Real Docker per-PE area (adder + scratchpad) plus OpenCake's own
    analytical inter-PE link area -- see module docstring."""

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
            "combine_add_result": None if self.combine_add_result is None else self.combine_add_result.to_dict(),
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
    """Real per-PE area, preferring the LOCAL sub-run's real adder (the
    combine phase reuses that exact same physical adder, so no double
    count) but falling back to the COMBINE sub-run's real adder when the
    local phase was itself degenerate (``elements_per_pe == 1``, i.e.
    ``total_elements == num_pes`` -- a real, reachable configuration
    ``validate()`` allows) and so instantiated no adder of its own, even
    though a real combine phase still ran and DID instantiate one. Fixed
    here (added after the first version of this module silently reported
    a fabricated 0.0 area in that configuration -- see `max`'s sibling
    ``_cross_pe_max.py`` module docstring's "AREA ACCOUNTING FIX" section
    for the fuller note, applied identically here for consistency).
    Returns a true 0.0 only when neither sub-run ever ran (num_pes=1 AND
    elements_per_pe=1, i.e. total_elements=1 -- nothing real computed at
    all)."""
    if local_component_area is not None:
        return local_component_area + (local_scratchpad_area or 0.0)
    if combine_component_area is not None:
        return combine_component_area + (combine_scratchpad_area or 0.0)
    return 0.0


def simulate_sum_cross_pe(
    spec: SumCrossPESpec, wafer: WaferSpec, out_dir: Path
) -> SumCrossPEResult:
    """Run the real local-reduction Docker sub-run, then (if num_pes > 1)
    the real single-add-firing Docker sub-run plus the real, sequential
    P-1-hop network relay simulation -- composing all of them into one
    real cross-PE `sum` cost. Raises (never fabricates) if any real
    Docker sub-run fails, same discipline as every other primitive in
    this project.
    """
    spec.validate()

    local_spec = SumSpec(
        elements_per_pe=spec.elements_per_pe, bytes_per_element=spec.bytes_per_element
    )
    local_result = simulate_sum(local_spec, wafer, out_dir / "local")

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
            local_result.adder_area_um2, local_result.pe_scratchpad_area_um2, None, None
        )
        area = estimate_area(
            wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
        )
        return SumCrossPEResult(
            spec=spec,
            local_result=local_result,
            combine_add_result=None,
            sim_time_cycles=sim_time_cycles,
            network_summary=empty_network_summary,
            energy=energy,
            area=area,
        )

    # One real Docker sub-run for a single combine-add firing, reused for
    # every one of the hop_count real combine steps (see module docstring).
    combine_add_spec = AddSpec(elements_per_pe=1, bytes_per_element=spec.bytes_per_element)
    combine_add_result = simulate_add(combine_add_spec, wafer, out_dir / "combine_add")

    # Real, sequential P-1-hop relay chain along one row of the mesh:
    # PE(0, num_pes-1) -> PE(0, num_pes-2) -> ... -> PE(0, 0). Real network
    # timing via PacketNetwork._send_hop (the same code GEMM's cross-PE
    # broadcast uses); real per-hop add cost from combine_add_result.
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
        cycle = hop_end + combine_add_result.local_compute_cycles

    sim_time_cycles = max(cycle, network.latest_cycle)
    network_summary = network.summary(sim_time_cycles)

    pe_internal_energy_uj = (
        local_result.local_energy_uj * spec.num_pes
        + combine_add_result.local_energy_uj * hop_count
    )
    energy = estimate_energy(
        wafer=wafer,
        active_pes=spec.num_pes,
        sim_time_cycles=sim_time_cycles,
        pe_internal_energy_uj=pe_internal_energy_uj,
        network=network_summary,
    )

    pe_internal_area_um2 = _pe_internal_area_um2(
        local_result.adder_area_um2,
        local_result.pe_scratchpad_area_um2,
        combine_add_result.adder_area_um2,
        combine_add_result.pe_scratchpad_area_um2,
    )
    area = estimate_area(
        wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
    )

    return SumCrossPEResult(
        spec=spec,
        local_result=local_result,
        combine_add_result=combine_add_result,
        sim_time_cycles=sim_time_cycles,
        network_summary=network_summary,
        energy=energy,
        area=area,
    )
