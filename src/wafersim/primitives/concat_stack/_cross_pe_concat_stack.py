"""Real, Docker-backed + real-network-simulated cross-PE extension of the
`concat`/`stack` primitive: assemble one ``total_elements``-element real
output tensor whose ``num_pes`` real input shards SHARD ACROSS ``num_pes``
distinct physical PEs (not all living on one PE), the way a real
tensor-/sequence-parallel layout actually requires it -- e.g. recombining
multi-head attention output when different heads live on different PEs
before the output projection, or appending a new K/V step into a KV cache
whose existing entries are sharded across the mesh (per this package's own
``__init__.py`` docstring, "most importantly for real autoregressive
generation -- appending each step's new K/V into the growing KV cache").
Local-only ``ConcatStackSpec``/``simulate_concat_stack`` (see
``_timeloop_concat.py``) stays correct and untouched for the case where
every input tensor already lives on one PE -- this module is additive,
for the other real case.

STRUCTURAL DIFFERENCE FROM `sum`/`max` CROSS-PE (read this before the rest
of this docstring -- concat/stack is NOT a reduction): `sum`/`max`'s
cross-PE modules both have a real LOCAL COMPUTE phase (every source PE
reduces its own shard before sending one scalar) followed by a real
COMBINE-VIA-ARITHMETIC relay chain. concat/stack has no reduction and no
per-source compute at all -- each source PE already holds its real shard's
values as-is and only needs to MOVE them to the PE assembling the final
output. So this module has no "local_result" the way `sum`/`max` do; the
only real Docker cost is the real memory-move WRITE that happens each time
a shard is appended into the growing output buffer -- see MECHANISM below.

WHY THIS WAS MISSING (real research, not assumed): Cerebras's own official
CSL SDK docs (``collectives_2d`` library, https://sdk.cerebras.net/csl/
language/libraries) do not expose a dedicated concat/stack primitive, but
DO expose ``gather(root, send_buf, recv_buf, count, callback)`` -- "an
MPI_Gather for buffers of f32", explicitly documented as accommodating
``count*N`` elements in ``recv_buf`` for ``N`` PEs, i.e. real hardware
support for exactly this "many real per-PE buffers -> one combined real
buffer on one PE" shape. Separately, WaferLLM (arXiv 2502.04563) uses the
specific phrase "shift-based redistribution" for real KV-cache-concat-style
data movement across the mesh -- a real, named mechanism distinct from the
allreduce/GEMV language it uses for `sum`-shaped reductions (see
`sum._cross_pe_sum`'s own module docstring). Both citations already
appeared in this project's own primitive-research discussion before this
module was written.

WHAT NEITHER SOURCE SAYS: neither Cerebras's docs nor WaferLLM's paper
publish the exact real hop-by-hop mechanics of `gather`/"shift-based
redistribution" (whether every source pushes directly to the root over a
dedicated multi-hop path, whether transfers pipeline, whether it is
literally an accumulating shift as modeled here). This module does not
claim to reproduce either's proprietary internal implementation.

MECHANISM (one concrete, named MODELING CHOICE -- a SEQUENTIAL SHIFT
CHAIN, chosen because it is the one real, named phrase in the sources
above, "shift-based redistribution", and because it stays consistent with
`sum`/`max`'s own "sequential relay, not parallel tree" convention):
PE(0, num_pes-1) sends its own real shard (``elements_per_pe`` elements)
west to PE(0, num_pes-2). That PE does a real memory-move WRITE (see
``_memory_move.py``'s own real ``bitwise`` identity-pass-through
component -- the exact same real Docker cost model local `concat`/`stack`
already uses) to append the ``elements_per_pe`` newly-arrived elements
into its own growing buffer (which already holds its own local shard),
then relays the COMBINED, now-larger payload further west to PE(0,
num_pes-3), which does the same real append-and-relay. This repeats
P-1 times until PE(0, 0) receives everything and performs the final real
append of its own local shard, ending up holding the complete, real,
``total_elements``-element concatenated/stacked output. Each hop's
network payload GROWS (hop k carries ``k`` shards' worth of real bytes,
matching the "shift" framing -- data physically accumulates as it moves,
not a fixed-size scalar the way `sum`/`max`'s combine payload is), chunked
into real packets via ``PacketNetwork``'s own ``packet_payload_bytes``
convention (the same chunking `wafer/network.py`'s ``broadcast_row``/
``broadcast_column`` already use for GEMM's large panel transfers -- NOT
skipped here the way `sum`/`max`'s single-scalar combine payload could
skip it, because concat/stack's real per-hop payload can genuinely exceed
one packet).

WHAT IS REAL DOCKER, WHAT IS REAL SIMULATION-BUT-NOT-DOCKER, AND WHAT IS
OPENCAKE'S OWN ANALYTICAL COEFFICIENT:
  1. REAL DOCKER: ONE real ``simulate_memory_move()`` sub-run for a
     single shard-sized append (``elements_per_pe`` real ``bitwise``
     identity firings, the exact same real component local `concat`/
     `stack` already uses) -- representative of every one of the
     ``num_pes - 1`` real append operations along the relay chain (every
     append writes the identical ``elements_per_pe``-sized NEW chunk,
     never the already-accumulated bytes passing through, so one real,
     representative Docker sub-run correctly covers all of them, same
     "run once, reuse for every identical firing" discipline `sum`/`max`
     already established for their combine steps).
  2. REAL SIMULATION, NOT DOCKER: the real hop-by-hop LINK/FIFO timing
     for the P-1 relay hops (with growing, real-packet-chunked payload
     sizes), via ``wafer.network.PacketNetwork._send_hop``/
     ``_packet_sizes`` -- the same real code GEMM's own cross-PE
     broadcast and `sum`/`max`'s cross-PE combine chains already use.
  3. OPENCAKE'S OWN ANALYTICAL COEFFICIENT: the ENERGY of those P-1 real
     hops (fifo/router/wire pJ-per-byte-hop terms) and static leakage --
     identical reuse of ``wafer.energy.estimate_energy``/
     ``wafer.area.estimate_area``, same discipline and same honesty about
     what they are as `sum`/`max`/GEMM.

SHARDING ASSUMPTION: ``total_elements`` must divide evenly by ``num_pes``
(raises otherwise) -- same convention as `sum`/`max`'s cross-PE modules;
an even shard per PE is what this module models.

DEGENERATE CASE (num_pes=1): the whole output already lives on one PE --
this is exactly local `concat`/`stack`'s own case, so this module calls
``simulate_concat_stack()`` directly (one real memory-move sub-run for
the FULL ``total_elements`` output, matching local concat/stack's own
"every output element gets one real write" convention) and returns it
with a true zero-cost network phase, not a fabricated near-zero number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from ...wafer.area import AreaBreakdown, estimate_area
from ...wafer.energy import EnergyBreakdown, estimate_energy
from ...wafer.network import PacketNetwork
from .._memory_move import MemoryMoveResult, MemoryMoveSpec, simulate_memory_move
from ._timeloop_concat import ConcatStackSpec, simulate_concat_stack


@dataclass(frozen=True, slots=True)
class ConcatStackCrossPESpec:
    """Assemble a ``total_elements``-element real concat/stack output,
    evenly sharded across ``num_pes`` distinct physical PEs arranged in
    one row of the mesh, into one real contiguous buffer on the leftmost
    PE -- see module docstring for the real mechanism (a P-1-hop
    sequential shift-relay chain of real appends)."""

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
    def elements_per_shard(self) -> int:
        """Real element count each of the num_pes PEs already holds
        locally before the cross-PE shift-relay phase."""
        return self.total_elements // self.num_pes

    @property
    def real_relay_hop_count(self) -> int:
        """Exact number of real relay hops (and real per-hop append
        firings) the sequential shift chain requires: num_pes - 1 (0
        when num_pes=1). Not an estimate -- every real shard except the
        destination's own must physically move exactly once."""
        return max(self.num_pes - 1, 0)


@dataclass(frozen=True, slots=True)
class ConcatStackCrossPEResult:
    """Real, Docker-verified + real-network-simulated cost of one
    cross-PE ``concat``/``stack`` assembly (see module docstring for
    exactly which parts are Docker, which are real-but-not-Docker
    simulation, and which are OpenCake's own analytical coefficients)."""

    spec: ConcatStackCrossPESpec
    local_result: Optional[MemoryMoveResult]
    """Real single-PE whole-output memory-move sub-run -- set ONLY in the
    ``num_pes=1`` degenerate case (see module docstring's DEGENERATE
    CASE), ``None`` otherwise."""
    append_result: Optional[MemoryMoveResult]
    """Real single-shard-sized append sub-run, representative of every
    one of ``spec.real_relay_hop_count`` real per-hop appends -- set
    ONLY when ``num_pes > 1``, ``None`` in the degenerate case (mutually
    exclusive with ``local_result``, see module docstring's STRUCTURAL
    DIFFERENCE section: concat/stack has no separate "local compute"
    phase the way `sum`/`max` do)."""
    sim_time_cycles: float
    """Real total wall-clock cycles: degenerate case is just the single
    real memory-move's cycles; otherwise the real, sequential P-1-hop
    shift-relay chain (real network hop timing, real-packet-chunked,
    growing payload, + real per-hop append, back to back since each hop
    depends on the previous one's result)."""
    network_summary: dict[str, float | int]
    """Real ``PacketNetwork.summary()`` output for the P-1-hop
    shift-relay chain -- empty-network zeros when num_pes=1."""
    energy: EnergyBreakdown
    """Real Docker PE-internal energy (either the single degenerate copy,
    or every real per-hop append summed across the ``real_relay_hop_count``
    distinct receiving PEs) plus OpenCake's own analytical network/static
    energy terms."""
    area: AreaBreakdown
    """Real Docker per-PE area (bitwise unit + scratchpad, from whichever
    real sub-run actually ran) plus OpenCake's own analytical inter-PE
    link area."""

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
                "elements_per_shard": self.spec.elements_per_shard,
                "bytes_per_element": self.spec.bytes_per_element,
                "real_relay_hop_count": self.spec.real_relay_hop_count,
            },
            "sim_time_cycles": self.sim_time_cycles,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "local_result": None if self.local_result is None else self.local_result.to_dict(),
            "append_result": None if self.append_result is None else self.append_result.to_dict(),
            "network_summary": self.network_summary,
            "energy": self.energy.to_dict(),
            "area": self.area.to_dict(),
        }


def _pe_internal_area_um2(component_area: Optional[float], scratchpad_area: Optional[float]) -> float:
    """Real per-PE area from whichever real sub-run actually ran (there
    is never ambiguity here the way `sum`/`max` needed a fallback for --
    ``local_result``/``append_result`` are mutually exclusive by
    construction, see the result dataclass's own field docstrings) --
    ``None`` only in the true zero-work case (``total_elements=0``, which
    ``validate()`` already rejects, so this is defensive, not reachable
    in practice)."""
    if component_area is None:
        return 0.0
    return component_area + (scratchpad_area or 0.0)


def simulate_concat_stack_cross_pe(
    spec: ConcatStackCrossPESpec, wafer: WaferSpec, out_dir: Path
) -> ConcatStackCrossPEResult:
    """Run the real, Docker-verified cost for assembling one real
    ``total_elements``-element concat/stack output whose shards are
    sharded across ``num_pes`` physical PEs -- degenerate single-PE case
    via local ``simulate_concat_stack()`` directly, else a real
    representative per-hop append Docker sub-run plus the real,
    sequential P-1-hop shift-relay network simulation (see module
    docstring). Raises (never fabricates) if any real Docker sub-run
    fails, same discipline as every other primitive in this project.
    """
    spec.validate()

    hop_count = spec.real_relay_hop_count
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
        local_spec = ConcatStackSpec(
            elements_per_pe=spec.total_elements, bytes_per_element=spec.bytes_per_element
        )
        local_result = simulate_concat_stack(local_spec, wafer, out_dir / "local")

        sim_time_cycles = float(local_result.local_compute_cycles)
        energy = estimate_energy(
            wafer=wafer,
            active_pes=spec.num_pes,
            sim_time_cycles=sim_time_cycles,
            pe_internal_energy_uj=local_result.local_energy_uj,
            network=empty_network_summary,
        )
        pe_internal_area_um2 = _pe_internal_area_um2(
            local_result.bitwise_area_um2, local_result.pe_scratchpad_area_um2
        )
        area = estimate_area(
            wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
        )
        return ConcatStackCrossPEResult(
            spec=spec,
            local_result=local_result,
            append_result=None,
            sim_time_cycles=sim_time_cycles,
            network_summary=empty_network_summary,
            energy=energy,
            area=area,
        )

    # One real Docker sub-run for a single shard-sized append, reused for
    # every one of the hop_count real per-hop appends (see module
    # docstring point 1).
    append_spec = MemoryMoveSpec(
        elements_per_pe=spec.elements_per_shard, bytes_per_element=spec.bytes_per_element
    )
    append_result = simulate_memory_move(append_spec, wafer, out_dir / "append")

    # Real, sequential P-1-hop shift-relay chain along one row of the
    # mesh: PE(0, num_pes-1) -> PE(0, num_pes-2) -> ... -> PE(0, 0), each
    # hop's real payload growing by one more shard's worth of real bytes
    # (see module docstring's MECHANISM section), chunked into real
    # packets the same way GEMM's broadcast_row/broadcast_column already
    # do for large transfers.
    network = PacketNetwork(wafer, rows=1, cols=spec.num_pes)
    shard_bytes = spec.elements_per_shard * spec.bytes_per_element
    cycle = 0.0  # sources already hold their real shard values -- no compute needed before the first send
    for col in range(spec.num_pes - 1, 0, -1):
        carried_shards = spec.num_pes - col
        total_hop_bytes = carried_shards * shard_bytes
        hop_end = cycle
        for packet_size in network._packet_sizes(total_hop_bytes):
            hop_end = network._send_hop(
                ready_cycle=cycle,
                size_bytes=packet_size,
                source=(0, col),
                destination=(0, col - 1),
                direction="west",
            )
        # Real memory-move append: the receiving PE writes the ONE
        # newly-arrived shard's elements_per_shard elements into its own
        # growing buffer before relaying onward (real work, not free --
        # see this package's own __init__.py: "appending is NOT free").
        cycle = hop_end + append_result.local_compute_cycles

    sim_time_cycles = max(cycle, network.latest_cycle)
    network_summary = network.summary(sim_time_cycles)

    pe_internal_energy_uj = append_result.local_energy_uj * hop_count
    energy = estimate_energy(
        wafer=wafer,
        active_pes=spec.num_pes,
        sim_time_cycles=sim_time_cycles,
        pe_internal_energy_uj=pe_internal_energy_uj,
        network=network_summary,
    )

    pe_internal_area_um2 = _pe_internal_area_um2(
        append_result.bitwise_area_um2, append_result.pe_scratchpad_area_um2
    )
    area = estimate_area(
        wafer=wafer, active_pes=spec.num_pes, pe_internal_area_um2=pe_internal_area_um2
    )

    return ConcatStackCrossPEResult(
        spec=spec,
        local_result=None,
        append_result=append_result,
        sim_time_cycles=sim_time_cycles,
        network_summary=network_summary,
        energy=energy,
        area=area,
    )
