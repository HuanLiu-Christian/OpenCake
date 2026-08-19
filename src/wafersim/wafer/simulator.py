"""Cycle-level analytical model for a tiled GEMM on a 2-D PE mesh."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from wafersim.contract import GemmMappingContract
from wafersim.wafer.area import AreaBreakdown, estimate_area
from wafersim.wafer.energy import EnergyBreakdown, estimate_energy
from wafersim.wafer.network import PacketNetwork


@dataclass(frozen=True, slots=True)
class WaferGemmResult:
    """Performance, energy, and area results for one complete GEMM execution."""

    contract: GemmMappingContract
    sim_time_cycles: float
    compute_busy_cycles: float
    mac_operations: int
    sram_access_bytes: int
    network: dict[str, float | int]
    energy: EnergyBreakdown
    area: AreaBreakdown

    @property
    def mean_pe_utilization(self) -> float:
        denominator = self.contract.active_pes * self.sim_time_cycles
        return self.compute_busy_cycles / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.contract.to_dict(),
            "sim_time_cycles": self.sim_time_cycles,
            "mac_operations": self.mac_operations,
            "compute_busy_cycles": self.compute_busy_cycles,
            "mean_pe_utilization": self.mean_pe_utilization,
            "sram_access_bytes": self.sram_access_bytes,
            "network": self.network,
            "energy": self.energy.to_dict(),
            "area": self.area.to_dict(),
        }


def _extent(start: int, tile: int, limit: int) -> int:
    return max(0, min(tile, limit - start))


def simulate_gemm(contract: GemmMappingContract) -> WaferGemmResult:
    """Execute a 2-D SUMMA-style, K-phased GEMM schedule with rotating panel
    ownership.

    Classic SUMMA: at K-phase p, the A panel for row r is resident at
    PE(r, p mod active_cols) -- the panel *owner* -- which broadcasts it to
    the rest of row r in both directions (east and west). Symmetrically,
    the B panel for column c is resident at PE(p mod active_rows, c) and
    broadcasts both south and north. The owner rotates with p rather than
    always sitting at column/row 0, so over a full run every PE must be
    able to both send and receive in all four directions -- see
    wafer/network.py's PacketNetwork, which gives each PE one independent
    FIFO per direction. Each PE computes its local C(r, c) update only
    after both panels of a phase have reached that PE.
    """

    contract.validate()
    gemm = contract.gemm
    wafer = contract.wafer
    network = PacketNetwork(wafer, contract.active_rows, contract.active_cols)

    # Real, TimeLoop-derived per-MAC compute rate for THIS mapping -- not an
    # assumed throughput constant (there used to be a `wafer.
    # macs_per_cycle_per_pe` config default here; it has been removed, see
    # specs.py, precisely because it let a made-up number silently stand in
    # for TimeLoop's own real Cycles result, defeating the point of running
    # TimeLoop at all). `contract.local_compute_cycles` is the real,
    # Accelergy-timed cycle count TimeLoop measured for exactly ONE PE's
    # full temporal loop nest under this mapping: pe_scratchpad's temporal
    # factors (tile_m, tile_n, tile_k) repeated over k_phases -- i.e.
    # tile_m * tile_n * K MACs total, since every accepted tile_k evenly
    # divides K (tile_k * k_phases == gemm.k exactly). Dividing the real
    # Cycles by that same real MAC count gives the real cycles/MAC rate
    # TimeLoop measured (confirmed against a real Docker run: 1,048,576
    # cycles / 1,048,576 MACs == 1.0, i.e. this project's single-scalar-MAC
    # architecture.yaml really does cost 1 cycle/MAC, not the old assumed
    # 32). Applying that real rate to each phase's/PE's MAC count below
    # (still needed per-PE, per-phase for boundary tiles and the
    # cycle-accurate broadcast/compute timeline) keeps every compute-time
    # number in this simulation anchored to TimeLoop's real measurement.
    interior_macs_per_pe = contract.tile_m * contract.tile_n * gemm.k
    cycles_per_mac = contract.local_compute_cycles / interior_macs_per_pe

    compute_ready = {
        (row, col): 0.0
        for row in range(contract.active_rows)
        for col in range(contract.active_cols)
    }
    # Single-buffered A/B, tracked independently (per real Cerebras CSL GEMM
    # collective benchmark, see https://sdk.cerebras.net/csl/code-examples/
    # benchmark-gemm-collectives -- confirmed against real doc content: its
    # broadcast/compute overlap is task-level block/unblock synchronization
    # (compute_task starts blocked; each direction's broadcast-done callback
    # unblocks it; compute re-blocks itself before the next phase's
    # broadcast begins), NOT a slot-alternating prefetch-ahead buffer -- so
    # phase P+1's broadcast into a PE's A/B space does not begin until phase
    # P's compute has fully finished reading that same space. One buffer per
    # PE is therefore enough; there is no second slot for a next phase to
    # write into early. A and B are tracked as two independent timestamps
    # (previously a single combined one) since real Cerebras keeps genuinely
    # separate per-direction task synchronization (x_done/y_done) for them.
    a_buffer_ready = {
        (row, col): 0.0
        for row in range(contract.active_rows)
        for col in range(contract.active_cols)
    }
    b_buffer_ready = {
        (row, col): 0.0
        for row in range(contract.active_rows)
        for col in range(contract.active_cols)
    }

    compute_busy = 0.0
    sram_access_bytes = 0
    mac_operations = 0

    for phase in range(contract.k_phases):
        k_start = phase * contract.tile_k
        k_extent = _extent(k_start, contract.tile_k, gemm.k)
        # Rotating SUMMA panel ownership: this phase's A panels live at
        # column `owner_col` of each row, this phase's B panels live at row
        # `owner_row` of each column -- not fixed at column/row 0.
        owner_col = phase % contract.active_cols
        owner_row = phase % contract.active_rows

        a_arrival_by_row: dict[int, float] = {}
        for row in range(contract.active_rows):
            m_extent = _extent(row * contract.tile_m, contract.tile_m, gemm.m)
            payload = m_extent * k_extent * gemm.bytes_per_element
            source_ready = a_buffer_ready[(row, owner_col)]
            a_arrival_by_row[row] = network.broadcast_row(
                row=row,
                ready_cycle=source_ready,
                size_bytes=payload,
                owner_col=owner_col,
            )

        b_arrival_by_col: dict[int, float] = {}
        for col in range(contract.active_cols):
            n_extent = _extent(col * contract.tile_n, contract.tile_n, gemm.n)
            payload = k_extent * n_extent * gemm.bytes_per_element
            source_ready = b_buffer_ready[(owner_row, col)]
            b_arrival_by_col[col] = network.broadcast_column(
                col=col,
                ready_cycle=source_ready,
                size_bytes=payload,
                owner_row=owner_row,
            )

        for row in range(contract.active_rows):
            m_extent = _extent(row * contract.tile_m, contract.tile_m, gemm.m)
            for col in range(contract.active_cols):
                n_extent = _extent(col * contract.tile_n, contract.tile_n, gemm.n)
                start = max(
                    a_arrival_by_row[row][(row, col)],
                    b_arrival_by_col[col][(row, col)],
                    compute_ready[(row, col)],
                    a_buffer_ready[(row, col)],
                    b_buffer_ready[(row, col)],
                )
                macs = m_extent * n_extent * k_extent
                compute_cycles = ceil(macs * cycles_per_mac)
                end = start + compute_cycles

                compute_ready[(row, col)] = end
                # Both freed at the same moment: this PE's compute for this
                # phase is what was holding them, real Cerebras task
                # synchronization frees the space for the next occupant only
                # once compute is done with it, not sooner.
                a_buffer_ready[(row, col)] = end
                b_buffer_ready[(row, col)] = end
                compute_busy += compute_cycles
                mac_operations += macs

                a_bytes = m_extent * k_extent * gemm.bytes_per_element
                b_bytes = k_extent * n_extent * gemm.bytes_per_element
                c_bytes = m_extent * n_extent * gemm.bytes_per_element
                # A/B write + read; C read-modify-write during this K update.
                sram_access_bytes += 2 * a_bytes + 2 * b_bytes + 2 * c_bytes

    sim_time = max(max(compute_ready.values(), default=0.0), network.latest_cycle)
    network_summary = network.summary(sim_time)
    energy = estimate_energy(
        wafer=wafer,
        pe_internal_energy_uj=contract.local_energy_uj,
        network=network_summary,
        sim_time_cycles=sim_time,
        active_pes=contract.active_pes,
    )
    area = estimate_area(
        wafer=wafer,
        active_pes=contract.active_pes,
        pe_internal_area_um2=contract.pe_internal_area_um2,
    )
    return WaferGemmResult(
        contract=contract,
        sim_time_cycles=sim_time,
        compute_busy_cycles=compute_busy,
        mac_operations=mac_operations,
        sram_access_bytes=sram_access_bytes,
        network=network_summary,
        energy=energy,
        area=area,
    )
