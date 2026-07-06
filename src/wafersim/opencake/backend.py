from __future__ import annotations

from dataclasses import dataclass

from wafersim.opencake.ir import AllocOp, ComputeOp, OpenCakeProgram, SendOp
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class OpenCakeBackendResult:
    ranks: int
    compute_ops: int
    compute_cycles: float
    memory_cycles: float
    network_cycles: float
    sim_time_cycles: float
    max_sram_bytes_per_rank: int
    max_sram_utilization: float
    max_sram_pressure_ratio: float
    sram_capacity_feasible: bool
    max_link_utilization: float
    compute_efficiency: float

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "ranks": self.ranks,
            "compute_ops": self.compute_ops,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "network_cycles": self.network_cycles,
            "sim_time_cycles": self.sim_time_cycles,
            "max_sram_bytes_per_rank": self.max_sram_bytes_per_rank,
            "max_sram_utilization": self.max_sram_utilization,
            "max_sram_pressure_ratio": self.max_sram_pressure_ratio,
            "sram_capacity_feasible": self.sram_capacity_feasible,
            "max_link_utilization": self.max_link_utilization,
            "compute_efficiency": self.compute_efficiency,
        }


class OpenCakeBackend:
    """Timestamp scheduler for first-pass OpenCake IR timing."""

    def run(self, program: OpenCakeProgram, hardware: HardwareSpec) -> OpenCakeBackendResult:
        hardware.validate()

        compute_ready_by_rank: dict[tuple[int, int], float] = {}
        memory_ready_by_rank: dict[tuple[int, int], float] = {}
        compute_busy_by_rank: dict[tuple[int, int], float] = {}
        memory_busy_by_rank: dict[tuple[int, int], float] = {}
        sram_bytes_by_rank: dict[tuple[int, int], int] = {}
        link_busy_cycles: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        link_ready_cycles: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        tensor_ready_by_rank: dict[tuple[tuple[int, int], str, int], float] = {}
        total_compute_ops = 0
        last_end_cycle = 0.0

        for op in program.ops:
            if isinstance(op, AllocOp):
                sram_bytes_by_rank[op.rank] = sram_bytes_by_rank.get(op.rank, 0) + op.bytes
            elif isinstance(op, ComputeOp):
                total_compute_ops += op.ops
                a_ready = tensor_ready_by_rank.get((op.rank, "A", op.k_start), 0.0)
                b_ready = tensor_ready_by_rank.get((op.rank, "B", op.k_start), 0.0)
                memory_start = max(
                    a_ready,
                    b_ready,
                    memory_ready_by_rank.get(op.rank, 0.0),
                )
                memory_duration = op.memory_bytes / hardware.sram_bandwidth_bytes_per_cycle_per_rank
                memory_end = memory_start + memory_duration
                memory_ready_by_rank[op.rank] = memory_end
                memory_busy_by_rank[op.rank] = (
                    memory_busy_by_rank.get(op.rank, 0.0) + memory_duration
                )

                compute_start = max(memory_end, compute_ready_by_rank.get(op.rank, 0.0))
                compute_duration = op.ops / hardware.compute_ops_per_cycle_per_rank
                compute_end = compute_start + compute_duration
                compute_ready_by_rank[op.rank] = compute_end
                compute_busy_by_rank[op.rank] = (
                    compute_busy_by_rank.get(op.rank, 0.0) + compute_duration
                )
                last_end_cycle = max(last_end_cycle, memory_end, compute_end)
            elif isinstance(op, SendOp):
                per_link_cycles = (
                    hardware.router_latency_cycles
                    + op.bytes / hardware.link_bandwidth_bytes_per_cycle
                )
                arrival_cycle = tensor_ready_by_rank.get((op.src_rank, op.tensor, op.k_start), 0.0)
                for src, dst in zip(op.route, op.route[1:]):
                    link = (src, dst)
                    start_cycle = max(arrival_cycle, link_ready_cycles.get(link, 0.0))
                    end_cycle = start_cycle + per_link_cycles
                    link_ready_cycles[link] = end_cycle
                    link_busy_cycles[link] = link_busy_cycles.get(link, 0.0) + per_link_cycles
                    arrival_cycle = end_cycle
                if op.tensor:
                    tensor_ready_by_rank[(op.dst_rank, op.tensor, op.k_start)] = arrival_cycle
                last_end_cycle = max(last_end_cycle, arrival_cycle)

        compute_cycles = max(compute_busy_by_rank.values(), default=0.0)
        memory_cycles = max(memory_busy_by_rank.values(), default=0.0)
        network_cycles = max(link_busy_cycles.values(), default=0.0)
        sim_time_cycles = last_end_cycle
        max_sram_bytes = max(sram_bytes_by_rank.values(), default=0)
        sram_pressure_ratio = max_sram_bytes / hardware.sram_bytes_per_rank
        peak_compute = sim_time_cycles * hardware.compute_ops_per_cycle_per_rank * hardware.ranks
        max_link_peak = sim_time_cycles if sim_time_cycles else 1.0

        return OpenCakeBackendResult(
            ranks=hardware.ranks,
            compute_ops=total_compute_ops,
            compute_cycles=compute_cycles,
            memory_cycles=memory_cycles,
            network_cycles=network_cycles,
            sim_time_cycles=sim_time_cycles,
            max_sram_bytes_per_rank=max_sram_bytes,
            max_sram_utilization=min(sram_pressure_ratio, 1.0),
            max_sram_pressure_ratio=sram_pressure_ratio,
            sram_capacity_feasible=sram_pressure_ratio <= 1.0,
            max_link_utilization=network_cycles / max_link_peak,
            compute_efficiency=(total_compute_ops / peak_compute) if peak_compute else 0.0,
        )
