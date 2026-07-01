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
    max_link_utilization: float
    compute_efficiency: float

    def summary(self) -> dict[str, float | int]:
        return {
            "ranks": self.ranks,
            "compute_ops": self.compute_ops,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "network_cycles": self.network_cycles,
            "sim_time_cycles": self.sim_time_cycles,
            "max_sram_bytes_per_rank": self.max_sram_bytes_per_rank,
            "max_sram_utilization": self.max_sram_utilization,
            "max_link_utilization": self.max_link_utilization,
            "compute_efficiency": self.compute_efficiency,
        }


class OpenCakeBackend:
    """Small deterministic backend for first-pass OpenCake IR timing."""

    def run(self, program: OpenCakeProgram, hardware: HardwareSpec) -> OpenCakeBackendResult:
        hardware.validate()

        compute_cycles_by_rank: dict[tuple[int, int], float] = {}
        memory_cycles_by_rank: dict[tuple[int, int], float] = {}
        sram_bytes_by_rank: dict[tuple[int, int], int] = {}
        link_busy_cycles: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        total_compute_ops = 0

        for op in program.ops:
            if isinstance(op, AllocOp):
                sram_bytes_by_rank[op.rank] = sram_bytes_by_rank.get(op.rank, 0) + op.bytes
            elif isinstance(op, ComputeOp):
                total_compute_ops += op.ops
                compute_cycles_by_rank[op.rank] = (
                    compute_cycles_by_rank.get(op.rank, 0.0)
                    + op.ops / hardware.compute_ops_per_cycle_per_rank
                )
                memory_cycles_by_rank[op.rank] = (
                    memory_cycles_by_rank.get(op.rank, 0.0)
                    + op.memory_bytes / hardware.sram_bandwidth_bytes_per_cycle_per_rank
                )
            elif isinstance(op, SendOp):
                per_link_cycles = (
                    hardware.router_latency_cycles
                    + op.bytes / hardware.link_bandwidth_bytes_per_cycle
                )
                for src, dst in zip(op.route, op.route[1:]):
                    link = (src, dst)
                    link_busy_cycles[link] = link_busy_cycles.get(link, 0.0) + per_link_cycles

        compute_cycles = max(compute_cycles_by_rank.values(), default=0.0)
        memory_cycles = max(memory_cycles_by_rank.values(), default=0.0)
        network_cycles = max(link_busy_cycles.values(), default=0.0)
        sim_time_cycles = max(compute_cycles, memory_cycles, network_cycles)
        max_sram_bytes = max(sram_bytes_by_rank.values(), default=0)
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
            max_sram_utilization=max_sram_bytes / hardware.sram_bytes_per_rank,
            max_link_utilization=network_cycles / max_link_peak,
            compute_efficiency=(total_compute_ops / peak_compute) if peak_compute else 0.0,
        )
