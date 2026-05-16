from __future__ import annotations

from wafersim.config import Coord


class ProcessingElement:
    def __init__(
        self,
        env,
        coord: Coord,
        compute_throughput_ops_per_cycle: float,
        local_sram_bytes: int,
        memory_bandwidth_bytes_per_cycle: float,
    ) -> None:
        self.env = env
        self.coord = coord
        self.compute_throughput_ops_per_cycle = compute_throughput_ops_per_cycle
        self.local_sram_bytes = local_sram_bytes
        self.memory_bandwidth_bytes_per_cycle = memory_bandwidth_bytes_per_cycle
        self.compute_busy_time = 0.0
        self.compute_tasks_completed = 0
        self.memory_busy_time = 0.0
        self.memory_tasks_completed = 0
        self.max_memory_footprint_bytes = 0

    def reserve_memory(self, bytes_needed: int) -> None:
        if bytes_needed > self.local_sram_bytes:
            raise ValueError(
                f"PE{self.coord} needs {bytes_needed} bytes, "
                f"but local SRAM is {self.local_sram_bytes} bytes"
            )
        self.max_memory_footprint_bytes = max(
            self.max_memory_footprint_bytes,
            bytes_needed,
        )

    def access_memory(self, bytes_accessed: int):
        duration = bytes_accessed / self.memory_bandwidth_bytes_per_cycle
        if duration > 0:
            yield self.env.timeout(duration)
        self.memory_busy_time += duration
        self.memory_tasks_completed += 1

    def compute(self, ops: int):
        duration = ops / self.compute_throughput_ops_per_cycle
        if duration > 0:
            yield self.env.timeout(duration)
        self.compute_busy_time += duration
        self.compute_tasks_completed += 1
