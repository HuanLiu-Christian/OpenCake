from __future__ import annotations

from dataclasses import dataclass

Coord = tuple[int, int]


@dataclass(frozen=True, slots=True)
class MeshConfig:
    rows: int = 32
    cols: int = 32
    link_bandwidth_bytes_per_cycle: float = 32.0
    router_latency_cycles: float = 1.0
    link_queue_capacity: int = 8
    compute_throughput_ops_per_cycle: float = 256.0
    local_sram_bytes_per_pe: int = 48 * 1024
    memory_bandwidth_bytes_per_cycle: float = 128.0

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("mesh rows and cols must be positive")
        if self.link_bandwidth_bytes_per_cycle <= 0:
            raise ValueError("link bandwidth must be positive")
        if self.router_latency_cycles < 0:
            raise ValueError("router latency cannot be negative")
        if self.link_queue_capacity <= 0:
            raise ValueError("link queue capacity must be positive")
        if self.compute_throughput_ops_per_cycle <= 0:
            raise ValueError("compute throughput must be positive")
        if self.local_sram_bytes_per_pe <= 0:
            raise ValueError("local SRAM size must be positive")
        if self.memory_bandwidth_bytes_per_cycle <= 0:
            raise ValueError("memory bandwidth must be positive")
