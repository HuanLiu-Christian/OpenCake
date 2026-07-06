from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GemmWorkload:
    """Chakra-like high-level workload for one GEMM node.

    This is intentionally smaller than real Chakra ET. It captures the part
    needed by the first OpenCake prototype: GEMM shape, datatype size, and the
    operation count that a mapper can lower into PE-block work.
    """

    m: int
    n: int
    k: int
    bytes_per_element: int = 4
    ops_per_mac: int = 2

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GEMM dimensions must be positive")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")
        if self.ops_per_mac <= 0:
            raise ValueError("ops_per_mac must be positive")

    @property
    def total_ops(self) -> int:
        return self.ops_per_mac * self.m * self.n * self.k

    @property
    def input_bytes(self) -> int:
        return (self.m * self.k + self.k * self.n) * self.bytes_per_element

    @property
    def output_bytes(self) -> int:
        return self.m * self.n * self.bytes_per_element


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Minimal wafer-scale machine model used by OpenCake mapping."""

    mesh_rows: int = 32
    mesh_cols: int = 32
    pe_block_rows: int = 1
    pe_block_cols: int = 1
    compute_ops_per_cycle_per_rank: float = 64.0
    sram_bandwidth_bytes_per_cycle_per_rank: float = 16.0
    link_bandwidth_bytes_per_cycle: float = 128.0
    router_latency_cycles: float = 1.0
    queue_capacity: int = 8
    sram_bytes_per_rank: int = 48 * 1024
    gemm_k_tile: int = 64
    double_buffer_ab: bool = True

    def validate(self) -> None:
        if self.mesh_rows <= 0 or self.mesh_cols <= 0:
            raise ValueError("mesh dimensions must be positive")
        if self.pe_block_rows <= 0 or self.pe_block_cols <= 0:
            raise ValueError("PE block dimensions must be positive")
        if self.mesh_rows % self.pe_block_rows or self.mesh_cols % self.pe_block_cols:
            raise ValueError("PE block dimensions must divide the mesh")
        if self.compute_ops_per_cycle_per_rank <= 0:
            raise ValueError("compute throughput must be positive")
        if self.sram_bandwidth_bytes_per_cycle_per_rank <= 0:
            raise ValueError("SRAM bandwidth must be positive")
        if self.link_bandwidth_bytes_per_cycle <= 0:
            raise ValueError("link bandwidth must be positive")
        if self.router_latency_cycles < 0:
            raise ValueError("router latency cannot be negative")
        if self.queue_capacity <= 0:
            raise ValueError("queue capacity must be positive")
        if self.sram_bytes_per_rank <= 0:
            raise ValueError("SRAM capacity must be positive")
        if self.gemm_k_tile <= 0:
            raise ValueError("GEMM K tile must be positive")

    @property
    def rank_rows(self) -> int:
        return self.mesh_rows // self.pe_block_rows

    @property
    def rank_cols(self) -> int:
        return self.mesh_cols // self.pe_block_cols

    @property
    def ranks(self) -> int:
        return self.rank_rows * self.rank_cols

    @property
    def pes_per_rank(self) -> int:
        return self.pe_block_rows * self.pe_block_cols

    @property
    def sram_banks_per_rank(self) -> int:
        return self.pes_per_rank

    @property
    def sram_bytes_per_bank(self) -> int:
        return self.sram_bytes_per_rank // self.sram_banks_per_rank

    @property
    def rank_model(self) -> str:
        return "pe" if self.pes_per_rank == 1 else "pe_block"
