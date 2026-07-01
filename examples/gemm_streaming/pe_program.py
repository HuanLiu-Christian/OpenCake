from __future__ import annotations

from dataclasses import dataclass

from wafersim.program import PeProgramSpec, PeTaskSpec


PE_PROGRAM = PeProgramSpec(
    kernel="gemm_streaming",
    tasks=(
        PeTaskSpec(
            name="recv_a",
            description="Receive an A panel from host or the PE to the west.",
        ),
        PeTaskSpec(
            name="recv_b",
            description="Receive a B panel from host or the PE above.",
        ),
        PeTaskSpec(
            name="compute_local_c",
            description="Compute the PE-local C tile from A and B panels.",
        ),
        PeTaskSpec(
            name="send_c_d2h",
            description="Return the final C tile to the host.",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class GemmStreamingConfig:
    m: int = 4
    n: int = 4
    k: int = 4
    kernel_rows: int = 2
    kernel_cols: int = 2
    bytes_per_element: int = 4
    packet_payload_bytes: int = 64
    ops_per_mac: int = 2

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GEMM M, N, and K must be positive")
        if self.kernel_rows <= 0 or self.kernel_cols <= 0:
            raise ValueError("GEMM kernel dimensions must be positive")
        if self.m % self.kernel_rows != 0:
            raise ValueError("M must be divisible by kernel_rows")
        if self.n % self.kernel_cols != 0:
            raise ValueError("N must be divisible by kernel_cols")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.ops_per_mac <= 0:
            raise ValueError("ops per MAC must be positive")

    @property
    def rows_per_pe(self) -> int:
        return self.m // self.kernel_rows

    @property
    def cols_per_pe(self) -> int:
        return self.n // self.kernel_cols

    @property
    def active_pes(self) -> int:
        return self.kernel_rows * self.kernel_cols

    @property
    def a_panel_bytes_per_row(self) -> int:
        return self.rows_per_pe * self.k * self.bytes_per_element

    @property
    def b_panel_bytes_per_col(self) -> int:
        return self.k * self.cols_per_pe * self.bytes_per_element

    @property
    def c_tile_bytes_per_pe(self) -> int:
        return self.rows_per_pe * self.cols_per_pe * self.bytes_per_element

    @property
    def compute_ops(self) -> int:
        return self.m * self.n * self.k * self.ops_per_mac

    @property
    def compute_ops_per_pe(self) -> int:
        return self.rows_per_pe * self.cols_per_pe * self.k * self.ops_per_mac

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        return (
            self.a_panel_bytes_per_row
            + self.b_panel_bytes_per_col
            + self.c_tile_bytes_per_pe
        )
