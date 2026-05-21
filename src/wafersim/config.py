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


@dataclass(frozen=True, slots=True)
class StencilWorkloadConfig:
    type: str = "stencil"
    steps: int = 8
    tile_rows: int = 32
    tile_cols: int = 32
    ops_per_cell: int = 64
    halo_bytes_per_edge: int = 512
    bytes_per_cell: int = 4
    memory_reads_per_cell: int = 5
    memory_writes_per_cell: int = 1
    double_buffer_tiles: bool = True

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("workload steps must be positive")
        if self.tile_rows <= 0 or self.tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        if self.ops_per_cell < 0:
            raise ValueError("ops per cell cannot be negative")
        if self.halo_bytes_per_edge < 0:
            raise ValueError("halo bytes per edge cannot be negative")
        if self.bytes_per_cell <= 0:
            raise ValueError("bytes per cell must be positive")
        if self.memory_reads_per_cell < 0:
            raise ValueError("memory reads per cell cannot be negative")
        if self.memory_writes_per_cell < 0:
            raise ValueError("memory writes per cell cannot be negative")

    @property
    def cells_per_tile(self) -> int:
        return self.tile_rows * self.tile_cols

    @property
    def ops_per_tile(self) -> int:
        return self.cells_per_tile * self.ops_per_cell

    @property
    def tile_storage_bytes(self) -> int:
        buffers = 2 if self.double_buffer_tiles else 1
        return self.cells_per_tile * self.bytes_per_cell * buffers

    @property
    def halo_storage_bytes(self) -> int:
        return 4 * self.halo_bytes_per_edge

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        return self.tile_storage_bytes + self.halo_storage_bytes

    @property
    def memory_access_bytes_per_step(self) -> int:
        accesses_per_cell = self.memory_reads_per_cell + self.memory_writes_per_cell
        return self.cells_per_tile * self.bytes_per_cell * accesses_per_cell


@dataclass(frozen=True, slots=True)
class MemoryCopyWorkloadConfig:
    type: str = "memory_copy"
    elements: int = 8_192
    bytes_per_element: int = 4
    packet_payload_bytes: int = 1024
    source_row: int = 0
    source_col: int = 0
    dest_row: int | None = None
    dest_col: int | None = None
    injection_gap_cycles: float = 0.0

    def validate(self) -> None:
        if self.elements <= 0:
            raise ValueError("array elements must be positive")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.source_row < 0 or self.source_col < 0:
            raise ValueError("source coordinate cannot be negative")
        if self.dest_row is not None and self.dest_row < 0:
            raise ValueError("destination row cannot be negative")
        if self.dest_col is not None and self.dest_col < 0:
            raise ValueError("destination col cannot be negative")
        if self.injection_gap_cycles < 0:
            raise ValueError("injection gap cannot be negative")

    @property
    def array_bytes(self) -> int:
        return self.elements * self.bytes_per_element

    @property
    def packet_count(self) -> int:
        return (self.array_bytes + self.packet_payload_bytes - 1) // (
            self.packet_payload_bytes
        )


@dataclass(frozen=True, slots=True)
class GemmWorkloadConfig:
    type: str = "gemm"
    m: int = 1024
    n: int = 1024
    k: int = 1024
    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32
    bytes_per_element: int = 4
    ops_per_mac: int = 2
    double_buffer_inputs: bool = True
    pipeline_k_tiles: bool = True

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("GEMM dimensions must be positive")
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError("GEMM tile dimensions must be positive")
        if self.m % self.tile_m != 0:
            raise ValueError("M must be divisible by tile_m")
        if self.n % self.tile_n != 0:
            raise ValueError("N must be divisible by tile_n")
        if self.k % self.tile_k != 0:
            raise ValueError("K must be divisible by tile_k")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.ops_per_mac <= 0:
            raise ValueError("ops per MAC must be positive")
        if self.pipeline_k_tiles and not self.double_buffer_inputs:
            raise ValueError("pipeline_k_tiles requires double_buffer_inputs")

    @property
    def pe_rows(self) -> int:
        return self.m // self.tile_m

    @property
    def pe_cols(self) -> int:
        return self.n // self.tile_n

    @property
    def k_tiles(self) -> int:
        return self.k // self.tile_k

    @property
    def a_tile_bytes(self) -> int:
        return self.tile_m * self.tile_k * self.bytes_per_element

    @property
    def b_tile_bytes(self) -> int:
        return self.tile_k * self.tile_n * self.bytes_per_element

    @property
    def c_tile_bytes(self) -> int:
        return self.tile_m * self.tile_n * self.bytes_per_element

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        input_buffers = 2 if self.double_buffer_inputs else 1
        return self.c_tile_bytes + input_buffers * (
            self.a_tile_bytes + self.b_tile_bytes
        )

    @property
    def ops_per_pe_per_k_tile(self) -> int:
        macs = self.tile_m * self.tile_n * self.tile_k
        return macs * self.ops_per_mac

    @property
    def ops_per_pe_total(self) -> int:
        return self.ops_per_pe_per_k_tile * self.k_tiles

    @property
    def total_ops(self) -> int:
        return self.m * self.n * self.k * self.ops_per_mac


@dataclass(frozen=True, slots=True)
class CslDemoWorkloadConfig:
    type: str = "csl_demo"
    active_rows: int = 4
    active_cols: int = 4
    elements_per_pe: int = 256
    bytes_per_element: int = 4
    packet_payload_bytes: int = 256
    fifo_capacity: int = 4
    wavelet_levels: int = 3
    sparse_density: float = 0.25
    collective_payload_bytes: int = 256
    microthreads_per_pe: int = 4
    output_queue_reuse_rounds: int = 2
    color_swap_payload_bytes: int = 128
    sentinel_bytes: int = 8

    def validate(self) -> None:
        if self.active_rows <= 0 or self.active_cols <= 0:
            raise ValueError("active rows and cols must be positive")
        if self.elements_per_pe <= 0:
            raise ValueError("elements per PE must be positive")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.fifo_capacity <= 0:
            raise ValueError("FIFO capacity must be positive")
        if self.wavelet_levels <= 0:
            raise ValueError("wavelet levels must be positive")
        if not 0.0 <= self.sparse_density <= 1.0:
            raise ValueError("sparse density must be between 0 and 1")
        if self.collective_payload_bytes <= 0:
            raise ValueError("collective payload bytes must be positive")
        if self.microthreads_per_pe <= 0:
            raise ValueError("microthreads per PE must be positive")
        if self.output_queue_reuse_rounds <= 0:
            raise ValueError("output queue reuse rounds must be positive")
        if self.color_swap_payload_bytes <= 0:
            raise ValueError("color swap payload bytes must be positive")
        if self.sentinel_bytes <= 0:
            raise ValueError("sentinel bytes must be positive")

    @property
    def active_pes(self) -> int:
        return self.active_rows * self.active_cols

    @property
    def local_array_bytes_per_pe(self) -> int:
        return self.elements_per_pe * self.bytes_per_element

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        fifo_bytes = self.fifo_capacity * self.packet_payload_bytes
        scratch_bytes = self.local_array_bytes_per_pe
        return self.local_array_bytes_per_pe + fifo_bytes + scratch_bytes

    @property
    def wavelet_ops_per_pe(self) -> int:
        return self.elements_per_pe * self.wavelet_levels * 4

    @property
    def sparse_ops_per_pe(self) -> int:
        return int(self.elements_per_pe * self.sparse_density) * 2

    @property
    def library_ops_per_pe(self) -> int:
        return self.elements_per_pe * 8

    @property
    def map_ops_per_pe(self) -> int:
        return self.elements_per_pe


@dataclass(frozen=True, slots=True)
class SdkExamplesWorkloadConfig:
    type: str = "sdk_examples"
    active_rows: int = 4
    active_cols: int = 4
    bytes_per_element: int = 4
    packet_payload_bytes: int = 256
    fifo_capacity: int = 8
    stream_bytes: int = 4096
    tile_rows: int = 16
    tile_cols: int = 16
    vector_length: int = 256
    matrix_rows: int = 256
    matrix_cols: int = 256
    gemm_m: int = 128
    gemm_n: int = 128
    gemm_k: int = 128
    iterations: int = 4
    stencil_halo_bytes_per_direction: int = 128
    spmv_nonzeros_per_row: int = 7
    fft_points: int = 256

    def validate(self) -> None:
        if self.active_rows <= 0 or self.active_cols <= 0:
            raise ValueError("active rows and cols must be positive")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes per element must be positive")
        if self.packet_payload_bytes <= 0:
            raise ValueError("packet payload bytes must be positive")
        if self.fifo_capacity <= 0:
            raise ValueError("FIFO capacity must be positive")
        if self.stream_bytes <= 0:
            raise ValueError("stream bytes must be positive")
        if self.tile_rows <= 0 or self.tile_cols <= 0:
            raise ValueError("tile dimensions must be positive")
        if self.vector_length <= 0:
            raise ValueError("vector length must be positive")
        if self.matrix_rows <= 0 or self.matrix_cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if self.gemm_m <= 0 or self.gemm_n <= 0 or self.gemm_k <= 0:
            raise ValueError("GEMM dimensions must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.stencil_halo_bytes_per_direction < 0:
            raise ValueError("stencil halo bytes cannot be negative")
        if self.spmv_nonzeros_per_row <= 0:
            raise ValueError("SpMV nonzeros per row must be positive")
        if self.fft_points <= 0:
            raise ValueError("FFT points must be positive")

    @property
    def active_pes(self) -> int:
        return self.active_rows * self.active_cols

    @property
    def tile_bytes_per_pe(self) -> int:
        return self.tile_rows * self.tile_cols * self.bytes_per_element

    @property
    def vector_bytes(self) -> int:
        return self.vector_length * self.bytes_per_element

    @property
    def memory_footprint_bytes_per_pe(self) -> int:
        fifo_bytes = self.fifo_capacity * self.packet_payload_bytes
        scratch_bytes = max(self.tile_bytes_per_pe, self.vector_bytes)
        return scratch_bytes + fifo_bytes + self.packet_payload_bytes

    @property
    def gemv_ops(self) -> int:
        return 2 * self.matrix_rows * self.matrix_cols

    @property
    def gemm_ops(self) -> int:
        return 2 * self.gemm_m * self.gemm_n * self.gemm_k

    @property
    def spmv_ops_per_iteration(self) -> int:
        return 2 * self.matrix_rows * self.spmv_nonzeros_per_row

    @property
    def stencil_ops_per_pe(self) -> int:
        return self.tile_rows * self.tile_cols * 25

    @property
    def game_of_life_ops_per_pe(self) -> int:
        return self.tile_rows * self.tile_cols * 8


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    mesh: MeshConfig = MeshConfig()
    workload: (
        StencilWorkloadConfig
        | MemoryCopyWorkloadConfig
        | GemmWorkloadConfig
        | CslDemoWorkloadConfig
        | SdkExamplesWorkloadConfig
    ) = StencilWorkloadConfig()

    def validate(self) -> None:
        self.mesh.validate()
        self.workload.validate()
