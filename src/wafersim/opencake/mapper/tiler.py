from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from wafersim.opencake.workload import GemmWorkload, HardwareSpec


@dataclass(frozen=True, slots=True)
class GemmTilePlan:
    """SRAM-feasible GEMM tile choice for one rank/PE.

    The first OpenCake policy maps one output C tile to one rank. M and N tiles
    therefore follow the rank grid. The auto choice is the K streaming chunk,
    constrained by PE-local SRAM capacity.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    ab_buffer_count: int
    a_buffer_bytes: int
    b_buffer_bytes: int
    c_tile_bytes: int
    sram_bytes_per_rank: int
    sram_utilization: float

    @property
    def streaming_dimension(self) -> str:
        return "K"


class SramAwareTiler:
    """Choose a GEMM K tile that fits PE-local SRAM.

    This is deliberately simple and deterministic: use the full rank grid for
    M/N and choose the largest feasible K chunk. Larger K chunks reduce stream
    count, while the SRAM constraint keeps A/B double buffers resident.
    """

    def choose_gemm_tile(self, workload: GemmWorkload, hardware: HardwareSpec) -> GemmTilePlan:
        candidates = self.feasible_gemm_tiles(workload, hardware)
        if not candidates:
            raise ValueError(
                "no feasible GEMM K tile fits PE-local SRAM; reduce tile_m/tile_n "
                "or increase sram_bytes_per_rank"
            )
        return max(candidates, key=lambda plan: plan.tile_k)

    @staticmethod
    def output_tile_shape(workload: GemmWorkload, hardware: HardwareSpec) -> tuple[int, int]:
        return ceil(workload.m / hardware.rank_rows), ceil(workload.n / hardware.rank_cols)

    def plan_for_tile(
        self,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        *,
        tile_m: int,
        tile_n: int,
        tile_k: int,
    ) -> GemmTilePlan:
        workload.validate()
        hardware.validate()
        if min(tile_m, tile_n, tile_k) <= 0:
            raise ValueError("GEMM tile dimensions must be positive")
        return self._build_plan(
            workload=workload,
            hardware=hardware,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            ab_buffer_count=2 if hardware.double_buffer_ab else 1,
        )

    def feasible_gemm_tiles(
        self,
        workload: GemmWorkload,
        hardware: HardwareSpec,
    ) -> tuple[GemmTilePlan, ...]:
        """Enumerate every SRAM-feasible K tile for the physical rank grid.

        This method only defines the candidate space. A strategy optimizer can
        rank these candidates using full-mesh communication and scheduling
        costs before MLIR Transform materializes the selected tile.
        """

        workload.validate()
        hardware.validate()
        tile_m, tile_n = self.output_tile_shape(workload, hardware)
        ab_buffer_count = 2 if hardware.double_buffer_ab else 1
        max_k = min(workload.k, hardware.gemm_k_tile)
        return tuple(
            plan
            for tile_k in self._k_candidates(max_k)
            if (
                plan := self._build_plan(
                    workload=workload,
                    hardware=hardware,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    ab_buffer_count=ab_buffer_count,
                )
            ).sram_bytes_per_rank
            <= hardware.sram_bytes_per_rank
        )

    @staticmethod
    def _k_candidates(max_k: int) -> tuple[int, ...]:
        return tuple(range(1, max_k + 1))

    @staticmethod
    def _build_plan(
        *,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        ab_buffer_count: int,
    ) -> GemmTilePlan:
        a_buffer_bytes = tile_m * tile_k * workload.bytes_per_element
        b_buffer_bytes = tile_k * tile_n * workload.bytes_per_element
        c_tile_bytes = tile_m * tile_n * workload.bytes_per_element
        sram_bytes = c_tile_bytes + ab_buffer_count * (a_buffer_bytes + b_buffer_bytes)
        return GemmTilePlan(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            ab_buffer_count=ab_buffer_count,
            a_buffer_bytes=a_buffer_bytes,
            b_buffer_bytes=b_buffer_bytes,
            c_tile_bytes=c_tile_bytes,
            sram_bytes_per_rank=sram_bytes,
            sram_utilization=sram_bytes / hardware.sram_bytes_per_rank,
        )
