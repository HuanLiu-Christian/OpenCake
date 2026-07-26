from __future__ import annotations

from dataclasses import dataclass

from wafersim.opencake.mapper.tiler import GemmTilePlan
from wafersim.opencake.strategy import GemmStreamingStrategyContract
from wafersim.opencake.workload import GemmWorkload


@dataclass(frozen=True, slots=True)
class GemmStreamingPlan:
    """K-streaming schedule used after SRAM-aware tiling."""

    dimension: str
    k_tile: int
    k_starts: tuple[int, ...]
    ab_buffer_count: int
    a_color: str = "a_east"
    b_color: str = "b_south"
    a_direction: str = "east"
    b_direction: str = "south"
    strategy_source: str = "builtin_wafer_policy"
    strategy_schema: str = "opencake.builtin_streaming_strategy.v1"
    strategy_target_mesh: tuple[int, int] | None = None
    strategy_objective_cycles: float | None = None
    strategy_optimality_gap: float | None = None

    @property
    def chunks(self) -> int:
        return len(self.k_starts)


class StreamingPlanner:
    """Create a deterministic streaming schedule from the tile plan."""

    def plan_gemm_streaming(
        self,
        workload: GemmWorkload,
        tile_plan: GemmTilePlan,
        strategy: GemmStreamingStrategyContract | None = None,
    ) -> GemmStreamingPlan:
        if strategy is not None:
            return GemmStreamingPlan(
                dimension=tile_plan.streaming_dimension,
                k_tile=tile_plan.tile_k,
                k_starts=tuple(range(0, workload.k, tile_plan.tile_k)),
                ab_buffer_count=tile_plan.ab_buffer_count,
                a_color=strategy.a_color,
                b_color=strategy.b_color,
                a_direction=strategy.a_direction,
                b_direction=strategy.b_direction,
                strategy_source=strategy.source,
                strategy_schema=strategy.schema,
                strategy_target_mesh=strategy.target_mesh,
                strategy_objective_cycles=strategy.objective_cycles,
                strategy_optimality_gap=strategy.optimality_gap,
            )
        return GemmStreamingPlan(
            dimension=tile_plan.streaming_dimension,
            k_tile=tile_plan.tile_k,
            k_starts=tuple(range(0, workload.k, tile_plan.tile_k)),
            ab_buffer_count=tile_plan.ab_buffer_count,
        )
