from __future__ import annotations

from dataclasses import dataclass

from wafersim.opencake.frontend.air import AirGemmHierarchy
from wafersim.opencake.frontend.tiled_mlir import TiledGemmLoopNest
from wafersim.opencake.strategy import GemmStreamingStrategyContract
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.mapper.core import OpenCakeMapper
from wafersim.opencake.mapper.streamer import GemmStreamingPlan
from wafersim.opencake.mapper.tiler import GemmTilePlan
from wafersim.opencake.workload import GemmWorkload, HardwareSpec


@dataclass(frozen=True, slots=True)
class MlirToSpatialLoweringResult:
    """Wafer mapping inferred from one transformed MLIR GEMM loop nest."""

    schema: str
    loop_nest: TiledGemmLoopNest
    tile_plan: GemmTilePlan
    streaming_plan: GemmStreamingPlan
    program: OpenCakeProgram

    def summary(self) -> dict[str, object]:
        uses_air = self.loop_nest.source == "air_operation_tree"
        return {
            "schema": self.schema,
            "source": self.loop_nest.source,
            "consumes_transformed_mlir": self.loop_nest.source == "mlir_operation_tree",
            "consumes_official_mlir_air": self.loop_nest.source == "air_operation_tree",
            "loop_nest": self.loop_nest.summary(),
            "inferred_mapping": {
                "parallel_iterations": (
                    "air.herd -> active rank/PE grid"
                    if uses_air
                    else "scf.forall -> active rank/PE grid"
                ),
                "reduction_iterations": "AIR-nested scf.for -> K streaming chunks"
                if uses_air
                else "scf.for -> K streaming chunks",
                "a_tile": (
                    "linalg.matmul operand 0 -> "
                    f"{self.streaming_plan.a_direction}-streamed A buffer"
                ),
                "b_tile": (
                    "linalg.matmul operand 1 -> "
                    f"{self.streaming_plan.b_direction}-streamed B buffer"
                ),
                "c_tile": "linalg.matmul output -> PE-local stationary C buffer",
                "a_color": self.streaming_plan.a_color,
                "a_direction": self.streaming_plan.a_direction,
                "b_color": self.streaming_plan.b_color,
                "b_direction": self.streaming_plan.b_direction,
            },
            "streaming_strategy": {
                "source": self.streaming_plan.strategy_source,
                "schema": self.streaming_plan.strategy_schema,
                "target_mesh": list(self.streaming_plan.strategy_target_mesh)
                if self.streaming_plan.strategy_target_mesh is not None
                else None,
                "objective_cycles": self.streaming_plan.strategy_objective_cycles,
                "optimality_gap": self.streaming_plan.strategy_optimality_gap,
            },
        }


class MlirToSpatialLowering:
    """Lower standard tiled MLIR structure into OpenCake Spatial IR."""

    def __init__(self, mapper: OpenCakeMapper | None = None) -> None:
        self._mapper = mapper or OpenCakeMapper()

    def lower_gemm(
        self,
        loop_nest: TiledGemmLoopNest,
        workload: GemmWorkload,
        hardware: HardwareSpec,
        *,
        air_hierarchy: AirGemmHierarchy | None = None,
        streaming_strategy: GemmStreamingStrategyContract | None = None,
    ) -> MlirToSpatialLoweringResult:
        tile_plan = loop_nest.to_tile_plan(workload, hardware)
        if streaming_strategy is not None:
            streaming_strategy.validate_target_mesh(loop_nest.grid_rows, loop_nest.grid_cols)
        channels = {
            channel.tensor: channel for channel in air_hierarchy.channels
        } if air_hierarchy is not None else {}
        a_channel = channels.get("A")
        b_channel = channels.get("B")
        if air_hierarchy is not None and (a_channel is None or b_channel is None):
            raise ValueError("AIR-to-OpenCake lowering requires explicit A and B AIR channels")
        if streaming_strategy is not None and a_channel is not None and b_channel is not None:
            actual = (a_channel.direction, b_channel.direction)
            expected = (streaming_strategy.a_direction, streaming_strategy.b_direction)
            if actual != expected:
                raise ValueError(
                    f"official AIR channels {actual} disagree with optimized strategy {expected}"
                )
        streaming_plan = GemmStreamingPlan(
            dimension="K",
            k_tile=loop_nest.k_step,
            k_starts=loop_nest.k_starts,
            ab_buffer_count=tile_plan.ab_buffer_count,
            a_color=streaming_strategy.a_color if streaming_strategy is not None else (
                a_channel.color if a_channel is not None else "a_east"
            ),
            b_color=streaming_strategy.b_color if streaming_strategy is not None else (
                b_channel.color if b_channel is not None else "b_south"
            ),
            a_direction=streaming_strategy.a_direction if streaming_strategy is not None else (
                a_channel.direction if a_channel is not None else "east"
            ),
            b_direction=streaming_strategy.b_direction if streaming_strategy is not None else (
                b_channel.direction if b_channel is not None else "south"
            ),
            strategy_source=streaming_strategy.source if streaming_strategy is not None else (
                "official_mlir_air" if air_hierarchy is not None else "builtin_wafer_policy"
            ),
            strategy_schema=streaming_strategy.schema if streaming_strategy is not None else (
                "opencake.official_air_streaming_strategy.v1"
                if air_hierarchy is not None
                else "opencake.builtin_streaming_strategy.v1"
            ),
            strategy_target_mesh=(
                streaming_strategy.target_mesh if streaming_strategy is not None else None
            ),
            strategy_objective_cycles=(
                streaming_strategy.objective_cycles
                if streaming_strategy is not None
                else None
            ),
            strategy_optimality_gap=(
                streaming_strategy.optimality_gap
                if streaming_strategy is not None
                else None
            ),
        )
        program = self._mapper.map_gemm(
            workload,
            hardware,
            tile_plan=tile_plan,
            stream_plan=streaming_plan,
            rank_grid=(loop_nest.grid_rows, loop_nest.grid_cols),
            mapping_source=(
                (
                    "stream_allocation_ir_to_air_channels"
                    if streaming_strategy.schema
                    == "opencake.gemm_stream_allocation_contract.v1"
                    else "full_mesh_strategy_to_air_channels"
                )
                if streaming_strategy is not None
                else (
                    "explicit_air_channels_to_opencake"
                    if air_hierarchy is not None
                    else "tiled_mlir"
                )
            ),
        )
        return MlirToSpatialLoweringResult(
            schema="opencake.air_channel_to_spatial_lowering.v2"
            if air_hierarchy is not None
            else "opencake.mlir_to_spatial_lowering.v1",
            loop_nest=loop_nest,
            tile_plan=tile_plan,
            streaming_plan=streaming_plan,
            program=program,
        )
