from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wafersim.opencake.frontend.air import AirDialectRunner
from wafersim.opencake.frontend.tiled_mlir import TiledGemmLoopNest, read_tiled_gemm_mlir
from wafersim.opencake.strategy import GemmStreamingStrategyContract
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.stages import OpenCakeStageCompiler, StagedGemmCompilation
from wafersim.opencake.workload import HardwareSpec


@dataclass(frozen=True, slots=True)
class OpenCakeTiledMlirPipelineResult:
    tiled_input: TiledGemmLoopNest
    spatial_ir_ops: int
    program: OpenCakeProgram
    stages: StagedGemmCompilation

    def summary(self) -> dict[str, object]:
        return {
            "frontend": {
                "type": "mlir.pre_tiled_gemm",
                "source": "existing_tiled_mlir_operation_tree",
                "loop_nest": self.tiled_input.summary(),
                "tile_search_rerun": False,
                "transform_dialect_rerun": False,
            },
            "workload": self.program.workload,
            "hardware": self.program.hardware,
            "spatial_ir_ops": self.spatial_ir_ops,
            "stages": self.stages.summary(),
        }


def run_tiled_mlir_to_wafer_sim(
    *,
    tiled_mlir_file: str | Path,
    hardware: HardwareSpec,
    spatial_ir_out: str | Path | None = None,
    air_opt: str | Path | None = None,
    air_docker_container: str | None = None,
    streaming_strategy: GemmStreamingStrategyContract | None = None,
) -> OpenCakeTiledMlirPipelineResult:
    """Lower an existing tiled GEMM MLIR operation tree to Spatial IR."""

    tiled_input = read_tiled_gemm_mlir(tiled_mlir_file)
    stages = OpenCakeStageCompiler(
        air_runner=AirDialectRunner(
            air_opt,
            docker_container=air_docker_container,
        )
    ).compile_tiled_gemm(
        tiled_mlir_file,
        hardware,
        streaming_strategy=streaming_strategy,
    )
    program = stages.program
    if spatial_ir_out is not None:
        program.write_jsonl(spatial_ir_out)
    return OpenCakeTiledMlirPipelineResult(
        tiled_input=tiled_input,
        spatial_ir_ops=len(program.ops),
        program=program,
        stages=stages,
    )
