"""MLIR/IREE and MLIR-AIR adapters used by the OpenCake GEMM pipeline."""

from wafersim.opencake.frontend.air import AirDialectRunner, AirTransformResult
from wafersim.opencake.frontend.iree_compiler import IreeCompileResult, IreeCompilerRunner
from wafersim.opencake.frontend.mlir import MlirMatmulOp, read_linalg_matmul
from wafersim.opencake.frontend.tiled_mlir import TiledGemmLoopNest, read_tiled_gemm_mlir
from wafersim.opencake.frontend.transform import MlirTransformResult, MlirTransformRunner

__all__ = [
    "AirDialectRunner",
    "AirTransformResult",
    "IreeCompileResult",
    "IreeCompilerRunner",
    "MlirMatmulOp",
    "MlirTransformResult",
    "MlirTransformRunner",
    "TiledGemmLoopNest",
    "read_linalg_matmul",
    "read_tiled_gemm_mlir",
]
