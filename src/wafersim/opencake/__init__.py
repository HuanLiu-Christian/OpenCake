"""Public API for the current OpenCake GEMM compiler pipeline."""

from wafersim.opencake.backend import OpenCakeBackend, OpenCakeBackendResult
from wafersim.opencake.chisel import ChiselExportPackage, ChiselTraceExporter
from wafersim.opencake.cpp_backend import CppBackendResult, OpenCakeCppBackendRunner
from wafersim.opencake.ir import OpenCakeProgram
from wafersim.opencake.pipeline import (
    OpenCakeTiledMlirPipelineResult,
    run_tiled_mlir_to_wafer_sim,
)
from wafersim.opencake.schema import validate_spatial_program
from wafersim.opencake.stream_strategy import StreamStrategyRunner
from wafersim.opencake.workload import GemmWorkload, HardwareSpec
from wafersim.opencake.zigzag import ZigZagGemmMapper, ZigZagGemmSearchResult

__all__ = [
    "ChiselExportPackage",
    "ChiselTraceExporter",
    "CppBackendResult",
    "GemmWorkload",
    "HardwareSpec",
    "OpenCakeBackend",
    "OpenCakeBackendResult",
    "OpenCakeCppBackendRunner",
    "OpenCakeProgram",
    "OpenCakeTiledMlirPipelineResult",
    "StreamStrategyRunner",
    "ZigZagGemmMapper",
    "ZigZagGemmSearchResult",
    "run_tiled_mlir_to_wafer_sim",
    "validate_spatial_program",
]
