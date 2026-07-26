from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafersim.opencake.astrasim import AstraSimExperimentExporter
from wafersim.opencake.backend import OpenCakeBackend
from wafersim.opencake.chisel import ChiselTraceExporter
from wafersim.opencake.cpp_backend import OpenCakeCppBackendRunner
from wafersim.opencake.frontend.iree_compiler import IreeCompilerRunner
from wafersim.opencake.frontend.mlir import (
    canonical_linalg_matmul_module,
    read_linalg_matmul,
)
from wafersim.opencake.frontend.tiled_mlir import parse_tiled_gemm_mlir
from wafersim.opencake.frontend.transform import MlirTransformRunner
from wafersim.opencake.pipeline import run_tiled_mlir_to_wafer_sim
from wafersim.opencake.schema import validate_spatial_program
from wafersim.opencake.stream_strategy import StreamStrategyRunner
from wafersim.opencake.workload import HardwareSpec
from wafersim.opencake.zigzag import ZigZagGemmMapper


def main() -> None:
    args = _parse_args()
    if args.cpp_backend_dir is not None and args.out_dir is None:
        raise SystemExit("--cpp-backend-dir requires --out-dir because the C++ backend reads the exported package")
    hardware = HardwareSpec(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        pe_block_rows=args.pe_block_rows,
        pe_block_cols=args.pe_block_cols,
        compute_ops_per_cycle_per_rank=args.compute_ops_per_cycle_per_rank,
        sram_bandwidth_bytes_per_cycle_per_rank=args.sram_bandwidth_bytes_per_cycle_per_rank,
        link_bandwidth_bytes_per_cycle=args.link_bandwidth_bytes_per_cycle,
        router_latency_cycles=args.router_latency_cycles,
        queue_capacity=args.queue_capacity,
        sram_bytes_per_rank=args.sram_bytes_per_rank,
        gemm_k_tile=args.gemm_k_tile,
        double_buffer_ab=not args.no_double_buffer,
    )

    mlir_input = args.mlir
    transform_mlir_input = args.mlir
    iree_compile = None
    if args.use_real_iree:
        iree_compile = IreeCompilerRunner().compile_workload(
            args.mlir,
            out_dir=args.iree_out_dir,
            phases=("input", "preprocessing", "flow", "stream"),
            emit_vmfb=args.emit_vmfb,
        )
        mlir_input = iree_compile.artifact(args.iree_phase).output_path
        recovered = read_linalg_matmul(mlir_input)
        transform_mlir_input = args.iree_out_dir / "matmul.opencake_canonical.mlir"
        transform_mlir_input.write_text(
            canonical_linalg_matmul_module(recovered),
            encoding="utf-8",
        )

    workload = read_linalg_matmul(mlir_input).to_workload()
    zigzag_result = ZigZagGemmMapper(
        optimization=args.zigzag_opt,
        temporal_engine=args.zigzag_temporal_engine,
        lpf_limit=args.zigzag_lpf_limit,
    ).search(
        workload,
        hardware,
        output_dir=args.zigzag_out_dir,
    )
    tile_plan = zigzag_result.tile_plan

    transform_result = MlirTransformRunner().tile_gemm(transform_mlir_input, tile_plan)
    if not transform_result.applied:
        raise SystemExit(transform_result.stderr)
    stream_input_dir = args.stream_strategy_out_dir / "materialized_input"
    stream_input_dir.mkdir(parents=True, exist_ok=True)
    tiled_mlir_path = stream_input_dir / "tiled_gemm.mlir"
    tiled_mlir_path.write_text(transform_result.tiled_payload_mlir, encoding="utf-8")
    loop_nest = parse_tiled_gemm_mlir(transform_result.tiled_payload_mlir)

    if not args.stream_python.exists() or not args.stream_root.exists():
        raise SystemExit(
            "The formal path requires an official Stream environment. Provide "
            "--stream-python and --stream-root."
        )
    route_families = tuple(
        family.strip() for family in args.stream_route_families.split(",") if family.strip()
    )
    streaming_strategy, stream_package, stream_results = StreamStrategyRunner(
        python=args.stream_python,
        stream_root=args.stream_root,
    ).run_tiled_gemm(
        loop_nest=loop_nest,
        hardware=hardware,
        output_dir=args.stream_strategy_out_dir / "tetra",
        route_families=route_families,
    )
    streaming_strategy.validate_target_mesh(loop_nest.grid_rows, loop_nest.grid_cols)

    result = run_tiled_mlir_to_wafer_sim(
        tiled_mlir_file=tiled_mlir_path,
        hardware=hardware,
        spatial_ir_out=args.spatial_ir_out,
        air_opt=args.air_opt,
        air_docker_container=args.air_docker_container,
        streaming_strategy=streaming_strategy,
    )
    validation = validate_spatial_program(result.program)
    backend_result = OpenCakeBackend().run(result.program, hardware)
    summary: dict[str, object] = result.summary()
    summary["validation"] = validation.summary()
    summary["performance"] = backend_result.summary()
    summary["mlir_transform"] = transform_result.summary()
    summary["stream_input_package"] = stream_package.summary()
    summary["stream_tetra_searches"] = [item.summary() for item in stream_results]
    summary["streaming_strategy"] = streaming_strategy.summary()
    summary["zigzag_tile_search"] = zigzag_result.summary()
    if iree_compile is not None:
        summary["iree_compile"] = iree_compile.summary()
        summary["iree_workload_phase"] = args.iree_phase
        summary["iree_transform_input_phase"] = "semantic_recovery_normalization"
        summary["iree_transform_input"] = str(transform_mlir_input)

    if args.stage_out_dir is not None:
        stage_artifacts = result.stages.export(args.stage_out_dir)
        summary["stage_artifacts"] = stage_artifacts.summary()

    if args.out_dir is not None:
        package = AstraSimExperimentExporter().export(
            program=result.program,
            hardware=hardware,
            validation=validation,
            reference_result=backend_result,
            out_dir=args.out_dir,
            execution_trace=result.stages.chakra_trace,
        )
        summary["astrasim_package"] = package.summary()

        if args.cpp_backend_dir is not None:
            cpp_result = OpenCakeCppBackendRunner(args.cpp_backend_dir).run(
                package_root=package.root,
                check_reference=not args.no_cpp_reference_check,
            )
            summary["cpp_backend"] = cpp_result.as_dict()

    if args.chisel_out_dir is not None:
        package = ChiselTraceExporter().export(
            program=result.program,
            hardware=hardware,
            validation=validation,
            out_dir=args.chisel_out_dir,
        )
        summary["chisel_package"] = package.summary()

    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLIR linalg.matmul through OpenCake.")
    parser.add_argument(
        "--mlir",
        type=Path,
        default=Path(__file__).with_name("matmul.mlir"),
        help="MLIR file containing one linalg.matmul op",
    )
    parser.add_argument("--spatial-ir-out", type=Path, default=None)
    parser.add_argument(
        "--use-real-iree",
        action="store_true",
        help="Run iree-compile and recover GEMM semantics from the selected IREE phase",
    )
    parser.add_argument(
        "--iree-out-dir",
        type=Path,
        default=Path("/tmp/opencake_gemm_1024_iree"),
        help="Directory for real IREE phase artifacts",
    )
    parser.add_argument(
        "--iree-phase",
        choices=("input", "preprocessing", "flow", "stream"),
        default="stream",
        help="IREE compiler phase consumed by OpenCake semantic recovery",
    )
    parser.add_argument("--emit-vmfb", action="store_true", help="Also emit a VMVX vmfb artifact")
    parser.add_argument(
        "--stage-out-dir",
        type=Path,
        default=None,
        help="Write explicit OpenCake compiler artifacts, including transformed MLIR",
    )
    parser.add_argument(
        "--air-opt",
        type=Path,
        default=None,
        help="Path to the official MLIR-AIR air-opt binary",
    )
    parser.add_argument(
        "--air-docker-container",
        default=None,
        help="Running Linux container that provides air-opt (useful on macOS)",
    )
    parser.add_argument(
        "--zigzag-out-dir",
        type=Path,
        default=Path("/tmp/opencake_zigzag_search"),
        help="Directory for ZigZag input descriptions and cost-model output",
    )
    parser.add_argument(
        "--zigzag-opt",
        choices=("latency", "energy", "EDP"),
        default="latency",
    )
    parser.add_argument(
        "--zigzag-temporal-engine",
        choices=("loma", "salsa"),
        default="loma",
    )
    parser.add_argument("--zigzag-lpf-limit", type=int, default=8)
    parser.add_argument(
        "--stream-python",
        type=Path,
        default=Path("/tmp/opencake-stream-toolchain/venv/bin/python"),
        help="Python executable from the Stream/ZigZag environment",
    )
    parser.add_argument(
        "--stream-root",
        type=Path,
        default=Path("/tmp/opencake-stream-toolchain/stream"),
        help="Stream checkout that provides its solver abstraction",
    )
    parser.add_argument(
        "--stream-strategy-out-dir",
        type=Path,
        default=Path("/tmp/opencake_gemm_stream_strategy"),
        help="Directory for exact-mesh Stream/TETRA inputs, AllocationIR, and selected strategy",
    )
    parser.add_argument(
        "--stream-route-families",
        default="a-east-b-south,a-east-b-north,a-west-b-south,a-west-b-north",
        help=(
            "Comma-separated legal wafer route families. Each family is solved by the "
            "official Stream/TETRA pipeline; the minimum Stream latency is selected."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Export Astra-Sim/OpenCake package")
    parser.add_argument("--chisel-out-dir", type=Path, default=None, help="Export Chisel replay package")
    parser.add_argument(
        "--cpp-backend-dir",
        type=Path,
        default=None,
        help="Build and run the formal C++ OpenCake wafer backend on --out-dir",
    )
    parser.add_argument(
        "--no-cpp-reference-check",
        action="store_true",
        help="Run the C++ backend without checking reference_result.json",
    )
    parser.add_argument("--mesh-rows", type=int, default=32)
    parser.add_argument("--mesh-cols", type=int, default=32)
    parser.add_argument("--pe-block-rows", type=int, default=1)
    parser.add_argument("--pe-block-cols", type=int, default=1)
    parser.add_argument("--compute-ops-per-cycle-per-rank", type=float, default=64.0)
    parser.add_argument("--sram-bandwidth-bytes-per-cycle-per-rank", type=float, default=16.0)
    parser.add_argument("--link-bandwidth-bytes-per-cycle", type=float, default=128.0)
    parser.add_argument("--router-latency-cycles", type=float, default=1.0)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--sram-bytes-per-rank", type=int, default=48 * 1024)
    parser.add_argument("--gemm-k-tile", type=int, default=64)
    parser.add_argument("--no-double-buffer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
