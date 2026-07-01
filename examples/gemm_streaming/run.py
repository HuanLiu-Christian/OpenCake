from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from layout import LAYOUT
from pe_program import PE_PROGRAM, GemmStreamingConfig
from wafersim.hardware.gemm import run_gemm_commands
from wafersim.mapping import GemmProgram, write_gemm_commands
from wafersim.program import GemmProblemSpec, LayoutSpec
from wafersim.workloads.kernels import gemm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Python-native GEMM streaming SimPy example."
    )
    parser.add_argument("--mesh-rows", type=int)
    parser.add_argument("--mesh-cols", type=int)
    parser.add_argument("--kernel-rows", type=int)
    parser.add_argument("--kernel-cols", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--bytes-per-element", type=int)
    parser.add_argument("--packet-payload-bytes", type=int)
    parser.add_argument("--ops-per-mac", type=int)
    parser.add_argument(
        "--ir-out",
        type=Path,
        help="Optional path for the generated multi-PE GEMM JSONL IR.",
    )
    args = parser.parse_args()

    layout = apply_layout_overrides(LAYOUT, args)
    problem = GemmProblemSpec(
        m=args.m or 4,
        n=args.n or 4,
        k=args.k or 4,
        bytes_per_element=args.bytes_per_element or 4,
        packet_payload_bytes=args.packet_payload_bytes or 64,
        ops_per_mac=args.ops_per_mac or 2,
    )
    summary = run_gemm_streaming(layout=layout, problem=problem)
    if args.ir_out is not None:
        write_gemm_commands(args.ir_out, summary["commands"])
        summary["generated_ir"] = str(args.ir_out)
    del summary["commands"]
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_gemm_streaming(
    *,
    layout: LayoutSpec,
    problem: GemmProblemSpec,
) -> dict[str, object]:
    config = GemmStreamingConfig(
        m=problem.m,
        n=problem.n,
        k=problem.k,
        kernel_rows=layout.kernel_rows,
        kernel_cols=layout.kernel_cols,
        bytes_per_element=problem.bytes_per_element,
        packet_payload_bytes=problem.packet_payload_bytes,
        ops_per_mac=problem.ops_per_mac,
    )
    config.validate()
    left = [index for index in range(config.m * config.k)]
    right = [1 for _ in range(config.k * config.n)]
    program = build_gemm_program(config, left, right)
    golden = run_gemm_commands(
        program.commands,
        ops_per_mac=config.ops_per_mac,
        kernel_cols=config.kernel_cols,
    )
    return {
        "type": "gemm_streaming_example",
        "m": config.m,
        "n": config.n,
        "k": config.k,
        "kernel_rows": config.kernel_rows,
        "kernel_cols": config.kernel_cols,
        "rows_per_pe": config.rows_per_pe,
        "cols_per_pe": config.cols_per_pe,
        "bytes_per_element": config.bytes_per_element,
        "a_panel_bytes_per_row": config.a_panel_bytes_per_row,
        "b_panel_bytes_per_col": config.b_panel_bytes_per_col,
        "c_tile_bytes_per_pe": config.c_tile_bytes_per_pe,
        "memory_footprint_bytes_per_pe": config.memory_footprint_bytes_per_pe,
        "compute_ops": config.compute_ops,
        "compute_ops_per_pe": config.compute_ops_per_pe,
        "sim_time_cycles": golden.sim_time_cycles,
        "compute_tasks_completed": golden.compute_tasks_completed,
        "memory_tasks_completed": golden.memory_tasks_completed,
        "result_c_tiles": {
            str(coord): list(tile)
            for coord, tile in sorted(golden.c_tiles.items())
        },
        "expected_c_tiles": {
            str(coord): list(tile)
            for coord, tile in expected_tiles(config, left, right).items()
        },
        "verified": golden.result_c == golden.expected_c,
        "program": {
            "layout": "examples/gemm_streaming/layout.py",
            "pe_program": "examples/gemm_streaming/pe_program.py",
            "kernel": PE_PROGRAM.kernel,
            "rectangle": [layout.kernel_rows, layout.kernel_cols],
            "colors": {
                color: direction.value if direction is not None else "ramp"
                for color, direction in layout.routes.items()
            },
            "tasks": [task.name for task in PE_PROGRAM.tasks],
        },
        "mapping": {
            "ir_type": "gemm_jsonl",
            "chisel_target": "GemmArrayTop",
        },
        "commands": program.commands,
    }


def build_gemm_program(
    config: GemmStreamingConfig,
    left: list[int],
    right: list[int],
) -> GemmProgram:
    config.validate()
    if len(left) != config.m * config.k:
        raise ValueError("left length does not match M x K")
    if len(right) != config.k * config.n:
        raise ValueError("right length does not match K x N")
    program = GemmProgram()

    for row in range(config.kernel_rows):
        program.load_a_panel(
            (row, 0),
            local_a_panel_values(config, left, row),
            rows=config.rows_per_pe,
            k=config.k,
        )
        for col in range(config.kernel_cols - 1):
            program.forward_a_east(
                src_pe=(row, col),
                dst_pe=(row, col + 1),
                color="a_color",
            )

    for col in range(config.kernel_cols):
        program.load_b_panel(
            (0, col),
            local_b_panel_values(config, right, col),
            cols=config.cols_per_pe,
            k=config.k,
        )
        for row in range(config.kernel_rows - 1):
            program.forward_b_south(
                src_pe=(row, col),
                dst_pe=(row + 1, col),
                color="b_color",
            )

    for row in range(config.kernel_rows):
        for col in range(config.kernel_cols):
            program.compute_c(
                (row, col),
                rows=config.rows_per_pe,
                cols=config.cols_per_pe,
                k=config.k,
            )

    for coord, tile in expected_tiles(config, left, right).items():
        program.expect_c(
            coord,
            tile,
            rows=config.rows_per_pe,
            cols=config.cols_per_pe,
        )
    return program


def local_a_panel_values(config: GemmStreamingConfig, left: list[int], row: int) -> list[int]:
    row0 = row * config.rows_per_pe
    values: list[int] = []
    for local_row in range(config.rows_per_pe):
        global_row = row0 + local_row
        start = global_row * config.k
        values.extend(left[start : start + config.k])
    return values


def local_b_panel_values(config: GemmStreamingConfig, right: list[int], col: int) -> list[int]:
    col0 = col * config.cols_per_pe
    values: list[int] = []
    for kk in range(config.k):
        start = kk * config.n + col0
        values.extend(right[start : start + config.cols_per_pe])
    return values


def expected_tiles(
    config: GemmStreamingConfig,
    left: list[int],
    right: list[int],
) -> dict[tuple[int, int], tuple[int, ...]]:
    expected = gemm(
        [float(value) for value in left],
        [float(value) for value in right],
        config.m,
        config.n,
        config.k,
    )
    tiles: dict[tuple[int, int], tuple[int, ...]] = {}
    for row in range(config.kernel_rows):
        for col in range(config.kernel_cols):
            row0 = row * config.rows_per_pe
            col0 = col * config.cols_per_pe
            values: list[int] = []
            for local_row in range(config.rows_per_pe):
                global_row = row0 + local_row
                start = global_row * config.n + col0
                values.extend(int(value) for value in expected[start : start + config.cols_per_pe])
            tiles[(row, col)] = tuple(values)
    return tiles


def apply_layout_overrides(
    layout: LayoutSpec,
    args: argparse.Namespace,
) -> LayoutSpec:
    return LayoutSpec(
        mesh_rows=args.mesh_rows or layout.mesh_rows,
        mesh_cols=args.mesh_cols or layout.mesh_cols,
        kernel_rows=args.kernel_rows or layout.kernel_rows,
        kernel_cols=args.kernel_cols or layout.kernel_cols,
        routes=layout.routes,
    )


if __name__ == "__main__":
    main()
