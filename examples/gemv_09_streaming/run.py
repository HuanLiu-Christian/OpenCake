from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import simpy


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from layout import LAYOUT
from pe_program import PE_PROGRAM, GemvStreamingConfig, GemvStreamingPeProgram
from wafersim.config import MeshConfig
from wafersim.config import Coord
from wafersim.csl import CslBuiltins, CslCompatibilityRuntime
from wafersim.hardware.network import WaferMesh
from wafersim.program import GemvProblemSpec, LayoutSpec
from wafersim.workloads.kernels import gemv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Python-native GEMV streaming SimPy example."
    )
    parser.add_argument("--mesh-rows", type=int)
    parser.add_argument("--mesh-cols", type=int)
    parser.add_argument("--kernel-rows", type=int)
    parser.add_argument("--kernel-cols", type=int)
    parser.add_argument("--matrix-rows", type=int)
    parser.add_argument("--matrix-cols", type=int)
    parser.add_argument("--bytes-per-element", type=int)
    parser.add_argument("--packet-payload-bytes", type=int)
    parser.add_argument("--ops-per-mac", type=int)
    parser.add_argument(
        "--full-vectors",
        action="store_true",
        help="print full expected_y/result_y vectors instead of compact samples",
    )
    args = parser.parse_args()

    layout = apply_layout_overrides(LAYOUT, args)
    problem = GemvProblemSpec(
        matrix_rows=args.matrix_rows or 6,
        matrix_cols=args.matrix_cols or 8,
        bytes_per_element=args.bytes_per_element or 4,
        packet_payload_bytes=args.packet_payload_bytes or 64,
        ops_per_mac=args.ops_per_mac or 2,
    )
    summary = run_gemv_streaming(layout=layout, problem=problem)
    if not args.full_vectors:
        compact_vectors(summary)
    summary["program"] = {
        "layout": "examples/gemv_09_streaming/layout.py",
        "pe_program": "examples/gemv_09_streaming/pe_program.py",
        "kernel": PE_PROGRAM.kernel,
        "rectangle": [layout.kernel_rows, layout.kernel_cols],
        "colors": {
            color: direction.value if direction is not None else "ramp"
            for color, direction in layout.routes.items()
        },
        "tasks": [task.name for task in PE_PROGRAM.tasks],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


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


def compact_vectors(summary: dict[str, object], sample: int = 8) -> None:
    workload = summary.get("workload")
    if not isinstance(workload, dict):
        return
    for key in ("expected_y", "result_y"):
        values = workload.get(key)
        if not isinstance(values, list) or len(values) <= 2 * sample:
            continue
        workload[f"{key}_sample"] = {
            "length": len(values),
            "head": values[:sample],
            "tail": values[-sample:],
        }
        del workload[key]


def run_gemv_streaming(
    *,
    layout: LayoutSpec,
    problem: GemvProblemSpec,
) -> dict[str, object]:
    config = GemvStreamingConfig(
        matrix_rows=problem.matrix_rows,
        matrix_cols=problem.matrix_cols,
        kernel_rows=layout.kernel_rows,
        kernel_cols=layout.kernel_cols,
        bytes_per_element=problem.bytes_per_element,
        packet_payload_bytes=problem.packet_payload_bytes,
        ops_per_mac=problem.ops_per_mac,
    )
    config.validate()

    env = simpy.Environment()
    mesh = WaferMesh(
        env=env,
        config=MeshConfig(rows=layout.mesh_rows, cols=layout.mesh_cols),
    )
    result: dict[str, object] = {}
    env.run(until=env.process(execute_gemv_program(env, mesh, layout, config, result)))

    elapsed = env.now
    summary = {
        "sim_time_cycles": elapsed,
        "mesh": {
            "rows": mesh.config.rows,
            "cols": mesh.config.cols,
            "pes": mesh.config.rows * mesh.config.cols,
        },
        "workload": result,
        **mesh.pe_summary(elapsed),
        **mesh.packet_summary(),
        **mesh.link_summary(elapsed),
    }
    summary.update(gemv_result_summary(config, result, elapsed))
    return summary


def execute_gemv_program(
    env: simpy.Environment,
    mesh: WaferMesh,
    layout: LayoutSpec,
    config: GemvStreamingConfig,
    result: dict[str, object],
):
    validate_layout(mesh, layout)
    region = CslBuiltins.set_rectangle(layout.kernel_rows, layout.kernel_cols)
    runtime = CslCompatibilityRuntime(
        env=env,
        mesh=mesh,
        region=region,
        packet_payload_bytes=config.packet_payload_bytes,
    )
    configure_colors(runtime, layout)
    coords = region.coords()

    matrix = [
        float(index)
        for index in range(config.matrix_rows * config.matrix_cols)
    ]
    vector = [1.0 for _ in range(config.matrix_cols)]
    bias = [2.0 for _ in range(config.matrix_rows)]
    expected = gemv(
        matrix,
        vector,
        bias,
        config.matrix_rows,
        config.matrix_cols,
    )

    pipeline_events = ["layout: create active PE programs and bind color routes"]
    runtime.debug.trace("gemv9_layout", (0, 0), config.active_pes)

    pe_programs = create_pe_programs(env, mesh, runtime, config, coords, matrix)
    yield from preload_a_tiles(env, mesh, config, coords)
    pipeline_events.append("h2d: preload each PE's local A tile")

    task_processes = [
        process
        for program in pe_programs.values()
        for process in program.launch()
    ]
    pe_program_task_count = sum(
        program.tasks_started for program in pe_programs.values()
    )
    pipeline_events.append("launch: start PE-local recv_x, recv_b, and reduce tasks")

    yield from inject_x_from_host(env, runtime, config, pe_programs, vector)
    pipeline_events.append("host: inject x chunks; PE programs forward x SOUTH")

    yield from inject_b_from_host(env, runtime, config, pe_programs, bias)
    pipeline_events.append("host: inject b chunks into west-edge PE programs")

    done_events = [
        pe_programs[(row, config.kernel_cols - 1)].done
        for row in range(config.kernel_rows)
    ]
    final_chunks = yield simpy.events.AllOf(env, done_events)
    result_y = [
        value
        for row in range(config.kernel_rows)
        for value in final_chunks[done_events[row]]
    ]
    if task_processes:
        yield simpy.events.AllOf(env, task_processes)
    pe_program_fmacs = sum(program.local_fmacs for program in pe_programs.values())
    pipeline_events.append("PE-local programs reduce partial y EAST and return final y")

    runtime.debug.trace("gemv9_done", (0, 0), env.now)
    runtime.simprint.print((0, 0), "gemv9 streaming done")
    pipeline_events.append("host: verify y against expected result")

    result.update(
        {
            "type": "gemv_09_streaming_example",
            "matrix_rows": config.matrix_rows,
            "matrix_cols": config.matrix_cols,
            "kernel_rows": config.kernel_rows,
            "kernel_cols": config.kernel_cols,
            "rows_per_pe": config.rows_per_pe,
            "cols_per_pe": config.cols_per_pe,
            "bytes_per_element": config.bytes_per_element,
            "packet_payload_bytes": config.packet_payload_bytes,
            "matrix_bytes": config.matrix_bytes,
            "x_bytes": config.x_bytes,
            "b_bytes": config.b_bytes,
            "y_bytes": config.y_bytes,
            "a_tile_bytes_per_pe": config.a_tile_bytes_per_pe,
            "memory_footprint_bytes_per_pe": config.memory_footprint_bytes_per_pe,
            "compute_ops": config.compute_ops,
            "compute_ops_per_pe": config.compute_ops_per_pe,
            "color_directions": color_direction_summary(layout),
            "color_packets": dict(runtime.colors.packets),
            "color_bytes": dict(runtime.colors.bytes),
            "pe_program_model": "event_driven_local_state",
            "pe_program_tasks_started": pe_program_task_count,
            "pe_program_fmacs": pe_program_fmacs,
            "pipeline_events": pipeline_events,
            "expected_y": expected,
            "result_y": result_y,
            "verified": result_y == expected,
            "runtime_summary": runtime.summary(),
        }
    )


def create_pe_programs(
    env: simpy.Environment,
    mesh: WaferMesh,
    runtime: CslCompatibilityRuntime,
    config: GemvStreamingConfig,
    coords: list[Coord],
    matrix: list[float],
) -> dict[Coord, GemvStreamingPeProgram]:
    pe_programs = {
        coord: GemvStreamingPeProgram(
            env=env,
            mesh=mesh,
            runtime=runtime,
            config=config,
            coord=coord,
            matrix=matrix,
        )
        for coord in coords
    }
    for program in pe_programs.values():
        program.bind_programs(pe_programs)
        mesh.pes[program.coord].reserve_memory(config.memory_footprint_bytes_per_pe)
    return pe_programs


def preload_a_tiles(
    env: simpy.Environment,
    mesh: WaferMesh,
    config: GemvStreamingConfig,
    coords: list[Coord],
):
    events = [
        env.process(mesh.pes[coord].access_memory(config.a_tile_bytes_per_pe))
        for coord in coords
    ]
    if events:
        yield simpy.events.AllOf(env, events)


def inject_x_from_host(
    env: simpy.Environment,
    runtime: CslCompatibilityRuntime,
    config: GemvStreamingConfig,
    pe_programs: dict[Coord, GemvStreamingPeProgram],
    vector: list[float],
):
    events = []
    for col in range(config.kernel_cols):
        x_values = vector[
            col * config.cols_per_pe : (col + 1) * config.cols_per_pe
        ]
        events.append(
            env.process(
                inject_x_chunk(runtime, config, pe_programs[(0, col)], x_values)
            )
        )
    if events:
        yield simpy.events.AllOf(env, events)


def inject_x_chunk(
    runtime: CslCompatibilityRuntime,
    config: GemvStreamingConfig,
    program: GemvStreamingPeProgram,
    x_values: list[float],
):
    yield from runtime.colors.send(
        "MEMCPYH2D_DATA_1",
        None,
        program.coord,
        config.x_chunk_bytes_per_col,
    )
    yield program.x_queue.put(list(x_values))


def inject_b_from_host(
    env: simpy.Environment,
    runtime: CslCompatibilityRuntime,
    config: GemvStreamingConfig,
    pe_programs: dict[Coord, GemvStreamingPeProgram],
    bias: list[float],
):
    events = []
    for row in range(config.kernel_rows):
        b_values = bias[
            row * config.rows_per_pe : (row + 1) * config.rows_per_pe
        ]
        events.append(
            env.process(
                inject_b_chunk(runtime, config, pe_programs[(row, 0)], b_values)
            )
        )
    if events:
        yield simpy.events.AllOf(env, events)


def inject_b_chunk(
    runtime: CslCompatibilityRuntime,
    config: GemvStreamingConfig,
    program: GemvStreamingPeProgram,
    b_values: list[float],
):
    yield from runtime.colors.send(
        "MEMCPYH2D_DATA_2",
        None,
        program.coord,
        config.b_chunk_bytes_per_row,
    )
    yield program.b_queue.put(list(b_values))


def configure_colors(runtime: CslCompatibilityRuntime, layout: LayoutSpec) -> None:
    for color, direction in layout.routes.items():
        runtime.colors.configure(color, direction)


def color_direction_summary(layout: LayoutSpec) -> dict[str, str]:
    return {
        color: direction.value if direction is not None else "ramp"
        for color, direction in layout.routes.items()
    }


def validate_layout(mesh: WaferMesh, layout: LayoutSpec) -> None:
    if layout.kernel_rows > mesh.config.rows:
        raise ValueError("GEMV kernel rows exceed mesh rows")
    if layout.kernel_cols > mesh.config.cols:
        raise ValueError("GEMV kernel cols exceed mesh cols")


def gemv_result_summary(
    config: GemvStreamingConfig,
    result: dict[str, object],
    elapsed_cycles: float,
) -> dict[str, float | int | bool]:
    color_packets = result.get("color_packets", {})
    color_bytes = result.get("color_bytes", {})
    if not isinstance(color_packets, dict) or not isinstance(color_bytes, dict):
        return {}
    return {
        "gemv_09_streaming_ops_per_cycle": (
            config.compute_ops / elapsed_cycles if elapsed_cycles > 0 else 0.0
        ),
        "gemv_09_streaming_color_packets": sum(color_packets.values()),
        "gemv_09_streaming_color_bytes": sum(color_bytes.values()),
        "gemv_09_streaming_verified": bool(result.get("verified", False)),
    }


if __name__ == "__main__":
    main()
