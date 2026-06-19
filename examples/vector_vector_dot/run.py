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
from pe_program import PE_PROGRAM, VectorDotConfig, VectorDotPeProgram
from wafersim.config import Coord, MeshConfig
from wafersim.csl import CslBuiltins, CslCompatibilityRuntime
from wafersim.hardware.vector_dot import run_vector_dot_commands
from wafersim.mapping import (
    VectorDotProgram,
    write_vector_dot_commands,
)
from wafersim.hardware.network import WaferMesh
from wafersim.program import LayoutSpec
from wafersim.workloads.kernels import vector_dot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the vector-vector dot product SimPy example."
    )
    parser.add_argument("--mesh-rows", type=int)
    parser.add_argument("--mesh-cols", type=int)
    parser.add_argument("--kernel-cols", type=int)
    parser.add_argument("--vector-length", type=int)
    parser.add_argument("--bytes-per-element", type=int)
    parser.add_argument("--packet-payload-bytes", type=int)
    parser.add_argument("--ops-per-mac", type=int)
    parser.add_argument(
        "--full-vectors",
        action="store_true",
        help="print full input vectors instead of compact samples",
    )
    parser.add_argument(
        "--ir-out",
        type=Path,
        help="Optional path for the generated vector-dot JSONL IR.",
    )
    args = parser.parse_args()

    layout = apply_layout_overrides(LAYOUT, args)
    config = VectorDotConfig(
        vector_length=args.vector_length or 16,
        kernel_cols=layout.kernel_cols,
        bytes_per_element=args.bytes_per_element or 4,
        packet_payload_bytes=args.packet_payload_bytes or 64,
        ops_per_mac=args.ops_per_mac or 2,
    )
    summary = run_vector_dot(layout=layout, config=config)
    if args.ir_out is not None:
        left_words = [index + 1 for index in range(config.vector_length)]
        right_words = [1 for _ in range(config.vector_length)]
        program = build_vector_dot_program(
            left_words,
            right_words,
            kernel_cols=config.kernel_cols,
        )
        write_vector_dot_commands(args.ir_out, program.commands)
        summary["generated_ir"] = str(args.ir_out)
        summary["mapping"] = {
            "ir_type": "vector_dot_jsonl",
            "simpy_golden_result": run_vector_dot_commands(program.commands).result,
            "chisel_target": "VectorDotRowTop",
        }
    if not args.full_vectors:
        compact_vectors(summary)
    summary["program"] = {
        "layout": "examples/vector_vector_dot/layout.py",
        "pe_program": "examples/vector_vector_dot/pe_program.py",
        "kernel": PE_PROGRAM.kernel,
        "rectangle": [layout.kernel_rows, layout.kernel_cols],
        "colors": {
            color: direction.value if direction is not None else "ramp"
            for color, direction in layout.routes.items()
        },
        "tasks": [task.name for task in PE_PROGRAM.tasks],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_vector_dot_program(
    left_words: list[int],
    right_words: list[int],
    *,
    kernel_cols: int = 4,
) -> VectorDotProgram:
    if len(left_words) != len(right_words):
        raise ValueError("left and right vectors must have the same length")
    if kernel_cols <= 0:
        raise ValueError("kernel_cols must be positive")
    if len(left_words) % kernel_cols != 0:
        raise ValueError("vector length must be divisible by kernel_cols")

    program = VectorDotProgram()
    chunk = len(left_words) // kernel_cols

    for col in range(kernel_cols):
        pe = (0, col)
        start = col * chunk
        stop = start + chunk
        program.load_left(left_words[start:stop], pe=pe)
        program.load_right(right_words[start:stop], pe=pe)

    for col in range(kernel_cols):
        program.compute_dot(chunk, pe=(0, col))

    for col in range(kernel_cols - 1):
        program.reduce_east(
            src_pe=(0, col),
            dst_pe=(0, col + 1),
            color="partial_sum_color",
        )

    program.expect_dot(
        sum(left * right for left, right in zip(left_words, right_words)),
        pe=(0, kernel_cols - 1),
    )
    return program


def apply_layout_overrides(
    layout: LayoutSpec,
    args: argparse.Namespace,
) -> LayoutSpec:
    return LayoutSpec(
        mesh_rows=args.mesh_rows or layout.mesh_rows,
        mesh_cols=args.mesh_cols or layout.mesh_cols,
        kernel_rows=layout.kernel_rows,
        kernel_cols=args.kernel_cols or layout.kernel_cols,
        routes=layout.routes,
    )


def compact_vectors(summary: dict[str, object], sample: int = 8) -> None:
    workload = summary.get("workload")
    if not isinstance(workload, dict):
        return
    for key in ("left_vector", "right_vector"):
        values = workload.get(key)
        if not isinstance(values, list) or len(values) <= 2 * sample:
            continue
        workload[f"{key}_sample"] = {
            "length": len(values),
            "head": values[:sample],
            "tail": values[-sample:],
        }
        del workload[key]


def run_vector_dot(
    *,
    layout: LayoutSpec,
    config: VectorDotConfig,
) -> dict[str, object]:
    config.validate()
    validate_layout(layout)

    env = simpy.Environment()
    mesh = WaferMesh(
        env=env,
        config=MeshConfig(rows=layout.mesh_rows, cols=layout.mesh_cols),
    )
    result: dict[str, object] = {}
    env.run(until=env.process(execute_vector_dot(env, mesh, layout, config, result)))

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
    summary.update(vector_dot_result_summary(config, result, elapsed))
    return summary


def execute_vector_dot(
    env: simpy.Environment,
    mesh: WaferMesh,
    layout: LayoutSpec,
    config: VectorDotConfig,
    result: dict[str, object],
):
    region = CslBuiltins.set_rectangle(layout.kernel_rows, layout.kernel_cols)
    runtime = CslCompatibilityRuntime(
        env=env,
        mesh=mesh,
        region=region,
        packet_payload_bytes=config.packet_payload_bytes,
    )
    configure_colors(runtime, layout)
    coords = region.coords()

    left = [float(index + 1) for index in range(config.vector_length)]
    right = [1.0 for _ in range(config.vector_length)]
    expected = vector_dot(left, right)

    pipeline_events = ["layout: create a 1D PE row and bind color routes"]
    runtime.debug.trace("vector_dot_layout", (0, 0), config.kernel_cols)

    pe_programs = create_pe_programs(env, mesh, runtime, config, coords)
    yield from inject_input_vectors(env, runtime, config, pe_programs, left, right)
    pipeline_events.append("h2d: inject local a/b chunks into each PE")

    task_processes = [
        process
        for program in pe_programs.values()
        for process in program.launch()
    ]
    pipeline_events.append("launch: start PE-local dot-product tasks in parallel")

    final_value = yield pe_programs[(0, config.kernel_cols - 1)].done
    if task_processes:
        yield simpy.events.AllOf(env, task_processes)
    pipeline_events.append("reduce: partial sums move EAST to the last PE")
    runtime.debug.trace("vector_dot_done", (0, config.kernel_cols - 1), env.now)
    runtime.simprint.print((0, config.kernel_cols - 1), "vector dot done")
    pipeline_events.append("host: verify scalar dot product")

    result.update(
        {
            "type": "vector_vector_dot_example",
            "vector_length": config.vector_length,
            "kernel_cols": config.kernel_cols,
            "elements_per_pe": config.elements_per_pe,
            "bytes_per_element": config.bytes_per_element,
            "packet_payload_bytes": config.packet_payload_bytes,
            "left_vector": left,
            "right_vector": right,
            "a_chunk_bytes_per_pe": config.chunk_bytes_per_pe,
            "b_chunk_bytes_per_pe": config.chunk_bytes_per_pe,
            "memory_footprint_bytes_per_pe": config.memory_footprint_bytes_per_pe,
            "compute_ops": config.compute_ops,
            "compute_ops_per_pe": config.compute_ops_per_pe,
            "color_directions": color_direction_summary(layout),
            "color_packets": dict(runtime.colors.packets),
            "color_bytes": dict(runtime.colors.bytes),
            "pe_program_model": "event_driven_local_state",
            "pe_program_tasks_started": sum(
                program.tasks_started for program in pe_programs.values()
            ),
            "pe_program_fmacs": sum(
                program.local_fmacs for program in pe_programs.values()
            ),
            "local_partials": {
                str(coord): program.local_partial for coord, program in pe_programs.items()
            },
            "pipeline_events": pipeline_events,
            "expected_dot": expected,
            "result_dot": final_value,
            "verified": final_value == expected,
            "runtime_summary": runtime.summary(),
        }
    )


def create_pe_programs(
    env: simpy.Environment,
    mesh: WaferMesh,
    runtime: CslCompatibilityRuntime,
    config: VectorDotConfig,
    coords: list[Coord],
) -> dict[Coord, VectorDotPeProgram]:
    pe_programs = {
        coord: VectorDotPeProgram(
            env=env,
            mesh=mesh,
            runtime=runtime,
            config=config,
            coord=coord,
        )
        for coord in coords
    }
    for program in pe_programs.values():
        program.bind_programs(pe_programs)
        mesh.pes[program.coord].reserve_memory(config.memory_footprint_bytes_per_pe)
    return pe_programs


def inject_input_vectors(
    env: simpy.Environment,
    runtime: CslCompatibilityRuntime,
    config: VectorDotConfig,
    pe_programs: dict[Coord, VectorDotPeProgram],
    left: list[float],
    right: list[float],
):
    events = []
    for col in range(config.kernel_cols):
        start = col * config.elements_per_pe
        stop = start + config.elements_per_pe
        program = pe_programs[(0, col)]
        events.append(
            env.process(inject_vector_chunk(runtime, config, program, "A_H2D", left[start:stop]))
        )
        events.append(
            env.process(inject_vector_chunk(runtime, config, program, "B_H2D", right[start:stop]))
        )
    if events:
        yield simpy.events.AllOf(env, events)


def inject_vector_chunk(
    runtime: CslCompatibilityRuntime,
    config: VectorDotConfig,
    program: VectorDotPeProgram,
    color: str,
    values: list[float],
):
    yield from runtime.colors.send(
        color,
        None,
        program.coord,
        config.chunk_bytes_per_pe,
    )
    if color == "A_H2D":
        yield program.a_queue.put(list(values))
    else:
        yield program.b_queue.put(list(values))


def configure_colors(runtime: CslCompatibilityRuntime, layout: LayoutSpec) -> None:
    for color, direction in layout.routes.items():
        runtime.colors.configure(color, direction)


def color_direction_summary(layout: LayoutSpec) -> dict[str, str]:
    return {
        color: direction.value if direction is not None else "ramp"
        for color, direction in layout.routes.items()
    }


def validate_layout(layout: LayoutSpec) -> None:
    if layout.kernel_rows != 1:
        raise ValueError("vector dot example expects a single active PE row")
    if layout.kernel_cols > layout.mesh_cols:
        raise ValueError("kernel cols exceed mesh cols")


def vector_dot_result_summary(
    config: VectorDotConfig,
    result: dict[str, object],
    elapsed_cycles: float,
) -> dict[str, float | int | bool]:
    color_packets = result.get("color_packets", {})
    color_bytes = result.get("color_bytes", {})
    if not isinstance(color_packets, dict) or not isinstance(color_bytes, dict):
        return {}
    return {
        "vector_dot_ops_per_cycle": (
            config.compute_ops / elapsed_cycles if elapsed_cycles > 0 else 0.0
        ),
        "vector_dot_color_packets": sum(color_packets.values()),
        "vector_dot_color_bytes": sum(color_bytes.values()),
        "vector_dot_verified": bool(result.get("verified", False)),
    }


if __name__ == "__main__":
    main()
