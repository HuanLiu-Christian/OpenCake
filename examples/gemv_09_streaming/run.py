from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wafersim.config import CslGemvStreamingWorkloadConfig, MeshConfig
from wafersim.config import SimulationConfig
from wafersim.csl import load_csl_example
from wafersim.simulator import run_simulation, strip_internal_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CSL-style GEMV 9 streaming SimPy example."
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

    example_dir = Path(__file__).resolve().parent
    program = load_csl_example(example_dir)
    config = build_config(program.constants, args)
    summary = strip_internal_records(run_simulation(config))
    if not args.full_vectors:
        compact_vectors(summary)
    summary["csl_example"] = {
        "layout": str(program.layout_path),
        "pe_program": str(program.pe_program_path),
        "kernel": program.kernel,
        "rectangle": list(program.rectangle) if program.rectangle else None,
        "colors": {
            color: direction.value if direction is not None else "ramp"
            for color, direction in program.colors.items()
        },
        "tasks": program.tasks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_config(
    constants: dict[str, int | float | str | bool],
    args: argparse.Namespace,
) -> SimulationConfig:
    mesh = MeshConfig(
        rows=_int_value(args.mesh_rows, constants, "mesh_rows"),
        cols=_int_value(args.mesh_cols, constants, "mesh_cols"),
    )
    workload = CslGemvStreamingWorkloadConfig(
        matrix_rows=_int_value(args.matrix_rows, constants, "matrix_rows"),
        matrix_cols=_int_value(args.matrix_cols, constants, "matrix_cols"),
        kernel_rows=_int_value(args.kernel_rows, constants, "kernel_rows"),
        kernel_cols=_int_value(args.kernel_cols, constants, "kernel_cols"),
        bytes_per_element=_int_value(
            args.bytes_per_element,
            constants,
            "bytes_per_element",
        ),
        packet_payload_bytes=_int_value(
            args.packet_payload_bytes,
            constants,
            "packet_payload_bytes",
        ),
        ops_per_mac=_int_value(args.ops_per_mac, constants, "ops_per_mac"),
    )
    return SimulationConfig(mesh=mesh, workload=workload)


def _int_value(
    override: int | None,
    constants: dict[str, int | float | str | bool],
    name: str,
) -> int:
    if override is not None:
        return override
    value = constants[name]
    if not isinstance(value, int):
        raise TypeError(f"expected integer CSL constant: {name}")
    return value


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


if __name__ == "__main__":
    main()
