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
from pe_program import PE_PROGRAM
from wafersim.program import GemvProblemSpec, LayoutSpec, build_gemv_streaming_config
from wafersim.simulator import run_simulation, strip_internal_records


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
    config = build_gemv_streaming_config(layout=layout, problem=problem)
    summary = strip_internal_records(run_simulation(config))
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


if __name__ == "__main__":
    main()
