from __future__ import annotations

import argparse
import json
from pathlib import Path

from wafersim.opencake import GemmWorkload, HardwareSpec, OpenCakeBackend, OpenCakeMapper


def run_opencake_gemm(
    *,
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    mesh_rows: int = 32,
    mesh_cols: int = 32,
    pe_block_rows: int = 1,
    pe_block_cols: int = 1,
    ir_out: Path | None = None,
) -> dict[str, object]:
    workload = GemmWorkload(m=m, n=n, k=k)
    hardware = HardwareSpec(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        pe_block_rows=pe_block_rows,
        pe_block_cols=pe_block_cols,
    )
    program = OpenCakeMapper().map_gemm(workload, hardware)
    result = OpenCakeBackend().run(program, hardware)

    if ir_out is not None:
        program.write_jsonl(ir_out)

    return {
        "workload": program.workload,
        "hardware": program.hardware,
        "spatial_ir_ops": len(program.ops),
        "result": result.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal OpenCake GEMM mapper/backend.")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--mesh-rows", type=int, default=32)
    parser.add_argument("--mesh-cols", type=int, default=32)
    parser.add_argument("--pe-block-rows", type=int, default=1)
    parser.add_argument("--pe-block-cols", type=int, default=1)
    parser.add_argument("--ir-out", type=Path, default=None)
    args = parser.parse_args()

    summary = run_opencake_gemm(
        m=args.m,
        n=args.n,
        k=args.k,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        pe_block_rows=args.pe_block_rows,
        pe_block_cols=args.pe_block_cols,
        ir_out=args.ir_out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
