from __future__ import annotations

import argparse
import json
from pathlib import Path

from wafersim.opencake import HardwareSpec, run_chakra_to_wafer_sim


def run_pipeline(
    *,
    chakra_trace: Path,
    spatial_ir_out: Path | None = None,
    mesh_rows: int = 32,
    mesh_cols: int = 32,
    pe_block_rows: int = 1,
    pe_block_cols: int = 1,
) -> dict[str, object]:
    result = run_chakra_to_wafer_sim(
        chakra_trace=chakra_trace,
        hardware=HardwareSpec(
            mesh_rows=mesh_rows,
            mesh_cols=mesh_cols,
            pe_block_rows=pe_block_rows,
            pe_block_cols=pe_block_cols,
        ),
        spatial_ir_out=spatial_ir_out,
    )
    return result.summary()


def main() -> None:
    default_trace = Path(__file__).with_name("chakra_gemm.jsonl")
    parser = argparse.ArgumentParser(
        description="Run Chakra trace -> OpenCake Mapper -> Spatial IR -> wafer simulator."
    )
    parser.add_argument("--chakra-trace", type=Path, default=default_trace)
    parser.add_argument("--spatial-ir-out", type=Path, default=None)
    parser.add_argument("--mesh-rows", type=int, default=32)
    parser.add_argument("--mesh-cols", type=int, default=32)
    parser.add_argument("--pe-block-rows", type=int, default=1)
    parser.add_argument("--pe-block-cols", type=int, default=1)
    args = parser.parse_args()

    summary = run_pipeline(
        chakra_trace=args.chakra_trace,
        spatial_ir_out=args.spatial_ir_out,
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        pe_block_rows=args.pe_block_rows,
        pe_block_cols=args.pe_block_cols,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
