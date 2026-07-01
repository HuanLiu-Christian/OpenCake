from __future__ import annotations

import argparse
import json
from pathlib import Path

from wafersim.opencake import HardwareSpec, run_formal_opencake_pipeline


def main() -> None:
    default_trace = Path(__file__).parents[1] / "opencake_pipeline" / "chakra_gemm.jsonl"
    parser = argparse.ArgumentParser(
        description="Generate the formal OpenCake package for Astra-Sim wafer backend integration."
    )
    parser.add_argument("--chakra-trace", type=Path, default=default_trace)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencake_astrasim_package"))
    parser.add_argument("--mesh-rows", type=int, default=32)
    parser.add_argument("--mesh-cols", type=int, default=32)
    parser.add_argument("--pe-block-rows", type=int, default=8)
    parser.add_argument("--pe-block-cols", type=int, default=8)
    args = parser.parse_args()

    result = run_formal_opencake_pipeline(
        chakra_trace=args.chakra_trace,
        out_dir=args.out_dir,
        hardware=HardwareSpec(
            mesh_rows=args.mesh_rows,
            mesh_cols=args.mesh_cols,
            pe_block_rows=args.pe_block_rows,
            pe_block_cols=args.pe_block_cols,
        ),
    )
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
