from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from wafersim.opencake import HardwareSpec, run_formal_opencake_pipeline


def run_cpp_backend(
    *,
    chakra_trace: Path,
    out_dir: Path,
    backend_dir: Path,
) -> dict[str, object]:
    formal_result = run_formal_opencake_pipeline(
        chakra_trace=chakra_trace,
        out_dir=out_dir,
        hardware=HardwareSpec(),
    )
    subprocess.run(["make"], cwd=backend_dir, check=True)
    binary = backend_dir / "build" / "opencake_wafer_backend"
    completed = subprocess.run(
        [str(binary), "--package", str(out_dir), "--check-reference"],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "formal_package": formal_result.package.summary(),
        "python_reference": formal_result.reference_summary,
        "cpp_backend": json.loads(completed.stdout),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate a formal package and validate it with the C++ OpenCake backend."
    )
    parser.add_argument(
        "--chakra-trace",
        type=Path,
        default=repo_root / "examples" / "opencake_pipeline" / "chakra_gemm.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencake_formal_package_cpp"))
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=repo_root / "cpp" / "opencake_wafer_backend",
    )
    args = parser.parse_args()

    result = run_cpp_backend(
        chakra_trace=args.chakra_trace,
        out_dir=args.out_dir,
        backend_dir=args.backend_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
