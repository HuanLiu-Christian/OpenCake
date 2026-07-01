from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.astra_sim_opencake.install import install_opencake_backend  # noqa: E402
from wafersim.opencake import HardwareSpec, run_formal_opencake_pipeline  # noqa: E402


def run_integrated_astrasim(
    *,
    chakra_trace: Path,
    out_dir: Path,
    astra_sim_root: Path,
    use_docker: bool,
    docker_image: str,
) -> dict[str, object]:
    formal_result = run_formal_opencake_pipeline(
        chakra_trace=chakra_trace,
        out_dir=out_dir,
        hardware=HardwareSpec(),
    )
    install_result = install_opencake_backend(repo_root=REPO_ROOT, astra_sim_root=astra_sim_root)

    if use_docker:
        micro_root = astra_sim_root.parent
        rel_build = astra_sim_root.relative_to(micro_root)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{micro_root}:/app/micro2024",
                docker_image,
                "bash",
                "-lc",
                f"cd /app/micro2024/{rel_build}/build/astra_analytical && ./build.sh -t congestion_aware",
            ],
            check=True,
        )
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{micro_root}:/app/micro2024",
                "-v",
                f"{out_dir}:/app/opencake_package",
                docker_image,
                "bash",
                "-lc",
                (
                    f"/app/micro2024/{rel_build}/build/astra_analytical/build/bin/"
                    "AstraSim_OpenCake_Wafer --package /app/opencake_package --check-reference"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        build_script = astra_sim_root / "build" / "astra_analytical" / "build.sh"
        subprocess.run([str(build_script), "-t", "congestion_aware"], check=True)
        binary = astra_sim_root / "build" / "astra_analytical" / "build" / "bin" / "AstraSim_OpenCake_Wafer"
        completed = subprocess.run(
            [str(binary), "--package", str(out_dir), "--check-reference"],
            check=True,
            capture_output=True,
            text=True,
        )
    binary = astra_sim_root / "build" / "astra_analytical" / "build" / "bin" / "AstraSim_OpenCake_Wafer"
    return {
        "integration": install_result,
        "formal_package": formal_result.package.summary(),
        "python_reference": formal_result.reference_summary,
        "astra_sim_binary": str(binary),
        "astra_sim_opencake_backend": json.loads(completed.stdout),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install and run the OpenCake backend inside the official Astra-Sim build tree."
    )
    parser.add_argument(
        "--chakra-trace",
        type=Path,
        default=REPO_ROOT / "examples" / "opencake_pipeline" / "chakra_gemm.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/opencake_astrasim_integrated_package"))
    parser.add_argument(
        "--astra-sim-root",
        type=Path,
        default=Path("/tmp/astra-sim-micro2024/micro2024/astra-sim"),
    )
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--docker-image", default="astrasim/tutorial-micro2024")
    args = parser.parse_args()

    result = run_integrated_astrasim(
        chakra_trace=args.chakra_trace,
        out_dir=args.out_dir,
        astra_sim_root=args.astra_sim_root,
        use_docker=args.use_docker,
        docker_image=args.docker_image,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
