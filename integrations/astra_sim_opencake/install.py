from __future__ import annotations

import argparse
from pathlib import Path
import shutil


BEGIN = "# BEGIN OPENCAKE WAFER BACKEND"
END = "# END OPENCAKE WAFER BACKEND"


def install_opencake_backend(repo_root: Path, astra_sim_root: Path) -> dict[str, str]:
    source = repo_root / "cpp" / "opencake_wafer_backend" / "src" / "opencake_wafer_backend.cpp"
    if not source.exists():
        raise FileNotFoundError(source)
    if not astra_sim_root.exists():
        raise FileNotFoundError(astra_sim_root)

    target_dir = astra_sim_root / "astra-sim" / "network_frontend" / "opencake"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_source = target_dir / "opencake_wafer_backend.cpp"
    shutil.copyfile(source, target_source)

    cmake_file = astra_sim_root / "astra-sim" / "network_frontend" / "analytical" / "CMakeLists.txt"
    text = cmake_file.read_text(encoding="utf-8")
    snippet = f"""
{BEGIN}
add_executable(AstraSim_OpenCake_Wafer
    ${{CMAKE_CURRENT_SOURCE_DIR}}/../opencake/opencake_wafer_backend.cpp
)
target_link_libraries(AstraSim_OpenCake_Wafer LINK_PRIVATE AstraSim)
target_include_directories(AstraSim_OpenCake_Wafer PRIVATE ${{CMAKE_CURRENT_SOURCE_DIR}}/../../../)
set_target_properties(AstraSim_OpenCake_Wafer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}/../bin/
        LIBRARY_OUTPUT_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}/../lib/
        ARCHIVE_OUTPUT_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}/../lib/
)
{END}
"""
    if BEGIN not in text:
        cmake_file.write_text(text.rstrip() + "\n" + snippet, encoding="utf-8")

    return {
        "source": str(target_source),
        "cmake": str(cmake_file),
        "target": "AstraSim_OpenCake_Wafer",
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Install OpenCake wafer backend into an Astra-Sim tree.")
    parser.add_argument(
        "--astra-sim-root",
        type=Path,
        required=True,
        help="Path to the official Astra-Sim root, for example /tmp/.../micro2024/astra-sim",
    )
    args = parser.parse_args()
    result = install_opencake_backend(repo_root=repo_root, astra_sim_root=args.astra_sim_root)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
