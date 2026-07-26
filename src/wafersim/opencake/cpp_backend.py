from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


@dataclass(frozen=True, slots=True)
class CppBackendResult:
    """Result from the C++ OpenCake wafer backend executable."""

    backend_dir: Path
    binary: Path
    package_root: Path
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_dir": str(self.backend_dir),
            "binary": str(self.binary),
            "package_root": str(self.package_root),
            **self.summary,
        }


class OpenCakeCppBackendRunner:
    """Build and run the formal C++ wafer backend on an exported package."""

    def __init__(self, backend_dir: str | Path) -> None:
        self.backend_dir = Path(backend_dir)

    @property
    def binary(self) -> Path:
        return self.backend_dir / "build" / "opencake_wafer_backend"

    def build(self) -> None:
        subprocess.run(
            ["make"],
            cwd=self.backend_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    def run(
        self,
        *,
        package_root: str | Path,
        check_reference: bool = True,
        build: bool = True,
    ) -> CppBackendResult:
        if build:
            self.build()
        package = Path(package_root)
        command = [str(self.binary), "--package", str(package)]
        if check_reference:
            command.append("--check-reference")
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return CppBackendResult(
            backend_dir=self.backend_dir,
            binary=self.binary,
            package_root=package,
            summary=json.loads(completed.stdout),
        )
