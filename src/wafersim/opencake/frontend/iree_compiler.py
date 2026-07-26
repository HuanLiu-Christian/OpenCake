from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys


@dataclass(frozen=True, slots=True)
class IreeCompileArtifact:
    phase: str
    output_path: Path
    command: tuple[str, ...]
    stderr: str

    def summary(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "output": str(self.output_path),
            "bytes": self.output_path.stat().st_size if self.output_path.exists() else 0,
            "command": list(self.command),
            "stderr": self.stderr.strip(),
        }


@dataclass(frozen=True, slots=True)
class IreeCompileResult:
    input_path: Path
    artifacts: tuple[IreeCompileArtifact, ...]

    def artifact(self, phase: str) -> IreeCompileArtifact:
        for artifact in self.artifacts:
            if artifact.phase == phase:
                return artifact
        raise KeyError(f"IREE phase was not compiled: {phase}")

    def summary(self) -> dict[str, object]:
        return {
            "input": str(self.input_path),
            "artifacts": [artifact.summary() for artifact in self.artifacts],
        }


class IreeCompilerRunner:
    """Thin wrapper around the real `iree-compile` binary.

    OpenCake still owns wafer-scale tiling and streaming. IREE is used here as
    the real frontend compiler stage that verifies and lowers linalg input IR
    into IREE phases such as flow and stream before OpenCake extracts workload
    shape information.
    """

    def __init__(self, binary: str | Path | None = None) -> None:
        self.binary = Path(binary) if binary is not None else self._find_binary()

    def compile_workload(
        self,
        input_file: str | Path,
        *,
        out_dir: str | Path,
        phases: tuple[str, ...] = ("input", "preprocessing", "flow", "stream"),
        emit_vmfb: bool = True,
        target_backend: str = "vmvx",
    ) -> IreeCompileResult:
        input_path = Path(input_file)
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[IreeCompileArtifact] = []
        for phase in phases:
            output_path = output_dir / f"{input_path.stem}.{phase}.mlir"
            command = (
                str(self.binary),
                str(input_path),
                f"--compile-to={phase}",
                "-o",
                str(output_path),
            )
            artifacts.append(self._run(command, phase=phase, output_path=output_path))

        if emit_vmfb:
            output_path = output_dir / f"{input_path.stem}.{target_backend}.vmfb"
            command = (
                str(self.binary),
                str(input_path),
                f"--iree-hal-target-backends={target_backend}",
                "-o",
                str(output_path),
            )
            artifacts.append(self._run(command, phase=f"{target_backend}_vmfb", output_path=output_path))

        return IreeCompileResult(input_path=input_path, artifacts=tuple(artifacts))

    def compile_conv2d(
        self,
        input_file: str | Path,
        *,
        out_dir: str | Path,
        phases: tuple[str, ...] = ("input", "preprocessing", "flow", "stream"),
        emit_vmfb: bool = True,
        target_backend: str = "vmvx",
    ) -> IreeCompileResult:
        """Backward-compatible alias for the original Conv2 example API."""

        return self.compile_workload(
            input_file,
            out_dir=out_dir,
            phases=phases,
            emit_vmfb=emit_vmfb,
            target_backend=target_backend,
        )

    @staticmethod
    def _find_binary() -> Path:
        sibling = Path(sys.executable).with_name("iree-compile")
        if sibling.exists():
            return sibling
        found = shutil.which("iree-compile")
        if found is not None:
            return Path(found)
        raise FileNotFoundError(
            "iree-compile was not found. Install it with `python -m pip install iree-compiler`."
        )

    @staticmethod
    def _run(
        command: tuple[str, ...],
        *,
        phase: str,
        output_path: Path,
    ) -> IreeCompileArtifact:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if not output_path.exists():
            raise FileNotFoundError(f"IREE did not produce expected output: {output_path}")
        return IreeCompileArtifact(
            phase=phase,
            output_path=output_path,
            command=command,
            stderr=completed.stderr,
        )
