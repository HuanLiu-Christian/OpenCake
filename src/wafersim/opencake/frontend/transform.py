from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from wafersim.opencake.mapper.tiler import GemmTilePlan


@dataclass(frozen=True, slots=True)
class MlirTransformResult:
    """Result of applying a real MLIR Transform Dialect GEMM schedule."""

    schema: str
    tile_m: int
    tile_n: int
    tile_k: int
    schedule_mlir: str
    scheduled_payload_mlir: str
    tiled_payload_mlir: str
    command: tuple[str, ...]
    stderr: str
    interpreter: str
    applied: bool

    def summary(self) -> dict[str, object]:
        tiled = self.tiled_payload_mlir
        return {
            "schema": self.schema,
            "tile_sizes": [self.tile_m, self.tile_n, self.tile_k],
            "interpreter": self.interpreter,
            "applied": self.applied,
            "command": list(self.command),
            "stderr": self.stderr.strip(),
            "tiled_ir_ops": {
                "scf.forall": tiled.count("scf.forall "),
                "scf.for": tiled.count("scf.for "),
                "tensor.extract_slice": tiled.count("tensor.extract_slice "),
                "linalg.matmul": tiled.count("linalg.matmul "),
            },
        }


class MlirTransformRunner:
    """Generate and interpret an MLIR Transform Dialect GEMM schedule.

    OpenCake chooses the tile sizes. The upstream MLIR transform interpreter
    performs the actual payload rewrite into ``scf.forall``, ``scf.for``, and
    ``tensor.extract_slice`` operations.
    """

    def __init__(self, binary: str | Path | None = None) -> None:
        self.binary = Path(binary) if binary is not None else self._find_binary()

    @classmethod
    def available(cls) -> bool:
        return cls._find_binary() is not None

    def tile_gemm(
        self,
        input_file: str | Path,
        tile_plan: GemmTilePlan,
    ) -> MlirTransformResult:
        input_path = Path(input_file)
        payload = input_path.read_text(encoding="utf-8")
        schedule = generate_gemm_transform_schedule(tile_plan)
        scheduled_payload = embed_transform_schedule(payload, tile_plan)

        if self.binary is None:
            return MlirTransformResult(
                schema="opencake.mlir_transform_result.v1",
                tile_m=tile_plan.tile_m,
                tile_n=tile_plan.tile_n,
                tile_k=tile_plan.tile_k,
                schedule_mlir=schedule,
                scheduled_payload_mlir=scheduled_payload,
                tiled_payload_mlir="",
                command=(),
                stderr="iree-opt was not found; install wafersim[iree] to materialize tiled MLIR",
                interpreter="unavailable",
                applied=False,
            )

        with tempfile.TemporaryDirectory(prefix="opencake_transform_") as temp_dir:
            root = Path(temp_dir)
            scheduled_path = root / "scheduled_payload.mlir"
            tiled_path = root / "tiled_payload.mlir"
            scheduled_path.write_text(scheduled_payload, encoding="utf-8")
            command = (
                str(self.binary),
                str(scheduled_path),
                "--transform-interpreter",
                "--transform-dialect-drop-schedule",
                "-o",
                str(tiled_path),
            )
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            tiled_payload = tiled_path.read_text(encoding="utf-8")

        self._verify_tiled_payload(tiled_payload, tile_plan)
        return MlirTransformResult(
            schema="opencake.mlir_transform_result.v1",
            tile_m=tile_plan.tile_m,
            tile_n=tile_plan.tile_n,
            tile_k=tile_plan.tile_k,
            schedule_mlir=schedule,
            scheduled_payload_mlir=scheduled_payload,
            tiled_payload_mlir=tiled_payload,
            command=command,
            stderr=completed.stderr,
            interpreter="iree-opt --transform-interpreter",
            applied=True,
        )

    @staticmethod
    def _find_binary() -> Path | None:
        sibling = Path(sys.executable).with_name("iree-opt")
        if sibling.exists():
            return sibling
        found = shutil.which("iree-opt")
        return Path(found) if found is not None else None

    @staticmethod
    def _verify_tiled_payload(payload: str, plan: GemmTilePlan) -> None:
        required = ("scf.forall ", "scf.for ", "tensor.extract_slice ", "linalg.matmul ")
        missing = [op for op in required if op not in payload]
        if missing:
            raise ValueError(f"MLIR transform output is missing expected tiled operations: {missing}")

        expected_a_tile = f"tensor<{plan.tile_m}x{plan.tile_k}x"
        expected_b_tile = f"tensor<{plan.tile_k}x{plan.tile_n}x"
        expected_c_tile = f"tensor<{plan.tile_m}x{plan.tile_n}x"
        missing_shapes = [
            shape
            for shape in (expected_a_tile, expected_b_tile, expected_c_tile)
            if shape not in payload
        ]
        if missing_shapes:
            raise ValueError(f"MLIR transform output is missing expected tile shapes: {missing_shapes}")


def generate_gemm_transform_schedule(tile_plan: GemmTilePlan) -> str:
    """Return a standalone, inspectable Transform Dialect schedule."""

    sequence = _named_gemm_sequence(tile_plan, indent="  ")
    return f"module attributes {{transform.with_named_sequence}} {{\n{sequence}\n}}\n"


def embed_transform_schedule(payload: str, tile_plan: GemmTilePlan) -> str:
    """Embed ``__transform_main`` in a top-level MLIR module."""

    module_match = re.search(r"\bmodule(?:\s+@[A-Za-z_.$-][A-Za-z0-9_.$-]*)?\s*\{", payload)
    if module_match is None:
        raise ValueError("expected a top-level MLIR module")
    if "transform.with_named_sequence" in payload:
        raise ValueError("payload already contains a Transform Dialect schedule")

    opening = module_match.group(0)
    replacement = opening[:-1].rstrip() + " attributes {transform.with_named_sequence} {"
    attributed = payload[: module_match.start()] + replacement + payload[module_match.end() :]
    closing = attributed.rfind("}")
    if closing < 0:
        raise ValueError("top-level MLIR module is missing its closing brace")

    sequence = _named_gemm_sequence(tile_plan, indent="  ")
    return attributed[:closing].rstrip() + "\n\n" + sequence + "\n" + attributed[closing:]


def _named_gemm_sequence(tile_plan: GemmTilePlan, *, indent: str) -> str:
    lines = [
        "transform.named_sequence @__transform_main(",
        "    %root: !transform.any_op {transform.readonly}) {",
        "  %matmul = transform.structured.match ops{[\"linalg.matmul\"]} in %root",
        "      : (!transform.any_op) -> !transform.any_op",
        "  %mn_tiled, %parallel = transform.structured.tile_using_forall %matmul",
        f"      tile_sizes [{tile_plan.tile_m}, {tile_plan.tile_n}, 0]",
        "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
        "  %mnk_tiled, %reduction = transform.structured.tile_using_for %mn_tiled",
        f"      tile_sizes [0, 0, {tile_plan.tile_k}]",
        "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
        "  transform.yield",
        "}",
    ]
    return "\n".join(indent + line for line in lines)
