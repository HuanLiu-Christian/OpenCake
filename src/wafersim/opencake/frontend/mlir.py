from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from wafersim.opencake.workload import GemmWorkload


_MATMUL_RE = re.compile(r"linalg\.matmul\b")
_TENSOR_RE = re.compile(r"tensor<\s*(\d+)\s*x\s*(\d+)\s*x\s*([a-zA-Z0-9_]+)\s*>")


@dataclass(frozen=True, slots=True)
class MlirMatmulOp:
    """Small MLIR linalg.matmul frontend record used by OpenCake.

    The parser intentionally accepts a compact subset of MLIR text instead of
    depending on MLIR Python bindings. It extracts the semantic contract the
    mapper needs: A[M,K], B[K,N], C[M,N], and element width.
    """

    m: int
    n: int
    k: int
    dtype: str
    bytes_per_element: int

    def to_workload(self) -> GemmWorkload:
        return GemmWorkload(
            m=self.m,
            n=self.n,
            k=self.k,
            bytes_per_element=self.bytes_per_element,
        )


def read_linalg_matmul(path: str | Path) -> MlirMatmulOp:
    return parse_linalg_matmul(Path(path).read_text(encoding="utf-8"))


def parse_linalg_matmul(text: str) -> MlirMatmulOp:
    line = _find_matmul_line(text)
    tensors = _TENSOR_RE.findall(line)
    if len(tensors) < 3:
        raise ValueError(
            "expected linalg.matmul with A, B, and C tensor types, for example "
            "tensor<1024x1024xf32>"
        )

    (a_m, a_k, a_dtype), (b_k, b_n, b_dtype), (c_m, c_n, c_dtype) = tensors[:3]
    m = int(a_m)
    k = int(a_k)
    n = int(b_n)
    if int(b_k) != k:
        raise ValueError(f"inconsistent K dimension: A has {k}, B has {b_k}")
    if int(c_m) != m or int(c_n) != n:
        raise ValueError(f"inconsistent C shape: expected ({m}, {n}), got ({c_m}, {c_n})")
    if a_dtype != b_dtype or a_dtype != c_dtype:
        raise ValueError(f"mixed matmul dtypes are not supported: {a_dtype}, {b_dtype}, {c_dtype}")

    return MlirMatmulOp(
        m=m,
        n=n,
        k=k,
        dtype=a_dtype,
        bytes_per_element=_dtype_bytes(a_dtype),
    )


def canonical_linalg_matmul_module(op: MlirMatmulOp) -> str:
    """Normalize an IREE-recovered matmul into standard upstream MLIR.

    IREE phase dumps use IREE-only ``util``, ``hal``, ``flow``, and ``stream``
    dialects that the standalone MLIR-AIR build does not register. OpenCake
    recovers the stable linalg semantic operation, then emits this equivalent
    standard ``func.func`` payload for MLIR Transform and MLIR-AIR.
    """

    a_type = f"tensor<{op.m}x{op.k}x{op.dtype}>"
    b_type = f"tensor<{op.k}x{op.n}x{op.dtype}>"
    c_type = f"tensor<{op.m}x{op.n}x{op.dtype}>"
    return f"""module {{
  func.func @main(%A: {a_type}, %B: {b_type}, %C: {c_type}) -> {c_type} {{
    %0 = linalg.matmul ins(%A, %B : {a_type}, {b_type}) outs(%C : {c_type}) -> {c_type}
    return %0 : {c_type}
  }}
}}
"""


def _find_matmul_line(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _MATMUL_RE.search(line):
            chunk = []
            for cursor in range(index, min(len(lines), index + 8)):
                chunk.append(lines[cursor])
                if len(_TENSOR_RE.findall(" ".join(chunk))) >= 3:
                    break
            return " ".join(chunk)
    raise ValueError("no linalg.matmul operation found")


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in {"f16", "bf16", "i16", "ui16"}:
        return 2
    if normalized in {"f32", "i32", "ui32"}:
        return 4
    if normalized in {"f64", "i64", "ui64"}:
        return 8
    if normalized in {"i8", "ui8"}:
        return 1
    raise ValueError(f"unsupported MLIR tensor element type: {dtype}")
