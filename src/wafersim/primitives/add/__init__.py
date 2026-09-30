"""add primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``add(a, b)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy elementwise add, no
    Docker/TimeLoop involved -- see ``Tensor``'s own module docstring for
    why building/operating on real tensor VALUES never needs TimeLoop,
    which only ever estimates hardware COST for an operation count).
  - Real cycle/energy COST MODEL for one PE's local elementwise add: REAL,
    Docker-verified, implemented in ``_timeloop_add.py`` -- ``AddSpec``/
    ``AddResult``/``simulate_add`` re-exported below. Mirrors how
    ``wafersim.primitives.matmul`` is a thin re-export of the project's
    real GEMM pipeline (``simulate_gemm`` never touches actual matrix
    values either -- see ``wafer/functional.py``'s docstring), rather than
    inventing a separate ad hoc estimation path. Uses a REAL, separately-
    characterized Accelergy ``fpadder`` component (confirmed by reading
    ``accelergy-aladdin-plug-in``'s actual source/data files inside this
    project's own pinned Docker image -- see ``_timeloop_add.py``'s module
    docstring for the full citation), not the GEMM ``mac``/``intmac``
    component reused as an approximation.

The two layers stay independent -- ``add(a, b)`` below never launches
Docker, and ``simulate_add()`` never sees ``a``/``b``'s real values, only
an operation count (``AddSpec.elements_per_pe``). Bridge them explicitly
when both numbers are wanted at once, e.g.:
    result = add(a, b)
    cost = simulate_add(AddSpec(elements_per_pe=result.size,
                                 bytes_per_element=result.bytes_per_element),
                         wafer, out_dir)

Used for: residual connections, and bias addition (e.g. Qwen2's QKV bias).
"""

from __future__ import annotations

from ._timeloop_add import AddResult, AddSpec, generate_add_timeloop_inputs, parse_add_result, simulate_add
from ..tensor import Tensor


def _require_same_dtype(a: Tensor, b: Tensor, op_name: str) -> None:
    if a.dtype != b.dtype:
        raise TypeError(
            f"{op_name}(a, b): dtype mismatch a.dtype={a.dtype} vs "
            f"b.dtype={b.dtype} -- cast one operand explicitly via "
            f"wafersim.primitives.cast.cast(...) first; this project models "
            f"mixed precision explicitly, not via numpy's implicit type "
            f"promotion."
        )


def add(a: Tensor, b: Tensor) -> Tensor:
    """Real elementwise (broadcasting) ``a + b`` on real Tensor data."""
    _require_same_dtype(a, b, "add")
    return Tensor(a.data + b.data)


__all__ = [
    "add",
    "AddSpec",
    "AddResult",
    "generate_add_timeloop_inputs",
    "parse_add_result",
    "simulate_add",
]
