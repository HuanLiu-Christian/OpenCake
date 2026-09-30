"""mul primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``mul(a, b)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy elementwise multiply,
    no Docker/TimeLoop involved -- see ``Tensor``'s own module docstring).
  - Real cycle/energy COST MODEL for one PE's local elementwise
    multiply: REAL, Docker-verified, implemented in ``_timeloop_mul.py``
    -- ``MulSpec``/``MulResult``/``simulate_mul`` re-exported below.
    Built directly on add's round-9 fix (``class: mac`` / ``subclass:
    <real class>``, required for TimeLoop's native topology parser to
    recognize the compute level at all -- see add's module docstring
    for the full real-source investigation) rather than repeating that
    multi-round crash history. Uses a REAL, separately-characterized
    Accelergy ``fpmultiplier`` component (confirmed by reading
    ``accelergy-aladdin-plug-in``'s actual source -- see
    ``_timeloop_mul.py``'s module docstring for the full citation), not
    GEMM's ``intmac`` or add's ``fpadder`` reused as an approximation.

The two layers stay independent, bridged only explicitly by the caller --
see ``wafersim.primitives.add``'s package docstring for the exact bridging
pattern (identical here with ``MulSpec``/``simulate_mul``).

Used for: RMSNorm scaling, SwiGLU gating (elementwise product of gate and up
projections), and multiplying by a causal mask.
"""

from __future__ import annotations

from ._timeloop_mul import MulResult, MulSpec, generate_mul_timeloop_inputs, parse_mul_result, simulate_mul
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


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Real elementwise (broadcasting) ``a * b`` on real Tensor data."""
    _require_same_dtype(a, b, "mul")
    return Tensor(a.data * b.data)


__all__ = [
    "mul",
    "MulSpec",
    "MulResult",
    "generate_mul_timeloop_inputs",
    "parse_mul_result",
    "simulate_mul",
]
