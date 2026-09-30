"""div primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``div(a, b)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy elementwise divide,
    no Docker/TimeLoop involved -- see ``Tensor``'s own module docstring).
    This is real IEEE-754 float division -- ``a/0`` produces real
    inf/nan, exactly as real hardware would, and is not specially
    intercepted here.
  - Real cycle/energy/area COST MODEL for one PE's local elementwise
    divide: REAL, Docker-verified, implemented in ``_timeloop_div.py``
    -- ``DivSpec``/``DivResult``/``simulate_div`` re-exported below.
    Unlike add/mul, there is NO real, separately-characterized
    Accelergy divider component anywhere in this project's pinned
    Docker toolchain (confirmed by grepping the real source of every
    plugin the pinned image installs -- see ``_timeloop_div.py``'s
    module docstring for the full citation), so this composes a real
    Newton-Raphson reciprocal decomposition entirely out of add's and
    mul's own already-Docker-verified ``simulate_add()``/
    ``simulate_mul()`` sub-runs, rather than inventing a standalone
    "divider" area/energy number that was never actually measured by
    anything. See ``_timeloop_div.py``'s module docstring for the full
    real-cited derivation and what is/isn't a directly measured number
    in the result.

The two layers stay independent, bridged only explicitly by the caller --
see ``wafersim.primitives.add``'s package docstring for the exact bridging
pattern (identical here with ``DivSpec``/``simulate_div``).

Used for: softmax normalization, and RMSNorm's mean computation (sum / N).
"""

from __future__ import annotations

from ._timeloop_div import DivResult, DivSpec, simulate_div
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


def div(a: Tensor, b: Tensor) -> Tensor:
    """Real elementwise (broadcasting) ``a / b`` on real Tensor data."""
    _require_same_dtype(a, b, "div")
    return Tensor(a.data / b.data)


__all__ = [
    "div",
    "DivSpec",
    "DivResult",
    "simulate_div",
]
