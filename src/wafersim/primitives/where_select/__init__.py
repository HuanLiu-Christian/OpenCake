"""where/select primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``where(condition, a, b)`` is REAL, implemented on
    top of ``wafersim.primitives.tensor.Tensor`` (real numpy ``where``, no
    Docker/TimeLoop involved -- see ``Tensor``'s own module docstring).
  - Real cycle/energy/area COST MODEL for one PE's local elementwise
    select: REAL, Docker-verified (real `run_bridge_check2.py` run:
    8-element ``where`` call -> ``real_bitwise_count=24``, real
    ``Cycles: 24`` in ``resolved.stats.txt`` -- exactly 3x the element
    count, matching the XOR/AND/XOR identity's exact firing count),
    implemented in ``_timeloop_where.py`` -- ``WhereSpec``/
    ``WhereResult``/``simulate_where`` re-exported below. Composes a real
    2:1 mux out of 3 real, sequential firings per element of the real,
    separately-characterized ``bitwise`` Accelergy component (an EXACT
    XOR/AND/XOR boolean identity, not an approximation). See
    ``wafersim.primitives._bitwise``'s module docstring's "ROUND-1 REAL
    DOCKER RESULT" section for the full real-output citation across all
    5 primitives that share this component pipeline.

The two layers stay independent, bridged only explicitly by the caller --
see ``wafersim.primitives.sum``'s package docstring for the exact bridging
pattern (identical here with ``WhereSpec``/``simulate_where``, using the
real elementwise count of the ``where()`` call).

Used for: causal attention masking.
"""

from __future__ import annotations

import numpy as np

from ._timeloop_where import WhereResult, WhereSpec, simulate_where
from ..tensor import Tensor


def where(condition: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Real elementwise select: ``a`` where ``condition`` is nonzero,
    ``b`` elsewhere. ``condition`` is a real numeric Tensor (this project's
    ``Tensor`` has no native bool dtype -- see its class docstring), so any
    nonzero value counts as true, matching this project's own convention
    of "attention masks encoded as 0/1 floats" (see
    ``wafersim.primitives.tensor``'s module docstring)."""
    if a.dtype != b.dtype:
        raise TypeError(
            f"where(): dtype mismatch a.dtype={a.dtype} vs b.dtype={b.dtype} "
            f"-- cast one operand explicitly via "
            f"wafersim.primitives.cast.cast(...) first."
        )
    return Tensor(np.where(condition.data != 0, a.data, b.data))


__all__ = ["where", "WhereSpec", "WhereResult", "simulate_where"]
