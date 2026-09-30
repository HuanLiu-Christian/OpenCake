"""cos primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy ``cos``, no Docker/TimeLoop involved -- see ``Tensor``'s own
module docstring). Also has a REAL, separate, Docker-backed hardware
cost-model layer (``CosSpec``/``CosResult``/``simulate_cos``, see
``_timeloop_cos.py``'s own module docstring for why this reuses `sin`'s
real decomposition via the ``cos(x) = sin(x + pi/2)`` identity rather
than an independent polynomial) -- see this package's parent
``__init__.py`` for how the two independent layers (numeric result vs.
hardware cost) are meant to be bridged explicitly by the caller when both
are wanted.

Used for: RoPE's rotary embedding tables, alongside `sin`.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ._timeloop_cos import CosResult, CosSpec, simulate_cos


def cos(x: Tensor) -> Tensor:
    """Real elementwise ``cos(x)`` on real Tensor data, dtype-preserving."""
    return Tensor(np.cos(x.data))


__all__ = ["cos", "CosSpec", "CosResult", "simulate_cos"]
