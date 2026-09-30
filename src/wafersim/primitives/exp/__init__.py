"""exp primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy ``exp``, no Docker/TimeLoop involved -- see ``Tensor``'s own
module docstring). Also has a REAL, separate, Docker-backed hardware
cost-model layer (``ExpSpec``/``ExpResult``/``simulate_exp``, see
``_timeloop_exp.py``'s own module docstring for the real range-reduction
+ degree-6-Taylor-polynomial decomposition this is built from) -- see
this package's parent ``__init__.py`` for how the two independent layers
(numeric result vs. hardware cost) are meant to be bridged explicitly by
the caller when both are wanted.

Used for: softmax.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ._timeloop_exp import ExpResult, ExpSpec, simulate_exp


def exp(x: Tensor) -> Tensor:
    """Real elementwise ``exp(x)`` on real Tensor data, dtype-preserving."""
    return Tensor(np.exp(x.data))


__all__ = ["exp", "ExpSpec", "ExpResult", "simulate_exp"]
