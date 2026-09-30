"""rsqrt primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy reciprocal-sqrt, no Docker/TimeLoop involved -- see
``Tensor``'s own module docstring). Also has a REAL, separate, Docker-
backed hardware cost-model layer (``RsqrtSpec``/``RsqrtResult``/
``simulate_rsqrt``, see ``_timeloop_rsqrt.py``'s own module docstring for
the real magic-number-seeded, 3-iteration Newton-Raphson decomposition
this is built from) -- see this package's parent ``__init__.py`` for how
the two independent layers (numeric result vs. hardware cost) are meant
to be bridged explicitly by the caller when both are wanted.

Used for: RMSNorm (x * rsqrt(mean(x^2) + eps)).
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ._timeloop_rsqrt import RsqrtResult, RsqrtSpec, simulate_rsqrt


def rsqrt(x: Tensor) -> Tensor:
    """Real elementwise ``1 / sqrt(x)`` on real Tensor data. Negative
    inputs produce real IEEE-754 nan (not specially intercepted) --
    callers computing RMSNorm are expected to add a real ``eps`` to a
    real non-negative mean-of-squares before calling this, exactly as
    the formula in this module's docstring shows."""
    return Tensor(1.0 / np.sqrt(x.data))


__all__ = ["rsqrt", "RsqrtSpec", "RsqrtResult", "simulate_rsqrt"]
