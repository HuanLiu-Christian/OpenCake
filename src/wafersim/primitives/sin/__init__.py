"""sin primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy ``sin``, no Docker/TimeLoop involved -- see ``Tensor``'s own
module docstring). Also has a REAL, separate, Docker-backed hardware
cost-model layer (``SinSpec``/``SinResult``/``simulate_sin``, see
``_timeloop_sin.py``'s own module docstring for the real range-reduction
+ degree-17-odd-power-Taylor-polynomial decomposition this is built
from, and for the real Cerebras CSL citation confirming a real chip
supports this operation even though its internal implementation is not
publicly disclosed) -- see this package's parent ``__init__.py`` for how
the two independent layers (numeric result vs. hardware cost) are meant
to be bridged explicitly by the caller when both are wanted.

Used for: RoPE's rotary embedding tables (previously precomputed via
plain, uncosted numpy in this project's Qwen3 composition scripts -- see
``_timeloop_sin.py``'s own module docstring for why that is also a
defensible choice, and when a composition script might instead want to
declare this as a real, costed primitive call).
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ._timeloop_sin import SinResult, SinSpec, simulate_sin


def sin(x: Tensor) -> Tensor:
    """Real elementwise ``sin(x)`` on real Tensor data, dtype-preserving."""
    return Tensor(np.sin(x.data))


__all__ = ["sin", "SinSpec", "SinResult", "simulate_sin"]
