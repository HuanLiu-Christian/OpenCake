"""silu primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy elementwise math, no Docker/TimeLoop involved -- see
``Tensor``'s own module docstring). Also has a REAL, separate, Docker-
backed hardware cost-model layer (``SiluSpec``/``SiluResult``/
``simulate_silu``, see ``_timeloop_silu.py``'s own module docstring for
how this is a pure composition of the already-real ``exp``/``div``/
``add``/``bitwise`` pipelines, needing no new numerical research of its
own) -- see this package's parent ``__init__.py`` for how the two
independent layers (numeric result vs. hardware cost) are meant to be
bridged explicitly by the caller when both are wanted.

Used for: SwiGLU's gate activation (silu(x) = x * sigmoid(x)) in the MLP
block of both LLaMA and Qwen.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor
from ._timeloop_silu import SiluResult, SiluSpec, simulate_silu


def silu(x: Tensor) -> Tensor:
    """Real ``silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`` on real
    Tensor data."""
    return Tensor(x.data / (1.0 + np.exp(-x.data)))


__all__ = ["silu", "SiluSpec", "SiluResult", "simulate_silu"]
