"""reshape/view primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``reshape()``/``view()`` are REAL, implemented on top
    of ``wafersim.primitives.tensor.Tensor`` (no Docker/TimeLoop involved
    for the numeric values themselves -- see ``Tensor``'s own module
    docstring).
  - Real cycle/energy/area COST MODEL: ``view()`` is permanently zero
    real hardware cost -- it raises if a real copy would ever be needed,
    so a successful ``view()`` call by construction moved zero real data
    (same "true zero-cost, not fabricated" category as `sum`'s K=1 case).
    ``reshape()`` is CONDITIONAL -- see ``_timeloop_reshape.py`` --
    ``ReshapeSpec``/``simulate_reshape`` re-exported below, built on the
    shared real ``wafersim.primitives._memory_move`` pipeline (a real
    identity-bitwise-pass-through per copied element -- see that module's
    own docstring for the real Accelergy-source citation and modeling
    choice).

Used for: multi-head split/merge in attention, and general shape bookkeeping
throughout a LLaMA/Qwen-style forward pass. `reshape` and `view` are treated
as one primitive (view = reinterpret shape metadata without moving data;
reshape falls back to a copy only when the requested shape is not viewable).

The reshape/view distinction here is a REAL one, not just naming: numpy's
own ``ndarray.reshape`` silently copies when the requested shape is not
compatible with the array's current memory layout (e.g. reshaping a
non-contiguous view produced by an earlier ``transpose``/``permute``).
``view()`` checks that real fact (via ``numpy.shares_memory``) and raises
if a copy would actually be required, instead of silently doing one;
``reshape()`` accepts either outcome -- and ``ReshapeSpec.from_call()``
performs that same real check to decide the real cost model's outcome too
(see ``_timeloop_reshape.py``'s own module docstring).
"""

from __future__ import annotations

import numpy as np

from ._timeloop_reshape import ReshapeSpec, simulate_reshape
from ..tensor import Tensor


def reshape(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Real reshape -- may copy if the requested shape is not viewable
    without moving data (exactly numpy's own real reshape semantics)."""
    return Tensor(tensor.data.reshape(shape))


def view(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Real, no-copy reshape. Raises ValueError if the requested shape
    genuinely cannot be produced without copying real data (checked via
    ``numpy.shares_memory`` against the real output array, not assumed)."""
    out = tensor.data.reshape(shape)
    if not np.shares_memory(out, tensor.data):
        raise ValueError(
            f"view({tensor.shape} -> {shape}) would require copying data "
            f"(the requested shape is not viewable in-place given this "
            f"tensor's current real memory layout) -- use reshape() instead "
            f"if a copy is acceptable"
        )
    return Tensor(out)


__all__ = ["reshape", "view", "ReshapeSpec", "simulate_reshape"]
