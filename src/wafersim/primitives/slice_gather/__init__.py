"""slice/gather primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``slice_()``/``gather()`` are REAL, implemented on
    top of ``wafersim.primitives.tensor.Tensor`` (no Docker/TimeLoop
    involved for the numeric values themselves -- see ``Tensor``'s own
    module docstring).
  - Real cycle/energy/area COST MODEL: ``slice_()`` is permanently zero
    real hardware cost -- a real, non-copying basic-indexing view, every
    call (same "true zero-cost" category as `sum`'s K=1 case). ``gather()``
    is unconditionally real-copying every call -- see
    ``_timeloop_gather.py`` -- ``GatherSpec``/``simulate_gather``
    re-exported below, built on the shared real
    ``wafersim.primitives._memory_move`` pipeline (a real identity-
    bitwise-pass-through per gathered element -- see that module's own
    docstring for the real Accelergy-source citation and modeling
    choice).

Used for: token embedding lookup (gather rows of the embedding table by
token id -- GGML's equivalent is `GET_ROWS`), and splitting a tensor in half
for RoPE's rotate_half step.

`slice_` (real numpy basic indexing) returns a real, non-copying VIEW --
splitting a tensor for RoPE's rotate_half is free, matching real hardware
behavior for a contiguous-range slice. `gather` (real numpy fancy indexing
via ``numpy.take``) always materializes a real, physically new array --
an embedding-table lookup is a real, non-contiguous memory gather, exactly
matching what a real embedding lookup costs on real hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._timeloop_gather import GatherSpec, simulate_gather
from ..tensor import Tensor


def slice_(tensor: Tensor, dim: int, start: int, end: int) -> Tensor:
    """Real, non-copying range slice (numpy basic indexing -- a view)."""
    index: list[Any] = [slice(None)] * tensor.ndim
    index[dim] = slice(start, end)
    return Tensor(tensor.data[tuple(index)])


def gather(tensor: Tensor, dim: int, index: Any) -> Tensor:
    """Real gather (numpy.take) -- always materializes a new real array,
    matching real embedding-lookup/index-select behavior."""
    return Tensor(np.take(tensor.data, np.asarray(index), axis=dim))


__all__ = ["slice_", "gather", "GatherSpec", "simulate_gather"]
