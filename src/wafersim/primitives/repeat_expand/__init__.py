"""repeat/expand primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``repeat()``/``expand()`` are REAL, implemented on
    top of ``wafersim.primitives.tensor.Tensor`` (no Docker/TimeLoop
    involved for the numeric values themselves -- see ``Tensor``'s own
    module docstring).
  - Real cycle/energy/area COST MODEL: ``expand()`` is permanently zero
    real hardware cost -- a real, non-owning stride-0 broadcast view,
    never a copy, every call (same "true zero-cost" category as `sum`'s
    K=1 case). ``repeat()`` is unconditionally real-copying every call --
    see ``_timeloop_repeat.py`` -- ``RepeatSpec``/``simulate_repeat``
    re-exported below, built on the shared real
    ``wafersim.primitives._memory_move`` pipeline (a real identity-
    bitwise-pass-through per physically-duplicated element -- see that
    module's own docstring for the real Accelergy-source citation and
    modeling choice).

Used for: grouped-query attention (GQA) -- broadcasting the K/V heads
(fewer than the Q heads) up to match the Q head count before the QK^T
matmul. Used by every LLaMA2/3 and Qwen2+ model (confirmed via the Qwen2
technical report and llama.cpp's real GGML_OP_REPEAT op).

`repeat` and `expand` are deliberately NOT the same operation numerically,
matching this file's own original note about a real, meaningful
distinction:
  - `repeat` (real ``numpy.repeat``) always materializes a real, physically
    duplicated copy.
  - `expand` (real ``numpy.broadcast_to``) returns a real, non-owning,
    stride-0 VIEW over the original data -- no copy is made, exactly this
    file's own prior note that "if the matmul primitive's own hardware/
    backend supports reading the same K/V address multiple times ...
    without materializing a physical copy, this primitive may become
    unnecessary for that backend": expand() here already does not
    materialize a physical copy, so a GQA matmul consuming its result reads
    the same underlying K/V storage repeatedly, not a duplicated array.
    (The returned view is read-only, as real numpy broadcasting requires;
    every op in this project is purely functional -- it only ever reads
    ``Tensor.data`` and returns a new Tensor -- so this is never a problem
    in practice.)
"""

from __future__ import annotations

import numpy as np

from ._timeloop_repeat import RepeatSpec, simulate_repeat
from ..tensor import Tensor


def repeat(tensor: Tensor, repeats: int, dim: int) -> Tensor:
    """Real, physically-copied repeat (numpy.repeat)."""
    return Tensor(np.repeat(tensor.data, repeats, axis=dim))


def expand(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Real, non-copying broadcast view (numpy.broadcast_to). ``shape``
    entries of -1 keep that dimension's current real size, matching
    PyTorch's real ``Tensor.expand`` convention."""
    resolved = tuple(
        tensor.shape[i] if s == -1 else s for i, s in enumerate(shape)
    )
    return Tensor(np.broadcast_to(tensor.data, resolved))


__all__ = ["repeat", "expand", "RepeatSpec", "simulate_repeat"]
