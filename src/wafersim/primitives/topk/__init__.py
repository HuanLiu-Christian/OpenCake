"""topk primitive.

STATUS: fully split and REAL on both sides:
  - Numeric compute: ``topk(x, k, dim)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy stable argsort, no
    Docker/TimeLoop involved). Returns ``(values, indices)`` sorted
    descending along ``dim`` (matching ``torch.topk(..., sorted=True)``).
    Ties break toward the LOWER index (stable), which is deterministic;
    ``torch.topk`` leaves tie order unspecified, so tests compare values
    exactly and indices only where scores are distinct. ``indices`` is an
    int64 Tensor -- it is routing data, not a float activation.
  - Real cycle/energy/area COST MODEL for one PE's local top-k:
    REAL, Docker-backed, implemented in ``_timeloop_topk.py`` --
    ``TopKSpec``/``TopKResult``/``simulate_topk`` re-exported below.
    Composed from the real ``comparator`` and ``bitwise`` components via
    k rounds of max-with-index + winner masking; see that module's
    docstring for the exact counts and the named modeling choices.

Bridging pattern (caller-side, as for every other primitive): after
``values, indices = topk(scores, k)`` with ``scores.shape == (rows, E)``,
cost it with ``TopKSpec(rows_per_pe=rows, num_elements=E, k=k,
bytes_per_element=scores.bytes_per_element)`` and ``simulate_topk``.

Used for: gpt-oss's MoE router (``torch.topk(gate_logits, k=4)`` over 128
experts). The ``indices`` it returns drive the expert-grouped GEMM shapes.
"""

from __future__ import annotations

import numpy as np

from ._timeloop_topk import TopKResult, TopKSpec, simulate_topk
from ..tensor import Tensor


def topk(x: Tensor, k: int, dim: int = -1) -> tuple[Tensor, Tensor]:
    """Real top-``k`` along ``dim``: returns ``(values, indices)``, both of
    shape ``x.shape`` with ``dim`` replaced by ``k``, sorted descending."""
    n = x.shape[dim]
    if not 1 <= k <= n:
        raise ValueError(f"topk(): k={k} must be in [1, {n}] for dim={dim} of shape {x.shape}")
    order = np.argsort(-x.data, axis=dim, kind="stable")
    idx = np.take(order, np.arange(k), axis=dim)
    vals = np.take_along_axis(x.data, idx, axis=dim)
    return Tensor(vals), Tensor(idx.astype(np.int64))


__all__ = ["topk", "TopKSpec", "TopKResult", "simulate_topk"]
