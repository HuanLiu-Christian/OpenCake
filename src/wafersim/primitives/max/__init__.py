"""max primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``max(x, dim)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy reduction, no
    Docker/TimeLoop involved -- see ``Tensor``'s own module docstring).
    Reduces with ``keepdims=True`` -- the shape stays broadcastable
    against the original tensor, matching how this reduction is actually
    consumed downstream (softmax's ``x - max(x, dim)`` needs the result to
    broadcast back against ``x``'s own shape).
  - Real cycle/energy/area COST MODEL for one PE's LOCAL max-reduction
    (all K elements already on one PE): REAL, Docker-verified,
    implemented in ``_timeloop_max.py`` -- ``MaxSpec``/``MaxResult``/
    ``simulate_max`` re-exported below. Unlike `sum` (which reuses
    `add`'s already-built pipeline), `max` needed a NEW real Accelergy
    component -- a genuine, separately-characterized ``comparator``
    (confirmed via the real upstream ``accelergy-aladdin-plug-in``
    source: real ``data/comparator.csv``, NOT `add`'s ``fpadder``/`mul`'s
    ``fpmultiplier`` reused as an approximation) -- but like `sum`,
    reducing K elements to their max is EXACTLY K-1 real, sequential
    comparator firings (an exact operation count from the definition of
    reduction, not a reasoned estimate like `div`'s Newton-Raphson
    decomposition). See ``_timeloop_max.py``'s module docstring for the
    full real-source citation and the "sequential, not tree, reduction"
    modeling choice.
  - Real cycle/energy/area COST MODEL for a CROSS-PE max-reduction (the K
    elements are sharded across multiple physical PEs, e.g. a
    tensor-parallel softmax numerical-stability row-max along a split
    dimension): REAL Docker (local reduce + one representative
    combine-compare firing) + REAL network simulation (the exact same
    ``PacketNetwork._send_hop`` link/FIFO timing GEMM's own cross-PE
    broadcast uses) + OpenCake's own analytical network/static energy and
    interconnect-area coefficients -- implemented in ``_cross_pe_max.py``
    -- ``MaxCrossPESpec``/``MaxCrossPEResult``/``simulate_max_cross_pe``
    re-exported below. Modeled as the same sequential relay chain `sum`'s
    own cross-PE module uses, for consistency -- but LESS certain than
    `sum`'s: Cerebras's official SDK documents a real cross-PE sum
    primitive (``collectives_2d.reduce_fadds``) but NO cross-PE max/min
    reduction primitive at all -- a real, confirmed gap in Cerebras's own
    published API, not an oversight in this project's research. This
    module's cross-PE relay topology is therefore a REASONABLE INFERENCE
    from `sum`'s real mechanism, not a directly-documented one -- see
    ``_cross_pe_max.py``'s own module docstring for the full citation
    trail and this explicit confidence caveat.

The two layers stay independent, bridged only explicitly by the caller --
see ``wafersim.primitives.sum``'s package docstring for the exact bridging
pattern (identical here with ``MaxSpec``/``simulate_max`` for a LOCAL
reduction lane, using ``x.shape[dim]`` for ``elements_per_pe``, not
``x.size``). For a CROSS-PE reduction lane split across ``num_pes``
physical PEs, use ``MaxCrossPESpec(total_elements=x.shape[dim],
num_pes=num_pes, bytes_per_element=x.bytes_per_element)`` with
``simulate_max_cross_pe`` instead.

Used for: softmax numerical stability (subtracting the row max before exp)
-- locally when the reduced dimension lives entirely on one PE, cross-PE
(via ``MaxCrossPESpec``/``simulate_max_cross_pe``) when a tensor-parallel
layout has split that dimension across the mesh.
"""

from __future__ import annotations

import numpy as np

from ._cross_pe_max import MaxCrossPEResult, MaxCrossPESpec, simulate_max_cross_pe
from ._timeloop_max import MaxResult, MaxSpec, simulate_max
from ..tensor import Tensor


def max(x: Tensor, dim: int) -> Tensor:
    """Real reduction: ``np.max(x.data, axis=dim, keepdims=True)`` on
    real Tensor data."""
    return Tensor(np.max(x.data, axis=dim, keepdims=True))


__all__ = [
    "max",
    "MaxSpec",
    "MaxResult",
    "simulate_max",
    "MaxCrossPESpec",
    "MaxCrossPEResult",
    "simulate_max_cross_pe",
]
