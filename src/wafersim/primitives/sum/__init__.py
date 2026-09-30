"""sum primitive.

STATUS: fully split and REAL on both sides now --
  - Numeric compute: ``sum(x, dim)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy reduction, no
    Docker/TimeLoop involved -- see ``Tensor``'s own module docstring).
    Reduces with ``keepdims=True`` -- the shape stays broadcastable
    against the original tensor, matching how this reduction is actually
    consumed downstream (softmax's ``x / sum(exp(x), dim)`` needs the
    denominator to broadcast back against ``exp(x)``'s own shape).
  - Real cycle/energy/area COST MODEL for one PE's LOCAL reduction (all K
    elements already on one PE): REAL, Docker-verified, implemented in
    ``_timeloop_sum.py`` -- ``SumSpec``/``SumResult``/``simulate_sum``
    re-exported below. Unlike `div`, this needed NO new Accelergy
    component research: reducing K elements is EXACTLY K-1 real,
    sequential firings of add's own already-Docker-verified ``fpadder``
    component (an exact operation count from the definition of
    reduction, not a reasoned estimate) -- see ``_timeloop_sum.py``'s
    module docstring for the full derivation and the "sequential, not
    tree, reduction" modeling choice.
  - Real cycle/energy/area COST MODEL for a CROSS-PE reduction (the K
    elements are sharded across multiple physical PEs, e.g. a
    tensor-parallel RMSNorm/softmax reduction along a split dimension):
    REAL Docker (local reduce + one representative combine-add firing)
    + REAL network simulation (the exact same ``PacketNetwork._send_hop``
    link/FIFO timing GEMM's own cross-PE broadcast uses) + OpenCake's own
    analytical network/static energy and interconnect-area coefficients
    -- implemented in ``_cross_pe_sum.py`` -- ``SumCrossPESpec``/
    ``SumCrossPEResult``/``simulate_sum_cross_pe`` re-exported below.
    Modeled as a sequential relay chain along one mesh row (consistent
    with local sum's own "sequential, not tree" modeling choice), a real,
    named mechanism confirmed to exist on real hardware via Cerebras's
    own official ``collectives_2d.reduce_fadds`` SDK primitive and
    WaferLLM's (arXiv 2502.04563) description of allreduce-based
    RMSNorm/Softmax -- see ``_cross_pe_sum.py``'s module docstring for
    the full citation and the explicit real/simulated/analytical
    breakdown of what each part of the cost is.

The two layers stay independent, bridged only explicitly by the caller.
For one real LOCAL reduction lane along ``dim`` (all K elements on one
PE), the bridge is
``SumSpec(elements_per_pe=x.shape[dim], bytes_per_element=x.bytes_per_element)``
-- ``x.shape[dim]`` (the K elements reduced), not ``x.size`` (which would
count every independent lane too, e.g. every row of a batch). For a
CROSS-PE reduction lane split across ``num_pes`` physical PEs, use
``SumCrossPESpec(total_elements=x.shape[dim], num_pes=num_pes,
bytes_per_element=x.bytes_per_element)`` with ``simulate_sum_cross_pe``
instead.

Used for: softmax's normalization denominator, and RMSNorm's mean-of-squares
reduction -- locally when the reduced dimension lives entirely on one PE,
cross-PE (via ``SumCrossPESpec``/``simulate_sum_cross_pe``) when a
tensor-parallel layout has split that dimension across the mesh.
"""

from __future__ import annotations

import numpy as np

from ._cross_pe_sum import SumCrossPEResult, SumCrossPESpec, simulate_sum_cross_pe
from ._timeloop_sum import SumResult, SumSpec, simulate_sum
from ..tensor import Tensor


def sum(x: Tensor, dim: int) -> Tensor:
    """Real reduction: ``sum(x.data, axis=dim, keepdims=True)`` on real
    Tensor data."""
    return Tensor(np.sum(x.data, axis=dim, keepdims=True))


__all__ = [
    "sum",
    "SumSpec",
    "SumResult",
    "simulate_sum",
    "SumCrossPESpec",
    "SumCrossPEResult",
    "simulate_sum_cross_pe",
]
