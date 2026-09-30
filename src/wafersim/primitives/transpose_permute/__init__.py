"""transpose/permute primitive.

STATUS: fully split and REAL on both sides now, and CONFIRMED PERMANENTLY
ZERO real hardware cost on the cost-model side -- not "no cost model
implemented yet":
  - Numeric compute: ``permute()``/``transpose()`` are REAL, implemented
    on top of ``wafersim.primitives.tensor.Tensor`` (no Docker/TimeLoop
    involved -- see ``Tensor``'s own module docstring).
  - Real cycle/energy/area COST MODEL: real numpy ``transpose`` is
    STRUCTURALLY always a view (unlike ``reshape``, there is no real input
    layout for which axis permutation alone would need to copy -- numpy's
    own transpose is defined to only ever rewrite strides/shape metadata),
    so every real call moves zero real data, on every real input, always
    -- not merely "usually", and not conditional the way ``reshape()`` is
    (see ``wafersim.primitives.reshape_view``'s package docstring). This
    is the same "true zero-cost, not fabricated" category `sum`'s K=1
    case and ``view()``/``expand()``/``slice_()`` (see the other
    shape/memory primitives' own package docstrings) already established
    -- confirmed this session's real cross-plugin Accelergy research
    (into ``accelergy-aladdin-plug-in``/``accelergy-neurosim-plugin``/
    ``accelergy-library-plug-in``/``accelergy-adc-plug-in``) turned up no
    reason to model this any other way. No Spec/Result/simulate_* trio
    exists for this primitive -- there is no real Docker run a cost model
    here could ever meaningfully launch.

Used for: swapping/reordering tensor axes before and after attention matmuls
(e.g. moving the head dimension next to the batch dimension). `permute` is
the general case (arbitrary axis reordering); `transpose` is the common
two-axis special case, kept as an alias here.

Both real numpy ``transpose`` operations return a real VIEW (no data is
copied) -- only the strides/shape metadata changes, exactly matching real
hardware behavior where a transpose is "free" until something downstream
actually needs the data in a different memory order (e.g. a matmul that
requires contiguous operands).
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor


def permute(tensor: Tensor, dims: tuple[int, ...]) -> Tensor:
    if sorted(dims) != list(range(tensor.ndim)):
        raise ValueError(
            f"permute(dims={dims}) is not a valid permutation of this "
            f"tensor's {tensor.ndim} real dimensions {tuple(range(tensor.ndim))}"
        )
    return Tensor(np.transpose(tensor.data, axes=dims))


def transpose(tensor: Tensor, dim0: int, dim1: int) -> Tensor:
    dims = list(range(tensor.ndim))
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return Tensor(np.transpose(tensor.data, axes=dims))


__all__ = ["permute", "transpose"]
