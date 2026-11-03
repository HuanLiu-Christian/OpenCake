"""clamp primitive.

STATUS: fully split and REAL on both sides:
  - Numeric compute: ``clamp(x, min, max)`` is REAL, implemented on top of
    ``wafersim.primitives.tensor.Tensor`` (real numpy ``maximum``/
    ``minimum``, no Docker/TimeLoop involved -- see ``Tensor``'s own module
    docstring). Semantics match ``torch.clamp``: either bound may be
    ``None`` (one-sided clamp), NaN propagates, and if ``min > max`` every
    element becomes ``max``.
  - Real cycle/energy/area COST MODEL for one PE's local elementwise clamp:
    REAL, Docker-backed, implemented in ``_timeloop_clamp.py`` --
    ``ClampSpec``/``ClampResult``/``simulate_clamp`` re-exported below.
    Composed from the real ``comparator`` (one firing per element per
    bound) and the real ``bitwise`` gate (an exact 3-gate XOR/AND/XOR mux
    per element per bound) -- see that module's docstring.

The two layers stay independent, bridged only explicitly by the caller --
see ``wafersim.primitives.sum``'s package docstring for the bridging
pattern; here use ``ClampSpec(elements_per_pe=x.size, num_bounds=1 or 2,
bytes_per_element=x.bytes_per_element)``.

Used for: gpt-oss's SwiGLU (``x_glu.clamp(max=7.0)`` one-sided,
``x_linear.clamp(-7.0, 7.0)`` two-sided).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._timeloop_clamp import ClampResult, ClampSpec, simulate_clamp
from ..tensor import Tensor


def clamp(x: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    """Real elementwise clamp of ``x`` into ``[min, max]`` (either bound
    optional, at least one required), preserving ``x``'s dtype."""
    if min is None and max is None:
        raise ValueError("clamp(): at least one of min/max must be given")
    out = x.data
    if min is not None:
        out = np.maximum(out, np.asarray(min, dtype=x.dtype))
    if max is not None:
        out = np.minimum(out, np.asarray(max, dtype=x.dtype))
    return Tensor(out.astype(x.dtype, copy=False))


__all__ = ["clamp", "ClampSpec", "ClampResult", "simulate_clamp"]
