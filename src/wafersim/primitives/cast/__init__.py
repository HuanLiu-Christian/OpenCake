"""cast primitive.

STATUS: REAL, implemented on top of ``wafersim.primitives.tensor.Tensor``
(real numpy ``astype``, no Docker/TimeLoop involved -- see ``Tensor``'s own
module docstring). Also has a REAL, separate, Docker-backed hardware
cost-model layer (``CastSpec``/``CastResult``/``simulate_cast``, see
``_timeloop_cast.py``'s own module docstring for the real IEEE-754
unpack/rebias/round/repack bit-manipulation decomposition this is built
from, modeling the fp32->fp16 narrowing conversion this primitive's real
use case below actually needs) -- see this package's parent
``__init__.py`` for how the two independent layers (numeric result vs.
hardware cost) are meant to be bridged explicitly by the caller when both
are wanted.

Used for: mixed-precision execution (e.g. running softmax in fp32 while the
rest of the model runs in a lower precision).
"""

from __future__ import annotations

from typing import Any

from ..tensor import Tensor
from ._timeloop_cast import CastResult, CastSpec, simulate_cast


def cast(x: Tensor, dtype: Any) -> Tensor:
    """Real dtype conversion (``numpy.ndarray.astype``) on real Tensor
    data -- e.g. ``cast(x, np.float16)``. Every other primitive in this
    project (add/mul/div/where/concat/stack/...) requires its operands to
    already share one real dtype and raises rather than silently
    up-casting -- this is the one, explicit place a mixed-precision
    pipeline is expected to insert a real conversion."""
    return Tensor(x.data.astype(dtype))


__all__ = ["cast", "CastSpec", "CastResult", "simulate_cast"]
