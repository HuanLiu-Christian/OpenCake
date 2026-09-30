"""tensor primitive -- the base data structure every other primitive operates on.

STATUS: REAL, implemented (no longer a stub). ``Tensor`` wraps a real,
already-known ``numpy.ndarray`` -- exactly the same "real numpy data, no
fabricated values" convention this project already uses in
``wafer/functional.py`` (its own module docstring: "This module is the
numerical counterpart [to the timing model] ... with real numpy matrices,
and checks the reconstructed C against a reference A @ B"). ``Tensor`` is
that same convention promoted to a reusable type, so every primitive's
*numeric* function (``add(a, b)``, ``reshape(tensor, shape)``, ``matmul``'s
own numeric path, ...) can be built on one real, shared data structure
instead of each one inventing its own ad hoc real-array wrapper.

Does this need Docker/TimeLoop? No -- and this is a real, structural
answer, not a simplification:

  - TimeLoop/Accelergy (``simulate_add``/``simulate_mul``/``simulate_div``/
    ``simulate_sum``/``simulate_max``/``simulate_gemm``, all already real
    and Docker-verified elsewhere under ``wafersim/primitives/*``) exist
    to answer ONE question: "if real hardware executed this operation, how
    many cycles/how much energy/how much area would it cost?". Answering
    that question requires a real Docker container running the real
    `timeloop-mapper`/`timeloop-model`/Accelergy tools against a real
    architecture description -- there is no shortcut, per this project's
    standing rule against synthetic/local cost numbers.
  - Building a ``Tensor`` and computing what the numeric result of an
    operation actually IS (e.g. "what are the 4 numbers produced by
    element-wise adding these two real length-4 arrays") is a completely
    different question -- ordinary arithmetic on real data, answered by
    real ``numpy``, the same way ``wafer/functional.py`` already answers
    "does the SUMMA tiling schedule reconstruct the real A @ B product"
    with plain numpy, never Docker. TimeLoop has no numeric-values API at
    all -- it only ever reports timing/energy/area for a given operation
    COUNT (e.g. AddSpec.elements_per_pe), never operates on real tensor
    contents, so there would be nothing for it to do here even if this
    module wanted to call it.

So: ``Tensor`` itself, and every future numeric-compute implementation
built on top of it (filling in the ``add(a, b)``/``reshape(tensor, shape)``/
etc. stubs elsewhere under ``wafersim/primitives/``), stays entirely in
real numpy and never touches Docker. The two layers stay independent and
are only bridged BY THE CALLER, on purpose, when both numbers are wanted
at once -- e.g. compute a real ``Tensor`` result with ``add(a, b)``, then
separately ask "what would that cost on real hardware" via
``simulate_add(AddSpec(elements_per_pe=a.size, bytes_per_element=
a.bytes_per_element), wafer, out_dir)``. Neither call depends on the
other; TimeLoop is invoked only when a real hardware cost number is
actually wanted, exactly as for every other primitive in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Real numpy dtype "kinds" this project accepts: f=float, i=signed int,
# u=unsigned int. Excludes complex/bool/object/string -- this project only
# ever models real floating/integer tensor data (matmul operands, RMSNorm
# scale factors, attention masks encoded as 0/1 floats, etc.).
_REAL_DTYPE_KINDS = "fiu"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Tensor:
    """A real, numpy-backed tensor.

    ``data`` must already be a real ``numpy.ndarray`` holding real numeric
    values -- this class never fabricates or approximates tensor contents,
    matching this project's standing rule against synthetic numbers (that
    rule is usually stated for TimeLoop/Accelergy cost numbers, but the
    same discipline applies here: a Tensor's values are either real input
    data or the real output of a real numpy computation, never a stand-in).

    ``eq``/``repr`` are both disabled on the generated dataclass methods on
    purpose: the default dataclass ``__eq__`` would compare the ``data``
    field with plain ``==``, which for a >1-element numpy array raises
    ``ValueError: truth value of an array ... is ambiguous`` instead of
    returning a bool -- use ``.allclose()`` below for real numeric
    comparison. The default ``__repr__`` would print the entire array,
    which is unusable for anything but a tiny tensor -- a custom, shape/
    dtype-only ``__repr__`` is defined below instead.
    """

    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"Tensor.data must be a real numpy.ndarray, got "
                f"{type(self.data).__name__} -- use Tensor.from_values(...) "
                f"to wrap a Python list/scalar as real array data instead "
                f"of constructing Tensor(...) directly with non-array data."
            )
        if self.data.dtype.kind not in _REAL_DTYPE_KINDS:
            raise TypeError(
                f"Tensor.data has dtype {self.data.dtype!r}, which is not a "
                f"real numeric (floating or integer) numpy dtype -- got "
                f"kind {self.data.dtype.kind!r}, expected one of "
                f"{list(_REAL_DTYPE_KINDS)!r}"
            )
        if self.data.ndim == 0:
            raise ValueError(
                "Tensor.data must have at least 1 dimension -- for a "
                "scalar use shape (1,), e.g. Tensor.from_values([3.0])"
            )

    # ---- shape / dtype introspection --------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total real element count (product of all dims) -- this is
        exactly the number to pass as e.g. AddSpec.elements_per_pe/
        SumSpec.elements_per_pe/MaxSpec.elements_per_pe when bridging a
        real numeric result into this project's real cost model for the
        same operation."""
        return int(self.data.size)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def bytes_per_element(self) -> int:
        """Matches the `bytes_per_element` field already used by every
        real cost-model Spec in this project (GemmSpec/AddSpec/MulSpec/
        DivSpec/SumSpec/MaxSpec, all under wafersim/primitives/*), so a
        real Tensor's real dtype can be handed straight to e.g.
        AddSpec(elements_per_pe=t.size, bytes_per_element=
        t.bytes_per_element) without the caller tracking a second,
        possibly-drifting element-size number by hand."""
        return self.data.dtype.itemsize

    # ---- construction -------------------------------------------------

    @classmethod
    def from_values(cls, values: Any, dtype: Any = np.float32) -> "Tensor":
        """Wrap real, already-known numbers (a Python scalar/list/nested
        list, or an existing numpy array) as a Tensor. This is the
        ordinary way to build a Tensor from real data, e.g.
        ``Tensor.from_values([[1.0, 2.0], [3.0, 4.0]])``."""
        return cls(np.asarray(values, dtype=dtype))

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: Any = np.float32) -> "Tensor":
        return cls(np.zeros(shape, dtype=dtype))

    @classmethod
    def random(cls, shape: tuple[int, ...], *, seed: int = 0, dtype: Any = np.float32) -> "Tensor":
        """Real pseudorandom data via ``numpy``'s ``Generator`` API -- the
        same ``np.random.default_rng(seed)`` convention
        ``wafer/functional.py`` already uses for its real A/B correctness-
        check matrices, so a Tensor built this way is reproducible and
        consistent with the rest of the project (not a second, different
        random convention)."""
        rng = np.random.default_rng(seed)
        return cls(rng.standard_normal(shape).astype(dtype))

    # ---- real numeric comparison (never `==`, see class docstring) ----

    def allclose(self, other: "Tensor", *, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
        """Real element-wise numeric comparison, mirroring the exact
        ``np.allclose(..., atol=atol, rtol=rtol)`` convention
        ``wafer/functional.py``'s ``simulate_gemm_functional`` already uses
        to check its real SUMMA reconstruction against ``A @ B``."""
        if self.shape != other.shape:
            return False
        return bool(np.allclose(self.data, other.data, atol=atol, rtol=rtol))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "bytes_per_element": self.bytes_per_element,
        }

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


__all__ = ["Tensor"]
