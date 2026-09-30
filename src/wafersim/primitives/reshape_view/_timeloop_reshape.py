"""Real, Docker-backed cost model for the COPYING branch of the `reshape`
primitive -- COMPOSED from the shared real ``_memory_move`` pipeline (see
``wafersim.primitives._memory_move``'s own module docstring for the real
Accelergy-source citation ruling out a dedicated DMA/copy-engine
component, and the "identity bitwise pass-through" modeling choice this
reuses).

WHY THIS IS CONDITIONAL, UNLIKE EVERY OTHER MEMORY-MOVE PRIMITIVE:
``reshape()`` (see ``reshape_view/__init__.py``) is the one function in
this whole package whose real copy-vs-view outcome is NOT fixed per call
-- it depends on whether the requested shape is viewable given the real
input tensor's CURRENT memory layout (numpy's own real reshape semantics;
``view()`` in the same module already documents and enforces the same
real fact for its own, always-must-not-copy contract). ``repeat()``/
``gather()``/`concat`/`stack` are each unconditionally real-copying every
call; ``view()``/``expand()``/``slice_()`` are each unconditionally
zero-cost every call -- only ``reshape()`` genuinely varies, so only it
needs a Spec that checks the REAL outcome of a specific call rather than
assuming one direction structurally.

``ReshapeSpec.from_call()`` below performs that real check itself (via
``numpy.shares_memory`` against the real ``ndarray.reshape()`` output,
the exact same check ``view()`` already uses) -- it does not trust the
caller to say whether a copy happened, it verifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...specs import WaferSpec
from .._memory_move import MemoryMoveResult, MemoryMoveSpec, simulate_memory_move
from ..tensor import Tensor


@dataclass(frozen=True, slots=True)
class ReshapeSpec:
    """A specific ``reshape(tensor, shape)`` call's real memory-move cost
    inputs. ``copied`` and ``elements_per_pe`` are determined from the
    REAL reshape outcome (see ``from_call()``), never assumed."""

    elements_per_pe: int
    bytes_per_element: int
    copied: bool

    @classmethod
    def from_call(cls, tensor: Tensor, shape: tuple[int, ...]) -> "ReshapeSpec":
        """Perform the real ``tensor.data.reshape(shape)`` and check via
        real ``numpy.shares_memory`` whether it actually copied -- the
        same real check ``reshape_view.view()`` already uses for its own
        must-not-copy contract."""
        out = tensor.data.reshape(shape)
        copied = not np.shares_memory(out, tensor.data)
        return cls(
            elements_per_pe=out.size if copied else 0,
            bytes_per_element=tensor.bytes_per_element,
            copied=copied,
        )

    def to_memory_move_spec(self) -> MemoryMoveSpec:
        return MemoryMoveSpec(
            elements_per_pe=self.elements_per_pe, bytes_per_element=self.bytes_per_element
        )


def simulate_reshape(reshape: ReshapeSpec, wafer: WaferSpec, out_dir: Path) -> MemoryMoveResult:
    """Run the real Docker ``timeloop-model`` pipeline for this specific
    reshape call's real memory-move cost -- or, when ``reshape.copied`` is
    False (the requested shape was viewable, exactly like ``view()``),
    skip Docker entirely and return the true zero-cost result (see
    ``simulate_memory_move()``'s own 0-elements degenerate case)."""
    return simulate_memory_move(reshape.to_memory_move_spec(), wafer, out_dir)
