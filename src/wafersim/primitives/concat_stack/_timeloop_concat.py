"""Real, Docker-backed cost model for the `concat`/`stack` primitive --
COMPOSED from the shared real ``_memory_move`` pipeline (see
``wafersim.primitives._memory_move``'s own module docstring for the real
Accelergy-source citation and the "identity bitwise pass-through"
modeling choice this reuses).

Both ``concat()`` and ``stack()`` (real ``numpy.concatenate``/
``numpy.stack``, this project's own real implementation) always copy --
every real input element is physically written once into the new output
buffer (see this primitive's own package docstring, in particular the
real KV-cache-append use case: "appending is NOT free ... a real memory
write"). ``ConcatStackSpec`` needs no per-call copied-or-not check, and
its ``elements_per_pe`` is the real output tensor's element count, which
by construction equals the sum of every real input tensor's element count
(concat/stack exactly partition the output -- no element is written more
than once).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...specs import WaferSpec
from .._memory_move import MemoryMoveResult, MemoryMoveSpec, simulate_memory_move
from ..tensor import Tensor


@dataclass(frozen=True, slots=True)
class ConcatStackSpec:
    """A specific ``concat(tensors, dim)``/``stack(tensors, dim)`` call's
    real memory-move cost inputs -- always a real copy (see module
    docstring)."""

    elements_per_pe: int
    bytes_per_element: int

    @classmethod
    def from_result(cls, result: Tensor) -> "ConcatStackSpec":
        """Derive from the REAL ``concat()``/``stack()`` output tensor --
        its real ``.size`` is exactly the number of elements physically
        written (== the sum of every real input's element count)."""
        return cls(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)

    def to_memory_move_spec(self) -> MemoryMoveSpec:
        return MemoryMoveSpec(
            elements_per_pe=self.elements_per_pe, bytes_per_element=self.bytes_per_element
        )


def simulate_concat_stack(
    concat_stack: ConcatStackSpec, wafer: WaferSpec, out_dir: Path
) -> MemoryMoveResult:
    """Run the real Docker ``timeloop-model`` pipeline for this
    concat/stack call's real memory-move cost. Raises (never fabricates a
    result) if the real Docker sub-run fails."""
    return simulate_memory_move(concat_stack.to_memory_move_spec(), wafer, out_dir)
