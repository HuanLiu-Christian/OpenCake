"""Real, Docker-backed cost model for the `gather` primitive (the
always-copying branch of this package -- ``slice_()`` stays permanently
zero real cost, see ``slice_gather/__init__.py``'s package docstring) --
COMPOSED from the shared real ``_memory_move`` pipeline (see
``wafersim.primitives._memory_move``'s own module docstring for the real
Accelergy-source citation and the "identity bitwise pass-through"
modeling choice this reuses).

``gather`` (real ``numpy.take``, this project's own real implementation)
always materializes a genuinely new array -- every real embedding-table
lookup is a real, non-contiguous memory gather (see this primitive's own
package docstring), so ``GatherSpec`` needs no per-call copied-or-not
check, unlike ``reshape()``. ``elements_per_pe`` is the real gathered
output's element count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...specs import WaferSpec
from .._memory_move import MemoryMoveResult, MemoryMoveSpec, simulate_memory_move
from ..tensor import Tensor


@dataclass(frozen=True, slots=True)
class GatherSpec:
    """A specific ``gather(tensor, dim, index)`` call's real memory-move
    cost inputs -- always a real copy (see module docstring)."""

    elements_per_pe: int
    bytes_per_element: int

    @classmethod
    def from_result(cls, result: Tensor) -> "GatherSpec":
        """Derive from the REAL ``gather()`` output tensor -- its real
        ``.size`` is exactly the number of elements physically gathered
        and written."""
        return cls(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)

    def to_memory_move_spec(self) -> MemoryMoveSpec:
        return MemoryMoveSpec(
            elements_per_pe=self.elements_per_pe, bytes_per_element=self.bytes_per_element
        )


def simulate_gather(gather: GatherSpec, wafer: WaferSpec, out_dir: Path) -> MemoryMoveResult:
    """Run the real Docker ``timeloop-model`` pipeline for this gather
    call's real memory-move cost. Raises (never fabricates a result) if
    the real Docker sub-run fails."""
    return simulate_memory_move(gather.to_memory_move_spec(), wafer, out_dir)
