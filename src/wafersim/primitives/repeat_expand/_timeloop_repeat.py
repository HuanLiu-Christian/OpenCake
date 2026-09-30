"""Real, Docker-backed cost model for the `repeat` primitive (the
always-copying branch of this package -- ``expand()`` stays permanently
zero real cost, see ``repeat_expand/__init__.py``'s package docstring) --
COMPOSED from the shared real ``_memory_move`` pipeline (see
``wafersim.primitives._memory_move``'s own module docstring for the real
Accelergy-source citation and the "identity bitwise pass-through"
modeling choice this reuses).

Unlike ``reshape()`` (see ``reshape_view._timeloop_reshape``), `repeat`'s
real copy outcome is NOT conditional -- ``numpy.repeat`` (this project's
own real implementation) always materializes a genuinely new, physically
duplicated array, every call, so ``RepeatSpec`` needs no per-call
copied-or-not check; ``elements_per_pe`` is simply the real repeated
output's element count (every one of those output elements is a real
write, even the physically-duplicated ones -- matching this primitive's
own package docstring: "appending is NOT free ... a real memory write,
not a metadata-only operation").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...specs import WaferSpec
from .._memory_move import MemoryMoveResult, MemoryMoveSpec, simulate_memory_move
from ..tensor import Tensor


@dataclass(frozen=True, slots=True)
class RepeatSpec:
    """A specific ``repeat(tensor, repeats, dim)`` call's real memory-move
    cost inputs -- always a real copy (see module docstring)."""

    elements_per_pe: int
    bytes_per_element: int

    @classmethod
    def from_result(cls, result: Tensor) -> "RepeatSpec":
        """Derive from the REAL ``repeat()`` output tensor -- its real
        ``.size`` is exactly the number of elements physically written."""
        return cls(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)

    def to_memory_move_spec(self) -> MemoryMoveSpec:
        return MemoryMoveSpec(
            elements_per_pe=self.elements_per_pe, bytes_per_element=self.bytes_per_element
        )


def simulate_repeat(repeat: RepeatSpec, wafer: WaferSpec, out_dir: Path) -> MemoryMoveResult:
    """Run the real Docker ``timeloop-model`` pipeline for this repeat
    call's real memory-move cost. Raises (never fabricates a result) if
    the real Docker sub-run fails."""
    return simulate_memory_move(repeat.to_memory_move_spec(), wafer, out_dir)
