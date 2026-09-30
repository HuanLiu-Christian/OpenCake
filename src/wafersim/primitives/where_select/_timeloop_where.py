"""Real, Docker-backed cost model for the `where`/`select` primitive --
COMPOSED from the real ``bitwise`` component (see
``wafersim.primitives._bitwise``'s own module docstring for the real
Accelergy-source citation establishing it as a genuine, separately-
characterized digital gate), not a new component.

WHY NO NEW ACCELERGY RESEARCH WAS NEEDED (same reasoning as `sum`/`max`
reusing `add`'s ``fpadder``): a real 2:1 multiplexer -- ``out = mask ? a :
b`` -- has an EXACT, well-known boolean identity built from three
elementary bitwise gates, with NO approximation error (unlike `div`'s
Newton-Raphson, which trades real accuracy for a real per-step count):

    t1 = a XOR b
    t2 = mask AND t1
    out = b XOR t2

(Substituting: if mask is all-1s, out = b XOR (a XOR b) = a. If mask is
all-0s, out = b XOR 0 = b. Exactly `where`'s real elementwise semantics,
for every real bit pattern -- not an approximation.) This is the standard
"mux via XOR-AND-XOR" digital identity (3 two-input gates), not this
project's own invention. This project's real, pinned Docker toolchain has
no separately-characterized standalone "mux" component wired up anywhere
(see `_memory_move`'s own module docstring for the real source citations
ruling out `accelergy-neurosim-plugin`'s crossbar-coupled ``mux`` and
`accelergy-library-plug-in`'s analog-PIM component library) -- decomposing
into 3 real, already-characterized ``bitwise`` firings per element is the
honest alternative, exactly matching `div`'s own "no real divider exists,
so decompose from what does" precedent, except here the decomposition is
EXACT rather than a reasoned numerical approximation.

MODELING CHOICE (named explicitly, not buried, same discipline as `sum`'s
"sequential not tree" note): the 3 gates per element are modeled as a
strictly SEQUENTIAL chain (t1 must complete before t2, t2 before out) --
not an assumption that a real ALU could evaluate t1's XOR and part of a
pipelined AND concurrently across different elements. This is
conservative (likely an OVERESTIMATE of a real, pipelined mux's actual
latency/energy), not fabricated.

WHAT IS AND ISN'T REAL IN THE RESULTING NUMBERS: every one of the
3*elements_per_pe gate firings' cycles/energy/area comes from a real
Docker ``timeloop-model`` run against the real, separately-characterized
``bitwise`` component. The firing count (3 per element) is EXACT (from the
boolean identity above), not a reasoned estimate -- same category as
`sum`'s K-1, not `div`'s Newton-Raphson step count.

SCOPE: same as add/sum/max -- one PE's local elementwise-select-over-N
cost only (used for causal attention masking, per this primitive's package
__init__.py), no fabric/network component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from .._bitwise import BitwiseResult, BitwiseSpec, simulate_bitwise

# Exact gate count per element for the XOR/AND/XOR 2:1 mux identity (see
# module docstring) -- not an estimate.
_GATES_PER_ELEMENT = 3


@dataclass(frozen=True, slots=True)
class WhereSpec:
    """Select ``elements_per_pe`` real elements from ``a``/``b`` by
    ``mask`` on ONE PE, via a real, sequential chain of
    ``real_bitwise_count`` real ``bitwise`` firings (see module
    docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")

    @property
    def real_bitwise_count(self) -> int:
        """Exact number of real bitwise firings a real 2:1 mux over
        ``elements_per_pe`` elements requires: 3 per element (see module
        docstring's XOR/AND/XOR identity). Not an estimate."""
        return _GATES_PER_ELEMENT * self.elements_per_pe


@dataclass(frozen=True, slots=True)
class WhereResult:
    """Real, Docker-verified per-PE cost of one `where`/`select` primitive
    call.

    ``bitwise_result`` is the real ``simulate_bitwise()`` sub-run covering
    ALL of ``where.real_bitwise_count``'s gate firings in ONE Docker
    round-trip (not one round-trip per gate).
    """

    where: WhereSpec
    bitwise_result: BitwiseResult

    @property
    def local_compute_cycles(self) -> int:
        return self.bitwise_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return self.bitwise_result.local_energy_uj

    @property
    def bitwise_area_um2(self) -> float:
        return self.bitwise_result.bitwise_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> float:
        return self.bitwise_result.pe_scratchpad_area_um2

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.where.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.where.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "where": {
                "elements_per_pe": self.where.elements_per_pe,
                "bytes_per_element": self.where.bytes_per_element,
            },
            "real_bitwise_count": self.where.real_bitwise_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "bitwise_area_um2": self.bitwise_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "bitwise_result": self.bitwise_result.to_dict(),
        }


def simulate_where(where: WhereSpec, wafer: WaferSpec, out_dir: Path) -> WhereResult:
    """Run ONE real Docker ``timeloop-model`` pipeline (via
    ``simulate_bitwise()``) covering all of ``where.real_bitwise_count``'s
    real XOR/AND/XOR mux-decomposition gate firings. Raises (never
    fabricates a result) if the real Docker sub-run fails -- same
    discipline as add/sum/max/div."""
    where.validate()
    bitwise_spec = BitwiseSpec(
        elements_per_pe=where.real_bitwise_count, bytes_per_element=where.bytes_per_element
    )
    bitwise_result = simulate_bitwise(bitwise_spec, wafer, out_dir / "bitwise")
    return WhereResult(where=where, bitwise_result=bitwise_result)
