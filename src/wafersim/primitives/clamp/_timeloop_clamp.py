"""Real, Docker-backed cost model for the `clamp` primitive -- COMPOSED from
two already-Docker-verified real components, not a new component:

  - the real ``comparator`` (see ``wafersim.primitives.max._timeloop_max``'s
    own module docstring for the real ``accelergy-aladdin-plug-in`` source
    citation), and
  - the real ``bitwise`` gate (see ``wafersim.primitives._bitwise``), used
    exactly the way ``where_select`` uses it, to build an EXACT 2:1 mux.

DECOMPOSITION (exact operation counts, not estimates). A one-sided clamp
``min(x, hi)`` (or ``max(x, lo)``) on one element is, in hardware terms,
"compare x against the bound, then select x or the bound by that result":

    flag = comparator(x, bound)                 # 1 real comparator firing
    out  = mux(flag, x, bound)                  # 3 real bitwise firings
                                                # (XOR/AND/XOR identity,
                                                #  see where_select)

A two-sided clamp ``min(max(x, lo), hi)`` is two such stages in sequence.
So for ``n_bounds`` bounds (1 or 2) over ``elements_per_pe`` elements:

    real_compare_count = n_bounds * elements_per_pe
    real_bitwise_count = 3 * n_bounds * elements_per_pe

MODELING CHOICES (named, same discipline as `sum`/`max`/`where`):
  - Strictly SEQUENTIAL firings, no overlap between comparator and bitwise
    stages or across elements -- conservative (likely an OVERESTIMATE of a
    real pipelined ``fmin``/``fmax`` datapath), not fabricated.
  - The two real Docker sub-runs (comparator, bitwise) are added: cycles
    and energy sum, areas are reported separately (a clamp unit physically
    contains both circuits).
  - The bound is a scalar held in the PE's own scratchpad -- no extra
    memory-move cost is charged for it (same simplification `mul`'s scalar
    scaling has elsewhere in this project).

SCOPE: one PE's local elementwise clamp over N elements only -- no fabric
or network component, same as add/sum/max/where.

Used for: gpt-oss's SwiGLU input clamp (``x_glu`` is clamped on its upper
side only, ``x_linear`` on both sides, limit = 7.0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ...specs import WaferSpec
from .._bitwise import BitwiseResult, BitwiseSpec, simulate_bitwise
from ..max._timeloop_max import _CompareResult, _CompareSpec, _simulate_compare

_GATES_PER_MUX = 3  # exact XOR/AND/XOR 2:1 mux identity (see where_select)


@dataclass(frozen=True, slots=True)
class ClampSpec:
    """Clamp ``elements_per_pe`` real elements against ``num_bounds`` scalar
    bounds (1 = one-sided, 2 = two-sided) on ONE PE."""

    elements_per_pe: int
    num_bounds: int = 1
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")
        if self.num_bounds not in (1, 2):
            raise ValueError("num_bounds must be 1 (one-sided) or 2 (two-sided)")

    @property
    def real_compare_count(self) -> int:
        """One real comparator firing per element per bound. Exact."""
        return self.num_bounds * self.elements_per_pe

    @property
    def real_bitwise_count(self) -> int:
        """One 3-gate XOR/AND/XOR mux per element per bound. Exact."""
        return _GATES_PER_MUX * self.num_bounds * self.elements_per_pe


@dataclass(frozen=True, slots=True)
class ClampResult:
    """Real, Docker-verified per-PE cost of one `clamp` primitive call:
    the comparator sub-run plus the mux (bitwise) sub-run, each ONE Docker
    round-trip."""

    clamp: ClampSpec
    compare_result: _CompareResult
    bitwise_result: BitwiseResult

    @property
    def local_compute_cycles(self) -> int:
        return self.compare_result.local_compute_cycles + self.bitwise_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return self.compare_result.local_energy_uj + self.bitwise_result.local_energy_uj

    @property
    def comparator_area_um2(self) -> float:
        return self.compare_result.comparator_area_um2

    @property
    def bitwise_area_um2(self) -> float:
        return self.bitwise_result.bitwise_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> float:
        return max(
            self.compare_result.pe_scratchpad_area_um2,
            self.bitwise_result.pe_scratchpad_area_um2,
        )

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.clamp.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.clamp.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "clamp": asdict(self.clamp),
            "real_compare_count": self.clamp.real_compare_count,
            "real_bitwise_count": self.clamp.real_bitwise_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "comparator_area_um2": self.comparator_area_um2,
            "bitwise_area_um2": self.bitwise_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
        }


def simulate_clamp(clamp: ClampSpec, wafer: WaferSpec, out_dir: Path) -> ClampResult:
    """Run TWO real Docker ``timeloop-model`` pipelines (comparator firings,
    then mux bitwise firings) for one clamp call. Raises (never fabricates a
    result) if either real sub-run fails -- same discipline as add/max/where."""
    clamp.validate()
    compare_result = _simulate_compare(
        _CompareSpec(
            elements_per_pe=clamp.real_compare_count,
            bytes_per_element=clamp.bytes_per_element,
        ),
        wafer,
        out_dir / "compare",
    )
    bitwise_result = simulate_bitwise(
        BitwiseSpec(
            elements_per_pe=clamp.real_bitwise_count,
            bytes_per_element=clamp.bytes_per_element,
        ),
        wafer,
        out_dir / "bitwise",
    )
    return ClampResult(clamp=clamp, compare_result=compare_result, bitwise_result=bitwise_result)
