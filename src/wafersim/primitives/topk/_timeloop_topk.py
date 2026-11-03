"""Real, Docker-backed cost model for the `topk` primitive -- COMPOSED from
the same two already-Docker-verified real components `clamp` uses (the real
``comparator`` and the real ``bitwise`` gate), not a new component.

ALGORITHM MODELED: iterative selection, ``k`` rounds per row of ``E``
elements. Round ``r`` finds the current maximum AND its index, then (for all
but the last round) masks that winner out so the next round finds the next
largest:

    max-with-index scan   : E-1 comparator firings, each followed by a
                            2:1 mux that carries the running INDEX
                            (3 bitwise firings, XOR/AND/XOR, see
                            where_select). The running VALUE select is
                            part of the comparator's own fmax datapath
                            (same convention `max` already uses).
    mask the winner       : E equality-style comparator firings
                            (position == winner) plus an E-element
                            where(mask, -inf, x), i.e. 3E bitwise firings.
                            Skipped in the last round.

Exact per-row counts:

    real_compare_count = k*(E-1) + (k-1)*E
    real_bitwise_count = 3*k*(E-1) + 3*(k-1)*E

multiplied by ``rows_per_pe``. Counts are exact FOR THIS ALGORITHM; the
algorithm itself is this project's own reasoned choice (a real chip may use
a cheaper insertion-style top-k), so treat the cost as a conservative
OVERESTIMATE -- same modeling-choice discipline as `sum`'s sequential
chain, named explicitly and not buried.

The final softmax over the k selected scores is NOT charged here: callers
cost it with the existing exp/sum/div primitives (gpt-oss softmaxes only
the k=4 values).

DEGENERATE CASE: E == 1 (then k == 1) needs no comparison at all, so no
Docker run is made and the true zero-cost result is returned -- same
precedent as `max`'s K=1 case.

SCOPE: one PE's local top-k over ``rows_per_pe`` rows only -- no fabric or
network component. Used for: gpt-oss's MoE router (E=128 experts, k=4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from .._bitwise import BitwiseResult, BitwiseSpec, simulate_bitwise
from ..max._timeloop_max import _CompareResult, _CompareSpec, _simulate_compare

_GATES_PER_MUX = 3  # exact XOR/AND/XOR 2:1 mux identity (see where_select)


@dataclass(frozen=True, slots=True)
class TopKSpec:
    """Select the top ``k`` of ``num_elements`` scores for each of
    ``rows_per_pe`` rows on ONE PE."""

    rows_per_pe: int
    num_elements: int
    k: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.rows_per_pe, self.num_elements, self.k, self.bytes_per_element) <= 0:
            raise ValueError("rows_per_pe, num_elements, k and bytes_per_element must be positive")
        if self.k > self.num_elements:
            raise ValueError(f"k={self.k} exceeds num_elements={self.num_elements}")

    @property
    def real_compare_count(self) -> int:
        e, k = self.num_elements, self.k
        return self.rows_per_pe * (k * (e - 1) + (k - 1) * e)

    @property
    def real_bitwise_count(self) -> int:
        e, k = self.num_elements, self.k
        return self.rows_per_pe * _GATES_PER_MUX * (k * (e - 1) + (k - 1) * e)


@dataclass(frozen=True, slots=True)
class TopKResult:
    """Real, Docker-verified per-PE cost of one `topk` call (comparator
    sub-run + bitwise sub-run); both ``None`` only in the E==1 degenerate
    case."""

    topk: TopKSpec
    compare_result: Optional[_CompareResult]
    bitwise_result: Optional[BitwiseResult]

    @property
    def local_compute_cycles(self) -> int:
        if self.compare_result is None or self.bitwise_result is None:
            return 0
        return self.compare_result.local_compute_cycles + self.bitwise_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        if self.compare_result is None or self.bitwise_result is None:
            return 0.0
        return self.compare_result.local_energy_uj + self.bitwise_result.local_energy_uj

    @property
    def comparator_area_um2(self) -> Optional[float]:
        return None if self.compare_result is None else self.compare_result.comparator_area_um2

    @property
    def bitwise_area_um2(self) -> Optional[float]:
        return None if self.bitwise_result is None else self.bitwise_result.bitwise_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> Optional[float]:
        if self.compare_result is None or self.bitwise_result is None:
            return None
        return max(
            self.compare_result.pe_scratchpad_area_um2,
            self.bitwise_result.pe_scratchpad_area_um2,
        )

    @property
    def cycles_per_row(self) -> float:
        return self.local_compute_cycles / self.topk.rows_per_pe

    @property
    def energy_uj_per_row(self) -> float:
        return self.local_energy_uj / self.topk.rows_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "topk": asdict(self.topk),
            "real_compare_count": self.topk.real_compare_count,
            "real_bitwise_count": self.topk.real_bitwise_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_row": self.cycles_per_row,
            "energy_uj_per_row": self.energy_uj_per_row,
            "comparator_area_um2": self.comparator_area_um2,
            "bitwise_area_um2": self.bitwise_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
        }


def simulate_topk(topk: TopKSpec, wafer: WaferSpec, out_dir: Path) -> TopKResult:
    """Run TWO real Docker ``timeloop-model`` pipelines (comparator firings,
    then index-mux/mask bitwise firings) for one topk call -- or none in the
    E==1 degenerate case. Raises (never fabricates a result) if a real
    sub-run fails."""
    topk.validate()
    if topk.real_compare_count == 0:
        return TopKResult(topk=topk, compare_result=None, bitwise_result=None)
    compare_result = _simulate_compare(
        _CompareSpec(
            elements_per_pe=topk.real_compare_count,
            bytes_per_element=topk.bytes_per_element,
        ),
        wafer,
        out_dir / "compare",
    )
    bitwise_result = simulate_bitwise(
        BitwiseSpec(
            elements_per_pe=topk.real_bitwise_count,
            bytes_per_element=topk.bytes_per_element,
        ),
        wafer,
        out_dir / "bitwise",
    )
    return TopKResult(topk=topk, compare_result=compare_result, bitwise_result=bitwise_result)
