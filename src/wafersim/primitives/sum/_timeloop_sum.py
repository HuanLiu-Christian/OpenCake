"""Real, Docker-backed cost model for the `sum` primitive -- COMPOSED from
add's real fpadder component, not a new reduction/accumulator circuit.

WHY NO NEW ACCELERGY RESEARCH WAS NEEDED (unlike `div`): summing K real
elements down to one scalar is EXACTLY K-1 real, sequential add
operations -- an accumulator chain ``s0 = x0; s1 = s0 + x1; s2 = s1 + x2;
...; s_{K-1} = s_{K-2} + x_{K-1}``. Every one of those K-1 firings is the
exact same real, already-Docker-verified Accelergy ``fpadder`` component
`add` already uses (see ``wafersim.primitives.add._timeloop_add``'s own
module docstring for the full real-source investigation that established
it). This is an EXACT operation-count derivation from the definition of
reduction, not an approximation and not a reasoned engineering estimate
like `div`'s Newton-Raphson decomposition -- `div` needed a reasoned
per-step count because no real divider component exists anywhere in this
project's pinned Docker toolchain; `sum` needs none of that, because a
K-element reduction is *by definition* K-1 adds, full stop.

MODELING CHOICE (confirmed with the user before implementing this file):
the K-1 real adds are modeled as a strictly SEQUENTIAL accumulator chain,
matching `div`'s own "no pipelining/overlap assumed" convention -- NOT a
parallel/tree reduction (which would need only ceil(log2(K)) sequential
depth instead of K-1, and a different TimeLoop mapping this project has
no infrastructure for yet). This is a conservative assumption (it likely
OVERESTIMATES real hardware latency/energy if a tree reduction is
actually available on the target hardware), not a fabricated number --
named here explicitly, not buried, so it can be revisited later if this
project ever adds a tree-reduction mapping.

DEGENERATE CASE (K=1): summing a single element takes zero real additions
-- there is nothing to accumulate. This module returns a true zero-cost
result WITHOUT launching a real Docker run in that case (see
``simulate_sum()`` below) -- this is not a fabricated number, it is the
honest structural answer (no real fpadder instantiation is required to
"sum" one element). Correspondingly, ``SumResult.adder_area_um2`` and
``.pe_scratchpad_area_um2`` are reported as ``None`` (not 0.0, and not
any other placeholder) in that case, since no real component was ever
instantiated for the call -- see ``SumResult`` below.

WHAT IS AND ISN'T REAL IN THE RESULTING NUMBERS: every add firing's
cycles/energy/area comes from a real Docker ``timeloop-model`` run
against the real, separately-characterized ``fpadder`` component (the
same one `add` already verified) -- nothing per-firing is invented, and
unlike `div`, the FIRING COUNT itself (K-1) is also exact, not a reasoned
estimate. The only modeling choice this file makes at all is the
sequential-chain (vs. tree) execution assumption above.

SCOPE: same as add -- one PE's local reduce-over-N cost only (used for
softmax's normalization denominator and RMSNorm's mean-of-squares
reduction, per this primitive's package __init__.py), no fabric/network
component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...specs import WaferSpec
from ..add import AddResult, AddSpec, simulate_add


@dataclass(frozen=True, slots=True)
class SumSpec:
    """Reduce ``elements_per_pe`` real elements ``x[0..K-1]`` to one
    scalar on ONE PE, via a real, sequential accumulator chain of
    ``real_add_count`` real `add` firings (see module docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")

    @property
    def real_add_count(self) -> int:
        """Exact number of real fpadder firings a K-element reduction
        requires: K-1 (0 when K=1 -- summing one element needs no add).
        Not an estimate -- this is the definition of reduction."""
        return max(self.elements_per_pe - 1, 0)


@dataclass(frozen=True, slots=True)
class SumResult:
    """Real, Docker-verified per-PE cost of one ``sum`` primitive call.

    ``add_result`` is the real ``simulate_add()`` sub-run covering ALL of
    ``sum.real_add_count``'s accumulator firings in ONE Docker
    round-trip (not one round-trip per firing) -- ``None`` only in the
    K=1 degenerate case, where zero real adds are needed and no Docker
    run is launched at all (see module docstring's "DEGENERATE CASE"
    section).
    """

    sum: SumSpec
    add_result: Optional[AddResult]

    @property
    def local_compute_cycles(self) -> int:
        return 0 if self.add_result is None else self.add_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return 0.0 if self.add_result is None else self.add_result.local_energy_uj

    @property
    def adder_area_um2(self) -> Optional[float]:
        """Real area of the fpadder this reduction's accumulator chain
        uses (from the real add sub-run's ART.yaml) -- ``None`` in the
        K=1 case, since no real adder is instantiated for a no-op
        reduction (see module docstring)."""
        return None if self.add_result is None else self.add_result.adder_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> Optional[float]:
        return None if self.add_result is None else self.add_result.pe_scratchpad_area_um2

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.sum.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.sum.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "sum": {
                "elements_per_pe": self.sum.elements_per_pe,
                "bytes_per_element": self.sum.bytes_per_element,
            },
            "real_add_count": self.sum.real_add_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "adder_area_um2": self.adder_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "add_result": None if self.add_result is None else self.add_result.to_dict(),
        }


def simulate_sum(sum_spec: SumSpec, wafer: WaferSpec, out_dir: Path) -> SumResult:
    """Run ONE real Docker ``timeloop-model`` pipeline (via
    ``simulate_add()``) covering all of ``sum_spec.real_add_count``'s
    real, sequential accumulator-chain firings a K-element reduction
    requires -- or, when K=1, skip Docker entirely and return the true
    zero-cost result (see module docstring's "DEGENERATE CASE" section).
    Raises (never fabricates a result) if the real Docker sub-run fails
    -- same discipline as add/mul/div.
    """
    sum_spec.validate()
    real_add_count = sum_spec.real_add_count

    if real_add_count == 0:
        return SumResult(sum=sum_spec, add_result=None)

    add_spec = AddSpec(
        elements_per_pe=real_add_count, bytes_per_element=sum_spec.bytes_per_element
    )
    add_result = simulate_add(add_spec, wafer, out_dir / "add")
    return SumResult(sum=sum_spec, add_result=add_result)
