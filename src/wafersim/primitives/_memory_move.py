"""Shared real cost model for pure data-movement operations: the copying
branch of ``reshape()``, `concat`/`stack`, the ``repeat()`` branch of
`repeat_expand`, and `gather`. Reused by all four rather than reinventing
the same real Docker pipeline four times -- exactly the "thin composition
of an already-real sub-run" discipline this project already applies via
`sum`/`max` reusing `add`.

WHY THIS MODELS A COPY AS A REAL BITWISE IDENTITY PASS-THROUGH (one real
``bitwise`` firing per moved element, via
``wafersim.primitives._bitwise.simulate_bitwise``): this project's real,
pinned Docker toolchain (7 Accelergy estimator plugins -- confirmed by
grepping every one of their real upstream sources: accelergy-aladdin-
plug-in, accelergy-cacti-plug-in, accelergy-neurosim-plugin, accelergy-
adc-plug-in, accelergy-library-plug-in, accelergy-table-based-plug-ins,
dummy_tables) has NO separately-characterized DMA/copy-engine component
anywhere. `accelergy-neurosim-plugin`'s components (mux/adder_tree/
max_pool/shift_add) all require first building a full simulated ReRAM
crossbar to extract a number (they are peripheral-circuit byproducts of an
analog PIM array simulation, not standalone digital components), and
`accelergy-library-plug-in`'s ~30 real component CSVs are all from
published analog/PIM accelerator papers (ISAAC/RAELLA/TIMELY/FORMS/
NEWTON/PRIME -- DAC/ADC/eDRAM/register-ladder parts), with nothing
matching a general digital data-mover either. So, exactly like `div`
needed a reasoned decomposition when no real divider component existed,
moving N real elements through one PE is modeled here as N real,
sequential firings of the real ``bitwise`` component computing
``out = x AND all_ones`` -- a real, physically meaningful identity
operation (masking with an all-ones value is a genuine digital technique
for routing a value through an ALU/datapath unchanged), built entirely out
of the one real, separately-characterized digital component this
project's toolchain actually has for elementwise data passage. This is a
modeling CHOICE, not a measured "copy engine" number -- named explicitly
here, not buried, matching this project's own established discipline
(see `sum`'s "sequential not tree" note, `div`'s Newton-Raphson note).
It very likely OVERESTIMATES a real dedicated DMA engine's actual
efficiency (a real copy engine is typically cheaper per byte than routing
every element through a full ALU), so this is a conservative, not
optimistic, real-component-grounded estimate.

DEGENERATE CASE (0 elements moved): nothing to copy, zero real firings,
no Docker run -- same "true zero-cost, not fabricated" discipline `sum`
established for K=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..specs import WaferSpec
from ._bitwise import BitwiseResult, BitwiseSpec, simulate_bitwise


@dataclass(frozen=True, slots=True)
class MemoryMoveSpec:
    """Move ``elements_per_pe`` real elements through ONE PE via a real,
    sequential chain of ``real_bitwise_count`` real ``bitwise`` identity
    firings (see module docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if self.elements_per_pe < 0 or self.bytes_per_element <= 0:
            raise ValueError(
                "elements_per_pe must be >= 0 and bytes_per_element must be positive"
            )

    @property
    def real_bitwise_count(self) -> int:
        """Exact firing count: one real identity-pass firing per moved
        element -- not an estimate, this IS the modeling choice (see
        module docstring)."""
        return self.elements_per_pe


@dataclass(frozen=True, slots=True)
class MemoryMoveResult:
    """Real, Docker-verified per-PE cost of moving ``elements_per_pe``
    real elements through one PE.

    ``bitwise_result`` is the real ``simulate_bitwise()`` sub-run covering
    ALL of ``move.real_bitwise_count``'s firings in ONE Docker round-trip
    -- ``None`` only in the 0-elements degenerate case (see module
    docstring), where no real Docker run is launched at all.
    """

    move: MemoryMoveSpec
    bitwise_result: Optional[BitwiseResult]

    @property
    def local_compute_cycles(self) -> int:
        return 0 if self.bitwise_result is None else self.bitwise_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return 0.0 if self.bitwise_result is None else self.bitwise_result.local_energy_uj

    @property
    def bitwise_area_um2(self) -> Optional[float]:
        return None if self.bitwise_result is None else self.bitwise_result.bitwise_area_um2

    @property
    def pe_scratchpad_area_um2(self) -> Optional[float]:
        return None if self.bitwise_result is None else self.bitwise_result.pe_scratchpad_area_um2

    @property
    def cycles_per_element(self) -> float:
        if self.move.elements_per_pe == 0:
            return 0.0
        return self.local_compute_cycles / self.move.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        if self.move.elements_per_pe == 0:
            return 0.0
        return self.local_energy_uj / self.move.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "move": {
                "elements_per_pe": self.move.elements_per_pe,
                "bytes_per_element": self.move.bytes_per_element,
            },
            "real_bitwise_count": self.move.real_bitwise_count,
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "bitwise_area_um2": self.bitwise_area_um2,
            "pe_scratchpad_area_um2": self.pe_scratchpad_area_um2,
            "bitwise_result": None if self.bitwise_result is None else self.bitwise_result.to_dict(),
        }


def simulate_memory_move(move: MemoryMoveSpec, wafer: WaferSpec, out_dir: Path) -> MemoryMoveResult:
    """Run ONE real Docker ``timeloop-model`` pipeline (via
    ``simulate_bitwise()``) covering all of ``move.real_bitwise_count``'s
    real identity-pass firings -- or, when 0 elements move, skip Docker
    entirely and return the true zero-cost result. Raises (never
    fabricates a result) if the real Docker sub-run fails."""
    move.validate()
    count = move.real_bitwise_count

    if count == 0:
        return MemoryMoveResult(move=move, bitwise_result=None)

    bitwise_spec = BitwiseSpec(elements_per_pe=count, bytes_per_element=move.bytes_per_element)
    bitwise_result = simulate_bitwise(bitwise_spec, wafer, out_dir / "bitwise")
    return MemoryMoveResult(move=move, bitwise_result=bitwise_result)
