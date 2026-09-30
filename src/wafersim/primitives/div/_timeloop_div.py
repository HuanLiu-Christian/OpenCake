"""Real, Docker-backed cost model for the `div` primitive -- COMPOSED from
add/mul's real components, not a standalone-measured divider circuit.

REAL, DOCKER-CONFIRMED GAP THIS FILE WORKS AROUND: unlike `add` (real
Accelergy ``fpadder``) and `mul` (real Accelergy ``fpmultiplier``), there
is NO real, separately-characterized division-circuit component anywhere
in this project's actual pinned Docker toolchain. Confirmed by cloning
and grepping the real source of every one of the 7 Accelergy plugins this
project's pinned image (``timeloopaccelergy/accelergy-timeloop-
infrastructure:latest``) actually installs -- per that image repo's own
real Dockerfile (Accelergy-Project/accelergy-timeloop-infrastructure):
``accelergy-aladdin-plug-in``, ``accelergy-cacti-plug-in``,
``accelergy-table-based-plug-ins``, ``accelergy-neurosim-plug-in``,
``accelergy-library-plug-in``, ``accelergy-adc-plug-in``, plus
``accelergy`` itself. None of them contain a divider/division-circuit
table or estimator -- ``aladdin_table.py``'s real
``get_supported_components()`` (the same source add/mul's own real
components were confirmed from) lists ``regfile``/``SRAM``/``counter``/
``comparator``/``crossbar``/``wire``/``FIFO``/``bitwise``/``intadder``/
``intmultiplier``/``intmac``/``fpadder``/``fpmultiplier``/``fpmac``/
``reg`` -- no divider anywhere; the string "divi" does not appear once
in that plugin's source or its ``data/`` CSVs. The other two plugins
whose source contains "divi" at all (``accelergy-library-plug-in``'s
``accelergywrapper.py``/``scaling.py``) turned out to be incidental
(``ZeroDivisionError``, a code comment) on inspection, not a hardware
component.

CHOSEN APPROACH (explicitly picked over two other options put to the
user -- see this conversation): decompose one division into a real
Newton-Raphson reciprocal iteration, itself built ENTIRELY from real
fpmultiplier/fpadder firings this project has already Docker-verified
via ``wafersim.primitives.mul``/``wafersim.primitives.add`` -- i.e. this
module runs TWO real, independent ``simulate_mul()``/``simulate_add()``
Docker round-trips per ``simulate_div()`` call and sums their real
results, rather than inventing a standalone "divider" area/energy number
that was never actually measured by anything. This is a REAL, standard
numerical-computing algorithm (not fabricated), but the specific
iteration count below IS this project's own reasoned engineering choice
built from real, cited per-step costs -- same precedent as
``specs.py``'s ``WaferSpec.link_area_um2_per_pe`` (a composite estimate
assembled from real citations, explicitly flagged as reasoned rather
than independently published).

REAL SOURCES for the algorithm and its real per-step operation counts
(both consulted directly, not from memory):
  - Wikipedia, "Division algorithm", Newton-Raphson division section:
    iteration ``X := X + X*(1 - D*X)`` (equivalently ``X*(2 - D*X)``);
    "the number of correct digits ... roughly doubles for every
    iteration" (quadratic convergence); "A quadratic initial estimate
    plus two iterations achieves single-precision accuracy ... A linear
    initial estimate ... plus four iterations suffices for both double
    and extended" (i.e. more iterations are needed from a cheaper,
    less-accurate linear seed); the double-precision combination it
    quotes elsewhere ("this method uses 10 multiplies, 9 adds, and 2
    shifts") is for THAT (different, more expensive) precision target,
    not reused here.
  - fp32.org, "Faster Division With Newton-Raphson Approximation": a
    concrete worked linear seed ``x0 = 48/17 - 32/17*D`` (1 multiply, 1
    subtract; real-cited worst-case error bound < 1/17), and confirms
    each Newton-Raphson iteration costs "2 multiplications and 1
    subtraction".

THIS PROJECT'S DERIVED OPERATION COUNT (not itself independently
published -- assembled from the real per-step counts above): a linear
seed (error < 1/17, i.e. roughly 4-5 correct bits) needs its correct-bit
count to at least double 3 times to comfortably clear fp32's 24-bit
mantissa (4-5 -> ~9-10 -> ~18-20 -> ~36-40), so this module uses 3
Newton-Raphson iterations from the linear seed (deliberately more
conservative than Wikipedia's cited "quadratic estimate + 2 iterations"
combination, since this project uses the cheaper LINEAR seed, not a
quadratic one whose own real operation count was not found stated
explicitly in either source), plus one final numerator multiply
(``a * reciprocal(b)``):
  multiplies = 1 (seed) + 3*2 (iterations) + 1 (final) = 8 per element
  adds/subtracts = 1 (seed) + 3*1 (iterations) = 4 per element
See ``_MULS_PER_ELEMENT``/``_ADDS_PER_ELEMENT`` below.

WHAT IS AND ISN'T REAL IN THE RESULTING NUMBERS: every multiply and add
firing's cycles/energy comes from a real Docker ``timeloop-model`` run
against the real, separately-characterized ``fpmultiplier``/``fpadder``
components (same ones add/mul already verified) -- nothing per-firing is
invented. What IS this project's own engineering choice, not a directly
measured "real divider" number, is (a) how many multiply/add firings one
division decomposes into (the 8/4 counts above), and (b) that this
module composes the multiply-stage and add-stage results by simple
addition (``local_compute_cycles = mul_result.cycles + add_result.
cycles``), i.e. assumes the two stages run sequentially with zero
overlap/pipelining -- a conservative (likely-overestimates-cost)
assumption, not a real measurement of a fused divider pipeline. Both
choices are named here, not buried, so this can be revisited with a
better real source later without archaeology.

SCOPE: same as add/mul -- one PE's local divide-over-N cost only (used
for softmax normalization and RMSNorm's mean computation, per this
primitive's package __init__.py), no fabric/network component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...specs import WaferSpec
from ..add import AddResult, AddSpec, simulate_add
from ..mul import MulResult, MulSpec, simulate_mul

# See this module's docstring's "THIS PROJECT'S DERIVED OPERATION COUNT"
# section for the full real-cited derivation: a linear Newton-Raphson
# seed (fp32.org's real worked formula, error < 1/17) plus 3 iterations
# (this project's own conservative choice, not itself independently
# published) plus one final numerator multiply.
_MULS_PER_ELEMENT = 8
_ADDS_PER_ELEMENT = 4


@dataclass(frozen=True, slots=True)
class DivSpec:
    """Problem ``Z[N] = X[N] / Y[N]``, evaluated for ONE PE's local N via
    a real Newton-Raphson reciprocal decomposition (see module
    docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")


@dataclass(frozen=True, slots=True)
class DivResult:
    """Real, Docker-verified per-PE cost of one ``div`` primitive call,
    composed from a real ``simulate_mul()`` sub-run (the Newton-Raphson
    multiply firings) and a real ``simulate_add()`` sub-run (the
    subtract/combine firings) -- both kept in full below, not collapsed,
    so every real number this is built from stays inspectable."""

    div: DivSpec
    mul_result: MulResult
    add_result: AddResult

    @property
    def local_compute_cycles(self) -> int:
        """Sum of the two real sub-runs' cycles -- assumes the multiply
        and add stages run sequentially with no overlap (see module
        docstring's "WHAT IS AND ISN'T REAL" section)."""
        return self.mul_result.local_compute_cycles + self.add_result.local_compute_cycles

    @property
    def local_energy_uj(self) -> float:
        return self.mul_result.local_energy_uj + self.add_result.local_energy_uj

    @property
    def multiplier_area_um2(self) -> float:
        """Real area of the fpmultiplier this division's Newton-Raphson
        stage uses (from the real mul sub-run's ART.yaml)."""
        return self.mul_result.multiplier_area_um2

    @property
    def adder_area_um2(self) -> float:
        """Real area of the fpadder this division's subtract/combine
        stage uses (from the real add sub-run's ART.yaml)."""
        return self.add_result.adder_area_um2

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.div.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.div.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "div": {"elements_per_pe": self.div.elements_per_pe, "bytes_per_element": self.div.bytes_per_element},
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "multiplier_area_um2": self.multiplier_area_um2,
            "adder_area_um2": self.adder_area_um2,
            "pe_scratchpad_area_um2_mul_stage": self.mul_result.pe_scratchpad_area_um2,
            "pe_scratchpad_area_um2_add_stage": self.add_result.pe_scratchpad_area_um2,
            "mul_result": self.mul_result.to_dict(),
            "add_result": self.add_result.to_dict(),
        }


def simulate_div(div: DivSpec, wafer: WaferSpec, out_dir: Path) -> DivResult:
    """Run TWO real Docker ``timeloop-model`` pipelines end to end --
    ``simulate_mul()`` for the Newton-Raphson multiply firings,
    ``simulate_add()`` for the subtract/combine firings -- and compose
    their real results into one ``DivResult``. Raises (never fabricates
    a result) if either real Docker run fails -- same discipline as
    add/mul/``wafersim.timeloop``.

    Each real element of ``div.elements_per_pe`` decomposes into
    ``_MULS_PER_ELEMENT`` real multiplies and ``_ADDS_PER_ELEMENT`` real
    adds (see module docstring for the full derivation) -- so this calls
    ``simulate_mul(MulSpec(elements_per_pe=N*8))`` and
    ``simulate_add(AddSpec(elements_per_pe=N*4))``, not N calls to each.
    """
    div.validate()
    n = div.elements_per_pe

    mul_spec = MulSpec(
        elements_per_pe=n * _MULS_PER_ELEMENT, bytes_per_element=div.bytes_per_element
    )
    add_spec = AddSpec(
        elements_per_pe=n * _ADDS_PER_ELEMENT, bytes_per_element=div.bytes_per_element
    )

    mul_result = simulate_mul(mul_spec, wafer, out_dir / "mul")
    add_result = simulate_add(add_spec, wafer, out_dir / "add")

    return DivResult(div=div, mul_result=mul_result, add_result=add_result)
