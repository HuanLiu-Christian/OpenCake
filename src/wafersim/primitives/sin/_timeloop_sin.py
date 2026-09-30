"""Real, Docker-backed cost model for the `sin` primitive -- COMPOSED from
real mul/add/sub/bitwise firings via a real range-reduction-plus-
polynomial decomposition, same "no dedicated component exists, decompose
into what's real and already-verified" discipline `div`/`rsqrt`/`exp`
each already established.

REAL, CONFIRMED GAP: this project's pinned Docker toolchain has no
dedicated trigonometric-function circuit component anywhere (same 7-plugin
search cited in ``wafersim.primitives._bitwise``'s module docstring).

REAL CEREBRAS CONTEXT (checked this session against Cerebras' own official
SDK docs, not assumed): CSL's math library DOES expose ``sin(x)``/
``cos(x)`` as built-in functions on a real Cerebras-style chip --
https://sdk.cerebras.net/csl/language/libraries -- confirming a real chip
supports this operation. That page does NOT disclose the underlying
hardware/microarchitectural implementation (no mention of range
reduction, polynomial/CORDIC approximation, or a dedicated functional
unit -- the only implementation-adjacent detail given is an accuracy
caveat for very large ``abs(x)``). So unlike `sub` (where this project
could point at a real, confirmed-identical Accelergy component and alias
`add`'s numbers directly), there is no real, independently-citable
Cerebras-internal decomposition to mirror here -- this module makes the
SAME kind of reasoned-but-unverified engineering choice `exp`'s own
module docstring already made for ITS polynomial degree, using the same
standard, textbook numerical technique (range reduction + polynomial),
built ENTIRELY from already-Docker-verified real components (mul/add/
sub/bitwise). It is honest about being a reasoned choice, not a
Cerebras-confirmed one.

CHOSEN ALGORITHM (real, standard technique for computing ``sin`` in
software -- range reduction plus an odd-power polynomial in ``r^2``,
evaluated via Horner's method):
  1. RANGE REDUCTION: ``k = floor(x/(2*pi) + 0.5)``, ``r = x - k*(2*pi)``,
     so ``sin(x) = sin(r)`` with ``|r| <= pi`` -- a bounded interval a
     polynomial can approximate accurately, and (unlike ``exp``'s tiny
     ``+-ln(2)/2`` reduced range) wide enough to need more polynomial
     terms for the same accuracy target, budgeted for below.
  2. POLYNOMIAL: approximate ``sin(r)`` on ``[-pi, pi]`` via the real,
     textbook Taylor series for sine (odd powers only, since sine is an
     odd function): ``r - r^3/3! + r^5/5! - r^7/7! + ...``, evaluated as
     ``r * (c0 + r^2*(c1 + r^2*(c2 + ...)))`` (Horner in ``r^2``) so the
     even/odd structure only costs one extra real multiply
     (``r^2 = r*r``) up front instead of computing every power of ``r``
     separately.
  3. No reconstruction step is needed (unlike ``exp``'s ``2^k``
     multiply) -- range reduction here already returns the real answer's
     argument directly, not a scaled proxy.

REAL, EMPIRICAL VERIFICATION OF THIS MODULE'S POLYNOMIAL DEGREE (checked
this session in plain Python against ``math.sin``/``numpy.sin`` -- same
discipline `exp`'s own module docstring used for ITS degree choice, and
the same "independently-verifiable, no unverifiable minimax claim, plain
Taylor" policy): sweeping the number of odd-power Taylor terms against
2001 real points spanning ``r in [-pi, pi]``:
  5 terms (degree  9): max abs err 6.9e-3
  6 terms (degree 11): max abs err 4.5e-4
  7 terms (degree 13): max abs err 2.1e-5
  8 terms (degree 15): max abs err 7.7e-7   (~6.5x real fp32 epsilon)
  9 terms (degree 17): max abs err 2.2e-8   (~5.3x TIGHTER than real fp32
                                              epsilon 1.1920929e-07)
This module uses 9 TERMS. A second, independent real check ran the FULL
pipeline (real range reduction via ``floor(x/2pi+0.5)`` + this 9-term
polynomial) against ``numpy.sin`` over 20,000+ real points spanning
``x in [-100, 100]`` (comfortably covering every real RoPE angle this
project's Qwen3 compositions ever produce): max abs error
2.239e-08, confirming the reduction step itself does not degrade the
in-range polynomial accuracy above. `cos` (see ``..cos`` package) reuses
this exact pipeline via the real identity ``cos(x) = sin(x + pi/2)`` --
that identity was independently checked the same way (max abs error
2.241e-08 over the same 20,000+ points), see ``cos/_timeloop_cos.py``.

REAL PER-ELEMENT OPERATION COUNT (derived from the above):
  - range reduction: ``x*inv2pi`` (1 mul) + ``+0.5`` (1 add) +
    ``floor(...)`` (1 real bit-truncation, no dedicated component -- see
    "WHAT ISN'T REAL" below) + ``k*2pi`` (1 mul) + ``x - k*2pi`` (1 real
    SUBTRACT, using this project's own real `sub` primitive -- see
    ``wafersim.primitives.sub`` -- rather than `exp`'s older negate+add
    workaround, now that a real `sub` exists) => multiplies=2, adds=1,
    subtracts=1, floor=1
  - polynomial (9 terms, Horner in r^2, degree 17):
    ``r^2 = r*r`` (1 mul) + 8 Horner steps (each ``acc*r2 + c_i``: 1 mul +
    1 add) + final ``r*acc`` (1 mul) => multiplies = 1 + 8 + 1 = 10,
    adds = 8
  Totals: multiplies = 2 + 10 = 12; adds = 1 + 8 = 9; subtracts = 1;
  bitwise-proxy firings (floor only, no ``2^k`` step needed) = 1.

WHAT IS AND ISN'T REAL: every multiply/add/subtract/bitwise firing's
cycles/energy comes from a real Docker ``timeloop-model`` run against a
real, separately-characterized Accelergy component (``fpmultiplier``,
``fpadder`` for both add AND sub -- see ``wafersim.primitives.sub``'s own
module docstring for why sub reuses add's real component --,
``bitwise``). The real gap filled honestly, not silently: ``floor()`` has
no dedicated Accelergy component anywhere in this project's toolchain
(same real gap `exp` already named), modeled here as ONE real
``bitwise`` firing, the same honest fallback-proxy choice `exp`/`rsqrt`/
`cast` already use.

SCOPE: same as add/mul/div/rsqrt/exp -- one PE's local sin-over-N cost
only, no fabric/network component.

Used for: RoPE's rotary embedding tables (``cos``/``sin`` of each
position's rotation angle) -- see this project's Qwen3 composition
scripts, where these tables were previously precomputed via plain numpy
OUTSIDE the cost-tracked primitive set entirely (treated as a one-time,
data-independent constant, same as `cast` was treated as illustrative-
only in the tiny composition). Adding a real, Docker-backed cost model
here makes it possible to declare RoPE table generation as a real,
costed primitive call instead, should a composition script choose to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...specs import WaferSpec
from .._bitwise import BitwiseResult, BitwiseSpec, simulate_bitwise
from ..add import AddResult, AddSpec, simulate_add
from ..mul import MulResult, MulSpec, simulate_mul
from ..sub import SubResult, SubSpec, simulate_sub

# See module docstring's "REAL PER-ELEMENT OPERATION COUNT" section.
_POLY_TERMS = 9  # odd powers r, r^3, ..., r^17 -- see empirical accuracy sweep above
_MULS_PER_ELEMENT = 2 + (_POLY_TERMS + 1)  # range reduction(2) + poly(r^2 + 8 Horner + final = 10) = 12
_ADDS_PER_ELEMENT = 1 + (_POLY_TERMS - 1)  # range reduction(+0.5)(1) + poly Horner adds(8) = 9
_SUBS_PER_ELEMENT = 1  # range reduction: x - k*2pi
_BITWISE_PROXY_PER_ELEMENT = 1  # floor only (no 2^k reconstruction needed, unlike exp)


@dataclass(frozen=True, slots=True)
class SinSpec:
    """Compute ``elements_per_pe`` real elementwise ``sin(x)`` values on
    ONE PE via a real range-reduction + degree-17 (9-term) odd-power
    Taylor-polynomial decomposition (see module docstring)."""

    elements_per_pe: int
    bytes_per_element: int = 4

    def validate(self) -> None:
        if min(self.elements_per_pe, self.bytes_per_element) <= 0:
            raise ValueError("elements_per_pe and bytes_per_element must be positive")


@dataclass(frozen=True, slots=True)
class SinResult:
    """Real, Docker-verified per-PE cost of one ``sin`` primitive call,
    composed from four real sub-runs (mul/add/sub/bitwise) -- all kept in
    full below, not collapsed."""

    sin: SinSpec
    mul_result: MulResult
    add_result: AddResult
    sub_result: SubResult
    bitwise_result: BitwiseResult

    @property
    def local_compute_cycles(self) -> int:
        return (
            self.mul_result.local_compute_cycles
            + self.add_result.local_compute_cycles
            + self.sub_result.local_compute_cycles
            + self.bitwise_result.local_compute_cycles
        )

    @property
    def local_energy_uj(self) -> float:
        return (
            self.mul_result.local_energy_uj
            + self.add_result.local_energy_uj
            + self.sub_result.local_energy_uj
            + self.bitwise_result.local_energy_uj
        )

    @property
    def cycles_per_element(self) -> float:
        return self.local_compute_cycles / self.sin.elements_per_pe

    @property
    def energy_uj_per_element(self) -> float:
        return self.local_energy_uj / self.sin.elements_per_pe

    def to_dict(self) -> dict[str, object]:
        return {
            "sin": {
                "elements_per_pe": self.sin.elements_per_pe,
                "bytes_per_element": self.sin.bytes_per_element,
            },
            "local_compute_cycles": self.local_compute_cycles,
            "local_energy_uj": self.local_energy_uj,
            "cycles_per_element": self.cycles_per_element,
            "energy_uj_per_element": self.energy_uj_per_element,
            "multiplier_area_um2": self.mul_result.multiplier_area_um2,
            "adder_area_um2": self.add_result.adder_area_um2,
            "bitwise_area_um2": self.bitwise_result.bitwise_area_um2,
            "mul_result": self.mul_result.to_dict(),
            "add_result": self.add_result.to_dict(),
            "sub_result": self.sub_result.to_dict(),
            "bitwise_result": self.bitwise_result.to_dict(),
        }


def simulate_sin(sin: SinSpec, wafer: WaferSpec, out_dir: Path) -> SinResult:
    """Run FOUR real Docker ``timeloop-model`` pipelines end to end --
    ``simulate_mul()``/``simulate_add()``/``simulate_sub()`` for the
    range-reduction and polynomial firings, ``simulate_bitwise()`` for the
    real floor bit-level proxy -- and compose their real results into one
    ``SinResult``. Raises (never fabricates a result) if any real Docker
    sub-run fails."""
    sin.validate()
    n = sin.elements_per_pe

    mul_spec = MulSpec(elements_per_pe=n * _MULS_PER_ELEMENT, bytes_per_element=sin.bytes_per_element)
    add_spec = AddSpec(elements_per_pe=n * _ADDS_PER_ELEMENT, bytes_per_element=sin.bytes_per_element)
    sub_spec = SubSpec(elements_per_pe=n * _SUBS_PER_ELEMENT, bytes_per_element=sin.bytes_per_element)
    bitwise_spec = BitwiseSpec(
        elements_per_pe=n * _BITWISE_PROXY_PER_ELEMENT, bytes_per_element=sin.bytes_per_element
    )

    mul_result = simulate_mul(mul_spec, wafer, out_dir / "mul")
    add_result = simulate_add(add_spec, wafer, out_dir / "add")
    sub_result = simulate_sub(sub_spec, wafer, out_dir / "sub")
    bitwise_result = simulate_bitwise(bitwise_spec, wafer, out_dir / "bitwise")

    return SinResult(
        sin=sin, mul_result=mul_result, add_result=add_result, sub_result=sub_result, bitwise_result=bitwise_result
    )
