"""Real, reproducible numerical-accuracy checks for the approximation
algorithms this project's `rsqrt`/`exp`/`silu`/`cast` REAL, Docker-backed
hardware cost models (see ``wafersim/primitives/{rsqrt,exp,silu,cast}/
_timeloop_*.py``) derive their real per-element operation counts from.

WHY THIS FILE EXISTS: none of TimeLoop/Accelergy has any way to check
whether a chosen approximation algorithm (a magic-number seed, a
Newton-Raphson iteration count, a Taylor polynomial degree, an IEEE-754
bit-manipulation conversion) is actually numerically correct -- Docker
only ever tells you the real cycles/energy/area of firing a given
sequence of real hardware components N times. Numerical correctness has
to be checked independently, in real numpy, against real reference
implementations (``numpy``/``math``). This file is that independent
check -- the SAME checks were first run ad hoc in a scratch script this
session; this is their permanent, reproducible form, and it is what each
of ``rsqrt/_timeloop_rsqrt.py``, ``exp/_timeloop_exp.py``,
``silu/_timeloop_silu.py``'s own module docstrings cite by this file's
name.

Every approximation implemented below is plain Python/NumPy -- entirely
separate from (and never imported by) the real Docker-backed cost-model
code, which only ever needs the real ELEMENT COUNT of each algorithmic
step (see each ``_timeloop_*.py`` module docstring's "REAL PER-ELEMENT
OPERATION COUNT" section) and never re-derives or re-checks the
algorithm's numerical accuracy itself. Keeping the two separate mirrors
this whole project's own "numeric compute layer vs. hardware cost-model
layer" split (see ``wafersim/primitives/tensor``'s module docstring).
"""

from __future__ import annotations

import math
import struct

import numpy as np

FP32_EPS = float(np.finfo(np.float32).eps)  # 1.1920929e-07


# ======================================================================
# rsqrt: Quake III magic-number seed + 3 Newton-Raphson iterations
# (see wafersim/primitives/rsqrt/_timeloop_rsqrt.py's module docstring;
# source: https://en.wikipedia.org/wiki/Fast_inverse_square_root)
# ======================================================================

_RSQRT_MAGIC = 0x5F3759DF
_RSQRT_ITERATIONS = 3
_RSQRT_TEST_VALUES = [7e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 3.14159]


def _rsqrt_magic_newton(x: float, iterations: int) -> float:
    """Real bit-level magic-number seed, reinterpreting the IEEE-754 bits
    of a real float32 as an integer, followed by ``iterations`` real
    Newton-Raphson refinement steps ``y := y*(1.5 - 0.5*x*y*y)`` -- the
    exact algorithm ``rsqrt``'s cost model counts real mul/add/bitwise/
    intadder firings for."""
    x32 = np.float32(x)
    bits = struct.unpack("<i", struct.pack("<f", x32))[0]
    seed_bits = np.int32(_RSQRT_MAGIC - (bits >> 1))
    y = np.float32(struct.unpack("<f", struct.pack("<i", int(seed_bits)))[0])
    x_half = np.float32(0.5) * x32
    for _ in range(iterations):
        y = y * (np.float32(1.5) - x_half * y * y)
    return float(y)


def test_rsqrt_magic_seed_alone_matches_cited_error_order():
    """The real cited source's own worked example (x=0.15625) reports
    about 3.4% error from the seed alone, before any Newton-Raphson
    refinement -- confirm this project's real seed implementation lands
    in that same real, cited error range (not exactly 3.4%, since that
    figure is for one specific input, but the same order of magnitude)."""
    seed_only = _rsqrt_magic_newton(0.15625, iterations=0)
    ref = 1.0 / math.sqrt(0.15625)
    rel_err = abs(seed_only - ref) / ref
    assert 0.01 < rel_err < 0.05, f"seed-only relative error {rel_err} not in the cited ~3.4% ballpark"


def test_rsqrt_three_newton_iterations_reach_fp32_epsilon():
    """The real, load-bearing claim ``rsqrt/_timeloop_rsqrt.py``'s own
    module docstring makes: 3 Newton-Raphson iterations from the
    magic-number seed reach real fp32-machine-epsilon-level accuracy
    across a wide real test range. Checked here against
    ``1/math.sqrt(x)``, not assumed."""
    max_rel_err = 0.0
    for v in _RSQRT_TEST_VALUES:
        approx = _rsqrt_magic_newton(v, _RSQRT_ITERATIONS)
        ref = 1.0 / math.sqrt(v)
        rel_err = abs(approx - ref) / ref
        max_rel_err = max(max_rel_err, rel_err)
        assert rel_err <= FP32_EPS, f"rsqrt({v}): real relative error {rel_err} exceeds fp32 eps {FP32_EPS}"
    # Real headroom check: 3 iterations should comfortably clear epsilon,
    # not sit exactly at the edge (the real reason this project chose 3
    # iterations over 2 -- see the module docstring's own justification).
    assert max_rel_err < FP32_EPS


def test_rsqrt_two_newton_iterations_are_measurably_worse():
    """Confirms 3 iterations is a real, load-bearing choice and not an
    arbitrary round number -- 2 iterations measurably fails to reach the
    same real accuracy on at least one real test value, which is exactly
    why this project's cost model uses 3."""
    max_rel_err_2 = 0.0
    for v in _RSQRT_TEST_VALUES:
        approx = _rsqrt_magic_newton(v, iterations=2)
        ref = 1.0 / math.sqrt(v)
        max_rel_err_2 = max(max_rel_err_2, abs(approx - ref) / ref)
    assert max_rel_err_2 > FP32_EPS, "2 iterations unexpectedly already reached fp32 epsilon"


# ======================================================================
# exp: range reduction (exp(x) = 2^k * exp(r)) + degree-6 Taylor polynomial
# (see wafersim/primitives/exp/_timeloop_exp.py's module docstring;
# source: https://justinwillmert.com/articles/2020/numerically-computing-the-exponential-function-with-polynomial-approximations/)
# ======================================================================

_LN2 = math.log(2.0)
_INV_LN2 = 1.0 / _LN2
_EXP_TEST_VALUES = [-10, -5, -1, -0.5, 0, 0.001, 0.5, 1, 2, 5, 10, 20]
_EXP_POLY_DEGREE = 6


def _taylor_exp_r(r: float, degree: int) -> float:
    """Real plain-Taylor polynomial for exp(r) around 0, evaluated in
    real Horner form (``degree`` real multiply-adds) -- deliberately NOT
    a minimax-optimized polynomial, since this project could not find a
    real, independently-citable minimax coefficient table (see the cost
    model's own module docstring for why plain Taylor was chosen over an
    unverifiable "minimax" claim)."""
    coeffs = [1.0]
    fact = 1.0
    for i in range(1, degree + 1):
        fact *= i
        coeffs.append(1.0 / fact)
    acc = coeffs[degree]
    for c in reversed(coeffs[:degree]):
        acc = acc * r + c
    return acc


def _exp_range_reduced(x: float, degree: int) -> float:
    """Real range-reduction + polynomial exp(x) approximation: ``k =
    floor(x/ln2 + 0.5)``, ``r = x - k*ln2``, ``exp(x) = 2^k * exp(r)`` --
    the exact algorithm ``exp``'s cost model counts real mul/add/bitwise
    firings for (floor and the 2^k reconstruction as bitwise-proxy
    firings, everything else as real mul/add)."""
    k = math.floor(x * _INV_LN2 + 0.5)
    r = x - k * _LN2
    poly = _taylor_exp_r(r, degree)
    return poly * (2.0**k)


def test_exp_degree4_taylor_insufficient_for_fp32_epsilon():
    """Real, independently-measured confirmation that the cited source's
    own suggested "3-4 multiply-add" polynomial range does NOT reach fp32
    epsilon with a plain (non-minimax) Taylor polynomial -- this is the
    real finding that justified this project going to degree 6 instead of
    stopping at the cited source's own lower suggestion."""
    max_rel_err = max(
        abs(_exp_range_reduced(v, 4) - math.exp(v)) / abs(math.exp(v)) for v in _EXP_TEST_VALUES
    )
    assert max_rel_err > FP32_EPS, "degree-4 Taylor unexpectedly already reached fp32 epsilon"


def test_exp_degree6_taylor_approaches_fp32_epsilon():
    """The real, load-bearing claim ``exp/_timeloop_exp.py``'s own module
    docstring makes: a degree-6 plain-Taylor polynomial (this project's
    real, final choice) gets close to real fp32 machine epsilon across a
    wide real test range spanning x in [-10, 20]. Checked here against
    ``math.exp``, not assumed."""
    max_rel_err = 0.0
    for v in _EXP_TEST_VALUES:
        approx = _exp_range_reduced(v, _EXP_POLY_DEGREE)
        ref = math.exp(v)
        rel_err = abs(approx - ref) / abs(ref)
        max_rel_err = max(max_rel_err, rel_err)
    # Real, measured bound: degree-6 lands within about an order of
    # magnitude of fp32 epsilon (not below it in every case) -- this
    # project's own module docstring names this honestly as "the closest
    # this project could get with a real, independently-verifiable
    # (plain Taylor, no unverifiable minimax claim) polynomial," not a
    # claim of reaching epsilon exactly.
    assert max_rel_err < 5 * FP32_EPS, f"degree-6 Taylor real max relative error {max_rel_err} too large"


# ======================================================================
# silu: composed exp(degree-6 Taylor) + real division
# (see wafersim/primitives/silu/_timeloop_silu.py's module docstring)
# ======================================================================

_SILU_TEST_VALUES = [-10, -5, -2, -1, -0.1, 0.1, 1, 2, 5, 10]  # 0 excluded: ref=0, relative error undefined


def _silu_composed(x: float) -> float:
    """Real ``silu(x) = x / (1 + exp(-x))``, using this project's own
    real degree-6-Taylor range-reduced ``exp`` approximation for the
    ``exp(-x)`` term -- the exact composition ``silu``'s cost model
    counts real bitwise/exp/add/div firings for."""
    e = _exp_range_reduced(-x, _EXP_POLY_DEGREE)
    return x / (1.0 + e)


def test_silu_composition_does_not_compound_exp_error():
    """The real, load-bearing claim ``silu/_timeloop_silu.py``'s own
    module docstring makes: composing the degree-6-Taylor ``exp``
    approximation with real division does not compound error into
    anything problematic for silu -- checked here against
    ``x/(1+math.exp(-x))``, not assumed."""
    max_rel_err = 0.0
    for v in _SILU_TEST_VALUES:
        approx = _silu_composed(v)
        ref = v / (1.0 + math.exp(-v))
        rel_err = abs(approx - ref) / abs(ref)
        max_rel_err = max(max_rel_err, rel_err)
        assert rel_err <= FP32_EPS, f"silu({v}): real relative error {rel_err} exceeds fp32 eps {FP32_EPS}"
    assert max_rel_err < FP32_EPS


# ======================================================================
# cast: real IEEE-754 unpack/rebias/round/repack fp32->fp16 conversion
# (see wafersim/primitives/cast/_timeloop_cast.py's module docstring;
# source: https://www.corsix.org/content/converting-fp32-to-fp16)
# ======================================================================

_CAST_TEST_VALUES = [
    1.0, -1.0, 3.14159, 0.1, 123.456, -0.5, 7.0, 65000.0, -65000.0,
    2.0, 0.333333, 1e-3, 12345.6789, -0.0087890625, 99.99,
]


def _fp32_to_fp16_manual(x: float) -> int:
    """Real, manual bit-manipulation fp32->fp16 conversion implementing
    the exact unpack/rebias/round(-to-nearest-even)/repack algorithm
    ``cast``'s cost model (``cast/_timeloop_cast.py``) counts real
    bitwise/intadder firings for. Returns the raw 16-bit pattern.
    Restricted to the normal (non-subnormal, non-overflow, non-nan) range
    for both formats, matching the real test values used below -- this
    project's cost model only claims to model that common case, named
    explicitly in its own module docstring."""
    bits32 = struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]
    sign = (bits32 >> 31) & 0x1
    exp32 = (bits32 >> 23) & 0xFF
    mant32 = bits32 & 0x7FFFFF

    # REBIAS: fp32 bias 127 -> fp16 bias 15 (real integer subtract).
    exp16 = exp32 - 127 + 15

    # ROUND: real round-to-nearest-even on the 23->10 bit mantissa
    # truncation (guard/round/sticky-bit test + conditional +1 increment).
    shift = 13
    mant16 = mant32 >> shift
    round_bit = (mant32 >> (shift - 1)) & 0x1
    sticky = 1 if (mant32 & ((1 << (shift - 1)) - 1)) != 0 else 0
    if round_bit and (sticky or (mant16 & 0x1)):
        mant16 += 1
        if mant16 == 0x400:  # mantissa overflow -> real exponent bump
            mant16 = 0
            exp16 += 1

    # REPACK.
    return (sign << 15) | (exp16 << 10) | mant16


def test_cast_manual_bit_algorithm_matches_real_numpy_fp16_cast():
    """The real, load-bearing claim ``cast/_timeloop_cast.py``'s own
    module docstring makes: the unpack/rebias/round/repack decomposition
    it counts real bitwise/intadder firings for is an actually-correct
    fp32->fp16 conversion algorithm, not a fictional one -- confirmed
    here bit-for-bit against real NumPy's own ``astype(np.float16)``
    (which this project's numeric `cast()` layer calls directly, see
    ``wafersim/primitives/cast/__init__.py``), across a real, varied test
    set spanning small/large/negative/fractional values within fp16's
    normal range."""
    for v in _CAST_TEST_VALUES:
        manual_bits = _fp32_to_fp16_manual(v)
        real_bits = int(np.float32(v).astype(np.float16).view(np.uint16))
        assert manual_bits == real_bits, (
            f"cast({v}): manual bit algorithm 0x{manual_bits:04x} != "
            f"real numpy fp16 cast 0x{real_bits:04x}"
        )
