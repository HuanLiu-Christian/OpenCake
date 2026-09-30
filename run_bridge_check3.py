#!/usr/bin/env python3
"""Real end-to-end check for the 4 NEW real hardware cost models added this
round: rsqrt, exp, silu, cast. Unlike ``run_bridge_check2.py``'s 5
primitives (which all shared ONE already-untested component pipeline,
``_bitwise``), these 4 introduce TWO new things at once:

  1. A brand-new Accelergy component pipeline, ``_intadder.py`` (the real
     ``intadder`` component) -- used by `rsqrt` (seed's integer subtract)
     and `cast` (exponent rebias + rounding increment) -- has NEVER been
     run against real Docker before this script.
  2. Multi-stage COMPOSITIONS -- `rsqrt` composes 4 real sub-runs (mul,
     add, bitwise, intadder), `exp` composes 3 (mul, add, bitwise), `silu`
     composes 4 (bitwise, exp [itself 3 sub-runs], add, div), `cast`
     composes 2 (bitwise, intadder) -- so `silu` alone triggers roughly
     8 real Docker ``timeloop-model`` invocations for one call.

If any step below crashes, that is real, expected-possible information
(the `add` primitive's own cost model needed 9 real Docker rounds to reach
its current, working design -- see ``add/_timeloop_add.py``'s module
docstring, and ``_bitwise.py``'s own docstring for how that fix was
transplanted verbatim into the first new component afterward) -- paste the
real Docker error back so the architecture/mapping files (most likely
``_intadder.py``, since it is the one genuinely new component pipeline
here) can be fixed at the actual root cause. No synthetic fallback
anywhere: if Docker/TimeLoop fails, this raises and prints nothing
fabricated for that step.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_bridge_check3.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import (
    CastSpec,
    ExpSpec,
    RsqrtSpec,
    SiluSpec,
    WaferSpec,
    cast,
    exp,
    rsqrt,
    silu,
    simulate_cast,
    simulate_exp,
    simulate_rsqrt,
    simulate_silu,
)
from wafersim.primitives.tensor import Tensor

import numpy as np

OUT_DIR = Path("outputs/bridge_check3")


def check_rsqrt() -> None:
    print("\n--- rsqrt: Tensor -> rsqrt(x) (real numpy) -> RsqrtSpec(from result) "
          "-> simulate_rsqrt (4 real sub-runs: mul/add/bitwise/intadder) ---")
    a = Tensor.from_values([0.25, 1.0, 4.0, 16.0])
    result = rsqrt(a)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    ref = 1.0 / np.sqrt(a.data)
    print(f"  numpy reference: {ref.tolist()}")
    assert np.allclose(result.data, ref), "real rsqrt() diverged from real numpy reference"
    spec = RsqrtSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"  bridged RsqrtSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_rsqrt(spec, WaferSpec(), OUT_DIR / "rsqrt")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"cycles_per_element={cost.cycles_per_element}, "
          f"intadder_area_um2={cost.intadder_result.intadder_area_um2}")


def check_exp() -> None:
    print("\n--- exp: Tensor -> exp(x) (real numpy) -> ExpSpec(from result) "
          "-> simulate_exp (3 real sub-runs: mul/add/bitwise) ---")
    a = Tensor.from_values([-1.0, 0.0, 1.0, 2.0])
    result = exp(a)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    ref = np.exp(a.data)
    print(f"  numpy reference: {ref.tolist()}")
    assert np.allclose(result.data, ref), "real exp() diverged from real numpy reference"
    spec = ExpSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"  bridged ExpSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_exp(spec, WaferSpec(), OUT_DIR / "exp")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"cycles_per_element={cost.cycles_per_element}")


def check_silu() -> None:
    print("\n--- silu: Tensor -> silu(x) (real numpy) -> SiluSpec(from result) "
          "-> simulate_silu (4 real sub-runs, incl. a nested 3-sub-run exp) ---")
    a = Tensor.from_values([-2.0, -0.5, 0.5, 2.0])
    result = silu(a)
    print(f"  numeric result: {result} values={result.data.tolist()}")
    ref = a.data / (1.0 + np.exp(-a.data))
    print(f"  numpy reference: {ref.tolist()}")
    assert np.allclose(result.data, ref), "real silu() diverged from real numpy reference"
    spec = SiluSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"  bridged SiluSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_silu(spec, WaferSpec(), OUT_DIR / "silu")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"cycles_per_element={cost.cycles_per_element}")


def check_cast() -> None:
    print("\n--- cast: Tensor(fp32) -> cast(x, np.float16) (real numpy astype) "
          "-> CastSpec(from real fp32 input) -> simulate_cast "
          "(2 real sub-runs: bitwise/intadder) ---")
    a = Tensor.from_values([1.0, 3.14159, -65000.0, 0.001])
    result = cast(a, np.float16)
    print(f"  numeric result: {result} dtype={result.data.dtype} values={result.data.tolist()}")
    ref = a.data.astype(np.float16)
    print(f"  numpy reference: {ref.tolist()}")
    assert np.array_equal(result.data, ref), "real cast() diverged from real numpy astype reference"
    # Cost is modeled at the wider (fp32, 4-byte) register width -- see
    # cast/_timeloop_cast.py's own module docstring's "BYTES_PER_ELEMENT
    # CHOICE" section.
    spec = CastSpec(elements_per_pe=a.size, bytes_per_element=a.bytes_per_element)
    print(f"  bridged CastSpec: elements_per_pe={spec.elements_per_pe}")
    cost = simulate_cast(spec, WaferSpec(), OUT_DIR / "cast")
    print(f"  REAL cost: cycles={cost.local_compute_cycles}, energy_uj={cost.local_energy_uj}, "
          f"cycles_per_element={cost.cycles_per_element}")


def main() -> None:
    print("Real cost-model check for the 4 NEW primitives added this round "
          "(rsqrt/exp/silu/cast)")
    print("This is the FIRST real Docker test of the new `intadder` component "
          "pipeline, and the first real Docker test of these 4 multi-stage "
          "compositions -- if any step below crashes, that is real, "
          "expected-possible information (see this file's own module "
          "docstring), not a bug in your setup.")
    check_rsqrt()
    check_exp()
    check_silu()
    check_cast()
    print("\nIf every step above printed a REAL cost line with no exception, "
          "all 4 new cost models (and the new `intadder` component) ran real "
          "Docker sub-runs successfully.")


if __name__ == "__main__":
    main()
