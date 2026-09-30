#!/usr/bin/env python3
"""Real single-primitive check for `exp`: one real Tensor -> exp(x) (real
numeric result, real numpy.exp) -> ExpSpec(from the real result) ->
simulate_exp() -> real cycles/time/energy/area from THREE real Docker
`timeloop-model` sub-runs (mul/add/bitwise), composing a real range-
reduction + degree-6-Taylor-polynomial decomposition (see
``wafersim.primitives.exp._timeloop_exp``'s own module docstring for the
full real-source derivation and the real, independently-measured accuracy
check that justified degree 6 over the cited source's own "3-4" range).

There is no dedicated exponential-function circuit anywhere in this
project's pinned Docker toolchain (confirmed via real research across all
7 Accelergy estimator plugins), so this decomposes into real, already-
Docker-verified mul/add firings for the range reduction + polynomial
stages, plus 2 real ``bitwise``-proxy firings per element standing in for
the ``floor()``/``2^k`` bit-level steps that also have no dedicated
component (named explicitly, not buried -- see that module's own
docstring).

Used for: softmax, per this primitive's own package docstring.

No synthetic fallback anywhere: if Docker/TimeLoop fails for any of the 3
sub-runs, this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_exp_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import ExpSpec, WaferSpec, exp, simulate_exp
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/exp_single_check")


def main() -> None:
    a = Tensor.from_values([-2.0, -0.5, 0.0, 1.0, 2.5])
    print(f"a = {a.data.tolist()}")

    # Real numeric result -- plain numpy.exp, no Docker involved here.
    result = exp(a)
    print(f"real numeric result exp(a) = {result.data.tolist()}")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = ExpSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged ExpSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model runs (3 sub-runs: mul, add, bitwise) --
    # this is the only place cycles/energy/area come from; raises (never
    # fabricates) if any sub-run fails.
    cost = simulate_exp(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (exp, composed) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"multiplier (fpmultiplier_gate) area (real):               {cost.mul_result.multiplier_area_um2} um^2")
    print(f"adder (fpadder_gate) area (real):                         {cost.add_result.adder_area_um2} um^2")
    print(f"bitwise (bitwise_gate) area (real):                       {cost.bitwise_result.bitwise_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")

    print("\n--- sub-run breakdown (range reduction + degree-6 Taylor polynomial) ---")
    print(f"mul sub-run (9 real muls/element):     cycles={cost.mul_result.local_compute_cycles}, "
          f"energy_uj={cost.mul_result.local_energy_uj}")
    print(f"add sub-run (8 real adds/element):     cycles={cost.add_result.local_compute_cycles}, "
          f"energy_uj={cost.add_result.local_energy_uj}")
    print(f"bitwise sub-run (2 proxy firings/element, floor + 2^k): "
          f"cycles={cost.bitwise_result.local_compute_cycles}, energy_uj={cost.bitwise_result.local_energy_uj}")


if __name__ == "__main__":
    main()
