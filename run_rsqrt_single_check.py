#!/usr/bin/env python3
"""Real single-primitive check for `rsqrt`: one real Tensor -> rsqrt(x)
(real numeric result, real 1/numpy.sqrt) -> RsqrtSpec(from the real
result) -> simulate_rsqrt() -> real cycles/time/energy/area from FOUR
real Docker `timeloop-model` sub-runs (mul/add/bitwise/intadder),
composing the real, cited "fast inverse square root" magic-number seed +
3-iteration Newton-Raphson decomposition (see
``wafersim.primitives.rsqrt._timeloop_rsqrt``'s own module docstring for
the full real-source derivation -- Quake III's own real, widely-
documented algorithm -- and the real, independently-measured accuracy
check confirming 3 iterations reaches real fp32 machine-epsilon-level
accuracy).

There is no dedicated reciprocal-square-root circuit anywhere in this
project's pinned Docker toolchain, so this decomposes into: the magic-
number seed's real bit-shift (1 ``bitwise``-proxy firing/element, since
this project's toolchain has no real, queryable bit-shifter component --
named explicitly, not buried) + the seed's real integer subtract (1 real
``intadder`` firing/element, a genuinely NEW real Accelergy component
this project added specifically for this) + 3 Newton-Raphson iterations
of real floating-point multiplies/subtracts (real mul/add firings).

Used for: RMSNorm (x * rsqrt(mean(x^2) + eps)), per this primitive's own
package docstring.

No synthetic fallback anywhere: if Docker/TimeLoop fails for any of the 4
sub-runs, this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_rsqrt_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import RsqrtSpec, WaferSpec, rsqrt, simulate_rsqrt
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/rsqrt_single_check")


def main() -> None:
    a = Tensor.from_values([0.25, 1.0, 4.0, 16.0])
    print(f"a = {a.data.tolist()}")

    # Real numeric result -- plain 1/numpy.sqrt, no Docker involved here.
    result = rsqrt(a)
    print(f"real numeric result rsqrt(a) = {result.data.tolist()}  (expect [2.0, 1.0, 0.5, 0.25])")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = RsqrtSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged RsqrtSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model runs (4 sub-runs: mul, add, bitwise,
    # intadder) -- this is the only place cycles/energy/area come from;
    # raises (never fabricates) if any sub-run fails.
    cost = simulate_rsqrt(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (rsqrt, composed) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"multiplier (fpmultiplier_gate) area (real):               {cost.mul_result.multiplier_area_um2} um^2")
    print(f"adder (fpadder_gate) area (real):                         {cost.add_result.adder_area_um2} um^2")
    print(f"bitwise (bitwise_gate) area (real):                       {cost.bitwise_result.bitwise_area_um2} um^2")
    print(f"intadder (intadder_gate) area (real):                     {cost.intadder_result.intadder_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")

    print("\n--- sub-run breakdown (magic-number seed + 3-iteration Newton-Raphson) ---")
    print(f"mul sub-run (10 real muls/element, x2 term + 3 iters * 3): "
          f"cycles={cost.mul_result.local_compute_cycles}, energy_uj={cost.mul_result.local_energy_uj}")
    print(f"add sub-run (3 real subs/element, 3 iters * 1):           "
          f"cycles={cost.add_result.local_compute_cycles}, energy_uj={cost.add_result.local_energy_uj}")
    print(f"bitwise sub-run (1 proxy firing/element, seed's i>>1):    "
          f"cycles={cost.bitwise_result.local_compute_cycles}, energy_uj={cost.bitwise_result.local_energy_uj}")
    print(f"intadder sub-run (1 real firing/element, seed's subtract): "
          f"cycles={cost.intadder_result.local_compute_cycles}, energy_uj={cost.intadder_result.local_energy_uj}")


if __name__ == "__main__":
    main()
