#!/usr/bin/env python3
"""Real single-primitive check for `div`: two real Tensors -> div(a, b)
(real numeric result) -> DivSpec(from that real result) -> simulate_div()
-> real cycles/time/energy/area from TWO real Docker `timeloop-model`
sub-runs (a real ``simulate_mul()`` sub-run for the Newton-Raphson
reciprocal's multiply firings, a real ``simulate_add()`` sub-run for its
subtract/combine firings -- see
``wafersim.primitives.div._timeloop_div``'s own module docstring for the
full real-source Newton-Raphson-reciprocal derivation this is built on).

No synthetic fallback anywhere: if Docker/TimeLoop fails for either
sub-run, this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_div_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec, div
from wafersim.primitives.div import DivSpec, simulate_div
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/div_single_check")


def main() -> None:
    a = Tensor.from_values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    b = Tensor.from_values([2.0, 4.0, 5.0, 8.0, 10.0, 3.0, 7.0, 16.0])
    print(f"a = {a.data.tolist()}")
    print(f"b = {b.data.tolist()}")

    # Real numeric result -- plain numpy divide, no Docker involved here.
    result = div(a, b)
    print(f"real numeric result a / b = {result.data.tolist()}")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = DivSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged DivSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model runs (2 sub-runs: mul, add) -- this is
    # the only place cycles/energy/area come from; raises (never
    # fabricates) if either sub-run fails.
    cost = simulate_div(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (div, composed) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"multiplier (fpmultiplier_gate) area (real):               {cost.multiplier_area_um2} um^2")
    print(f"adder (fpadder_gate) area (real):                         {cost.adder_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")

    print("\n--- sub-run breakdown (Newton-Raphson reciprocal: mul stage + add stage) ---")
    print(f"mul sub-run:  cycles={cost.mul_result.local_compute_cycles}, "
          f"energy_uj={cost.mul_result.local_energy_uj}")
    print(f"add sub-run:  cycles={cost.add_result.local_compute_cycles}, "
          f"energy_uj={cost.add_result.local_energy_uj}")


if __name__ == "__main__":
    main()
