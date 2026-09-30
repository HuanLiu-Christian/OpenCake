#!/usr/bin/env python3
"""Real single-primitive check for `add`: two real Tensors -> add(a, b)
(real numeric result) -> AddSpec(from that real result) -> simulate_add()
-> real cycles/time/energy/area from one real Docker `timeloop-model`
run against the real, already-verified ``fpadder`` Accelergy component
(see ``wafersim.primitives.add._timeloop_add``'s own module docstring for
the full real-source investigation this is built on).

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_add_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec, add
from wafersim.primitives.add import AddSpec, simulate_add
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/add_single_check")


def main() -> None:
    a = Tensor.from_values([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = Tensor.from_values([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    print(f"a = {a.data.tolist()}")
    print(f"b = {b.data.tolist()}")

    # Real numeric result -- plain numpy add, no Docker involved here.
    result = add(a, b)
    print(f"real numeric result a + b = {result.data.tolist()}")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = AddSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged AddSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run -- this is the only place cycles/
    # energy/area come from; raises (never fabricates) on failure.
    cost = simulate_add(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (add) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"adder (fpadder_gate) area (real):                         {cost.adder_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
