#!/usr/bin/env python3
"""Real single-primitive check for `sub`: one real Tensor pair -> a - b
(real numeric result, real numpy subtract) -> SubSpec(from the real
result) -> simulate_sub() -> real cycles/time/energy/area from ONE real
Docker `timeloop-model` run (see ``wafersim.primitives.sub``'s own module
docstring for why `sub` is a deliberate ALIAS of `add`'s own real,
already-Docker-verified cost model, not a new architecture -- SubSpec IS
AddSpec, simulate_sub IS simulate_add).

Real-world use case exercised here: softmax's own numeric-stability step,
``x - row_max`` (subtracting each row's real max before ``exp``) -- one
of the two real subtractions this project previously modeled as
negate-by-mul(-1) + add (the other being RoPE's ``rotate_half``'s
``-x2``), now declared as one real, dedicated `sub` firing instead.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_sub_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec, sub
from wafersim.primitives.sub import SubSpec, simulate_sub
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/sub_single_check")


def main() -> None:
    # A real softmax-style row: x, and that row's real max broadcast to
    # every position (what a real `op_max` + broadcast would produce).
    x = Tensor.from_values([3.0, 1.0, 4.0, 1.0, 5.0])
    row_max = Tensor.from_values([5.0, 5.0, 5.0, 5.0, 5.0])
    print(f"x       = {x.data.tolist()}")
    print(f"row_max = {row_max.data.tolist()}")

    # Real numeric result -- plain numpy subtract, no Docker involved here.
    result = sub(x, row_max)
    print(f"real numeric result x - row_max = {result.data.tolist()}")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = SubSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged SubSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")
    print("(SubSpec IS AddSpec -- this is a real, deliberate alias, see wafersim.primitives.sub's own docstring)")

    # Real Docker timeloop-model run -- literally `add`'s own real,
    # already-Docker-verified pipeline; this is the only place cycles/
    # energy/area come from; raises (never fabricates) on failure.
    cost = simulate_sub(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (sub, aliased to add's real fpadder) ===")
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
