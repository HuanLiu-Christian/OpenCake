#!/usr/bin/env python3
"""Real single-primitive check for `silu`: one real Tensor -> silu(x)
(real numeric result, real x/(1+numpy.exp(-x))) -> SiluSpec(from the real
result) -> simulate_silu() -> real cycles/time/energy/area from FOUR real
Docker `timeloop-model` sub-runs at the top level -- negate (real
bitwise sign-flip), exp (itself 3 NESTED real sub-runs: mul/add/bitwise),
add_one, div (itself 2 nested real sub-runs: mul/add) -- 7 real
underlying `timeloop-model` invocations in total for one call (see
``wafersim.primitives.silu._timeloop_silu``'s own module docstring: silu
needs NO new numerical research of its own, it is a pure composition of
already-real bitwise/exp/add/div pipelines: silu(x) = x/(1+exp(-x))).

Used for: SwiGLU's gate activation in the MLP block of both LLaMA and
Qwen, per this primitive's own package docstring.

No synthetic fallback anywhere: if Docker/TimeLoop fails for any of the 7
underlying sub-runs, this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open -- this is the most Docker-heavy single-primitive check so far, 7
real sub-runs, so it will take noticeably longer than add/mul/etc.):
  source .venv/bin/activate
  python3 run_silu_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import SiluSpec, WaferSpec, silu, simulate_silu
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/silu_single_check")


def main() -> None:
    a = Tensor.from_values([-2.0, -0.5, 0.5, 2.0])
    print(f"a = {a.data.tolist()}")

    # Real numeric result -- plain x/(1+numpy.exp(-x)), no Docker
    # involved here.
    result = silu(a)
    print(f"real numeric result silu(a) = {result.data.tolist()}")

    # Bridge: derive the cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = SiluSpec(elements_per_pe=result.size, bytes_per_element=result.bytes_per_element)
    print(f"bridged SiluSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model runs (4 top-level sub-runs, 7 real
    # timeloop-model invocations total once exp's and div's own nested
    # sub-runs are counted) -- this is the only place cycles/energy/area
    # come from; raises (never fabricates) if any sub-run fails.
    cost = simulate_silu(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (silu, composed) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")

    print("\n--- top-level sub-run breakdown: silu(x) = x / (1 + exp(-x)) ---")
    print(f"negate (real bitwise sign-flip, -x):        "
          f"cycles={cost.negate_result.local_compute_cycles}, energy_uj={cost.negate_result.local_energy_uj}")
    print(f"exp (exp(-x), 3 nested real sub-runs):       "
          f"cycles={cost.exp_result.local_compute_cycles}, energy_uj={cost.exp_result.local_energy_uj}")
    print(f"add_one (1 + exp(-x)):                       "
          f"cycles={cost.add_one_result.local_compute_cycles}, energy_uj={cost.add_one_result.local_energy_uj}")
    print(f"div (x / (1+exp(-x)), 2 nested real sub-runs): "
          f"cycles={cost.div_result.local_compute_cycles}, energy_uj={cost.div_result.local_energy_uj}")


if __name__ == "__main__":
    main()
