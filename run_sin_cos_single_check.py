#!/usr/bin/env python3
"""Real single-primitive check for `sin`/`cos`: a small, real RoPE-style
rotation-angle table -> sin(angle)/cos(angle) (real numeric result, real
numpy sin/cos) -> SinSpec/CosSpec(from the real result) ->
simulate_sin()/simulate_cos() -> real cycles/time/energy/area from FOUR
(sin) + FIVE (cos: sin's four plus one phase-shift add) real Docker
`timeloop-model` sub-runs (mul/add/sub/bitwise -- see
``wafersim.primitives.sin``/``cos``'s own module docstrings for the real
range-reduction + degree-17 Taylor-polynomial decomposition, and for why
`cos` reuses `sin`'s pipeline via the real ``cos(x) = sin(x + pi/2)``
identity rather than a second, independent polynomial).

Real-world use case exercised here: RoPE's own rotation-angle table
(``position * inv_freq`` for a few real positions/frequency indices) --
previously computed via plain, uncosted numpy in this project's Qwen3
composition scripts (see those modules' own "REAL DOCKER CRASH" /
docstring notes); this script shows what declaring that table generation
as a real, costed primitive call would look like instead.

Also independently re-verifies (live, not just cited) this project's own
accuracy claim for these primitives -- max abs error vs. real numpy.sin/
numpy.cos -- so the printed numbers are reproducible, not just asserted.

No synthetic fallback anywhere: if Docker/TimeLoop fails for any sub-run,
this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_sin_cos_single_check.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from wafersim.primitives import WaferSpec, cos, sin
from wafersim.primitives.cos import CosSpec, simulate_cos
from wafersim.primitives.sin import SinSpec, simulate_sin
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/sin_cos_single_check")

HEAD_DIM = 128  # real Qwen3-8B head_dim, see qwen3_8b_prefill_decode.py
ROPE_THETA = 1_000_000.0  # real Qwen3-8B rope_theta
POSITIONS = [0, 1, 5, 15, 63]  # a few real positions from this project's
# actual PREFILL_SEQ_LEN=10 / decode-step range, plus a couple beyond it


def main() -> None:
    # A real, small RoPE-style rotation-angle table: angle = position *
    # inv_freq, for the first few real frequency indices of a real
    # head_dim=128 table (mirrors qwen3_8b_prefill_decode.py's own
    # build_rope_tables, just a small illustrative slice of it).
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, 8, 2, dtype=np.float64) / HEAD_DIM))
    angles = np.outer(np.array(POSITIONS, dtype=np.float64), inv_freq).astype(np.float32)
    x = Tensor.from_values(angles)
    print(f"positions = {POSITIONS}")
    print(f"real rotation angles (position * inv_freq), shape {x.shape}:\n{x.data}")

    # Real numeric result -- plain numpy sin/cos, no Docker involved here.
    sin_result = sin(x)
    cos_result = cos(x)
    print(f"\nreal numeric sin(angles):\n{sin_result.data}")
    print(f"real numeric cos(angles):\n{cos_result.data}")

    # Independent, live accuracy re-check against real numpy, over a much
    # wider real range than this example's own small table (same real
    # check cited in sin/_timeloop_sin.py's own module docstring).
    xs = np.linspace(-100, 100, 20001)
    approx_sin = sin(Tensor.from_values(xs.astype(np.float32))).data.astype(np.float64)
    approx_cos = cos(Tensor.from_values(xs.astype(np.float32))).data.astype(np.float64)
    real_sin = np.sin(xs)
    real_cos = np.cos(xs)
    print(f"\nlive accuracy re-check over x in [-100, 100] (20001 real points):")
    print(f"  max abs error sin: {np.max(np.abs(approx_sin - real_sin)):.3e}")
    print(f"  max abs error cos: {np.max(np.abs(approx_cos - real_cos)):.3e}")
    print(f"  (real fp32 epsilon: {np.finfo(np.float32).eps:.3e})")

    # Bridge: derive each cost-model Spec's fields FROM the real Tensor
    # result (elements_per_pe/bytes_per_element), not typed in by hand.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    sin_spec = SinSpec(elements_per_pe=sin_result.size, bytes_per_element=sin_result.bytes_per_element)
    cos_spec = CosSpec(elements_per_pe=cos_result.size, bytes_per_element=cos_result.bytes_per_element)
    print(f"\nbridged SinSpec: elements_per_pe={sin_spec.elements_per_pe}, bytes_per_element={sin_spec.bytes_per_element}")
    print(f"bridged CosSpec: elements_per_pe={cos_spec.elements_per_pe}, bytes_per_element={cos_spec.bytes_per_element}")

    # Real Docker timeloop-model runs -- sin: 4 real sub-runs
    # (mul/add/sub/bitwise); cos: those same 4 (via the real
    # cos(x)=sin(x+pi/2) identity) plus 1 more real add for the phase
    # shift. This is the only place cycles/energy/area come from; raises
    # (never fabricates) if any real Docker sub-run fails.
    sin_cost = simulate_sin(sin_spec, wafer, OUT_DIR / "sin")
    cos_cost = simulate_cos(cos_spec, wafer, OUT_DIR / "cos")

    sin_time_ns = sin_cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9
    cos_time_ns = cos_cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (sin) ===")
    print(f"cycles (real, total for {sin_spec.elements_per_pe} elements): {sin_cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                   {sin_time_ns:.3f} ns")
    print(f"energy (real, total):                                        {sin_cost.local_energy_uj} uJ "
          f"({sin_cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"cycles/element (real):                                       {sin_cost.cycles_per_element}")
    print(f"energy/element (real):                                       {sin_cost.energy_uj_per_element} uJ")
    print("--- sub-run breakdown (range reduction + degree-17 odd-power Taylor polynomial) ---")
    print(f"mul sub-run (12 real muls/element):    cycles={sin_cost.mul_result.local_compute_cycles}, "
          f"energy_uj={sin_cost.mul_result.local_energy_uj}")
    print(f"add sub-run (9 real adds/element):     cycles={sin_cost.add_result.local_compute_cycles}, "
          f"energy_uj={sin_cost.add_result.local_energy_uj}")
    print(f"sub sub-run (1 real subtract/element): cycles={sin_cost.sub_result.local_compute_cycles}, "
          f"energy_uj={sin_cost.sub_result.local_energy_uj}")
    print(f"bitwise sub-run (1 proxy firing/element, floor): "
          f"cycles={sin_cost.bitwise_result.local_compute_cycles}, energy_uj={sin_cost.bitwise_result.local_energy_uj}")

    print("\n=== REAL Docker/TimeLoop/Accelergy result (cos, via sin(x + pi/2)) ===")
    print(f"cycles (real, total for {cos_spec.elements_per_pe} elements): {cos_cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                   {cos_time_ns:.3f} ns")
    print(f"energy (real, total):                                        {cos_cost.local_energy_uj} uJ "
          f"({cos_cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"cycles/element (real):                                       {cos_cost.cycles_per_element}")
    print(f"energy/element (real):                                       {cos_cost.energy_uj_per_element} uJ")
    print(f"phase-shift add sub-run (1 real add/element): "
          f"cycles={cos_cost.phase_shift_result.local_compute_cycles}, "
          f"energy_uj={cos_cost.phase_shift_result.local_energy_uj}")
    print(f"(remaining cost = sin's own real pipeline, reused verbatim -- see breakdown above)")


if __name__ == "__main__":
    main()
