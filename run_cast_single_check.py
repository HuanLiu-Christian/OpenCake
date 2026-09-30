#!/usr/bin/env python3
"""Real single-primitive check for `cast`: one real fp32 Tensor ->
cast(x, np.float16) (real numeric result, real numpy.astype) ->
CastSpec(from the real INPUT tensor -- see below) -> simulate_cast() ->
real cycles/time/energy/area from TWO real Docker `timeloop-model`
sub-runs (bitwise/intadder), composing a real IEEE-754 fp32->fp16
unpack/rebias/round(-to-nearest-even)/repack bit-manipulation
decomposition (see ``wafersim.primitives.cast._timeloop_cast``'s own
module docstring for the full real-source derivation, and
``tests/test_cost_model_numeric_approximation.py`` for an independent,
bit-for-bit check of a manual implementation of this exact algorithm
against real numpy's own fp16 cast, across these SAME test values, which
matched exactly).

There is no dedicated floating-point-format-converter circuit anywhere in
this project's pinned Docker toolchain, so this decomposes into: 3 real
``bitwise``-proxy firings/element (unpack, round-test, repack -- genuine
bit-level mask/shift/test steps) + 2 real ``intadder`` firings/element
(exponent rebias, round-to-nearest-even increment -- genuine integer
arithmetic).

Note the bridge uses the real INPUT tensor's ``bytes_per_element`` (4,
fp32) rather than the output's (2, fp16) -- the cost model is
deliberately modeled at the WIDER register width, since that is what the
real unpack/round/repack bit manipulation actually operates over for a
narrowing conversion (see that module's own "BYTES_PER_ELEMENT CHOICE"
docstring section).

Used for: mixed-precision execution (e.g. running softmax in fp32 while
the rest of the model runs in a lower precision), per this primitive's
own package docstring.

No synthetic fallback anywhere: if Docker/TimeLoop fails for either
sub-run, this raises and prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_cast_single_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wafersim.primitives import CastSpec, WaferSpec, cast, simulate_cast
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/cast_single_check")


def main() -> None:
    a = Tensor.from_values([1.0, 3.14159, -65000.0, 0.001])  # real fp32
    print(f"a (fp32) = {a.data.tolist()}  dtype={a.data.dtype}")

    # Real numeric result -- plain numpy astype(float16), no Docker
    # involved here.
    result = cast(a, np.float16)
    print(f"real numeric result cast(a, np.float16) = {result.data.tolist()}  dtype={result.data.dtype}")

    # Bridge: bytes_per_element comes from the real INPUT tensor (fp32,
    # the wider register width the bit manipulation actually operates
    # over), not the fp16 output.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = CastSpec(elements_per_pe=a.size, bytes_per_element=a.bytes_per_element)
    print(f"bridged CastSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model runs (2 sub-runs: bitwise, intadder) --
    # this is the only place cycles/energy/area come from; raises (never
    # fabricates) if either sub-run fails.
    cost = simulate_cast(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (cast, fp32->fp16) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"bitwise (bitwise_gate) area (real):                       {cost.bitwise_result.bitwise_area_um2} um^2")
    print(f"intadder (intadder_gate) area (real):                     {cost.intadder_result.intadder_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")

    print("\n--- sub-run breakdown: unpack/round-test/repack (bitwise) + rebias/round-increment (intadder) ---")
    print(f"bitwise sub-run (3 proxy firings/element):  "
          f"cycles={cost.bitwise_result.local_compute_cycles}, energy_uj={cost.bitwise_result.local_energy_uj}")
    print(f"intadder sub-run (2 real firings/element):  "
          f"cycles={cost.intadder_result.local_compute_cycles}, energy_uj={cost.intadder_result.local_energy_uj}")


if __name__ == "__main__":
    main()
