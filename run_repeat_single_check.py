#!/usr/bin/env python3
"""Real single-primitive check for `repeat`: one real Tensor -> repeat(x,
repeats, dim) (real numeric result, real numpy.repeat) -> RepeatSpec.
from_result() (bridged from the real OUTPUT tensor's size) ->
simulate_repeat() -> real cycles/time/energy/area from one real Docker
`timeloop-model` run against the shared real ``bitwise`` component (see
``wafersim.primitives._memory_move``'s own module docstring: a real
identity-bitwise-pass-through per element -- there is no dedicated DMA/
copy-engine component in this project's pinned Docker toolchain).

Unlike `reshape` (whose real copy-vs-view outcome is conditional on the
specific call), `repeat` is UNCONDITIONAL -- every real call physically
duplicates data, every time (see
``wafersim.primitives.repeat_expand._timeloop_repeat``'s own module
docstring), so no permute trick is needed here to force the copying
branch -- a plain repeat() call already triggers a real Docker run.

Used for: grouped-query attention (GQA) -- broadcasting fewer K/V heads up
to match the Q head count, per this primitive's own package docstring.
This example mirrors that real use case at toy scale: 2 KV heads (each a
4-dim vector) repeated 3x to produce 6 heads.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_repeat_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives.repeat_expand import RepeatSpec, repeat, simulate_repeat
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/repeat_single_check")


def main() -> None:
    # 2 "KV heads", each a 4-dim vector -- GQA-style toy example.
    a = Tensor.from_values([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # shape (2, 4)
    print(f"a = {a.data.tolist()}  shape={a.shape}")

    # Real numeric result -- plain numpy.repeat, no Docker involved here.
    result = repeat(a, repeats=3, dim=0)
    print(f"real numeric result repeat(a, repeats=3, dim=0) = {result.data.tolist()}  shape={result.shape}")

    # Bridge: derived from the real OUTPUT tensor's size (every one of
    # those output elements is a real physical write, even the
    # duplicated ones) -- not from the input.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = RepeatSpec.from_result(result)
    print(f"bridged RepeatSpec: elements_per_pe={spec.elements_per_pe}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run -- this is the only place cycles/
    # energy/area come from; raises (never fabricates) on failure.
    cost = simulate_repeat(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (repeat, always-copying) ===")
    print(f"cycles (real, total for {spec.elements_per_pe} elements): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                     {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"bitwise (bitwise_gate) area (real):                       {cost.bitwise_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real):                                    {cost.cycles_per_element}")
    print(f"energy/element (real):                                    {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
