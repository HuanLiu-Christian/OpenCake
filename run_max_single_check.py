#!/usr/bin/env python3
"""Real single-primitive check for `max`: one real Tensor -> max(x, dim)
(real numeric reduction) -> MaxSpec(from the real input's reduced axis)
-> simulate_max() -> real cycles/time/energy/area from one real Docker
`timeloop-model` run against a real, DEDICATED ``comparator`` Accelergy
component (see ``wafersim.primitives.max._timeloop_max``'s own module
docstring: unlike `sum` [which reuses `add`'s fpadder], `max` needed a
genuinely NEW component -- confirmed via the real upstream
``accelergy-aladdin-plug-in`` source, backed by real ``data/comparator.csv``
-- but reducing K elements to their max is still EXACTLY K-1 real
sequential comparator firings, an exact operation count, not an
estimate).

Note the real bridge formula is the same shape as `sum`'s:
``MaxSpec(elements_per_pe=x.shape[dim], ...)`` -- the real size of the
axis being reduced, not ``result.size``.

Also note the real K=1 degenerate case (the max of a single element needs
zero real comparisons and skips Docker entirely) does NOT apply here:
this example reduces K=4 elements per row, a real, non-degenerate
Docker-triggering case.

No synthetic fallback anywhere: if Docker/TimeLoop fails, this raises and
prints nothing fabricated.

Usage (from the project root, with your venv active, Docker Desktop
open):
  source .venv/bin/activate
  python3 run_max_single_check.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives import WaferSpec
from wafersim.primitives import max as t_max
from wafersim.primitives.max import MaxSpec, simulate_max
from wafersim.primitives.tensor import Tensor

OUT_DIR = Path("outputs/max_single_check")


def main() -> None:
    a = Tensor.from_values([[3.0, 7.0, 1.0, 5.0], [9.0, 2.0, 8.0, 4.0]])  # shape (2, 4)
    print(f"a = {a.data.tolist()}  shape={a.shape}")

    # Real numeric result -- plain numpy max(axis=1, keepdims=True), no
    # Docker involved here.
    result = t_max(a, dim=1)
    print(f"real numeric result max(a, dim=1) = {result.data.tolist()}  shape={result.shape}")

    # Bridge: K is the real size of the axis being reduced
    # (a.shape[dim]), NOT result.size -- same pattern as `sum`.
    wafer = WaferSpec()  # default clock_ghz = 1.0
    spec = MaxSpec(elements_per_pe=a.shape[1], bytes_per_element=a.bytes_per_element)
    print(f"bridged MaxSpec: elements_per_pe (K)={spec.elements_per_pe}, "
          f"real_compare_count (K-1)={spec.real_compare_count}, bytes_per_element={spec.bytes_per_element}")

    # Real Docker timeloop-model run (K-1=3 real sequential comparator
    # firings) -- this is the only place cycles/energy/area come from;
    # raises (never fabricates) on failure.
    cost = simulate_max(spec, wafer, OUT_DIR)

    real_time_ns = cost.local_compute_cycles / (wafer.clock_ghz * 1e9) * 1e9

    print("\n=== REAL Docker/TimeLoop/Accelergy result (max, K-1 real comparisons) ===")
    print(f"cycles (real, total for K-1={spec.real_compare_count} comparisons): {cost.local_compute_cycles}")
    print(f"time (real, @ {wafer.clock_ghz} GHz clock):                        {real_time_ns:.3f} ns")
    print(f"energy (real, total):                                             {cost.local_energy_uj} uJ "
          f"({cost.local_energy_uj * 1e6:.3f} pJ)")
    print(f"comparator area (real):                                           {cost.comparator_area_um2} um^2")
    print(f"pe_scratchpad area (real):                                        {cost.pe_scratchpad_area_um2} um^2")
    print(f"cycles/element (real, divided by K={spec.elements_per_pe}):         {cost.cycles_per_element}")
    print(f"energy/element (real):                                            {cost.energy_uj_per_element} uJ")


if __name__ == "__main__":
    main()
