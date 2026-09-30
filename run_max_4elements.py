#!/usr/bin/env python3
"""Real Docker timeloop-model run: cycles/energy/area for ONE max-reduction
over 4 elements (e.g. softmax's numerical-stability row-max, over a small
representative slice).

Uses this project's real, Docker-verified max cost model (simulate_max()),
which reduces K elements via K-1 real, sequential `comparator` firings --
a real, separately-characterized Accelergy component (NOT add's fpadder or
mul's fpmultiplier reused as an approximation -- see
wafersim/primitives/max/_timeloop_max.py's module docstring for the full
real-source citation of the real data/comparator.csv table). K-1 is an
EXACT operation count from the definition of reduction, not an
approximation. No synthetic fallback -- if the real Docker sub-run fails,
this raises.

Note: the numeric result itself is NOT computed by this primitive --
max(x, dim) is still a stub (depends on the still-stub Tensor class).
This script only runs the real cost model: cycles/energy/area for one
4-element max reduction.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_max_4elements.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.max import MaxSpec, simulate_max
from wafersim.specs import WaferSpec

VALUES = [950, 1050, 200, 300]
OUT_DIR = Path("outputs/max_4elements")


def main() -> None:
    max_spec = MaxSpec(elements_per_pe=len(VALUES), bytes_per_element=4)
    wafer = WaferSpec()

    print(
        f"Running real Docker timeloop-model for max({VALUES}) "
        f"({max_spec.real_compare_count} real sequential comparisons) ..."
    )
    result = simulate_max(max_spec, wafer, OUT_DIR)

    print("\n=== REAL result (from a real comparator sub-run) ===")
    print(f"real_compare_count (K-1)   = {max_spec.real_compare_count}")
    print(f"local_compute_cycles       = {result.local_compute_cycles}")
    print(f"local_energy_uj            = {result.local_energy_uj}")
    print(f"comparator_area_um2        = {result.comparator_area_um2}")
    print(f"pe_scratchpad_area_um2     = {result.pe_scratchpad_area_um2}")

    if result.compare_result is not None:
        print(f"\ncompare sub-run stats.txt: {result.compare_result.stats_path}")
        print(f"compare sub-run ART.yaml : {result.compare_result.art_path}")
    else:
        print("\n(no real Docker sub-run launched -- K=1 needs zero real comparisons)")


if __name__ == "__main__":
    main()
