#!/usr/bin/env python3
"""Real Docker timeloop-model run: cycles/energy/area for ONE reduction-sum
over 4 elements (e.g. softmax's normalization denominator, or RMSNorm's
mean-of-squares reduction, over a small representative slice).

Uses this project's real, Docker-verified sum cost model (simulate_sum()),
which reduces K elements via K-1 real, sequential `add` firings -- i.e.
add's own already-Docker-verified fpadder component, reused exactly K-1
times (an EXACT operation count from the definition of reduction, not an
approximation -- see wafersim/primitives/sum/_timeloop_sum.py's module
docstring for the full derivation and the "sequential, not tree,
reduction" modeling choice). No synthetic fallback -- if the real Docker
sub-run fails, this raises.

Note: the numeric result itself is NOT computed by this primitive --
sum(x, dim) is still a stub (depends on the still-stub Tensor class).
This script only runs the real cost model: cycles/energy/area for one
4-element reduction.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_sum_4elements.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.sum import SumSpec, simulate_sum
from wafersim.specs import WaferSpec

VALUES = [950, 1050, 200, 300]
OUT_DIR = Path("outputs/sum_4elements")


def main() -> None:
    sum_spec = SumSpec(elements_per_pe=len(VALUES), bytes_per_element=4)
    wafer = WaferSpec()

    print(
        f"Running real Docker timeloop-model for sum({VALUES}) "
        f"({sum_spec.real_add_count} real sequential adds) ..."
    )
    result = simulate_sum(sum_spec, wafer, OUT_DIR)

    print("\n=== REAL result (composed from a real add sub-run) ===")
    print(f"real_add_count (K-1)       = {sum_spec.real_add_count}")
    print(f"local_compute_cycles       = {result.local_compute_cycles}")
    print(f"local_energy_uj            = {result.local_energy_uj}")
    print(f"adder_area_um2             = {result.adder_area_um2}")
    print(f"pe_scratchpad_area_um2     = {result.pe_scratchpad_area_um2}")

    if result.add_result is not None:
        print(f"\nadd sub-run stats.txt: {result.add_result.stats_path}")
        print(f"add sub-run ART.yaml : {result.add_result.art_path}")
    else:
        print("\n(no real Docker sub-run launched -- K=1 needs zero real adds)")


if __name__ == "__main__":
    main()
