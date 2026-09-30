#!/usr/bin/env python3
"""Real Docker timeloop-model run: cycles/energy/area for ONE elementwise
divide (950 / 1050 -- a single scalar pair, i.e. elements_per_pe=1).

Uses this project's real, Docker-verified div cost model (simulate_div()),
which is COMPOSED from two real Docker sub-runs -- simulate_mul() for the
Newton-Raphson reciprocal's multiply firings, simulate_add() for its
subtract/combine firings -- since no real Accelergy divider-circuit
component exists in this project's pinned Docker toolchain (see
wafersim/primitives/div/_timeloop_div.py's module docstring for the full
real-cited investigation and the multiply+add decomposition this project
chose in its place). No synthetic fallback -- if either real Docker
sub-run fails, this raises.

Note: the numeric result (950 / 1050 = 0.9047619...) is NOT computed by
this primitive -- div(a, b) itself is still a stub (depends on the
still-stub Tensor class). This script only runs the real cost model:
cycles/energy/area for one elementwise divide op.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_div_950_1050.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.div import DivSpec, simulate_div
from wafersim.specs import WaferSpec

A, B = 950, 1050
OUT_DIR = Path("outputs/div_950_1050")


def main() -> None:
    div_spec = DivSpec(elements_per_pe=1, bytes_per_element=4)
    wafer = WaferSpec()

    print(f"Running real Docker timeloop-model for one divide ({A} / {B}) ...")
    print("(this runs TWO real Docker sub-pipelines: mul, then add -- see")
    print(" _timeloop_div.py's module docstring for why)")
    result = simulate_div(div_spec, wafer, OUT_DIR)

    print("\n=== REAL result (composed from real mul + add sub-runs) ===")
    print(f"local_compute_cycles (mul+add) = {result.local_compute_cycles}")
    print(f"local_energy_uj (mul+add)      = {result.local_energy_uj}")
    print(f"multiplier_area_um2            = {result.multiplier_area_um2}")
    print(f"adder_area_um2                 = {result.adder_area_um2}")
    print(f"pe_scratchpad_area_um2 (mul)   = {result.mul_result.pe_scratchpad_area_um2}")
    print(f"pe_scratchpad_area_um2 (add)   = {result.add_result.pe_scratchpad_area_um2}")

    print("\n--- mul sub-run (Newton-Raphson multiplies) ---")
    print(f"mul stats.txt: {result.mul_result.stats_path}")
    print(f"mul ART.yaml : {result.mul_result.art_path}")

    print("\n--- add sub-run (subtract/combine) ---")
    print(f"add stats.txt: {result.add_result.stats_path}")
    print(f"add ART.yaml : {result.add_result.art_path}")


if __name__ == "__main__":
    main()
