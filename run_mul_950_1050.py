#!/usr/bin/env python3
"""Real Docker timeloop-model run: cycles/energy/area for ONE elementwise
multiply (950 * 1050 -- a single scalar pair, i.e. elements_per_pe=1).

Uses this project's real, Docker-verified mul cost model (simulate_mul()).
No synthetic fallback -- if Docker/TimeLoop fails, this raises.

Note: the numeric result (950 * 1050 = 997500) is NOT computed by this
primitive -- mul(a, b) itself is still a stub (depends on the still-stub
Tensor class). This script only runs the real cost model: cycles/energy/
area for one elementwise multiply op.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_mul_950_1050.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.mul import MulSpec, simulate_mul
from wafersim.specs import WaferSpec

A, B = 950, 1050
OUT_DIR = Path("outputs/mul_950_1050")


def main() -> None:
    mul_spec = MulSpec(elements_per_pe=1, bytes_per_element=4)
    wafer = WaferSpec()

    print(f"Running real Docker timeloop-model for one multiply ({A} * {B}) ...")
    result = simulate_mul(mul_spec, wafer, OUT_DIR)

    print("\n=== REAL result (from real stats.txt / ART.yaml) ===")
    print(f"local_compute_cycles       = {result.local_compute_cycles}")
    print(f"local_energy_uj            = {result.local_energy_uj}")
    print(f"multiplier_area_um2        = {result.multiplier_area_um2}")
    print(f"pe_scratchpad_area_um2     = {result.pe_scratchpad_area_um2}")
    print(f"\nstats.txt: {result.stats_path}")
    print(f"ART.yaml : {result.art_path}")


if __name__ == "__main__":
    main()
