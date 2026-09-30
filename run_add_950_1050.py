#!/usr/bin/env python3
"""Real Docker timeloop-model run: cycles/energy for ONE elementwise add
(950 + 1050 -- a single scalar pair, i.e. elements_per_pe=1).

Uses this project's real, Docker-verified add cost model (simulate_add()).
No synthetic fallback -- if Docker/TimeLoop fails, this raises.

Usage (run from the project root, with your usual venv active, Docker
Desktop open):
  source .venv/bin/activate
  python3 run_add_950_1050.py
"""

from __future__ import annotations

from pathlib import Path

from wafersim.primitives.add import AddSpec, simulate_add
from wafersim.specs import WaferSpec

A, B = 950, 1050
OUT_DIR = Path("outputs/add_950_1050")


def main() -> None:
    add_spec = AddSpec(elements_per_pe=1, bytes_per_element=4)
    wafer = WaferSpec()

    print(f"Running real Docker timeloop-model for one add ({A} + {B}) ...")
    result = simulate_add(add_spec, wafer, OUT_DIR)

    print("\n=== REAL result (from real stats.txt / ART.yaml) ===")
    print(f"local_compute_cycles       = {result.local_compute_cycles}")
    print(f"local_energy_uj            = {result.local_energy_uj}")
    print(f"adder_area_um2 (1 fpadder) = {result.adder_area_um2}")
    print(f"pe_scratchpad_area_um2     = {result.pe_scratchpad_area_um2}")
    print(f"\nstats.txt: {result.stats_path}")
    print(f"ART.yaml : {result.art_path}")


if __name__ == "__main__":
    main()
