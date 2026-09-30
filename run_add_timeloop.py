#!/usr/bin/env python3
"""Real Docker run of the new `add` primitive's TimeLoop/Accelergy cost model.

Calls wafersim.primitives.add.simulate_add() end to end: generates a real
single-PE TimeLoop problem/architecture/mapping for Z[N] = X[N] + Y[N]
(N=1024, arbitrary -- see below), runs the real `timeloop-model` binary
inside Docker against it, and parses the real stats.txt/ART.yaml output.
No synthetic fallback -- if Docker/TimeLoop fails, this raises and prints
nothing fabricated.

Also prints the real cycles-per-element and energy-per-element rate this
run measured, the same way wafer/simulator.py derives a real cycles/MAC
rate from GEMM's local_compute_cycles (see that module's comment) --
useful for comparing against the real 1.0 cycle/MAC rate already confirmed
for this project's `intmac` GEMM pipeline.

Usage (run from the project root, with your usual venv active):
  source .venv/bin/activate
  python3 run_add_timeloop.py            # N=1024 (default)
  python3 run_add_timeloop.py --n 8      # customize N, e.g. for a quick
                                          # cheap ablation while debugging
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wafersim.primitives.add import AddSpec, simulate_add
from wafersim.specs import WaferSpec

DEFAULT_ELEMENTS_PER_PE = 1024  # arbitrary; see cycles_per_element/energy_uj_per_element
OUT_DIR = Path("outputs/add_primitive_check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=DEFAULT_ELEMENTS_PER_PE,
        help=f"elements_per_pe for AddSpec (default: {DEFAULT_ELEMENTS_PER_PE})",
    )
    args = parser.parse_args()

    add = AddSpec(elements_per_pe=args.n)
    wafer = WaferSpec()  # defaults: 45nm, clock_ghz=1.0, bytes_per_element via AddSpec

    print(f"Running real Docker timeloop-model for add: N={args.n} elements, "
          f"bytes_per_element={add.bytes_per_element} (fp32), technology={wafer.technology_node} ...")
    result = simulate_add(add, wafer, OUT_DIR)

    print("\n=== REAL result (from real stats.txt / ART.yaml) ===")
    print(f"local_compute_cycles       = {result.local_compute_cycles}")
    print(f"local_energy_uj            = {result.local_energy_uj}")
    print(f"cycles_per_element         = {result.cycles_per_element}")
    print(f"energy_uj_per_element      = {result.energy_uj_per_element}")
    print(f"adder_area_um2 (1 fpadder) = {result.adder_area_um2}")
    print(f"pe_scratchpad_area_um2     = {result.pe_scratchpad_area_um2}")
    print(f"\nstats.txt: {result.stats_path}")
    print(f"ART.yaml : {result.art_path}")


if __name__ == "__main__":
    main()
