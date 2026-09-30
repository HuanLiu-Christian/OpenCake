#!/usr/bin/env python3
"""Calibration/validation run: forces OpenCake onto the SAME K-tiling
structure Cerebras's real SUMMA reference kernel used for the 100x100x100 /
20x20 case (Kt=5, i.e. K split into 5 phases of 20 each), instead of letting
search_tile_k() pick its own free-search winner (which was tile_k=100,
k_phases=1 -- see run_opencake_gemm_100_p20.py).

WHY THIS RUN EXISTS -- READ BEFORE USING THE NUMBER:
The free-search comparison showed OpenCake predicting ~2,752 cycles against
Cerebras's real, hardware-measured 31,367-31,641 cycles for this GEMM (an
~11.5x gap). That comparison is NOT evidence that OpenCake's design is
better, because the two sides ran genuinely different algorithms (OpenCake:
1 phase, no K-split at all; Cerebras: 5 phases, forced by Kt=K/P grid
geometry). A simulator's prediction for an algorithm nobody measured on
real silicon cannot by itself prove anything about real silicon.

What CAN be tested here: pin OpenCake to the exact same 5-phase structure
Cerebras actually used (tile_k=20, matching Kt=5), and compare OpenCake's
predicted cycles against Cerebras's real measured cycles UNDER THAT SAME
ALGORITHM. This isolates "is OpenCake's compute+network cost model an
accurate predictor of real Cerebras hardware behavior" from "did OpenCake
pick a different (possibly better) tiling than Cerebras did" -- two
separate questions that the free-search run conflated.

How to read the two results together:
  - If THIS run's cycles land close to Cerebras's real 31,367-31,641 (same
    algorithm), that's real evidence OpenCake's cost model is trustworthy.
    Only then does the free-search run's ~2,752-cycle prediction for the
    untested 1-phase alternative become meaningful evidence that avoiding
    the K-split really would be faster on real hardware too -- because now
    you have grounds to trust the model's predictions in this regime.
  - If THIS run's cycles are still far from Cerebras's real numbers even
    under the SAME algorithm, that means OpenCake's cost model itself has
    a real calibration gap (missing overhead, wrong per-hop timing, etc.),
    and the free-search run's 1-phase number cannot be trusted as evidence
    of anything until that gap is understood -- the 11.5x free-search gap
    would then be mostly "model inaccuracy", not "your tiling choice is
    smarter than Cerebras's."
Either outcome is a real, honest answer -- this run is what actually
decides which one you're looking at.

Bypasses search_tile_k() (which searches all SRAM-feasible tile_k) and
calls the lower-level generate_timeloop_inputs(..., tile_k=20) /
run_timeloop_model / parse_mapping_contract / simulate_gemm directly, one
real Docker call, exactly the same real machinery search_tile_k uses per
candidate -- just pinned to one specific candidate instead of searching.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE: same standing rule as
every other real-Docker script in this project.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_100_p20_tilek20.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from wafersim.specs import GemmSpec, WaferSpec
from wafersim.timeloop import (
    MODEL_MODE,
    generate_timeloop_inputs,
    parse_mapping_contract,
    run_timeloop_model,
)
from wafersim.wafer.simulator import simulate_gemm

REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_100_p20_tilek20"

CEREBRAS_REFERENCE = {
    "precision": "FP32",
    "matrix_m": 100,
    "matrix_n": 100,
    "matrix_k": 100,
    "pe_grid": "20x20",
    "total_pes": 400,
    "Mt": 5,
    "Kt": 5,
    "Nt": 5,
    "broadcast_cycles_min": 12020,
    "broadcast_cycles_max": 12294,
    "compute_cycles_min": 17480,
    "compute_cycles_max": 17480,
    "total_cycles_min": 31367,
    "total_cycles_max": 31641,
    "max_per_pe_gemm_cycles": 31641,
}

PINNED_TILE_K = 20  # = K / Kt = 100 / 5, matching Cerebras's Kt=5 exactly


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    gemm = GemmSpec(m=100, n=100, k=100, bytes_per_element=4)
    wafer = WaferSpec(mesh_rows=20, mesh_cols=20)

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element}")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols}")
    print(f"PINNED tile_k={PINNED_TILE_K} (k_phases={gemm.k // PINNED_TILE_K}) "
          f"-- forced to match Cerebras's Kt=5, NOT search_tile_k's free choice")
    print("Running ONE real Docker timeloop-model call against this pinned mapping ...")

    t0 = time.monotonic()
    artifacts = generate_timeloop_inputs(gemm, wafer, OUT_ROOT, tile_k=PINNED_TILE_K)
    run_timeloop_model(artifacts)
    contract = parse_mapping_contract(artifacts, gemm, wafer, mode=MODEL_MODE)
    result = simulate_gemm(contract)
    elapsed = time.monotonic() - t0

    print(f"\nDone in {elapsed:.1f}s wall time.")
    print(f"  tile_m={contract.tile_m}  tile_n={contract.tile_n}  tile_k={contract.tile_k}  "
          f"k_phases={contract.k_phases}")
    print(f"  real local_compute_cycles (TimeLoop, per active PE) = {contract.local_compute_cycles:,}")
    print(f"  real OpenCake sim_time_cycles (compute + network)   = {result.sim_time_cycles:,}")
    print(f"  real OpenCake total_energy_uj                       = {result.energy.total_uj:,.4f}")

    out = {
        "gemm_m": gemm.m,
        "gemm_n": gemm.n,
        "gemm_k": gemm.k,
        "bytes_per_element": gemm.bytes_per_element,
        "mesh_rows": wafer.mesh_rows,
        "mesh_cols": wafer.mesh_cols,
        "sram_kib_per_pe": wafer.sram_kib_per_pe,
        "fifo_capacity_packets": wafer.fifo_capacity_packets,
        "active_rows": contract.active_rows,
        "active_cols": contract.active_cols,
        "active_pes": contract.active_pes,
        "tile_m": contract.tile_m,
        "tile_n": contract.tile_n,
        "tile_k": contract.tile_k,
        "k_phases": contract.k_phases,
        "pinned": True,
        "opencake_sim_time_cycles": result.sim_time_cycles,
        "opencake_total_energy_uj": result.energy.total_uj,
        "opencake_local_compute_cycles": contract.local_compute_cycles,
        "opencake_network_packet_hops": result.network["packet_hops"],
        "opencake_mean_pe_utilization": result.mean_pe_utilization,
        "wall_clock_seconds": elapsed,
        "cerebras_reference": CEREBRAS_REFERENCE,
    }
    out_json = OUT_ROOT / "result.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")

    print("\n" + "=" * 74)
    print("SAME-ALGORITHM comparison (both sides: 5 K-phases, tile_k=20/Kt=5)")
    print("=" * 74)
    print(f"OpenCake real sim_time_cycles : {result.sim_time_cycles:,.0f}")
    print(f"Cerebras real total cycles    : {CEREBRAS_REFERENCE['total_cycles_min']:,} - "
          f"{CEREBRAS_REFERENCE['total_cycles_max']:,}")
    ratio_min = result.sim_time_cycles / CEREBRAS_REFERENCE["total_cycles_min"]
    ratio_max = result.sim_time_cycles / CEREBRAS_REFERENCE["total_cycles_max"]
    print(f"ratio (OpenCake / Cerebras)   : {ratio_min:.3f}x - {ratio_max:.3f}x")
    print("\nThis ratio, not the free-search one, is the real test of whether")
    print("OpenCake's cost model is calibrated against real Cerebras hardware.")


if __name__ == "__main__":
    main()
