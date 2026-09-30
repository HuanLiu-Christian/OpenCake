#!/usr/bin/env python3
"""Real OpenCake/wafersim GEMM run, FP16, between the previous two scales:

    Precision : FP16   (bytes_per_element=2)
    Matrix A/B/C : 250 x 250
    PE Grid   : 50 x 50  (Mt=Nt=5, i.e. 250 = 50 * 5)
    Total PEs : 2,500

NOTE ON THE CONFIG YOU SENT -- same pattern as the 500x500x500 case:
your message said "PE Grid: 50x50" but also "Total PEs: 100", which are
inconsistent (50x50 = 2,500 PEs, not 100). Using the same internal-
consistency check as last time: M=250 with mesh 50x50 gives tile_m =
250/50 = 5, matching the Mt=5 you gave; a 10x10/100-PE reading would give
tile_m=25, contradicting Mt=5. So this script uses the literal 50x50 grid =
2,500 PEs (same resolution you confirmed for the 500x500x500 case). Flagging
this again rather than silently assuming -- let me know if you actually
meant something else.

SCALE CONTEXT: 2,500 active PEs is 6.25x more than the 50x50x50/10x10 case
(100 PEs) but 4x fewer than the 500x500x500/100x100 case you just ran
(10,000 PEs, took 121.3s real wall time for all candidates). Rough
extrapolation only -- expect this one to land somewhere between the two,
plausibly well under the 500^3 case's ~2 minutes, but this isn't a
guarantee.

CEREBRAS_REFERENCE cycle fields are left as None -- no real Cerebras
hardware TSC numbers have been given for this exact 250x250x250/FP16 case
yet.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE: same standing rule as
every other real-Docker script in this project.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_250_p50_fp16.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from wafersim.primitives import (
    GemmSpec,
    WaferSpec,
    best_by_sim_time,
    search_tile_k,
)

REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_250_p50_fp16"

CEREBRAS_REFERENCE = {
    "precision": "FP16",
    "matrix_m": 250,
    "matrix_n": 250,
    "matrix_k": 250,
    "pe_grid": "50x50",
    "total_pes": 2500,
    "Mt": 5,
    "Kt": 5,
    "Nt": 5,
    "broadcast_cycles_min": None,
    "broadcast_cycles_max": None,
    "compute_cycles_min": None,
    "compute_cycles_max": None,
    "total_cycles_min": None,
    "total_cycles_max": None,
    "max_per_pe_gemm_cycles": None,
}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    gemm = GemmSpec(m=250, n=250, k=250, bytes_per_element=2)  # FP16
    wafer = WaferSpec(mesh_rows=50, mesh_cols=50)  # 50x50 = 2,500 PEs

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element} (FP16)")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols} = {wafer.mesh_rows * wafer.mesh_cols:,} PEs "
          f"(sram_kib_per_pe={wafer.sram_kib_per_pe}, "
          f"fifo_capacity_packets={wafer.fifo_capacity_packets}, "
          f"link_bandwidth_bytes_per_cycle={wafer.link_bandwidth_bytes_per_cycle}, "
          f"clock_ghz={wafer.clock_ghz})")
    print("Running real Docker timeloop-model once per SRAM-feasible tile_k candidate ...")

    t0 = time.monotonic()
    candidates = search_tile_k(gemm, wafer, OUT_ROOT)
    elapsed = time.monotonic() - t0

    print(f"\n{len(candidates)} real tile_k candidate(s) evaluated in {elapsed:.1f}s wall time:")
    for c in candidates:
        print(
            f"  tile_k={c.tile_k:>5}  k_phases={c.result.contract.k_phases:>3}  "
            f"sim_time_cycles={c.result.sim_time_cycles:>10,}  "
            f"total_energy_uj={c.result.energy.total_uj:>14,.4f}"
        )

    winner = best_by_sim_time(candidates)
    contract = winner.result.contract

    result = {
        "gemm_m": gemm.m,
        "gemm_n": gemm.n,
        "gemm_k": gemm.k,
        "bytes_per_element": gemm.bytes_per_element,
        "precision": "FP16",
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
        "opencake_sim_time_cycles": winner.result.sim_time_cycles,
        "opencake_total_energy_uj": winner.result.energy.total_uj,
        "opencake_local_compute_cycles": contract.local_compute_cycles,
        "opencake_network_packet_hops": winner.result.network["packet_hops"],
        "opencake_mean_pe_utilization": winner.result.mean_pe_utilization,
        "wall_clock_seconds_all_candidates": elapsed,
        "cerebras_reference": CEREBRAS_REFERENCE,
    }

    out_json = OUT_ROOT / "result.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("WINNING real tile_k candidate (lowest full-system sim_time_cycles):")
    print(f"  active_rows x active_cols = {contract.active_rows} x {contract.active_cols} "
          f"({contract.active_pes:,} active PEs of {wafer.mesh_rows * wafer.mesh_cols:,} in the mesh)")
    print(f"  tile_m={contract.tile_m}  tile_n={contract.tile_n}  tile_k={contract.tile_k}  "
          f"k_phases={contract.k_phases}")
    print(f"  real local_compute_cycles (TimeLoop, per active PE) = {contract.local_compute_cycles:,}")
    print(f"  real OpenCake sim_time_cycles (compute + network)   = {winner.result.sim_time_cycles:,}")
    print(f"  real OpenCake total_energy_uj                       = {winner.result.energy.total_uj:,.4f}")
    print(f"  total wall-clock time for all candidates            = {elapsed:.1f}s")
    print(f"\nwrote {out_json}")
    print("\nIf you have real Cerebras FP16 hardware TSC numbers for this exact case,")
    print("send them and I'll fill in cerebras_reference and run the comparison.")


if __name__ == "__main__":
    main()
