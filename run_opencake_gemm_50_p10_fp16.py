#!/usr/bin/env python3
"""Real OpenCake/wafersim GEMM run, FP16 version of run_opencake_gemm_50_p10.py:

    Precision : FP16   (bytes_per_element=2, was FP32/bytes_per_element=4)
    Matrix A/B/C : 50 x 50
    PE Grid   : 10 x 10  (Mt=Kt=Nt=5, i.e. 50 = 10 * 5)
    Total PEs : 100

WHY THIS IS A SEPARATE REAL RUN, NOT A REINTERPRETATION OF THE FP32 RESULT:
bytes_per_element flows into the real generated problem.yaml/architecture.yaml
(datawidth = bytes_per_element * 8, a_bytes/b_bytes/c_bytes tile sizes, SRAM
depth-in-words for the pe_scratchpad/fifo_buffer/router_buffer CACTI sizing --
see src/wafersim/timeloop.py lines ~216-522). Halving bytes_per_element
changes real memory footprint and can change which tile_k candidates are
SRAM-feasible, and changes Accelergy's real energy estimate for the mac/SRAM/
network components (16-bit datapath vs 32-bit). So this needs its own real
Docker timeloop-mapper + timeloop-model + Accelergy run per tile_k candidate
-- the FP32 result (outputs/cerebras_gemm_compare/gemm_50_p10/result.json)
cannot just be scaled/reused.

NOTE ON WHAT MIGHT *NOT* CHANGE: the TimeLoop "mac" component's cycle count
has, so far in this project's real runs, come out to exactly 1 cycle per MAC
regardless of datawidth (confirmed via real resolved.stats.txt for both the
100x100x100/FP32 and 50x50x50/FP32 cases: Computes(total)/Cycles = active_pes
exactly). If that holds here too, local_compute_cycles should come out the
same as the FP32 run (1,250) -- but that is a prediction to be checked against
this run's real output, not assumed. Do not treat it as known until the real
resolved.stats.txt is read.

CEREBRAS_REFERENCE cycle fields are left as None -- no real Cerebras FP16
hardware TSC numbers have been given for this exact case yet. Fill in /
compare once real hardware numbers are available, same pattern as the other
scripts in this project.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE: same standing rule as
every other real-Docker script in this project -- real Docker/TimeLoop calls
run on your own Mac terminal, where Docker actually exists.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_50_p10_fp16.py
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
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_50_p10_fp16"

CEREBRAS_REFERENCE = {
    "precision": "FP16",
    "matrix_m": 50,
    "matrix_n": 50,
    "matrix_k": 50,
    "pe_grid": "10x10",
    "total_pes": 100,
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

    gemm = GemmSpec(m=50, n=50, k=50, bytes_per_element=2)  # FP16
    wafer = WaferSpec(mesh_rows=10, mesh_cols=10)  # pinned to match PE Grid 10x10

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element} (FP16)")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols} "
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

    print("\nCompare against the FP32 result at "
          "outputs/cerebras_gemm_compare/gemm_50_p10/result.json once this is done --")
    print("same GEMM size and PE grid, only bytes_per_element differs (2 vs 4).")
    print("If you have real Cerebras FP16 hardware TSC numbers for this exact case,")
    print("send them and I'll fill in cerebras_reference and run the comparison.")


if __name__ == "__main__":
    main()
