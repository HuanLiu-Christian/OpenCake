#!/usr/bin/env python3
"""Real OpenCake/wafersim GEMM run, FP16, larger scale than the previous
50x50x50/10x10 case:

    Precision : FP16   (bytes_per_element=2)
    Matrix A/B/C : 500 x 500
    PE Grid   : 100 x 100  (Mt=Nt=5, i.e. 500 = 100 * 5)
    Total PEs : 10,000

NOTE ON THE CONFIG YOU SENT -- READ BEFORE RUNNING:
Your message said "PE Grid: 100x100" but also "Total PEs: 100", which are
inconsistent (100x100 = 10,000 PEs, not 100). You confirmed via the
clarifying question that you want the literal 100x100 grid = 10,000 PEs --
this script uses that. It's also the internally-consistent reading: with
M=N=500 and mesh 100x100, tile_m = 500/100 = 5, which matches the Mt=5 you
gave; the 100-PE/10x10 reading would have implied tile_m=50, contradicting
Mt=5.

WHY THIS MAY TAKE NOTICEABLY LONGER THAN YOUR PREVIOUS RUNS:
Every earlier GEMM run in this project topped out at 400 active PEs (the
100x100x100/20x20 case), which took ~62s wall time for all 9 tile_k
candidates. This run uses 10,000 active PEs -- 25x more. The one real data
point this project has on how real Docker timeloop-mapper/timeloop-model
runtime scales with PE count is the abandoned 3775x3775x3775/755x755 run
(570,025 PEs), which took 7+ real hours. Extrapolating VERY roughly from
those two real data points (400 PEs -> ~62s, 570,025 PEs -> 7+ hours)
suggests this 10,000-PE run could plausibly take anywhere from a few minutes
to tens of minutes for all real tile_k candidates -- this is a rough
extrapolation, not a guarantee, since real Docker/TimeLoop runtime depends
on more than just PE count. Don't assume it's stuck if it runs longer than
your previous 50x50x50 experiments; if it's still going after a long while,
tell me and I'll check real docker stats/docker top on your machine (I can't
run Docker myself, but I can inspect a running container's real state
through you).

CEREBRAS_REFERENCE cycle fields are left as None -- no real Cerebras
hardware TSC numbers have been given for this exact 500x500x500/FP16 case
yet. Fill in / compare once real hardware numbers are available, same
pattern as every other script in this project.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE: same standing rule as
every other real-Docker script in this project -- real Docker/TimeLoop calls
run on your own Mac terminal, where Docker actually exists.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_500_p100_fp16.py
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
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_500_p100_fp16"

CEREBRAS_REFERENCE = {
    "precision": "FP16",
    "matrix_m": 500,
    "matrix_n": 500,
    "matrix_k": 500,
    "pe_grid": "100x100",
    "total_pes": 10000,
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

    gemm = GemmSpec(m=500, n=500, k=500, bytes_per_element=2)  # FP16
    wafer = WaferSpec(mesh_rows=100, mesh_cols=100)  # 100x100 = 10,000 PEs

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element} (FP16)")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols} = {wafer.mesh_rows * wafer.mesh_cols:,} PEs "
          f"(sram_kib_per_pe={wafer.sram_kib_per_pe}, "
          f"fifo_capacity_packets={wafer.fifo_capacity_packets}, "
          f"link_bandwidth_bytes_per_cycle={wafer.link_bandwidth_bytes_per_cycle}, "
          f"clock_ghz={wafer.clock_ghz})")
    print("Running real Docker timeloop-model once per SRAM-feasible tile_k candidate ...")
    print("This uses 10,000 active PEs -- expect this to take noticeably longer than the")
    print("50x50x50 runs (those used only 100 PEs). See docstring for a rough estimate.")

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
