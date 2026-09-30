#!/usr/bin/env python3
"""Real OpenCake/wafersim GEMM run matched to another small Cerebras-style
config, same family as run_opencake_gemm_100_p20.py:

    Precision : FP32
    Matrix A/B/C : 50 x 50
    PE Grid   : 10 x 10  (Mt=Kt=Nt=5, i.e. 50 = 10 * 5)
    Total PEs : 100

Same real Docker/TimeLoop machinery as every other GEMM call in this project
(GemmSpec + WaferSpec + search_tile_k + best_by_sim_time) -- no synthetic
shortcut, one real `timeloop-model` Docker call per SRAM-feasible tile_k
candidate.

WHY THIS ONE: one more real data point on the PE-count-vs-runtime curve,
between the 100x100x100/20x20 case (400 PEs, ~62s total real wall time for
all 9 candidates) and the still-unfinished 3775x3775x3775/755x755 case
(570,025 PEs, 7+ hours before it was abandoned). 100 active PEs here is a
useful low-end anchor.

CAUTION ON THE COMPARISON ITSELF: you gave me Precision/Matrix/PE-Grid/
Total-PEs/Mt,Kt,Nt for this case, but no real Cerebras hardware cycle
counts. CEREBRAS_REFERENCE below leaves those fields as None -- send me the
real numbers if/when you have them and I'll wire up the comparison.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE: same standing rule as
every other real-Docker script in this project -- run it in your own Mac
terminal, in the project root (next to `src/wafersim`).

WHAT WAFER PARAMETERS THIS USES: mesh_rows=mesh_cols=10 is pinned to match
PE Grid 10x10 exactly. Every other WaferSpec field (sram_kib_per_pe=48,
fifo_capacity_packets=8, link_bandwidth_bytes_per_cycle=32, clock_ghz=1.0)
is left at this project's own existing WaferSpec() defaults
(src/wafersim/specs.py) -- no new assumptions invented for this case either.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_50_p10.py
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
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_50_p10"

CEREBRAS_REFERENCE = {
    "precision": "FP32",
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

    gemm = GemmSpec(m=50, n=50, k=50, bytes_per_element=4)  # FP32, matches Cerebras precision
    wafer = WaferSpec(mesh_rows=10, mesh_cols=10)  # pinned to match PE Grid 10x10 exactly

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element}")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols} "
          f"(sram_kib_per_pe={wafer.sram_kib_per_pe}, "
          f"fifo_capacity_packets={wafer.fifo_capacity_packets}, "
          f"link_bandwidth_bytes_per_cycle={wafer.link_bandwidth_bytes_per_cycle}, "
          f"clock_ghz={wafer.clock_ghz})")
    print("Running real Docker timeloop-model once per SRAM-feasible tile_k candidate ...")
    print("(100 active PEs total -- should be the fastest one yet.)")

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


if __name__ == "__main__":
    main()
