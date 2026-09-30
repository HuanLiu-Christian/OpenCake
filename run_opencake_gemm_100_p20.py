#!/usr/bin/env python3
"""Real OpenCake/wafersim GEMM run matched to a SMALL Cerebras-style config,
used as a fast sanity check / speed comparison against the much larger
3775x3775x3775 / 755x755 run (see run_opencake_gemm_3775_p755.py):

    Precision : FP32
    Matrix A/B/C : 100 x 100
    PE Grid   : 20 x 20  (Mt=Kt=Nt=5, i.e. 100 = 20 * 5)
    Total PEs : 400

Same real Docker/TimeLoop machinery as every other GEMM call in this project
(GemmSpec + WaferSpec + search_tile_k + best_by_sim_time) -- no synthetic
shortcut, one real `timeloop-model` Docker call per SRAM-feasible tile_k
candidate.

WHY THIS ONE MATTERS RIGHT NOW:
The 3775^3 / 755x755 run (570,025 PEs) has been running for 7+ real hours.
The working hypothesis (from reading the real generated architecture.yaml
for that run -- see PE[0..570024] and the DRAM-level spatial factors
M=755 N=755) is that timeloop-model's real runtime scales super-linearly
with the number of spatially-mapped PE instances, not with the per-PE
workload (which is tiny in both cases). This run is a real, cheap way to
test that: 400 PEs is about 1,425x FEWER active PEs than the 3775 case. If
the hypothesis is right, this should finish in well under a minute; if it
still takes a long time, the PE-count theory is wrong and something else is
the bottleneck.

CAUTION ON THE COMPARISON ITSELF:
You gave me Precision/Matrix/PE-Grid/Total-PEs/Mt,Kt,Nt for this small case,
but NOT the actual Cerebras hardware cycle counts (broadcast/compute/total
cycles) the way you did for the 3775 case. CEREBRAS_REFERENCE below is left
with those cycle fields as None -- I am not going to invent numbers. If you
already ran this small case on real Cerebras hardware too, send me those
numbers and I'll fill them in / rerun the comparison; if not, this run is
just an OpenCake-side speed/sanity check for now.

WHY THIS HAS TO RUN ON YOUR OWN MACHINE, NOT HERE:
Same standing rule as every other *_by_area.py / qwen3_8b_prefill_decode.py
script in this project: real Docker/TimeLoop calls run on your own Mac
terminal, where Docker actually exists. This file is meant to sit in the
project root (next to `src/wafersim`) and run from there.

WHAT WAFER PARAMETERS THIS USES, AND WHY:
mesh_rows=mesh_cols=20 is pinned to match PE Grid 20x20 exactly. Every other
WaferSpec field (sram_kib_per_pe=48, fifo_capacity_packets=8,
link_bandwidth_bytes_per_cycle=32, clock_ghz=1.0) is left at this project's
own existing WaferSpec() defaults (src/wafersim/specs.py), same convention
as the 3775 run -- no new assumptions invented for this smaller case either.

Usage (on your Mac, in the project root, real Docker required):
    python3 run_opencake_gemm_100_p20.py
"""

from __future__ import annotations

import json
from pathlib import Path

from wafersim.primitives import (
    GemmSpec,
    WaferSpec,
    best_by_sim_time,
    search_tile_k,
)

REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "outputs" / "cerebras_gemm_compare" / "gemm_100_p20"

# What you gave me for this small case. Cycle numbers are None because you
# did not paste any this time -- fill them in (or tell me and I will) once
# you have a real Cerebras run for this exact size, same shape as the 3775
# CEREBRAS_REFERENCE dict in run_opencake_gemm_3775_p755.py.
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

    gemm = GemmSpec(m=100, n=100, k=100, bytes_per_element=4)  # FP32, matches Cerebras precision
    wafer = WaferSpec(mesh_rows=20, mesh_cols=20)  # pinned to match PE Grid 20x20 exactly

    print(f"GEMM: m={gemm.m} n={gemm.n} k={gemm.k} bytes_per_element={gemm.bytes_per_element}")
    print(f"Wafer mesh: {wafer.mesh_rows}x{wafer.mesh_cols} "
          f"(sram_kib_per_pe={wafer.sram_kib_per_pe}, "
          f"fifo_capacity_packets={wafer.fifo_capacity_packets}, "
          f"link_bandwidth_bytes_per_cycle={wafer.link_bandwidth_bytes_per_cycle}, "
          f"clock_ghz={wafer.clock_ghz})")
    print("Running real Docker timeloop-model once per SRAM-feasible tile_k candidate ...")
    print("(400 active PEs total -- this should be fast; timing it is the point.)")

    import time
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
    print("\nIf/when you have real Cerebras cycle numbers for this 100x100x100 / "
          "20x20 case, send them to me and I'll fill in cerebras_reference and "
          "run the comparison (same format as compare_gemm_3775_cerebras_vs_opencake.py).")


if __name__ == "__main__":
    main()
