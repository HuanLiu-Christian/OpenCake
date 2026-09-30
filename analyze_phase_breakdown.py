#!/usr/bin/env python3
"""Per-K-phase breakdown of a real OpenCake GEMM run's broadcast/compute
timing, in the SAME shape as the real Cerebras TSC printout format
("Broadcast cycles (min/max)", "Compute cycles (min/max)", per phase
instead of only summed across all phases).

WHY THIS EXISTS: the aggregate numbers we already have (e.g. Cerebras's
real "Broadcast cycles (min/max): 12020/12294, Compute cycles (min/max):
17480/17480" for the 100x100x100/20x20/Kt=5 case) sum across all 5 K-phases
into one min/max pair each. That hides whether every phase behaves
similarly or one phase (e.g. the first, if it also had to wait on initial
data load) is an outlier. This script re-runs the exact same real,
already-Docker-verified per-PE compute rate (taken from a completed run's
result.json -- no new Docker call needed, this is pure Python, deterministic
given that real TimeLoop-measured local_compute_cycles) through the same
real wafer/network.py timing model, but instruments it to report EACH
phase's own broadcast/compute min/max across all active PEs, plus each
phase's end time -- ready to line up directly against whatever
per-phase real numbers jiasheng provides from real Cerebras hardware.

Does NOT touch wafer/simulator.py's real simulate_gemm() (used by every
other script in this project, covered by the existing test suite) -- this
is a read-only, side-channel instrumentation of the identical real
algorithm, built from a completed run's own real numbers.

Usage:
    python3 analyze_phase_breakdown.py outputs/cerebras_gemm_compare/gemm_100_p20_tilek20/result.json
"""

from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path

from wafersim.specs import GemmSpec, WaferSpec
from wafersim.wafer.network import PacketNetwork


def analyze(gemm: GemmSpec, wafer: WaferSpec, tile_m: int, tile_n: int, tile_k: int,
            active_rows: int, active_cols: int, local_compute_cycles: float) -> list[dict]:
    k_phases = ceil(gemm.k / tile_k)
    interior_macs_per_pe = tile_m * tile_n * gemm.k
    cycles_per_mac = local_compute_cycles / interior_macs_per_pe

    network = PacketNetwork(wafer, active_rows, active_cols)

    compute_ready = {(r, c): 0.0 for r in range(active_rows) for c in range(active_cols)}
    a_buffer_ready = {(r, c): 0.0 for r in range(active_rows) for c in range(active_cols)}
    b_buffer_ready = {(r, c): 0.0 for r in range(active_rows) for c in range(active_cols)}

    phase_records: list[dict] = []
    prev_phase_end = 0.0

    for phase in range(k_phases):
        k_start = phase * tile_k
        k_extent = max(0, min(tile_k, gemm.k - k_start))
        owner_col = phase % active_cols
        owner_row = phase % active_rows

        a_arrival_by_row: dict[int, dict] = {}
        for row in range(active_rows):
            m_extent = max(0, min(tile_m, gemm.m - row * tile_m))
            payload = m_extent * k_extent * gemm.bytes_per_element
            source_ready = a_buffer_ready[(row, owner_col)]
            a_arrival_by_row[row] = network.broadcast_row(
                row=row, ready_cycle=source_ready, size_bytes=payload, owner_col=owner_col
            )

        b_arrival_by_col: dict[int, dict] = {}
        for col in range(active_cols):
            n_extent = max(0, min(tile_n, gemm.n - col * tile_n))
            payload = k_extent * n_extent * gemm.bytes_per_element
            source_ready = b_buffer_ready[(owner_row, col)]
            b_arrival_by_col[col] = network.broadcast_column(
                col=col, ready_cycle=source_ready, size_bytes=payload, owner_row=owner_row
            )

        arrivals: list[float] = []
        compute_durations: list[float] = []
        compute_ends: list[float] = []

        for row in range(active_rows):
            m_extent = max(0, min(tile_m, gemm.m - row * tile_m))
            for col in range(active_cols):
                n_extent = max(0, min(tile_n, gemm.n - col * tile_n))
                arrival = max(
                    a_arrival_by_row[row][(row, col)],
                    b_arrival_by_col[col][(row, col)],
                    compute_ready[(row, col)],
                    a_buffer_ready[(row, col)],
                    b_buffer_ready[(row, col)],
                )
                macs = m_extent * n_extent * k_extent
                compute_cycles = ceil(macs * cycles_per_mac)
                end = arrival + compute_cycles

                compute_ready[(row, col)] = end
                a_buffer_ready[(row, col)] = end
                b_buffer_ready[(row, col)] = end

                arrivals.append(arrival)
                compute_durations.append(compute_cycles)
                compute_ends.append(end)

        phase_records.append({
            "phase": phase,
            "k_extent": k_extent,
            "broadcast_cycles_min": min(arrivals) - prev_phase_end,
            "broadcast_cycles_max": max(arrivals) - prev_phase_end,
            "compute_cycles_min": min(compute_durations),
            "compute_cycles_max": max(compute_durations),
            "phase_end_min": min(compute_ends),
            "phase_end_max": max(compute_ends),
        })
        prev_phase_end = max(compute_ends)

    return phase_records


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <result.json>")
        raise SystemExit(1)

    data = json.loads(Path(sys.argv[1]).read_text())
    gemm = GemmSpec(m=data["gemm_m"], n=data["gemm_n"], k=data["gemm_k"],
                     bytes_per_element=data["bytes_per_element"])
    wafer = WaferSpec(mesh_rows=data["mesh_rows"], mesh_cols=data["mesh_cols"],
                       sram_kib_per_pe=data["sram_kib_per_pe"],
                       fifo_capacity_packets=data["fifo_capacity_packets"])

    records = analyze(
        gemm, wafer,
        tile_m=data["tile_m"], tile_n=data["tile_n"], tile_k=data["tile_k"],
        active_rows=data["active_rows"], active_cols=data["active_cols"],
        local_compute_cycles=data["opencake_local_compute_cycles"],
    )

    print(f"Per-phase real OpenCake breakdown for GEMM {gemm.m}x{gemm.n}x{gemm.k}, "
          f"mesh {wafer.mesh_rows}x{wafer.mesh_cols}, tile_k={data['tile_k']} "
          f"(k_phases={len(records)}):\n")
    print(f"{'phase':>5} {'broadcast(min/max)':>22} {'compute(min/max)':>20} "
          f"{'phase_end(min/max)':>22}")
    for r in records:
        print(f"{r['phase']:>5} "
              f"{r['broadcast_cycles_min']:>10.1f}/{r['broadcast_cycles_max']:<10.1f} "
              f"{r['compute_cycles_min']:>9.1f}/{r['compute_cycles_max']:<9.1f} "
              f"{r['phase_end_min']:>10.1f}/{r['phase_end_max']:<10.1f}")

    total_broadcast_min = sum(r["broadcast_cycles_min"] for r in records)
    total_broadcast_max = sum(r["broadcast_cycles_max"] for r in records)
    total_compute_min = sum(r["compute_cycles_min"] for r in records)
    total_compute_max = sum(r["compute_cycles_max"] for r in records)
    print(f"\nSummed across all {len(records)} phases:")
    print(f"  broadcast_cycles (min/max) : {total_broadcast_min:.1f} / {total_broadcast_max:.1f}")
    print(f"  compute_cycles   (min/max) : {total_compute_min:.1f} / {total_compute_max:.1f}")
    print(f"  (cross-check) OpenCake's own aggregate sim_time_cycles from the full run: "
          f"{data['opencake_sim_time_cycles']}")

    out_path = Path(sys.argv[1]).with_name("phase_breakdown.json")
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
