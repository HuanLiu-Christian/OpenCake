#!/usr/bin/env python3
"""Compares the real OpenCake GEMM result (from
run_opencake_gemm_3775_p755.py, run with real Docker) against the Cerebras
numbers you were given for the same M=N=K=3775, PE grid 755x755 GEMM.

Usage:
    python3 compare_gemm_3775_cerebras_vs_opencake.py outputs/cerebras_gemm_compare/gemm_3775_p755/result.json
"""

from __future__ import annotations

import json
import sys


def pct_diff(a: float, b: float) -> float:
    """% difference of a relative to b."""
    return (a - b) / b * 100.0


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <result.json>")
        raise SystemExit(1)

    with open(sys.argv[1]) as f:
        r = json.load(f)

    ref = r["cerebras_reference"]
    oc_cycles = r["opencake_sim_time_cycles"]

    same_shape = (r["gemm_m"], r["gemm_n"], r["gemm_k"]) == (
        ref["matrix_m"], ref["matrix_n"], ref["matrix_k"]
    )
    same_pes = (r["mesh_rows"], r["mesh_cols"]) == tuple(
        int(x) for x in ref["pe_grid"].split("x")
    )

    print("=" * 74)
    print("GEMM cycle comparison: OpenCake (real Docker/TimeLoop) vs. Cerebras")
    print("=" * 74)
    print(f"Shape (M,N,K): OpenCake ({r['gemm_m']},{r['gemm_n']},{r['gemm_k']})  "
          f"vs Cerebras ({ref['matrix_m']},{ref['matrix_n']},{ref['matrix_k']})  "
          f"-- match: {same_shape}")
    print(f"PE grid      : OpenCake {r['mesh_rows']}x{r['mesh_cols']}  "
          f"vs Cerebras {ref['pe_grid']}  -- match: {same_pes}")
    print(f"Active PEs   : OpenCake {r['active_pes']:,} of "
          f"{r['mesh_rows']*r['mesh_cols']:,} mesh PEs  "
          f"vs Cerebras {ref['total_pes']:,} (all used, by construction of the SUMMA grid)")
    print()
    print("Tiling comparison (two DIFFERENT mapping strategies, not the same"
          " algorithm -- read the note at the bottom before trusting the ratio):")
    print(f"  OpenCake : tile_m={r['tile_m']} tile_n={r['tile_n']} tile_k={r['tile_k']} "
          f"-> k_phases={r['k_phases']} (real SRAM-feasibility search, "
          f"sram_kib_per_pe={r['sram_kib_per_pe']})")
    print(f"  Cerebras : Mt={ref['Mt']} Kt={ref['Kt']} Nt={ref['Nt']} "
          f"-> {ref['Kt']} SUMMA broadcast/accumulate steps (fixed by the "
          f"P x P grid geometry, not searched)")
    print()

    print(f"OpenCake real sim_time_cycles (compute+network) : {oc_cycles:,}")
    print(f"Cerebras total cycles, given range               : "
          f"{ref['total_cycles_min']:,} - {ref['total_cycles_max']:,}")
    print(f"Cerebras max per-PE GEMM cycles (sync-safe bound) : "
          f"{ref['max_per_pe_gemm_cycles']:,}")
    print()

    for label, cerebras_val in [
        ("total_cycles_min", ref["total_cycles_min"]),
        ("total_cycles_max", ref["total_cycles_max"]),
        ("max_per_pe_gemm_cycles", ref["max_per_pe_gemm_cycles"]),
    ]:
        ratio = oc_cycles / cerebras_val
        diff = pct_diff(oc_cycles, cerebras_val)
        print(f"  vs Cerebras {label:<24} ({cerebras_val:>10,}): "
              f"ratio={ratio:.3f}x  diff={diff:+.1f}%")

    print()
    print(f"OpenCake real total_energy_uj : {r['opencake_total_energy_uj']:,.4f}")
    print("(Cerebras printout has no energy number to compare against --")
    print(" this project has not found a public Cerebras SDK power/energy API,")
    print(" so energy stays a one-sided OpenCake-only number here.)")

    print()
    print("-" * 74)
    print("READ BEFORE reporting a \"similarity %\" to your teacher:")
    print("-" * 74)
    print(
        "OpenCake's tile_k and Cerebras's Kt are NOT the same kind of split.\n"
        "Cerebras's 2D-SUMMA example ties K-chunking directly to the P x P grid\n"
        "(Kt = K / P, and there are exactly P row/column broadcast steps -- an\n"
        "architectural choice, not a search). OpenCake's tile_k instead comes\n"
        "from a real, independent SRAM-feasibility search across every\n"
        "candidate that evenly divides K, picked by lowest real full-system\n"
        "(compute + network) cycles -- it is not required to equal K/755 just\n"
        "because the grid is 755x755. If the two happen to land on the same\n"
        "tile_k / phase count, that is a real point of agreement worth noting;\n"
        "if they differ, that difference is itself part of the honest answer\n"
        "to \"how similar are the two approaches\" -- not something to paper\n"
        "over by forcing the same number."
    )


if __name__ == "__main__":
    main()
