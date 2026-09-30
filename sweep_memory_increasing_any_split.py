#!/usr/bin/env python3
"""Step sweep: memory's share walks UP from the three-way even split
(1/3, 1/3, 1/3 -- the real best-ranked combo from the very first 10-combo
sweep), compute+network's combined share walks DOWN to match, and at
every step compute_fraction + network_fraction = 1 - memory_fraction
EXACTLY (no leftover budget, no overlap -- the three always sum to 1.0,
never less). This is the memory-led counterpart to sweep_compute_
increasing_any_split.py (same "just needs to run, sum to 1, no fixed
50/50" rule, applied with memory as the dimension that increases).

Real, source-confirmed structural fact this exploits (see area_budget.py):
mesh size (hence PE count) is derived from ONLY compute_area_fraction --
_derive_mesh(budget.compute_area_budget_um2, mac_area_um2, gemm.m, gemm.n)
runs before memory/network are even looked at. So here, since compute is
now part of the shrinking remainder (not the swept lead dimension), giving
compute a SMALLER share of the remainder directly means a SMALLER mesh
(fewer PEs) -- and a smaller mesh means a LARGER per-PE budget for both
memory (this sweep's growing lead share) and network (whatever's left),
since per_pe_budget = category_fraction * total_budget / num_pes. That is
why this sweep tries compute-shares-of-the-remainder from SMALL to LARGE
(opposite ordering from the network-share order used in the compute-led
sweep) -- a smaller mesh is expected to be the easier case here, so trying
it first should short-circuit sooner on average.

RATIO_CANDIDATES are compute's share of (1 - memory), tried in this order
(network gets the rest): 0.20, 0.35, 0.50, 0.65, 0.80. Every attempt
(success or real failure) is logged; the sweep reports whichever ratio was
first to succeed at each memory step, or FAILED if none of the 5 did.

COST WARNING (real, bounded but still nontrivial): up to 5 real Docker-
backed CLI calls per memory step (fewer once one ratio succeeds).
Individual real derivations at these mesh sizes have taken ~90s-2200s
EACH in earlier runs -- a step where every ratio fails can cost 5x that.
An early-stop after MAX_CONSECUTIVE_STEP_FAILURES consecutive memory
steps where EVERY ratio failed stops the sweep (any steps skipped this
way are printed and logged explicitly, never silently dropped). Every
real trial is logged to trials.csv as it happens.

Usage (run from the project root, with your usual venv active):
  source .venv/bin/activate
  nohup python3 sweep_memory_increasing_any_split.py > sweep_memory_increasing_any_split.log 2>&1 &
  disown
  tail -f sweep_memory_increasing_any_split.log
  cat outputs/area_budget_memory_sweep_any_split/trials.csv
"""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import time
from pathlib import Path

GEMM_M = 1024
GEMM_N = 1024
GEMM_K = 1024
AREA_BUDGET_UM2 = 200_000_000
OUT_ROOT = Path("outputs/area_budget_memory_sweep_any_split")
TRIALS_CSV = OUT_ROOT / "trials.csv"

MEMORY_STEPS = [
    1.0 / 3.0,  # 0.3333... -- the real three-way-even-split baseline
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]

# compute's share of the remainder (1 - memory), tried in this order at
# every memory step; network gets (remainder - compute_share*remainder) =
# the exact float complement, so memory+compute+network always sums to
# exactly 1.0 -- never independently rounded, avoiding the float-sum-
# overshoot bug found earlier in this project's history.
RATIO_CANDIDATES = [0.20, 0.35, 0.50, 0.65, 0.80]

MAX_CONSECUTIVE_STEP_FAILURES = 3  # early-stop if this many memory steps in a row have EVERY ratio fail

trial_counter = 0


def fractions_for(memory: float, compute_share_of_remainder: float) -> tuple[float, float, float]:
    remainder = 1.0 - memory
    compute = remainder * compute_share_of_remainder
    network = remainder - compute  # exact remainder, avoids float-sum drift
    return compute, memory, network


def run_trial(memory: float, compute_share: float, step_idx: int, ratio_idx: int) -> dict:
    global trial_counter
    trial_counter += 1
    idx = trial_counter
    c, m, n = fractions_for(memory, compute_share)
    name = f"trial{idx:03d}_step{step_idx:02d}_m{memory:.4f}_compshare{compute_share:.2f}"
    out_dir = OUT_ROOT / name
    log_path = OUT_ROOT / f"{name}.log"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "wafersim",
        "--m", str(GEMM_M), "--n", str(GEMM_N), "--k", str(GEMM_K),
        "--area-budget-um2", str(AREA_BUDGET_UM2),
        "--compute-fraction", f"{c:.10f}",
        "--memory-fraction", f"{m:.10f}",
        "--network-fraction", f"{n:.10f}",
        "--out-dir", str(out_dir),
    ]
    print(f"    [ratio {ratio_idx+1}/{len(RATIO_CANDIDATES)}, compute_share={compute_share:.2f}] "
          f"compute={c:.4f} memory={memory:.4f} network={n:.4f} (sum={c+m+n:.10f}) ...", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    row = {
        "trial": idx, "step": step_idx, "compute_frac": c, "memory_frac": m, "network_frac": n,
        "compute_share_of_remainder": compute_share, "elapsed_s": round(elapsed, 1),
        "success": proc.returncode == 0,
    }
    if proc.returncode == 0:
        with open(out_dir / "result.json") as f:
            result = json.load(f)
        with open(out_dir / "area_budget.json") as f:
            budget = json.load(f)
        row.update({
            "sim_time_cycles": result["sim_time_cycles"],
            "total_energy_uj": result["energy"]["total_uj"],
            "mesh": f'{budget["mesh_rows"]}x{budget["mesh_cols"]}',
            "sram_kib_per_pe": budget["sram_kib_per_pe"],
            "fifo_capacity_packets": budget["fifo_capacity_packets"],
        })
        print(f"      -> OK in {elapsed:.0f}s (mesh {row['mesh']}, {row['sim_time_cycles']:.0f} cycles, "
              f"{row['total_energy_uj']:.2f} uJ)", flush=True)
    else:
        err_lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
        row["error"] = err_lines[-1] if err_lines else "(no stderr captured)"
        print(f"      -> FAILED in {elapsed:.0f}s: {row['error']}", flush=True)

    write_row(row)
    return row


def write_row(row: dict) -> None:
    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TRIALS_CSV.exists()
    fieldnames = [
        "trial", "step", "compute_frac", "memory_frac", "network_frac",
        "compute_share_of_remainder", "elapsed_s", "success", "sim_time_cycles",
        "total_energy_uj", "mesh", "sram_kib_per_pe", "fifo_capacity_packets", "error",
    ]
    with open(TRIALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def run_step(step_idx: int, memory: float) -> dict:
    """Try each ratio in RATIO_CANDIDATES until one succeeds (or all fail).
    Returns the winning row (success=True) or the LAST attempted row
    (success=False) if none worked."""
    print(f"=== [step {step_idx}/{len(MEMORY_STEPS)}] memory={memory:.4f} "
          f"(remainder={1.0-memory:.4f} split across up to {len(RATIO_CANDIDATES)} ratios) "
          f"-- started {time.strftime('%H:%M:%S')} ===", flush=True)
    last_row = None
    for ratio_idx, compute_share in enumerate(RATIO_CANDIDATES):
        row = run_trial(memory, compute_share, step_idx, ratio_idx)
        last_row = row
        if row["success"]:
            return row
    return last_row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if TRIALS_CSV.exists():
        TRIALS_CSV.unlink()

    rows = []
    consecutive_step_failures = 0
    stopped_early_at = None
    for step_idx, memory in enumerate(MEMORY_STEPS, start=1):
        row = run_step(step_idx, memory)
        rows.append(row)
        if row["success"]:
            consecutive_step_failures = 0
        else:
            consecutive_step_failures += 1
            if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                stopped_early_at = step_idx
                remaining = MEMORY_STEPS[step_idx:]
                print(f"\n=== STOPPING EARLY after {consecutive_step_failures} consecutive memory "
                      f"steps where EVERY ratio failed (memory={memory:.4f}) -- {len(remaining)} "
                      f"step(s) NOT run: {[f'{x:.4f}' for x in remaining]}. Raise "
                      f"MAX_CONSECUTIVE_STEP_FAILURES and re-run if you want them attempted "
                      f"anyway. ===\n", flush=True)
                break

    print("\n=== summary: memory increasing from the 1/3 baseline, compute+network always "
          f"summing to exactly (1 - memory), whichever of {RATIO_CANDIDATES} compute-share "
          f"ratios was first to work at each step (area budget {AREA_BUDGET_UM2} um^2) ===")
    header = (
        f"{'step':5} {'memory':>8} {'compute':>8} {'network':>8} {'comp_share':>10} {'mesh':>9} "
        f"{'sram_kib':>9} {'fifo_pkts':>9} {'sim_time_cycles':>16} {'total_energy_uj':>16}"
    )
    print(header)
    for r in rows:
        if r["success"]:
            print(
                f"{r['step']:<5} {r['memory_frac']:>8.4f} {r['compute_frac']:>8.4f} {r['network_frac']:>8.4f} "
                f"{r['compute_share_of_remainder']:>10.2f} {r['mesh']:>9} {r['sram_kib_per_pe']:>9} "
                f"{r['fifo_capacity_packets']:>9} {float(r['sim_time_cycles']):>16.0f} "
                f"{float(r['total_energy_uj']):>16.4f}"
            )
        else:
            print(
                f"{r['step']:<5} {r['memory_frac']:>8.4f}   ALL {len(RATIO_CANDIDATES)} RATIOS FAILED -- "
                f"last tried compute_share={r['compute_share_of_remainder']:.2f}: {r.get('error', '')[:70]}"
            )
    if stopped_early_at is not None:
        skipped = MEMORY_STEPS[stopped_early_at:]
        print(f"\n{len(skipped)} step(s) skipped by early-stop: {[f'{x:.4f}' for x in skipped]}")

    print(f"\ndone -- every real trial (all ratios, all memory values) logged at {TRIALS_CSV}")


if __name__ == "__main__":
    main()
