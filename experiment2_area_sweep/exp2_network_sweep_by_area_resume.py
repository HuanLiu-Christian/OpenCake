#!/usr/bin/env python3
"""RESUME wrapper for exp2_network_sweep_by_area.py -- written after the
original run crashed with a real OSError: [Errno 23] Too many open files in
system while writing outputs/experiment2_area_sweep/network_sweep/
area_1600000000/trial016_step08_n0.7500_compshare0.20.log (see the original
script's own log, exp2_network_sweep_by_area.log, lines 183-200, for the
real traceback -- and line 4/77/142 for the three real "AREA BUDGET" banners
that prove 200M and 400M each ran to a natural, non-crashed completion
before 1,600,000,000 started).

Real state verified against real files on disk right before writing this
script (not assumed, not carried over from an earlier session):
  - area=200,000,000:  DONE.  22 real trial rows, ran to completion
    naturally (the log's very next line after this area's own "done --"
    print is the 400,000,000 AREA BUDGET banner -- no crash in between).
  - area=400,000,000:  DONE.  18 real trial rows, same evidence.
  - area=1,600,000,000: PARTIAL. 15 real trial rows on disk. step1: all 5
    ratios failed. step2: all 5 ratios failed. steps3-7: each succeeded on
    the first ratio tried (1 row each). consecutive_step_failures is
    genuinely 0 going into step 8 (step 7's row has success=True). The
    crash happened mid-step-8, ratio 1/5 (network=0.75, comp_share=0.20).
    Confirmed via that crashed trial's own output directory:
    area_budget.json shows the real area-budget binary search actually
    finished (mesh 64x128, sram=54KiB, fifo=21 packets, no saturation),
    but there is NO result.json there -- the real TimeLoop/GEMM simulation
    phase never finished, so this one specific trial attempt has no
    usable real result and must be genuinely re-run from scratch, not
    treated as recoverable.
  - area=3,200,000,000: NEVER STARTED. No banner for it appears anywhere
    in the crashed log.

This script does not touch the 200M/400M real data at all (never opens
those trials.csv files), and reuses every other real mechanism byte-for-
byte from the original module via import (fractions_for/run_trial/
write_row/run_step -- same real `python3 -m wafersim` subprocess
invocation, same real CSV schema, same real early-stop rule). It only
replaces run_for_area()'s "always start at step 1, always delete
trials.csv" behavior and main()'s "always run all 4 area budgets from
scratch" behavior for the one area that needs resuming. The per-area local
trial-numbering counter and the global all_trial_counter (used for the
"trial" column shared across the whole experiment) are both DERIVED AT
RUNTIME from the real row counts already on disk, not hardcoded, so this
stays correct even if the exact prior counts were mis-stated anywhere
above. No synthetic results are fabricated anywhere -- the resumed step-8
attempt and everything after it still goes through a real subprocess call
to `python3 -m wafersim`, exactly like before.

Usage (same real-Docker-required preconditions as the original -- Docker
Desktop running; strongly recommend tmux/screen or caffeinate given how
long a single trial can take (this branch has shown real single-trial
durations up to ~66575s / ~18.5h), see the original script's own docstring
for why a plain Terminal window risks getting interrupted):
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_network_sweep_by_area_resume.py > exp2_network_sweep_by_area_resume.log 2>&1 &
  disown
  tail -f exp2_network_sweep_by_area_resume.log
  cat ../outputs/experiment2_area_sweep/network_sweep/all_trials.csv
"""

from __future__ import annotations

import csv

import exp2_network_sweep_by_area as base

# Real, verified state (see module docstring for the exact evidence this
# was derived from) -- only these two facts are hardcoded; every trial
# counter is self-derived from real files on disk at runtime below.
RESUME_AREA = 1_600_000_000
RESUME_FROM_STEP = 8            # steps 1-7 already have real rows on disk
REMAINING_NEW_AREAS = [3_200_000_000]  # never started in the crashed run


def _real_row_count(csv_path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _load_existing_step_rows(trials_csv, up_to_step: int) -> list[dict]:
    """Reconstruct the same one-row-per-step 'rows' list run_for_area()
    builds during a normal run, from the real trials.csv already on disk,
    for steps 1..up_to_step. Mirrors run_step()'s own real selection rule:
    the first real success for a step wins; if every real ratio for that
    step failed, the LAST real attempt for that step is what gets kept
    (matches "last tried comp_share=..." in the original summary print).
    """
    by_step: dict[int, list[dict]] = {}
    with open(trials_csv, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if step <= up_to_step:
                by_step.setdefault(step, []).append(row)

    rows = []
    for step in sorted(by_step):
        step_rows = by_step[step]
        success_rows = [r for r in step_rows if r["success"] == "True"]
        chosen = dict(success_rows[0] if success_rows else step_rows[-1])
        chosen["step"] = step
        chosen["success"] = chosen["success"] == "True"
        chosen["network_frac"] = float(chosen["network_frac"])
        chosen["compute_frac"] = float(chosen["compute_frac"])
        chosen["memory_frac"] = float(chosen["memory_frac"])
        chosen["compute_share_of_remainder"] = float(chosen["compute_share_of_remainder"])
        rows.append(chosen)
    return rows


def run_for_area_resume(area_budget: int, start_step_idx: int) -> None:
    """Same real logic as base.run_for_area(), except: does NOT delete the
    existing trials.csv, does NOT re-run steps before start_step_idx, and
    seeds trial_idx/rows/consecutive_step_failures from the real data
    already on disk (row counts, not guesses) instead of from zero."""
    out_root = base.EXPERIMENT_ROOT / f"area_{area_budget}"
    trials_csv = out_root / "trials.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _load_existing_step_rows(trials_csv, start_step_idx - 1) if trials_csv.exists() else []
    local_trial_idx_start = _real_row_count(trials_csv)  # derived, not hardcoded

    consecutive_step_failures = 0
    for r in reversed(rows):
        if r["success"]:
            break
        consecutive_step_failures += 1

    print(f"\n{'#'*70}\n# AREA BUDGET {area_budget:,} um^2 -- network-primary any-split sweep "
          f"(GEMM {base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} fixed) -- RESUMING from real step "
          f"{start_step_idx}/{len(base.NETWORK_STEPS)} ({local_trial_idx_start} real per-area trial(s) "
          f"already on disk, consecutive_step_failures reconstructed as {consecutive_step_failures})\n"
          f"{'#'*70}", flush=True)

    if consecutive_step_failures >= base.MAX_CONSECUTIVE_STEP_FAILURES:
        print(f"[area={area_budget:,}] real data on disk already shows "
              f"{consecutive_step_failures} consecutive full-step failures going into step "
              f"{start_step_idx} -- this area budget was already effectively early-stopped; not "
              f"running anything new for it.", flush=True)
        return

    stopped_early_at = None
    trial_idx = local_trial_idx_start
    for step_idx, network in enumerate(base.NETWORK_STEPS, start=1):
        if step_idx < start_step_idx:
            continue
        row, trial_idx = base.run_step(area_budget, out_root, trials_csv, step_idx, network, trial_idx)
        rows.append(row)
        if row["success"]:
            consecutive_step_failures = 0
        else:
            consecutive_step_failures += 1
            if consecutive_step_failures >= base.MAX_CONSECUTIVE_STEP_FAILURES:
                stopped_early_at = step_idx
                remaining = base.NETWORK_STEPS[step_idx:]
                print(f"\n=== [area={area_budget:,}] STOPPING EARLY after {consecutive_step_failures} "
                      f"consecutive network steps where EVERY ratio failed (network={network:.4f}) -- "
                      f"{len(remaining)} step(s) NOT run for this area budget: "
                      f"{[f'{x:.4f}' for x in remaining]}. ===\n", flush=True)
                break

    print(f"\n=== [area={area_budget:,}] summary (real steps 1-{len(base.NETWORK_STEPS)}; steps before "
          f"{start_step_idx} reconstructed from the real data already on disk): network 5%->95%, "
          f"compute+memory always summing to exactly (1 - network), whichever of "
          f"{base.RATIO_CANDIDATES} compute-share ratios was first to work at each step "
          f"(GEMM {base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} fixed) ===")
    header = (
        f"{'step':5} {'network':>8} {'compute':>8} {'memory':>8} {'comp_share':>10} {'mesh':>9} "
        f"{'sram_kib':>9} {'fifo_pkts':>9} {'sim_time_cycles':>16} {'total_energy_uj':>16}"
    )
    print(header)
    for r in rows:
        if r["success"]:
            print(
                f"{r['step']:<5} {r['network_frac']:>8.4f} {r['compute_frac']:>8.4f} {r['memory_frac']:>8.4f} "
                f"{r['compute_share_of_remainder']:>10.2f} {r['mesh']:>9} {r['sram_kib_per_pe']:>9} "
                f"{r['fifo_capacity_packets']:>9} {float(r['sim_time_cycles']):>16.0f} "
                f"{float(r['total_energy_uj']):>16.4f}"
            )
        else:
            print(
                f"{r['step']:<5} {r['network_frac']:>8.4f}   ALL {len(base.RATIO_CANDIDATES)} RATIOS FAILED -- "
                f"last tried comp_share={r['compute_share_of_remainder']:.2f}: {r.get('error', '')[:70]}"
            )
    if stopped_early_at is not None:
        skipped = base.NETWORK_STEPS[stopped_early_at:]
        print(f"\n[area={area_budget:,}] {len(skipped)} step(s) skipped by early-stop: "
              f"{[f'{x:.4f}' for x in skipped]}")

    print(f"\n[area={area_budget:,}] done -- every real trial for this area budget logged at {trials_csv}")


def main() -> None:
    base.EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    # Do NOT unlink ALL_TRIALS_CSV -- it already holds real rows (200M +
    # 400M + 1.6B steps 1-7) that must be preserved, not wiped.
    prior_global_trials = _real_row_count(base.ALL_TRIALS_CSV)
    base.all_trial_counter = prior_global_trials  # derived, not hardcoded

    print(f"experiment 2 (network-primary branch) -- RESUME run. Real state: area=200,000,000 and "
          f"area=400,000,000 already fully done (not touched here); area={RESUME_AREA:,} resuming "
          f"from real step {RESUME_FROM_STEP}; then {REMAINING_NEW_AREAS} (never started before). "
          f"Global trial counter resumes at {prior_global_trials + 1} (real trials 1-{prior_global_trials} "
          f"already exist in {base.ALL_TRIALS_CSV}).", flush=True)

    run_for_area_resume(RESUME_AREA, start_step_idx=RESUME_FROM_STEP)

    for area_budget in REMAINING_NEW_AREAS:
        base.run_for_area(area_budget)

    print(f"\n{'='*70}\nRESUME RUN DONE (network-primary branch). Combined real results (every "
          f"trial, every area, including the pre-crash 200M/400M/1.6B-steps-1-7 rows) at "
          f"{base.ALL_TRIALS_CSV}\n(the 800M um^2 point is separately at "
          f"../outputs/area_sweep_network_any_split_800M/trials.csv, same schema minus the "
          f"area_budget_um2 column -- treat that as area_budget_um2=800000000)\n{'='*70}")


if __name__ == "__main__":
    main()
