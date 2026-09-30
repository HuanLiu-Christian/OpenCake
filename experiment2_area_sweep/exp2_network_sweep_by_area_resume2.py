#!/usr/bin/env python3
"""SECOND resume wrapper for exp2_network_sweep_by_area.py.

Context: the original network_sweep run crashed once already (real
OSError: too many open files) and was resumed via
exp2_network_sweep_by_area_resume.py, which successfully finished
area=1,600,000,000 (real steps 8-10 all completed for real, area marked
done) and started area=3,200,000,000. Then a real Mac reboot (2026-09-08,
reported by the user) killed that resumed process too, mid area=3,200,000,000
step 1.

Real state verified against real files on disk right before writing this
script:
  - area=200,000,000:  DONE.  22 real trial rows.
  - area=400,000,000:  DONE.  18 real trial rows.
  - area=1,600,000,000: DONE.  18 real trial rows (finished by the FIRST
    resume -- steps 1-7 from before that resume, steps 8-10 real results
    added by it: step8 mesh 64x128/134942 cycles, step9 mesh
    64x64/265410 cycles, step10 mesh 32x64/529704 cycles, all real
    successes).
  - area=3,200,000,000: PARTIAL. 1 real trial row on disk, step 1
    (network=0.05), comp_share=0.20, FAILED for real (elapsed_s 4393.5s).
    The reboot killed it before comp_share=0.35 (the 2nd ratio) produced
    any result. No success recorded for step 1 yet.
  - area=3,200,000,000 is network_sweep's LAST area budget -- nothing new
    to start after it finishes.

Same discipline as the first resume and as the other two branches' resume
scripts written alongside this one: does not touch the 200M/400M/1.6B real
data, reuses every real mechanism byte-for-byte from the original module
via import, and for the one partially-completed step resumes at the NEXT
UNTRIED ratio (comp_share=0.35) instead of re-attempting comp_share=0.20,
which already has a real, on-disk result -- so it does not throw away the
~1.2 real hours that attempt cost (4393.5s). Every trial-numbering counter
and the resume point itself are derived at runtime from the real
trials.csv/all_trials.csv row counts already on disk via
_resume_common.py, not hardcoded as literal numbers. No synthetic results
are fabricated anywhere -- every resumed trial still goes through a real
subprocess call to `python3 -m wafersim`, exactly like before.

Usage (same real-Docker-required preconditions as before -- Docker Desktop
running; use tmux/screen/caffeinate, not a plain Terminal window, since a
plain window is what let both previous runs get killed):
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_network_sweep_by_area_resume2.py > exp2_network_sweep_by_area_resume2.log 2>&1 &
  disown
  tail -f exp2_network_sweep_by_area_resume2.log
  cat ../outputs/experiment2_area_sweep/network_sweep/all_trials.csv
"""

from __future__ import annotations

import _resume_common as common
import exp2_network_sweep_by_area as base


def run_for_area_resume(area_budget: int) -> None:
    out_root = base.EXPERIMENT_ROOT / f"area_{area_budget}"
    trials_csv = out_root / "trials.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    resume_step_idx, ratio_start_idx, rows = common.find_resume_point(
        trials_csv, len(base.RATIO_CANDIDATES)
    )
    local_trial_idx_start = common.real_row_count(trials_csv)  # derived, not hardcoded

    consecutive_step_failures = 0
    for r in reversed(rows):
        if r["success"]:
            break
        consecutive_step_failures += 1

    print(f"\n{'#'*70}\n# AREA BUDGET {area_budget:,} um^2 -- network-primary any-split sweep "
          f"(GEMM {base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} fixed) -- RESUMING (2nd time) at real "
          f"step {resume_step_idx}/{len(base.NETWORK_STEPS)}"
          + (f", ratio {ratio_start_idx+1}/{len(base.RATIO_CANDIDATES)} within that step "
             f"(ratios before it already have real on-disk results)" if ratio_start_idx else "")
          + f" ({local_trial_idx_start} real per-area trial(s) already on disk, "
            f"consecutive_step_failures reconstructed as {consecutive_step_failures})\n{'#'*70}", flush=True)

    if resume_step_idx > len(base.NETWORK_STEPS):
        print(f"[area={area_budget:,}] all {len(base.NETWORK_STEPS)} real steps already complete on disk; "
              f"nothing new to run for this area.", flush=True)
        return

    if consecutive_step_failures >= base.MAX_CONSECUTIVE_STEP_FAILURES:
        print(f"[area={area_budget:,}] real data on disk already shows "
              f"{consecutive_step_failures} consecutive full-step failures going into step "
              f"{resume_step_idx} -- this area budget was already effectively early-stopped; not "
              f"running anything new for it.", flush=True)
        return

    stopped_early_at = None
    trial_idx = local_trial_idx_start
    for step_idx, network in enumerate(base.NETWORK_STEPS, start=1):
        if step_idx < resume_step_idx:
            continue
        if step_idx == resume_step_idx and ratio_start_idx:
            row = None
            for ratio_idx in range(ratio_start_idx, len(base.RATIO_CANDIDATES)):
                compute_share = base.RATIO_CANDIDATES[ratio_idx]
                trial_idx += 1
                row = base.run_trial(area_budget, out_root, trials_csv, network, compute_share,
                                      step_idx, ratio_idx, trial_idx)
                if row["success"]:
                    break
        else:
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
          f"{resume_step_idx} reconstructed from the real data already on disk): network 5%->95%, "
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
    # 400M + 1.6B + 3.2B step1's 1 real attempt) that must be preserved,
    # not wiped.
    prior_global_trials = common.real_row_count(base.ALL_TRIALS_CSV)
    base.all_trial_counter = prior_global_trials  # derived, not hardcoded

    print(f"experiment 2 (network-primary branch) -- 2nd RESUME run (after real Mac reboot). Real "
          f"state: area=200,000,000 / 400,000,000 / 1,600,000,000 already fully done (not touched "
          f"here); area=3,200,000,000 (network_sweep's last area) resuming from real progress on "
          f"disk. Global trial counter resumes at {prior_global_trials + 1} "
          f"(real trials 1-{prior_global_trials} already exist in {base.ALL_TRIALS_CSV}).", flush=True)

    run_for_area_resume(3_200_000_000)

    print(f"\n{'='*70}\nRESUME RUN DONE (network-primary branch). Combined real results (every "
          f"trial, every area, including every pre-reboot row) at {base.ALL_TRIALS_CSV}\n"
          f"(the 800M um^2 point is separately at "
          f"../outputs/area_sweep_network_any_split_800M/trials.csv, same schema minus the "
          f"area_budget_um2 column -- treat that as area_budget_um2=800000000)\n{'='*70}")


if __name__ == "__main__":
    main()
