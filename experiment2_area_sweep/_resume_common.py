"""Shared resume helpers for exp2_*_sweep_by_area.py -- written after a
real Mac reboot (2026-09-08, reported by the user and consistent with all
three branches' logs going silent at different real times) took down all
three branch processes mid-trial:
  - compute_sweep: died mid area=1,600,000,000 step 7 (2 of 5 real ratio
    attempts done, both failed, no success yet).
  - memory_sweep: died mid area=3,200,000,000 step 1 (4 of 5 real ratio
    attempts done, all failed, no success yet).
  - network_sweep (already resumed once before -- this is a SECOND resume
    on top of that): died mid area=3,200,000,000 step 1 (1 of 5 real ratio
    attempts done, failed, no success yet).

Every gap found by this module still gets a real subprocess call through
the original script's own run_trial()/run_step(); this module only
locates, from the real trials.csv content already on disk, exactly where
each branch's real progress stopped -- so a resume never redoes an
already-completed real Docker call (no wasted real GPU/Docker time) and
never fabricates a result for the ones left incomplete.
"""
from __future__ import annotations

import csv


def real_rows(csv_path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def real_row_count(csv_path) -> int:
    return len(real_rows(csv_path))


def pick_step_row(step_rows: list[dict]) -> dict:
    """Same one-row-per-step selection rule run_for_area()'s own summary
    print uses: the first real success for a step wins; if every real
    ratio attempted for that step failed, the LAST real attempt is what
    gets kept."""
    success_rows = [r for r in step_rows if r["success"] == "True"]
    chosen = dict(success_rows[0] if success_rows else step_rows[-1])
    chosen["step"] = int(chosen["step"])
    chosen["success"] = chosen["success"] == "True"
    for key in list(chosen):
        if key.endswith("_frac") or key.endswith("_of_remainder"):
            chosen[key] = float(chosen[key])
    return chosen


def find_resume_point(trials_csv, num_ratio_candidates: int):
    """Returns (resume_step_idx, ratio_start_idx, prior_step_rows) derived
    from the real trials.csv already on disk.

      - No file yet -> (1, 0, []) -- nothing real has run for this area.
      - The max real step present already has a real success, OR already
        has all num_ratio_candidates real ratio attempts recorded (all
        failed) -> that step is genuinely resolved; resume at the NEXT
        step, ratio_start_idx=0.
      - Otherwise the max real step is genuinely incomplete (this is what
        a mid-step process kill looks like on disk): resume AT that same
        step, but starting from the next untried ratio index (= however
        many real rows that step already has) instead of re-attempting
        ratios that already have real, on-disk results.

    prior_step_rows is a summary-ready list (one dict per step strictly
    before resume_step_idx, via pick_step_row()) so the resume run's own
    final summary table can still show the full real history, not just
    the newly re-run tail.
    """
    rows = real_rows(trials_csv)
    if not rows:
        return 1, 0, []

    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append(r)

    max_step = max(by_step)
    max_rows = by_step[max_step]
    had_success = any(r["success"] == "True" for r in max_rows)

    if had_success or len(max_rows) >= num_ratio_candidates:
        resume_step, ratio_start = max_step + 1, 0
        prior_steps = sorted(by_step)
    else:
        resume_step, ratio_start = max_step, len(max_rows)
        prior_steps = sorted(s for s in by_step if s < max_step)

    prior_step_rows = [pick_step_row(by_step[s]) for s in prior_steps]
    return resume_step, ratio_start, prior_step_rows
