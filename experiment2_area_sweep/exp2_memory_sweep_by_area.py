#!/usr/bin/env python3
"""实验二(memory 主变量分支)：固定矩阵大小 M=N=K=1024(方阵)，芯片总面积
预算(area-budget-um2)在 AREA_BUDGETS 里取 4 个真实值依次测试，每个面积
预算下都用"内存占比为主变量"的任意切法扫描——内层扫描逻辑跟实验一的
exp1_memory_sweep_by_gemm_size.py（以及它复制自的 sweep_memory_any_
split_800M.py）完全一字不改：MEMORY_STEPS 10 个点(5%..95%)、
RATIO_CANDIDATES=[0.20,0.35,0.50,0.65,0.80]（compute 拿剩余份额的比例，
从小到大试）、网络拿精确浮点余量、第一个真实跑成功的切法就停、连续 3 个
memory 点全灭就提前停止这一个面积预算的扫描(不是整个实验)。跟同目录
exp2_compute_sweep_by_area.py / exp2_network_sweep_by_area.py 是同一组
实验的另外两个变量分支，三个脚本可以同时后台跑。

跟实验一(exp1_*_sweep_by_gemm_size.py)的关系：实验一是"面积固定
(800M)，矩阵大小变"，这个实验二反过来，是"矩阵大小固定(1024)，面积
变"——改的地方正好对调：
  1. GEMM_M=GEMM_N=GEMM_K 从外层循环变量变回固定常量 1024。
  2. AREA_BUDGET_UM2 从固定常量变成外层循环的变量。
  3. 输出路径按面积预算分开(outputs/experiment2_area_sweep/memory_sweep/
     area_<um2>/)，并且每一行真实结果都多记一列 area_budget_um2(替换掉
     实验一里的 gemm_size 列)，汇总进同一个
     outputs/experiment2_area_sweep/memory_sweep/all_trials.csv。

面积预算选择(AREA_BUDGETS)：以你已有的真实基准点 800,000,000 um^2 为
中心对称缩放，两侧各扩/缩 2 倍和 4 倍：[200M, 400M, 800M, 1.6B, 3.2B]。
其中 **800M 这一档不在这个脚本的 AREA_BUDGETS 里重新跑**——1024x1024x
1024、800M um^2、memory-primary any-split 这个组合，你已经有一份真实
的历史数据在 outputs/area_sweep_memory_any_split_800M/trials.csv（14
次真实 Docker 调用，真实累计约 2.30 小时，我在这次会话里重新加总过
elapsed_s 列核实过这个数字，不是转述）。这份历史数据的 MEMORY_STEPS/
RATIO_CANDIDATES 跟本脚本完全一致（本来就是同一份代码复制出来的），
可以直接当这 5 个面积点里"800M"这一个点用，不需要再烧一次真实 Docker
时间重跑——所以 AREA_BUDGETS 只列了 4 个新值：[200_000_000, 400_000_000,
1_600_000_000, 3_200_000_000]。

代价提醒(如实说明，不是估计——用 800M/1024 这个已有真实历史点的
elapsed_s 总和做参考基准，不是外推公式)：
  - memory_any_split_800M(1024)：14 次真实调用，总共约 2.30 小时，是
    三个"任意切法"版本里最快的一个。
  - 面积比 800M 更小(200M/400M)会不会更容易撞墙、提前触发"连续 3 步
    全灭"提前停止，还是面积更大(1.6B/3.2B)会不会因为 mesh 更大导致单
    次 trial 更慢——这两个方向哪个更贵、贵多少，只有真跑这 4 个新面积
    点才知道，这里不假装能预测。已知的真实机制是：面积预算直接决定
    wafer_area_budget.py 里能二分搜出的 sram_kib_per_pe /
    fifo_capacity_packets 的搜索空间上限，面积越小这个空间越窄，越容易
    出现"no sram_kib_per_pe in [1, 4096] KiB fits the per-PE memory
    budget"这类真实失败(实验一的 compute/network 分支在 800M 上已经真实
    观测到大量这类失败)，所以 200M 这一档 ALL 5 RATIOS FAILED 甚至提前
    停止整个面积点的可能性不低，不是脚本bug，是真实的资源约束。

用法(在项目根目录下的 experiment2_area_sweep/ 子目录执行，先确认Docker
Desktop已经启动，建议跟另外两个"主变量"脚本同时后台跑，用 tmux/screen
而不是普通 Terminal 窗口，避免中途被打断——参考实验一脚本docstring里
"重要"那一段，理由完全一样)：
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_memory_sweep_by_area.py > exp2_memory_sweep_by_area.log 2>&1 &
  disown
  tail -f exp2_memory_sweep_by_area.log
  cat ../outputs/experiment2_area_sweep/memory_sweep/all_trials.csv
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# 固定矩阵大小 -- 见模块docstring，跟实验一反过来，这次面积是外层循环变量。
GEMM_M = 1024
GEMM_N = 1024
GEMM_K = 1024

# 4 个新面积点 -- 见模块docstring"面积预算选择"一节，800M 那一档复用已有
# 真实历史数据(outputs/area_sweep_memory_any_split_800M/trials.csv)，不
# 在这里重跑。
AREA_BUDGETS = [200_000_000, 400_000_000, 1_600_000_000, 3_200_000_000]

# 项目根目录 -- 用脚本自己的真实文件路径定位，不依赖调用者的当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiment2_area_sweep" / "memory_sweep"
ALL_TRIALS_CSV = EXPERIMENT_ROOT / "all_trials.csv"

# 跟 sweep_memory_any_split_800M.py / exp1_memory_sweep_by_gemm_size.py 完全一致，一字不改。
MEMORY_STEPS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
RATIO_CANDIDATES = [0.20, 0.35, 0.50, 0.65, 0.80]
MAX_CONSECUTIVE_STEP_FAILURES = 3  # 每个面积预算自己的连续失败计数，互不影响

all_trial_counter = 0


def fractions_for(memory: float, compute_share_of_remainder: float) -> tuple[float, float, float]:
    remainder = 1.0 - memory
    compute = remainder * compute_share_of_remainder
    network = remainder - compute  # 精确余量，避免浮点漂移
    return compute, memory, network


def run_trial(
    area_budget: int, out_root: Path, trials_csv: Path,
    memory: float, compute_share: float, step_idx: int, ratio_idx: int, trial_idx_in_area: int,
) -> dict:
    global all_trial_counter
    all_trial_counter += 1
    c, m, n = fractions_for(memory, compute_share)
    name = f"trial{trial_idx_in_area:03d}_step{step_idx:02d}_m{memory:.4f}_compshare{compute_share:.2f}"
    out_dir = out_root / name
    log_path = out_root / f"{name}.log"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "wafersim",
        "--m", str(GEMM_M), "--n", str(GEMM_N), "--k", str(GEMM_K),
        "--area-budget-um2", str(area_budget),
        "--compute-fraction", f"{c:.10f}",
        "--memory-fraction", f"{m:.10f}",
        "--network-fraction", f"{n:.10f}",
        "--out-dir", str(out_dir),
    ]
    print(f"    [area={area_budget:,} ratio {ratio_idx+1}/{len(RATIO_CANDIDATES)}, comp_share={compute_share:.2f}] "
          f"compute={c:.4f} memory={m:.4f} network={n:.4f} (sum={c+m+n:.10f}) ...", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    row = {
        "area_budget_um2": area_budget, "trial": all_trial_counter, "step": step_idx,
        "compute_frac": c, "memory_frac": m, "network_frac": n,
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

    write_row(trials_csv, row)
    write_row(ALL_TRIALS_CSV, row)
    return row


def write_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    fieldnames = [
        "area_budget_um2", "trial", "step", "compute_frac", "memory_frac", "network_frac",
        "compute_share_of_remainder", "elapsed_s", "success", "sim_time_cycles",
        "total_energy_uj", "mesh", "sram_kib_per_pe", "fifo_capacity_packets", "error",
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def run_step(area_budget: int, out_root: Path, trials_csv: Path, step_idx: int, memory: float, trial_counter_start: int) -> tuple[dict, int]:
    """依次试 RATIO_CANDIDATES 里的每个切法，第一个真实跑成功的就返回；
    全部失败就返回最后一次尝试的结果(success=False)。"""
    print(f"=== [area={area_budget:,} step {step_idx}/{len(MEMORY_STEPS)}] memory={memory:.4f} "
          f"(remainder={1.0-memory:.4f} split across up to {len(RATIO_CANDIDATES)} ratios) "
          f"-- started {time.strftime('%H:%M:%S')} ===", flush=True)
    last_row = None
    trial_idx = trial_counter_start
    for ratio_idx, compute_share in enumerate(RATIO_CANDIDATES):
        trial_idx += 1
        row = run_trial(area_budget, out_root, trials_csv, memory, compute_share, step_idx, ratio_idx, trial_idx)
        last_row = row
        if row["success"]:
            return row, trial_idx
    return last_row, trial_idx


def run_for_area(area_budget: int) -> None:
    out_root = EXPERIMENT_ROOT / f"area_{area_budget}"
    trials_csv = out_root / "trials.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    if trials_csv.exists():
        trials_csv.unlink()

    print(f"\n{'#'*70}\n# AREA BUDGET {area_budget:,} um^2 -- memory-primary any-split sweep "
          f"(GEMM {GEMM_M}x{GEMM_N}x{GEMM_K} fixed)\n{'#'*70}", flush=True)

    rows = []
    consecutive_step_failures = 0
    stopped_early_at = None
    trial_idx = 0
    for step_idx, memory in enumerate(MEMORY_STEPS, start=1):
        row, trial_idx = run_step(area_budget, out_root, trials_csv, step_idx, memory, trial_idx)
        rows.append(row)
        if row["success"]:
            consecutive_step_failures = 0
        else:
            consecutive_step_failures += 1
            if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                stopped_early_at = step_idx
                remaining = MEMORY_STEPS[step_idx:]
                print(f"\n=== [area={area_budget:,}] STOPPING EARLY after {consecutive_step_failures} consecutive "
                      f"memory steps where EVERY ratio failed (memory={memory:.4f}) -- {len(remaining)} "
                      f"step(s) NOT run for this area budget: {[f'{x:.4f}' for x in remaining]}. ===\n", flush=True)
                break

    print(f"\n=== [area={area_budget:,}] summary: memory 5%->95%, compute+network always summing to exactly "
          f"(1 - memory), whichever of {RATIO_CANDIDATES} compute-share ratios was first "
          f"to work at each step (GEMM {GEMM_M}x{GEMM_N}x{GEMM_K} fixed) ===")
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
                f"last tried comp_share={r['compute_share_of_remainder']:.2f}: {r.get('error', '')[:70]}"
            )
    if stopped_early_at is not None:
        skipped = MEMORY_STEPS[stopped_early_at:]
        print(f"\n[area={area_budget:,}] {len(skipped)} step(s) skipped by early-stop: {[f'{x:.4f}' for x in skipped]}")

    print(f"\n[area={area_budget:,}] done -- every real trial for this area budget logged at {trials_csv}")


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    if ALL_TRIALS_CSV.exists():
        ALL_TRIALS_CSV.unlink()

    print(f"experiment 2 (memory-primary branch): matrix size fixed at {GEMM_M}x{GEMM_N}x{GEMM_K}, "
          f"area budget varies over {AREA_BUDGETS} um^2 (800M um^2 point reused from existing real "
          f"outputs/area_sweep_memory_any_split_800M/trials.csv, not rerun here) ...", flush=True)

    for area_budget in AREA_BUDGETS:
        run_for_area(area_budget)

    print(f"\n{'='*70}\nALL {len(AREA_BUDGETS)} AREA BUDGETS DONE (memory-primary branch). "
          f"Combined real results (every trial, every area) at {ALL_TRIALS_CSV}\n"
          f"(the 800M um^2 point is separately at "
          f"../outputs/area_sweep_memory_any_split_800M/trials.csv, same schema minus the "
          f"area_budget_um2 column -- treat that as area_budget_um2=800000000)\n{'='*70}")


if __name__ == "__main__":
    main()
