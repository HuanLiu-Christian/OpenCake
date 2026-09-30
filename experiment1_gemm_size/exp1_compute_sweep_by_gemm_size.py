#!/usr/bin/env python3
"""实验一(compute 主变量分支):固定芯片总面积 800,000,000 um^2,矩阵乘法
大小(M=N=K,方阵)在 GEMM_SIZES 里取 5 个真实值依次测试，每个矩阵大小下
都用"计算占比为主变量"的任意切法扫描(逻辑跟你项目里已经真实验证跑通过
的 sweep_compute_any_split_800M.py 一字不改，只是外面套了一层矩阵大小
的循环)。

跟 sweep_compute_any_split_800M.py 的关系：内层每个矩阵大小下的扫描逻辑
是那个文件的真实、已验证代码原样复制——COMPUTE_STEPS 10 个点(5%..95%)、
RATIO_CANDIDATES=[0.80,0.65,0.50,0.35,0.20](网络拿剩余份额的比例，从大
到小试)、内存拿精确浮点余量、第一个真实跑成功的切法就停、连续3个
compute点全灭就提前停止这一个矩阵大小的扫描(不是整个实验)。改的地方
只有两处：
  1. GEMM_M=GEMM_N=GEMM_K 从固定 1024 变成外层循环的变量。
  2. 输出路径按矩阵大小分开(outputs/experiment1_gemm_size/compute_sweep/
     gemm_<size>/)，并且每一行真实结果都多记一列 gemm_size，汇总进同一
     个 outputs/experiment1_gemm_size/compute_sweep/all_trials.csv，方便
     跨矩阵大小直接比较。

矩阵大小选择(GEMM_SIZES)：没有重复放 1024，因为这次会话里
sweep_compute_any_split_800M.py 已经在 1024x1024x1024 上真实跑完过(结果
就在 outputs/area_sweep_compute_any_split_800M/trials.csv)，可以直接当
这个"矩阵大小"实验的第6个数据点用，不用再烧一次真实Docker时间重跑。这
里选的5个新尺寸是 [128, 256, 512, 2048, 4096]，都是2的幂(和1024保持同
一种因数结构——只有2的幂这一种除数，这样"mesh只能踩2的幂档位"这个既有
发现在不同矩阵大小之间可以直接比较，不会混进"除数变丰富了"这个新变量)，
覆盖比1024小8倍到大4倍的范围。

代价提醒(如实说明，不是估计——用这次会话里1024x1024x1024上其余两个
"任意切法"扫描的真实历史数据做参考基准，不是外推公式)：
  - compute_any_split_800M(1024)：22次真实调用，从trials.csv里的
    elapsed_s列直接加总约 7.2 小时(跟这次会话里另一份脚本docstring引用
    的同一个真实数字一致，这里是我自己重新加总trials.csv验证过的)。
  - matrix尺寸越大，单PE要扛的tile_m/tile_n通常越大，越容易撞上SRAM
    建不出来的墙(这次会话里已经用真实代码 tile_m=ceil(gemm.m/active_rows)
    验证过这个机制)，具体某个尺寸会不会因此更快/更慢，只能真跑才知道，
    这里不假装能预测。
  - 5个尺寸各自独立跑，最坏情况(每个尺寸都跟1024差不多贵)大约
    5 * 7.2 ≈ 36 小时；如果某个尺寸(尤其是4096)因为mesh更难选、SRAM/
    FIFO二分搜索范围内失败更多而变慢，实际时间可能明显更长。这是三个
    脚本(compute/memory/network)之一，你说可以三个同时跑，那真实瓶颈
    是三个里最慢的那个，不是三个加起来。

路径bug修复记录(2026-09-02)：第一次真实后台跑的时候，因为是在
experiment1_gemm_size/ 子目录下启动的，而 EXPERIMENT_ROOT 当时是写成相对
路径 "outputs/experiment1_gemm_size/compute_sweep"，实际就写到了
experiment1_gemm_size/outputs/experiment1_gemm_size/compute_sweep/ 这个
嵌套错误的位置(而不是项目根目录下的 outputs/experiment1_gemm_size/
compute_sweep/)。现在改成用 Path(__file__) 定位项目根目录，不管你从项目
根目录还是从 experiment1_gemm_size/ 子目录启动这个脚本，输出都会真实落在
项目根目录的 outputs/experiment1_gemm_size/compute_sweep/ 下，不会再嵌套。
那次跑串了、也被提前中断的部分数据(只到 gemm=128 一半左右)已经删掉重跑。

用法(在项目根目录下的 experiment1_gemm_size/ 子目录执行，先确认Docker
Desktop已经启动，建议跟另外两个"主变量"脚本同时后台跑)：
  cd experiment1_gemm_size
  source ../.venv/bin/activate
  nohup python3 exp1_compute_sweep_by_gemm_size.py > exp1_compute_sweep_by_gemm_size.log 2>&1 &
  disown
  tail -f exp1_compute_sweep_by_gemm_size.log
  cat ../outputs/experiment1_gemm_size/compute_sweep/all_trials.csv

重要:上一次真实后台跑到一半被中断了(大概率是Terminal窗口被关掉/App被
退出，nohup+disown只保护"shell退出"这一种情况，保护不了"Terminal.app
整个被关掉"这种情况杀掉子进程)。这次(compute分支最坏情况约36小时)如果
还用普通Terminal窗口挂着跑，中途不能把Terminal.app整个退出/重启电脑/
让Mac真的睡眠(合盖大概率会真的休眠，跟"锁屏"不是一回事)。更稳妥的做法
(任选一种)：
  1. 用 tmux/screen 起一个真正独立于Terminal的会话(`tmux new -s exp1c`，
     里面跑上面的命令，然后 Ctrl-b d 退出附着——这样即使Terminal.app整个
     退出，跑在tmux里的进程也不受影响，回来用 `tmux attach -t exp1c`
     接回去看)。
  2. 没装tmux/screen的话，至少全程留着这个Terminal窗口/App不要关，且用
     `caffeinate -i nohup python3 ... &` 防止Mac合盖/超时自动睡眠打断它。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

AREA_BUDGET_UM2 = 800_000_000

# 见模块docstring："矩阵大小选择"一节 -- 没有重复放1024(已有真实数据)，
# 都是2的幂，跟1024保持同一种除数结构，方便跨尺寸直接比较。
GEMM_SIZES = [128, 256, 512, 2048, 4096]

# 项目根目录 -- 用脚本自己的真实文件路径定位，不依赖调用者的当前工作目录
# (见上面"路径bug修复记录")。this_file = .../experiment1_gemm_size/xxx.py，
# .parent 是 experiment1_gemm_size/，再 .parent 一次才是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiment1_gemm_size" / "compute_sweep"
ALL_TRIALS_CSV = EXPERIMENT_ROOT / "all_trials.csv"

# 跟 sweep_compute_any_split_800M.py 完全一致，一字不改。
COMPUTE_STEPS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
RATIO_CANDIDATES = [0.80, 0.65, 0.50, 0.35, 0.20]
MAX_CONSECUTIVE_STEP_FAILURES = 3  # 每个矩阵大小自己的连续失败计数，互不影响

all_trial_counter = 0


def fractions_for(compute: float, network_share_of_remainder: float) -> tuple[float, float, float]:
    remainder = 1.0 - compute
    network = remainder * network_share_of_remainder
    memory = remainder - network  # 精确余量，避免浮点漂移
    return compute, memory, network


def run_trial(
    gemm_size: int, out_root: Path, trials_csv: Path,
    compute: float, network_share: float, step_idx: int, ratio_idx: int, trial_idx_in_size: int,
) -> dict:
    global all_trial_counter
    all_trial_counter += 1
    c, m, n = fractions_for(compute, network_share)
    name = f"trial{trial_idx_in_size:03d}_step{step_idx:02d}_c{compute:.4f}_netshare{network_share:.2f}"
    out_dir = out_root / name
    log_path = out_root / f"{name}.log"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "wafersim",
        "--m", str(gemm_size), "--n", str(gemm_size), "--k", str(gemm_size),
        "--area-budget-um2", str(AREA_BUDGET_UM2),
        "--compute-fraction", f"{c:.10f}",
        "--memory-fraction", f"{m:.10f}",
        "--network-fraction", f"{n:.10f}",
        "--out-dir", str(out_dir),
    ]
    print(f"    [gemm={gemm_size} ratio {ratio_idx+1}/{len(RATIO_CANDIDATES)}, net_share={network_share:.2f}] "
          f"compute={compute:.4f} memory={m:.4f} network={n:.4f} (sum={c+m+n:.10f}) ...", flush=True)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    row = {
        "gemm_size": gemm_size, "trial": all_trial_counter, "step": step_idx,
        "compute_frac": c, "memory_frac": m, "network_frac": n,
        "network_share_of_remainder": network_share, "elapsed_s": round(elapsed, 1),
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
        "gemm_size", "trial", "step", "compute_frac", "memory_frac", "network_frac",
        "network_share_of_remainder", "elapsed_s", "success", "sim_time_cycles",
        "total_energy_uj", "mesh", "sram_kib_per_pe", "fifo_capacity_packets", "error",
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def run_step(gemm_size: int, out_root: Path, trials_csv: Path, step_idx: int, compute: float, trial_counter_start: int) -> tuple[dict, int]:
    """依次试 RATIO_CANDIDATES 里的每个切法，第一个真实跑成功的就返回；
    全部失败就返回最后一次尝试的结果(success=False)。返回 (row, 用掉的
    trial 计数)，trial 计数只在这个矩阵大小内部编号，方便跟单尺寸版脚本
    的目录命名习惯保持一致。"""
    print(f"=== [gemm={gemm_size} step {step_idx}/{len(COMPUTE_STEPS)}] compute={compute:.4f} "
          f"(remainder={1.0-compute:.4f} split across up to {len(RATIO_CANDIDATES)} ratios) "
          f"-- started {time.strftime('%H:%M:%S')} ===", flush=True)
    last_row = None
    trial_idx = trial_counter_start
    for ratio_idx, network_share in enumerate(RATIO_CANDIDATES):
        trial_idx += 1
        row = run_trial(gemm_size, out_root, trials_csv, compute, network_share, step_idx, ratio_idx, trial_idx)
        last_row = row
        if row["success"]:
            return row, trial_idx
    return last_row, trial_idx


def run_for_gemm_size(gemm_size: int) -> None:
    out_root = EXPERIMENT_ROOT / f"gemm_{gemm_size}"
    trials_csv = out_root / "trials.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    if trials_csv.exists():
        trials_csv.unlink()

    print(f"\n{'#'*70}\n# GEMM SIZE {gemm_size}x{gemm_size}x{gemm_size} -- compute-primary any-split sweep\n{'#'*70}", flush=True)

    rows = []
    consecutive_step_failures = 0
    stopped_early_at = None
    trial_idx = 0
    for step_idx, compute in enumerate(COMPUTE_STEPS, start=1):
        row, trial_idx = run_step(gemm_size, out_root, trials_csv, step_idx, compute, trial_idx)
        rows.append(row)
        if row["success"]:
            consecutive_step_failures = 0
        else:
            consecutive_step_failures += 1
            if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                stopped_early_at = step_idx
                remaining = COMPUTE_STEPS[step_idx:]
                print(f"\n=== [gemm={gemm_size}] STOPPING EARLY after {consecutive_step_failures} consecutive "
                      f"compute steps where EVERY ratio failed (compute={compute:.4f}) -- {len(remaining)} "
                      f"step(s) NOT run for this matrix size: {[f'{x:.4f}' for x in remaining]}. ===\n", flush=True)
                break

    print(f"\n=== [gemm={gemm_size}] summary: compute 5%->95%, memory+network always summing to exactly "
          f"(1 - compute), whichever of {RATIO_CANDIDATES} network-share ratios was first "
          f"to work at each step (area budget {AREA_BUDGET_UM2} um^2) ===")
    header = (
        f"{'step':5} {'compute':>8} {'memory':>8} {'network':>8} {'net_share':>9} {'mesh':>9} "
        f"{'sram_kib':>9} {'fifo_pkts':>9} {'sim_time_cycles':>16} {'total_energy_uj':>16}"
    )
    print(header)
    for r in rows:
        if r["success"]:
            print(
                f"{r['step']:<5} {r['compute_frac']:>8.4f} {r['memory_frac']:>8.4f} {r['network_frac']:>8.4f} "
                f"{r['network_share_of_remainder']:>9.2f} {r['mesh']:>9} {r['sram_kib_per_pe']:>9} "
                f"{r['fifo_capacity_packets']:>9} {float(r['sim_time_cycles']):>16.0f} "
                f"{float(r['total_energy_uj']):>16.4f}"
            )
        else:
            print(
                f"{r['step']:<5} {r['compute_frac']:>8.4f}   ALL {len(RATIO_CANDIDATES)} RATIOS FAILED -- "
                f"last tried net_share={r['network_share_of_remainder']:.2f}: {r.get('error', '')[:70]}"
            )
    if stopped_early_at is not None:
        skipped = COMPUTE_STEPS[stopped_early_at:]
        print(f"\n[gemm={gemm_size}] {len(skipped)} step(s) skipped by early-stop: {[f'{x:.4f}' for x in skipped]}")

    print(f"\n[gemm={gemm_size}] done -- every real trial for this matrix size logged at {trials_csv}")


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    if ALL_TRIALS_CSV.exists():
        ALL_TRIALS_CSV.unlink()

    print(f"experiment 1 (compute-primary branch): area budget fixed at {AREA_BUDGET_UM2} um^2, "
          f"matrix size varies over {GEMM_SIZES} (M=N=K, square) ...", flush=True)

    for gemm_size in GEMM_SIZES:
        run_for_gemm_size(gemm_size)

    print(f"\n{'='*70}\nALL {len(GEMM_SIZES)} MATRIX SIZES DONE (compute-primary branch). "
          f"Combined real results (every trial, every size) at {ALL_TRIALS_CSV}\n{'='*70}")


if __name__ == "__main__":
    main()
