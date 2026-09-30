#!/usr/bin/env python3
"""网络占比(network-fraction)密集扫描 -- "任意切法"版本，不强制计算=内存
二等分剩余份额，只要求剩余份额里有某种切法能让程序真的跑成功即可。

这版是直接照抄你项目里已经真实验证跑通过的
sweep_network_increasing_any_split.py 的核心逻辑，只改了两处，跟这次会
话里其余几轮800M扫描(包括刚跑完的内存any_split版)保持一致：
  1. AREA_BUDGET_UM2 改成 800,000,000。
  2. NETWORK_STEPS 改成跟网络占比等分那版一样的10个点：5%,15%,...,95%
     （原版是从1/3基线开始往上走13步）。

其余逻辑和真实依据一字不改，跟内存那版是同一套原理：mesh大小(进而PE数)
只由compute_area_fraction决定，网络这里是被扫的主变量，compute变成
"剩余份额"里的一部分——给compute剩余份额里更小的比例，意味着更小的mesh
(更少PE)，更小的mesh意味着内存和网络每PE能分到的预算都更大。所以这版
的RATIO_CANDIDATES(compute占剩余份额的比例)也是从小到大试
[0.20, 0.35, 0.50, 0.65, 0.80]，跟内存那版顺序一致(先试小compute份额
->小mesh，预期更容易成功、更快短路)。

内存拿的是精确浮点余量(remainder - compute)，不是各自独立取整。第一个
真实跑成功的切法就停，不找"最优切法"。每一次真实尝试(成功或失败)都记
进trials.csv。安全阀：连续3个网络点、5种切法全部失败，就提前停止。

参考数据（内存any_split版跑完的真实结果，供预期时间参考）：10个点里9
个第一个切法(comp_share=0.20)就成功，只有1个点(memory=5%)5种全试了都
失败，14次真实调用总共约2.3小时。这版网络参数结构完全一样，实际表现
以真跑的结果为准。

用法（在项目根目录，也就是这个文件所在目录执行，先确认Docker Desktop
已经启动）：
  source .venv/bin/activate
  nohup python3 sweep_network_any_split_800M.py > sweep_network_any_split_800M.log 2>&1 &
  disown
  tail -f sweep_network_any_split_800M.log
  cat outputs/area_sweep_network_any_split_800M/trials.csv
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
AREA_BUDGET_UM2 = 800_000_000
OUT_ROOT = Path("outputs/area_sweep_network_any_split_800M")
TRIALS_CSV = OUT_ROOT / "trials.csv"

# 网络占比从最小到最大，10个点，跟这次会话里"等分"那版
# (area_sweep_network_800M.sh) 的点位完全对齐，方便逐点比较。
NETWORK_STEPS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# compute拿"剩余份额(1-network)"的比例，按这个顺序依次真跑Docker试(从小
# 到大，小compute份额->小mesh->更容易成功，预期更快短路)；memory拿精确
# 浮点余量(remainder - compute)。
RATIO_CANDIDATES = [0.20, 0.35, 0.50, 0.65, 0.80]

MAX_CONSECUTIVE_STEP_FAILURES = 3  # 连续这么多个网络点、5种切法全部失败，就提前停止

trial_counter = 0


def fractions_for(network: float, compute_share_of_remainder: float) -> tuple[float, float, float]:
    remainder = 1.0 - network
    compute = remainder * compute_share_of_remainder
    memory = remainder - compute  # 精确余量，避免浮点漂移
    return compute, memory, network


def run_trial(network: float, compute_share: float, step_idx: int, ratio_idx: int) -> dict:
    global trial_counter
    trial_counter += 1
    idx = trial_counter
    c, m, n = fractions_for(network, compute_share)
    name = f"trial{idx:03d}_step{step_idx:02d}_n{network:.4f}_compshare{compute_share:.2f}"
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
    print(f"    [ratio {ratio_idx+1}/{len(RATIO_CANDIDATES)}, comp_share={compute_share:.2f}] "
          f"compute={c:.4f} memory={m:.4f} network={network:.4f} (sum={c+m+n:.10f}) ...", flush=True)
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


def run_step(step_idx: int, network: float) -> dict:
    """依次试 RATIO_CANDIDATES 里的每个切法，第一个真实跑成功的就返回；
    全部失败就返回最后一次尝试的结果(success=False)。"""
    print(f"=== [step {step_idx}/{len(NETWORK_STEPS)}] network={network:.4f} "
          f"(remainder={1.0-network:.4f} split across up to {len(RATIO_CANDIDATES)} ratios) "
          f"-- started {time.strftime('%H:%M:%S')} ===", flush=True)
    last_row = None
    for ratio_idx, compute_share in enumerate(RATIO_CANDIDATES):
        row = run_trial(network, compute_share, step_idx, ratio_idx)
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
    for step_idx, network in enumerate(NETWORK_STEPS, start=1):
        row = run_step(step_idx, network)
        rows.append(row)
        if row["success"]:
            consecutive_step_failures = 0
        else:
            consecutive_step_failures += 1
            if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
                stopped_early_at = step_idx
                remaining = NETWORK_STEPS[step_idx:]
                print(f"\n=== STOPPING EARLY after {consecutive_step_failures} consecutive network "
                      f"steps where EVERY ratio failed (network={network:.4f}) -- {len(remaining)} "
                      f"step(s) NOT run: {[f'{x:.4f}' for x in remaining]}. Raise "
                      f"MAX_CONSECUTIVE_STEP_FAILURES and re-run if you want them attempted "
                      f"anyway. ===\n", flush=True)
                break

    print("\n=== summary: network 5%->95%, compute+memory always summing to exactly "
          f"(1 - network), whichever of {RATIO_CANDIDATES} compute-share ratios was first "
          f"to work at each step (area budget {AREA_BUDGET_UM2} um^2) ===")
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
                f"{r['step']:<5} {r['network_frac']:>8.4f}   ALL {len(RATIO_CANDIDATES)} RATIOS FAILED -- "
                f"last tried comp_share={r['compute_share_of_remainder']:.2f}: {r.get('error', '')[:70]}"
            )
    if stopped_early_at is not None:
        skipped = NETWORK_STEPS[stopped_early_at:]
        print(f"\n{len(skipped)} step(s) skipped by early-stop: {[f'{x:.4f}' for x in skipped]}")

    print(f"\ndone -- every real trial (all ratios, all network values) logged at {TRIALS_CSV}")


if __name__ == "__main__":
    main()
