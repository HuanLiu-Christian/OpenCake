#!/usr/bin/env bash
# 网络占比(network-fraction)扫描，总面积预算 800,000,000 um^2（跟计算、
# 内存两轮800M用同一个预算，方便三个变量的"墙"位置直接对比）。
#
# 配置和前两版对齐：从最小到最大10个点，每10%一个台阶，
# 计算 = 内存 = (1-网络)/2，平分剩余份额。
#
# 结果落到单独的 outputs/area_sweep_network_800M/ 目录，不会覆盖之前任何
# 一轮(计算200M/800M、内存800M)的数据。
#
# 结构和前两版完全一致，已经在你机器上真实验证跑通过：
#   - set -uo pipefail (不用 set -e，避免单点失败中断整个扫描)
#   - 每个点 python 输出直接重定向到独立 log 文件（不经过 | tee）
#   - 每个点用 if/then/else 包裹：成功就把真实结果追加进 summary.csv；
#     失败就打印 FAILED 并跳到下一个点，不会中断整个扫描
#   - 全部跑完后按真实 sim_time_cycles + total_energy_uj 排名输出一张表，
#     并单独标出触及SRAM/FIFO安全上限(4096 KiB / 2048 packets)的点
#
# 用法（在项目根目录，也就是这个文件所在目录执行）：
#   chmod +x area_sweep_network_800M.sh
#   ./area_sweep_network_800M.sh
# 或者放后台跑：
#   nohup ./area_sweep_network_800M.sh > area_sweep_network_800M.log 2>&1 &
#   disown
#   tail -f area_sweep_network_800M.log
#
# 跑之前确认 Docker Desktop 已经启动。请直接执行这个脚本文件本身，
# 不要把内容当命令块粘贴进交互式 zsh 终端。

set -uo pipefail

source .venv/bin/activate

GEMM_M=1024
GEMM_N=1024
GEMM_K=1024
AREA_BUDGET_UM2=800000000
OUT_ROOT="outputs/area_sweep_network_800M"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"
echo "point,compute_frac,memory_frac,network_frac,sim_time_cycles,total_energy_uj,mesh,sram_kib_per_pe,fifo_capacity_packets,sram_search_saturated,fifo_search_saturated" > "$SUMMARY_CSV"

# name  network  compute=memory  (network 从最小到最大排列；compute=memory=(1-network)/2)
POINTS=(
  "n05 0.05 0.475"
  "n15 0.15 0.425"
  "n25 0.25 0.375"
  "n35 0.35 0.325"
  "n45 0.45 0.275"
  "n55 0.55 0.225"
  "n65 0.65 0.175"
  "n75 0.75 0.125"
  "n85 0.85 0.075"
  "n95 0.95 0.025"
)

total=${#POINTS[@]}
idx=0
for point in "${POINTS[@]}"; do
  idx=$((idx+1))
  read -r name n cm <<< "$point"
  out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  start_ts=$(date +%s)
  echo "=== [$idx/$total] $name: network=$n compute=$cm memory=$cm -- started $(date) ==="

  if python -m wafersim --m "$GEMM_M" --n "$GEMM_N" --k "$GEMM_K" \
      --area-budget-um2 "$AREA_BUDGET_UM2" \
      --compute-fraction "$cm" --memory-fraction "$cm" --network-fraction "$n" \
      --out-dir "$out_dir" > "$OUT_ROOT/$name.log" 2>&1; then
    end_ts=$(date +%s)
    echo "    -> OK in $((end_ts-start_ts))s, extracting real results"
    python3 - "$out_dir" "$name" "$cm" "$cm" "$n" "$SUMMARY_CSV" <<'PYEOF'
import json, sys, csv

out_dir, name, c, m, n, summary_csv = sys.argv[1:7]
with open(f"{out_dir}/result.json") as f:
    result = json.load(f)
with open(f"{out_dir}/area_budget.json") as f:
    budget = json.load(f)

row = [
    name, c, m, n,
    result["sim_time_cycles"],
    result["energy"]["total_uj"],
    f'{budget["mesh_rows"]}x{budget["mesh_cols"]}',
    budget["sram_kib_per_pe"],
    budget["fifo_capacity_packets"],
    budget["sram_search_saturated"],
    budget["fifo_search_saturated"],
]
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)
PYEOF
  else
    end_ts=$(date +%s)
    echo "    -> FAILED after $((end_ts-start_ts))s -- see $OUT_ROOT/$name.log, skipping this point"
  fi
done

echo ""
echo "=== all $total network-fraction points attempted, ranking real successes ==="
python3 - "$SUMMARY_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no successful points to rank -- check the per-point .log files")
    sys.exit(0)

for r in rows:
    r["network_frac"] = float(r["network_frac"])
    r["sim_time_cycles"] = float(r["sim_time_cycles"])
    r["total_energy_uj"] = float(r["total_energy_uj"])

rows.sort(key=lambda r: r["network_frac"])

header = (
    f"{'point':8} {'network':>8} {'mesh':>9} {'sram_kib':>9} {'fifo_pkts':>9} "
    f"{'sim_time_cycles':>16} {'total_energy_uj':>16} {'sat?':>8}"
)
print(header)
for r in rows:
    sat_parts = []
    if r["sram_search_saturated"] == "True":
        sat_parts.append("SRAM")
    if r["fifo_search_saturated"] == "True":
        sat_parts.append("FIFO")
    sat = "+".join(sat_parts)
    print(
        f"{r['point']:8} {r['network_frac']:>8.2f} {r['mesh']:>9} "
        f"{r['sram_kib_per_pe']:>9} {r['fifo_capacity_packets']:>9} "
        f"{r['sim_time_cycles']:>16.0f} {r['total_energy_uj']:>16.4f} {sat:>8}"
    )

best_time = min(rows, key=lambda r: r["sim_time_cycles"])
best_energy = min(rows, key=lambda r: r["total_energy_uj"])
print()
print(f"最快 (sim_time_cycles 最小): {best_time['point']} network={best_time['network_frac']:.2f} "
      f"mesh={best_time['mesh']} sim_time_cycles={best_time['sim_time_cycles']:.0f}")
print(f"最省能 (total_energy_uj 最小): {best_energy['point']} network={best_energy['network_frac']:.2f} "
      f"mesh={best_energy['mesh']} total_energy_uj={best_energy['total_energy_uj']:.4f}")

sat_rows = [r for r in rows if r["sram_search_saturated"] == "True" or r["fifo_search_saturated"] == "True"]
if sat_rows:
    print()
    print("注意: 以下点的 SRAM 或 FIFO 搜索卡在了人为设的安全上限 (4096 KiB / 2048 packets) 上，")
    print("      不代表真实预算就只能到这个数字，见 area_budget.py 的 saturation_warning 说明:")
    for r in sat_rows:
        print(f"  {r['point']} network={r['network_frac']:.2f}")
PYEOF

echo ""
echo "done -- full summary at $SUMMARY_CSV, per-point logs at $OUT_ROOT/<point>.log"
