#!/usr/bin/env bash
# 计算占比(compute-fraction)扫描：从最小到最大，10个点，每10%一个台阶
# 内存/网络平分剩余份额 (memory = network = (1-compute)/2)，作为你已确认
# 可以采用的等分策略，避免引入额外变量。
#
# 结构完全比照项目里已验证跑通过的 sweep_area_fractions.sh：
#   - set -uo pipefail (不用 set -e，避免单点失败中断整个扫描)
#   - 每个点 python 输出直接重定向到独立 log 文件 (不经过 | tee，
#     避免管道退出码被吞掉，也避免你之前遇到的 zsh 交互粘贴问题)
#   - 每个点用 if/then/else 包裹：成功就把 result.json/area_budget.json
#     里的真实数字追加进 summary.csv；失败就打印 FAILED 并跳到下一个点，
#     不会让整个扫描因为某一个点（尤其是靠近可行性上限的点）而中断退出
#   - 全部跑完后按真实 sim_time_cycles + total_energy_uj 排名输出一张表
#
# 用法（在项目根目录，也就是这个文件所在目录执行）：
#   chmod +x area_sweep_compute.sh
#   ./area_sweep_compute.sh
# 或者放后台跑（10个点，每个点都是一次完整的真实 Docker 3阶段搜索，
# 可能会跑比较久）：
#   nohup ./area_sweep_compute.sh > area_sweep_compute.log 2>&1 &
#   disown
#   tail -f area_sweep_compute.log
#
# 注意：请直接执行这个脚本文件本身，不要把内容当命令块粘贴进交互式
# zsh 终端 —— zsh 默认不会把粘贴内容里的 "#" 当注释解析，会报
# "command not found: #"。

set -uo pipefail

source .venv/bin/activate

GEMM_M=1024
GEMM_N=1024
GEMM_K=1024
AREA_BUDGET_UM2=200000000
OUT_ROOT="outputs/area_sweep_compute"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"
echo "point,compute_frac,memory_frac,network_frac,sim_time_cycles,total_energy_uj,mesh,sram_kib_per_pe,fifo_capacity_packets,sram_search_saturated,fifo_search_saturated" > "$SUMMARY_CSV"

# name  compute  mem=net  (compute 从最小到最大排列；memory=network=(1-compute)/2)
POINTS=(
  "c05 0.05 0.475"
  "c15 0.15 0.425"
  "c25 0.25 0.375"
  "c35 0.35 0.325"
  "c45 0.45 0.275"
  "c55 0.55 0.225"
  "c65 0.65 0.175"
  "c75 0.75 0.125"
  "c85 0.85 0.075"
  "c95 0.95 0.025"
)

total=${#POINTS[@]}
idx=0
for point in "${POINTS[@]}"; do
  idx=$((idx+1))
  read -r name c mn <<< "$point"
  out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  start_ts=$(date +%s)
  echo "=== [$idx/$total] $name: compute=$c memory=$mn network=$mn -- started $(date) ==="

  if python -m wafersim --m "$GEMM_M" --n "$GEMM_N" --k "$GEMM_K" \
      --area-budget-um2 "$AREA_BUDGET_UM2" \
      --compute-fraction "$c" --memory-fraction "$mn" --network-fraction "$mn" \
      --out-dir "$out_dir" > "$OUT_ROOT/$name.log" 2>&1; then
    end_ts=$(date +%s)
    echo "    -> OK in $((end_ts-start_ts))s, extracting real results"
    python3 - "$out_dir" "$name" "$c" "$mn" "$mn" "$SUMMARY_CSV" <<'PYEOF'
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
echo "=== all $total compute-fraction points attempted, ranking real successes ==="
python3 - "$SUMMARY_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no successful points to rank -- check the per-point .log files")
    sys.exit(0)

for r in rows:
    r["compute_frac"] = float(r["compute_frac"])
    r["sim_time_cycles"] = float(r["sim_time_cycles"])
    r["total_energy_uj"] = float(r["total_energy_uj"])

rows.sort(key=lambda r: r["compute_frac"])

header = (
    f"{'point':8} {'compute':>8} {'mesh':>9} {'sram_kib':>9} {'fifo_pkts':>9} "
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
        f"{r['point']:8} {r['compute_frac']:>8.2f} {r['mesh']:>9} "
        f"{r['sram_kib_per_pe']:>9} {r['fifo_capacity_packets']:>9} "
        f"{r['sim_time_cycles']:>16.0f} {r['total_energy_uj']:>16.4f} {sat:>8}"
    )

best_time = min(rows, key=lambda r: r["sim_time_cycles"])
best_energy = min(rows, key=lambda r: r["total_energy_uj"])
print()
print(f"最快 (sim_time_cycles 最小): {best_time['point']} compute={best_time['compute_frac']:.2f} "
      f"mesh={best_time['mesh']} sim_time_cycles={best_time['sim_time_cycles']:.0f}")
print(f"最省能 (total_energy_uj 最小): {best_energy['point']} compute={best_energy['compute_frac']:.2f} "
      f"mesh={best_energy['mesh']} total_energy_uj={best_energy['total_energy_uj']:.4f}")
PYEOF

echo ""
echo "done -- full summary at $SUMMARY_CSV, per-point logs at $OUT_ROOT/<point>.log"
