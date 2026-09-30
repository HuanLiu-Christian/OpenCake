#!/usr/bin/env bash
# Real-Docker sweep over compute/memory/network area-fraction combinations,
# built directly on top of the already-verified
# `python -m wafersim --area-budget-um2 ... --compute-fraction ... \
#  --memory-fraction ... --network-fraction ...` pipeline
# (see cli.py's _derive_wafer_from_area_budget / area_budget.py).
#
# Every number this script reports comes from a real timeloop-model/
# accelergy Docker run for that exact fraction combo -- this script only
# decides WHICH combos to try and how to rank their real results. It does
# not estimate, interpolate, or fabricate any number itself.
#
# Usage (run this from the project root, with your usual venv active):
#   source .venv/bin/activate
#   nohup bash sweep_area_fractions.sh > sweep_area_fractions.log 2>&1 &
#   disown
#   tail -f sweep_area_fractions.log                # watch progress
#   cat outputs/area_budget_sweep/summary.csv        # watch results land
#
# This can realistically take HOURS: each combo runs a full 3-stage
# real-Docker area-budget derivation (1 compute probe + up to ~24 SRAM
# binary-search steps + up to ~24 FIFO binary-search steps) plus a full
# tile_k search (one real timeloop-model run per SRAM-feasible tile_k).
# That's why it's meant to run in the background on your own machine
# rather than inside a single foreground command.

set -uo pipefail

GEMM_M=1024
GEMM_N=1024
GEMM_K=1024
AREA_BUDGET_UM2=200000000
OUT_ROOT="outputs/area_budget_sweep"
SUMMARY_CSV="$OUT_ROOT/summary.csv"

mkdir -p "$OUT_ROOT"
echo "combo,compute_frac,memory_frac,network_frac,sim_time_cycles,total_energy_uj,mesh,sram_kib_per_pe,fifo_capacity_packets,sram_search_saturated,fifo_search_saturated" > "$SUMMARY_CSV"

# name          compute  memory  network   (each row sums to 1.0 -- full,
# non-overlapping allocation of the 200,000,000 um^2 budget, same three
# categories as contract.py's compute_area_um2/memory_area_um2/
# network_area_um2). Spans low->high compute (including past the ~13.58%
# threshold where mac_area_um2=6628.00 lets the mesh jump from 32x32 to
# 64x64, since 4096*6628/200,000,000 = 0.1358), and low->high network/
# memory splits around that.
COMBOS=(
  "c03_m62_n35 0.03 0.62 0.35"
  "c05_m70_n25 0.05 0.70 0.25"
  "c05_m45_n50 0.05 0.45 0.50"
  "c05_m80_n15 0.05 0.80 0.15"
  "c05_m20_n75 0.05 0.20 0.75"
  "c15_m50_n35 0.15 0.50 0.35"
  "c20_m40_n40 0.20 0.40 0.40"
  "c30_m40_n30 0.30 0.40 0.30"
  "c10_m45_n45 0.10 0.45 0.45"
  "c33_m33_n34 0.33 0.33 0.34"
)

total=${#COMBOS[@]}
idx=0
for combo in "${COMBOS[@]}"; do
  idx=$((idx+1))
  read -r name c m n <<< "$combo"
  out_dir="$OUT_ROOT/$name"
  start_ts=$(date +%s)
  echo "=== [$idx/$total] $name: compute=$c memory=$m network=$n -- started $(date) ==="

  if python -m wafersim --m "$GEMM_M" --n "$GEMM_N" --k "$GEMM_K" \
      --area-budget-um2 "$AREA_BUDGET_UM2" \
      --compute-fraction "$c" --memory-fraction "$m" --network-fraction "$n" \
      --out-dir "$out_dir" > "$OUT_ROOT/$name.log" 2>&1; then
    end_ts=$(date +%s)
    echo "    -> OK in $((end_ts-start_ts))s, extracting real results"
    python3 - "$out_dir" "$name" "$c" "$m" "$n" "$SUMMARY_CSV" <<'PYEOF'
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
    echo "    -> FAILED after $((end_ts-start_ts))s -- see $OUT_ROOT/$name.log, skipping this combo"
  fi
done

echo ""
echo "=== all $total combos attempted, ranking by real sim_time_cycles + total_energy_uj ==="
python3 - "$SUMMARY_CSV" <<'PYEOF'
import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no successful combos to rank -- check the per-combo .log files")
    sys.exit(0)

for r in rows:
    r["sim_time_cycles"] = float(r["sim_time_cycles"])
    r["total_energy_uj"] = float(r["total_energy_uj"])

by_time = sorted(rows, key=lambda r: r["sim_time_cycles"])
by_energy = sorted(rows, key=lambda r: r["total_energy_uj"])
time_rank = {r["combo"]: i for i, r in enumerate(by_time, 1)}
energy_rank = {r["combo"]: i for i, r in enumerate(by_energy, 1)}
for r in rows:
    r["time_rank"] = time_rank[r["combo"]]
    r["energy_rank"] = energy_rank[r["combo"]]
    r["avg_rank"] = (r["time_rank"] + r["energy_rank"]) / 2

ranked = sorted(rows, key=lambda r: (r["avg_rank"], r["sim_time_cycles"]))

header = (
    f"{'combo':16} {'compute':>8} {'memory':>8} {'network':>8} {'mesh':>9} "
    f"{'sram_kib':>9} {'fifo_pkts':>9} {'sim_time_cycles':>16} "
    f"{'total_energy_uj':>16} {'t_rank':>7} {'e_rank':>7} {'avg_rank':>9} {'sat?':>8}"
)
print(header)
for r in ranked:
    sat_parts = []
    if r["sram_search_saturated"] == "True":
        sat_parts.append("SRAM")
    if r["fifo_search_saturated"] == "True":
        sat_parts.append("FIFO")
    sat = "+".join(sat_parts)
    print(
        f"{r['combo']:16} {r['compute_frac']:>8} {r['memory_frac']:>8} {r['network_frac']:>8} "
        f"{r['mesh']:>9} {r['sram_kib_per_pe']:>9} {r['fifo_capacity_packets']:>9} "
        f"{r['sim_time_cycles']:>16.0f} {r['total_energy_uj']:>16.4f} "
        f"{r['time_rank']:>7} {r['energy_rank']:>7} {r['avg_rank']:>9.1f} {sat:>8}"
    )

best = ranked[0]
print()
print(f"BEST (lowest avg of time-rank + energy-rank): {best['combo']}")
print(f"  compute={best['compute_frac']} memory={best['memory_frac']} network={best['network_frac']}")
print(f"  sim_time_cycles={best['sim_time_cycles']:.0f}, total_energy_uj={best['total_energy_uj']:.4f}")
if best["sram_search_saturated"] == "True" or best["fifo_search_saturated"] == "True":
    print("  WARNING: this combo's derivation saturated a search ceiling -- its hardware config")
    print("  is NOT confirmed to be a real budget-driven optimum, see area_budget.py's")
    print("  saturation_warning docstring before trusting this as the final answer.")
PYEOF

echo ""
echo "done -- full summary at $SUMMARY_CSV"
