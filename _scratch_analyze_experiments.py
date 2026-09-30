#!/usr/bin/env python3
"""Real-data analysis over every completed trial in experiment 1 (GEMM size
sweep @ fixed area=800M) and experiment 2 (area sweep @ fixed GEMM=1024).
Reads the real trials.csv files already on disk -- no synthetic numbers,
no re-running anything. Reports, per (experiment, branch, size/area) group:
best real sim_time_cycles and best real total_energy_uj among SUCCESSFUL
real trials only, plus success/attempt counts so partial groups are
visible as partial, not silently treated as complete.
"""
import csv
import glob
import os

ROOT = os.path.expanduser("~/mnt/wafer-scale-cerebras-simpy-map-hardware") if os.path.isdir(
    os.path.expanduser("~/mnt/wafer-scale-cerebras-simpy-map-hardware")) else "."

groups = []  # (experiment, branch, size_or_area, filepath)

# Experiment 1: GEMM size sweep @ area=800M fixed
for branch in ["compute_sweep", "memory_sweep", "network_sweep"]:
    for path in sorted(glob.glob(f"{ROOT}/outputs/experiment1_gemm_size/{branch}/gemm_*/trials.csv")):
        size = int(path.split("/")[-2].split("gemm_")[1])
        groups.append(("exp1_gemm_sweep@area800M", branch, size, path))
    # reused gemm=1024 point (shared with exp2's area=800M point)
    reused = f"{ROOT}/outputs/area_sweep_{branch.split('_')[0]}_any_split_800M/trials.csv"
    if os.path.exists(reused):
        groups.append(("exp1_gemm_sweep@area800M", branch, 1024, reused))

# Experiment 2: area sweep @ gemm=1024 fixed
for branch in ["compute_sweep", "memory_sweep", "network_sweep"]:
    for path in sorted(glob.glob(f"{ROOT}/outputs/experiment2_area_sweep/{branch}/area_*/trials.csv")):
        area = int(path.split("/")[-2].split("area_")[1])
        groups.append(("exp2_area_sweep@gemm1024", branch, area, path))
    reused = f"{ROOT}/outputs/area_sweep_{branch.split('_')[0]}_any_split_800M/trials.csv"
    if os.path.exists(reused):
        groups.append(("exp2_area_sweep@gemm1024", branch, 800_000_000, reused))


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


results = []
for exp, branch, key, path in groups:
    rows = load_rows(path)
    attempted = len(rows)
    succeeded = [r for r in rows if r.get("success") == "True"]
    n_ok = len(succeeded)
    best_cycles = None
    best_energy = None
    if succeeded:
        best_cycles = min(succeeded, key=lambda r: float(r["sim_time_cycles"]))
        best_energy = min(succeeded, key=lambda r: float(r["total_energy_uj"]))
    results.append({
        "exp": exp, "branch": branch, "key": key, "path": path,
        "attempted": attempted, "n_ok": n_ok,
        "best_cycles": best_cycles, "best_energy": best_energy,
    })

# --- Print per-group table ---
for exp in ["exp1_gemm_sweep@area800M", "exp2_area_sweep@gemm1024"]:
    print(f"\n{'='*100}\n{exp}\n{'='*100}")
    for branch in ["compute_sweep", "memory_sweep", "network_sweep"]:
        print(f"\n--- {branch} ---")
        rows = sorted([r for r in results if r["exp"] == exp and r["branch"] == branch], key=lambda r: r["key"])
        for r in rows:
            key_label = f"gemm={r['key']}" if "gemm_sweep" in exp else f"area={r['key']:,}"
            if r["n_ok"] == 0:
                print(f"  {key_label:>18}: {r['n_ok']}/{r['attempted']} real trials succeeded so far -- NO successful result yet")
                continue
            bc, be = r["best_cycles"], r["best_energy"]
            print(f"  {key_label:>18}: {r['n_ok']}/{r['attempted']} succeeded | "
                  f"BEST cycles={float(bc['sim_time_cycles']):>10.0f} @ c={float(bc['compute_frac']):.3f} "
                  f"m={float(bc['memory_frac']):.3f} n={float(bc['network_frac']):.3f} (mesh {bc['mesh']}) | "
                  f"BEST energy={float(be['total_energy_uj']):>10.2f}uJ @ c={float(be['compute_frac']):.3f} "
                  f"m={float(be['memory_frac']):.3f} n={float(be['network_frac']):.3f} (mesh {be['mesh']})")

# --- Global best across everything (real, successful only) ---
all_succeeded = []
for r in results:
    path = r["path"]
    rows = load_rows(path)
    for row in rows:
        if row.get("success") == "True":
            row["_exp"] = r["exp"]
            row["_branch"] = r["branch"]
            row["_key"] = r["key"]
            all_succeeded.append(row)

print(f"\n{'='*100}\nGLOBAL BEST ACROSS ALL REAL SUCCESSFUL TRIALS ({len(all_succeeded)} total)\n{'='*100}")
if all_succeeded:
    best_cycles_global = min(all_succeeded, key=lambda r: float(r["sim_time_cycles"]))
    best_energy_global = min(all_succeeded, key=lambda r: float(r["total_energy_uj"]))
    for label, r in [("LOWEST sim_time_cycles", best_cycles_global), ("LOWEST total_energy_uj", best_energy_global)]:
        key_label = f"gemm={r['_key']}" if "gemm_sweep" in r["_exp"] else f"area={r['_key']:,}"
        print(f"\n{label}:")
        print(f"  experiment={r['_exp']}  branch={r['_branch']}  {key_label}")
        print(f"  compute_frac={float(r['compute_frac']):.4f}  memory_frac={float(r['memory_frac']):.4f}  network_frac={float(r['network_frac']):.4f}")
        print(f"  sim_time_cycles={float(r['sim_time_cycles']):.0f}  total_energy_uj={float(r['total_energy_uj']):.2f}  mesh={r['mesh']}  "
              f"sram_kib_per_pe={r['sram_kib_per_pe']}  fifo_capacity_packets={r['fifo_capacity_packets']}")

print(f"\n{'='*100}\nATTEMPT/SUCCESS COUNTS PER GROUP (completeness check)\n{'='*100}")
for exp in ["exp1_gemm_sweep@area800M", "exp2_area_sweep@gemm1024"]:
    for branch in ["compute_sweep", "memory_sweep", "network_sweep"]:
        rows = sorted([r for r in results if r["exp"] == exp and r["branch"] == branch], key=lambda r: r["key"])
        total_attempted = sum(r["attempted"] for r in rows)
        total_ok = sum(r["n_ok"] for r in rows)
        print(f"  {exp} / {branch}: {total_ok}/{total_attempted} real trials succeeded across {len(rows)} size/area points")
