#!/usr/bin/env python3
"""100M um^2 快速探测 -- memory-primary any-split 分支.

跟 exp2_compute_sweep_100M.py 是同一个思路、同一份代码骨架，只是换成了
memory-primary 这个分支：100% 复用 exp2_memory_sweep_by_area.py 里已经
真实验证过的机制 (run_for_area/run_step/run_trial/真实 CSV schema/早停
规则)，只新加一个之前从没真实跑过的 area_budget=100,000,000 um^2 面积
点，不改动、不重跑 200M/400M/800M/1.6B/3.2B 任何一个已有真实数据点，也
完全不碰 all_trials.csv（只在 memory_sweep/area_100000000/ 这个全新目
录下写文件）。

同样不是"resume"脚本，没有断点续跑逻辑 -- 中途被打断的话，重新跑会把
这个 100M 目录下的 trials.csv 整个删掉重新开始。100M 量级预期比 3.2B
快很多，重跑代价不大。每个候选依然是一次真实的 `python3 -m wafersim`
子进程调用，不可行就真实报错记录，绝不编造结果。

用法（必须在你自己的 Mac 终端里跑，用真实 Docker）：
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_memory_sweep_100M.py > exp2_memory_sweep_100M.log 2>&1 &
  disown
  tail -f exp2_memory_sweep_100M.log

注意：和现在正在跑的 3.2B compute_sweep/network_sweep 真实进程会抢同一
台机器上的真实 Docker/CPU 资源，可能互相拖慢，取舍同 compute 版本脚本
里的说明。
"""

from __future__ import annotations

import exp2_memory_sweep_by_area as base

AREA_BUDGET_UM2 = 100_000_000


if __name__ == "__main__":
    print(
        f"100M um^2 快速探测 (memory-primary any-split, GEMM "
        f"{base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} 固定) -- 全新面积点，"
        f"不影响 200M/400M/800M/1.6B/3.2B 任何一个已有真实数据点 ...",
        flush=True,
    )
    base.run_for_area(AREA_BUDGET_UM2)
    out_csv = base.EXPERIMENT_ROOT / f"area_{AREA_BUDGET_UM2}" / "trials.csv"
    print(f"\n100M um^2 (memory-primary) 真实探测完成，结果在 {out_csv}")
