#!/usr/bin/env python3
"""100M um^2 快速探测 -- network-primary any-split 分支.

跟另外两个 100M 脚本是同一个思路，换成 network-primary 分支：100% 复用
exp2_network_sweep_by_area.py 里已经真实验证过的机制
(run_for_area/run_step/run_trial/真实 CSV schema/早停规则)，只新加一个
之前从没真实跑过的 area_budget=100,000,000 um^2 面积点，不改动、不重
跑 200M/400M/800M/1.6B/3.2B 任何一个已有真实数据点，也完全不碰
all_trials.csv（只在 network_sweep/area_100000000/ 这个全新目录下写文
件）。

同样不是"resume"脚本，没有断点续跑逻辑。要注意：network-primary 分支
本身内部有 SRAM 预算 + 网络 FIFO 预算两层真实探测，是三个分支里单次
真实调用最慢的一个（3.2B 那一档目前的真实历史数据显示，它比另外两个
分支慢了不止一倍）——100M 这么小的面积预算下,两层探测的候选数量应该
会少很多，但具体还是要看真实跑出来的结果，这里不做无根据的预测。

用法（必须在你自己的 Mac 终端里跑，用真实 Docker）：
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_network_sweep_100M.py > exp2_network_sweep_100M.log 2>&1 &
  disown
  tail -f exp2_network_sweep_100M.log

注意：和现在正在跑的 3.2B compute_sweep/network_sweep 真实进程会抢同一
台机器上的真实 Docker/CPU 资源，可能互相拖慢，取舍同另外两个 100M 脚
本里的说明。
"""

from __future__ import annotations

import exp2_network_sweep_by_area as base

AREA_BUDGET_UM2 = 100_000_000


if __name__ == "__main__":
    print(
        f"100M um^2 快速探测 (network-primary any-split, GEMM "
        f"{base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} 固定) -- 全新面积点，"
        f"不影响 200M/400M/800M/1.6B/3.2B 任何一个已有真实数据点 ...",
        flush=True,
    )
    base.run_for_area(AREA_BUDGET_UM2)
    out_csv = base.EXPERIMENT_ROOT / f"area_{AREA_BUDGET_UM2}" / "trials.csv"
    print(f"\n100M um^2 (network-primary) 真实探测完成，结果在 {out_csv}")
