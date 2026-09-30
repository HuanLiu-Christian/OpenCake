#!/usr/bin/env python3
"""100M um^2 快速探测 -- compute-primary any-split 分支.

背景：3.2B 那一档面积预算跑得太慢（单次真实 Docker 调用经常要几十分钟
到几个小时），所以先在一个小得多的面积预算上做一次快速的真实 Docker
探测，看看 100M 这个量级下计算/存储/网络三者怎么分配。这是一个全新的、
之前从来没有真实跑过的面积点，不属于 AREA_BUDGETS = [200M, 400M,
1.6B, 3.2B]（也不是被复用的 800M 那个历史点），纯粹是为了快速拿到一个
参考结果而临时加的。

这个脚本不是"resume"脚本 -- 100M 这个面积点没有任何历史数据要保护，也
没有断点续跑逻辑：如果中途被打断，重新跑这个脚本会把
outputs/experiment2_area_sweep/compute_sweep/area_100000000/trials.csv
整个删掉重新开始（不像 3.2B 那几个 _resume.py 脚本那样，会从真实已有
数据的断点续跑）。100M 这个量级预期本来就应该比 3.2B 快很多（更小的
面积 -> 更小的可行网格 -> 更小的真实 TimeLoop mapper 搜索空间），所以
即使要整个重跑一次，代价也不大。

100% 复用 exp2_compute_sweep_by_area.py 里已经真实验证过的机制
(run_for_area/run_step/run_trial/真实 CSV schema/早停规则)，只是换了一
个新的 area_budget 数值 -- 不改动、不重跑 200M/400M/800M/1.6B/3.2B 任
何一个已有真实数据点，也完全不碰 all_trials.csv（只在
compute_sweep/area_100000000/ 这个全新目录下写文件）。跟其它分支的真实
Docker 调用机制完全一致：每个候选都是一次真实的 `python3 -m wafersim`
子进程调用，不可行就真实报错记录，绝不编造结果。

用法（必须在你自己的 Mac 终端里跑，用真实 Docker -- 这个脚本不能在云端
会话里跑，因为那边没有 docker，也看不到你终端里的真实进程）：
  cd experiment2_area_sweep
  source ../.venv/bin/activate
  nohup python3 exp2_compute_sweep_100M.py > exp2_compute_sweep_100M.log 2>&1 &
  disown
  tail -f exp2_compute_sweep_100M.log

注意：这个脚本会和你现在正在跑的 3.2B compute_sweep/network_sweep 真实
进程抢同一台机器上的真实 Docker/CPU 资源，可能会让两边都变慢。如果想
避免互相拖慢，可以先跑完/暂停 3.2B 那两个进程，再跑这个；如果不介意
互相慢一点，也可以直接并行跑。
"""

from __future__ import annotations

import exp2_compute_sweep_by_area as base

AREA_BUDGET_UM2 = 100_000_000


if __name__ == "__main__":
    print(
        f"100M um^2 快速探测 (compute-primary any-split, GEMM "
        f"{base.GEMM_M}x{base.GEMM_N}x{base.GEMM_K} 固定) -- 全新面积点，"
        f"不影响 200M/400M/800M/1.6B/3.2B 任何一个已有真实数据点 ...",
        flush=True,
    )
    base.run_for_area(AREA_BUDGET_UM2)
    out_csv = base.EXPERIMENT_ROOT / f"area_{AREA_BUDGET_UM2}" / "trials.csv"
    print(f"\n100M um^2 (compute-primary) 真实探测完成，结果在 {out_csv}")
