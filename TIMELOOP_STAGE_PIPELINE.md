# 从给定负载到 TimeLoop 完成使命:TimeLoop 阶段流水线详解

这份文档只讲整条 pipeline 里"TimeLoop 阶段"这一段——从用户给定一个负载(GEMM 规模 + 硬件规格)开始,到 TimeLoop 真实算出每一个候选 tile_k 的 Cycles/Energy、把结果打包交给 OpenCake 为止。OpenCake 拿到这些真实数字之后怎么选最优解、怎么叠加网络开销,不在这份文档范围内(见 `PIPELINE_SUMMARY.md` 第四、五节)。

## 一、整体流程

```
GemmSpec(M, N, K) + WaferSpec(mesh_rows, mesh_cols, sram_kib_per_pe, ...)
        │
        │  用户/上层调用者给定的负载和硬件规格,这是唯一的输入
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 第 0 步 —— 确定几何形状(纯计算,不涉及 TimeLoop)                    │
│                                                                 │
│   active_rows = min(mesh_rows, M)     ← 这次例子: min(32,1024)=32│
│   active_cols = min(mesh_cols, N)     ← 32                     │
│   tile_m = ceil(M / active_rows)      ← 1024/32 = 32           │
│   tile_n = ceil(N / active_cols)      ← 32                     │
│                                                                 │
│   这四个数不是搜索出来的,是 mesh 尺寸和问题规模的算术结果——           │
│   每个 PE 固定负责哪一块输出,在这一步就定了,TimeLoop 不参与。        │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 第 1 步 —— feasible_tile_k_candidates()                        │
│                                                                 │
│   遍历 K 的所有约数,保留同时满足:                                  │
│     (a) 能整除 K(每个 phase 工作量相同)                           │
│     (b) 单缓冲 SRAM 预算: c_bytes+a_bytes+b_bytes ≤               │
│         每PE SRAM - fifo_router_bytes_per_pe                    │
│   的 tile_k,从大到小排序。A/B 不做双缓冲(真实 Cerebras CSL 的任务   │
│   同步语义,不预取下一 phase),右边的预算也不是原始 SRAM 容量,        │
│   要先扣掉 FIFO/router 现在真实占用的字节数(见第 2 步)。            │
│                                                                 │
│   这次例子(K=1024, 48KiB SRAM/PE): [64, 32, 16, 8, 4, 2, 1]      │
│   —— 7 个候选,这是唯一需要真实交给 TimeLoop 评估的自由变量。         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼  对上面 7 个候选逐一重复下面这一整套(各自独立目录,互不覆盖)
┌───────────────────────────────────────────────────────────────┐
│ 第 2 步 —— generate_timeloop_inputs(gemm, wafer, tile_k=候选值)   │
│                                                                 │
│   在 outputs/.../tile_k_<候选值>/timeloop/ 下真实写出:            │
│     problem.yaml                 GEMM 问题形状(M×N×K, dense)     │
│     architecture.yaml            DRAM → PE[0..1023] → pe_scratchpad │
│                                   (SRAM) + fifo_buffer(SRAM,新增) │
│                                   + router_buffer(SRAM,新增)      │
│                                   + mac(intmac)                  │
│     architecture_constraints.yaml pe_scratchpad 保留 A/B/C;       │
│                                   fifo_buffer/router_buffer 对    │
│                                   A/B/C 都设 bypass(它们不装      │
│                                   GEMM 数据,只为了拿真实面积)      │
│     mapping.yaml                 完整指定的单点映射,把这个候选的     │
│                                   tile_k 写死在 pe_scratchpad 的  │
│                                   temporal 因子里,fifo_buffer/    │
│                                   router_buffer 同样 bypass       │
│     primitive_components.yaml    告诉 accelergy 自定义类支持哪些 action │
│     accelergy_config.yaml        accelergy 的插件/组件库配置       │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 第 3 步 —— run_timeloop_model():真实 Docker 容器                 │
│                                                                 │
│   docker run --rm                                              │
│     -v <candidate_dir>:/work                                   │
│     -v accelergy_config.yaml:/root/.config/accelergy/...:ro    │
│     -w /work                                                   │
│     timeloopaccelergy/accelergy-timeloop-infrastructure:latest │
│     timeloop-model architecture.yaml problem.yaml mapping.yaml │
│                                                                 │
│   容器内部先跑 accelergy(用真实 45nm 工艺表生成 ERT/ART 能耗/面积表)│
│   再跑 timeloop-model 对这一个映射做确定性求值(不搜索,只算这一点)。 │
│   Docker 失败/没产出 stats.txt 或 ART.yaml → 直接抛 RuntimeError, │
│   没有本地兜底。                                                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 第 4 步 —— parse_mapping_contract(mode=MODEL_MODE)               │
│                                                                 │
│   从真实 resolved.stats.txt 的 "Summary Stats" 段解析:            │
│     local_compute_cycles  ← 这个候选 tile_k 下,PE 内部真实 Cycles │
│     local_energy_uj       ← 这个候选 tile_k 下,PE 内部真实能耗(µJ)│
│                                                                 │
│   从真实 resolved.ART.yaml 解析(FIFO/router-in-PE 改动新增):     │
│     mac_area_um2 / pe_scratchpad_area_um2 /                    │
│     fifo_area_um2 / router_area_um2  ← 单实例真实面积(µm²)       │
│     ——四项缺任何一项(如 CACTI 处理不了某个很小的组件)都直接         │
│       RuntimeError,不会用 0 或估计值悄悄补上。                    │
│                                                                 │
│   tiling 字段(active_rows/cols、tile_m/n/k)直接读回 resolved.map.yaml│
│   ——这个文件就是第 2 步真实喂给 Docker 的那份 mapping.yaml 原样记录 │
│   (timeloop-model 不会像 timeloop-mapper 那样重新吐出一份 .map.yaml)│
└───────────────────────────────────────────────────────────────┘
        │
        ▼
GemmMappingContract(active_rows, active_cols, tile_m, tile_n, tile_k,
                     local_compute_cycles, local_energy_uj, ...)
        │
        │  ★ TimeLoop 阶段到此为止,使命完成:
        │    为这一个候选 tile_k 交出了真实、可验证、来自 Docker 的
        │    PE 内部 Cycles 和 Energy。
        ▼
交给 OpenCake(simulate_gemm)—— 这一部分见 PIPELINE_SUMMARY.md 第四节,
不在本文档范围
```

## 二、逐候选重复第 2~4 步,凑出完整的"真实代价表"

第 2~4 步不是只做一次——`search.search_tile_k()` 会对第 1 步列出的每一个候选(这次是 7 个)都完整跑一遍第 2~4 步,每个候选各自独立起一个 Docker 容器、各自产出一份真实的 `resolved.stats.txt`。TimeLoop 在整条 pipeline 里的"使命",严格来说不是"选出一个最优 tile_k",而是**对每一个候选,分别给出一个真实、可信、来自真实工艺表的 PE 内部代价数字**——选哪个候选最优,是 OpenCake 拿到这张真实代价表之后才做的事,TimeLoop 自己并不知道网络开销,也不参与最终选择。

这次(K=1024, 48KiB SRAM/PE)7 个候选,TimeLoop 真实给出的代价表(数据来自 `outputs/fifo_router_in_pe_check/candidates.json`):

| tile_k | k_phases | local_compute_cycles(TimeLoop真实) | local_energy_uj(TimeLoop真实) |
|---|---|---|---|
| 64 | 16 | 1,048,576 | 26,005.4 |
| 32 | 32 | 1,048,576 | 26,005.4 |
| 16 | 64 | 1,048,576 | 26,005.4 |
| 8 | 128 | 1,048,576 | 26,005.4 |
| 4 | 256 | 1,048,576 | 26,005.4 |
| 2 | 512 | 1,048,576 | 26,005.4 |
| 1 | 1024 | 1,048,576 | 26,005.4 |

7 个数字看起来完全一样不是偷懒或者出错——已经验证过(见 `PIPELINE_SUMMARY.md` 第五节):每个 PE 总 MAC 次数只取决于问题规模,和 K 怎么切成几个 phase 数学上无关,真正的差异(集中在 C 的读改写次数上)被四舍五入到了小数点后看不见的位置。**这恰恰说明 TimeLoop 单独看这张表是分不出候选优劣的**——7 个候选在它眼里几乎一样好,真正拉开差距的是 OpenCake 那边算出来的网络通信开销(`sim_time_cycles` 从 1,053,650 到 1,071,622 不等,虽然这次差距的相对幅度不算大,最优解恰好和贪心取最大 tile_k 重合,见 `PIPELINE_SUMMARY.md` 第五节的完整分析)——这也是这次架构调整"决策权交给 OpenCake"的真实数据依据。

## 三、这一路上踩过的坑(对"TimeLoop 完成使命"这件事本身有意义)

- **`mapping.yaml` 曾经把 M/N 循环挂在 `mac`(计算单元)层级上**——真实 Docker 报错 `ERROR: target storage level not found: mac`。TimeLoop 的映射 schema 只认存储层级(DRAM、pe_scratchpad)当 target,不认计算单元。改成让 `pe_scratchpad` 一次性带满 `M/N/K` 三个因子,照抄真实 `timeloop-mapper` 产出过的 `resolved.map.yaml` 结构,才通过。
- **`timeloop-model` 不会重新吐出 `.map.yaml`**——它只做单点求值,真实容器输出目录里只有 `timeloop-model.map.txt`(纯文本,不是 yaml)。修法是直接把第 2 步真实喂给 Docker 的 `mapping.yaml` 原样记成 `resolved.map.yaml`,反正这个模式下 tile_k 本来就是 OpenCake 自己决定、真实交给 Docker 评估的,不是 TimeLoop 搜出来的。

这两个坑都是第一次真实跑 Docker 才暴露出来的(`run_timeloop_model` 这条路径此前只写了代码,没有真实跑过),修完之后 7 个候选全部真实跑通,重新用真实的 `candidates.json`/`resolved.stats.txt` 核对过。
