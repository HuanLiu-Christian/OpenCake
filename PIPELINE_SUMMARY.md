# TimeLoop + OpenCake 晶圆级 GEMM 仿真流程总结

以 `outputs/gemm1024_v11`(1024×1024×1024 GEMM,32×32 PE mesh,48KB SRAM/PE)为例,完整走通并逐项核对过真实 Docker 产出的数据。这是这套流程"OpenCake 遍历 TimeLoop 真实方案选最优"的新架构第一次端到端跑通,取代了此前"TimeLoop 自己搜一次就定案"的做法。

## 一、这次架构调整是为了解决什么问题

最初的目标很单纯:TimeLoop 必须自己真实计算 PE 内部的 tiling(尤其是 K 维度怎么切),不能由我们自己的 Python 代码算好后喂给它一个假数字。这个要求本身从没变过。

但后来讨论"给定一个负载,怎么找到最佳的资源划分"这个更大的目标时,发现原来的做法有个结构性缺陷:`timeloop-mapper` 自己做的那次内部搜索,优化目标只有它自己能看到的 delay/energy 打分,**它根本看不到 PE 之间的网络通信开销**——TimeLoop 的 `architecture.yaml` 里 `spatial` 这个维度就是单纯的"N 个并行副本",没有跳数、没有排队、没有链路争用的概念(这也是为什么 `wafer/network.py` 要单独存在)。所以 TimeLoop 自己选出来的"最优" tile_k,只是它自己看得到的那部分最优,不是这个项目真正关心的全系统(计算+网络)最优。

调整后的分工:

- **TimeLoop 仍然是唯一的、真实的 PE 内部代价来源**——每一个候选 tile_k 的 Cycles/Energy,都来自一次真实的 Docker `timeloop-model` 单点确定性求值,不是猜的,也不是本地公式估的。这一点完全没有让步。
- **OpenCake(这个项目自己)拿到决策权**——因为 TimeLoop 能对给定负载的所有合法 tile_k 划分分别给出真实代价,OpenCake 索性把它们全部枚举出来,每个都真实跑一遍 `timeloop-model`,再叠加上只有 OpenCake 自己看得到的网络通信代价,选出真正的全系统最优解。

这就是这次要交付的两个想法:(1) OpenCake 的能耗核算直接用 TimeLoop 真实算出来的数字,不再用自己拍脑袋的系数;(2) OpenCake 遍历 TimeLoop 给出的所有候选方案,加上通信成本后选最优。

## 二、整体流程(新的默认路径:`--mode search`)

```
GemmSpec/WaferSpec (问题规模 + 硬件规格)
        │
        ▼
feasible_tile_k_candidates()   —— 列出所有满足双缓冲 SRAM 约束、能整除 K 的 tile_k 候选
        │
        ▼
┌─────────────────────────────────────────────┐
│  对每一个候选 tile_k(此例是 7 个):              │
│                                               │
│  generate_timeloop_inputs(tile_k=候选值)       │
│         │                                    │
│         ▼                                    │
│  run_timeloop_model()  —— 真实 Docker 容器     │
│    跑 timeloop-model,单点确定性求值             │
│    (accelergy 先生成能耗表,再评估这一个映射)     │
│         │                                    │
│         ▼                                    │
│  parse_mapping_contract(mode=MODEL_MODE)       │
│    —— 从真实 stats.txt 解析这个候选的           │
│       Cycles/Energy(TimeLoop 真实数字)         │
│         │                                    │
│         ▼                                    │
│  simulate_gemm()  —— OpenCake:算出这个候选的    │
│    全系统开销(PE 计算 + 跨 PE 网络通信)         │
└─────────────────────────────────────────────┘
        │
        ▼
best_by_sim_time()   —— 按全系统 sim_time_cycles(平局按总能耗)选出真正最优的那个候选
        │
        ▼
candidates.json(全部候选对比表) + mapping_contract.json / result.json(最优候选)
```

旧的 `timeloop-mapper` 单次搜索路径还在,改叫 `--mode mapper`,留作对比/调试用——它只需要跑一次 Docker,比 `search` 模式快,但选出来的 tile_k 没有考虑网络开销,仅供参考。

## 三、TimeLoop 阶段细节

### 候选枚举(`feasible_tile_k_candidates`)

给定 `tile_m/tile_n`(由 mesh 尺寸和问题规模的比值决定,是硬件几何,不是搜索空间的一部分),遍历 K 的所有能整除的因子,保留满足双缓冲 SRAM 预算(`c_bytes + 2·a_bytes + 2·b_bytes ≤ 每 PE SRAM 容量`)的那些。这次例子(K=1024,48KB SRAM/PE)得到 7 个候选:`tile_k ∈ {64, 32, 16, 8, 4, 2, 1}`。

### 单点输入生成(`generate_timeloop_inputs(tile_k=...)`)

对每个候选,生成一套独立的 TimeLoop/Accelergy 输入(存在各自的 `outputs/.../tile_k_<值>/timeloop/` 子目录,互不覆盖):

- `problem.yaml` —— GEMM 问题形状(M×N×K,dense)
- `architecture.yaml` —— TimeLoop v0.3 原生 `subtree`/`local` 架构描述:DRAM → 32×32 复制的 PE → 每个 PE 内的 `pe_scratchpad`(SRAM 类)+ `mac`(intmac 类)
- `mapping.yaml` —— **这一步是关键**:一份完整指定、不留搜索空间的映射,把这个候选的 tile_k 直接写死在 `pe_scratchpad` 的 temporal 因子里(`M=32 N=32 K=<候选值>`),连同 DRAM 层的 spatial(mesh 展开)和 temporal(K phase 数)因子。这份映射的结构是照抄真实 `timeloop-mapper` 成功产出过的 `resolved.map.yaml`,不是猜的格式——中间踩过一个坑:一开始曾经把 M/N 循环单独挂在 `mac`(计算单元)这个层级上,真实 Docker 报错 `target storage level not found: mac`(TimeLoop 的映射 schema 只认存储层级,不认计算单元),后来改成让 `pe_scratchpad` 一次性带满 `M/N/K` 三个因子,和真实产出的结构完全一致才通过。
- `primitive_components.yaml` + `accelergy_config.yaml` —— 告诉 accelergy 我们自定义的 `SRAM`/`DRAM`/`intmac` 类各自支持哪些 action(解决"ERT 表全空"的关键)

### 真实执行(`run_timeloop_model`)

```
docker run --rm \
  -v <candidate_dir>:/work \
  -v <candidate_dir>/accelergy_config.yaml:/root/.config/accelergy/accelergy_config.yaml:ro \
  -w /work \
  timeloopaccelergy/accelergy-timeloop-infrastructure:latest \
  timeloop-model architecture.yaml problem.yaml mapping.yaml
```

`timeloop-model` 和 `timeloop-mapper` 不一样:它不做搜索,只对给定的这一份 `mapping.yaml` 做真实的确定性求值,跑得比 mapper 快很多(mapper 要跑几百个候选点才收敛,model 只算一个点)。这里还踩过第二个坑:`timeloop-model` 跟 `timeloop-mapper` 不同,**不会**重新吐出一份 `.map.yaml`(真实容器输出目录里只有 `timeloop-model.map.txt`,纯文本格式,不是 yaml)。修法是:既然这个模式下 tile_k 本来就是 OpenCake 自己决定、真实喂给 Docker 的输入,不是 TimeLoop 搜出来的,那就直接把这份真实喂给 Docker 的 `mapping.yaml` 原样记录成 `resolved.map.yaml`——TimeLoop 在这个模式下真正的贡献是 `stats.txt` 里的 Cycles/Energy,不是 tiling 本身。

失败(Docker 不可用、二进制报错、没产出 `.stats.txt`)直接抛异常,没有任何本地兜底——7 个候选里只要有一个 Docker 真实失败,整个流程就会在那里停下来,不会用假数字填坑接着走。

### 结果解析(`parse_mapping_contract(mode=MODEL_MODE)`)

跟 mapper 模式共用同一套解析逻辑:从 `resolved.map.yaml` 正则读出 `active_rows/active_cols/tile_m/tile_n/tile_k`,从 `resolved.stats.txt` 的 "Summary Stats" 段解析出这个候选真实的 `local_compute_cycles`/`local_energy_uj`。跟 mapper 模式不同的是,`mapper_mode` 字段会记成 `docker-timeloop-model`,标明这份数据的来源方式(单点求值 vs 搜索),但两种模式下数字本身都是 100% 真实 Docker 产出。

## 四、OpenCake 阶段细节(`simulate_gemm`)

拿到每个候选各自的 `GemmMappingContract` 后,分别在 32×32 PE mesh 上做一次周期级仿真:

- **调度**:SUMMA 风格,按 K phase 循环,但用的是**真实 SUMMA 的"轮换 owner"广播**,不是简化的"固定从边缘注入"——第 `p` 个 phase,A 的行块常驻在 `PE(row, p mod active_cols)`(不是永远列 0),向本行剩下的 PE 做**东西双向**广播;B 的列块常驻在 `PE(p mod active_rows, col)`,向本列剩下的 PE 做**南北双向**广播。owner 会随 phase 轮换,所以一整个运行下来每个 PE 都必须能双向收发四个方向,这是 v9/v10 那次改动补上的。
- **网络模型**(`PacketNetwork`):四个方向(东/西/南/北)各自独立的发送/接收 FIFO(参考了 Cerebras CSL 的 fabric queue 设计,同一个 PE 上不同方向的数据流用不同队列,互不阻塞),每条有向链路(东西南北各自独立,不是一条链路两个方向共用)都有独立的排队/争用建模,逐跳(hop-by-hop)显式仿真,不是简单的带宽/延迟公式。
- **双缓冲**:A/B 各有 2 个缓冲 slot(奇偶 phase 轮换),让通信和计算重叠,这也是 TimeLoop 侧要求"SRAM 容量减半"这个安全约束的来源。
- **能耗模型**(`estimate_energy`):PE 内部(MAC 计算 + 本地 SRAM 读写)的能耗**直接用 TimeLoop/Accelergy 真实算出来的 `local_energy_uj`**,不再用项目自己拍脑袋的系数;FIFO 入队、路由器转发、导线传输、静态功耗四项 OpenCake 自己核算(TimeLoop 根本不知道 PE 之间的距离和布线)。

## 五、这次(v11)全部候选的真实对比

问题规模:1024×1024×1024 GEMM,32×32 PE mesh(1024 个 PE 全部激活),48KB SRAM/PE。7 个候选全部真实跑通 `timeloop-model`:

| tile_k | K phase 数 | SRAM 利用率 | 全系统耗时(cycles) | 平均 PE 利用率 | 总能耗(µJ) |
|---|---|---|---|---|---|
| 64 | 16 | 75.0% | 33,357 | 98.23% | 2375.90 |
| 32 | 32 | 41.7% | 33,213 | 98.66% | 2375.90 |
| **16** | **64** | **25.0%** | **33,141** | **98.87%** | **2375.90** |
| 8 | 128 | 16.7% | 33,900 | 96.66% | 2375.91 |
| 4 | 256 | 12.5% | 38,108 | 85.99% | 2375.93 |
| 2 | 512 | 10.4% | 46,716 | 70.14% | 2375.97 |
| 1 | 1024 | 9.4% | 50,244 | 65.22% | 2375.99 |

**最优解是 `tile_k=16`**,不是"贪心取最大能装下的 tile_k"(64)会选的那个——虽然差距不大(33,141 对 33,357,约 0.65%),但方向已经证明:全系统开销的最优点不一定和 TimeLoop 单独看到的最优点重合,因为 K 切得太大(64)双缓冲填充的首阶段延迟更长,切得太小(1、2)又会让通信频率上升、PE 利用率明显掉下去(65%~70%),中间存在一个真正的甜点。

三个额外验证过的真实现象(不是巧合,是模型内在逻辑的必然结果):

1. **`local_compute_cycles`(1,048,576)和 `local_energy_uj`(2365.33)几乎不随 tile_k 变化**——因为每个 PE 总 MAC 次数(`tile_m×tile_n×K`)和 A/B 总读写字节数,天然只取决于问题规模,和 K 怎么切无关;真正能耗差异集中在 C 的读改写次数上,但这部分占比在真实 `stats.txt` 里不到 0.01%,被四舍五入掉了。
2. **`tile_k=1` 的网络包跳数(2,031,616)恰好是其它候选(1,015,808)的 2 倍**——因为这一档每 phase 的载荷(128 字节)不足一个网络包容量(256 字节),浪费了一半带宽,真实反映在包数上。
3. 最优候选(`tile_k=16`)的最终能耗六项,逐项都能用真实产出的物理量手算精确对上(见 `OPENCAKE_CODE_WALKTHROUGH.md` 第四节),没有黑盒。

最优候选(`tile_k=16`)完整结果:

| 指标 | 数值 |
|---|---|
| 每 PE tile | 32×32,K 切 64 个 phase,每 phase 16 个 K |
| SRAM 占用 | 12,288 / 49,152 bytes(25.0%) |
| 仿真总耗时 | 33,141 cycles |
| 平均 PE 利用率 | 98.87% |
| 总 MAC 操作数 | 1,073,741,824(=1024³) |
| 网络 packet 跳数 | 1,015,808 |
| 峰值链路利用率 | 13.47% |
| 总能耗 | 2375.90 µJ |
| ├─ PE 内部能耗(TimeLoop 真实数字) | 2365.33 µJ(99.6%) |
| ├─ FIFO 入队能耗 | 2.60 µJ |
| ├─ 路由器能耗 | 5.20 µJ |
| ├─ 导线能耗 | 2.60 µJ |
| └─ 静态功耗 | 0.17 µJ |

值得一提:这次真实的 PE 内部能耗(2365.33 µJ)比这个项目之前用自己拍脑袋系数(`mac_energy_pj=0.30`、`sram_energy_pj_per_byte=0.01`)估出来的数字(约 328.8 µJ)高了约 7 倍——说明真实 45nm 工艺下的计算能耗比项目早期的估计系数要高不少,这正是"改用 TimeLoop 真实数字"这件事本身的价值所在。

## 六、产出文件位置

Mac 上:`outputs/gemm1024_v11/`
- `tile_k_64/`、`tile_k_32/`、…、`tile_k_1/` —— 每个候选各自独立的 `timeloop/` 子目录(输入 YAML、真实的 `resolved.map.yaml`/`resolved.stats.txt`、accelergy 日志、ERT/ART 能耗表等),互不覆盖,可以逐一核对
- `candidates.json` —— 全部候选的对比表(tile_k / K phase 数 / TimeLoop 真实 Cycles/Energy / 全系统耗时 / 总能耗)
- `mapping_contract.json` —— 最优候选的 TimeLoop → OpenCake 交接契约
- `result.json` —— 最优候选的 OpenCake 完整仿真结果(即上表数据来源)

## 七、结论

链路已经完整打通并逐项用真实 Docker 数据核对过:每一个候选 tile_k 的 Cycles/Energy 都来自真实的 `timeloop-model` 单点求值(不是搜索,也不是本地估算),`wafersim` 自己的 OpenCake 阶段把这些真实数字叠加上只有它自己看得到的网络通信开销,选出全系统真正最优的方案。任一候选的 Docker/TimeLoop 失败都会直接报错,没有静默降级到本地近似值的路径。旧的 `timeloop-mapper` 单次搜索路径保留作对比(`--mode mapper`),但默认路径已经切换成这套"OpenCake 遍历真实方案选最优"的新架构。
