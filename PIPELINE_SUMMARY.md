# TimeLoop + OpenCake 晶圆级 GEMM 仿真流程总结

以 `outputs/fifo_router_in_pe_check`(1024×1024×1024 GEMM,32×32 PE mesh,48KB SRAM/PE)为例,完整走通并逐项核对过真实 Docker 产出的数据。这是 FIFO/router 建到 PE 内部(用 SRAM 空间)这次改动之后第一次端到端真实跑通;取代了更早的 `outputs/gemm1024_v11` 示例。那次示例的数据现在已经作废,不要再引用它里面的具体数字——它是在两处后来都被发现并修正的问题之上算出来的:(1) PE 内部的 cycles/MAC 速率当时用的是一个假设的常数 `macs_per_cycle_per_pe=32`,后来发现和真实架构(1 cycle/MAC)不符并改掉了;(2) A/B 当时是双缓冲(奇偶 phase 轮换两个槽位),后来照真实 Cerebras CSL 的任务同步语义改成了单缓冲。这份文档下面统一用修正之后的真实数据重新走一遍。

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
feasible_tile_k_candidates()   —— 列出所有满足单缓冲 SRAM 约束(已扣除 FIFO/router 占用)、能整除 K 的 tile_k 候选
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

给定 `tile_m/tile_n`(由 mesh 尺寸和问题规模的比值决定,是硬件几何,不是搜索空间的一部分),遍历 K 的所有能整除的因子,保留满足**单缓冲** SRAM 预算的那些:`c_bytes + a_bytes + b_bytes ≤ 每 PE SRAM 容量 - fifo_router_bytes_per_pe`。等号右边不再是原始的 48KiB(49,152 字节),而是先扣掉 FIFO/router 现在真实占用的部分(`8×8×256 + 5×4 = 16,404` 字节,见下一节 `architecture.yaml` 里新增的 `fifo_buffer`/`router_buffer` 组件),剩下 32,748 字节才是 GEMM 数据真正能用的预算。这次例子(K=1024,48KiB SRAM/PE)算出来仍然是 7 个候选:`tile_k ∈ {64, 32, 16, 8, 4, 2, 1}`——和更早的双缓冲版本数字巧合地一样,但推导逻辑已经完全不同(单缓冲、且预算里多扣了一块 FIFO/router 占用),不要把这次的候选集合和旧版的候选集合当成同一套逻辑的产物。

### 单点输入生成(`generate_timeloop_inputs(tile_k=...)`)

对每个候选,生成一套独立的 TimeLoop/Accelergy 输入(存在各自的 `outputs/.../tile_k_<值>/timeloop/` 子目录,互不覆盖):

- `problem.yaml` —— GEMM 问题形状(M×N×K,dense)
- `architecture.yaml` —— TimeLoop v0.3 原生 `subtree`/`local` 架构描述:DRAM → 32×32 复制的 PE → 每个 PE 内的 `pe_scratchpad`(SRAM 类,GEMM 数据)+ `fifo_buffer`(SRAM 类,新增)+ `router_buffer`(SRAM 类,新增)+ `mac`(intmac 类)。`fifo_buffer`/`router_buffer` 是这次 FIFO/router-in-PE 改动新增的两个组件——对应真实 Cerebras WSE 把 fabric router 和它的 FIFO 队列都做在每个 PE tile 内部这件事,让 Accelergy(CactiSRAM)给它们各自算一个真实(对 router 而言是近似)的面积数字,不再是 0 或者拍脑袋的系数;它们按字节粒度声明(`block-size:1, word-bits:8, width:8`),不同于 `pe_scratchpad` 的 GEMM 元素粒度,而且在 `architecture_constraints.yaml`/`mapping.yaml` 里都设成 `bypass:[A,B,C]`——它们不装 GEMM 数据,存在的唯一目的是拿到真实面积。
- `mapping.yaml` —— **这一步是关键**:一份完整指定、不留搜索空间的映射,把这个候选的 tile_k 直接写死在 `pe_scratchpad` 的 temporal 因子里(`M=32 N=32 K=<候选值>`),连同 DRAM 层的 spatial(mesh 展开)和 temporal(K phase 数)因子,以及 `fifo_buffer`/`router_buffer` 各自的 bypass 声明。这份映射的结构是照抄真实 `timeloop-mapper` 成功产出过的 `resolved.map.yaml`,不是猜的格式——中间踩过一个坑:一开始曾经把 M/N 循环单独挂在 `mac`(计算单元)这个层级上,真实 Docker 报错 `target storage level not found: mac`(TimeLoop 的映射 schema 只认存储层级,不认计算单元),后来改成让 `pe_scratchpad` 一次性带满 `M/N/K` 三个因子,和真实产出的结构完全一致才通过。
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
- **单缓冲、任务级同步**(不是双缓冲):A、B 各自只有一块缓冲空间,不是奇偶 phase 轮换的两个 slot。这是照真实 Cerebras CSL 的 GEMM collective benchmark 改的(https://sdk.cerebras.net/csl/code-examples/benchmark-gemm-collectives)——它的计算任务在每个 phase 开始前是"阻塞"的,等这一 phase 的广播完成回调才解除阻塞,算完之后再重新阻塞、等下一 phase 的广播,不存在"下一 phase 提前把数据搬进另一个槽位"这种预取。所以第 `phase+1` 的广播必须等第 `phase` 的计算真正读完这块缓冲才能开始写入(`a_buffer_ready`/`b_buffer_ready` 在计算结束时才被释放),通信和计算并不重叠。这也是为什么 TimeLoop 侧的 SRAM 预算不再需要"容量减半"这个安全系数——单缓冲下 `pe_scratchpad` 只需要装得下一份 A/B/C,不需要多留一份给下一 phase 提前用。真正需要从 48KiB 里扣掉的是 FIFO/router 现在真实占用的字节数(`fifo_router_bytes_per_pe`,见上一节),而不是给双缓冲预留的安全余量。
- **能耗模型**(`estimate_energy`):PE 内部(MAC 计算 + 本地 SRAM 读写)的能耗**直接用 TimeLoop/Accelergy 真实算出来的 `local_energy_uj`**,不再用项目自己拍脑袋的系数;FIFO 入队、路由器转发、导线传输、静态功耗四项依然 OpenCake 自己核算(TimeLoop 的 `spatial` 架构维度根本不知道 PE 之间的距离、跳数、排队——它只知道"有 N 个并行副本"),系数现在锚定到真实文献(见第五节)而不是拍脑袋的数字。**注意能耗和面积走的是两条不同的路**:FIFO/router 的**面积**现在是真实的、来自 Accelergy 的 `fifo_buffer`/`router_buffer` 组件(见 `GemmMappingContract.pe_internal_area_um2`);但 FIFO/router 的**能耗**依然是 OpenCake 自己按 `network.py` 统计出的真实字节数/跳数乘系数算的——因为 TimeLoop 从来不知道 OpenCake 自己仿真出的这些流量(`fifo_enqueued_bytes`/`router_byte_hops`),这一点即使把 fifo/router 声明成真实 SRAM 组件也没法改变。

## 五、这次(`fifo_router_in_pe_check`)全部候选的真实对比

问题规模:1024×1024×1024 GEMM,32×32 PE mesh(1024 个 PE 全部激活),48KiB SRAM/PE。7 个候选全部真实跑通 `timeloop-model`(读的是 Mac 上真实产出的 `candidates.json`,不是重算/估计的):

| tile_k | K phase 数 | SRAM 利用率 | 全系统耗时(cycles) | 平均 PE 利用率 | 总能耗(µJ) |
|---|---|---|---|---|---|
| **64** | **16** | **75.04%** | **1,053,650** | **99.52%** | **29,879.58** |
| 32 | 32 | 58.37% | 1,054,106 | 99.48% | 29,880.52 |
| 16 | 64 | 50.04% | 1,055,018 | 99.39% | 29,882.39 |
| 8 | 128 | 45.87% | 1,056,842 | 99.22% | 29,886.12 |
| 4 | 256 | 43.79% | 1,060,490 | 98.88% | 29,893.59 |
| 2 | 512 | 42.75% | 1,067,786 | 98.20% | 29,908.53 |
| 1 | 1024 | 42.23% | 1,071,622 | 97.85% | 29,916.39 |

**这次最优解就是 `tile_k=64`——贪心取最大能装下的那个候选,而且是单调的**:`sim_time_cycles` 随 tile_k 变小单调递增,PE 利用率单调递减,中间没有像更早的、现在已经作废的 `gemm1024_v11` 示例(见第一节说明)那样出现一个非单调的内部甜点(那次是 `tile_k=16` 优于 `tile_k=64`)。这不是这次改动引入的新现象,而是把 PE 内部 cycles 从假设的 `macs_per_cycle_per_pe=32` 改成真实的 1 cycle/MAC 之后必然的结果:单 PE 计算量从原来的"33,357 cycles 量级"变成了"1,053,650 cycles 量级",而 SUMMA 广播的首阶段填充延迟只有几千 cycles 量级——计算彻底压过通信之后,phase 数越多,只是越多次地重复缴纳这笔固定的填充延迟,不再有任何效应能反过来把总时间往下拉,单调性也就是必然的。**这不代表 OpenCake 遍历全部候选这件事没有意义**:它依然是把 7 个候选都真实跑一遍 `timeloop-model`、真实叠加网络开销之后才"发现"是单调的,而不是预先假设贪心策略成立——只是这次真实数据下,贪心和全系统最优恰好重合;换一组 SRAM 更紧张、或者网络延迟占比更高的参数,完全可能重新出现非单调的情形,不能把"这次贪心也对"当成"贪心总是对"的证据。

四个额外验证过的真实现象(不是巧合,是模型内在逻辑的必然结果):

1. **`local_compute_cycles`(1,048,576)和 `local_energy_uj`(26,005.4)完全不随 tile_k 变化**——因为每个 PE 总 MAC 次数(`tile_m×tile_n×K`)只取决于问题规模,和 K 怎么切无关;这次 7 个候选的这两个数字在 `candidates.json` 里逐位精确相同,不是四舍五入凑出来的巧合。
2. **`tile_k=1` 的网络包跳数(2,031,616)恰好是其它候选(1,015,808)的 2 倍**——因为这一档每 phase 的载荷(128 字节)不足一个网络包容量(256 字节),浪费了一半带宽,真实反映在包数上。
3. **SRAM 利用率这次(75.04%,tile_k=64)不能直接和更早文档里的百分比比较**——分子分母的定义这几版一直在变:`gemm1024_v11`(双缓冲)是"GEMM 数据 ×2 / 48KiB";`cycles_fix_check`(单缓冲、FIFO 未占 SRAM)是"GEMM 数据(单份)/ 48KiB",tile_k=64 时是 41.67%;这次(单缓冲、FIFO/router 真实占 SRAM)是`(GEMM 数据 + 16,404 字节 fifo/router)/ 48KiB`。三个数字看着都在同一个"利用率"名下,实际衡量的是三件不同的事,只能在同一版代码内部横向比较不同 tile_k,不能跨版本直接比大小。
4. 最优候选(`tile_k=64`)的最终能耗五项,逐项都能用真实产出的物理量手算精确对上(见 `OPENCAKE_CODE_WALKTHROUGH.md` 第四节),没有黑盒。

最优候选(`tile_k=64`)完整结果:

| 指标 | 数值 |
|---|---|
| 每 PE tile | 32×32,K 切 16 个 phase,每 phase 64 个 K |
| SRAM 占用 | 36,884 / 49,152 bytes(75.04%,含 16,404 字节 fifo/router 预留) |
| 仿真总耗时 | 1,053,650 cycles |
| 平均 PE 利用率 | 99.52% |
| 总 MAC 操作数 | 1,073,741,824(=1024³) |
| 网络 packet 跳数 | 1,015,808 |
| 峰值链路利用率 | 0.44% |
| PE 内部面积(1 PE,真实 Accelergy) | 172,414.45 µm²(mac 6,628 + pe_scratchpad 106,030 + fifo_buffer 59,674 + router_buffer 82.45) |
| 总能耗 | 29,879.58 µJ |
| ├─ PE 内部能耗(TimeLoop 真实数字) | 26,005.40 µJ(87.0%) |
| ├─ FIFO 入队能耗 | 988.18 µJ |
| ├─ 路由器能耗 | 546.10 µJ |
| ├─ 导线能耗 | 182.03 µJ |
| └─ 静态功耗 | 2,157.88 µJ |

值得一提两个真实的对比:

- **这次真实的 PE 内部能耗(26,005.40 µJ)比刚修完 cycles bug、还没做 FIFO/router-in-PE 改动那次(`cycles_fix_check`,27,440.43 µJ)低了约 5.2%**——这是预期之内的副作用,不是回归:`pe_scratchpad` 现在要给 `fifo_buffer`/`router_buffer` 让出一部分 SRAM 深度(`pe_scratchpad_depth_words` 从原来的 12,288 降到 8,187),CactiSRAM 对更小的 depth 算出更小的面积和更低的访问能耗,`pe_scratchpad_area_um2` 相应从 170,108 µm² 降到 106,030 µm²。
- **FIFO/router 的能耗系数这次换成了有真实文献依据的值**(`fifo=1.9pJ/byte`、`router=2.1pJ/byte`、`wire=0.7pJ/byte`,来自 DAC 2012 一颗真实流片测量的 45nm mesh NoC;`static_power_mw_per_pe=2.0mW`,来自 DSENT 验证过的 45nm router 模型的约 20%),取代了项目早期完全拍脑袋的 `0.005/0.02/0.01/0.005`——FIFO+路由器+导线+静态这四项加起来,从老系数下的约 15.80 µJ(`cycles_fix_check` 里对应四项之和)涨到了新系数下的 3,874.18 µJ,占总能耗的比例从不到 0.06% 涨到了约 13%,不再是可以忽略的零头。

## 六、产出文件位置

Mac 上:`outputs/fifo_router_in_pe_check/`
- `tile_k_64/`、`tile_k_32/`、…、`tile_k_1/` —— 每个候选各自独立的 `timeloop/` 子目录(输入 YAML、真实的 `resolved.map.yaml`/`resolved.stats.txt`/`resolved.ART.yaml`、accelergy 日志、ERT/ART 能耗/面积表等),互不覆盖,可以逐一核对
- `candidates.json` —— 全部候选的对比表(tile_k / K phase 数 / TimeLoop 真实 Cycles/Energy / 全系统耗时 / 总能耗),本节表格的数据来源
- `mapping_contract.json` —— 最优候选的 TimeLoop → OpenCake 交接契约(含新的 `fifo_area_um2`/`router_area_um2` 字段)
- `result.json` —— 最优候选的 OpenCake 完整仿真结果(即上面"完整结果"表的数据来源)

## 七、结论

链路已经完整打通并逐项用真实 Docker 数据核对过:每一个候选 tile_k 的 Cycles/Energy/Area 都来自真实的 `timeloop-model` 单点求值(不是搜索,也不是本地估算),`wafersim` 自己的 OpenCake 阶段把这些真实数字叠加上只有它自己看得到的网络通信开销,选出全系统真正最优的方案。任一候选的 Docker/TimeLoop/Accelergy 失败都会直接报错,没有静默降级到本地近似值的路径。旧的 `timeloop-mapper` 单次搜索路径保留作对比(`--mode mapper`),但默认路径已经切换成这套"OpenCake 遍历真实方案选最优"的新架构。这次(FIFO/router-in-PE)的改动之后,面积核算里 PE 内部四项(mac/pe_scratchpad/fifo_buffer/router_buffer)也全部换成了真实 Accelergy 数字,项目里唯一还留着的 OpenCake 自己拍的系数只剩下 link(PE 间导线)的面积——因为链路不属于任何单个 PE 的 architecture.yaml 子树,Accelergy 没有地方给它一个"真实"数字。
