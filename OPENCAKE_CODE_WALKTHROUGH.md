# OpenCake 阶段代码详解:simulator.py / network.py / energy.py

这三个文件是 `wafersim` 的 **OpenCake 阶段**——拿到 TimeLoop 决定/评估的映射方案(`GemmMappingContract`:每个 PE 负责的 tile_m×tile_n 输出块、tile_k、k_phases,以及 TimeLoop 真实算出的 `local_compute_cycles`/`local_energy_uj`)之后,在 32×32 PE mesh 上做一次**真正的周期级(cycle-level)离散事件仿真**,算出总耗时、网络负载、各项能耗。三者的关系:

```
simulator.py  (调度逻辑,谁在第几个 cycle 开始/结束做什么)
     │ 调用
     ▼
network.py    (每一次数据搬运具体经过哪些链路、排了多久队、几点到)
     │ 产出 network summary
     ▼
energy.py     (PE 内部能耗直接用 TimeLoop 真实数字,网络/静态能耗用 network.py 统计出的物理量按 pJ 系数换算)
```

下面按文件逐个讲,并在最后用 `gemm1024_v11` 里真实的最优候选(`tile_k=16, k_phases=64`,来自 7 个候选里 OpenCake 真实选出的最优解)的**真实产出数字**把每一条公式都手算验证一遍——所有数字都跟 `result.json` 严丝合缝对上了,可以直接放进汇报里作为"公式与实现一致"的证据。

---

## 一、`simulator.py` —— 调度主逻辑

### 1.1 `WaferGemmResult`(结果容器)

```python
@dataclass(frozen=True, slots=True)
class WaferGemmResult:
    contract: GemmMappingContract
    sim_time_cycles: float
    compute_busy_cycles: float
    mac_operations: int
    sram_access_bytes: int
    network: dict[str, float | int]
    energy: EnergyBreakdown
```

装仿真结果的只读数据类。`mean_pe_utilization` 是一个派生属性,不存字段而是现算:

```python
mean_pe_utilization = compute_busy_cycles / (active_pes * sim_time_cycles)
```

分母是"1024 个 PE × 总仿真周期数"这个理论上限(如果每个 PE 从第 0 cycle 到最后一刻都在算),分子是实际发生的计算 cycle 总和。这个比值就是"PE 平均有百分之多少的时间在真正做乘加",越接近 100% 说明通信/等待对计算的拖累越小。

### 1.2 `_extent()` —— 处理边界 tile

```python
def _extent(start: int, tile: int, limit: int) -> int:
    return max(0, min(tile, limit - start))
```

如果 `active_rows * tile_m` 不能整除 `gemm.m`(比如 mesh 比矩阵稍大,或矩阵不是 tile 大小的整数倍),最后一行/列的 PE 分到的实际数据量要比 `tile_m` 小。这个函数就是算"某个 PE 这次实际能拿到多少行/列/多少 K",而不是无脑用 `tile_m`。在 v11 这个例子里 1024 正好被 32(mesh)和 16(tile_k)整除,所有 PE 每次拿到的都是满 tile,没有边界效应——这也是后面手算能跟真实输出精确对上的原因。

### 1.3 `simulate_gemm()` —— 主仿真循环

**整体思路**:这是**经典 SUMMA(Scalable Universal Matrix Multiplication Algorithm)**风格的调度,采用**轮换 owner(rotating panel ownership)**——K 维度被切成 `k_phases` 个阶段,第 `p` 个阶段:

1. A 的行块这一阶段常驻在 `PE(row, p mod active_cols)`(**不是永远固定在第 0 列**),从这个 owner 位置向本行剩下的 PE 做**东、西双向**广播;
2. B 的列块常驻在 `PE(p mod active_rows, col)`,从这个 owner 位置向本列剩下的 PE 做**南、北双向**广播;
3. 每个 PE(r,c)必须等到**这一阶段**的 A 分片和 B 分片都送达、且自己上一次计算已经做完、且要写入的缓冲区槽位空闲后,才能开始这一阶段的乘加。

owner 会随 phase 轮换(不是永远从边缘注入),意味着一整个运行下来,每个 PE 都有可能既要接收东边/南边来的数据,也要接收西边/北边来的数据——这正是本项目 SUMMA 算法的真实语义(而不是"DRAM 固定从边缘喂数据"的简化版本),对应的网络实现见下一节 `network.py`。

**状态字典**:

```python
compute_ready = {(row, col): 0.0 ...}          # 每个 PE 的计算单元下一次空闲的时间
buffer_ready  = {(row, col, slot): 0.0 ...}    # 每个 PE 的两个缓冲槽(slot 0/1)何时空闲
```

`slot = phase % 2` 就是**双缓冲(double buffering)**的实现:偶数阶段用槽 0,奇数阶段用槽 1,槽位每两个阶段轮换复用一次。这样第 `phase+1` 阶段的数据可以在第 `phase-1` 阶段的数据还在被消费时,提前搬进另一个槽——通信和计算因此可以重叠,这正是 TimeLoop 侧把 SRAM 容量声明减半(只允许它规划出单缓冲下能装得下、双缓冲下也能装得下的 tile_k)的原因。

**每个阶段内部**(核心是三段):

```python
owner_col = phase % contract.active_cols
owner_row = phase % contract.active_rows
```

先算出这一阶段 A/B 面板各自的 owner 位置。

**(a) 行广播(A)**——对每一行,取该行 PE 需要的 M 范围(`m_extent`)和本阶段 K 范围(`k_extent`),算出 payload 字节数,以 owner 位置上"该槽位上一次被清空的时间"(`buffer_ready[(row, owner_col, slot)]`)作为最早可发送时间,调用 `network.broadcast_row(row, ready_cycle, payload, owner_col=owner_col)` 得到这一行每个 PE 收到 A 的具体时刻。

**(b) 列广播(B)**——对每一列同理,调用 `network.broadcast_column(col, ready_cycle, payload, owner_row=owner_row)`。

**(c) 逐 PE 计算**——对每个 `(row, col)`:

```python
start = max(
    a_arrival_by_row[row][(row, col)],   # A 数据到达时刻
    b_arrival_by_col[col][(row, col)],   # B 数据到达时刻
    compute_ready[(row, col)],           # 这个 PE 的计算单元上次何时空闲
    buffer_ready[(row, col, slot)],      # 这次要用的缓冲槽何时空闲
)
macs = m_extent * n_extent * k_extent
compute_cycles = ceil(macs / wafer.macs_per_cycle_per_pe)
end = start + compute_cycles
```

四个依赖取最大值——这是标准的"资源可用时间"建模:数据没到不能算,上一次算的还没算完不能算,缓冲槽还占着不能覆盖写。算完之后更新 `compute_ready`/`buffer_ready` 为 `end`,并把这次的 MAC 数、SRAM 读写字节数(A/B 各算 2 次——写入+读出,C 算 2 次——读改写)累加进全局统计。

**循环结束后**:

```python
sim_time = max(max(compute_ready.values()), network.latest_cycle)
network_summary = network.summary(sim_time)
energy = estimate_energy(
    wafer=wafer,
    pe_internal_energy_uj=contract.local_energy_uj,   # TimeLoop 真实数字,不再是本地系数算出来的
    network=network_summary,
    sim_time_cycles=sim_time,
    active_pes=contract.active_pes,
)
```

`sim_time_cycles` 取"所有 PE 里最晚完工的那个"和"网络里最后一个包/FIFO 完成的时刻"两者的较大值——保证即使某条链路上还有包没走完,也不会把仿真时间算短了。这里能看到 `simulate_gemm()` 和 TimeLoop 交接的关键一步:PE 内部能耗直接用 `contract.local_energy_uj`(真实 accelergy 数字),不再用 `mac_operations`/`sram_access_bytes` 乘本地系数。

---

## 二、`network.py` —— 周期级网络仿真

这是三个文件里最"底层"的一个,把广播这件事拆到**每一跳、每一个包**的粒度去建模,而不是用一个简单的"字节数/带宽+延迟"公式。这一版实现了两件事:(1)四个方向(东/西/南/北)各自独立的 FIFO,参考 Cerebras CSL 的 fabric queue 设计;(2)真实 SUMMA 语义要求的、owner 位置可以轮换的双向广播。

### 2.1 拓扑与状态

```python
directions = ("east", "west", "south", "north")
self.send_fifos = {(r, c, d): FifoState() for r in rows for c in cols for d in directions}
self.receive_fifos = {(r, c, d): FifoState() for r in rows for c in cols for d in directions}

self.links = {}
for r in rows:
    for c in cols:
        if c + 1 < cols:
            links[((r, c), (r, c + 1))] = LinkState()   # 东向
            links[((r, c + 1), (r, c))] = LinkState()   # 西向
        if r + 1 < rows:
            links[((r, c), (r + 1, c))] = LinkState()   # 南向
            links[((r + 1, c), (r, c))] = LinkState()   # 北向
```

每个 PE 有 **8 个独立 FIFO**(东/西/南/北各一个发送 + 一个接收),每条物理链路也拆成**两条独立的有向链路**(东向和西向、南向和北向分别用各自的 `LinkState`,不共享一个链路槽位)——同一个 PE 上同时往东继电 A 和往北继电 B,不会因为共用一个队列/一条链路而互相阻塞。容量由 `wafer.fifo_capacity_packets` 限制(此例是 8)。这次 32×32 mesh 下 `links` 总数是 3,968(= 2 × 1,984,即每条物理边两个方向各算一条)。

### 2.2 `_admit_fifo()` —— FIFO 容量的排队建模

```python
def _admit_fifo(self, fifo, ready_cycle, size_bytes) -> float:
    self._discard_completed(fifo, ready_cycle)          # 先清掉已经处理完的包
    admission = ready_cycle
    if len(fifo.completions) >= capacity:                 # 满了
        admission = min(fifo.completions)                 # 等最早的那个腾出位置
        self._discard_completed(fifo, admission)
    fifo.wait_cycles += admission - ready_cycle            # 记录排队等待时间
    ...
    return admission
```

这不是一个"无限缓冲、只算带宽延迟"的简化模型——FIFO 真的有容量上限,满了就必须真实地等待,等待时间会被累计进 `send_fifo_wait_cycles`/`receive_fifo_wait_cycles`,直接影响后面算出来的 `sim_time`。

### 2.3 `_send_hop()` —— 一跳的完整时序

```python
send_fifo = self.send_fifos[(*source, direction)]           # 按方向选对应的队列
send_admitted = self._admit_fifo(send_fifo, ready_cycle, size_bytes)     # ① 排队等发送槽位
link = (source, destination)                                              # 该方向对应的有向链路
serialization = ceil(size_bytes / link_bandwidth_bytes_per_cycle)         # ② 这个包过链路要多少 cycle
service = router_latency_cycles + serialization                           # ③ 路由延迟 + 传输时间
link_start = max(send_admitted, state.next_free_cycle)                    # ④ 链路本身可能还在忙上一个包
link_end = link_start + service
receive_fifo = self.receive_fifos[(*destination, direction)]              # 按方向选对应的接收队列
receive_admitted = self._admit_fifo(receive_fifo, link_end, size_bytes)   # ⑤ 到了对端还要排队进接收 FIFO
hop_end = receive_admitted + receive_dequeue_latency_cycles               # ⑥ 接收端出队延迟
```

一跳里有 **两次排队(发送 FIFO、接收 FIFO,按方向各自独立)+ 一次链路争用(同一条有向链路上包与包之间也要排队,靠 `state.next_free_cycle` 维护)+ 固定的路由/出队延迟**。这四类延迟分别累计进 `queue_wait_cycles`(链路)、`wait_cycles`(FIFO),最后汇总到 `summary()` 里,能清楚看出瓶颈到底出在哪一环。

### 2.4 `broadcast_row`/`broadcast_column` —— 从 owner 位置双向中继

```python
def broadcast_row(self, row, ready_cycle, size_bytes, owner_col):
    arrivals = {(row, owner_col): ready_cycle}
    for size in self._packet_sizes(size_bytes):
        ready = ready_cycle
        for col in range(owner_col, self.cols - 1):        # 向东继电,直到最后一列
            ready = self._send_hop(ready, size, (row, col), (row, col + 1), "east")
            arrivals[(row, col + 1)] = max(arrivals.get((row, col + 1), 0.0), ready)
        ready = ready_cycle
        for col in range(owner_col, 0, -1):                 # 向西继电,直到第一列
            ready = self._send_hop(ready, size, (row, col), (row, col - 1), "west")
            arrivals[(row, col - 1)] = max(arrivals.get((row, col - 1), 0.0), ready)
    return arrivals
```

和旧版(固定从第 0 列向东单向继电)的核心区别:广播起点是**这一阶段的 owner_col**,不是永远的列 0;而且从 owner 出发**同时**往东、往西两个方向继电,各自独立走一遍上面 `_send_hop` 的排队逻辑(east 用 east 队列,west 用 west 队列,互不阻塞)。`broadcast_column` 是对称的南/北版本。

**一个数学上的巧合(已验证,不是意外)**:对一条长度为 N 的链路,从任意位置 `p` 出发双向继电覆盖全部 N 个节点,总跳数永远是 `p(向西) + (N-1-p)(向东) = N-1`,和 `p` 在哪里无关。这意味着不管 owner 固定在边缘(旧版)还是随 phase 轮换(新版),`packet_hops`/`link_bytes` 这两个聚合指标的公式和旧版**完全一样**——变的只是"谁在等谁"这种排队细节(体现在 `send_fifo_wait_cycles`/`link_queue_wait_cycles` 上),聚合吞吐量指标不变。这也是为什么下面第四节的手算公式,形式上和这份文档更早版本里的公式几乎一样,只是代入了新的 tile_k/k_phases 数字。

### 2.5 `summary()` —— 对外汇总统计

汇总出 `packet_hops`(总跳数)、`link_bytes`(所有链路上传输的总字节,`router_byte_hops`/`wire_byte_hops` 是同一个数字,只是分别喂给下面能耗模型里路由器和导线两项不同的系数)、`fifo_enqueued_bytes`、各种等待 cycle、`max_link_utilization`(最繁忙那条链路的忙碌时间占总仿真时间的比例)。

---

## 三、`energy.py` —— 能耗核算

```python
@dataclass(frozen=True, slots=True)
class EnergyBreakdown:
    pe_internal_uj: float   # 原来是 mac_uj + sram_uj 两项,现在合并成一项、直接来自 TimeLoop 真实数字
    fifo_uj: float
    router_uj: float
    wire_uj: float
    static_uj: float

    @property
    def total_uj(self) -> float:
        return sum(asdict(self).values())
```

五项能耗,只有 `pe_internal_uj` 这一项来源变了——不再是"某个仿真阶段统计出来的物理量 × 项目自己拍脑袋的 pJ 系数",而是**直接等于 `GemmMappingContract.local_energy_uj`**,也就是 accelergy 用真实 45nm 工艺表算出来的、这个候选 tile_k 下 PE 内部(计算 + 本地 SRAM 读写)的真实能耗。TimeLoop 的 `architecture.yaml` 把 PE 阵列声明成一个 `spatial` 子树(`PE[0..active_pes-1]`),TimeLoop 自己的 Energy 核算(不同于 Cycles——Cycles 是关键路径,不随副本数变化)本身就已经是对全部 1024 个 PE 实例的加总,所以这里**不需要**再乘一次 `active_pes`。

其余四项 OpenCake 自己核算,公式没变,依然完全透明可核对:

```python
fifo_uj   = fifo_enqueued_bytes  * fifo_energy_pj_per_byte    / 1e6
router_uj = router_byte_hops     * router_energy_pj_per_byte_hop / 1e6
wire_uj   = wire_byte_hops       * wire_energy_pj_per_byte_hop   / 1e6
static_uj = static_power_mw_per_pe * active_pes * sim_time_cycles / clock_ghz / 1e6
```

为什么这四项还是 OpenCake 自己算,而不是也让 TimeLoop 算:TimeLoop 的 `spatial` 维度只表示"N 个并行副本",完全没有跳数/排队/链路争用的概念,天然算不出"PE 之间通信要花多少能量"——这正是 `network.py` 要单独存在的原因。`static_uj` 是漏电类静态功耗,公式是"功率 × 时间":`mW × cycles/GHz = mW × ns = pJ`,再除以 1e6 换算成 µJ,这一项也是 TimeLoop 的单副本模型算不出来的(它不知道整个 wafer 有多少激活的 PE 在同时漏电)。

---

## 四、拿 v11 最优候选(tile_k=16)的真实数字手算验证

参数:`active_rows=active_cols=32`(1024 个 PE)、`tile_m=tile_n=32`、**`tile_k=16`、`k_phases=64`**(这是 OpenCake 从 7 个真实候选里选出来的最优解,不是贪心取最大的 tile_k=64)、`bytes_per_element=4`、`macs_per_cycle_per_pe=32`、`link_bandwidth=32 bytes/cycle`、`packet_payload_bytes=256`。因为 1024 正好被 32 和 16 整除,所有 PE 每阶段拿到的都是满 tile,没有边界截断,所以下面每个公式都能和 `outputs/gemm1024_v11/result.json` **精确对上**,不是估算。

**1. 计算量与计算耗时**

单个 PE 每阶段:`macs = 32×32×16 = 16,384`,`compute_cycles = ceil(16384/32) = 512`。

- `mac_operations = 1024 PE × 16,384 macs × 64 phases = 1,073,741,824`(= 1024³)—— **与 `result.json` 完全一致**,而且和 tile_k=64 那次(v8/v10)的结果完全相同——因为每个 PE 总 MAC 数只取决于问题规模(`tile_m×tile_n×K`),和 K 怎么切成几个 phase 无关。
- 单 PE 64 阶段共 `512×64 = 32,768` busy cycles;1024 个 PE 合计 `compute_busy_cycles = 1024×32,768 = 33,554,432` —— **与 `result.json` 完全一致**,同样和 tile_k 无关(`ceil` 在这个规模下总能整除,不产生额外 cycle)。
- `sim_time_cycles = 33,141`,只比单 PE 的纯计算时间(32,768)多了 373 cycles——比 v8 那次(tile_k=64,多出 814 cycles)反而更少,因为这次每个 phase 的 payload 更小(2,048 字节而不是 8,192),首阶段的"数据还没到达"填充延迟更短。`mean_pe_utilization = 33,554,432 / (1024×33,141) = 98.87%`,是 7 个候选里最高的一档。

**2. 网络负载**

单次行/列广播 payload = `32×16×4 = 2,048` bytes,切成 `2048/256 = 8` 个包。owner 位置随 phase 轮换,但**总跳数只取决于链路长度**(见第二节 2.4 的数学证明):一条广播链总共还是要走 `32-1=31` 跳(东西两个方向加起来)。

- 单次广播的跳数 = `8 包 × 31 跳 = 248`;每阶段有 32 次行广播 + 32 次列广播 = 64 次广播 → `64×248 = 15,872`;64 个阶段 → `15,872×64 = 1,015,808` —— **与 `result.json` 的 `packet_hops` 完全一致**,而且和 v8/v10(tile_k=64)的结果**数值上相同**,印证了第二节里"总跳数与 tile_k 怎么切无关"的结论。
- 单次广播搬运的总字节(每一跳都要重传一次 2,048 字节)= `2,048×31 = 63,488`;每阶段 64 次广播(32 行 A + 32 列 B)→ `63,488×64 = 4,063,232`;64 阶段 → `4,063,232×64 = 260,046,848` —— **与 `result.json` 的 `link_bytes` 完全一致**,同样和 tile_k=64 那次数值相同。

**3. SRAM 访问(这一项会随 tile_k 变化,和上面两项不同)**

单 PE 单阶段:`a_bytes=2,048`、`b_bytes=2,048`、`c_bytes=4,096`,按代码 `2×a + 2×b + 2×c = 12,288` bytes;1024 PE × 64 阶段 → `12,288×65,536 = 1,073,741,824` —— **与 `result.json` 的 `sram_access_bytes` 完全一致**,是 v8/v10(tile_k=64,671,088,640)的 1.6 倍——因为 K 切得更细,C 的读改写次数更多,这一项确实会随 tile_k 变化(这也是为什么 TimeLoop 真实的 `local_energy_uj` 里 SRAM 分量理论上应该有极小的差异,只是幅度太小被四舍五入抹平了,见 `PIPELINE_SUMMARY.md` 第五节)。

**4. 能耗五项(全部用上面验证过的物理量直接代公式)**

| 项 | 公式 | 结果 | result.json |
|---|---|---|---|
| PE 内部(TimeLoop 真实数字) | 直接取 `contract.local_energy_uj` | 2365.33 µJ | 2365.33 ✓ |
| FIFO | 520,093,696 × 0.005pJ / 1e6 | 2.60046848 µJ | 2.60046848 ✓ |
| 路由器 | 260,046,848 × 0.02pJ / 1e6 | 5.20093696 µJ | 5.20093696 ✓ |
| 导线 | 260,046,848 × 0.01pJ / 1e6 | 2.60046848 µJ | 2.60046848 ✓ |
| 静态 | 0.005mW×1024×33,141/1.0 /1e6 | 0.16968192 µJ | 0.16968192 ✓ |
| **总计** | | **2375.90155584 µJ** | **2375.90155584 ✓** |

`fifo_enqueued_bytes = 2×link_bytes = 520,093,696`(发送+接收各计一次)。五项全部精确对齐,说明代码实现和这里描述的公式完全一致,`result.json` 里的每一个数字都可以从 `mapping_contract.json` 里的映射参数一步步手算复现出来,不存在黑盒或者不可解释的中间步骤——包括现在这个"PE 内部能耗直接来自 TimeLoop 真实 Docker 输出"的新环节。
