# OpenCake 阶段代码详解:simulator.py / network.py / energy.py / area.py

这四个文件是 `wafersim` 的 **OpenCake 阶段**——拿到 TimeLoop 决定/评估的映射方案(`GemmMappingContract`:每个 PE 负责的 tile_m×tile_n 输出块、tile_k、k_phases,TimeLoop 真实算出的 `local_compute_cycles`/`local_energy_uj`,以及自 FIFO/router-in-PE 改动之后新增的 `mac_area_um2`/`pe_scratchpad_area_um2`/`fifo_area_um2`/`router_area_um2` 四项真实面积)之后,在 32×32 PE mesh 上做一次**真正的周期级(cycle-level)离散事件仿真**,算出总耗时、网络负载、各项能耗、各项面积。四者的关系:

```
simulator.py  (调度逻辑,谁在第几个 cycle 开始/结束做什么)
     │ 调用
     ▼
network.py    (每一次数据搬运具体经过哪些链路、排了多久队、几点到)
     │ 产出 network summary
     ├──────────────────────────────┐
     ▼                              ▼
energy.py                       area.py
(PE 内部能耗直接用 TimeLoop      (PE 内部面积直接用 TimeLoop/Accelergy
 真实数字,网络/静态能耗用         真实数字——mac+pe_scratchpad+fifo_buffer+
 network.py 统计出的物理量        router_buffer 四项;互连面积现在只剩
 按 pJ 系数换算)                  link 这一项还是 OpenCake 自己的系数)
```

下面按文件逐个讲,并在最后用 `fifo_router_in_pe_check` 里真实的最优候选(`tile_k=64, k_phases=16`,来自 7 个候选里 OpenCake 真实选出的全系统最优解——这次恰好也是贪心会选的那个,原因见 `PIPELINE_SUMMARY.md` 第五节)的**真实产出数字**把每一条公式都手算验证一遍——所有数字都跟 `result.json` 严丝合缝对上了,可以直接放进汇报里作为"公式与实现一致"的证据。

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

**状态字典**(现在是三个,不是两个——A、B 各自独立追踪,不共用一个 buffer 时间戳):

```python
compute_ready  = {(row, col): 0.0 ...}   # 每个 PE 的计算单元下一次空闲的时间
a_buffer_ready = {(row, col): 0.0 ...}   # 每个 PE 的 A 缓冲(唯一一份,不是两个 slot)何时空闲
b_buffer_ready = {(row, col): 0.0 ...}   # 每个 PE 的 B 缓冲(唯一一份)何时空闲
```

**这里不是双缓冲**(项目更早的版本是双缓冲,`slot = phase % 2` 轮换两个槽位;现在已经改掉了)。改成单缓冲、任务级同步是照真实 Cerebras CSL 的 GEMM collective benchmark 来的(https://sdk.cerebras.net/csl/code-examples/benchmark-gemm-collectives):它的计算任务在每个 phase 开始前处于"阻塞"状态,等这一 phase 各方向的广播完成回调才解除阻塞开始算,算完之后任务再重新阻塞、等下一 phase 的广播——不存在"下一 phase 的数据提前搬进另一个槽"这种预取重叠。代码里对应的就是:`a_buffer_ready[(row,col)]`/`b_buffer_ready[(row,col)]` 只有在这个 PE 这一 phase 的计算**真正结束**(`end`)时才会被更新,下一 phase 对同一块空间的广播必须等到这个时刻之后才能把数据送到。也因为这样,TimeLoop 侧的 SRAM 预算不再需要"双缓冲、容量减半"这道安全系数——单缓冲下 `pe_scratchpad` 只需要装得下一份 A/B/C 就够了(见 `contract.py` 的 `data_sram_bytes_per_pe`)。

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
    a_buffer_ready[(row, col)],          # A 缓冲(唯一一份)何时空闲
    b_buffer_ready[(row, col)],          # B 缓冲(唯一一份)何时空闲
)
macs = m_extent * n_extent * k_extent
compute_cycles = ceil(macs * cycles_per_mac)
end = start + compute_cycles
```

五个依赖取最大值(比双缓冲版本多一个,因为 A、B 各自的缓冲空闲时刻现在分开追踪了)——这是标准的"资源可用时间"建模:数据没到不能算,上一次算的还没算完不能算,A/B 各自的缓冲还占着不能覆盖写。`cycles_per_mac` 不是一个假设的吞吐常数,而是 `contract.local_compute_cycles / (tile_m*tile_n*gemm.k)`——TimeLoop 真实测出的这个映射下每个 PE 全部时间循环的 Cycles,除以同一份映射下每个 PE 真实要做的总 MAC 数,反推出的真实 cycles/MAC 速率(项目早期这里曾经是一个假设的 `wafer.macs_per_cycle_per_pe=32` 常数,和真实架构声明的"每个 PE 只有一个标量 MAC、1 cycle 一次"不符,后来发现并改掉了——真实速率其实是 1.0,不是 32,这也是为什么改完之后 `sim_time_cycles` 比老版本涨了约 28 倍,不是 bug,是修 bug)。算完之后更新 `compute_ready`/`a_buffer_ready`/`b_buffer_ready` 为 `end`,并把这次的 MAC 数、SRAM 读写字节数(A/B 各算 2 次——写入+读出,C 算 2 次——读改写)累加进全局统计。

**循环结束后**:

```python
sim_time = max(max(compute_ready.values(), default=0.0), network.latest_cycle)
network_summary = network.summary(sim_time)
energy = estimate_energy(
    wafer=wafer,
    pe_internal_energy_uj=contract.local_energy_uj,   # TimeLoop 真实数字,不再是本地系数算出来的
    network=network_summary,
    sim_time_cycles=sim_time,
    active_pes=contract.active_pes,
)
area = estimate_area(
    wafer=wafer,
    active_pes=contract.active_pes,
    pe_internal_area_um2=contract.pe_internal_area_um2,   # 同样是 TimeLoop 真实数字
)
```

`sim_time_cycles` 取"所有 PE 里最晚完工的那个"和"网络里最后一个包/FIFO 完成的时刻"两者的较大值——保证即使某条链路上还有包没走完,也不会把仿真时间算短了。这里能看到 `simulate_gemm()` 和 TimeLoop 交接的关键一步:PE 内部能耗直接用 `contract.local_energy_uj`,PE 内部面积直接用 `contract.pe_internal_area_um2`(两个都是真实 accelergy 数字),不再用 `mac_operations`/`sram_access_bytes` 乘本地系数去估。

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

## 四、`area.py` —— 面积核算

```python
@dataclass(frozen=True, slots=True)
class AreaBreakdown:
    pe_internal_area_um2: float
    interconnect_area_um2_per_pe: float
    active_pes: int
    mesh_pes: int

    @property
    def area_um2_per_pe(self) -> float:
        return self.pe_internal_area_um2 + self.interconnect_area_um2_per_pe

    @property
    def active_pe_array_area_um2(self) -> float:
        return self.area_um2_per_pe * self.active_pes

    @property
    def die_pe_array_area_um2(self) -> float:
        return self.area_um2_per_pe * self.mesh_pes
```

结构上和 `energy.py` 是同一个套路:`pe_internal_area_um2` 不是 OpenCake 自己算的,是 `estimate_area()` 调用方(`simulator.py`)直接传进来的 `contract.pe_internal_area_um2`——**FIFO/router-in-PE 这次改动之后,这个数字是 mac + pe_scratchpad + fifo_buffer + router_buffer 四项 Accelergy 真实面积之和**(见 `contract.py`),不再只是 mac + pe_scratchpad 两项。`interconnect_area_um2_per_pe` 现在**只剩 `wafer.link_area_um2_per_pe`** 这一项——FIFO/router 的面积已经真实地并进 `pe_internal_area_um2` 了,`area.py` 里原来那个 `_FIFOS_PER_PE = 8` 常量和 fifo/router 系数项已经删掉,不再重复计算。

```python
def estimate_area(wafer, *, active_pes, pe_internal_area_um2):
    interconnect_area_um2_per_pe = wafer.link_area_um2_per_pe
    return AreaBreakdown(
        pe_internal_area_um2=pe_internal_area_um2,
        interconnect_area_um2_per_pe=interconnect_area_um2_per_pe,
        active_pes=active_pes,
        mesh_pes=wafer.mesh_rows * wafer.mesh_cols,
    )
```

`link_area_um2_per_pe` 现在默认是 **1684.25**——link 是两个相邻 PE 的 router 之间的一段导线,不属于任何单个 PE 的 `architecture.yaml` 子树,Accelergy 没有地方给它一个真实数字,Cerebras 和 DSENT(NoCS 2012)两个真实来源都查过,都没有直接给出可用的每 PE 导线面积。这个默认值是本项目自己拼出来的**复合估算**,但每一步输入都可核实为真实文献值,不是拍脑袋:导线 pitch 取自真实 ITRS 2007 Interconnect Roadmap(Table INTC2a,45nm 节点对应的 2010 年,"Minimum global wiring pitch" = 135nm);总线宽度用 `router_port_bytes=4` 字节(32 bit,同一个真实 Cerebras "32-bit bi-directional interfaces" 引用)代入 ORION 2.0(TVLSI 2012)论文给出的真实导线面积公式 `Arealink = fw×(ww+sw)+sw` 的主项;单跳导线长度用真实 Cerebras 引用的 PE tile 面积 38,000 µm²(cerebras.ai 架构深度解析博客)开方近似成正方形 tile 的边长,每个 PE 分摊半跳;4 个 mesh 方向(不含 router 第 5 个本地端口)各自算一遍再求和,详细推导见 `specs.py` 里 `link_area_um2_per_pe` 字段上方的注释。这仍然是一个**推理组合出来的工程估算**,不是某一篇论文直接发表的单一数字——所以严格说它和 PE 内部四项(纯 Accelergy 真实输出)、`local_energy_uj`(纯 TimeLoop 真实输出)相比,可信度低一档,但已经不再是无引用支撑的占位 0。

两个 PE 数量并排报告,不合并成一个数:`active_pes` 是这次 GEMM 真正用到的 PE 数(`contract.active_pes`)——"这个问题实际用了多少硅片";`mesh_pes` 是物理 wafer 上全部的 PE 数(`wafer.mesh_rows * wafer.mesh_cols`)——芯片的 die 面积在流片时就固定了,不会因为某次问题没用满而变小,所以这是回答"这块芯片有多大"该用的数字。

## 五、拿 `fifo_router_in_pe_check` 最优候选(tile_k=64)的真实数字手算验证

参数:`active_rows=active_cols=32`(1024 个 PE)、`tile_m=tile_n=32`、**`tile_k=64`、`k_phases=16`**(这是 OpenCake 从 7 个真实候选里选出来的全系统最优解——这次恰好也是贪心取最大 tile_k 会选的那个,原因见 `PIPELINE_SUMMARY.md` 第五节,不是巧合是必然)、`bytes_per_element=4`、`cycles_per_mac=1.0`(真实测出来的,不是假设的 32)、`link_bandwidth=32 bytes/cycle`、`packet_payload_bytes=256`。因为 1024 正好被 32 和 64 整除,所有 PE 每阶段拿到的都是满 tile,没有边界截断,所以下面每个公式都能和 `outputs/fifo_router_in_pe_check/result.json` **精确对上**,不是估算。

**1. 计算量与计算耗时**

单个 PE 每阶段:`macs = 32×32×64 = 65,536`,`compute_cycles = ceil(65536×1.0) = 65,536`(`cycles_per_mac=1.0` 是精确的整数比,`ceil` 不引入任何多余 cycle)。

- `mac_operations = 1024 PE × 65,536 macs × 16 phases = 1,073,741,824`(= 1024³)—— **与 `result.json` 完全一致**,而且和 tile_k=16 那个候选的结果完全相同——因为每个 PE 总 MAC 数只取决于问题规模(`tile_m×tile_n×K`),和 K 怎么切成几个 phase 无关。
- 单 PE 16 阶段共 `65,536×16 = 1,048,576` busy cycles;1024 个 PE 合计 `compute_busy_cycles = 1024×1,048,576 = 1,073,741,824` —— **与 `result.json` 完全一致**。值得指出:这次 `compute_busy_cycles` 和 `mac_operations` 是**同一个数字**——因为 `cycles_per_mac` 精确等于 1.0(真实架构就是 1 cycle/MAC),不像项目早期假设 `macs_per_cycle_per_pe=32` 那会儿,`compute_busy_cycles` 只有 `mac_operations` 的 1/32。
- `sim_time_cycles = 1,053,650`,只比单 PE 的纯计算时间(1,048,576)多了 5,074 cycles——这部分是 SUMMA 广播的排队/填充延迟,占比不到 0.5%。`mean_pe_utilization = 1,073,741,824 / (1024×1,053,650) = 99.52%`,是 7 个候选里最高的一档(tile_k 越小,这个填充延迟被重复缴纳的次数越多,利用率单调下降,见 `PIPELINE_SUMMARY.md` 第五节的完整对比表)。

**2. 网络负载**

单次行/列广播 payload = `32×64×4 = 8,192` bytes,切成 `8192/256 = 32` 个包。owner 位置随 phase 轮换,但**总跳数只取决于链路长度**(见第二节 2.4 的数学证明):一条广播链总共还是要走 `32-1=31` 跳(东西两个方向加起来)。

- 单次广播的跳数 = `32 包 × 31 跳 = 992`;每阶段有 32 次行广播 + 32 次列广播 = 64 次广播 → `64×992 = 63,488`;16 个阶段 → `63,488×16 = 1,015,808` —— **与 `result.json` 的 `packet_hops` 完全一致**,而且和 tile_k=16 那个候选的结果**数值上相同**,印证了第二节里"总跳数与 tile_k 怎么切无关"的结论。
- 单次广播搬运的总字节(每一跳都要重传一次 8,192 字节)= `8,192×31 = 253,952`;每阶段 64 次广播(32 行 A + 32 列 B)→ `253,952×64 = 16,252,928`;16 阶段 → `16,252,928×16 = 260,046,848` —— **与 `result.json` 的 `link_bytes` 完全一致**。

**3. SRAM 访问(这一项会随 tile_k 变化,和上面两项不同)**

单 PE 单阶段:`a_bytes=8,192`、`b_bytes=8,192`、`c_bytes=4,096`,按代码 `2×a + 2×b + 2×c = 40,960` bytes;1024 PE × 16 阶段 → `40,960×16,384 = 671,088,640` —— **与 `result.json` 的 `sram_access_bytes` 完全一致**。K 切得越细(tile_k 越小、phase 越多),C 的读改写次数越多,这一项会相应增大(tile_k=16 那个候选算出来会是这个数字的 1.6 倍)。

**4. 面积四项(FIFO/router-in-PE 改动之后新增的手算验证)**

| 项 | 来源 | 结果(µm²/PE) | mapping_contract.json |
|---|---|---|---|
| mac | TimeLoop/Accelergy 真实 ART.yaml | 6,628.0 | 6628.0 ✓ |
| pe_scratchpad | 同上 | 106,030.0 | 106030.0 ✓ |
| fifo_buffer | 同上(新增) | 59,674.0 | 59674.0 ✓ |
| router_buffer | 同上(新增,近似) | 82.45 | 82.45 ✓ |
| **pe_internal_area_um2(四项之和)** | | **172,414.45** | **172414.45 ✓** |

`active_pe_array_area_um2 = 172,414.45 × 1,024 = 176,552,396.8` —— 与 `outputs/fifo_router_in_pe_check/result.json` 的 `area.active_pe_array_area_um2` 完全一致;这次 `active_pes == mesh_pes == 1024`(mesh 正好用满),所以 `die_pe_array_area_um2` 是同一个数字。**这组数字对应的是 `link_area_um2_per_pe` 还是占位 0.0 时跑的那次真实 Docker 结果**——当时 `interconnect_area_um2_per_pe = 0.0`,`active_pe_array_area_um2` 只有 `pe_internal_area_um2 × active_pes` 一项。现在 `link_area_um2_per_pe` 默认值已经改成 1684.25(见上一节),同样的命令重新跑一次,`interconnect_area_um2_per_pe` 会变成 1684.25,`area_um2_per_pe = 172,414.45 + 1,684.25 = 174,098.70`,`active_pe_array_area_um2` 也会相应变成 `174,098.70 × 1,024 = 178,277,068.8`——这两个新数字目前还只是按 `wafer/area.py` 公式手算出来的,没有用新配置重新跑过真实 Docker 核对,标注为推算值,不是本节其余数字那种"已核对"状态。

**5. 能耗五项(全部用上面验证过的物理量直接代公式)**

| 项 | 公式 | 结果 | result.json |
|---|---|---|---|
| PE 内部(TimeLoop 真实数字) | 直接取 `contract.local_energy_uj` | 26,005.4 µJ | 26005.4 ✓ |
| FIFO | 520,093,696 × 1.9pJ / 1e6 | 988.1780224 µJ | 988.1780224 ✓ |
| 路由器 | 260,046,848 × 2.1pJ / 1e6 | 546.0983808 µJ | 546.0983808 ✓ |
| 导线 | 260,046,848 × 0.7pJ / 1e6 | 182.0327936 µJ | 182.0327936 ✓ |
| 静态 | 2.0mW×1024×1,053,650/1.0 /1e6 | 2,157.8752 µJ | 2157.8752 ✓ |
| **总计** | | **29,879.5843968 µJ** | **29879.5843968 ✓** |

`fifo_enqueued_bytes = 2×link_bytes = 520,093,696`(发送+接收各计一次)。五项能耗、四项面积全部精确对齐,说明代码实现和这里描述的公式完全一致,`result.json` 里的每一个数字都可以从 `mapping_contract.json` 里的映射参数一步步手算复现出来,不存在黑盒或者不可解释的中间步骤——包括这次新增的"FIFO/router 面积直接来自 TimeLoop/Accelergy 真实 Docker 输出、能耗系数锚定到真实文献"这两个环节。
