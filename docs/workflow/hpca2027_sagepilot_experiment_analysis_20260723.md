# SagePilot 多 Workflow 异构 GPU 实验分析（2026-07-23）

> 本文面向 `paper/hpca2027-sagepilot` 的论文写作者，冻结并解释
> `output/serve` 中截至 2026-07-23 14:09 UTC 已完成的多 workflow 实验。
> 本文不是 runtime 实现说明，也不把尚未实现或尚未测量的完整 SagePilot 机制写成实验结论。

## 阅读前的实验背景

### 实验要回答什么问题

这里研究的不是单个模型的一次推理，而是多个大型语言模型（LLM）workflow 共享异构 GPU 池时
的端到端执行。
一个用户请求对应一个 **session**；session 按有向无环图（DAG）依次或并行执行多个节点。
LLM 节点需要先获得合适的本地模型副本，CPU function 节点则直接执行。一次 session 的完成时间
因此不仅包含模型推理，还包含排队、依赖等待、模型加载或复用、输出背压以及最终汇合。

当前运行时把所有 workflow 放进一个共享 fleet：

- workflow 共享同一个调度器、A100/V100 池和模型副本池；
- 完全相同的模型部署身份不含 workflow 名，因此可以跨 session、跨 workflow 复用；
- 一个模型副本独占一张 GPU，但同一副本最多可连续批处理3个并发请求；
- GPU 容量不足或未来需要别的模型时，运行时必须在保留、加载、预加载和驱逐之间选择。

这使实验的核心问题变成：

> 调度器能否利用 workflow 的未来节点和冻结资源估计，让模型在正确的时间、正确的 GPU 上
> 出现，并在仍有近期复用价值时保留，从而同时降低 session 平均完成时间与无效模型驻留？

### 两类 workload 与当前 DAG

QMSum 是面向会议记录的查询式摘要 workload，用来制造 fan-out/fan-in 和跨 session 流水线：

```text
split [CPU]
  └─> chunk_0 ... chunk_5 [Qwen3-4B，并行]
        └─> merge [Qwen3-8B]
              └─> result
```

每个 QMSum session 有6个并行 chunk 推理和1个 merge 推理，共产生7次逻辑模型获取。

MBPP sanitized 是 Python 编程题 workload，用来制造多模型串接、模型切换和后续复用：

```text
coder [Qwen3-14B] -> tester [CPU]
tester -> diagnoser [Qwen3-1.7B]
tester + diagnoser -> reviewer [Qwen3-4B]
tester + diagnoser + reviewer -> repair [Qwen3-8B]
tester + repair -> final_tester [CPU] -> result
```

每个 MBPP session 有4个 LLM 节点，共产生4次逻辑模型获取；`tester` 和 `final_tester` 是 CPU
函数，不占用模型副本。当前每个 trial 固定运行60个 QMSum session 和60个 MBPP session，
因此逻辑模型获取总数为：

```text
60 × 7 + 60 × 4 = 660
```

这里的“模型获取”不等于“物理加载”。如果所需模型已经驻留，获取会成为 reuse；否则才会
触发 load。于是每个 trial 均满足 `loads + reuses = 660`。这正是后文用加载、复用和驱逐次数
解释模型生命周期决策的基础。

### 为什么这两个 workload 能体现跨 workflow 模型复用

当前 DAG 只包含4种共享模型部署：

| 模型 | 节点角色 | 生命周期压力 |
|---|---|---|
| Qwen3-1.7B | MBPP diagnoser | 小模型，位于代码 workflow 中段 |
| Qwen3-4B | QMSum 六个 chunk；MBPP reviewer | 同时被两类 workflow 使用，复用最密集 |
| Qwen3-8B | QMSum merge；MBPP repair | 同时位于 fan-in 后和代码链后段 |
| Qwen3-14B | MBPP coder | 受显存约束，只能放在 A100 |

Qwen3-4B 和 Qwen3-8B 的部署身份分别跨越 QMSum 与 MBPP，`qmsum1...4` 或 `mbpp1...4`
之间也不会复制模型身份。因此“共享模型池”不是把每个 workflow 的私有模型统计相加，而是
让不同 workflow 的节点直接复用同一个已驻留副本。

这两类数据集在本文中是系统 workload fixture：它们提供不同 DAG 形状、输入长度和模型序列。
本文评估的是系统执行行为，不从这些实验外推摘要质量或代码正确率。

### 模型生命周期与 workflow 可见性

一个模型副本在 trace 中经历如下状态：

```text
不存在 -> | LOADING | -> | IDLE <-> BUSY | -> | EVICTING | -> 不存在
             loading          resident          evicting
             \_____________ lifecycle ______________/
```

- **ready** 表示节点的所有依赖已经满足，现在必须获取模型才能执行；
- **near-ready** 表示节点尚未 ready，但调度器从 DAG 和前驱进度已能看到它即将需要某个模型；
- **prefetch** 是在显式模型获取请求到来前，由 near-ready 信号提前发起加载；
- **reuse-distance eviction** 根据各 workflow 距离下一次使用某副本还有多远，选择更适合驱逐的模型；
- **acquire** 是节点向共享池申请模型执行权，可能直接复用，也可能等待加载或其他请求释放 lease。

SagePilot 当前模型级机制的因果链是：

```text
未来 DAG 节点可见
  -> 估计前驱完成时间与模型加载时间
  -> 决定何时预加载、保留或驱逐
  -> 改变 ready 时模型是否可用及 GPU 被无效占用多久
  -> 影响端到端平均耗时、驻留时间和空泡率
```

后文的机制分析会区分“动作确实被触发”和“动作已经产生收益”。例如一次预加载最终被使用，
并不代表它在 consumer ready 前完成，也不代表它隐藏了模型加载。

### 实验名称和缩写速查

| 名称 | 含义 |
|---|---|
| `2w` | 实际提交 session 的2个 workflow：`qmsum1` 和 `mbpp1`；不是2张 GPU或2个 session |
| `8w` | 8个 workflow 命名空间：`qmsum1...4` 和 `mbpp1...4`，每个15个 session |
| `A1V2` / `a1v2` | 1张 A100 加2张 V100，共3张 GPU；表示硬件配比，不是算法版本 |
| `A1V1` | 1张 A100 加1张 V100，共2张 GPU；legacy 结果目录没有硬件后缀 |
| `hetero` | heterogeneous，A100/V100 异构共享池 |
| `burst` | 到达间隔为0，120个 session 在运行开始时近乎同时提交 |
| `Poisson` | 开放到达；相邻 session 的到达间隔服从指数分布 |
| `ρ` | offered load 相对实测 burst 饱和速率的比例 |
| `r050` | `ρ≈0.50`；聚合到达率约0.025 session/s，不是 cache hit ratio |
| `DAG` | directed acyclic graph，描述节点及其依赖关系 |
| `fan-out/fan-in` | 一个节点分叉到多个并行节点，再等待多个前驱结果汇合 |
| `backpressure` | 下游有界队列已满，导致上游输出无法立即入队的阻塞 |
| `EMA` | exponential moving average，History 对本轮已观测耗时的在线平滑估计 |
| `ETA` | estimated time of availability，用于估计任务或模型何时可用 |
| `SRPT` | shortest remaining processing time，优先剩余关键路径较短的请求 |
| `LRU` | least recently used，按最近使用时间选择驱逐对象 |
| `vLLM` | 本实验使用的本地 LLM serving engine，负责连续批处理与 token 生成 |
| `OOM` | out of memory，模型加载或 batch 执行超过 GPU 显存 |
| `GNN` | graph neural network；完整论文计划使用的离线资源预测器，当前实验未接入 |
| `KV cache` | 自回归生成保存的 key/value 中间状态；当前实验未管理其生命周期 |
| `BubblePath` | 论文规划的完整有序准备路径机制；当前只验证其模型级先导机制 |
| `RQ` | research question；RQ2、RQ4、RQ5 的编号沿用当前论文草稿 |
| `pp` | percentage points，两个百分比直接相减得到的百分点 |
| `GPU-s` | GPU·second；跨 GPU/副本累计的占用时间，不是 wall-clock 秒 |
| `trial` | 一个策略在一个固定 workload、到达轨迹和硬件配置下的完整系统运行 |

`8w` 与 `2w` 都是120个 session、60个 QMSum 加60个 MBPP、660次模型获取。8w 不是把
工作量扩大4倍，而是把同等工作量分散到更多 workflow 命名空间，检验共享模型池、全局调度
和跨 workflow 复用能否保持有效。

目录名 `output/serve/2w_poisson/hetero_r050_cache_a1v2` 可依次读作：
“2个活跃 workflow、Poisson 到达、约50%负载、Cache 策略、1×A100+2×V100”。

### “Cache”在本文中具体指什么

本文的 **Cache 策略**不是 KV cache，也不是简单的模型权重文件缓存。四种策略都读取同一份
只读 resource-contract cache。该表按模型、GPU 类型、batch size、输入长度和输出长度索引，
记录加载耗时、运行耗时和峰值显存等离线 profile。四种策略都用显存条目进行可行性与 batch
准入检查；策略差异在于如何使用时间估计做软决策：

- Cache 从 trial 开始就使用冻结的模型加载与运行点估计，计算 ETA、预加载时机和未来复用距离；
- History 不用冻结时间估计做这些软决策，而是在当前 trial 内用实际观察逐步建立 EMA；
- FIFO 和本项目的 Kairos-style 基线不做 near-ready 预加载，也不做预测式 reuse-distance 驱逐。

三组正式 Cache trial 的660次 placement trace 均将冻结估计来源标为
`empirical_gpu_profile`。运行时不会用本轮结果更新这些值，也没有在线调用 GNN。因此本文能
讨论“冻结 profile 信息对生命周期决策的价值”，不能把 Cache 的收益直接写成 GNN 预测收益。

## 结论摘要

当前最适合论文使用的证据由三组互补实验构成：

- **当前 DAG、2 workflows、A1V2 burst：用于 Cache 对 History 的干净机制比较。**
  Cache 的平均端到端耗时比 History 低 `2.946%`，makespan 低 `2.895%`，
  模型驻留低 `1.795%`，空闲驻留低 `2.491%`，模型生命周期 GPU 开销低 `3.206%`。
  两者的 active GPU-s、acquire 等待、fan-in 等待、预加载次数和调度动作数接近，
  但 Cache 少一次模型迁移/重载。trace 中该差异使后继模型提前 `47.2s` 就绪，
  首个 session 相应提前 `46.5s` 完成。
- **当前 DAG、8 workflows、A1V2 burst：用于多 workflow 扩展和 Cache 对 Kairos 的比较。**
  Cache 相对 Kairos 将平均端到端耗时降低 `3.145%`、makespan 降低 `6.811%`、
  模型驻留降低 `28.939%`、空闲驻留降低 `45.252%`、空泡率降低 `12.313` 个百分点。
  Cache 在该组的总体平均耗时也是四种策略中最低。
- **当前 DAG、2 workflows、A1V2、ρ≈0.50 Poisson：用于开放到达和生命周期动作分析。**
  Cache 的平均端到端耗时为 `117.487s`，低于 History 的 `118.514s`、FIFO 的
  `120.724s` 和 Kairos 的 `128.862s`。Cache 相对 History 的完整模型生命周期 GPU
  开销仅高 `0.116%`，可表述为以近似相同的生命周期开销获得 `0.867%` 的平均延迟改善。
  该组触发了 23 次 Cache 预加载和 24 次 near-ready 驱逐，但没有一次预加载在请求到达前
  完成，因此它证明机制被执行，不证明当前预加载已经充分隐藏加载延迟。

综合来看，当前数据支持以下有限但清晰的结论：

> Cache 使用运行开始前即可获得的冻结资源估计，在在线 History 尚未充分建立时改善模型
> 驱逐与加载顺序。该优势在当前 2w A1V2 burst 中表现为少一次模型迁移/重载、更低的非活跃
> 生命周期占用和更早的首个完成；在 8w burst 中，Cache 相对 Kairos 同时改善总体平均延迟、
> 模型驻留和空泡；在低负载 Poisson 中，Cache 保持最低总体平均延迟，但预加载尚未形成完整
> 的加载隐藏。

当前数据**不支持**以下更强说法：

- Cache 在所有负载、所有基线上全面最优；
- 预加载是当前端到端收益的主要原因；
- 当前结果验证了 GNN 预测器、KV 生命周期、BubblePath 或 path-wise admission；
- Cache 的控制器 CPU/内存开销低于 History；
- 单次 trial 的点估计具有统计显著性。

## 与 SagePilot 论文范围的关系

当前实验验证的是完整 SagePilot 之前的模型级生命周期运行时：

- 多 workflow 共享一个异构 A100/V100 GPU 池；
- 共享模型副本和跨 workflow 模型复用；
- 基于 ready/near-ready frontier 的加载、预加载与驱逐；
- Cache、History、FIFO 和 Kairos 四种调度策略；
- 有界队列、fan-out/fan-in、连续批处理和背压。

当前实验尚未覆盖论文草稿中的以下完整机制：

- GNN Agent 性能预测器作为运行时资源契约来源；
- KV 持久化、迁移、恢复和重建；
- 完整 BubblePath 的候选生成、路径排序和逐状态内存检查；
- leases、reservations、generation validation 对并发准备路径的保护；
- calibrated transient-memory bound 和 path-wise admission；
- 控制器 CPU 时间、内存占用和规划搜索开销。

当前 `cache` 策略读取由实机 profile 生成的冻结资源点估计。它能够验证“预测信息如何影响
模型级 ETA、预加载和 reuse-distance 驱逐”，但不能把观察到的收益表述为“GNN 带来的收益”。
在论文中，这些结果更适合作为：

- RQ2 中 workflow 可见性和 near-ready 生命周期动作的先导证据；
- RQ4 中共享模型池和模型级生命周期调度的端到端证据；
- 完整 BubblePath 设计的动机与边界证据；
- 后续完整系统实验的 baseline sanity check。

## 冻结实验矩阵

### 正式分析组

| 组别 | 到达模式 | Workflow 构成 | Sessions | GPU | 策略 | 论文用途 |
|---|---|---|---:|---|---|---|
| 2w A1V2 burst | 约 `0.3–0.6s` 内全部提交 | QMSum×60 + MBPP×60 | 120 | 1×A100 + 2×V100 | FIFO/History/Cache/Kairos | Cache-vs-History 机制主证据 |
| 2w r050 A1V2 Poisson | 到达窗口 `4469.988s` | QMSum×60 + MBPP×60 | 120 | 1×A100 + 2×V100 | FIFO/History/Cache/Kairos | 开放到达、预加载和驱逐时序 |
| 8w A1V2 burst | 小于 `0.6s` 内全部提交 | 4×QMSum + 4×MBPP，各15个 session | 120 | 1×A100 + 2×V100 | FIFO/History/Cache/Kairos | 多 workflow 扩展、Cache-vs-Kairos |

三组正式分析共 12 个 trial、1440 个完整 session、7920 次模型获取。所有 trial 均为：

- `120/120` session 完成；
- `0` session 失败；
- `0` OOM；
- `0` 不可调度请求；
- acquire、fan-in 和 item enqueue/emission trace 未配对数均为 `0`。

每个配置当前只有一个 trial。相同组内共享相同输入和到达轨迹，可以做配对描述，但不能把
120 个相互影响的 session 当作 120 次独立系统重复来计算置信区间。

### 补充组与排除组

| 组别 | 状态 | 使用建议 |
|---|---|---|
| 2w A1V1 legacy burst | 四策略完成，Cache 数值优势强 | 旧版 MBPP DAG，只有600次模型获取且缺少当前 `diagnoser` 节点，只可作补充证据 |
| 2w A1V3 | 缺少 History，目录名与 trace 中实际三张卡不完全一致 | 不进入论文主比较 |
| 8w r075 Poisson | FIFO/History/Kairos 完成，缺少 Cache | 无法形成 Cache 配对比较 |

## 策略口径

| 策略 | 本轮实验中的含义 |
|---|---|
| FIFO | ready 请求按 FIFO 处理；不做 near-ready 预加载，也不使用预测 reuse distance 选择驱逐对象 |
| History | 使用当前 trial 内逐步积累的加载与运行 EMA 估计 ETA、预加载时间和 reload cost；每个 trial 从空历史开始 |
| Cache | 从 trial 开始即可读取冻结缓存中的运行与加载点估计，用于 ETA、预加载和 reuse-distance 驱逐 |
| Kairos | 使用全局 remaining-critical-path SRPT 排序和 memory-aware placement；不预加载，驱逐退化为 LRU |

Cache 和 History 都使用相同的 workflow DAG、请求、模型、设备候选、显存门禁和连续批处理。
二者的核心区别是估计来源：Cache 从第一项任务起有冻结估计，History 需要先观察本轮执行。

## 指标口径

本文不使用尾延迟作为主要判断指标，重点使用以下端到端和生命周期指标：

| 指标 | 定义 |
|---|---|
| 平均端到端耗时 | 每个 session 从提交到最终结果完成的时间，再对120个 session 求算术平均；包含排队、依赖等待、模型获取和执行 |
| Makespan | 从 run 开始到最后一个业务 session 完成的 wall-clock 时间 |
| Active GPU-s | 每个副本至少有一个推理正在执行的时间并集，再跨副本求和；并发 batch 不重复计时 |
| Resident GPU-s | 模型完成加载后到开始驱逐之间的累计 GPU 时间，包含 active 与 idle |
| Idle-resident GPU-s | Resident 区间内没有任何推理执行的累计 GPU 时间 |
| Pipeline bubble ratio | 业务窗口内 idle-resident GPU-s 占 workload resident GPU-s 的比例 |
| Loading GPU-s | 模型开始加载到完成加载的累计 GPU 时间 |
| Evicting GPU-s | 模型开始驱逐到完成驱逐的累计 GPU 时间 |
| Lifecycle GPU-s | `loading + resident + evicting`，表示完整模型生命周期占用 |
| Non-active lifecycle GPU-s | `lifecycle − active`，表示模型生命周期内未用于实际推理的 GPU 占用 |
| Loads/Reuses/Evictions | 完成加载、直接复用已驻留副本和完成驱逐的事件数 |
| Prefetches | 后继节点显式请求模型前启动的加载次数 |
| Wasted prefetches | 预加载后未被任何任务使用便被驱逐，或运行结束时仍未使用的次数 |

GPU-s 是跨 GPU 和模型副本累加的时间。例如两个副本各驻留50秒等于100 GPU-s，即使
wall-clock 只经过50秒。Resident GPU-s 衡量的是模型占住 GPU 的时间，不是显存字节积分，
也不是能耗。Pipeline bubble ratio 是从模型驻留与推理 trace 推导出的系统空泡，不等同于
硬件采样得到的 GPU 核心利用率。本文用它回答“模型已经占住 GPU 时，有多少时间没有服务推理”。

总体平均端到端耗时对120个 session 等权；因为三组正式实验都是60个 QMSum 和60个 MBPP，
也等于两类 workload 平均值的算术平均。QMSum/MBPP 分项平均用于判断总体收益是否由牺牲某一类
workload 换来。

`summarize_trial()` 中的 `resident_gpu_seconds` 只统计完成加载后的驻留区间；当前
`run_summary.json` 中名为 `resident_gpu_seconds` 的字段实际等于完整 lifecycle GPU-s。
本文显式分开这两个口径，避免把加载时间误写成驻留时间。

除百分点外，本文的“降低百分比”均使用：

```text
(baseline − cache) / baseline × 100%
```

## 2w A1V2 burst：当前 DAG 下的主要机制比较

### 端到端结果

| 策略 | 平均端到端耗时 (s) | QMSum 平均 (s) | MBPP 平均 (s) | Makespan (s) | Sessions/min | 首次完成 (s) |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | **844.501** | 835.192 | 853.811 | 1384.198 | 5.202 | 224.956 |
| History | 892.156 | 1011.177 | 773.135 | 1415.352 | 5.087 | 341.907 |
| Cache | 865.871 | 975.439 | **756.302** | **1374.377** | **5.239** | 295.375 |
| Kairos | 924.497 | **689.699** | 1159.295 | 1494.889 | 4.816 | **187.652** |

FIFO 的总体平均耗时最低，因此本组不能写成 Cache 对所有策略全面获胜。Cache 的主要用途是
与 History 和 Kairos 比较：

- 相对 History，Cache 平均端到端耗时降低 `2.946%`，makespan 降低 `2.895%`；
- 相对 Kairos，Cache 平均端到端耗时降低 `6.341%`，makespan 降低 `8.062%`；
- Kairos 强烈偏向 QMSum，Cache 在 QMSum 与 MBPP 之间更平衡；
- Cache 的 MBPP 平均耗时在四策略中最低。

### 模型生命周期结果

| 策略 | Active GPU-s | Loading GPU-s | Resident GPU-s | Idle GPU-s | Lifecycle GPU-s | Bubble | Loads | Reuses | Prefetches | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1559.640 | 1112.737 | **2872.409** | **1312.768** | 3996.778 | **45.604%** | 20 | 640 | 0 | 20 |
| History | 1596.536 | 901.354 | 3169.283 | 1572.746 | 4080.248 | 49.541% | 17 | 643 | 1 | 17 |
| Cache | 1578.839 | **827.659** | 3112.405 | 1533.566 | **3949.423** | 49.189% | **16** | **644** | 1 | **16** |
| Kairos | 1577.721 | 1010.199 | 3283.820 | 1706.099 | 4303.245 | 51.882% | 17 | 643 | 0 | 17 |

相对 History，Cache 同时获得：

- Resident GPU-s 降低 `1.795%`；
- Idle-resident GPU-s 降低 `2.491%`；
- Lifecycle GPU-s 降低 `3.206%`；
- Pipeline bubble 降低 `0.353` 个百分点；
- 加载和驱逐各少一次，复用多一次。

相对 Kairos，Cache 同时获得：

- Resident GPU-s 降低 `5.220%`；
- Idle-resident GPU-s 降低 `10.113%`；
- Lifecycle GPU-s 降低 `8.222%`；
- Pipeline bubble 降低 `2.693` 个百分点。

相对 FIFO，Cache 的平均端到端耗时高 `2.530%`、Resident GPU-s 高 `8.355%`、空泡率高
`3.585` 个百分点。这是需要保留的边界结果：三张 GPU 上的两 workflow burst 并不是 Cache
相对简单 FIFO 最有利的工作区间。

## Cache 对 History：最干净的生命周期证据

### 相似的基础工作与控制动作

2w A1V2 中，Cache 和 History 的下列指标相近：

| 指标 | Cache | History | 差异 |
|---|---:|---:|---:|
| 完成 sessions | 120 | 120 | 相同 |
| 模型获取次数 | 660 | 660 | 相同 |
| Active GPU-s | 1578.839 | 1596.536 | Cache低1.109% |
| Mean acquire wait | 29.736s | 29.444s | Cache高0.993% |
| Mean fan-in wait | 345.801s | 340.686s | Cache高1.502% |
| Prefetches | 1 | 1 | 相同 |
| Wasted prefetches | 0 | 0 | 相同 |
| Scheduler decisions | 689 | 691 | 近似相同 |

Cache 的收益不是来自更多预加载、更多调度动作或更少 fan-in/acquire 等待。更有解释力的差异是：

| 指标 | Cache | History | Cache 改善 |
|---|---:|---:|---:|
| 平均端到端耗时 | 865.871s | 892.156s | 2.946% |
| Makespan | 1374.377s | 1415.352s | 2.895% |
| 首次完成 | 295.375s | 341.907s | 13.610% |
| Lifecycle GPU-s | 3949.423 | 4080.248 | 3.206% |
| Non-active lifecycle GPU-s | 2370.584 | 2483.711 | 4.555% |
| Loading+evicting GPU-s | 837.018 | 910.965 | 8.117% |
| Backpressure | 1101.745s | 1260.757s | 12.612% |
| vLLM mean queue | 0.0310s | 0.0331s | 6.213% |

Cache 在 120 个配对 session 中有 80 个比 History 更快，History 有 40 个更快。两类 workload
均得到改善：

- QMSum 平均耗时降低 `3.534%`；
- MBPP 平均耗时降低 `2.177%`。

### Trace 级模型迁移案例

`e219` 和 `b3dd` 是 trace 中的模型部署哈希前缀。运行开始约 223–337 秒时出现以下差异：

```text
Cache:
  t=223.4s  evict e219
  t=224.1s  load b3dd
  t=290.2s  b3dd ready
  t=295.4s  first session complete

History:
  t=224.9s  evict e219
  t=225.6s  migrate/reload e219
  t=267.6s  e219 ready
  t=269.8s  evict e219
  t=270.5s  load b3dd
  t=337.4s  b3dd ready
  t=341.9s  first session complete
```

Cache 避免了中间一次 `e219` 迁移/重载，使 `b3dd` 提前 `47.2s` 就绪；首个 session 相应提前
`46.5s` 完成。该时间关系与聚合计数一致：

- Cache：12 次 ready-load 驱逐、1 次 near-ready 驱逐、3 次结束清理；
- History：13 次 ready-load 驱逐、1 次 near-ready 驱逐、3 次结束清理。

这里的“结束清理”对应 trace 中 run shutdown 发起的3个 `reason=requested` 驱逐，不是调度策略
在业务窗口内选择的 victim。

这一案例最适合解释 Cache 相对 History 的机制优势：

1. Cache 从 trial 开始即可使用冻结运行时间和加载成本；
2. History 必须先积累本轮 EMA，冷启动阶段缺少稳定的 ETA 和 reload cost；
3. 不同估计导致不同的模型迁移顺序；
4. Cache 避免一次迁移/重载，降低非活跃 lifecycle GPU-s 和背压；
5. 后继模型更早就绪，形成可观察的首完成和平均端到端收益。

这是一条 trace 支持的机制链，但当前只有一次 trial。正式论文需要在随机运行顺序和一致冷启动
条件下重复验证“少一次迁移”是否稳定出现。

## 2w r050 A1V2 Poisson：开放到达与预加载边界

### 端到端结果

| 策略 | 平均端到端耗时 (s) | QMSum 平均 (s) | MBPP 平均 (s) | Makespan (s) | Sessions/min | 首次完成 (s) |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 120.724 | 86.899 | 154.548 | 4640.458 | 1.5516 | **187.652** |
| History | 118.514 | 86.853 | **150.176** | 4639.079 | 1.5520 | 214.872 |
| Cache | **117.487** | **84.073** | 150.902 | **4638.883** | **1.5521** | 193.687 |
| Kairos | 128.862 | 93.034 | 164.690 | 4641.304 | 1.5513 | 195.239 |

Cache 的总体平均端到端耗时在四策略中最低：

- 相对 FIFO 降低 `2.681%`，88/120 个配对 session 更快；
- 相对 History 降低 `0.867%`，63/120 个配对 session 更快；
- 相对 Kairos 降低 `8.827%`，103/120 个配对 session 更快。

所有策略的 makespan 和吞吐几乎相同，因为总 wall-clock 主要由固定 Poisson 到达窗口决定。
这组应比较到达后的平均完成时间，而不应把 makespan 的小数点差异写成系统加速。

### 模型生命周期结果

| 策略 | Active GPU-s | Loading GPU-s | Resident GPU-s | Idle GPU-s | Lifecycle GPU-s | Bubble | Loads | Reuses | Prefetches | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1633.766 | 4310.779 | 9231.294 | 7597.527 | 13582.975 | 82.290% | 82 | 578 | 0 | 82 |
| History | 1650.092 | 4318.331 | 9256.961 | 7606.869 | 13616.625 | 82.164% | **81** | **579** | **27** | **81** |
| Cache | 1642.218 | 4334.241 | 9256.923 | 7614.705 | 13632.488 | 82.248% | 82 | 578 | 23 | 82 |
| Kairos | **1622.474** | **4271.451** | **9064.039** | **7441.565** | **13376.416** | **82.085%** | **81** | **579** | 0 | **81** |

Cache 相对 History：

- Resident GPU-s 基本相同，Cache 低 `0.0004%`；
- Lifecycle GPU-s 高 `0.116%`；
- Idle-resident GPU-s 高 `0.103%`；
- Bubble 高 `0.083` 个百分点；
- 平均端到端耗时低 `0.867%`。

因此可表述为：

> 在低负载开放到达下，Cache 以与 History 实质相同的模型生命周期占用获得小幅平均延迟改善。

Cache 相对 Kairos 的平均端到端耗时低 `8.827%`，但 Resident GPU-s 高 `2.128%`、Lifecycle
GPU-s 高 `1.914%`。Kairos 是该组资源最省的策略，Cache 是总体平均延迟最低的策略。

### 预加载时序

Cache 和 History 的预加载事件如下：

| 指标 | Cache | History |
|---|---:|---:|
| Prefetch started | 23 | 27 |
| Wasted prefetch | 0 | 0 |
| 请求到达前完成 | 0 | 0 |
| 平均比请求早启动 | 5.426s | 5.144s |
| 平均实际加载耗时 | 60.354s | 60.491s |
| 平均预期加载耗时 | 197.338s | 63.034s |
| 相对计划时刻平均晚启动 | 161.622s | 59.859s |

`wasted_prefetch=0` 只说明预加载最终被使用，不说明加载被隐藏。当前所有预加载都在后继请求到达
时仍未完成；Cache 平均只隐藏约5.4秒，而一次实际加载约60.4秒。

Cache 的预加载估计平均为197.3秒，实际约60.4秒，明显过高；同时 GPU 容量和 ready work
使预加载无法在计划时刻启动。该组最重要的结论不是“Cache 预加载成功加速”，而是：

> 单独预测一个预加载时刻不足以保证准备在真实 dependency window 内完成。准备动作必须与
> 当前设备状态、被驱逐对象、ready-work 优先级和完整有序转换共同调度。

这一负面结果与 SagePilot 论文中把完整有序准备路径作为调度单位的 BubblePath 动机一致。
它可以作为设计动机或 eager/isolated prefetch 的边界证据，不能直接当作完整 SagePilot 已经解决
该问题的结果。

### 驱逐动作如何变化

| 策略 | Ready-load 驱逐 | Near-ready 驱逐 | 结束清理 | 总驱逐 |
|---|---:|---:|---:|---:|
| FIFO | 79 | 0 | 3 | 82 |
| History | 54 | 24 | 3 | 81 |
| Cache | 55 | 24 | 3 | 82 |
| Kairos | 78 | 0 | 3 | 81 |

Cache 和 History 都把24次原本发生在请求 ready 后的换模提前到了 near-ready 阶段，但并未减少
总 churn。该结果证明 workflow-aware lifecycle 路径被触发，也说明当前策略主要是在移动换模
时机，而不是稳定减少换模次数。

## 8w A1V2 burst：多 Workflow 扩展

### 端到端结果

| 策略 | 平均端到端耗时 (s) | QMSum 平均 (s) | MBPP 平均 (s) | Makespan (s) | Sessions/min | 首次完成 (s) |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 1337.585 | 1384.430 | 1290.740 | 1949.226 | 3.694 | 851.596 |
| History | 1522.645 | 1622.063 | 1423.226 | 1770.384 | 4.067 | 1382.264 |
| Cache | **1250.232** | 1412.981 | **1087.483** | **1600.573** | **4.498** | 1047.082 |
| Kairos | 1290.823 | **973.492** | 1608.153 | 1717.564 | 4.192 | **242.646** |

Cache 的总体平均端到端耗时在四策略中最低：

- 相对 FIFO 降低 `6.531%`，93/120 个配对 session 更快；
- 相对 History 降低 `17.891%`，117/120 个配对 session 更快；
- 相对 Kairos 降低 `3.145%`，69/120 个配对 session 更快。

Kairos 优先缩短 QMSum critical path，但 MBPP 平均耗时达到1608.2秒；Cache 的总体优势主要来自
更均衡地服务两类 workflow，而不是每一类都比 Kairos 更快。

### 模型生命周期结果

| 策略 | Active GPU-s | Loading GPU-s | Resident GPU-s | Idle GPU-s | Lifecycle GPU-s | Bubble | Loads | Reuses | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1970.906 | 1525.332 | 3603.519 | 1632.613 | 5133.298 | 45.177% | 5 | 655 | 5 |
| History | **1749.818** | 1450.018 | **2799.918** | **1050.100** | 4252.608 | **37.318%** | **4** | **656** | **4** |
| Cache | 1899.757 | 833.427 | 3226.109 | 1326.353 | **4061.863** | 40.967% | **4** | **656** | **4** |
| Kairos | 2117.223 | **293.661** | 4539.886 | 2422.663 | 4837.073 | 53.280% | 5 | 655 | 5 |

Cache 相对 Kairos 同时获得：

- 平均端到端耗时降低 `3.145%`；
- makespan 降低 `6.811%`；
- Resident GPU-s 降低 `28.939%`；
- Idle-resident GPU-s 降低 `45.252%`；
- Lifecycle GPU-s 降低 `16.026%`；
- Bubble 降低 `12.313` 个百分点；
- 少一次模型加载和运行期驱逐，多一次直接复用。

这组是当前最适合展示 Cache 在多 workflow 环境中相对 Kairos 的综合收益。

Cache 相对 FIFO 的观察值也同时改善平均延迟、Resident GPU-s 和 Bubble，但端到端数字受冷加载
差异影响，正式论文应先完成随机运行顺序的重复实验。

Cache 相对 History 是明确的资源—延迟折中：

- 平均端到端耗时低 `17.891%`；
- makespan 低 `9.592%`；
- Resident GPU-s 高 `15.222%`；
- Idle-resident GPU-s 高 `26.307%`；
- Bubble 高 `3.649` 个百分点。

History 的低驻留和低空泡不等于更快。它把更多等待放在模型尚未驻留或应用队列阶段，因此
平均端到端耗时最高。论文中应将 Cache 描述为更平衡的 latency–residency operating point，
而不是所有资源指标的最小值。

### 共享模型池与换模

8w burst 中：

- Cache/History：4次加载、656次复用，复用率 `99.394%`；
- FIFO/Kairos：5次加载、655次复用，复用率 `99.242%`；
- Cache/History：1次运行期驱逐和3次结束清理；
- FIFO/Kairos：2次运行期驱逐和3次结束清理。

该组支持“多 workflow 共享池能够跨 workflow 复用相同部署，并减少一次不必要的换模”。
它不支持“预加载带来收益”，因为所有策略的 prefetch count 都为0。

### 8w 冷加载混杂因素

8w 的首轮加载时间存在明显不一致。例如相同模型/GPU组合：

- `60ef`/V100：Cache `57.7s`，History `337.2s`；
- `880b`/A100：Cache `404.8s`，History `735.6s`；
- `b3dd`/V100：Cache `263.6s`，History `267.3s`；
- `e219`/V100：Cache `94.2s`，History `93.6s`。

前两个模型的差异远大于调度策略能直接解释的范围，可能来自文件系统页缓存、模型缓存、
运行顺序或平台状态。由于 History 的加载更慢，Cache-vs-History 的端到端和 total lifecycle
优势不能全部归因于策略。

Cache-vs-Kairos 的方向更稳健：Kairos 的 Loading GPU-s 反而远低于 Cache，但 Cache 仍然获得
更低总体平均耗时、Resident GPU-s、Idle GPU-s 和 Bubble。即使如此，正式结果仍需随机化策略
运行顺序并控制冷启动。

## 旧 A1V1 burst：受限资源下的补充结果

无后缀的 `output/serve/2w/hetero_{policy}` 实际使用1张A100和1张V100。该组中 Cache 数值优势
最强：

| Cache 相对基线 | 平均耗时降低 | Makespan 降低 | Resident 降低 | Idle 降低 | Lifecycle 降低 | Bubble 降低 |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 20.098% | 25.138% | 14.255% | 22.791% | 26.218% | 6.177pp |
| History | 20.823% | 10.694% | 1.784% | 5.373% | 11.947% | 2.135pp |
| Kairos | 6.745% | 14.126% | 7.132% | 12.397% | 15.342% | 3.372pp |

Cache 和 History 的 loads/reuses/evictions 完全相同，均为 `13/587/13`，但 Cache 的平均端到端
耗时低 `20.823%`。这提示冻结预测可能在受限 GPU 池中比在线历史更早形成有效顺序。

该组不能作为当前主结果，原因是：

- 使用旧版 MBPP DAG，缺少当前 `diagnoser` 节点；
- 每个 trial 有600次模型获取，而当前 DAG 为660次；
- 首轮模型加载时间存在显著冷启动差异；
- 只有一个 trial。

建议使用当前 DAG 重跑 A1V1，并随机化策略顺序。若趋势复现，该组会成为模型生命周期创新点
最强的受限容量证据。

## 面向论文的证据选择

### Claim–dataset 映射

| 拟支持的论文论点 | 首选实验与比较 | 可使用的数字 | 必须附带的限制 |
|---|---|---|---|
| 冻结预测比在线 History 更早形成有效生命周期决策 | 2w A1V2 Cache vs History | 平均耗时−2.946%，lifecycle−3.206%，non-active lifecycle−4.555%，少一次重载 | n=1，需要冷启动受控重复 |
| Cache 避免一次模型迁移并提前解锁后继 | 2w A1V2 trace | `b3dd` 提前47.2s就绪，首次完成提前46.5s | 单个机制案例，需重复统计 avoidable reload |
| 多 workflow 下 Cache 比 Kairos 更平衡 | 8w A1V2 Cache vs Kairos | 平均耗时−3.145%，resident−28.939%，idle−45.252%，bubble−12.313pp | Kairos对QMSum更快，必须同时报告MBPP |
| 共享模型池实现高复用并减少换模 | 8w A1V2 | Cache 656/660复用、1次运行期驱逐；Kairos/FIFO 655/660、2次 | 复用差异只有一次事件 |
| 低负载下 Cache 不引入显著 History 生命周期开销 | r050 Cache vs History | 平均耗时−0.867%，lifecycle+0.116%，resident近似相同 | 不应宣称资源节省 |
| Workflow-aware near-ready 动作确实执行 | r050 Cache trace | 23次prefetch、24次near-ready eviction、0次未使用 | 没有一次在请求前完成，不证明加速 |
| 受限容量放大 Cache 生命周期收益 | A1V1 legacy | Cache相对FIFO平均耗时−20.098%、resident−14.255% | 旧DAG，只能补充，优先重跑 |

### 推荐主表

论文主表优先使用当前 DAG 的三组 A1V2 数据：

1. 2w burst：展示 Cache-vs-History/Kairos，并完整保留 FIFO 边界结果；
2. 2w r050 Poisson：展示开放到达下四策略平均端到端耗时与 lifecycle；
3. 8w burst：展示多 workflow 扩展和 Cache-vs-Kairos。

主表应同时给出：

- mean workflow completion time；
- makespan；
- resident GPU-s；
- idle-resident GPU-s；
- pipeline bubble ratio；
- loads/reuses/prefetches/evictions；
- failed/OOM/infeasible counts。

不要只报告 Cache 有利的列。特别是：

- 2w burst 中 FIFO 的总体平均耗时和 Resident GPU-s 更低；
- r050 中 Kairos 的 Resident/Lifecycle GPU-s 更低；
- 8w 中 History 的 post-load Resident GPU-s 和 Bubble 更低。

这些边界不会削弱论文，反而明确了 Cache 的工作区间和 latency–resource trade-off。

### 推荐机制图

**模型迁移时间线。**

使用 2w A1V2 Cache/History 在 `t≈160–350s` 的 trace，画出两个V100上的模型 load/evict 区间、
`b3dd` ready 时间和首次 session 完成时间。该图直接连接“预测来源—驱逐顺序—模型就绪—端到端结果”。

**Lifecycle 分解图。**

使用 2w A1V2，将 GPU 时间分解为：

```text
loading + active resident + idle resident + evicting
```

突出 Cache 相对 History 的 non-active lifecycle GPU-s 降低 `4.555%`。这比单独报告
resident GPU-s 更能说明避免换模和减少空闲占用的共同作用。

**Poisson 预加载时序图。**

对 r050 的每次 prefetch 画：

```text
planned prefetch_at → actual start → actual finish → downstream acquire
```

该图应展示所有预加载都在 acquire 后才完成，并将其解释为 isolated prefetch 的边界，而不是
把 `wasted=0` 等同于成功隐藏加载。

**8w workload 分解图。**

分别画 QMSum 和 MBPP 的平均端到端耗时。该图能说明 Kairos 对 QMSum 的强优先与 MBPP 代价，
以及 Cache 的总体平均优势来自更平衡的跨 workflow 调度。

## 可直接改写进论文的表述

### Cache 对 History

> In a matched two-workflow burst replay on one A100 and two V100s, the
> cache-guided policy reduced mean workflow completion time by 2.95% and
> makespan by 2.90% relative to the online-history policy. Active GPU time,
> acquire wait, fan-in wait, and the number of prefetches remained within
> 1.6%, while total model-lifecycle GPU time fell by 3.21% and non-active
> lifecycle time by 4.56%. The trace attributes the difference to one avoided
> model migration/reload: the downstream model became ready 47.2 s earlier,
> closely matching a 46.5 s improvement in time to first completion.

### Cache 对 Kairos

> With eight concurrent workflows on the same heterogeneous pool, the
> cache-guided policy reduced mean workflow completion time by 3.14%,
> model residency by 28.94%, idle residency by 45.25%, and the pipeline
> bubble ratio by 12.31 percentage points relative to Kairos. Kairos completed
> QMSum workflows earlier but substantially delayed MBPP, whereas the
> cache-guided policy achieved the lowest aggregate mean completion time.

### 开放到达

> At an aggregate Poisson load of approximately 0.5, the cache-guided policy
> achieved the lowest mean completion time (117.5 s), compared with 118.5 s
> for online history, 120.7 s for FIFO, and 128.9 s for Kairos. Its total
> model-lifecycle GPU time was within 0.12% of online history. Although the
> trace recorded 23 prefetches and no unused prefetched replica, none completed
> before its consumer became ready; the result therefore validates mechanism
> activation, not effective hiding of model-load latency.

以上英文只能作为单次 trial 的观察性表述。完成重复实验前，不应加入 “statistically
significant”、“consistently across runs” 或置信区间语言。

## 不应使用的论证方式

### 不把低 Bubble 单独解释成更快

History 在8w中的 Bubble 最低，但平均端到端耗时最高。低 Bubble 可能只是把等待移到模型加载前
或应用队列中。Bubble 必须与平均端到端耗时、backpressure 和 resident GPU-s 联合解释。

### 不把 `wasted_prefetch=0` 解释成加载被隐藏

r050 中所有预加载最终都被使用，但没有一次在请求前完成。论文应同时报告：

- ready-before-acquire count；
- hidden load seconds；
- prefetch start lateness；
- ready-work delay；
- unused preparation。

### 不把8w加载时间差归因于策略

相同模型/GPU的首轮加载时间相差数百秒，说明存在冷缓存或运行顺序混杂。随机化运行顺序和显式
控制冷启动前，不应声称 Cache 将单次模型加载本身加速。

### 不把当前结果写成 GNN 或完整 BubblePath 收益

当前 Cache 使用冻结缓存点估计，未验证 GNN 预测、KV 路径和 transient-memory admission。
这些实验可以说明模型级预测信息有调度价值，不能证明完整 SagePilot 的三个贡献已经端到端成立。

### 不使用 session 数代替系统重复数

同一 trial 内的 session 共享队列、模型状态和 GPU，彼此不独立。当前每个配置的系统重复数为
`n=1`，不是 `n=120`。

## 后续实验优先级

### 最高优先级：重复当前三组 A1V2

- 每个配置至少5次独立 trial；
- 策略运行顺序随机化或拉丁方平衡；
- 明确区分冷文件缓存、热文件缓存和已驻留模型三种起始状态；
- 配对使用相同数据样本和到达 trace；
- 失败、超时和异常冷加载不从分母中删除。

### 重跑当前 DAG 的 A1V1

旧 A1V1 组显示最强 Cache 收益。使用当前含 `diagnoser` 的660次获取 DAG 重跑后，可验证受限
GPU容量是否稳定放大预测驱逐的价值。

### 补齐8w r075 Poisson Cache

当前 `output/serve/8w_poisson` 缺少 Cache。补齐后可同时观察：

- 多 workflow 开放到达；
- near-ready window 数量；
- Cache 与 History 的预加载及时性；
- workflow mix 对公平性和驻留的影响。

### 增加机制专用指标

建议 trace/分析直接输出：

- `prefetch_ready_before_acquire_count`；
- `hidden_loading_seconds`；
- `prefetch_overrun_seconds`；
- `ready_work_delay_caused_by_preparation`；
- `avoidable_reload_count`；
- 驱逐候选的 reuse distance、reload cost 和最终选择；
- History 首次具有有效 load/run EMA 的时间；
- Cache/History prediction source；
- displaced-state recovery cost。

### 增加控制器开销遥测

当前只能比较 GPU 生命周期开销，不能比较控制器开销。完整 RQ5 需要测量：

- prediction-cache lookup；
- history update；
- near-ready frontier 构造；
- eviction candidate scoring；
- scheduling loop wall time和CPU time；
- controller resident set size（RSS，常驻内存）；
- 每次 decision 和每个 active workflow 的摊销开销。

### 使用 GNN 与完整 BubblePath

若要让这些结果支撑完整 SagePilot 论文，应进一步：

- 将当前冻结的 empirical GPU profile 点估计替换为冻结的 GNN Agent profile；
- 记录 estimate source、预测误差和 calibration scope；
- 将 model/KV action 组合成完整路径；
- 比较 isolated prefetch、independent lifecycle policies 和 BubblePath；
- 报告 transient peak、false admission/deferral、window fit、overrun 和 recovery cost。

## 数据来源与复核入口

每个 trial 的原始产物由三部分组成：

- `workflow_trace.jsonl`：fleet 级事件流，是本文重算端到端、驻留、空泡和生命周期动作的主数据源；
- `run_summary.json`：完成数、失败数和基础事件计数的快速摘要；
- `<workflow>/session_results.jsonl`：每个 workflow 的最终 session 结果与完成状态。

本文的表格以 `workflow_trace.jsonl` 经 `summarize_trial()` 重算的口径为准。特别是
`run_summary.json` 的 `resident_gpu_seconds` 实际包含 loading、resident 和 evicting，不能
直接当作 post-load Resident GPU-s 使用。

### 结果目录

- 2w A1V2 burst：`output/serve/2w/hetero_{fifo,history,cache,kairos}_a1v2`
- 2w r050 A1V2 Poisson：`output/serve/2w_poisson/hetero_r050_{fifo,history,cache,kairos}_a1v2`
- 8w A1V2 burst：`output/serve/8w/hetero_{fifo,history,cache,kairos}_a1v2`
- A1V1 legacy：`output/serve/2w/hetero_{fifo,history,cache,kairos}`

### 配置

- `config/workflow/serve/qmsum{1,2,3,4}.yaml`
- `config/workflow/serve/mbpp{1,2,3,4}.yaml`
- `config/workflow/serve/replay_2w.yaml`
- `config/workflow/serve/replay_2w_poisson_r050.yaml`
- `config/workflow/serve/replay_8w.yaml`
- `config/workflow/serve/scheduler_fifo.yaml`
- `config/workflow/serve/scheduler_history.yaml`
- `config/workflow/serve/scheduler_cache.yaml`
- `config/workflow/serve/scheduler_kairos.yaml`
- `cache/profile/predictions.yaml`

### 指标实现

- Trace 汇总：`src/experiment/workflow/analysis.py`
- 报告生成：`src/experiment/workflow/report.py`
- 调度与 lifecycle：`src/workflow/scheduler.py`
- 驱逐 victim 选择：`src/workflow/policy.py`

## 最终建议

面向当前 HPCA 论文，最稳妥的实验叙事是：

1. 用 **2w A1V2 Cache-vs-History** 证明冻结预测在在线历史冷启动阶段能够避免一次模型迁移，
   降低非活跃生命周期占用和背压；
2. 用 **8w A1V2 Cache-vs-Kairos** 证明该选择在多 workflow 异构池中形成更好的总体
   latency–residency balance；
3. 用 **r050 Poisson** 证明 workflow-aware near-ready 生命周期动作在开放到达下被真实触发，
   同时诚实展示 isolated prefetch 未能完整隐藏加载；
4. 把 **FIFO 在2w burst中的优势、History 在8w中的低驻留、Kairos 在r050中的低 lifecycle**
   作为工作区间边界，而不是隐藏；
5. 把当前结果称为“model-level lifecycle precursor evidence”，待重复实验、GNN profile、
   KV actions 和 BubblePath admission 完成后，再升级为完整 SagePilot 的端到端结论。
