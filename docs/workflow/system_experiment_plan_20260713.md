# Workflow 系统实验方案（2026-07-13）

## 文档定位

本文记录 `src/workflow` 已完成系统实验的设计、实现状态和复现约束。正式矩阵与结果
均已冻结；后续系统版本或新增实验应建立新的方案，不回写本轮预注册设计。

本文自 2026-07-14 起采用精简后的系统验证口径。修订版正式产物使用新的 experiment ID
`system_20260713_v2`，不得与旧 `system_20260713` 目录中的 60-session 工程试跑混合。
修订版 runner、正式矩阵和结果产物均已完成并冻结。

- 实验方案：`docs/workflow/system_experiment_plan_20260713.md`
- 实验结果：`docs/workflow/system_experiment_results_20260713.md`
- 当前系统目标：`docs/dev/workflow_system_phase1.md`
- 历史动机方案：`docs/workflow/motivation_20260703.md`
- 历史动机结果：`docs/workflow/motivation_20260704.md`

状态标记：

- **已确认**：可以直接进入实现或实验。
- **待实现**：方案已经确认，但当前代码尚不支持。
- **待讨论**：不能据此实现，需继续确认原理或取舍。

结果文档与本文分离。本文记录实验为什么这样设计；结果文档记录运行环境、原始数据
索引、统计结果、图和受证据约束的解释，不改变预注册矩阵或事后筛选结果。

## 实验目标

本轮使用两种具有不同拓扑和资源行为的 workflow，比较 LangGraph 粗粒度图执行与
当前细粒度数据流运行时：

| 场景 | 数据集 | 拓扑 | 主要观察对象 |
|---|---|---|---|
| 长文章总结 | QMSum | fan-out/fan-in | 跨 session 流水线、continuous batching、有界队列和背压 |
| 代码生成 | MBPP sanitized | chain | 模型加载复用、预取、驱逐和有限 GPU 下的资源效率 |

QMSum 和 MBPP 在本文中是固定的系统 workload fixture，不是算法 benchmark。正式实验只需
保留足以触发 batching、排队、fan-in、模型切换和资源竞争的代表性 workload，不以扩大样本
覆盖来估计数据集级模型质量。

本轮不设置单一大模型 direct quality reference。不同运行时使用相同 workflow、模型、
prompt、采样参数和输出上限，质量指标用于确认各运行时实际完成的是同一任务，而不是
比较 workflow 与单模型能力。

本轮不接入、训练或验证 GNN。workflow runtime 只读取一份冻结的 synthetic prediction
cache，用它验证缓存驱动的 ETA、预取、复用和驱逐机制。本文不对缓存值的预测准确率、
模型泛化能力或 GNN 收益作任何结论。

实验只列出数据、效应量、离散度和置信区间，不在方案中预先规定必须达到的加速比，
也不根据结果自动给出“成功”或“失败”的陈述性结论。

本文允许得出以下范围内的结论：

- QMSum burst 下细粒度运行时相对真实 LangGraph baseline 的端到端性能差异。
- QMSum 在低于容量和超过容量时的吞吐、延迟与排队变化。
- batching 和有界队列端点对 QMSum 流水线行为的影响。
- MBPP 在 1、2、3 张 GPU 下的性能与资源曲线，以及 2-GPU 下的调度策略差异。

本文不对数据集级准确率、完整负载曲线、未运行参数组合、MBPP open-loop、GNN、预测准确率
或模型泛化能力作结论。

## 实验设计原则

- 实验由论文主张驱动，不为覆盖所有参数组合而扩张矩阵。
- 每项正式实验必须明确对应的研究问题、对照组、自变量、指标和原始产物。
- 正式矩阵在运行前冻结；不得根据阶段性结果临时增加、删除或挑选配置。
- 系统方差优先由独立 trial 重复估计，不用增加同一 trial 内的数据集样本冒充独立重复。
- 每个场景使用 24 条固定样本；质量只作为运行时语义 guardrail。
- 后续新增创新点必须先更新本文、trial manifest 和结果索引，再生成任何正式结果。
- 不单独扫描 workflow in-flight session 数；burst、Poisson load 和 vLLM `max_num_seqs`
  已覆盖本轮需要的负载与 batching 变化。

## 历史动机与工程准备

2026-06-26 的 QMSum pilot、2026-07-03 至 2026-07-04 的 offline pipeline oracle 和
active-frontier lower bound 只用于选择场景和确认研究动机。它们不是真实新运行时的端到端
对比，不重跑、不进入修订版正式矩阵，也不与正式结果合并。具体方法和数字保留在
`docs/workflow/motivation_20260703.md` 与 `docs/workflow/motivation_20260704.md`。

以下已经完成的检查属于工程准备，不作为论文实验：

- Qwen3-4B/8B/14B 在目标 V100 上的初始化与 KV 容量检查。
- 固定 60+60 parent sample 的 prompt/token preflight；它已经覆盖修订版 24+24 子集。
- QMSum 2-GPU 的 3 次 capacity calibration；它只提供开放到达率。

只要模型、prompt、token 上限、硬件和 24 条子集不变，这些检查不重复执行。任一条件变化时
必须重新验证，不能沿用旧结论。

## 当前系统基础

当前 `src/workflow` 已具备通用 DAG 校验、Ray actor、Ray Queue、跨 session 并发、
session fan-in、目标路由、有界缓冲、vLLM 模型副本、模型生命周期管理、真实 LangGraph
vLLM baseline、FIFO/History/Cache 策略、统一 trace、外部 GPU telemetry 和结果汇总。

场景 splitter、prompt、数据加载和 evaluator 放在实验层，不在 `src/workflow` 核心中加入
QMSum 或 MBPP 特判。

修订版运行前只同步实验层硬约束：24 条 sample manifest、75-trial matrix、新 experiment ID、
QMSum capacity 引用以及 report completeness。该同步不得改变 workflow 核心语义。

## 统一运行条件

### 硬件与 vLLM

**已确认**使用本机 4 张 Tesla V100-SXM2-32GB。正式 serving 固定：

- vLLM `0.10.2` V0 engine。
- XFormers attention backend。
- FP16、eager mode、TP=1、PP=1。
- `block_size=16`、FCFS。
- 关闭 prefix caching、chunked prefill、speculative decoding、量化、CPU offload 和 swap。
- `do_sample=false`、`enable_thinking=false`。

这些配置用于控制 serving 优化变量并稳定复现实验。vLLM 的 PagedAttention 和 continuous
batching 与 eager 静态 batch 不同；本轮不据此验证 GNN 预测结果。

2026-07-13 已完成目标部署初始化预检：

| 模型 | 预检 max model length | GPU KV blocks | block size | GPU KV tokens | load 后显存 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 8192 | 10397 | 16 | 166352 | 31144.6 MiB |
| Qwen3-8B | 4096 | 6975 | 16 | 111600 | 31338.9 MiB |
| Qwen3-14B | 1024 | 1370 | 16 | 21920 | 31627.1 MiB |

三个实例均正常 shutdown，预检结束后四张 GPU 均恢复到 `0 MiB`。正式 trace 应同时记录
单请求 `max_model_len` 和实例实际 KV block/token 容量，二者不是同一个指标。

### 冷启动与重复

- 每个正式 trial 开始前 GPU 必须为空。
- 独立 preflight 可以预热文件系统和编译缓存，但必须退出并释放 GPU。
- trial 的计时起点在模型加载前，模型加载时间计入端到端结果。
- 同时报告包含冷启动的完整结果和首轮加载结束后的 steady-state 结果。
- 每个正式实验单元重复 5 次。
- 五次使用固定 seed、固定样本集合和固定 arrival trace；不同策略共享相同输入顺序。
- burst workload 在 `t=0` 提交该 trial 的全部 24 个 session，五次重复使用固定的五种样本排列。
  这只表示 session 同时到达；下游节点仍按 DAG 依赖逐步产生，并受队列和 serving 容量约束。

统计以 trial 为独立重复，不把同一 trial 内的 24 个 session 当成 24 次独立系统实验。

## QMSum fan-out/fan-in

### 数据与拓扑

**已确认**从现有 QMSum `ALL/test` 的 60 条 seed 42 parent sample 中固定选择 24 条，
short、medium、long 各 8 条。它们是系统 workload，不用于估计数据集级 ROUGE。

选择过程只使用正式结果产生前已经存在的系统成本特征：

- 每个 stratum 按 6 个 chunk prompt 的 input token 总量排序，覆盖最短、最长和中间分位。
- 在接近同一分位时优先增加不同 meeting 的覆盖，并保持 general/specific query 比例接近
  parent sample；最终以 sample ID 确定性打破并列。
- 不读取模型输出、质量分数或策略性能来决定样本。
- 固化精确 sample IDs、分层、选择规则、parent manifest hash 和正式 manifest hash。

24 条样本大于默认 `queue_capacity=16`，且能被 `max_num_seqs=3` 整除。每个 QMSum trial
仍产生 144 个 chunk 请求和 24 个 merge 请求，足以观察 batching、fan-in、有界队列和背压。

```text
split
  ├── chunk_0 ──┐
  ├── chunk_1 ──┤
  ├── chunk_2 ──┤
  ├── chunk_3 ──┼── merge
  ├── chunk_4 ──┤
  └── chunk_5 ──┘
```

- splitter 按 transcript turn 数连续等分为 6 份，不按负载或 token 做动态均衡。
- 6 个 chunk 是逻辑节点，共享一个 Qwen3-4B vLLM engine。
- merge 使用一个 Qwen3-8B vLLM engine。
- 不增加 refine 节点，不使用外部 LLM judge。
- 固定使用 2 张 V100：一张驻留 4B chunk engine，一张驻留 8B merge engine。
- chunk：`max_model_len=8192`、`max_new_tokens=384`。
- merge：`max_model_len=4096`、`max_new_tokens=384`。
- 已完成的 60 条 parent preflight 覆盖正式子集；只要 prompt 和输出上限不变，不再重复执行。
- 任一配置变化后的 preflight 发现 prompt 加输出上限越界时直接失败，不静默截断。

质量使用 deterministic ROUGE-1、ROUGE-2 和 ROUGE-L。

### 正式对比

主配置使用 `max_num_seqs=3`、`queue_capacity=16`，比较：

| 组别 | 含义 |
|---|---|
| `LG-Batch` | LangGraph 并发执行完整 workflow；静态加载并共享 4B/8B vLLM engines |
| `WF-FIFO` | 当前数据流执行器；按到达顺序、按需加载、固定 token 上限、简单驱逐 |
| `WF-Cache` | 使用冻结的 synthetic cache 做 ETA、预取和 workflow-aware 驱逐 |

`LG-Batch` 必须是真实 LangGraph graph execution，并允许 24 个完整 workflow invocation
并发进入；不能继续使用历史 session-at-a-time 循环，也不能用节点锁串行化同一 engine 的
请求。所有运行时共享同一 vLLM backend、tokenizer、chat template、模型实例规格和生成参数。

QMSum 中两个模型固定驻留在两张 GPU，`WF-History` 的生命周期信息不会形成独立的论文
证据，因此不进入正式矩阵。

### QMSum 消融

不运行 3×3 全网格。默认 `WF-Cache(seq3, queue16)` 已包含在主对比中，只增加两个
单变量端点：

| 配置 | 控制变量 | 论文用途 |
|---|---|---|
| `WF-Cache(seq1, queue16)` | 仅关闭 continuous batching | 观察 batching 对吞吐和空泡的影响 |
| `WF-Cache(seq3, queue1)` | 仅收紧缓冲 | 观察有界队列和背压的影响 |

`max_num_seqs=2`、`queue_capacity=4` 以及交叉组合不进入正式矩阵。本文不由两个端点外推
完整参数敏感性曲面，也不把 `queue_capacity=16` 宣称为无界队列。

QMSum 主实验不涉及模型驱逐，因为两个模型分别固定驻留在两张 GPU。主要指标为：

- 全部 24 个 session 的 makespan 和 sessions/min。
- session latency p50、p95、max 和 first completion latency。
- vLLM queue time、TTFT、每节点执行时长和 replica inflight。
- fan-in wait、pipeline bubble、stage utilization。
- queue occupancy、queue put blocked time 和 backpressure duration。
- 以时间区间并集计算的 active/idle/resident GPU seconds。
- ROUGE-1、ROUGE-2、ROUGE-L、空输出和 token-limit hit。

## MBPP chain

### 数据与拓扑

**已确认**从现有 MBPP sanitized test 的 60 条 seed 42 parent sample 中固定选择 24 条，
short、medium、long 各 8 条。所有样本固定经过 repair，不根据首次测试结果提前退出。

每个 stratum 按已有 preflight 中 repair prompt 的 input token 数覆盖最短、最长和中间分位，
必须包含 1600-token 的最长 repair 样本，并以 sample ID 确定性打破并列。选择过程不读取
代码是否通过测试或任何策略结果。最终固化精确 IDs、选择规则、parent manifest hash 和正式
manifest hash。

24 个 session 能被 CPU evaluator 并发 8 整除，仍执行 72 个 LLM stage，并完整经过三种模型
和两次 CPU evaluator，足以观察模型加载、复用、预取、驱逐与资源竞争。该集合只作为系统
workload，不用于估计数据集级 pass@1。

```text
coder(Qwen3-14B)
  -> tester(CPU)
  -> reviewer(Qwen3-4B)
  -> repair(Qwen3-8B)
  -> final_tester(CPU)
```

| 节点 | 配置 |
|---|---|
| coder | `max_model_len=1024`、`max_new_tokens=512` |
| reviewer | `max_model_len=2048`、`max_new_tokens=512` |
| repair | `max_model_len=3072`、`max_new_tokens=512` |
| tester/final_tester | `evaluate_mbpp`，固定 CPU 并发 8 |

完整 60 条 parent sample 的 prompt preflight 中，repair 最大输入为 1600 tokens，加固定输出
上限后为 2112 tokens。原 2048 配置会使 6 条样本越界，因此固定上调到 3072；不裁剪输入，
也不降低 `max_new_tokens`。正式 24 条子集包含该最长样本，因此不重复执行相同 preflight。

两种运行时共享同一个 CPU evaluator 并发限制，避免 LangGraph 并发执行而 workflow runtime
串行执行 evaluator。质量同时报告 initial pass@1 和 final pass@1，并记录 `no_code`、
syntax error、runtime error、assertion error 和 timeout。

### 正式对比

burst 资源曲线：

| 组别 | GPU 数 | 含义 |
|---|---:|---|
| `LG-Batch` | 3 | 三个模型静态驻留，LangGraph 并发执行完整 chain |
| `WF-Cache1` | 1 | 三模型按 workflow 进度加载、复用和驱逐 |
| `WF-Cache2` | 2 | 允许下一模型在空闲 GPU 上预取 |
| `WF-Cache3` | 3 | 与 LangGraph 使用相同 GPU 数，比较执行与生命周期策略 |

同资源策略对比只在 2-GPU 运行 `WF-FIFO`、`WF-History`、`WF-Cache`。`WF-Cache2`
已经包含在资源曲线中，因此只新增 FIFO 和 History 两个实验单元。1-GPU 无法验证预取，
其三策略横向对比不进入正式矩阵。

MBPP 主要指标为：

- makespan、sessions/min、session latency p50/p95/max。
- 每 stage latency、queue wait、model-switch wait、TTFT。
- 模型 load 次数、load time、reuse、prefetch、eviction 和无效预取。
- peak resident replicas、resident GPU seconds 和 resident VRAM-GiB·s。
- active GPU seconds 使用同一 replica 上推理区间的并集，不累计重叠请求时长。
- sessions/min/GPU、GPU seconds/completion 和 energy/completion。
- initial/final pass@1 与 evaluator 失败类型。

结果同时列出绝对延迟、资源数和单位 GPU 效率，不只报告驻留实例数量比例。

## 开放到达 workload

只有 QMSum 运行 workflow session 级 Poisson arrival，不把 LLM 节点当成独立请求。QMSum
的直接先例是使用 LongBench QMSum 并以 Poisson RPS 扫描负载的
[FlowKV](https://arxiv.org/html/2504.03775)；agent program 级 Poisson 到达可参考
[Agentix](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_luo_prepub.pdf)，
一般 serving 依据可参考 [Splitwise](https://arxiv.org/abs/2311.18677)。

本文不宣称复现这些论文的硬件、模型或绝对 RPS。只保留一个低于容量和一个超过容量的负载点：

```text
load factor ∈ {75%, 125%}
```

- 复用已经完成的 QMSum `WF-Cache2` 三次 saturated closed-loop calibration。
- parent-60 workload 的三次 completion slope 中位数为 `0.0774826203 sessions/s`。
- 75% 和 125% 的绝对到达率分别为 `0.0581119652` 和 `0.0968532754 sessions/s`。
- calibration 是开放到达率的冻结输入，不进入正式 trial 数和论文性能对比。
- 每个开放 trial 使用固定 24 个 session；到达结束后继续 drain，直到全部完成或显式失败。
- `LG-Batch` 与 `WF-Cache` 在相同 load factor、repetition 下共享完全相同的 arrival timestamps。
- 结果同时报告 parent capacity load factor 和绝对 sessions/s，避免把它误解为 24 条子集的
  重新标定容量。

25%、50% 主要受冷启动主导，100% 对 calibration 波动敏感，三者不进入正式矩阵。QMSum
`WF-FIFO` 和 `WF-History` 不运行 open-loop；它们不能为两个保留负载点增加独立论文主张。

MBPP 不运行 open-loop，也不运行 `WF-Cache1/2/3` capacity calibration。MBPP burst 已经覆盖
模型切换、资源预算和策略差异；增加开放负载会显著扩大矩阵，但不能支撑新的核心结论。

## 正式实验矩阵

修订版矩阵固定为 15 个实验单元，每个单元重复 5 次：

| 场景与论文证据 | 实验单元 | 计算 | Trials |
|---|---|---:|---:|
| QMSum burst 主性能 | `LG-Batch`、`WF-FIFO`、`WF-Cache` | 3×5 | 15 |
| QMSum 机制消融 | `WF-Cache(seq1,q16)`、`WF-Cache(seq3,q1)` | 2×5 | 10 |
| QMSum open-loop | `LG-Batch`、`WF-Cache` × 75%、125% | 2×2×5 | 20 |
| MBPP burst 资源曲线 | `LG-Batch-g3`、`WF-Cache-g1/g2/g3` | 4×5 | 20 |
| MBPP 2-GPU 策略 | 新增 `WF-FIFO-g2`、`WF-History-g2` | 2×5 | 10 |
| 总计 | 15 个实验单元 | 15×5 | 75 |

按 workload 汇总：

| 场景 | burst | open-loop | 合计 |
|---|---:|---:|---:|
| QMSum | 25 | 20 | 45 |
| MBPP | 30 | 0 | 30 |
| 总计 | 55 | 20 | 75 |

每个 trial 有 24 个 workflow session，共 1,800 个正式 session，其中 QMSum 1,080 个、
MBPP 720 个。相对旧方案的 24,000 个正式 session 减少 92.5%。已有 QMSum calibration、
初始化检查和 parent-60 preflight 不包含在该数量中。

## Synthetic prediction cache 边界

**已确认**：本轮缓存是外部提供的测试输入，不是 GNN 输出，也不作为 empirical profile。

- 三种 WF 策略都读取冻结 cache，用于固定请求可行性、GPU 选择、显存门禁和精确
  batch 准入。
- `WF-FIFO` 按 acquire 到达顺序处理，不用缓存耗时建立运行完成 ETA，也不生成
  near-ready prefetch。
- `WF-History` 只用于 MBPP 2-GPU 正式对比；已有观测时使用当前 trial 的加载和执行
  历史估计 ETA、预取和 reload cost，并在 trial 间清空。缺少加载历史时不做 near-ready
  prefetch，ready 请求仍使用冻结 cache 完成放置。
- `WF-Cache` 使用冻结 cache 的加载和执行耗时做 ETA、预取和 workflow-aware 驱逐，
  不用当前 trial 结果修正缓存值。
- cache 覆盖正式 workflow 需要的 model、GPU、batch、input 和 output buckets，并随实验
  固化生成方式、完整文件和 hash。
- cache 值只驱动调度决策，不作为预测准确率结果，也不与实际运行值计算预测误差。
- 调度器使用覆盖固定请求的最小 bucket，但不得修改实际输入或 `max_new_tokens`。
- 缺少 batch 1 覆盖条目时显式失败；缺少更高 batch 条目时只表示该并发组合不可准入。

`src/gnn_model` 接入、GNN checkpoint、训练/测试划分、校准和 predictor-level 实验全部不在
本轮范围内。未来接入时另行扩展本文，不在当前实现中预留额外策略分支。

## Trace、遥测与统计

当前 `workflow_trace.jsonl` 已通过 storage 与 trace contract tests，具备连续 `event_seq`、
统一 `run_id` 和 item/task/acquire/replica 标识，可还原核心运行时因果链。正式实验沿用该事件
模型，不重新设计 trace。

每个 trial 至少保存：

- session submit/start/finish/fail 时间和端到端 latency。
- node ready、queue enqueue/dequeue、queue blocked 和 fan-in ready 时间。
- acquire request/grant、granted batch size、configured `max_new_tokens` 和等待原因。
- replica load/start/finish、reuse、prefetch、evict 和 shutdown。
- inference start/finish、input/output tokens、finish reason、queue time 和 TTFT。
- vLLM version、engine mode、attention backend、max model length、KV block 数与 block size。
- GPU memory、utilization 和 power 的固定周期采样。

`item_emitted` 到 `item_enqueued` 的时间差作为 queue put blocked duration，不在 trace 中重复
存储派生值。队列占用和 GPU telemetry 作为带时间戳的 sidecar；LangGraph baseline 的事件
在分析前归一化到相同的 session/node/engine 时间语义。

统计约束：

- workload makespan 从 `run_started` 计到最后一个 session terminal；模型清理到
  `run_finished` 的时间单独报告，不混入吞吐分母。
- active GPU time 对同一 replica 的推理区间取并集，避免 continuous batching 重复计时。
- loading、active、idle-resident 和 evicting 分开统计。
- pipeline bubble 是 workload 窗口内 resident replica 时间减去与推理区间交集后的 GPU·s，
  ratio 以同一窗口的 resident GPU·s 为分母。
- GPU power 按时间积分为能耗，并同时保留未经结论化处理的原始采样。
- trial 级结果报告 5 次重复的均值、标准差和 95% confidence interval。
- session latency 另报每个 trial 内的 p50、p95 和 max。
- 质量按固定 24 条 sample IDs 做 paired comparison。

论文主指标只保留能够回答研究问题的部分：

- QMSum：makespan、sessions/min、p95、first completion、pipeline bubble、queue blocked 和
  batching/inflight。
- MBPP：makespan、sessions/min、p95、GPU seconds、resident VRAM-GiB·s、load、reuse、
  prefetch 和 eviction。
- 两个场景的质量结果只作为空输出、截断、执行失败或运行时语义偏差的 guardrail。

TTFT、stage latency、原始 power、能耗和完整错误分类继续随同一 trial 采集，但只作为诊断或
附录材料，不为这些指标增加独立实验。

### 结果呈现与论文复用

结果文档按论文复用顺序组织：先报告绝对值、基于 n=5 独立 trial 的 Student-t 95% CI、同一
repetition 的配对效应、机制伴随指标变化与质量结果，再后置工程运行状态、失败归档和 artifact
hash。固定 24 个 sample IDs 上的质量 CI 仅表示重复运行波动，不解释为数据集抽样不确定性，
也不外推为 QMSum 或 MBPP 的数据集级质量。

阶段性结果必须注明已完成 trial、场景、策略和负载覆盖，既不外推到未运行配置，也不自动生成
因果性或“胜负”结论。只有某个实验单元的 5 次重复全部完成且输入 hash 对齐，才生成该单元
的正式组统计；不得用 session 数代替缺失的 trial 重复。

## 可复现产物

正式实现需要固化：

- QMSum/MBPP 精确 sample manifest、分层信息和 hash。
- 每次 repetition 的 burst permutation。
- QMSum 每个 load factor 和 repetition 的 arrival trace。
- workflow、runtime、serving 和 scheduler 配置快照。
- synthetic prediction cache、生成 manifest 和 hash；不能只引用被忽略的 `tmp/` 路径。
- git commit、dirty diff hash、Python/vLLM/Transformers/Ray 版本。
- GPU UUID、驱动版本、主机名、实验开始时显存状态和 seed。
- 原始 session results、workflow trace、GPU telemetry、run summary 和质量结果。

原始产物固定布局：

```text
output/workflow/experiments/system_20260713_v2/
  prepared/
  calibration/
  trials/<trial_id>/
  analysis/
```

trial ID 必须由场景、策略、GPU 数、load factor、参数网格和 repetition 唯一决定。runner
发现一个 trial 的完整 manifest 与完成标记时跳过；发现残缺目录时显式失败，不覆盖原始数据。

旧 `system_20260713` 目录中的 60-session trial 只作为工程试跑归档。不能事后从其输出中抽取
24 个 session 计算修订版系统指标，因为原运行的排队、batching、fan-in 和资源竞争由全部
60 个 session 共同形成。修订版分析必须拒绝旧 experiment ID、旧 sample hash 和旧矩阵产物。

## 运行预算与停止规则

- 75 个正式 trial 顺序运行，不在同一主机并行多个 formal trial；不同 GPU 上的并行 trial
  仍会共享 CPU evaluator、模型文件 I/O 和主机资源。
- 预计完整运行需要 18 至 22 小时；该数字是运行预算，不是性能结论。
- 单个 trial 的 hard timeout 为 60 分钟，完整 campaign 的 wall-clock 上限为 24 小时。
- 只允许基础设施故障最多重跑一次；性能超时本身是显式失败，不自动重跑。
- 先完成一个论文证据组的全部配对策略和 5 次重复，再进入下一组；组内使用冻结种子做
  repetition-blocked 确定性交错，减少时间漂移。
- 24 小时内未完成时按 partial experiment 报告，不根据已观察结果临时删除慢组或补跑有利组。
- 每个正式 trial 前确认 GPU 空闲，结束后确认 Ray、vLLM 和 telemetry 已释放。

## 固定 workflow 业务语义

**已确认**：workflow 配置拥有模型、prompt、图结构、路由、采样参数和固定的
`max_new_tokens`。系统只决定请求何时执行、在哪里执行以及如何并发，不根据资源状态改变
业务请求。

- `max_new_tokens` 作为正整数放入节点的 `ExecutionConfig`。
- 删除 `TokenBudgetConfig` 范围、upscale/downscale 决策和 granted token budget。
- 旧配置中的 `token_budget` 显式迁移为 `execution.max_new_tokens`，不保留兼容分支。
- 暂时没有资源时保持 pending；固定请求在所有候选部署上都不可行时显式失败。
- prediction bucket 只用于保守估计；实际 serving 始终使用配置值。
- `hit_token_limit` 只进入 trace 和质量结果，不触发系统自动调参。
- 不设置动态 token budget 或人工 KV 配额消融，也不为该机制增加 trial。

该 schema、调度、trace 和测试调整已经完成。修订版不得为缩短运行时间而重新启用动态 token
budget、降低 `max_new_tokens`、减少 QMSum chunk 数、跳过 MBPP repair 或改为 warm start。

## 实现时固化的工程参数

以下参数使用单一固定值并写入配置和 manifest，不扩展为新的实验维度：

- QMSum capacity 的 steady-state completion slope 取完成序列的 20% 至 80%。
- GPU telemetry 的采样周期和空载能耗扣除方式。
- FIFO、History 和 Cache 的公开配置字段。
- synthetic cache 的确定性生成方式和 schema 校验。

历史 offline pipeline oracle 与 active-frontier lower bound 只保留为动机，不重新加入正式矩阵。

GPU telemetry 周期固定为 0.2 秒。每张 GPU 在模型加载前同步采集第一条样本，能耗和显存
派生结果同时保留原始值与基于该样本的非负空载扣除值。完整原始采样不被派生分析覆盖。

修订版实验的统一入口为：

```bash
PYTHONPATH=src uv run python -m experiment.workflow prepare
PYTHONPATH=src uv run python -m experiment.workflow list
PYTHONPATH=src uv run python -m experiment.workflow run --all --gpu-ids 0,1,2,3
PYTHONPATH=src uv run python -m experiment.workflow analyze
```

`prepare` 必须生成 24+24 正式 sample、75-trial matrix 和 10 条 QMSum arrival traces，并
引用已完成的 QMSum capacity artifact。修订版不执行 `preflight` 或 `calibrate --all`；
`prepare` 负责校验复用产物的配置与 hash。完整 manifest 与完成标记存在时跳过；残缺目录
或 hash 不一致时直接失败，不覆盖原始数据。

## 实施状态

- [x] 阅读 Phase 1、历史动机方案、结果和当前 workflow 实现。
- [x] 固定 QMSum/MBPP 场景、拓扑、模型和质量 guardrail。
- [x] 将正式 workload 固定为每个场景 24 条，short、medium、long 各 8 条。
- [x] 将正式矩阵固定为 75 trials、1,800 sessions 和五次独立重复。
- [x] 删除 MBPP open-loop、MBPP capacity calibration、QMSum 三个冗余负载点和 3×3 全网格。
- [x] 完成 Qwen3-4B/8B/14B 在目标 V100 context 下的初始化与 KV 容量预检。
- [x] 固定 workflow 业务语义与系统调度职责边界。
- [x] 确认本轮排除 GNN，使用冻结 synthetic cache 验证调度机制。
- [x] 确认现有 runtime trace contract，不新增重复的派生事件。
- [x] 创建持续维护的实验方案文档。
- [x] 用 `ExecutionConfig.max_new_tokens` 替换动态 `TokenBudgetConfig` 并更新测试。
- [x] 固化 sample、arrival、synthetic cache 和环境 manifests。
- [x] 实现通用 LangGraph vLLM baseline。
- [x] 实现 FIFO/History/Cache 策略配置，不加入数据集特判。
- [x] 修正 batch cache 缺失语义和 active GPU time 统计。
- [x] 补齐 trial manifest、LangGraph trace 归一化、队列占用、KV capacity 和 GPU telemetry sidecar。
- [x] 实现统一 experiment runner、resume、analysis 和结果文档生成。
- [x] 完成固定 60+60 样本和保守代表性中间输出的 prompt/token preflight。
- [x] 完成 QMSum 2-GPU 的 3 次 capacity calibration，冻结容量为
  `0.0774826203 sessions/s`。
- [x] 固化 QMSum/MBPP 24 条正式 sample manifests 与五种 burst permutations。
- [x] 将 config、trial plan、prepare、report 和测试从旧 60/400 口径同步到 24/75 口径。
- [x] 实现 60 分钟 trial timeout、单次基础设施重跑、确定性交错顺序和 24 小时 campaign 上限。
- [x] 使用 `system_20260713_v2` 新目录引用 QMSum calibration，生成 10 条 arrival traces。
- [x] 将旧 `system_20260713` 正式 trial 归档为工程试跑，不纳入修订版分析。
- [x] 运行 75 个正式 trial，完成对齐汇总和结果文档。
