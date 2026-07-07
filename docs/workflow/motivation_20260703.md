# 20260703 workflow motivation 实验方案

## 目标

本文只规划 motivation 实验，不实现新的调度系统。实验聚焦两个已有数据集和两类
workflow：

| workflow | 数据集 | 目标现象 |
|---|---|---|
| fan-out/fan-in | QMSum | 粗粒度 DAG barrier 导致 chunk 节点 idle，跨样本流水线存在理论加速空间 |
| chain | MBPP sanitized | 连续生成、测试、修复链路中，静态模型驻留高于实际活跃 frontier 所需资源 |

这两个实验统一使用 LangGraph native DAG 作为 baseline。该 baseline 表示已有
workflow-aware serving 系统和 LangGraph 都能表达 workflow 结构，但不显式管理本地模型
实例加载、复用、预取、卸载和驱逐的场景。实验不声称复现 SAGA、Kairos、Murakkab、
Parrot 或 Grape，只复用它们常见的 motivation 写法：先选代表性 workflow，再量化
request/task 级执行抽象造成的 idle、排队、缓存或资源浪费。

## 相关论文的动机实验参考

这些系统论文的 motivation 或主实验通常采用代表性 workflow 与采样/trace-driven workload。
只有少数实验明确跑了某个 benchmark 的完整公开 split；多数 motivation 图只给 request rate、
trace 时间窗或随机采样数。

| 论文 | 动机实验或最接近动机的实验 | 模型 | 数据规模 | 是否全量 |
|---|---|---|---|---|
| SAGA | Figure 1 用 SWE-bench coding agent workload 量化 KV cache regeneration、HBM 利用率和端到端延迟 | Llama-3-70B-Instruct | Figure 1 写明 32 A100、10 trials、SWE-bench，但未说明该图具体任务数；主实验写明 SWE-bench Verified 500 任务、WebArena full 812 任务、BurstGPT synthetic 10 tenants | 主实验中 WebArena 是 full 812；SWE-bench 是 Verified 500 子集，不是完整 SWE-bench；motivation 图任务数未明说 |
| Kairos | Section 2.1/2.2 分析 agent 输出长度、推理延迟、排队和 preemption；使用 QA/RG/CG 混部 workload | Llama3-8B；扩展实验用 Llama2-13B | Survey 30 个 GitHub multi-agent 项目；动机分析用 G+M/TQ/HE，三组扩展为 G+M/TQ/HE、M+W/NCD/MBPP、S+S/NQ/APPS；到达时间来自生产 trace，图中出现 8 req/s | 未说明每个数据集跑多少样本；不是声明全量 benchmark |
| Murakkab | Section 2.5 用 Video Q/A 和 Code Generation 的配置空间说明 accuracy/token/cost trade-off；Section 4 用 trace 驱动资源优化 | Video Q/A: Gemma-3-27B、Llava-OneVision-7B、NVLM-D-72B；Code: DeepSeek-Qwen-32B、Gemma-3-27B、Phi-4、NVLM-D-72B | 主实验用 Azure LLM serving trace 的 24 小时子集，chat 请求映射到 Video Q/A，coding 请求映射到 Code Generation；附录给常见配置表 | 不是全量公开数据集；是 24 小时 trace 子集 |
| Parrot | Section 3 motivation 用生产 chain-style 应用和 map-reduce summary 说明 request-centric API 的开销；Section 8.2 做长文档实验 | 长文档 chain/map-reduce 用 LLaMA 13B；mixed workload 用 4 个 LLaMA 7B engine；MetaGPT 用 LLaMA 13B | 长文档实验从 Arxiv-March 随机选 10 篇 20k token 以上文档；Figure 13 是 25 个 chain-summary applications；Bing Copilot 合成 64 requests；mixed workload 叠加 1 req/s chat 与 map-reduce | 明确不是全量 Arxiv-March；是随机 10 篇和合成/并发 workload |
| Grape | Section II-B 在 long document summarization 上展示 TFLOPs 分布和 agentic latency spike；Section IV-A 评估 code/summary/search | Llama3-8B、Qwen3-14B | Code 输入来自 LiveCodeBench，summary 用 arxiv papers，search 用 HotpotQA；图 12/13/14 按 request/s 扫描，未给每个数据集样本数 | 未声明全量；更像 serving workload 下的采样/请求流实验 |

对本文最有参考价值的是 Parrot 和 Grape：它们都把长文档摘要作为 map-reduce 或 chain
workflow；Grape 还明确使用 Llama3-8B 和 Qwen3-14B 这样的中等规模开源模型，而不是统一换成
最大模型。Murakkab 也说明 code generation 对模型能力更敏感，会在 DeepSeek-Qwen-32B、
Gemma-3-27B 和 Phi-4 之间切换，而不是默认小模型足够。

## 本文模型与样本规模

本地可用模型包含 `Qwen3-4B`、`Qwen3-8B`、`Qwen3-14B` 和 `Qwen3-32B`。已有实验显示：
QMSum 的 `Qwen3-4B` chunk 与 `Qwen3-8B` merge 可以支撑任务；MBPP 的 weak workflow 质量不足，
strong workflow 也仍只是在小样本上接近 direct baseline。因此两个 workflow 的模型选择应分开：

| workflow | 节点 | 默认模型 | 原因 |
|---|---|---|---|
| QMSum fan-out/fan-in | `chunk_*` | Qwen3-4B | 局部摘要对模型能力要求低于全局代码生成，已有 2-way/3-way 结果可用 |
| QMSum fan-out/fan-in | `merge` | Qwen3-8B | merge 需要综合多个局部答案，但 8B 已能支撑当前 QMSum pilot |
| QMSum fan-out/fan-in | `judge` | 外部 judge 或离线 evaluator | 不计入被调度 workflow 的本地加速卡资源 |
| MBPP chain | `coder` | Qwen3-14B | 4B/8B 在 MBPP 上能力风险过高，14B 是本地可用的最小强候选 |
| MBPP chain | `tester` | CPU deterministic | 直接使用 `evaluate_mbpp`，不占用本地 LLM 实例 |
| MBPP chain | `repair` | Qwen3-14B | 修复节点需要理解失败断言和已有代码，默认不再使用 4B/8B |
| MBPP chain | `final_tester` | CPU deterministic | 只做 pass@1 判定 |

`Qwen3-32B` 只作为 direct quality reference 或能力上限，不进入 motivation 主 workflow。
这样可以避免“一个大模型解决所有节点”的偷懒设置，同时保证 MBPP chain 不是因模型太弱而失效。

样本规模采用代表性采样：

| 数据集 | 本地规模 | motivation 规模 |
|---|---:|---:|
| QMSum `ALL/test` | 281 条 query | 60 条 seed42 分层样本，short、medium、long 各 20 条 |
| MBPP sanitized `test` | 257 条任务 | 60 条 seed42 分层样本 |

动机实验的目标是证明调度抽象产生 idle 和资源驻留浪费，不是刷新 benchmark 分数。因此不建议首轮全量跑
QMSum 或 MBPP。更合理的做法是先用小而稳定的采样集做 trace，再用相同 trace 离线计算
pipeline oracle 和 active-frontier oracle。

## 实验 A：QMSum fan-out/fan-in idle

### 数据集

使用 `src/dataset/summarization.py::load_qmsum_split` 读取 QMSum `ALL/test` split。
样本采用 60 条 seed42 分层样本，按输入长度分成 short、medium、long 三层，每层 20 条。
已有 24 条分层样本只作为耗时估算和历史对照：

```text
output/workflow_parallelism_trace_20260628/sample_ids/qmsum_24_seed42.jsonl
```

样本字段：

| 字段 | 用途 |
|---|---|
| `input_text` | 会议 transcript，用于切分 chunk |
| `metadata.query` | query-focused summarization 的查询 |
| `gold_answer` | ROUGE 和 LLM judge 的参考答案 |

### Workflow

使用 chunk -> merge 范式：

```text
input
  -> chunk_0, chunk_1, ..., chunk_k
  -> merge
  -> judge
  -> output
```

首轮只保留 `k=2` 和 `k=3`，直接复用 20260626 已有结果。后续如需要更强
fan-in 压力，再扩展到 `config/workflow/summarization_compare_20260621/qmsum_strong_parallel_6way.yaml`。

### Baseline 与 oracle

| 组别 | 含义 |
|---|---|
| `langgraph_native` | 每个样本内部按 DAG barrier 执行；merge 等所有 chunk 完成后启动 |
| `pipeline_oracle` | 离线 trace 模拟：chunk worker 在 merge 当前样本时继续处理后续样本 |

`pipeline_oracle` 不改变单个样本的 fan-in 语义，只回收跨样本流水线机会。

### Pipeline idle 定义

本文把 pipeline idle 定义为：

```text
一个 worker、模型实例或拓扑 stage 已经完成当前工作，并且在流水线执行语义下存在可继续推进的数据，
但由于粗粒度 DAG barrier、session 级阻塞或静态调度策略，不能开始下一项工作而被迫空闲的时间。
```

这个定义对应流水线系统中的 stall 或 bubble：不是节点本身计算慢，而是执行语义让可用执行单元
处于等待状态。它可以完全由 trace 后处理得到，不需要在运行时额外引入复杂探针。

对任意可复用执行单元 `r`，设它完成一次工作区间的时间为 `end(r, i)`，下一次真正开始工作的时间为
`start(r, i+1)`。如果在 `[end(r, i), start(r, i+1))` 内已经存在按流水线语义可执行的下一项工作，
但 baseline 没有调度它，则这段时间计入 `pipeline_idle_time(r)`。

对 QMSum map-reduce summary，具体表现为：

```text
同一个 session 内，较快 chunk agent 先完成局部摘要；
merge agent 仍必须等待最慢 chunk agent；
较快 chunk agent 在 merge 启动或当前 session 释放前不能继续处理下一个 session 的 chunk。
```

因此，QMSum 中的 pipeline idle 可以从每个 session 的 chunk 节点 trace 计算：

```text
idle(chunk_i, session_s) = release_time(session_s) - end(chunk_i, session_s)
```

其中 `release_time(session_s)` 取 baseline 允许该 chunk worker 继续处理下一 session 的最早时间。
在严格 session-at-a-time baseline 中，它可以取当前 session 完全结束时间；在只分析 fan-in barrier
时，它可以取 merge 启动时间。论文主文只报告统一的 `pipeline_idle_time` 和利用率，附录或脚注再说明
QMSum 中的 release time 选择。

### 指标口径

| 指标 | 定义 |
|---|---|
| `pipeline_idle_time` | 可复用执行单元因粗粒度 DAG barrier 或 session 阻塞产生的空闲时间 |
| `stage_utilization` | stage busy time / (stage busy time + pipeline idle time) |
| `pipeline_bubble_ratio` | pipeline idle time / (stage busy time + pipeline idle time) |
| `end_to_end_total_sec` | baseline 完成全部样本的总生成耗时 |
| `oracle_total_sec` | pipeline oracle 完成全部样本的模拟总耗时 |
| `speedup_upper_bound` | end_to_end_total_sec / oracle_total_sec |
| `quality` | ROUGE-L、LLM score、pass rate |

### 判定标准

实验成立需要同时满足：

- `langgraph_native` 存在非零 `pipeline_idle_time`。
- `pipeline_bubble_ratio` 在 mean 或 p95 上明显高于 0，说明粗粒度执行确实留下流水线空泡。
- `pipeline_oracle` 相比 `langgraph_native` 有明显理论加速比。
- 摘要质量不作为优化目标，只用于确认 workflow 仍是有效任务。

已有结果已经可以重算成该统一口径：20260626 中记录的 chunk 节点开始/结束时间足以计算
`pipeline_idle_time`、`pipeline_bubble_ratio` 和 pipeline oracle 理论加速比。原文中的
chunk 等待时间、merge 等待时间只作为 QMSum 场景下的中间计算量，不作为论文主指标命名。

### 术语参考

`pipeline_idle_time`、`pipeline_bubble_ratio` 和 `stage_utilization` 借鉴传统流水线系统里的
stall、bubble、throughput 和 utilization 口径。Pipelining 的核心收益来自多个任务在不同
stage 上重叠执行；stall 或 bubble 会降低吞吐，stage 不均衡和 fill/drain 时间也会降低理论加速比。
本文不直接套用 CPU pipeline 的周期级模型，只复用其“可执行单元因依赖或调度约束空转”的度量思想。

## 实验 B：MBPP chain 模型驻留资源浪费

### 数据集

使用 `src/dataset/mbpp.py::load_mbpp_samples` 读取 MBPP sanitized 数据集。首轮只使用
loader 判定为 `test` 的样本；当前本地文件中该 split 为 257 条任务。motivation 实验使用
60 条 seed42 分层样本。

样本字段：

| 字段 | 用途 |
|---|---|
| `input_text` | 编程题 prompt |
| `test_imports` | 执行测试前导入 |
| `test_list` | pass@1 测试断言 |
| `reference_code` | 只用于离线诊断，不进入 prompt |

质量评估使用 `src/dataset/mbpp.py::evaluate_mbpp`。

### Workflow

使用固定 chain：

```text
input
  -> coder
  -> tester
  -> repair
  -> final_tester
  -> output
```

节点定义：

| 节点 | 类型 | 模型 | 说明 |
|---|---|---|---|
| `coder` | LLM | Qwen3-14B | 根据 prompt 生成 Python 函数 |
| `tester` | deterministic | CPU | 运行 `evaluate_mbpp`，生成失败类型和失败断言 |
| `repair` | LLM | Qwen3-14B | 根据原题、代码和测试反馈修复 |
| `final_tester` | deterministic | CPU | 重新运行 `evaluate_mbpp` |

首轮为了稳定触发 chain，所有样本都经过 `repair`，不因为 `tester` 通过而提前结束。
后续可增加 dynamic branch 版本，但不进入本轮 motivation。

### Baseline 与 oracle

| 组别 | 含义 |
|---|---|
| `workflow_static` | workflow 开始前加载 `coder` 和 `repair` 需要的所有本地模型实例，直到样本结束才释放 |
| `active_frontier_oracle` | 离线 trace 下，只在 LLM 节点实际执行区间统计其模型实例为 resident |

如果 `coder` 和 `repair` 使用同一个模型规格，但作为两个 LangGraph 节点各自持有模型实例，
`workflow_static` 仍按两个实例计入。这是 motivation baseline，用于暴露缺少模型实例复用和
生命周期管理时的资源上限。

### 指标

| 指标 | 定义 |
|---|---|
| `resident_model_count_peak` | 任意时刻 resident LLM 模型实例数量峰值 |
| `active_model_count_peak` | 任意时刻真实执行 LLM 节点数量峰值 |
| `resident_model_seconds` | resident 模型实例数量对时间积分 |
| `active_model_seconds` | active LLM 执行数量对时间积分 |
| `resource_gap` | resident_model_seconds / active_model_seconds |
| `frontier_gap` | resident_model_count_peak / active_model_count_peak |
| `node_idle_time` | 模型 resident 但节点未执行 LLM 的时间 |
| `pass_at_1` | final_tester 通过率 |

### 判定标准

实验成立需要满足：

- `workflow_static` 的 `resident_model_count_peak` 高于 `active_model_count_peak`。
- `resource_gap` 明显大于 1，说明静态驻留显著高于真实活跃 frontier。
- `node_idle_time` 主要集中在 chain 上下游等待区间，而不是测试函数执行本身。
- `pass_at_1` 只用于确认 chain workflow 是有效编程任务，不作为本轮调度优化目标。

## 统一 trace 字段

两个实验都记录同一套最小 trace，便于后续复用：

| 字段 | 说明 |
|---|---|
| `workflow_name` | workflow 名称 |
| `sample_id` | 数据样本 ID |
| `node_name` | 节点名 |
| `node_type` | `llm`、`deterministic`、`input`、`output` |
| `started_at` | 节点执行开始时间 |
| `ended_at` | 节点执行结束时间 |
| `duration_sec` | 节点执行耗时 |
| `ready_at` | 按数据依赖计算的节点最早可执行时间 |
| `release_at` | baseline 允许同一 worker 处理下一项工作的时间 |
| `pipeline_idle_time` | 由 `ready_at`、`release_at` 和实际执行区间离线计算得到 |
| `model_name` | LLM 节点使用的模型名 |
| `model_instance_id` | baseline 中的模型实例 ID |
| `resident_started_at` | 模型驻留开始时间 |
| `resident_ended_at` | 模型驻留结束时间 |
| `input_token_count` | LLM 输入 token 数 |
| `output_token_count` | LLM 输出 token 数 |
| `status` | `ok`、`oom`、`timeout`、`empty_output`、`eval_fail` |
| `quality_result` | ROUGE、judge 或 MBPP 测试结果 |

## 输出目录

建议统一输出到：

```text
output/workflow_motivation_20260703/
```

建议文件：

```text
output/workflow_motivation_20260703/qmsum_fan_in_trace.jsonl
output/workflow_motivation_20260703/qmsum_fan_in_summary.json
output/workflow_motivation_20260703/mbpp_chain_trace.jsonl
output/workflow_motivation_20260703/mbpp_chain_summary.json
```

## 论文表述口径

推荐结论写法：

```text
These motivation experiments do not reproduce prior systems. Instead, they use
representative workflow patterns from prior work to expose two limitations of a
LangGraph-style baseline without local model-instance lifecycle management:
coarse-grained DAG barriers leave pipeline parallelism unused, and static model
residency consumes more accelerator resources than the active workflow frontier.
```

避免写法：

```text
SAGA、Kairos、Murakkab、Parrot 都无法运行这些 workflow。
```

更准确的边界是：这些系统证明了 workflow-aware serving 的价值，但它们的 motivation
重点分别落在 request-level API、KV cache、memory-aware dispatch、profile-guided
配置优化或 micro-task 并行上；本项目补充的是本地多模型 workflow 的模型实例生命周期
管理动机。
