# 20260626 workflow 动机实验方案

## Agent工作

agent 只能修改实验内容和实验结果部分，其他内容全部禁止修改。

## Langgraph 端到端耗时实验

### 实验数据集

从 QMSum test 数据集中按输入长度分层采样 18 条数据，short、medium、long 各 6 条，随机种子固定为 42。

### 并行度实验

#### 实验描述

实验在不同并行度下，各数据项的端到端耗时，汇总总耗时，计算平均耗时。用于证明在该类场景中，将文本切片并行，可以显著降低端到端耗时。

##### 实验设置 1

使用 qwen-32b 大模型，占满所有显卡，直接处理任务

##### 实验设置 2

使用 3个 qwen-4b 小模型，分切片并行处理任务，每个小模型占用一张显卡，显卡数量为 3。merge agent 单独使用一个 qwen-8b 模型，占用一张显卡负责合并所有小模型的输出，生成最终摘要。

每个实验项必须记录：

- 端到端耗时：total、mean、p50、p95。
- 摘要质量：ROUGE 或 LLM judge pass rate。
- token 统计：输入 token、输出 token、merge prompt token。
- 失败统计：OOM、截断、空输出、judge fail。

实验的 workflow 为
``` text
input -> dispatcher -> chunk agents -> merge agent -> judge -> output
```

其中 dispatcher 负责把一条会议记录按并行度切分成多个 chunk，并把每个 chunk 分发给对应的 chunk_* agent。dispatcher 不使用 LLM，不参与文本生成，只负责确定性的任务拆分和分发。

chunk_* agents 负责分别阅读各自的文本片段，并生成面向 query 的局部摘要。不同并行度对应不同数量的 chunk agent，例如并行度为 1 时只生成 1 个 chunk，并行度为 4 时生成 4 个 chunk。每个 agent 单独占用一张卡。

merge agent 负责收集所有局部摘要，去重、整合并直接生成最终摘要。merge 阶段同时承担原先 refine 的职责，因此 workflow 中不再额外设置 refine agent。

judge 只负责在最终摘要生成后进行质量评估，不参与摘要生成流程。端到端生成耗时只统计 dispatcher -> chunk agents -> merge agent，不包含 judge 的评分耗时。

本实验比较不同 chunk agent 数量下的生成耗时和摘要质量，用于观察文本切片并行是否能够降低摘要生成耗时，以及切分过细是否会导致上下文丢失、合并压力增大或摘要质量下降。

#### 实验内容

本轮把并行度实验、节点耗时分析实验、粗粒度 DAG 等待损失实验合并到一次带 trace 的正式实验中完成。

旧目录 `output/workflow_parallelism_20260626/` 中的结果因为大量命中输出上限，不再用于结论。上一轮 `output/workflow_parallelism_redo_20260628/` 保留作历史对照，本轮正式目录为：

``` text
output/workflow_parallelism_trace_20260628/
```

样本使用 QMSum test 的 24 条分层样本，short、medium、long 各 8 条，随机种子为 42。样本文件为：

``` text
output/workflow_parallelism_trace_20260628/sample_ids/qmsum_24_seed42.jsonl
```

正式实验项为：

| 实验项 | workflow | 模型与显卡 | 输入上限 | 输出上限 |
|---|---|---|---:|---:|
| 32B 直连 | input -> summarizer -> judge -> output | Qwen3-32B 使用 cuda:0/1/2/3 | 4096 | 1024 |
| 切成 2 份 | input -> 2 个 chunk agent -> merge -> judge -> output | 两个 Qwen3-4B chunk agent 使用 cuda:0/1，Qwen3-8B merge 使用 cuda:3 | chunk 8192, merge 4096 | 1024 |
| 切成 3 份 | input -> 3 个 chunk agent -> merge -> judge -> output | 三个 Qwen3-4B chunk agent 使用 cuda:0/1/2，Qwen3-8B merge 使用 cuda:3 | chunk 8192, merge 4096 | 1024 |

dispatcher 不是独立 LLM 节点，而是 workflow 运行器中的确定性切块逻辑。它不生成文本，也不占用 GPU。

正式运行前使用 24 条样本中的最长样本 `test_covid_9_specific_003` 做真实 GPU 上限预检。该样本文本约 12.7 万字符。

| 预检项 | 结果 |
|---|---|
| 32B direct 6144/1024 | OOM |
| 32B direct 5120/1024 | OOM |
| 32B direct 4096/1024 | 成功，最大输出 13 token |
| 2 chunks 10240/1024 | OOM |
| 2 chunks 9216/1024 | OOM |
| 2 chunks 8192/1024 | 成功，最大输出 100 token |
| 3 chunks 10240/1024 | OOM |
| 3 chunks 9216/1024 | OOM |
| 3 chunks 8192/1024 | 成功，最大输出 186 token |

因此正式配置继续使用 `32B 4096/1024`、chunk `8192/1024`、merge `4096/1024`。本轮不把输入截断率作为自动废弃条件，只完整记录截断率、质量和耗时；输出上限命中、OOM、空输出、judge error 才视为异常。

正式配置文件为：

``` text
output/workflow_parallelism_trace_20260628/configs/qmsum_direct_32b.yaml
output/workflow_parallelism_trace_20260628/configs/qmsum_chunk_2parts.yaml
output/workflow_parallelism_trace_20260628/configs/qmsum_chunk_3parts.yaml
```

正式结果和日志为：

``` text
output/workflow_parallelism_trace_20260628/results/qmsum_direct_32b.jsonl
output/workflow_parallelism_trace_20260628/results/qmsum_chunk_2parts.jsonl
output/workflow_parallelism_trace_20260628/results/qmsum_chunk_3parts.jsonl
output/workflow_parallelism_trace_20260628/logs/
```

#### 实验结果

三组正式实验均完成 24 条样本，每个结果文件均包含 24 条 sample 记录和 1 条 summary 记录。三组都没有 CUDA OOM、没有空输出、没有 judge error、没有输出上限命中。每个 model/evaluator 节点都记录了 `started_at`、`ended_at` 和 `duration_sec`。

主耗时不包含 judge 评分耗时。

| 实验项 | 样本数 | 主耗时 total | 主耗时 mean | p50 | p95 | ROUGE-L | LLM score | pass rate | 输入上限命中 | 输出上限命中 | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B 直连 | 24 | 731.547 | 30.481 | 29.833 | 44.912 | 0.1991 | 2.750 | 0.292 | 23 | 0 | 0 |
| 切成 2 份 | 24 | 669.934 | 27.914 | 28.067 | 39.464 | 0.1888 | 2.875 | 0.250 | 15 | 0 | 0 |
| 切成 3 份 | 24 | 721.839 | 30.077 | 30.536 | 41.956 | 0.1880 | 3.125 | 0.333 | 7 | 0 | 0 |

输入截断情况：

| 实验项 | 节点 | 调用数 | 输入上限命中 | 输入命中率 | 最大输入 token | 最大输出 token |
|---|---|---:|---:|---:|---:|---:|
| 32B 直连 | summarizer | 24 | 23 | 95.8% | 4096 | 180 |
| 切成 2 份 | chunk agents | 48 | 15 | 31.2% | 8192 | 204 |
| 切成 2 份 | merge | 24 | 0 | 0.0% | 483 | 250 |
| 切成 3 份 | chunk agents | 72 | 7 | 9.7% | 8192 | 181 |
| 切成 3 份 | merge | 24 | 0 | 0.0% | 581 | 442 |

节点耗时：

| 实验项 | direct mean | chunk stage mean | merge mean |
|---|---:|---:|---:|
| 32B 直连 | 30.479 | 0.000 | 0.000 |
| 切成 2 份 | 0.000 | 17.304 | 10.605 |
| 切成 3 份 | 0.000 | 17.313 | 12.756 |

结论：

- 32B 直连在 4 张 V100 上仍只能稳定使用 `4096/1024`。它的输入截断率为 95.8%，因此只能作为有限显存下的直连 baseline，不能代表充分读完整会议记录的质量上限。
- 切成 2 份的平均主耗时最低，为 27.914 秒；但 chunk 输入截断率仍有 31.2%。
- 切成 3 份的 chunk 输入截断率降到 9.7%，LLM score 和 pass rate 最高，但 merge 平均耗时高于切成 2 份。
- 本轮不能证明“切得越多一定越快”。更准确的结论是：切分能降低单个 chunk agent 的输入压力，但 merge 阶段和调度等待会抵消一部分收益。
- 输出上限 `1024` 对本轮足够，三组最大输出均低于 1024；上一次输出上限过低导致的实验异常已消除。

### 节点耗时分析实验

#### 实验描述

参照 config/workflow/summarization_compare_20260621/multi_news_strong_parallel_6way.yaml 的方案，设置合适并行度的 chunk agent 。

采集各 chunk agent 节点平均耗时情况，记录每次处理时，节点输入的 prompt 长度，输出的 summary 长度，以及处理耗时，采集各 chunk agent 节点的 idle 时间和 busy 时间。

用于分析，伴随不同输入长度，输出长度和节点耗时的关系；分析在 LangGraph 原生粗粒度并行策略下，节点的 idle 时间和 busy 时间的分布情况。

我还想要探索，使用节点历史 prompt 长度、节点历史输出长度、节点历史耗时等信息，是否可以应用于一个非机器学习、偏统计的方式，来预测节点的处理耗时。这个应该需要采集第 2 项的实验数据，来分析节点耗时和输入输出长度的关系。

#### 实验内容

本实验复用 `output/workflow_parallelism_trace_20260628/` 的同一批 24 条正式结果，不再单独重跑。

执行器在每个 model/evaluator 节点记录：

- `started_at`：节点相对当前样本开始的启动时间。
- `ended_at`：节点相对当前样本开始的结束时间。
- `duration_sec`：节点运行耗时。
- `input_token_count`、`output_token_count`：节点输入输出 token 数。

分析对象包括：

- 32B 直连的 summarizer 节点。
- 切成 2 份和切成 3 份的 chunk agents。
- 切成 2 份和切成 3 份的 merge 节点。

统计每类节点的调用数、输入 token 均值、输出 token 均值、平均耗时，以及 token 数和耗时之间的相关性。

#### 实验结果

节点耗时：

| 实验项 | 节点 | 调用数 | mean | p50 | p95 |
|---|---|---:|---:|---:|---:|
| 32B 直连 | summarizer | 24 | 30.479 | 29.831 | 44.911 |
| 切成 2 份 | chunk agents | 48 | 15.930 | 15.822 | 24.714 |
| 切成 2 份 | chunk stage | 24 | 17.304 | 17.969 | 24.714 |
| 切成 2 份 | merge | 24 | 10.605 | 11.534 | 17.223 |
| 切成 3 份 | chunk agents | 72 | 15.606 | 15.356 | 22.921 |
| 切成 3 份 | chunk stage | 24 | 17.313 | 16.534 | 24.255 |
| 切成 3 份 | merge | 24 | 12.756 | 13.282 | 21.593 |

token 与耗时关系：

| 实验项 | 节点 | 调用数 | input mean | output mean | duration mean | corr(input,duration) | corr(output,duration) | corr(input,output) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 32B 直连 | summarizer | 24 | 4087.1 | 103.0 | 30.479 | 0.206 | 1.000 | 0.208 |
| 切成 2 份 | chunk agents | 48 | 5889.8 | 116.4 | 15.930 | 0.782 | 0.769 | 0.222 |
| 切成 2 份 | merge | 24 | 325.5 | 149.8 | 10.605 | 0.769 | 1.000 | 0.770 |
| 切成 3 份 | chunk agents | 72 | 4496.2 | 107.9 | 15.606 | 0.632 | 0.772 | 0.057 |
| 切成 3 份 | merge | 24 | 425.9 | 180.8 | 12.756 | 0.665 | 1.000 | 0.662 |

结论：

- chunk agent 的输入 token 和耗时存在明显正相关：切成 2 份为 0.782，切成 3 份为 0.632。
- chunk agent 的输出 token 和耗时也存在明显正相关：切成 2 份为 0.769，切成 3 份为 0.772。
- chunk agent 的输入 token 和输出 token 相关性较弱：切成 2 份为 0.222，切成 3 份为 0.057。
- merge 的耗时和输出长度高度相关；切成 3 份后 merge 输入和输出都变长，因此 merge 平均耗时从 10.605 秒升到 12.756 秒。
- merge 的输入 token 和输出 token 存在较明显正相关：切成 2 份为 0.770，切成 3 份为 0.662。
- 仅靠增加 chunk 数不能保证端到端更快，因为 chunk 阶段省下的输入压力可能被 merge 阶段吃掉。

### 粗粒度 DAG 等待损失实验

#### 实验描述

参照实验2采集到的节点级 trace，分析 LangGraph 原生 parallel workflow 在 chunk fan-out 后等待最慢节点完成的损失。

当前 parallel workflow 中，`merge` 节点必须等待所有 `chunk_*` 节点完成后才能开始执行。如果不同 chunk 的输入长度、输出长度或生成耗时差异较大，较快完成的 chunk 节点会进入等待状态。这个实验用于量化这种 barrier 等待损失，并判断 Buffered Node 是否有继续实现的必要。

每个样本至少记录：

- 每个 `chunk_*` 节点的开始时间、结束时间和耗时。
- `chunk_max_duration`：最慢 chunk 的耗时。
- `chunk_wait_time`：所有 chunk 相对最慢 chunk 的等待时间总和。
- `merge_wait_time`：第一个 chunk 完成到最后一个 chunk 完成之间的时间差。
- `idle_ratio`：chunk 等待时间相对 chunk 总占用时间的比例。

如果等待损失在 mean 或 p95 上都较小，则 Buffered Node 的优先级应降低。如果等待损失明显，则继续用 trace 做 Buffered Node 的 oracle simulation。

#### 实验内容

本实验复用 `output/workflow_parallelism_trace_20260628/` 的同一批节点 trace。

对每个切分样本计算：

- `chunk_wait_time`：同一样本中较快 chunk 等待最慢 chunk 的时间总和。
- `merge_wait_time`：第一个 chunk 完成到最后一个 chunk 完成的时间差。
- `idle_ratio`：chunk 等待时间除以 chunk busy 时间和等待时间之和。
- `barrier_gap`：最后一个 chunk 完成到 merge 实际启动之间的调度空隙。

另外做一个离线流水线模拟：假设 chunk worker 可以继续处理后续样本，merge worker 只在某个样本的所有 chunk 完成后处理该样本。这个模拟只用于估算 Buffered Node 风格的跨样本流水线收益，本轮没有实现新的 Buffered Node runner。

#### 实验结果

等待损失：

| 实验项 | trace 样本数 | chunk wait mean | chunk wait p50 | chunk wait p95 | merge wait mean | idle ratio mean | barrier gap mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 切成 2 份 | 24 | 2.747 | 2.785 | 6.698 | 2.747 | 0.083 | 0.001 |
| 切成 3 份 | 24 | 5.124 | 4.310 | 12.081 | 3.501 | 0.098 | 0.001 |

跨样本流水线离线模拟：

| 实验项 | 当前主耗时 total | 当前 chunk+merge total | 流水线模拟 total | 理论加速比 |
|---|---:|---:|---:|---:|
| 切成 2 份 | 669.934 | 669.813 | 399.560 | 1.676 |
| 切成 3 份 | 721.839 | 721.675 | 395.680 | 1.824 |

结论：

- 两个切分 workflow 都存在 chunk 节点 idle。切成 2 份平均等待 2.747 秒，切成 3 份平均等待 5.124 秒。
- `barrier_gap` 接近 0，说明同一样本内最后一个 chunk 完成后，merge 基本会立刻启动；主要损失不是本地调度延迟，而是 fan-out/fan-in barrier 和样本间串行。
- 切成 3 份的等待损失更大，但它比切成 2 份慢的主要原因还包括 merge 更重：merge mean 从 10.605 秒升到 12.756 秒。
- 离线模拟显示，如果按 Buffered Node 的思路让 chunk worker 在 merge 处理当前样本时继续处理后续样本，24 条样本的生成阶段理论 total 可降到约 400 秒。
- Buffered Node 更适合优化批量吞吐，不会消除“同一个样本必须等所有 chunk 完成后才能 merge”的语义依赖。

### 有限资源下 OOM 实验

#### 实验描述

为了验证引入 GNN 预测器的必要性，考虑设置一个实验，让 llm 在尽量不受限制的情况下，输出 token 以完成其任务。在这个过程中，llm 可能会因为输出 token 过多而导致 OOM。如果在提示词中有意识地添加输出 token 长度的限制，是否可以避免 OOM 并依然能完成任务？

这里所说的 OOM 不一定是指真实的 GPU OOM，而是 llm 在输出 tokens 时，其显存峰值占用是否会超过一个理论的阈值。

我希望这个实验用以验证这个事情：在有限资源下，llm 的输出 token 长度是一个关键因素。通过限制输出 token 长度，是否可以避免 OOM 并依然完成任务？

这个实验如果成立，则说明在 docs/dev/workflow_plugin_20260626.md 的 Buffered Node 中添加 GNN 预测器是有必要的。对于一个即将发送到下游节点的提示词，预测器根据下游节点的 llm 参数信息，和提示词长度，规划一个最大输出 token 长度，避免下游节点的 llm 在处理时出现 OOM。同时在提示词中嵌入这个最大输出 token 长度的限制，确保下游节点在限制内尽可能完成任务，而不是直接被截断。

#### 实验内容

#### 实验结果
