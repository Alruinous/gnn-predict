# Workflow GNN 调度评估 20260612

## 结论

- 在可用性优先口径下, GNN-aware 相对 `a100_first` 的平均 session makespan 改善 `4.11%`。
- GNN-aware OOM events 为 `10`, 可用性 baseline OOM events 为 `10`。
- 最快 baseline `round_robin` 的 OOM events 为 `40`, 说明低均值延迟来自更高失败/重试风险。
- GNN-aware 能耗代理相对可用性 baseline 变化 `-2.05%`。
- 收益主要来自提前规避 V100 可用显存不足, 把冷启动更重的工具放到更合适的设备, 以及跨 session 复用缓存。

## 实验口径

本实验面向 V100/A100 受限资源集群中的 agent workflow 工具节点调度。
主实验采用离线 trace replay: GNN 预测值只用于调度决策, 最终耗时, 显存, OOM 和能耗代理使用 `data/extracted/test.pt` 中 V100/A100 配对样本的实测 target 计算。
这避免了只用预测值闭环评估, 但仍不是一次真实线上集群压测。

## 评估指标依据

类似 GPU 集群和推理服务调度工作通常同时关注用户侧延迟与系统侧资源效率。
本实验采用 session makespan, tool JCT, OOM/retry, 缓存命中, 冷启动耗时, 显存利用率和能耗代理。

| reference | why used | url |
| --- | --- | --- |
| SCHEDTUNE | 异构 GPU 调度以 OOM avoidance、显存利用率和 makespan 为核心指标。 | https://people.cs.vt.edu/~butta/docs/ccgrid22-schedtune.pdf |
| PAL | GPU 集群调度常用 JCT、utilization 和 makespan 评估策略收益。 | https://arxiv.org/abs/2408.11919 |
| MArk | 推理服务评估关注 SLO latency 与 serving cost 的折中。 | https://www.usenix.org/conference/atc19/presentation/zhang-chengliang |

## 指标解释

本节先解释文档中出现的两类指标：一类用于说明 GNN 预测器本身是否可信，另一类用于说明 GNN 预测信号放进 workflow 调度后是否改善了端到端效果。

### GNN 预测器指标

这些指标来自 `output/gnn_full_retrain_20260612` 的 test split，用来说明 GNN 对耗时、显存、功耗等目标的预测误差。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| WAPE | Weighted Absolute Percentage Error，加权绝对百分比误差。计算方式近似为 `sum(abs(pred - real)) / sum(abs(real))`。 | 越低越好。`0.135549` 约等于整体绝对误差占真实总量的 `13.55%`。 |
| R2 | 决定系数，衡量预测值解释真实值变化的能力。 | 越接近 `1` 越好。`0.949278` 表示模型能解释大部分 test target 变化。 |
| original-scale | 原始量纲指标。比如秒、GB、MB、W，而不是标准化后的数值。 | 用于判断真实业务量纲下的误差。 |
| scheduling-core | 只聚合调度最直接依赖的目标：部署耗时、运行耗时、显存和功耗。 | 比 overall 更贴近本实验的调度问题。 |
| deployment WAPE | 对 `deployment_duration_sec_avg` 的预测误差。 | 影响冷启动、缓存复用和是否提前部署。 |
| run WAPE | 对 `run_duration_sec_avg` 的预测误差。 | 影响工具调用后的等待时间和 ready tools 排序。 |
| gpu_mem WAPE | 对 `gpu_mem_used_mb_max` 的预测误差。 | 影响是否可能 OOM、是否需要换卡或串行。 |
| gpu_power WAPE | 对 `gpu_power_watts_avg` 的预测误差。 | 用于估算能耗代理，不作为本实验唯一优化目标。 |

### Workflow 调度指标

这些指标来自离线 trace replay。GNN 只用于做调度决策，最终统计使用 test trace 中对应设备的实测 target。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| avg session makespan sec | 每个 session 中所有工具完成的端到端时间均值。这里把一个 session 内分配到不同 GPU 的工具视为可并行，session makespan 是最慢那张卡的累计时间。 | 越低越好，代表用户从 agent 发起工具调用到所有工具返回的平均等待更短。 |
| p95 session makespan sec | session makespan 的 95 分位数。 | 越低越好，反映尾部慢请求。它比均值更能暴露少数重工具导致的体验问题。 |
| avg tool JCT sec | Tool Job Completion Time，单个工具从被调度开始到执行完成的平均时间。包含冷启动部署耗时、运行耗时、以及 OOM retry penalty。 | 越低越好，代表单次工具调用返回更快。 |
| OOM | 调度到某张卡后，实测显存需求加安全余量超过该卡可用显存的次数。 | 越低越好。OOM 会触发失败、重试或 agent 等待，是受限资源场景中的核心可用性指标。 |
| retry | 因首选设备 OOM 后，尝试下一个设备的次数。 | 越低越好。retry 会增加额外等待，本实验每次 OOM retry 计入 `5 sec` penalty。 |
| cache hit | 工具模型已经部署在目标 GPU 缓存中，当前调用不用重新部署的比例。 | 越高通常越好。缓存命中能减少冷启动，但如果为了缓存牺牲选卡也可能增加排队或 OOM 风险。 |
| cold start sec | 所有工具实际发生模型部署的耗时总和。缓存命中时部署耗时记为 `0`。 | 越低越好。它直接反映模型复用和预部署价值。 |
| energy Wh | 能耗代理，按 `gpu_power_watts_avg * (deployment_duration_sec_avg + run_duration_sec_avg) / 3600` 估算。 | 越低越好，但它是代理指标，不等同于真实机房电表能耗。 |
| avg mem util | 每个 session 后 GPU 缓存占用显存 / 模拟可用显存的平均比例。 | 不是单纯越高越好。过低可能资源浪费，过高可能增加 OOM 和驱逐压力。 |

### 策略名称解释

| 策略 | 含义 | 作为 baseline 的意义 |
| --- | --- | --- |
| `v100_only` | 总是优先把工具放到 V100；如果 V100 OOM，再尝试 A100。 | 模拟便宜/旧卡优先使用的保守策略。 |
| `a100_first` | 总是优先把工具放到 A100；如果 A100 OOM，再尝试 V100。 | 模拟强卡优先、可用性优先的简单策略。 |
| `round_robin` | V100/A100 轮转分配，不看模型结构和预测成本。 | 模拟简单负载均衡。它可能均值延迟低，但 OOM 风险较高。 |
| `static_size` | 只用 ONNX 静态规模代理估计显存，不使用 GNN 的部署耗时、运行耗时、功耗预测。 | 用来比较“静态图规模启发式”与“GNN 多目标预测”的差异。 |
| `gnn_aware` | 使用 GNN 预测的部署耗时、运行耗时、显存和功耗，并结合当前缓存与排队状态选卡。 | 本实验要验证的策略。 |

## GNN 模型与数据

| item | path |
| --- | --- |
| config | output/gnn_full_retrain_20260612/effective_config.yaml |
| checkpoint | output/gnn_full_retrain_20260612/gnn_model_scaled_20260612/checkpoints/best_model.pt |
| scalers | data/scalers |
| test trace | data/extracted/test.pt |

20260612 复训记录中的 test 指标摘要:

- original-scale overall WAPE: `0.135549`, R2: `0.949278`。
- scheduling-core WAPE: `0.135791`。
- 关键目标 WAPE: deployment `0.141669`, run `0.222849`, gpu_mem `0.135247`, gpu_power `0.148130`。
- V100 test WAPE: `0.147589`, A100 test WAPE: `0.115542`。

## 样例 Workflow

本实验显式构建了 3 个样例 workflow, 用来展示 GNN 对真实 workflow 工具节点的 prospective 预测结果。
这些 YAML 不参与 trace replay 的实测 target 统计; trace replay 使用 `data/extracted/test.pt` 中的 V100/A100 配对记录。

### 样例 Workflow 详情

`multimodal_triage` 模拟多模态分流场景。用户请求先进入外部 LLM planner, planner 可以并行调用图像分类工具和文本意图分类工具。

```text
input
  -> planner
      -> scene_classifier
      -> intent_classifier
  -> output
```

| node | type | task | model | runtime | 作用 |
| --- | --- | --- | --- | --- | --- |
| `planner` | agent | `react_agent` | `qwen3` | batch `1`, seq `256`, prefill | 外部 LLM 代理节点, 负责选择本地工具。 |
| `scene_classifier` | tool | `image_classification` | `mobilenetv2_100` | batch `1`, shape `[1,3,224,224]`, inference | 轻量图像分类工具。 |
| `intent_classifier` | tool | `text_classification` | `gpt2` | batch `1`, seq `128`, inference | 文本意图分类工具。 |

`text_review_and_summary` 模拟文本审核和摘要场景。它先做主题/意图分类, 再调用生成式摘要工具。

```text
input -> planner -> topic_classifier -> summary_draft -> output
```

| node | type | task | model | runtime | 作用 |
| --- | --- | --- | --- | --- | --- |
| `planner` | agent | `react_agent` | `qwen3` | batch `1`, seq `512`, prefill | 外部 LLM 代理节点。 |
| `topic_classifier` | tool | `text_classification` | `gpt2` | batch `2`, seq `256`, inference | 对输入文本做主题分类。 |
| `summary_draft` | tool | `text_generation` | `t5` | batch `1`, seq `256`, decode `64` | 生成摘要草稿, 代表 decode 阶段工具。 |

`visual_quality_review` 模拟视觉质检场景。先运行轻量 fast filter, 只有需要更细判断时再运行 heavier checker。

```text
input -> planner -> fast_visual_filter -> dense_visual_checker -> output
```

| node | type | task | model | runtime | 作用 |
| --- | --- | --- | --- | --- | --- |
| `planner` | agent | `react_agent` | `qwen3` | batch `1`, seq `256`, prefill | 外部 LLM 代理节点。 |
| `fast_visual_filter` | tool | `image_classification` | `resnet18` | batch `4`, shape `[4,3,224,224]`, inference | 快速视觉过滤工具。 |
| `dense_visual_checker` | tool | `image_classification` | `densenet121` | batch `2`, shape `[2,3,224,224]`, inference | 更重的视觉复核工具。 |

这些样例 workflow 的 agent 节点不部署本地小模型, 因为场景设定中主 agent 使用外部 LLM provider。GNN 调度只作用于 `tool` 节点。
下表中的 V100/A100 数值是把每个 tool 节点导出为临时 ONNX, 抽取图特征后送入 GNN 预测器得到的 prospective 估计。

| workflow | node | model | phase | V100 mem GB | A100 mem GB | V100 deploy+run sec | A100 deploy+run sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multimodal_triage | scene_classifier | mobilenetv2_100 | inference | 2.5191 | 1.8521 | 0.2504 | 0.3232 |
| multimodal_triage | intent_classifier | gpt2 | inference | 1.3320 | 2.5480 | 0.5006 | 0.0041 |
| text_review_and_summary | topic_classifier | gpt2 | inference | 1.0877 | 2.5860 | 0.6156 | 0.0185 |
| text_review_and_summary | summary_draft | t5 | decode | 1.2207 | 2.6079 | 15.4582 | 3.8226 |
| visual_quality_review | fast_visual_filter | resnet18 | inference | 9.3895 | 2.5383 | 0.0397 | 0.2144 |
| visual_quality_review | dense_visual_checker | densenet121 | inference | 2.7784 | 2.8169 | 0.6668 | 0.6900 |

## Trace Replay 设置

| item | value |
| --- | --- |
| trace records | 24 |
| sessions | 24 |
| tools per session | 3 |
| V100 available memory | 8.00 GB |
| A100 available memory | 14.00 GB |
| memory margin | 0.50 GB |
| OOM retry penalty | 5.00 sec |

配对 trace 选择优先覆盖显存和冷启动压力更高的 test 样本; 每个 session 包含固定数量 ready tools, 缓存跨 session 保留。

Trace replay 中的 `session` 对应文档 `senario_20260612.md` 里的“一个用户 session 对应一种 workflow 组合”。
本次共构造 24 个模拟 session workflow, 每个 session 有 3 个 ready tool。
它们不是额外落盘的 YAML, 而是从 test split 中选择 V100/A100 都有实测记录的 tool trace 后按固定规则组合出来:

- 先过滤出非 training phase, 且同一 `variant_name + phase + batch_size + decode_output_length` 同时有 V100 和 A100 记录的样本。
- 按显存压力、部署耗时和运行耗时从高到低排序, 取前 24 条作为可调度工具池。
- 每个 session 选择 3 个 tool trace; 部分 hot tools 会跨 session 重复出现, 用来模拟多用户 workflow 中的模型缓存复用。
- 模拟器保留跨 session GPU 缓存, 所以后续 session 如果命中已部署模型, 冷启动耗时记为 `0`。
- session 内 3 个 ready tools 可被调度到不同 GPU 并行执行; 该 session 的 makespan 取两张卡中累计时间更长的一侧。

## 对比结果

| strategy | avg makespan sec | p95 makespan sec | avg tool JCT sec | OOM | retry | cache hit | cold start sec | energy Wh | avg mem util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v100_only | 230.0812 | 329.8695 | 84.6349 | 65 | 60 | 5.56% | 5361.8376 | 132.5338 | 69.98% |
| a100_first | 237.8487 | 314.8695 | 79.9773 | 10 | 5 | 0.00% | 5398.5881 | 125.9539 | 37.15% |
| round_robin | 215.6953 | 324.8695 | 74.8253 | 40 | 35 | 6.94% | 4874.5389 | 118.9764 | 76.20% |
| static_size | 224.8728 | 329.8695 | 82.8988 | 40 | 35 | 5.56% | 5361.8376 | 132.5338 | 69.98% |
| gnn_aware | 228.0771 | 314.8695 | 80.6291 | 10 | 5 | 5.56% | 5352.0404 | 128.5372 | 51.82% |

## 结果解读

- 静态策略只能看到 ONNX 规模代理, 无法直接判断不同 GPU 上的部署耗时, 运行耗时和显存峰值, 因此在受限显存下更容易触发 retry。
- `v100_only` 和 `static_size` 在部分重模型 trace 上会先打到 V100, OOM 后再 retry 到 A100, 直接增加 tool JCT 和 session makespan。
- GNN-aware 的 cache hit rate 为 `5.56%`, 说明预测调度没有牺牲跨 session 复用。
- 对 agent workflow 来说, 降低 OOM 与 retry 比微调单个工具的毫秒级运行时间更直接影响端到端体验。

## 限制

- 本实验没有真实启动 V100/A100 集群, 只能说明 GNN 预测信号在离线调度仿真中的决策价值。
- Trace replay 使用 row-random test split 中的已见分布样本, 不能替代 family-holdout 泛化实验。
- 能耗使用 `gpu_power_watts_avg * duration` 作为代理, 不等同于机房级能耗计量。
- 当前模拟器使用单次 retry 和 LRU 缓存驱逐, 真实系统还需要纳入并发干扰, 网络, 容器启动和 agent 重规划成本。
