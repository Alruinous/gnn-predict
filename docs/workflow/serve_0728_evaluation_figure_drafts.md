# serve_0728 Evaluation 图片细分草图

本文档只用于选择论文图片，不修改论文正文或论文目录中的正式图片。

## 数据口径

- 系统性能、模型切分和消融数据是 `serve_0728_reference.md` 第 10 节定义的统一加载时间反事实回放。
- `a1v4` 完全排除。
- 系统性能图使用 `a1v3` 的 Burst、Poisson 0.050、Poisson 0.025 和 Poisson 0.0125；模型切分和
  消融图只使用 `a1v3 + Burst`。这些结果都严格采用第 11 节选择的轮次，`Mean(...)` 只平均其中列出的轮次。
- GPU time distribution 例外使用 `output/eval_20260727` 的原始 Burst 运行，不使用
  `serve_0728` 反事实回放或第 11 节人工选择轮次。
- 系统性能图中的 `Oracle` 是从所选 SagePilot 结果派生的 trace-conditioned 放松下界，只读取
  已有 `workflow_trace.jsonl`；不启动系统、不提交 GPU 任务，也不运行 `ReplaySimulator`。它不是
  新增实验运行或可执行调度器的实测结果。GPU time distribution 不包含 Oracle。
- 除 GPU time distribution 外，所有 session、workflow 和模型级指标都从同一批回放重新提取，
  不从原集群 wall-clock trace 拼接。

## 当前候选：a1v3 论文式布局

当前系统性能草图使用第 11 节选择的 `a1v3` 四档到达方式；模型切分和消融仍固定使用
`a1v3 + Burst`。GPU time distribution 单独使用 `output/eval_20260727` 的三卡 Burst 数据。
每一行只比较同一种到达方式，不会跨 Burst 和 Poisson arrival span 比较 makespan。

`scripts/workflow/plot_serve_0728_bar_drafts.py` 负责从选中运行生成
`output/serve_0728_bar_drafts/analysis_metrics.json`；当前图片由
`scripts/workflow/plot_serve_0728_layout_drafts.py` 读取该冻结结果并生成。布局脚本不重新选择轮次，
也不读取 a1v4。GPU time distribution 由
`scripts/workflow/plot_eval_20260727_gpu_time_draft.py` 直接读取原始运行并独立生成。双栏图宽
7.0 英寸，单栏 GPU 图宽 3.33 英寸，图内最小字号为 8 pt。

系统颜色沿用论文现有定义：Parrot 为蓝色、Kairos 为绿色、Analytical 为紫色、GBDT 为橙色、
SagePilot 为红色；Oracle 使用深灰色，强调它只是参考下界。纹理提供黑白打印时的第二重
区分。消融项不是独立系统，不占用新的系统颜色：SagePilot 保持红色，三个消融项使用由深到浅
的灰度。

### 系统性能：makespan 与 p95 latency

![System makespan and p95 latency](../../output/serve_0728_layout_drafts/fig_system_performance_layout_draft.png)

矢量版本：[PDF](../../output/serve_0728_layout_drafts/fig_system_performance_layout_draft.pdf)

该图采用 4 行 4 列布局：四行依次为 Burst、Poisson 0.050、Poisson 0.025 和 Poisson 0.0125，
四列依次为 Overall、4B、8B 和 14B。每个子图中，五种系统和一个 Oracle 参考各有一对柱；
实色柱表示 makespan，白色填充、系统颜色描边和斜线纹理表示 p95，不再展示 p50。子图标题在
顶部居中标明到达方式与统计范围，例如 `Poisson 0.025 + 8B`。该图沿用开放坐标轴，只保留左、
下边框和水平网格；系统图例与 makespan/p95 图例位于整图顶部，柱顶数值按秒取整并逆时针旋转。

- Overall makespan 是从 workload 开始到最后一个 session 完成的时间，Overall p95 是 session
  完成 latency 的第 95 百分位。
- 模型列先对每个 session 取使用该模型的最后一个 agent 节点完成 latency，再分别取最大值和
  第 95 百分位。它包含 DAG 依赖、排队、加载和执行，不是模型的纯 inference latency。

#### Oracle 候选说明草稿（暂不进入实验正文）

图中将该参考项简称为 Oracle。严格来说，它是针对当前 trace 的放松理论下界，不是已经实现的
调度器，也不是重新运行系统得到的测量结果。Oracle 直接读取第 11 节所选 SagePilot 轮次的
`workflow_trace.jsonl` 和 `run_manifest.json`。
到达时间取 `session_submitted`，每个节点的执行时长取 `task_execution_finished.duration_sec`，模型
访问顺序取 `acquire_granted`，加载代价取该轮 `model_load_started/finished` 对中每个模型实际观测
到的最短时长。计算不调用 counterfactual replay，也不重新启动系统。

每个 session 去掉排队、加载和生命周期等待后按 DAG 关键路径计算 latency，并对这 60 个理论
latency 仍使用与实测柱完全相同的 nearest-rank p95，而不是最大 latency。Overall makespan 再取
“带到达时间的 DAG 关键路径跨度”和
`(trace active GPU-seconds + 离线最优 loading GPU-seconds) / 4` 的较大值。离线加载项是在固定
trace grant 顺序上按模型加载时长做的精确加权缓存最优，是论文第一项动机实验 Belady miss-count
下界的 cost-aware 版本。模型列按每个 session 的最早节点完成时间计算，不额外施加全局四卡
工作量约束。

这个下界假设四张异构 GPU 可被理想打包、预加载可在正确时刻发生、无 idle residence，并忽略
真实调度中改变顺序后对 batching 和执行时长的反馈。因此它用于表示当前 trace 尚存的理论空间，
不能写成 SagePilot 已经达到的性能，也不能写成整个调度问题经过求解后的严格全局最优。

四档到达下，Oracle 的 Overall makespan/p95 依次为 415.5/68.2、469.6/67.9、
928.5/65.1 和 1846.4/65.8 秒。对应 trace 的实际加载次数为 21/19/25/46，固定 grant 顺序上的
加权离线最优下界为 7/9/13/16 次。Burst 的 makespan 下界由四卡 GPU 工作量约束决定；三个
Poisson 档中到达跨度逐渐成为主导，因此 0.025 和 0.0125 的 makespan 下界仍接近实测 makespan，
而 p95 关键路径下界保持在 65--68 秒。这一分离说明低到达率 makespan 主要受 offered arrival
span 限制，不能据此认为调度优化空间很小。

只刷新 Oracle 而不运行 replay 的命令为：

```sh
PYTHONPATH=src:scripts/workflow uv run python \
  scripts/workflow/plot_serve_0728_bar_drafts.py --refresh-oracle-from-traces
```

Burst 下，SagePilot 的 Overall makespan/p95 为 442.4/409.5 秒，4B 为 421.8/386.7 秒，8B
为 442.4/409.5 秒；三个范围均低于其他系统。14B 的 130.6/124.2 秒则没有形成优势，说明收益
集中在决定尾部的 4B 和 8B 路径，而不是所有模型都均匀加速。

Poisson 0.050 下，SagePilot 的 Overall makespan 为 673.3 秒，仍是五种系统中最低，但 p95
为 378.0 秒，高于 Kairos 的 347.0 秒和 GBDT 的 375.6 秒。SagePilot 的 4B 和 8B makespan
分别为 367.2 和 403.6 秒，均略低于 GBDT；对应 p95 并不领先。Poisson 0.025 下也呈现相同
分离：SagePilot 的 Overall makespan 为 967.2 秒、8B makespan 为 311.9 秒，均为最低，但
Overall p95 为 273.5 秒，明显高于 Kairos 的 162.7 秒。

Poisson 0.0125 下，SagePilot 的 Overall makespan/p95 为 1883.6/156.1 秒；makespan 仅比 GBDT
低 0.2 秒、比 Analytical 低 0.4 秒。该档 makespan 主要由到达跨度决定，因此只能在同一行内
比较，不能将四行的绝对 makespan 直接用于负载强弱结论。整体上，这张图支持 SagePilot 更早
结束关键模型路径，但不支持其在所有到达方式和模型上都取得最低 p95。

候选 caption：`Makespan and p95 completion latency across arrival processes and model groups; Oracle is a trace-conditioned relaxed lower bound derived from SagePilot.`

### 系统 makespan：分模型

![Per-model makespan](../../output/serve_0728_layout_drafts/fig_completion_models_layout_draft.png)

矢量版本：[PDF](../../output/serve_0728_layout_drafts/fig_completion_models_layout_draft.pdf)

五个子图分别展示 0.6B、1.7B、4B、8B 和 14B 模型的完成时间。模型 makespan 先对每个
session 取使用该模型的最后一个 agent 节点完成时间，再取全部 session 的最大值。它包含 DAG
依赖、排队、加载和执行，不是模型的纯 inference latency。柱顶数值按秒取整，并逆时针轻微旋转，
避免相邻数字遮挡。

总体排空时间几乎由 QMSum 和 8B 模型决定；SagePilot 在 4B 和 8B 两个关键子图中最早排空，
但没有在 0.6B、1.7B 和 14B 上全部领先。该图用于说明 SagePilot 优先缩短决定 workload
尾部的模型路径，而不是均匀改善所有模型。

候选 caption：`Completion time grouped by model under burst arrivals.`

### 消融实验：makespan 与 p50/p95 latency

![Ablation makespan and latency](../../output/serve_0728_layout_drafts/fig_ablation_performance_layout_draft.png)

矢量版本：[PDF](../../output/serve_0728_layout_drafts/fig_ablation_performance_layout_draft.pdf)

该图固定使用 `a1v3 + Burst`，不包含 Poisson 结果。图中采用 2 行 4 列布局，并只保留与 workload
尾部直接相关的 Overall、QMSum、4B 和 8B。
第一行展示 makespan，第二行展示同一分组的 p95 实色柱和 p50 斜线柱。SagePilot、NoFusion、
NoPrefetch 和 NoXWF 的图例位于顶部，p95/p50 图例只放在第二行上方。柱顶数值按秒取整，成对
的 p95/p50 数值逆时针旋转。SagePilot、NoFusion、NoPrefetch 和 NoXWF 的总体 makespan
分别为 442.4、459.6、459.5 和 515.1 秒。相对 SagePilot，关闭 fusion 和关闭 prefetch 都使
总体 makespan 增加约 3.9%，关闭跨 workflow 生命周期管理使总体 makespan 增加 16.4%。

NoFusion 的总体 p50 从 222.8 秒增加到 226.3 秒，总体 p95 从 409.5 秒增加到 444.2 秒，增幅
分别为 1.6% 和 8.5%。分到 QMSum 后，SagePilot 的 p95 为 424.1 秒，低于 NoFusion 的
451.3 秒，关闭 fusion 后增加 6.4%；新选择的 Burst 轮次能同时从总体 tail 和长链 workflow
体现节点融合收益。

NoXWF 的总体 p95 为 492.8 秒、QMSum p95 为 501.8 秒，均明显变差。NoPrefetch 的总体 p95
为 433.7 秒、QMSum p95 为 446.4 秒，相对 SagePilot 分别增加 5.9% 和 5.3%；当前 Burst
选择能体现 prefetch 的小幅收益，但幅度明显弱于跨 workflow 生命周期管理。

候选 caption：`Effect of SagePilot components on makespan and workflow latency under burst arrivals.`

### GPU time distribution

![GPU time distribution](../../output/eval_20260727_gpu_time_draft/fig_gpu_time_distribution_eval_20260727_draft.png)

矢量版本：[PDF](../../output/eval_20260727_gpu_time_draft/fig_gpu_time_distribution_eval_20260727_draft.pdf)；
冻结指标：[JSON](../../output/eval_20260727_gpu_time_draft/analysis_metrics.json)

数据明确来自 `output/eval_20260727`，实验条件为 Burst、60 个 session 和三张 GPU
（1xA100 + 2xV100）。每种系统取两次重复运行，原始运行目录如下。

| 系统 | `output/eval_20260727` 下的运行目录 |
| --- | --- |
| Parrot | `burst_parrot`、`burst_parrot_r2` |
| Kairos | `burst_kairos`、`burst_kairos_r2` |
| Analytical | `burst_sys_static`、`burst_sys_static_r2` |
| GBDT | `burst_sys_tabular`、`burst_sys_tabular_r2` |
| SagePilot | `burst_sys_gnn_v2`、`burst_sys_gnn_v2_r2` |

绘图脚本重新分析各运行的 `workflow_trace.jsonl`，并校验 60 个 session 全部完成且没有失败。
四种状态先分别对两轮 GPU-seconds 取平均，再除以 `3 × Mean(run duration)`，因此每根柱表示
完整三卡 GPU window 的 100%。柱内四部分互斥：Generation、Idle resident、Loading 和 Other；
五个系统以竖直的 100% 堆叠柱排列，填充色只表示 GPU 状态，柱段仅使用白色分隔边，不使用
系统颜色外框，名称也统一使用中性文本色。图内不设置实验小标题，按论文展示方式标注一位小数
百分比，精确值保存在冻结指标 JSON 中。

Parrot、Kairos、Analytical、GBDT 和 SagePilot 的 generation share 分别为 48.2%、41.9%、
45.3%、45.2% 和 48.6%；idle resident share 分别为 30.6%、29.6%、28.7%、31.5% 和
29.2%；loading share 分别为 19.8%、27.1%、24.4%、21.8% 和 20.9%。SagePilot 的 generation
share 最高。其 idle resident 绝对时间为 734.1 GPU-seconds，较 GBDT 的 875.2 GPU-seconds
减少 16.1%；loading 绝对时间为 525.6 GPU-seconds，较 GBDT 的 606.3 GPU-seconds 减少
13.3%，且与 Parrot 的 527.1 GPU-seconds 基本一致。正文仍应区分“时间占比”和“绝对
GPU-seconds”，不要把百分比分布直接写成同等比例的绝对时间降幅。

候选 caption：`GPU time distribution across three GPUs under burst arrivals (output/eval_20260727).`
