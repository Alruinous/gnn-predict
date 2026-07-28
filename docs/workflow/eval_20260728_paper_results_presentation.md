# 四项实验结果呈现草案（2026-07-28）

## 数据集与工作负载

本文档只整理已有实验记录，不重新运行实验。实验说明从数据集开始，不包含集群和
GPU 环境。

| 实验 | 数据集或数据范围 | 工作负载 | 每次运行的数据量 |
|---|---|---|---|
| 端到端性能 | GSM8K、Sanitized MBPP、QMSum | 数学题多模型求解、代码生成与修复、会议摘要 | 三类各 20 个会话，共 60 个 |
| 模型驻留时间 | GSM8K、Sanitized MBPP、QMSum | 与端到端实验相同；分别观察集中到达和泊松到达 | 三类各 20 个会话，共 60 个 |
| GNN 预测器准确率 | 2,848 个服务配置 | 在不同模型、输入长度、输出长度和批处理配置上预测运行时间 | 50 个实测配置，2,798 个未参与校准的配置 |
| 节点融合 | QMSum、Sanitized MBPP | 两条三段滚动摘要链和一条三段定稿链，与代码修复会话共同到达 | 两类各 10 个会话；图片只统计 QMSum |

三类主实验工作流的作用如下。

| 数据集 | 工作流名称 | 工作流做什么 |
|---|---|---|
| GSM8K | `moa_gsm8k` | 五个求解节点分别回答数学题，随后汇总答案并精修 |
| Sanitized MBPP | `repair_mbpp` | 生成代码、运行测试、分析错误、修复代码并再次测试 |
| QMSum | `chain_qmsum` | 将会议文本分成两条滚动摘要链，随后合并、补充并定稿 |

端到端和模型驻留数据来自 `output/eval_20260727/`。预测器的网格、预算与 SageRadar
数据来自 `cache/canonical_metrics.json`，Analytical 和 GBDT 的后处理结果由
`cache/profile/predictions.yaml` 和 `cache/gnn/metrics.json` 中的同一批 50 个锚点
重新计算。节点融合数据来自 `output/fusion_exp/`。四张图片和统一结果文件位于
`output/eval_20260728_paper_drafts/`，均为临时产物，不属于论文目录。

## 实验一：端到端性能

### 问题与指标

这个实验回答：同样处理 60 个会话时，SagePilot 能否更快完成整批工作，同时缩短一个
普通会话需要等待的时间。

“批次完成时间”从运行开始计时，直到最后一个会话完成。它反映系统排空整批工作的速度。
“会话 p50”是一次运行中 60 个会话完成时间的中间值，它反映一个典型会话的完成速度。
每种方案相互独立地重复运行两次；柱子表示两次运行的平均值。

Parrot 和 Kairos 是调度顺序基线。Analytical 和 GBDT 保留 SagePilot 的其他系统机制，
只把资源预测结果分别换成解析公式和梯度提升树。因此，这两项展示预测方法变化最终会对
完整工作流产生多大影响，而不是两种新的调度器。

| 方法 | 批次完成时间（秒） | 会话 p50（秒） |
|---|---:|---:|
| Parrot | 885.3（870.7–899.9） | 499.7（484.9–514.5） |
| Kairos | 934.1（875.8–992.3） | 533.9（494.6–573.2） |
| Analytical | 866.3（788.7–943.9） | 541.4（496.2–586.7） |
| GBDT | 923.1（912.2–933.9） | 554.1（521.6–586.6） |
| **SagePilot** | **837.5（817.9–857.1）** | **475.1（427.2–522.9）** |

![端到端性能临时图](../../output/eval_20260728_paper_drafts/fig_e2e_performance_draft.png)

[矢量 PDF](../../output/eval_20260728_paper_drafts/fig_e2e_performance_draft.pdf)

### 图片表达的含义

上半图比较完成全部 60 个会话需要多久。柱子越短，说明系统越早处理完整批工作。
SagePilot 平均需要 837.5 秒，比最接近的 Analytical 缩短 3.3%，比 Parrot、Kairos
和 GBDT 分别缩短 5.4%、10.3% 和 9.3%。加入两种预测器基线后，SagePilot 的平均批次
完成时间仍然最低。

下半图比较会话完成时间的中间值。SagePilot 的结果为 475.1 秒，比 Parrot 缩短
4.9%，比 Kairos、Analytical 和 GBDT 分别缩短 11.0%、12.3% 和 14.3%。两组结果
放在一起说明，SagePilot 不仅更早完成整批工作，也让典型会话更早结束。

### Caption 草案

> **End-to-end performance for 60 burst-arrival sessions drawn equally from
> GSM8K, sanitized MBPP, and QMSum.** Results are averaged over two independent
> runs. Analytical and GBDT retain SagePilot's
> system mechanisms while replacing its predictor cache. SagePilot achieves the
> lowest mean batch completion time, improving on the next-best Analytical
> result by 3.3%, and the
> lowest median session completion time, improving on the next-best Parrot result
> by 4.9%.

## 实验二：模型驻留时间优化

### 问题与指标

这个实验回答：SagePilot 是否减少了模型已经装入 GPU 但没有生成内容的时间，以及一个
请求是否更少被其他模型的装载过程挡住。

上半图采用集中到达的两次运行，并按照动机实验的口径展示完整的 100% GPU time
distribution。每种方法的总量是三张 GPU 的运行时间之和；Generation 表示模型实际生成
内容，Idle resident 表示模型仍在 GPU 中但没有生成内容，Loading 表示将模型装入 GPU，
Other 表示其余未被前三项占用的 GPU 时间。图片展示比例，绝对 GPU 秒仍列在下表中。
Generation 与 Idle resident 的和就是模型驻留时间。

下半图采用泊松到达的一次运行。它统计所有模型获取等待时间中，有多少比例与另一个模型
的装载过程重叠。比例越低，说明请求越少因为 GPU 正在装载别的模型而停下来。

| 方法 | Generation（GPU 秒） | Idle resident（GPU 秒） | 驻留时间（GPU 秒） | Loading（GPU 秒） | 被其他模型装载挡住的等待 |
|---|---:|---:|---:|---:|---:|
| Parrot | 1,282.5 | 815.2 | 2,097.7 | 527.1 | 37.3% |
| Kairos | 1,179.7 | 832.8 | 2,012.5 | 763.1 | 37.6% |
| Analytical | 1,181.0 | 747.9 | **1,928.9** | 634.8 | 34.9% |
| GBDT | 1,253.7 | 875.2 | 2,128.9 | 606.3 | 30.2% |
| **SagePilot** | 1,224.4 | **734.1** | 1,958.5 | **525.6** | **29.8%** |

![模型驻留时间临时图](../../output/eval_20260728_paper_drafts/fig_model_residency_draft.png)

[矢量 PDF](../../output/eval_20260728_paper_drafts/fig_model_residency_draft.pdf)

### 图片表达的含义

上半图展示的是时间构成而不是绝对时长。SagePilot 将 48.6% 的可用 GPU 时间用于生成，
是五种方法中最高的；空闲驻留、模型装载和其他时间分别占 29.2%、20.9% 和 1.4%。
由于五种方法的总完成时间不同，比例不能代替绝对量比较。绝对量仍由上表给出：
SagePilot 的空闲驻留时间为 734.1 GPU 秒，比 Parrot、Kairos、Analytical 和 GBDT
分别减少 9.9%、11.9%、1.8% 和 16.1%；其模型装载时间也最低，与 Parrot 基本相同，
并比 Kairos、Analytical 和 GBDT 分别减少 31.1%、17.2% 和 13.3%。

Analytical 的生成时间更短，因此“生成加空闲”的原始驻留总和为 1,928.9 GPU 秒，比
SagePilot 的 1,958.5 GPU 秒低 1.5%。这个数字需要单独说明：Analytical 并没有更少的
空闲驻留或装载时间，而且其端到端完成时间更长。驻留优化图要表达的优势是减少没有产出
时的占用和模型切换开销，而不是把真正用于生成内容的时间也当成浪费。

下半图从请求的角度展示这一差别。SagePilot 被其他模型装载挡住的等待比例为 29.8%，
在五种方法中最低；GBDT 为 30.2%，Analytical 为 34.9%，Parrot 和 Kairos 约为 37%。
这说明 SagePilot 更常让当前请求需要的模型及时获得 GPU，而不是让它等待一个无关模型
完成装载。

### Caption 草案

> **Model residency and cross-model loading interference.** The upper panel
> reports the mean GPU time distribution across all three GPUs over two
> burst-arrival runs; resident time is the sum of generation and idle-resident
> time. The lower panel reports the
> fraction of acquire wait overlapping another model's loading under Poisson
> arrivals. SagePilot devotes the largest share of GPU time to generation and
> has the lowest absolute idle-resident time, loading time, and cross-model
> blocked-wait fraction. Analytical has 1.5% lower raw resident time because it
> spends less time generating, but it incurs more idle residency and loading
> and has longer end-to-end completion time.

## 实验三：GNN 预测器准确率

### 问题与指标

这个实验回答：只测量少量配置后，GNN 预测器能否准确预测未测配置的运行时间和峰值显存，
并给出调度真正需要的正确比较结果。

实验覆盖 2,848 个服务配置，其中 50 个配置用于提供实测参照，占完整配置空间的 1.47%；
其余 2,798 个配置用于检验预测结果。图中比较 SageRadar、解析公式和 GBDT。

三种方法只使用这 50 个配置进行校准。Analytical 在每个模型和 GPU 组合内只拟合一个
比例系数，避免带偏移的线性修正在未测输出长度上失真。GBDT 先在同一组合内完成线性校准，
再依据已知的输出 token 数，将 512-token 锚点预测扩展到目标输出长度。后处理没有使用
2,798 个检验配置的真实运行时间。峰值显存沿用冻结的 50-anchor 统一校准结果。

WAPE 将所有配置的预测误差绝对值相加，再除以对应实测值之和。它越低，整体预测越接近
实测结果。图中分别计算运行时间和峰值显存 WAPE。“Pairwise runtime ordering”从配置对
中抽样，检查预测器判断谁更快的顺序是否与实测顺序一致。“Predicted vs. measured
window fit”比较预测和实测对配置能否在固定窗口内完成的判断是否一致；窗口长度取
2,798 个检验配置实测运行时间的中位数 48.85 秒。

| 方法 | 运行时间 WAPE | 峰值显存 WAPE | 两两顺序准确率 | 中位运行时间窗口判断一致率 |
|---|---:|---:|---:|---:|
| **SageRadar** | **7.25%** | 14.08% | **97.94%** | **97.82%** |
| Analytical | 15.29% | **11.98%** | 96.37% | 96.82% |
| GBDT | 11.05% | 15.70% | 96.53% | 97.18% |

![GNN 预测器准确率临时图](../../output/eval_20260728_paper_drafts/fig_predictor_accuracy_draft.png)

[矢量 PDF](../../output/eval_20260728_paper_drafts/fig_predictor_accuracy_draft.pdf)

### 图片表达的含义

上半图比较运行时间和峰值显存预测与实测值的整体差距。SageRadar 的运行时间误差为
7.25%，低于 Analytical 的 15.29% 和 GBDT 的 11.05%；换算成误差降幅，分别减少
52.6% 和 34.3%。峰值显存方面，Analytical 以 11.98% 最低，SageRadar 为 14.08%，
GBDT 为 15.70%。因此 SageRadar 在运行时间预测上明显最好，显存预测则处于两种基线
之间。

下半图没有继续重复“预测值差多少”，而是直接检查预测结果会不会让调度器做出正确选择。
SageRadar 在两个配置谁更快的判断上达到 97.94%；对于长度等于实测运行时间中位数
48.85 秒的窗口，其预测结果与实测结果对“能否在窗口内完成”的判断一致率为 97.82%。
两项结果仍分别高于 Analytical 的 96.37% 和 96.82%，以及 GBDT 的 96.53% 和
97.18%。这说明三个方法经过后处理后都能较好地保留调度判断，而 SageRadar 的运行时间
数值误差和两类决策准确率仍然最好。

### Caption 草案

> **Predictor accuracy on 2,798 held-out serving configurations using 50 measured
> anchors.** After method-appropriate calibration using only the anchors,
> SageRadar achieves 7.25% runtime WAPE, versus 15.29% for Analytical and 11.05%
> for GBDT. Its peak-VRAM WAPE is 14.08%, between Analytical at 11.98% and GBDT
> at 15.70%. SageRadar also preserves 97.94% of pairwise runtime orderings and
> reaches 97.82% agreement between predicted and measured fit decisions for a
> 48.85-s window set to the held-out median runtime, remaining the best of the
> three methods on both decisions.

## 实验四：节点融合减少重复模型获取

### 问题与指标

这个实验回答：把连续使用同一模型的节点合并后，是否能减少中间节点重新排队获取模型的
次数，并缩短完成较慢的 QMSum 会话。

每种策略都包含三组一一对应的运行。每次运行先将 10 个 QMSum 会话按完成时间排序，并取
第 95 百分位位置的完成时间；表中和大圆点展示三次运行的平均值。图中的细线连接同一组
融合前后运行，空心圆表示未融合，实心圆表示融合，大圆表示平均值。横轴越靠左越好。
右侧数据列补充完整混合负载的 makespan 变化；正值表示融合后更长，负值表示更短。
SagePilot 的 makespan 变化剔除了异常的第二组运行及其配对基线，其余两种策略使用三组
完整配对。

| 调度方法 | 未融合 p95（秒） | 融合后 p95（秒） | 缩短比例 | Makespan 变化 | 三组配对中融合更快 |
|---|---:|---:|---:|---:|---:|
| Parrot | 240.1（223.4–249.8） | 214.7（210.0–217.1） | 10.6% | +0.1% | 3/3 |
| Kairos | 341.6（292.8–366.0） | 208.9（178.7–225.3） | 38.8% | +0.1% | 3/3 |
| SagePilot | 280.6（222.1–312.8） | 191.3（180.7–209.9） | 31.8% | −0.3% | 3/3 |

![节点融合临时图](../../output/eval_20260728_paper_drafts/fig_node_fusion_tail_draft.png)

[矢量 PDF](../../output/eval_20260728_paper_drafts/fig_node_fusion_tail_draft.pdf)

### 图片表达的含义

每一行都从右侧的未融合结果移动到左侧的融合结果，说明三种调度方法都从节点融合中获益。
Parrot、Kairos 和 SagePilot 的 p95 完成时间分别缩短 10.6%、38.8% 和 31.8%，九组
一一对应的运行全部变快。这个结果说明收益来自节点融合本身，而不是只在某一种调度方法下
才会出现。

右侧数据列显示 Parrot、Kairos 和 SagePilot 的 makespan 分别变化 +0.1%、+0.1% 和
−0.3%，均接近不变。这是因为当前负载受会话到达速度限制，完整运行的结束时间对融合并不
敏感；QMSum p95 更直接地反映融合所减少的链内等待。

融合在编译期将每个 QMSum 会话的模型获取次数从九次降到三次。连续节点在一次获取后直接
完成整条链，省去六次释放、重新排队和再次获取。配对线展示的时间下降说明，减少这些中间
操作最终缩短了较慢会话的完成时间。

### Caption 草案

> **Effect of node fusion on p95 QMSum workflow completion time.** Each thin line
> connects the fused and unfused result from the same repeat; the thick line
> connects their three-run means. Fusion lowers p95 completion time by 10.6%
> under Parrot, 38.8% under Kairos, and 31.8% under SagePilot; all nine paired
> runs improve. The right column reports whole-run makespan change, excluding
> the matched SagePilot r2 outlier pair; makespan remains within 0.3%.

## 绘图约定与复现

四张图均按单栏 3.33 英寸宽度生成，使用至少 8 pt 字体、可嵌入的 TrueType 字体、
颜色与纹理双重区分、矢量 PDF 和 600 DPI PNG。布局遵循
[PPoPP 2027 投稿要求](https://conf.researchr.org/track/PPoPP-2027/PPoPP-2027-papers)，
并参考 [Parrot](https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf)、
[ServerlessLLM](https://www.usenix.org/system/files/osdi24-fu.pdf)、
[LDB](https://www.usenix.org/system/files/nsdi24-cho.pdf) 和
[Llumnix](https://www.usenix.org/system/files/osdi24-sun-biao.pdf) 的结果图组织方式：
一张图只回答一个主要问题，保留独立运行点，直接标出主要变化，并把实验背景放在 caption
而不是图内。

统一生成命令如下：

```bash
PYTHONPATH=src uv run python scripts/workflow/plot_eval_20260728_drafts.py \
  --output-dir output/eval_20260728_paper_drafts
```

脚本直接读取已有 trace 和冻结的预测器指标，并在生成图片前检查运行数量、完成会话数量、
失败会话数量以及融合前后的模型获取次数。`metrics.json` 是本文档表格和图片数值的统一
来源。
