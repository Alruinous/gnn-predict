# 深度学习模型性能预测研究对照调研

调研时间：2026-05-19

## 摘要

当前项目与现有研究的交集很明确：都希望用模型结构和运行环境特征，在真实运行前预测深度学习模型的运行时间、显存、利用率等资源指标。最接近的是 PerfSeer、DNNPerf 和 DIPPM 这类图级 GNN 性能预测器；PerfSAGE、BRP-NAS 和 nn-Meter 更偏向边缘设备或 NAS 场景；Habitat、NeuSight 和 BOOM 更偏向跨 GPU 或系统级性能建模。

主结论如下：

- 现有工作通常不会默认按完整模型类型做测试集留出，随机配置划分或固定搜索空间内划分更常见。
- DNNPerf 是明确设置未见模型族测试的代表；DIPPM 只在 MIG profile 示例里额外展示 seen、partially seen、unseen 架构。
- 本项目的多模型族覆盖更宽，尤其包含 YOLO、GPT-2/T5 和推荐模型，但当前行级 split 不能支持严格泛化结论。
- 本项目当前图特征比 PerfSeer/DNNPerf 更轻，缺少显式 op type、完整 tensor shape、算术强度、训练反向图和运行时优化信息。
- 如果要形成严谨论文口径，建议至少报告常规随机或 variant split、模型族留出 split、硬件留出 split 三套结果。

## 本项目定位

当前项目用 ONNX 图构造 PyG `Data`，再用 GNN 做图级多目标回归。相关实现集中在：

- `src/gnn_model/data/onnx_graph.py`：从 ONNX 和 `onnx_tool` profile 中提取节点、边和图级特征。
- `src/gnn_model/data/extract.py`：从监控 CSV 生成图样本，当前按样本行随机划分 train/val/test。
- `src/gnn_model/models/predictor.py`：用节点、边和图级状态共同预测多个目标。
- `src/gnn_model/models/fusion.py`：用 `TransformerConv` 更新节点，用 MLP 更新边，并融合图级状态。

当前特征口径：

| 层级 | 当前特征 |
| --- | --- |
| node | MACs、memory、params、input/output 数、attr 数、in/out degree |
| edge | tensor bytes、rank、element count、source out-degree、target in-degree |
| graph | phase、batch、sample count、GPU specs、参数输入统计、全图 MACs/FLOPs/memory/params |

当前目标口径来自监控 CSV，核心包括：

- `duration_sec_avg`
- `cpu_cores_p95`
- `memory_gb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

已有项目报告显示，2026-05-17 等权基线在行级 split 下 test WAPE 为 `0.087023`、R2 为 `0.979250`，但 train/val/test 之间存在 `variant_name` 重叠，不能解释为严格未见架构泛化。报告还显示推荐模型和 GPT/T5 是当前最难模型族。

## 代表性工作对照

| 工作 | 主要任务 | 表示方式 | 数据与硬件 | 目标指标 | 划分与泛化口径 | 与本项目关系 |
| --- | --- | --- | --- | --- | --- | --- |
| PerfSeer, IJCAI 2025 | 训练和推理性能预测 | ONNX 计算图，含 node/edge/global features | 53k+ 配置，RTX 3090，VGG、GoogLeNet、ResNe(X)t、MobileNet、DenseNet | time、memory、SM utilization | 公开论文未看到以模型族留出的主实验描述 | 最接近本项目；多指标、训练+推理、ONNX/GNN 口径相似 |
| DNNPerf, 2021 | 训练时间和 GPU memory 预测 | TensorFlow computation graph，node/edge/device features | 18,641 配置，TensorFlow v1.13.1，真实模型+NAS 合成模型 | training time、GPU memory | 五个模型族做 70/10/20，另五个模型族作为 UNSEEN | 明确支持 leave-model-family-out 泛化评估 |
| PerfSAGE / EdgeDLPerf, 2023 | 边缘推理性能预测 | TFLite graph，GraphSAGE 风格模型 | 134,912 个 TFLite 模型，CPU/NPU/EdgeTPU | latency、energy、memory/SRAM | 80/20 随机划分；specialized/generalized 覆盖不同 design space | 证明多 design space 需要重采样和泛化评估，但任务偏 edge inference |
| DIPPM, 2023 | A100 推理性能和 MIG profile 预测 | TVM Relay 转图，GraphSAGE + static features | 10,508 个 DL 模型，A100 40GB | latency、energy、memory、MIG profile | 70/15/15 随机划分；另有 seen/partially seen/unseen 示例 | 与本项目同为 GPU 图级多回归，但只做 A100 inference |
| nn-Meter, MobiSys 2021 | 边缘推理 latency 预测 | kernel-level decomposition | 26,000 CNN 模型，mobile CPU/GPU/VPU | latency | 面向 unseen model graph 的 kernel 泛化，不是 GNN 图级回归 | 提醒本项目需要关注 operator fusion 和 runtime optimization |
| BRP-NAS / Eagle, NeurIPS 2020 | NAS latency 和 accuracy/rank 预测 | NAS graph + GCN | NAS-Bench-201，LatBench 覆盖多设备 | latency、accuracy rank | 固定 NAS 空间内随机采样，900 train/100 val/其余 test | 对 NAS 很强，但模型空间窄于本项目 |
| HELP, NeurIPS 2021 | 硬件自适应 latency 预测 | meta-learning latency predictor | 多硬件 NAS latency 场景 | latency | 重点是 unseen device few-shot adaptation | 对本项目后续跨 GPU 泛化很有参考价值 |
| Habitat, USENIX ATC 2021 | 从已有 GPU 预测其他 GPU 训练迭代时间 | operation runtime scaling / MLP | ResNet-50、Inception v3、Transformer、GNMT、DCGAN，六种 GPU | training iteration time | 跨 GPU 预测 | 对本项目硬件特征和 cross-GPU split 有参考价值 |
| NeuSight, ASPLOS 2025 | 预测 unseen GPU 上训练/推理性能 | kernel tile-level modeling | 多类 DL workloads 和新 GPU | training/inference latency | 强调 unseen model、unseen GPU、二者同时未见 | 对本项目从图级回归走向硬件可迁移建模有参考价值 |
| BOOM, PACT 2024 | 大模型训练 GPU 选择 | memory predictor + runtime predictor | CNN 和 Transformer，大模型训练优化 | memory footprint、runtime | 强调 memory generalization 和训练优化 | 对本项目的显存预测、混合精度、checkpointing 等因素有参考价值 |

## 数据集划分上的异同

现有研究可以分成几种泛化层级：

| 泛化层级 | 说明 | 代表做法 |
| --- | --- | --- |
| 行级随机 split | 同一模型族、甚至相似变体可跨 split | DIPPM 主实验、PerfSAGE 主实验、本项目当前数据 |
| 固定搜索空间未见架构 | 测试模型是未见采样点，但来自同一 NAS/search space | BRP-NAS、PerfSAGE |
| variant-level split | 同一 `variant_name` 不跨 split | 本项目应该优先补齐 |
| model-family split | 测试集模型族训练中完全没有 | DNNPerf 明确采用 |
| design-space split | 某一 task/search space 完整留出 | PerfSAGE 没有作为主实验，但适合本项目扩展 |
| hardware split | 训练硬件和测试硬件不同 | HELP、Habitat、NeuSight |
| runtime/compiler split | 框架版本、后端优化或 kernel fusion 规则变化 | nn-Meter 和 NeuSight 的问题意识更强 |

对本项目而言，只有行级随机 split 会高估可泛化性。更稳的评估矩阵是：

| 评估名 | train/test 关系 | 用途 |
| --- | --- | --- |
| row-random | 样本行随机 | 观察当前数据分布下的拟合稳定性 |
| variant-group | `variant_name` 不跨 split | 防止同一模型配置泄漏 |
| family-holdout | 完整留出 `model_family` | 评估未见模型类型泛化 |
| domain-holdout | 留出 vision/text/detection/recommender 之一 | 评估跨任务域泛化 |
| hardware-holdout | 留出 A100 或 V100 | 评估硬件迁移 |
| time/runtime-holdout | 留出后采集批次或框架版本 | 评估长期可维护性 |

## 与现有研究的相同点

本项目与 PerfSeer、DNNPerf、DIPPM 的共同点：

- 都把深度学习模型视为计算图，做图级回归。
- 都使用真实 profiling 或监控结果作为标签，而不是只用 FLOPs、params 这类 proxy。
- 都关注部署前预测，目标服务于资源调度、批量实验筛选、OOM 风险降低或架构搜索。
- 都需要处理不同模型规模、不同输入 shape、不同 batch 和不同运行阶段带来的性能变化。

本项目与 PerfSeer 的共同点最强：

- 都使用 ONNX 作为跨框架图表示入口。
- 都预测训练和推理阶段性能。
- 都把 time、memory、SM/utilization 一类 GPU 指标作为关键目标。
- 都可以走多目标预测，而不是为每个指标训练完全独立模型。

## 与现有研究的差异

### 模型族覆盖

多数现有数据集主要集中在 CNN、NAS vision search space 或少量 RNN/Transformer。当前项目覆盖更宽：

- 视觉分类：ResNet、VGG、DenseNet、MobileNet、EfficientNet、Inception、Swin、ViT、BEiT、ConvNeXt。
- 检测：YOLOv5、YOLOv9、YOLOv11 等。
- 文本：BERT、GPT-2、T5。
- 推荐：DeepFM、DCN，并计划扩展 DCNv2、EDCN、AutoInt、FiBiNet。

这可以成为项目差异点，但也会放大分布不均和 out-of-family 泛化难度。

### 目标指标

现有工作常见目标是 latency/time、memory、energy/power。当前项目多了 CPU cores、GPU utilization、SM occupancy 的 P95 口径，更贴近集群调度和资源画像。

需要注意的是，P95 资源指标更容易受监控采样间隔、后台噪声、容器隔离、Prometheus/DCGM 更新频率影响。论文结果里最好同时报告标签采集协议，否则指标可重复性会弱于 latency/memory。

### 训练阶段表示

DNNPerf 显式讨论 forward/backward graph 和训练迭代；当前项目训练样本仍主要由 ONNX inference graph 加 phase token 表示。这样可以学习到训练和推理的平均差异，但反向传播、optimizer state、activation liveness、gradient tensor、mixed precision、activation checkpointing 等训练特有因素没有进入图结构。

这对 `duration_sec_avg` 和 `gpu_mem_used_mb_p95` 的训练阶段预测尤其关键。后续可以考虑加入训练扩展特征，至少包括：

- 参数量、activation memory 估计、optimizer state 倍数。
- 每类 op 的 backward cost 近似。
- 是否使用 AMP、checkpointing、fused optimizer。
- 训练 batch、sequence length、detection target size、推荐 sparse fields 等任务输入特征。

### 图特征丰富度

PerfSeer 和 DNNPerf 都强调 op type、hyperparameter、tensor shape、FLOPs、memory access、edge tensor 信息和 device feature。当前项目特征更轻，尤其缺少：

- 显式 op type embedding。
- 完整 tensor shape 向量，只保留 rank、element count 和 bytes。
- arithmetic intensity、memory access cost、weight/input/output tensor 分解。
- op 级占全图比例特征。
- global graph density、op histogram、critical path 或 topo depth。
- runtime/compiler/backend 特征。

这可能解释为什么推荐模型、GPT-2/T5 比视觉模型更难：不同语义的 op 可能在当前特征里被压成相近的 MACs/memory/counts。

### 硬件和运行时范围

当前项目已有 A100/V100 GPU specs 字段，但现有报告主要体现 V100 数据。PerfSAGE、nn-Meter、HELP、Habitat、NeuSight 都说明，跨硬件泛化不能只靠同硬件随机 split 证明。

如果本项目要主张硬件可迁移，需要单独构造：

- V100 train / A100 test。
- A100 train / V100 test。
- V100+A100 train / new GPU test。
- 少量 target GPU samples fine-tune 或 calibration。

### 数据生成方式

本项目大量样本来自人工配置网格、架构 mutation 和 fake batch。PerfSAGE 也使用随机初始化权重和 sampled model graph，因此“不是完整真实训练任务”并不自动构成问题。但需要在文档中明确：

- 标签代表固定输入形状和固定 phase rounds 下的 step-level runtime，不代表完整任务收敛成本。
- 推荐模型的 `phase_rounds` 是 step/batch 口径，不是真正完整 epoch。
- 检测、文本、推荐模型使用 fake input，数据 pipeline 开销不在标签内。

这个边界越清楚，越不容易被误解为端到端训练成本预测。

## 值得优先考虑的实验与改进

### 划分协议

建议优先补齐三套 split：

| 优先级 | split | 原因 |
| --- | --- | --- |
| P0 | `variant_name` group split | 立即消除当前报告已确认的变体泄漏 |
| P0 | train-only scaler fit | 避免 scaler 使用 test 分布信息 |
| P1 | `model_family` holdout | 回答未见模型类型泛化问题 |
| P1 | phase-stratified split | 防止 training/inference 比例漂移 |
| P2 | hardware holdout | 支撑跨 GPU 泛化声明 |
| P2 | time-based split | 模拟后续新增模型和新增采集批次 |

### 数据平衡

当前 YOLO 样本多，推荐和文本困难族样本少。PerfSAGE 证明 upsampling 可以改善小 design space 的 generalized model。建议评估：

- family-balanced sampler。
- target-range balanced sampler，尤其长耗时和高显存区间。
- phase-balanced sampler。
- hard-family upweighting，但验证集和 checkpoint 选择要保持目标口径一致。

### 基线与消融

论文或报告至少应包含以下基线：

| 基线 | 目的 |
| --- | --- |
| global MLP/XGBoost | 只用 params、FLOPs、memory、batch、GPU specs，证明 GNN 有必要 |
| op histogram + MLP | 评估 op type 分布是否足够 |
| node-only GNN | 对照 edge features 价值 |
| node+edge GNN | 对照 graph/global features 价值 |
| current model | 作为现有实现基线 |
| PerfSeer-style feature set | 评估更丰富 node/edge/global features 的收益 |

当前模型已经有节点、边、图级状态，可以通过逐步加特征而不是换模型来做最小可控实验。

### 目标建模

建议同时报告：

- per-target MAE/RMSE/WAPE/R2。
- per-family、per-phase 指标。
- 对小分母敏感的 inference latency，单独报告 MAE、median AE、P95 AE，不只看 MAPE/sMAPE。
- OOM 或 boundary 配置单独做分类/风险预测，不要混入常规回归。

### 运行时因素

nn-Meter 和 NeuSight 都说明，图结构不等于实际执行计划。当前项目可以逐步加入：

- framework/runtime 标识：PyTorch、Transformers、Ultralytics、Torch-RecHub、ONNX Runtime 等。
- dtype / AMP / TF32 / cudnn benchmark / torch compile 状态。
- operator fusion 或 compiled graph 统计。
- GPU 独占状态、MIG、power cap、driver/CUDA/cuDNN 版本。
- warmup 次数、采样窗口和 phase rounds。

### 推荐模型专项

推荐模型当前是误差高风险区。与视觉模型相比，推荐模型常见瓶颈是 embedding lookup、sparse fields、feature interaction 和 memory bandwidth，FLOPs 对耗时解释力弱。建议给推荐模型增加显式特征：

- sparse feature 数、dense feature 数。
- embedding table 数、总 vocab、embed dim、embedding 参数量。
- cross/interaction 层数和 attention head 数。
- 每 step 输入 token/index 数。
- embedding 参数占比和 MLP 参数占比。

这类特征可以先放 graph-level，不必立即改 ONNX node 编码。

## 可以形成的项目贡献点

如果后续补齐实验，本项目相比现有工作可以强调：

- 比 PerfSAGE/nn-Meter 更贴近云端 GPU 训练和推理，而不是只做边缘推理。
- 比 DNNPerf 覆盖更多现代模型族，包括 detector、Transformer 文本模型和推荐模型。
- 比 DIPPM 目标更贴近集群资源调度，包含 CPU、GPU utilization、SM occupancy、GPU memory P95。
- 比 PerfSeer 更强调跨任务域模型族和推荐模型/文本模型的资源行为。
- 如果加入 family-holdout 和 hardware-holdout，可在泛化评估上比多数随机 split 工作更严格。

这些贡献成立的前提是 split、scaler、采集协议和 baseline 做严谨，否则主张会被当前行级 split 削弱。

## 风险清单

| 风险 | 影响 | 建议 |
| --- | --- | --- |
| `variant_name` 泄漏 | 高估 test 指标 | 先做 group split |
| scaler 用全量数据拟合 | test 信息泄漏 | 只用 train fit scaler |
| ONNX inference graph 预测 training | 训练显存和耗时解释力不足 | 加训练特征或 backward 近似 |
| op type 缺失 | 不同模型族被压成相近数值特征 | 加 op type embedding / histogram |
| 模型族不均衡 | 大族指标掩盖小族失败 | 报告 family macro 指标 |
| 低耗时 inference 小分母 | MAPE/sMAPE 失真 | 同时报 MAE、P95 AE、WAPE |
| runtime 优化不可见 | 跨框架/跨硬件泛化弱 | 加 runtime/backend 特征 |
| 推荐模型 embedding-heavy | FLOPs 解释力弱 | 加推荐专项图级特征 |

## 建议的下一步

- 先实现 `variant_name` group split 和 train-only scaler，复跑当前等权基线。
- 在同一数据上补一个 `model_family` holdout 实验，优先留出 `gpt2`、`t5`、`recommender` 和一个视觉族。
- 加一版 op type histogram + graph-level 推荐专项特征，先验证是否改善 `deepfm/dcn/gpt2/t5`。
- 把论文对照中的 baseline 做成实验表：global MLP、XGBoost、node-only GNN、node+edge+graph GNN。
- 将所有报告固定为 micro 指标和 family macro 指标并列，避免样本量大的 YOLO 主导结论。

## 参考资料

- PerfSeer: An Efficient and Accurate Deep Learning Models Performance Predictor, IJCAI 2025. <https://www.ijcai.org/proceedings/2025/793>
- PerfSeer PDF. <https://www.ijcai.org/proceedings/2025/0793.pdf>
- PerfSeer GitHub. <https://github.com/upuuuuuu/PerfSeer>
- Runtime Performance Prediction for Deep Learning Models with Graph Neural Network, DNNPerf. <https://www.microsoft.com/en-us/research/wp-content/uploads/2021/02/dnnperf.pdf>
- PerfSAGE: Generalized Inference Performance Predictor for Arbitrary Deep Learning Models on Edge Devices. <https://arxiv.org/pdf/2301.10999>
- DIPPM: a Deep Learning Inference Performance Predictive Model using Graph Neural Networks. <https://arxiv.org/pdf/2303.11733>
- nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices. <https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf>
- BRP-NAS: Prediction-based NAS using GCNs. <https://proceedings.neurips.cc/paper/2020/hash/768e78024aa8fdb9b8fe87be86f64745-Abstract.html>
- HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning. <https://arxiv.org/abs/2106.08630>
- A Runtime-Based Computational Performance Predictor for Deep Neural Network Training, Habitat. <https://arxiv.org/abs/2102.00527>
- Forecasting GPU Performance for Deep Learning Training and Inference, NeuSight. <https://arxiv.org/abs/2407.13853>
- BOOM: Use your Desktop to Accurately Predict the Performance of Large Deep Neural Networks. <https://www.cs.toronto.edu/~qdsu/papers/boom.pdf>
