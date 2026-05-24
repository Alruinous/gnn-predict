# GNN 特征工程 TODO

## 范围

本文档汇总 P0 之外的特征工程后续任务。P0 的详细实现设计见
`docs/dev/gnn_p0_feature_engineering_implementation.md`。

优先级定义：

- P0：先补 `op_type`、tensor shape、推荐模型专项 graph features。
- P1：P0 稳定后，补强训练阶段、硬件效率和文本模型结构特征。
- P2：更依赖采集环境、runtime/backend 或更复杂图分析的增强项。

## 评估卫生前置项

这些不是新特征，但应在 P1/P2 前完成，否则特征收益难以归因。

- [ ] 实现 `variant_name` group split。
  - 同一 `variant_name` 不允许跨 train/val/test。
  - manifest 记录 split 策略和随机种子。
  - 报告中同时保留 row split 和 group split 对照。

- [ ] 修复 train-only scaler。
  - feature scalers 和 target scalers 只用 train split fit。
  - val/test 只 transform。
  - scaler 输出目录记录 fit split、target names、feature schema version。

- [ ] 固定 family macro 指标。
  - 每次报告同时输出 micro overall 和 per-family macro。
  - 推荐模型、GPT-2、T5 单独成表。
  - 小分母目标同时报告 MAE、RMSE、WAPE、P95 AE。

- [ ] 增加 feature schema 追踪。
  - manifest 写入 node/edge/graph feature names。
  - loader 校验 manifest feature names 与当前 constants 一致。
  - 旧 schema 数据失败时给出明确错误。

## P1: Arithmetic Intensity 和 Roofline Proxy

目标：让模型显式看到计算密集、访存密集、显存容量压力之间的差异。

- [ ] 新增节点级强度特征。
  - `node_macs_per_input_byte`
  - `node_macs_per_output_byte`
  - `node_macs_per_total_tensor_byte`
  - `node_param_bytes_log`
  - `node_activation_bytes_log`

- [ ] 新增图级硬件归一化特征。
  - `graph_flops_to_fp32_peak_ratio`
  - `graph_flops_to_tensor_peak_ratio`
  - `graph_activation_bytes_to_bandwidth_ratio`
  - `graph_param_bytes_to_gpu_memory_ratio`
  - `graph_activation_bytes_to_gpu_memory_ratio`

- [ ] 保持数值稳定。
  - 分母为 0 时显式输出 0。
  - bytes、params、elements 继续使用 `log1p`。
  - ratio 特征保持有限值，禁止 NaN/Inf。

- [ ] 消融验证。
  - P0 baseline。
  - P0 + arithmetic intensity。
  - 重点观察 `duration_sec_avg`、`gpu_util_percent_p95`、`gpu_mem_used_mb_p95`。

## P1: 训练阶段特征

目标：解决当前用 inference ONNX graph 加 phase token 预测 training 的表达不足。

- [ ] 新增 graph-level training proxy。
  - `is_training_phase`
  - `estimated_backward_macs_log`
  - `estimated_backward_activation_bytes_log`
  - `estimated_optimizer_state_bytes_log`
  - `estimated_gradient_bytes_log`

- [ ] 按 optimizer state 做可配置倍数。
  - SGD 默认参数状态倍数为 0。
  - Adam/AdamW 默认参数状态倍数为 2。
  - 当前结果文档没有 optimizer 类型时先用项目训练默认值，并在 manifest 记录。

- [ ] 增加 activation liveness 近似。
  - 先用节点输出 activation bytes 总和作为上界。
  - 后续再考虑拓扑生命周期区间。
  - 不在 P1 中构造完整 backward graph。

- [ ] 标记训练协议。
  - fake image/text/recommender dataset。
  - training batch size。
  - training measurement min seconds。
  - phase rounds 的语义仍只用于标签解释，不作为 batch/step 数误用。

- [ ] 消融验证。
  - training 与 inference 分开报。
  - 重点观察 training `duration_sec_avg`、`memory_gb_p95`、`gpu_mem_used_mb_p95`。

## P1: 文本模型结构特征

目标：补 GPT-2/T5 中 sequence length、attention head、FFN 宽度等结构信号。

- [ ] 从 result JSON resolved `variant_config` 提取 GPT-2 特征。
  - `text_is_gpt2`
  - `text_vocab_size_log`
  - `text_sequence_length_log`
  - `text_hidden_size_log`
  - `text_layer_count`
  - `text_head_count`
  - `text_ffn_size_log`
  - `text_attention_pair_count_log`

- [ ] 从 result JSON resolved `variant_config` 提取 T5 特征。
  - `text_is_t5`
  - `text_encoder_layer_count`
  - `text_decoder_layer_count`
  - `text_d_model_log`
  - `text_d_ff_log`
  - `text_num_heads`
  - `text_d_kv_log`
  - `text_relative_attention_bucket_count_log`

- [ ] 非文本模型输出零向量。

- [ ] 消融验证。
  - GPT-2/T5 单独 family 表。
  - 重点观察显存、duration、小分母 SM occupancy。

## P1: 数据平衡和采样

目标：避免 YOLO 等大族主导训练，困难族群被 overall 指标掩盖。

- [ ] family-balanced sampler。
  - 每个 batch 内尽量均衡 family。
  - 不改变 val/test 分布。

- [ ] phase-balanced sampler。
  - training/inference 都有稳定采样概率。
  - 报告中单独对比 per-phase 指标。

- [ ] target-range balanced sampler。
  - 对长耗时、高显存尾部适度重采样。
  - 只作用于 train split。

- [ ] hard-family upweighting 实验。
  - 单独实验，不作为默认训练策略。
  - checkpoint 选择仍以统一验证指标为准。

## P2: Runtime 和 Backend 特征

目标：记录图结构之外的执行计划因素。

- [ ] 扩展结果文档 metadata。
  - PyTorch version。
  - CUDA version。
  - cuDNN version。
  - driver version。
  - Transformers/Ultralytics/Torch-RecHub version。

- [ ] 记录执行开关。
  - dtype。
  - AMP enabled。
  - TF32 enabled。
  - `torch.compile` enabled。
  - `cudnn.benchmark` enabled。
  - ONNX export mode。

- [ ] 记录 GPU 状态。
  - MIG profile。
  - power cap。
  - exclusive process mode。
  - visible device count。

- [ ] 加入 graph-level runtime one-hot。
  - PyTorch eager。
  - Transformers eager。
  - Ultralytics。
  - Torch-RecHub。
  - ONNX Runtime 仅在真实使用时记录。

- [ ] 消融验证。
  - 同一模型图不同 runtime/backend 不应被当成同一输入。
  - 跨 GPU 实验前必须固定 runtime metadata。

## P2: 拓扑全局特征

目标：让模型看到 critical path、branch、residual、fan-in/fan-out 等结构信息。

- [ ] 图密度特征。
  - `node_count_log`
  - `edge_count_log`
  - `edge_to_node_ratio`
  - `max_in_degree`
  - `max_out_degree`

- [ ] 拓扑深度特征。
  - DAG topo depth。
  - longest path node count。
  - longest path MACs。
  - longest path activation bytes。

- [ ] branch 和 join 特征。
  - fan-out 节点比例。
  - fan-in 节点比例。
  - concat/split/layout op 比例。
  - residual-like Add 节点比例。

- [ ] 验证图假设。
  - ONNX graph 应按有向无环图处理。
  - 遇到异常环或缺失 tensor 时 fail fast。

## P2: Kernel/Fusion 近似

目标：处理“ONNX 图不等于实际 kernel 执行计划”的误差来源。

- [ ] 添加可融合 pattern 统计。
  - Conv + BatchNorm。
  - Conv/Gemm + Activation。
  - MatMul + Add。
  - LayerNorm 子图。
  - Attention 子图。

- [ ] 添加 layout overhead proxy。
  - Transpose count ratio。
  - Reshape/Squeeze/Unsqueeze count ratio。
  - Cast count ratio。
  - layout op activation bytes ratio。

- [ ] 不在 P2 中直接实现 kernel latency lookup。
  - 先只做 pattern count。
  - 若 pattern count 有收益，再考虑 kernel-level baseline。

## P2: 硬件留出和少样本校准

目标：支撑跨 GPU 泛化声明。

- [ ] 实现 hardware-holdout split。
  - V100 train / A100 test。
  - A100 train / V100 test。
  - mixed train / held-out GPU test。

- [ ] 设计 target GPU calibration。
  - 每个 family 少量样本 fine-tune。
  - 比较 0-shot 与 few-shot。

- [ ] 报告硬件维度指标。
  - per hardware。
  - per family per hardware。
  - per phase per hardware。

## 基线和消融 TODO

- [ ] global MLP。
  - 只用 graph-level features。
  - 证明 GNN 是否真的必要。

- [ ] XGBoost 或 LightGBM。
  - 只用 graph/global summary。
  - 作为非神经模型强基线。

- [ ] op histogram + MLP。
  - 验证图级 op 分布是否已经足够。

- [ ] node-only GNN。
  - 移除 edge_attr 和 graph_features。

- [ ] node + edge GNN。
  - 移除 graph_features。

- [ ] full P0/P1/P2 GNN。
  - 按阶段逐步打开特征，禁止一次性混入多类改动。

## 默认执行顺序

- [ ] 完成 P0 并重建数据。
- [ ] 修 group split 和 train-only scaler。
- [ ] 跑 P0 row split 与 P0 variant-group split。
- [ ] 加 P1 arithmetic intensity。
- [ ] 加 P1 training proxy。
- [ ] 加 P1 text config features。
- [ ] 评估采样策略。
- [ ] 加 P2 runtime/backend metadata。
- [ ] 加 P2 topology features。
- [ ] 设计 hardware-holdout 和 few-shot calibration。
