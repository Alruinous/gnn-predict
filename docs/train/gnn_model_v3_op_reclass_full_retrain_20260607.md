# GNN 预测器 v3 算子重分类全量复训记录

## 结论

- 本次按用户更正后的口径重新在旧 v3 数据上进行，不使用 v2 数据。
- 使用临时代码重实现算子类别抽取，重新生成 `data/extracted_v3_op_reclass`、`data/scaled_v3_op_reclass` 和 `data/scalers_v3_op_reclass`。
- 新口径保留原 v3 的 `19,915` 条样本与 train/val/test split，样本数为 `11,949/3,983/3,983`。
- 新口径把旧 `op_other` 覆盖到更具体类别，并新增 `op_identity`；生成后的 `op_other` 为 `0`。
- GNN 完成 `100` epoch，best epoch 为 `99`，best val loss 为 `0.073407`。
- Test original-scale overall WAPE 为 `0.083935`，R2 为 `0.971238`，MAE 为 `39.978870`。
- 相比旧 v3 20260601 复训，overall MAE/WAPE 下降 `8.83%`，RMSE 下降 `1.59%`，R2 提升 `0.000938`；max abs error 升高 `3.77%`。
- 改进主要来自 duration、CPU、memory delta、GPU util、SM active 和 GPU memory；SM occupancy 的 WAPE 略降，但 R2 基本持平且轻微下降。
- 结论是：在 row-random v3 场景下，算子重分类能稳定改善单一 GNN predictor 的主指标；它不能证明 family-holdout 已解决，因为本次 split 不是未见模型族留出。

## 产物

| 项目 | 路径 |
| --- | --- |
| extracted data | `data/extracted_v3_op_reclass` |
| scaled data | `data/scaled_v3_op_reclass` |
| scalers | `data/scalers_v3_op_reclass` |
| effective config | `output/gnn_v3_op_reclass_full_retrain_20260607/effective_config.yaml` |
| result JSON | `output/gnn_v3_op_reclass_full_retrain_20260607/gnn_model_v3_op_reclass_20260607/results/gnn_model_v3_op_reclass_20260607_1780820687_result.json` |
| supplemental eval JSON | `output/gnn_v3_op_reclass_full_retrain_20260607/gnn_model_v3_op_reclass_20260607/results/supplemental_evaluation_20260607.json` |
| checkpoint | `output/gnn_v3_op_reclass_full_retrain_20260607/gnn_model_v3_op_reclass_20260607/checkpoints/best_model.pt` |
| train log | `output/gnn_v3_op_reclass_full_retrain_20260607/gnn_model_v3_op_reclass_20260607/logs/run_1780820687.log` |

目录大小：

| 路径 | 大小 |
| --- | ---: |
| `data/extracted_v3_op_reclass` | 2.2G |
| `data/scaled_v3_op_reclass` | 2.2G |
| `data/scalers_v3_op_reclass` | 4.5K |
| `output/gnn_v3_op_reclass_full_retrain_20260607` | 3.9M |

## 背景与问题

当前源码中算子类别映射位于 `src/gnn_model/data/onnx_graph.py` 的 `OP_TYPE_CATEGORY_BY_RAW_OP`。它只覆盖部分 ONNX raw op，缺失项通过 `resolve_op_type_category()` 统一落到 `op_other`。

这不是 ONNX 或 `onnx_tool` 完全没有这些算子实现。`onnx_tool` 在图解析、shape inference 和 profile 阶段仍能给出节点、边、shape、memory、MAC 等信息；`op_other` 的直接原因是项目自己的粗粒度语义分类表没有覆盖这些 raw op。也就是说，`op_other` 是本项目特征工程层面的兜底类别，不是 ONNX 图不可解析的错误类别。

旧 v3 中 `op_other` 占比很高：

| split | samples | nodes | old `op_other` count | old `op_other` ratio |
| --- | ---: | ---: | ---: | ---: |
| train | 11,949 | 7,203,153 | 1,356,673 | 0.188344 |
| val | 3,983 | 2,335,762 | 442,543 | 0.189464 |
| test | 3,983 | 2,370,947 | 451,189 | 0.190299 |

这会让一个 embedding 同时承载多种不相似语义。尤其 `Identity` 在导出的 ONNX 图中非常高频，它本身几乎不代表计算量，却会改变图拓扑、边连接和中间 tensor 生命周期。如果把它与 `Sqrt`、`Resize`、`ArgMax`、`Clip` 等都压入 `op_other`，模型只能学习一个混合 embedding，容易把 no-op、elementwise、activation、layout、reduce 的运行时含义混在一起。

## 临时特征处理口径

本次没有修改 `src/`、`config/` 或 `tests/`。特征生成和训练均使用 here-doc 临时代码完成；临时脚本没有保存到仓库。

重分类映射如下：

| raw ONNX op | 新类别 | 理由 |
| --- | --- | --- |
| `Identity` | `op_identity` | no-op 单独建类，避免高频 identity 污染其他类别 |
| `Sqrt` | `op_elementwise` | 逐元素数学算子 |
| `Mod` | `op_elementwise` | 逐元素数学算子 |
| `Not` | `op_elementwise` | 逐元素逻辑算子 |
| `Trilu` | `op_elementwise` | tensor mask/逐元素结构化处理，当前更接近 elementwise |
| `Erf` | `op_activation` | 常见于 GELU 近似路径 |
| `Clip` | `op_activation` | 激活/截断语义，常见于 bounded activation |
| `PRelu` | `op_activation` | 参数化激活 |
| `Resize` | `op_layout` | tensor layout/shape 变换 |
| `Pad` | `op_layout` | tensor layout/shape 变换 |
| `ArgMax` | `op_reduce` | reduction/index selection |

新 `op_type_names` 为：

```text
op_conv, op_dense, op_embedding, op_attention, op_norm, op_pool,
op_activation, op_elementwise, op_reduce, op_shape, op_layout,
op_join_split, op_cast, op_constant, op_other, op_identity
```

注意：这使 `OP_TYPE_COUNT` 从 `15` 变为 `16`。项目源码当前仍是 `15` 类，因此训练时使用临时代码在进程内 patch 了：

```text
gnn_model.data.constants.OP_TYPE_NAMES / OP_TYPE_TO_INDEX / OP_TYPE_COUNT
gnn_model.data.dataset.OP_TYPE_COUNT
gnn_model.data.prepared_dataset.OP_TYPE_NAMES
gnn_model.runner.OP_TYPE_COUNT
```

因此，本次结果是“临时新特征口径”的实验结果。若要把它作为可复现主线，需要把 op map 和 `op_identity` 正式合入源码，再用普通 CLI 复现。

## 数据生成

数据来源为 `data/extracted_v3`，临时代码从每条样本的 `onnx_path` 重新构图、重新抽取 node/edge/graph feature，并保留原 v3 的 label、metadata 与 split。

Manifest 关键信息：

| 项目 | 值 |
| --- | --- |
| schema_version | `4.1.0-temp` |
| feature_source | `onnx_tool_static_metrics_shape_topology_features_op_reclass_identity_v1` |
| source_data_dir | `/home/wangjh/gnn_predict/data/extracted_v3` |
| split_seed | 42 |
| val_ratio | 0.2 |
| test_ratio | 0.2 |
| sample_count | 19,915 |

生成耗时：

| split | count | duration sec |
| --- | ---: | ---: |
| train | 11,949 | 1,381.383467 |
| val | 3,983 | 433.652042 |
| test | 3,983 | 414.919353 |

样本数与原 v3 完全一致。节点数不完全一致，因为本次是基于当前代码和临时映射从 ONNX 重新抽取，而不是对旧 `.pt` 文件中的 `op_type_ids` 原地改写；因此本次对比应理解为“旧 v3 抽取口径 vs 新临时抽取口径”的端到端对比。

## Scaler

Scaler 使用现有 `gnn_model.data.scaler`：

```bash
PYTHONPATH=src uv run python -m gnn_model.data.scaler \
  --data_dir data/extracted_v3_op_reclass \
  --scaler_output_path data/scalers_v3_op_reclass \
  --scaled_data_output_path data/scaled_v3_op_reclass \
  --target_fields duration_sec_avg,cpu_cores_p95,memory_delta_gb_p95,gpu_util_percent_p95,gpu_sm_active_percent_p95,gpu_sm_occupancy_percent_p95,gpu_mem_used_mb_p95
```

## 数据验证

基础验证：

| split | samples | node dim | edge dim | graph dim | target dim | all finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 11,949 | 22 | 15 | 30 | 7 | true |
| val | 3,983 | 22 | 15 | 30 | 7 | true |
| test | 3,983 | 22 | 15 | 30 | 7 | true |

新口径下 `op_other` 已被消除：

| split | nodes | `op_other` count | `op_other` ratio | `op_identity` count | `op_identity` ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,175,205 | 0 | 0.000000 | 1,113,506 | 0.155188 |
| val | 2,326,692 | 0 | 0.000000 | 365,240 | 0.156978 |
| test | 2,360,641 | 0 | 0.000000 | 372,490 | 0.157792 |

主要 op 分布对比：

| split | old top ratios | new top ratios |
| --- | --- | --- |
| train | elementwise 0.213433, other 0.188344, constant 0.179810, conv 0.122440, activation 0.097603 | elementwise 0.229918, constant 0.178285, identity 0.155188, conv 0.122917, activation 0.113402 |
| val | elementwise 0.212044, other 0.189464, constant 0.177790, conv 0.125596, activation 0.100127 | elementwise 0.227755, constant 0.176255, identity 0.156978, conv 0.126085, activation 0.116005 |
| test | elementwise 0.211505, other 0.190299, constant 0.177666, conv 0.125286, activation 0.099578 | elementwise 0.227129, constant 0.175947, identity 0.157792, conv 0.125833, activation 0.115724 |

这个分布说明旧 `op_other` 中最大的一类是 `Identity`，但也有一部分被转移到 elementwise、activation、layout 和 reduce。新口径让模型可以把高频 no-op 与真实计算/shape 变换分开学习。

## 训练口径

训练配置保存为 `output/gnn_v3_op_reclass_full_retrain_20260607/effective_config.yaml`。

| 项目 | 值 |
| --- | ---: |
| device | `cuda:0` |
| GPU | Tesla V100 |
| batch_size | 32 |
| num_epochs | 100 |
| learning_rate | 0.0008 |
| weight_decay | 0.0001 |
| loss_weights | `[1, 1, 1, 1, 1, 2, 1]` |
| hidden_dim | 128 |
| num_layers | 2 |
| num_heads | 8 |
| dropout_rate | 0.1 |
| readout_mode | `mean_sum_max` |
| structural_context_mode | `basic` |
| parameter_count | 983,944 |

参数量比旧 v3 的 `979,328` 增加 `4,616`，原因是新增 `op_identity` 后 op embedding 和后续融合层输入发生变化。这是特征类别数变化带来的自然结果，不是额外后处理模型。

训练结果：

| best epoch | best val loss | last train loss |
| ---: | ---: | ---: |
| 99 | 0.073407 | 0.069668 |

训练耗时：

| stage | start UTC | end UTC | duration sec |
| --- | --- | --- | ---: |
| data | `2026-06-07T08:24:47.825562+00:00` | `2026-06-07T08:24:56.814208+00:00` | 8.988646 |
| model_build | `2026-06-07T08:24:56.814296+00:00` | `2026-06-07T08:24:57.287410+00:00` | 0.473114 |
| training | `2026-06-07T08:24:57.287470+00:00` | `2026-06-07T09:02:37.907695+00:00` | 2,260.620225 |
| evaluation | `2026-06-07T09:02:37.907809+00:00` | `2026-06-07T09:02:41.312479+00:00` | 3.404670 |
| full | `2026-06-07T08:24:47.825562+00:00` | `2026-06-07T09:02:41.312592+00:00` | 2,273.487030 |

部分 epoch 轨迹：

| epoch | train loss | val loss |
| ---: | ---: | ---: |
| 1 | 0.252019 | 0.225541 |
| 5 | 0.130819 | 0.117294 |
| 10 | 0.150176 | 0.174292 |
| 20 | 0.104613 | 0.110455 |
| 30 | 0.098058 | 0.099980 |
| 40 | 0.082812 | 0.104121 |
| 50 | 0.081437 | 0.082492 |
| 60 | 0.097706 | 0.083316 |
| 70 | 0.073069 | 0.079036 |
| 80 | 0.070492 | 0.077730 |
| 90 | 0.070458 | 0.074211 |
| 99 | 0.069191 | 0.073407 |
| 100 | 0.069668 | 0.117503 |

## Test Overall

| scale | MAE | MSE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 0.147129 | 0.872478 | 0.934065 | 0.880639 | 0.193098 | 37.044819 |
| original | 39.978870 | 50,974.882812 | 225.776184 | 0.971238 | 0.083935 | 10,968.582031 |

## Split 指标

| split | count | original MAE | original RMSE | original R2 | original WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 11,949 | 35.513847 | 180.668839 | 0.981495 | 0.074794 | 6,567.434570 |
| val | 3,983 | 40.998566 | 224.449448 | 0.972149 | 0.084758 | 7,025.576172 |
| test | 3,983 | 39.978870 | 225.776184 | 0.971238 | 0.083935 | 10,968.582031 |

train、val、test 的 WAPE 接近，说明本次 row-random v3 split 仍是 seen-distribution interpolation。它与 `data/scaled_unseen_densenet` 那种 family-holdout 评估不是同一难度。

## Test By Target

| target | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.003657 | 0.010746 | 0.984784 | 0.046139 | 0.320441 |
| `cpu_cores_p95` | 0.059710 | 0.272433 | 0.997569 | 0.025039 | 3.533966 |
| `memory_delta_gb_p95` | 0.214695 | 1.089102 | 0.654944 | 0.113861 | 16.707212 |
| `gpu_util_percent_p95` | 6.134938 | 10.118466 | 0.870140 | 0.106578 | 59.523697 |
| `gpu_sm_active_percent_p95` | 0.050458 | 0.085002 | 0.907900 | 0.104662 | 0.550731 |
| `gpu_sm_occupancy_percent_p95` | 0.029246 | 0.055233 | 0.841178 | 0.155388 | 0.442067 |
| `gpu_mem_used_mb_p95` | 273.359406 | 597.260864 | 0.891472 | 0.083556 | 10,968.582031 |

主要误差仍来自 `gpu_mem_used_mb_p95` 的原始量纲绝对值；相对误差最高的目标仍是 `gpu_sm_occupancy_percent_p95`、`memory_delta_gb_p95` 和 GPU/SM 利用率相关指标。

## Test By Phase

| phase | count | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 1,989 | 47.698238 | 233.409424 | 0.970731 | 0.097267 | 8,181.910645 |
| inference | 1,994 | 32.278862 | 217.895844 | 0.971791 | 0.069827 | 10,968.582031 |

training 仍明显难于 inference，符合过往记录：训练阶段包含 backward/optimizer/resource interaction，当前特征只显式表达 ONNX forward graph 和 `phase_token_id`，对训练态的运行时行为仍是弱表示。

## Test By Family

Top family 指标：

| family | count | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo11` | 1,114 | 24.307707 | 112.092628 | 0.991020 | 0.055253 | 2,428.835205 |
| `densenet121` | 241 | 22.179276 | 100.452209 | 0.989952 | 0.053805 | 1,037.785156 |
| `vgg11` | 226 | 76.764961 | 414.023102 | 0.940105 | 0.115287 | 5,293.998047 |
| `vgg19` | 207 | 87.528313 | 378.494293 | 0.968679 | 0.100991 | 6,043.041992 |
| `vgg16` | 196 | 69.722084 | 308.146118 | 0.979157 | 0.080954 | 4,320.980469 |
| `yolov5` | 185 | 23.266991 | 87.918510 | 0.971730 | 0.109604 | 1,061.542603 |
| `resnet18` | 168 | 19.949110 | 57.103210 | 0.976744 | 0.119146 | 500.072876 |
| `mobilenetv2` | 144 | 13.828685 | 57.367554 | 0.990673 | 0.055901 | 630.331055 |
| `bert-base-uncased` | 131 | 26.040924 | 124.878555 | 0.976781 | 0.078432 | 1,713.387451 |
| `beit` | 123 | 29.302914 | 115.443581 | 0.978394 | 0.091505 | 1,446.879639 |
| `efficientnet` | 110 | 25.066946 | 78.862816 | 0.992910 | 0.066333 | 802.854248 |
| `gpt2` | 101 | 62.609955 | 460.675690 | 0.759477 | 0.210201 | 8,181.910645 |
| `convnext` | 98 | 132.385696 | 637.390503 | 0.907215 | 0.170506 | 10,968.582031 |

本次 `densenet121` WAPE 为 `0.053805`，但这不能与 DenseNet unseen 的 `0.222993` 直接等价比较。这里的 v3 split 中 DenseNet 不是 holdout family，train/val/test 都来自同一 row-random 主线分布；DenseNet unseen 则是 train/val 完全不含 DenseNet 的 extrapolation。

## 与旧 v3 复训对比

旧 v3 复训使用：

```text
output/gnn_model_v3_full_retrain_20260601/gnn_model_scaled_v3_20260531/results/gnn_model_scaled_v3_20260531_1780381836_result.json
```

总体指标：

| 指标 | 旧 v3 20260601 | 新 op-reclass v3 | 变化 |
| --- | ---: | ---: | ---: |
| original MAE | 43.852512 | 39.978870 | -8.83% |
| original RMSE | 229.429550 | 225.776184 | -1.59% |
| original R2 | 0.970300 | 0.971238 | +0.000938 |
| original WAPE | 0.092068 | 0.083935 | -8.83% |
| max abs error | 10,570.238281 | 10,968.582031 | +3.77% |
| best val loss | 0.075801 | 0.073407 | -3.16% |
| best epoch | 77 | 99 | +22 |
| parameter_count | 979,328 | 983,944 | +4,616 |

Per-target WAPE：

| target | old WAPE | new WAPE | 变化 |
| --- | ---: | ---: | ---: |
| `duration_sec_avg` | 0.065012 | 0.046139 | -29.03% |
| `cpu_cores_p95` | 0.027865 | 0.025039 | -10.14% |
| `memory_delta_gb_p95` | 0.118616 | 0.113861 | -4.01% |
| `gpu_util_percent_p95` | 0.110984 | 0.106578 | -3.97% |
| `gpu_sm_active_percent_p95` | 0.109946 | 0.104662 | -4.81% |
| `gpu_sm_occupancy_percent_p95` | 0.156451 | 0.155388 | -0.68% |
| `gpu_mem_used_mb_p95` | 0.091761 | 0.083556 | -8.94% |

Per-target MAE 与 R2：

| target | old MAE | new MAE | old R2 | new R2 |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.005153 | 0.003657 | 0.978661 | 0.984784 |
| `cpu_cores_p95` | 0.066448 | 0.059710 | 0.994063 | 0.997569 |
| `memory_delta_gb_p95` | 0.223661 | 0.214695 | 0.649032 | 0.654944 |
| `gpu_util_percent_p95` | 6.388545 | 6.134938 | 0.866607 | 0.870140 |
| `gpu_sm_active_percent_p95` | 0.053006 | 0.050458 | 0.900862 | 0.907900 |
| `gpu_sm_occupancy_percent_p95` | 0.029446 | 0.029246 | 0.841281 | 0.841178 |
| `gpu_mem_used_mb_p95` | 300.201355 | 273.359406 | 0.887931 | 0.891472 |

历史主线对照：

| 基线 | WAPE | 与本次差异 |
| --- | ---: | ---: |
| docs/refactor 旧 v3 保留方案 | 0.085933 | 本次低 `2.33%` |
| docs/refactor 刷新后 v2 | 0.091725 | 本次低 `8.49%` |
| docs/refactor v2 baseline | 0.092669 | 本次低 `9.43%` |

## 为什么新特征能改善 row-random v3

本次改善的核心原因不是模型结构换了，而是输入语义噪声降低了。旧口径下 `op_other` 约占全部节点的 `19%`，已经不是一个低频兜底类。GNN 的 op embedding 会把 `op_other` 当成一个稳定类别学习，但这个类别内部混有 no-op identity、elementwise、activation、layout 和 reduce。它们对 runtime 的含义不同，却共享同一个类别 id，等价于向模型注入系统性 label noise。

新口径把 `Identity` 单独拆出后，三份 split 中 `op_identity` 均约为 `15.5%-15.8%`。这很重要：高频 no-op 不再主导 `op_other`，也不会与 `Resize/Pad/Sqrt/ArgMax/Clip/PRelu` 等真实操作共享 embedding。对于使用 `TransformerConv` 和 mean/sum/max readout 的图级回归器，这会让 message passing 和 graph pooling 中的节点语义更稳定。

从指标看，收益最明显的是 `duration_sec_avg` 和 `gpu_mem_used_mb_p95`。这符合预期：

- duration 对图中细粒度 op mix 很敏感，粗糙的 `op_other` 会弱化不同 lightweight op 的语义差异。
- GPU memory 不只受参数量和激活量影响，也受 layout/identity/concat 后的 tensor 生命周期影响；把 identity 与 layout 类 op 分开后，图结构摘要更容易形成稳定关系。
- GPU util、SM active、SM occupancy 的收益较小，因为这些目标还受到 kernel scheduling、cuDNN/cuBLAS algorithm、runtime overlap 和训练态 backward 行为影响，当前特征仍没有完整表达。

`max_abs_error` 变差说明新口径没有消除少数极端样本。family 表中 `convnext` 和 `gpt2` 仍有较高 WAPE 或较大 max error，表明剩余误差不只是 `op_other` 问题，还包括模型族特定结构、训练态资源行为、以及当前静态图特征无法覆盖的 runtime 因素。

## 与 Family Holdout 的关系

本次 v3 op-reclass 实验不能推翻 DenseNet unseen 文档中的结论。原因是两套评估协议不同：

| 评估 | train/val/test 关系 | 主要问题 |
| --- | --- | --- |
| 本次 v3 op-reclass | row-random split，同一模型族可跨 split | seen-family interpolation |
| DenseNet unseen | train/val 不含 DenseNet，test 全部 DenseNet | out-of-family extrapolation |

在本次 row-random v3 中，`densenet121` test WAPE 为 `0.053805`，表现很好；但这主要说明当 DenseNet 或相近样本已参与训练分布时，重分类后的 GNN 能在 DenseNet 附近插值。DenseNet unseen 的 test WAPE 为 `0.222993`，对应的是完全没有 DenseNet 训练信号时的外推失败。

因此更准确的论文表述应是：

> Refining the operator taxonomy improves in-distribution prediction on the v3 row-random split by reducing the semantic heterogeneity of the catch-all `op_other` category. However, this improvement addresses feature ambiguity within the seen-family distribution and does not by itself solve model-family holdout generalization. The latter remains an extrapolation problem where the predictor must transfer structural-runtime relationships to an unseen architecture family.

如果后续要验证它对 family holdout 的影响，需要重新生成 `data/extracted_unseen_densenet` 的 op-reclass 版本，并在 train/val 无 DenseNet 的协议下重新训练与评估。只看本次 v3 row-random 指标不能判断 family-holdout 是否明显改善。

## 实验 Caveats

- 本次是临时特征口径，源码常量仍是 15 类；训练可复现需要运行时 patch 或正式合入 `op_identity`。
- 新旧数据不是简单替换 `op_type_ids`，而是从 ONNX 重新抽取，节点总数存在轻微差异。
- Scaler 按项目现有 `gnn_model.data.scaler` 口径生成；本记录只比较同类主线训练结果，不引入 stacker、校准器或目标侧后处理。
- 本次只训练单一多目标 GNN predictor，没有使用 residual stacker，因此指标可与旧 v3 单模型直接比较。
- `op_identity` 增加了参数量，严格说收益来自“算子分类更细 + 模型输入维度对应扩展”的端到端组合。

## 验证记录

- `data/extracted_v3_op_reclass/manifest.json` 写入成功，split counts 为 `11,949/3,983/3,983`。
- `data/scaled_v3_op_reclass` 抽样检查无 NaN/Inf，node/edge/graph/target 维度为 `22/15/30/7`。
- 新 split 的 `op_other` count 均为 `0`。
- 训练完成 `100` epoch，并写出 result JSON、supplemental eval JSON、checkpoint 和 train log。
- 训练过程中最初按 v2 启动的尝试已在用户更正后停止，未保留 v2 op-reclass 数据目录或 v2 训练输出。
- 临时代码未保存到仓库；持久产物仅为本节列出的数据、scaler、训练输出、config 和本文档。
