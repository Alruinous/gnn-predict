# GNN 预测器新数据训练评估报告

## 执行摘要

本次使用当前已生成的 `data/scaled` 新数据集重新训练并评估
`src/gnn_model` 中的 GNN 性能预测器。数据集来自 2026-05-04 生成的
`data/extracted/manifest.json`，共 `18,677` 条样本，较 2026-04-30 报告中的
`18,239` 条增加 `438` 条。

训练目标沿用上一轮 6 项：

- `duration_sec_avg`
- `cpu_cores_p95`
- `memory_gb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

训练完整跑完 100 epoch，设备为 `cuda:0` 上的
`Tesla V100-SXM2-32GB`。临时训练配置、checkpoint、日志和 result JSON 均位于
`/tmp/gnn_model_newdata_20260504_trdk8bdd`，报告生成后已删除。

核心结论：

- 本轮训练未发现跳过数据、异常过快或训练中断，平均约 `21.56s/epoch`。
- best checkpoint 出现在 epoch 97，说明 100 epoch 仍有实际收益。
- 相比 2026-04-30 训练，本轮数据增加不大，但 test 指标整体变差，尤其是
  `duration_sec_avg`。
- 追加 no-02 对照实验后，去掉 `*_02_monitor.csv` 的 440 条样本，best val loss 从
  `33.364559` 降到 `29.509258`，test original-scale WAPE 从 `0.100086` 降到
  `0.087108`。
- `cpu_cores_p95` 的 scaler scale 仍约为 `0.001`，normalized loss 继续被 CPU
  目标强烈主导。
- 当前评估仍偏乐观：split 间存在大量 `variant_name` 重叠，且 scaler 由全量 split
  共同拟合。

## 运行配置

本次没有修改仓库内的 `config/gnn_model/scaled_training.yaml`，因为该文件仍是 4 目标配置。
运行时使用临时 YAML，等价配置如下：

```yaml
experiment_name: gnn_model_scaled_train_newdata_20260504
data:
  kind: split
  data_dir: data/scaled
  target_names:
    - duration_sec_avg
    - cpu_cores_p95
    - memory_gb_p95
    - gpu_util_percent_p95
    - gpu_sm_occupancy_percent_p95
    - gpu_mem_used_mb_p95
  scaler_dir: data/scalers
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout_rate: 0.1
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
```

数据契约校验通过：

| split | 样本数 | target shape |
| --- | ---: | --- |
| train | 11,207 | `(1, 6)` |
| val | 3,735 | `(1, 6)` |
| test | 3,735 | `(1, 6)` |

`data/scalers/target_scalers.pkl` 包含全部 6 个目标的 scaler。

## 与上一轮对比

上一轮报告路径：

`docs/agent/gnn_model_newdata_training_eval_2026-04-30.md`

| 项目 | 2026-04-30 | 2026-05-04 |
| --- | ---: | ---: |
| train 样本数 | 10,943 | 11,207 |
| val 样本数 | 3,648 | 3,735 |
| test 样本数 | 3,648 | 3,735 |
| 总样本数 | 18,239 | 18,677 |
| epoch 数 | 100 | 100 |
| batch size | 32 | 32 |
| 训练耗时 | 2,107.647s | 2,156.023s |
| 平均 epoch 耗时 | 21.000s | 21.560s |
| best epoch | 97 | 97 |
| best val loss | 24.163776 | 33.364559 |
| last train loss | 31.388226 | 40.795914 |

新增 `*_02` CSV 在当前数据集中贡献 `440` 条记录，但部分旧 CSV 的有效记录数略有变化，
最终净增 `438` 条。

## 训练过程

训练时间窗口：

| 阶段 | 起止时间 UTC | 耗时 |
| --- | --- | ---: |
| data load | 14:56:57 - 14:57:05 | 8.207s |
| training | 14:57:05 - 15:33:01 | 2,156.023s |
| evaluation | 15:33:01 - 15:33:03 | 1.962s |
| full run | 14:56:57 - 15:33:03 | 2,166.465s |

关键 epoch：

| epoch | train loss | val loss | 说明 |
| ---: | ---: | ---: | --- |
| 1 | 586.912495 | 423.471405 | CPU normalized 目标主导初始 loss |
| 10 | 63.025528 | 59.709660 | 快速下降后进入震荡区间 |
| 20 | 58.128912 | 46.541794 | 前 20 轮继续改善 |
| 31 | 51.100288 | 40.867298 | 中段刷新低点 |
| 48 | 49.569977 | 37.959576 | 后半程继续收益 |
| 57 | 45.187799 | 36.911617 | val loss 进入 36 区间 |
| 66 | 45.646778 | 35.185505 | 后半程明显刷新 |
| 73 | 43.452556 | 34.818283 | 最后四分之一前继续改善 |
| 85 | 43.748294 | 34.429184 | late-stage checkpoint 改善 |
| 95 | 42.791523 | 34.372318 | 接近最终 best |
| 96 | 41.906578 | 33.968761 | 首次低于 34 |
| 97 | 42.323568 | 33.364559 | best checkpoint |
| 100 | 40.795914 | 35.170219 | 最后一轮未刷新 |

训练曲线特征：

- `val_loss` 在后半程仍持续刷新，best checkpoint 出现在 epoch 97。
- 最后一轮 train loss 是全程较低值，但 val loss 未刷新，最终评估使用 epoch 97 的 checkpoint。
- normalized aggregate loss 不适合直接解释业务误差，因为 `cpu_cores_p95` 的 scaler scale 极小。

## Test 指标

以下指标来自本次 result JSON 的 test split 评估。

### 整体指标

| 指标 | normalized | original-scale |
| --- | ---: | ---: |
| MAE | 35.482582 | 55.142548 |
| RMSE | 301.536377 | 239.449371 |
| WAPE | 0.060405 | 0.100086 |
| max abs error | 13,863.145508 | 6,601.460938 |

整体 original-scale 指标混合了秒、CPU 核、GB、百分比、MB 等不同单位，只适合粗略观察。

### 分目标 original-scale 指标

| target | MAE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 4.038074s | 7.778777s | 0.331406 | 68.102440s |
| `cpu_cores_p95` | 0.211776 cores | 0.738643 | 0.046838 | 13.863793 |
| `memory_gb_p95` | 0.731422 GB | 1.760353 | 0.280431 | 16.208294 GB |
| `gpu_util_percent_p95` | 6.619610 | 11.092913 | 0.112553 | 101.003433 |
| `gpu_sm_occupancy_percent_p95` | 0.035566 | 0.059560 | 0.187957 | 0.618868 |
| `gpu_mem_used_mb_p95` | 319.218872 MB | 586.369202 MB | 0.098909 | 6,601.460938 MB |

### 分目标 normalized 指标

| target | MAE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.216347 | 0.416761 | 0.331754 | 3.648701 |
| `cpu_cores_p95` | 211.766403 | 738.609009 | 0.060138 | 13,863.145508 |
| `memory_gb_p95` | 0.511126 | 1.230156 | 0.560820 | 11.326551 |
| `gpu_util_percent_p95` | 0.112197 | 0.188015 | 0.257571 | 1.711923 |
| `gpu_sm_occupancy_percent_p95` | 0.171817 | 0.287729 | 0.311473 | 2.989700 |
| `gpu_mem_used_mb_p95` | 0.117619 | 0.216054 | 0.218913 | 2.432373 |

## 指标变化观察

与 2026-04-30 报告相比：

| target | 2026-04-30 original MAE | 2026-05-04 original MAE | 变化 |
| --- | ---: | ---: | --- |
| `duration_sec_avg` | 1.166572s | 4.038074s | 明显变差 |
| `cpu_cores_p95` | 0.148305 cores | 0.211776 cores | 变差 |
| `memory_gb_p95` | 0.735548 GB | 0.731422 GB | 基本持平 |
| `gpu_util_percent_p95` | 5.576684 | 6.619610 | 变差 |
| `gpu_sm_occupancy_percent_p95` | 0.028414 | 0.035566 | 变差 |
| `gpu_mem_used_mb_p95` | 267.162720 MB | 319.218872 MB | 变差 |

本轮的 `duration_sec_avg` 退化最明显，WAPE 从上一轮 `0.087414` 升至 `0.331406`。
`memory_gb_p95` 基本持平，CPU 和 GPU 资源目标有轻中度退化。

## 去除 `*_02` CSV 对照实验

为确认 `*_02_monitor.csv` 是否是指标退化的主要来源，追加了一次对照实验：

- 从当前 `data/extracted` 的 train/val/test 中删除所有 `*_02_monitor.csv` 样本。
- 保持原有 split，不重新随机划分。
- 在临时目录中重新拟合 feature scalers 和 target scalers。
- 使用相同模型结构、训练参数和 `cuda:0`，重新训练 100 epoch。
- 临时数据、配置、checkpoint、日志和 result JSON 均在实验后删除。

### no-02 数据集

| split | 含 `*_02` 样本数 | 删除 `*_02` 后 | 删除数 |
| --- | ---: | ---: | ---: |
| train | 11,207 | 10,932 | 275 |
| val | 3,735 | 3,653 | 82 |
| test | 3,735 | 3,652 | 83 |
| total | 18,677 | 18,237 | 440 |

被删除的 `*_02` 样本来源：

| CSV | 删除样本数 |
| --- | ---: |
| `bert_02_monitor.csv` | 44 |
| `bert_large_02_monitor.csv` | 28 |
| `convnext_02_monitor.csv` | 86 |
| `densenet_02_monitor.csv` | 73 |
| `mobilenet_02_monitor.csv` | 142 |
| `vit_02_monitor.csv` | 67 |

### no-02 训练结果

| 项目 | 含 `*_02` | 去除 `*_02` |
| --- | ---: | ---: |
| train 样本数 | 11,207 | 10,932 |
| val 样本数 | 3,735 | 3,653 |
| test 样本数 | 3,735 | 3,652 |
| best epoch | 97 | 84 |
| best val loss | 33.364559 | 29.509258 |
| last train loss | 40.795914 | 40.327323 |
| 训练耗时 | 2,156.023s | 2,094.084s |
| 平均 epoch 耗时 | 21.560s | 20.941s |

no-02 关键 epoch：

| epoch | train loss | val loss | 说明 |
| ---: | ---: | ---: | --- |
| 1 | 596.440888 | 439.919250 | 初始 loss 仍由 CPU normalized 目标主导 |
| 10 | 77.111785 | 59.706944 | 前 10 轮与含 `*_02` 训练接近 |
| 20 | 54.797733 | 46.096634 | 中前段开始追平含 `*_02` 训练 |
| 28 | 51.357837 | 36.628731 | 明显低于含 `*_02` 同阶段 |
| 34 | 49.471585 | 34.097191 | 接近含 `*_02` 最终 best |
| 46 | 45.299417 | 33.960266 | 进入优于含 `*_02` 的区间 |
| 52 | 44.349648 | 31.905685 | validation 优势扩大 |
| 56 | 43.659874 | 31.751303 | 后半程继续改善 |
| 63 | 43.473606 | 29.719357 | 首次低于 30 |
| 84 | 41.820010 | 29.509258 | best checkpoint |
| 92 | 42.811831 | 29.668970 | 接近 best |
| 100 | 40.327323 | 38.874950 | 最后一轮未刷新 |

### no-02 Test 整体指标

| 指标 | 含 `*_02` original-scale | 去除 `*_02` original-scale | 变化 |
| --- | ---: | ---: | --- |
| MAE | 55.142548 | 48.541866 | 变好 |
| RMSE | 239.449371 | 230.350327 | 变好 |
| WAPE | 0.100086 | 0.087108 | 变好 |
| max abs error | 6,601.460938 | 6,769.258789 | 略差 |

| 指标 | 含 `*_02` normalized | 去除 `*_02` normalized | 变化 |
| --- | ---: | ---: | --- |
| MAE | 35.482582 | 30.439554 | 变好 |
| RMSE | 301.536377 | 235.640106 | 变好 |
| WAPE | 0.060405 | 0.051489 | 变好 |
| max abs error | 13,863.145508 | 7,471.197266 | 变好 |

### no-02 分目标 original-scale 指标

| target | 含 `*_02` MAE | 去除 `*_02` MAE | 含 `*_02` WAPE | 去除 `*_02` WAPE | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `duration_sec_avg` | 4.038074s | 1.348078s | 0.331406 | 0.108204 | 明显变好 |
| `cpu_cores_p95` | 0.211776 | 0.181681 | 0.046838 | 0.039981 | 变好 |
| `memory_gb_p95` | 0.731422 | 0.735589 | 0.280431 | 0.278571 | 基本持平 |
| `gpu_util_percent_p95` | 6.619610 | 6.635563 | 0.112553 | 0.111535 | 基本持平 |
| `gpu_sm_occupancy_percent_p95` | 0.035566 | 0.033497 | 0.187957 | 0.174081 | 变好 |
| `gpu_mem_used_mb_p95` | 319.218872 MB | 282.316772 MB | 0.098909 | 0.086488 | 变好 |

no-02 分目标完整指标：

| target | MAE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 1.348078s | 2.589775s | 0.108204 | 29.086517s |
| `cpu_cores_p95` | 0.181681 cores | 0.577224 | 0.039981 | 7.471548 |
| `memory_gb_p95` | 0.735589 GB | 1.778435 | 0.278571 | 16.134655 GB |
| `gpu_util_percent_p95` | 6.635563 | 11.139411 | 0.111535 | 97.535263 |
| `gpu_sm_occupancy_percent_p95` | 0.033497 | 0.057201 | 0.174081 | 0.594606 |
| `gpu_mem_used_mb_p95` | 282.316772 MB | 564.121765 MB | 0.086488 | 6,769.258789 MB |

### no-02 scaler 参数

| target | 含 `*_02` center | 去除 `*_02` center | 含 `*_02` scale | 去除 `*_02` scale |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.035843 | 0.049454 | 18.664844 | 18.768580 |
| `cpu_cores_p95` | 1.000000 | 1.000000 | 0.001000 | 0.001000 |
| `memory_gb_p95` | 1.523000 | 1.531000 | 1.431000 | 1.427000 |
| `gpu_util_percent_p95` | 58.000000 | 60.000000 | 59.000000 | 59.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.177000 | 0.182000 | 0.207000 | 0.208000 |
| `gpu_mem_used_mb_p95` | 2805.000000 | 2881.000000 | 2714.000000 | 2722.000000 |

去除 `*_02` 后，`duration_sec_avg`、GPU 利用率、SM occupancy 和显存 center 都向主数据集的
较高负载区域回移。

### 为什么去掉后会变好

`*_02` 数据和主数据分布差异很大，是一批明显偏短、偏轻的样本：

| target | `*_02` 均值 | no-02 均值 | `*_02` median | no-02 median |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.117796 | 12.661537 | 0.009114 | 0.049454 |
| `cpu_cores_p95` | 3.518077 | 4.674118 | 1.000000 | 1.000000 |
| `memory_gb_p95` | 1.154543 | 2.645439 | 1.106000 | 1.531000 |
| `gpu_util_percent_p95` | 27.196477 | 60.387701 | 24.000000 | 60.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.046584 | 0.196093 | 0.029000 | 0.182000 |
| `gpu_mem_used_mb_p95` | 1552.120455 | 3329.421681 | 1087.000000 | 2881.000000 |

它们影响训练的方式主要有四点：

- 样本太少但分布偏移明显：`*_02` 只有 `440 / 18,677`，约 `2.4%`，却集中在极短耗时和低 GPU
  负载区域，不能平滑扩展主分布。
- 模型族覆盖不均衡：`*_02` 只来自 BERT、BERT large、ConvNeXt、DenseNet、MobileNet、ViT，
  没有覆盖 YOLO、VGG、ResNet 等主数据大头，导致“新增数据”更像局部族群补丁。
- 耗时目标被强烈拉低：`*_02` 的 `duration_sec_avg` median 是 `0.009114s`，no-02 median 是
  `0.049454s`，均值差距超过 100 倍；这会让模型在低耗时区域多学一组很窄但不成比例的模式。
- scaler center 被拉向低负载：含 `*_02` 时 `duration_sec_avg` center 为 `0.035843`，
  no-02 为 `0.049454`；GPU 利用率 center 从 `60` 被拉到 `58`，显存 center 从 `2881` 被拉到
  `2805`。

这次 no-02 训练结果支持上述判断：删除 `*_02` 后，best val loss 从 `33.364559` 降到
`29.509258`，test original-scale WAPE 从 `0.100086` 降到 `0.087108`。改善最明显的是
`duration_sec_avg`，MAE 从 `4.038074s` 降到 `1.348078s`，说明退化核心更接近这批新增样本的
目标分布和主训练分布不一致，而非单纯样本量不足。

不过这个对照实验的 test split 也删除了 `83` 条 `*_02` 测试样本，并重新拟合了 scaler。
因此它证明的是“在 no-02 分布上重新训练和评估会更好”；模型在完整 test split 上对每个
样本是否都更好，还需要固定 test split 和 scaler，再分别训练 with-02/no-02 模型做同一
test 集评估。

## 数据诊断

### CSV 覆盖情况

当前 `data/extracted` 中进入数据集的记录数：

| CSV | 样本数 |
| --- | ---: |
| `beit_monitor.csv` | 902 |
| `bert_02_monitor.csv` | 44 |
| `bert_large_02_monitor.csv` | 28 |
| `bert_large_monitor.csv` | 272 |
| `bert_monitor.csv` | 651 |
| `convnext_02_monitor.csv` | 86 |
| `convnext_monitor.csv` | 470 |
| `densenet_02_monitor.csv` | 73 |
| `densenet_monitor.csv` | 793 |
| `mobilenet_02_monitor.csv` | 142 |
| `mobilenet_monitor.csv` | 648 |
| `resnet101_monitor.csv` | 400 |
| `resnet152_monitor.csv` | 400 |
| `resnet_monitor.csv` | 1,084 |
| `vgg11_monitor.csv` | 1,115 |
| `vgg16_monitor.csv` | 975 |
| `vgg19_monitor.csv` | 988 |
| `vit_02_monitor.csv` | 67 |
| `vit_monitor.csv` | 243 |
| `yolov11_monitor.csv` | 7,030 |
| `yolov5_monitor.csv` | 1,789 |
| `yolov9_monitor.csv` | 477 |

`*_02` CSV 合计贡献 `440` 条：

| CSV | 样本数 |
| --- | ---: |
| `bert_02_monitor.csv` | 44 |
| `bert_large_02_monitor.csv` | 28 |
| `convnext_02_monitor.csv` | 86 |
| `densenet_02_monitor.csv` | 73 |
| `mobilenet_02_monitor.csv` | 142 |
| `vit_02_monitor.csv` | 67 |

### phase 分布

| split | inference | training |
| --- | ---: | ---: |
| train | 5,603 | 5,604 |
| val | 1,935 | 1,800 |
| test | 1,902 | 1,833 |

训练和推理样本整体仍较均衡。

### YOLO 占比

| split | YOLO 样本数 |
| --- | ---: |
| train | 5,509 |
| val | 1,906 |
| test | 1,881 |

总 YOLO 样本数为 `9,296 / 18,677`，约 `49.8%`。

### split 间 variant 重叠

| overlap | 重叠 `variant_name` 数 |
| --- | ---: |
| train / val | 2,195 |
| train / test | 2,174 |
| val / test | 768 |

当前仍是行级随机拆分，同一 `variant_name` 可能跨 split 出现。这会让 test 结果偏乐观，
不能视为严格的未见架构泛化评估。

### 目标原始分布

| target | min | mean | median | max |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.001189 | 12.366027 | 0.035843 | 255.045456 |
| `cpu_cores_p95` | 0.959000 | 4.646883 | 1.000000 | 38.915001 |
| `memory_gb_p95` | 0.820000 | 2.610316 | 1.523000 | 18.100000 |
| `gpu_util_percent_p95` | 0.000000 | 59.605769 | 58.000000 | 100.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.000000 | 0.192570 | 0.177000 | 0.709000 |
| `gpu_mem_used_mb_p95` | 0.000000 | 3287.551331 | 2805.000000 | 14049.000000 |

`duration_sec_avg` 继续高度偏斜，中位数只有 `0.035843s`，最大值达到 `255.045456s`。
`cpu_cores_p95` 最大值从上一轮的约 `20.126` 提升到 `38.915`，尾部更重。

### scaler 参数

| target | center | scale |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.035843 | 18.664844 |
| `cpu_cores_p95` | 1.000000 | 0.001000 |
| `memory_gb_p95` | 1.523000 | 1.431000 |
| `gpu_util_percent_p95` | 58.000000 | 59.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.177000 | 0.207000 |
| `gpu_mem_used_mb_p95` | 2805.000000 | 2714.000000 |

`cpu_cores_p95` 的 scale 极小，原因仍是大量样本集中在 `1.0` 附近。未加权
`SmoothL1Loss` 会让 CPU 目标在 normalized 空间压倒其他目标。

## 判断

本轮训练流程本身可信：

- 训练跑满 100 epoch。
- 每个 epoch 约 `21.56s`，与数据规模和上一轮训练速度匹配。
- result JSON 包含全部 6 个目标和 `original_scale_*` 指标。
- best checkpoint 出现在 epoch 97，说明训练没有提前停滞。

但本轮模型在 test split 上的业务指标弱于 2026-04-30 结果，尤其是耗时预测。追加 no-02
对照实验后，主要判断更新为：

- `*_02` 数据是本轮退化的重要原因，尤其影响 `duration_sec_avg`。
- `*_02` 数据自身偏短、偏低负载，且只覆盖少数模型族，和主数据分布不一致。
- 去除 `*_02` 后，best val loss、overall WAPE、耗时 MAE、CPU MAE、SM occupancy MAE、显存 MAE
  均改善。
- `memory_gb_p95` 和 `gpu_util_percent_p95` 基本持平，说明退化不是所有目标均匀发生。
- `cpu_cores_p95` 尾部仍很重，normalized loss 仍难解释。
- 当前 split 方式和 scaler 拟合方式仍会泄漏部分 test 分布信息，严格泛化能力尚未单独验证。

当前更合理的训练基线是暂时排除 `*_02_monitor.csv`，或先将这些 CSV 标记为独立数据域单独评估。
后续建议按同一固定 test 集比较 with-02/no-02 模型，并优先做按 `variant_name` 或模型族分组的
split，再重新审视 CPU target 的归一化或 loss 权重。

## 追加：当前 CSV 训练参数短轮次探索

`csv/` 中的 `*_02_monitor.csv` 已迁走后，使用临时脚本重新读取当前 CSV 目录并做短轮次
参数探索。脚本只在 `/tmp` 下创建临时 raw/scaled split、scaler、checkpoint，结束后已清理。
为避免重复执行全量 ONNX 图解析，图特征复用 `data/extracted` 中与当前 CSV 的
`variant_name + phase` 匹配的缓存；当前有效 CSV 行与缓存完全对齐。

当前 CSV 原始行数为 `18,253`。按 [extract.py](../../src/gnn_model/data/extract.py) 的正式过滤规则，
`resnet101_monitor.csv` 和 `resnet152_monitor.csv` 各有 8 条新增行缺少对应 ONNX 文件，因此有效训练样本为
`18,237` 条。临时 split 规模如下：

| split | 样本数 |
| --- | ---: |
| train | 10,943 |
| val | 3,647 |
| test | 3,647 |

本次探索保持 6 个目标不变，先跑 9 组 `6` epoch scan，再选出较好的配置和基线跑 `18` epoch refine。
模型规模和训练参数逐步变化，没有同时大幅改变多个维度。

### scan 结果

| 配置 | 主要变化 | val mean target WAPE | test mean target WAPE | test original WAPE | 训练耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| `scan_cpu_w0p03` | `cpu_cores_p95` loss 权重 `0.03` | 0.163324 | 0.163673 | 0.126759 | 127.4s |
| `scan_cpu_w0p10` | `cpu_cores_p95` loss 权重 `0.10` | 0.171428 | 0.171570 | 0.127011 | 127.1s |
| `scan_batch64` | batch size `64` | 0.191689 | 0.190145 | 0.139823 | 72.5s |
| `scan_lr_7e4` | learning rate `0.0007` | 0.202412 | 0.202100 | 0.152948 | 127.1s |
| `scan_base_128_l2_lr1e3_bs32` | 原基线参数 | 0.208615 | 0.204412 | 0.149077 | 127.2s |
| `scan_lr_1p3e3` | learning rate `0.0013` | 0.209206 | 0.210819 | 0.168259 | 127.5s |
| `scan_dropout15` | dropout `0.15` | 0.209428 | 0.209431 | 0.160845 | 127.0s |
| `scan_hidden160` | hidden dim `160` | 0.222240 | 0.222020 | 0.198304 | 128.5s |
| `scan_layers3` | GNN layer `3` | 0.284744 | 0.279579 | 0.196369 | 138.6s |

scan 阶段最稳定的改善来自降低 CPU 目标权重。单独增大学习率、dropout、hidden dim 或层数没有改善；
`batch_size=64` 训练更快，但精度明显不如 CPU 降权。

### refine 结果

| 配置 | 主要变化 | val mean target WAPE | test mean target WAPE | test original WAPE | best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `refine_cpu_w0p10` | `cpu_cores_p95` loss 权重 `0.10` | 0.140810 | 0.141226 | 0.098448 | 17 |
| `refine_cpu_w0p03` | `cpu_cores_p95` loss 权重 `0.03` | 0.142679 | 0.144462 | 0.103841 | 18 |
| `refine_batch64` | batch size `64` | 0.158825 | 0.160490 | 0.155501 | 18 |
| `refine_base_128_l2_lr1e3_bs32` | 原基线参数 | 0.176693 | 0.178572 | 0.154052 | 13 |

`cpu_w0p10` 相比 18-epoch 基线明显改善：

| 指标 | 基线 | `cpu_w0p10` |
| --- | ---: | ---: |
| val mean target WAPE | 0.176693 | 0.140810 |
| test mean target WAPE | 0.178572 | 0.141226 |
| test original WAPE | 0.154052 | 0.098448 |
| `duration_sec_avg` WAPE | 0.232464 | 0.125480 |
| `cpu_cores_p95` WAPE | 0.054975 | 0.058037 |
| `gpu_mem_used_mb_p95` WAPE | 0.154035 | 0.098179 |

`cpu_w0p03` 的 CPU WAPE 略好，但 duration、overall 和显存目标都弱于 `cpu_w0p10`。
这说明 CPU 目标确实需要降权，但降到 `0.03` 已开始牺牲整体均衡。

### 推荐配置

当前最好的下一轮完整训练配置如下：

```yaml
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout_rate: 0.1
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  loss_weights:
    - 1.0
    - 0.1
    - 1.0
    - 1.0
    - 1.0
    - 1.0
```

`loss_weights` 顺序对应：

| target | weight |
| --- | ---: |
| `duration_sec_avg` | 1.0 |
| `cpu_cores_p95` | 0.1 |
| `memory_gb_p95` | 1.0 |
| `gpu_util_percent_p95` | 1.0 |
| `gpu_sm_occupancy_percent_p95` | 1.0 |
| `gpu_mem_used_mb_p95` | 1.0 |

推荐保留原模型规模而不是加大模型。短轮次结果显示当前主要瓶颈不是容量不足，而是
`cpu_cores_p95` 的 scaler scale 过小导致 normalized loss 被 CPU 目标主导。`cpu_w0p10`
在不增加参数量的情况下改善了 duration 和显存目标，同时只轻微牺牲 CPU WAPE，更适合作为下一轮
100 epoch 完整训练基线。
