# 推荐模型簇大规模配置评估

本文给出推荐模型下一轮接入的最终配置建议。结论只保留可落地的大模型配置，不再保留
早期过轻、几毫秒级或不适合作为主力数据源的探索结果。

## 结论

首批建议引入 4 个单任务 ranking 模型簇：

- `DCNv2`
- `EDCN`
- `AutoInt`
- `FiBiNet`

配置策略应以模型规模为主，而不是先调大 batch size：

- 优先增大 `sparse feature` 数量。
- 优先增大 `embed_dim`。
- 优先增大 MLP 宽度和深度。
- 对 `DCNv2/EDCN` 增大 cross 层数和 low-rank 维度。
- 对 `AutoInt` 增大 attention 层数和 head 数。
- batch 默认保持 `2048`，只作为补充维度。

最终推荐分三档：

| 档位 | 用途 | 目标耗时 | 建议占比 |
| --- | --- | --- | ---: |
| large | 主力推荐模型数据 | 单次训练约 `0.4s - 1.8s`，单次推理约 `0.05s - 0.45s` | 70% |
| xlarge | 长耗时与高显存样本 | 单次训练约 `1.4s - 3.4s`，单次推理约 `0.24s - 1.17s` | 25% |
| boundary | OOM 边界附近探测 | 只少量运行，验证上限 | 5% |

不建议继续生成大量 tiny/small 推荐模型。它们会重新制造 `1ms` 级低
`duration_sec_avg`，削弱 GNN 预测器对推荐模型资源行为的学习价值。

## 背景

当前已接入推荐模型只有 `DeepFM` 和 `DCN`。在当前 CSV 数据中，极低
`duration_sec_avg` 主要由这两个推荐模型的 inference 样本产生：

- `duration_sec_avg <= 0.002s` 的样本共 `980` 条。
- 其中 `dcn inference` 为 `480` 条，`deepfm inference` 为 `480` 条。
- 两者合计占 `97.96%`。

项目的目标耗时口径为：

```text
duration_sec_avg = duration_sec / phase_rounds
```

推荐模型训练中的 `phase_rounds` 对应训练 step/batch 数，不是真正完整数据集 epoch。
因此，要避免推荐模型样本继续落在毫秒级，需要让单次训练 step 和单次 inference step
本身变重。

## 公开尺度参考

公开推荐系统工程和论文都支持更大的 embedding-heavy 配置：

- DLRM 使用大量 categorical embedding table 和 dense bottom/top MLP，是推荐系统性能研究的标准模型。
- NVIDIA Merlin 的 DLRM block 默认 embedding 维度为 `64`，并强调 DLRM 需要 categorical features。
- AMD DLRM 示例使用 `embedding_dim=128`。
- NVIDIA Merlin distributed embeddings 文档讨论 Criteo 1TB 和 TiB 级 embedding table 的分布式训练。
- DCNv2、AutoInt、FiBiNet 都是面向 CTR/ranking 的特征交互模型，适合放大 sparse fields、embedding 和交互层。

因此，单卡 V100 32GB 上使用 `embed_dim=128/192/256`、`24/32` 个 sparse fields、
`[2048,1024,512]` 到 `[4096,2048,1024]` 级 MLP 是合理的实验尺度。不能照搬
TiB 级 embedding table，但不应停留在 `embed_dim=16/32` 的轻量玩具配置。

参考资料：

- Torch-RecHub 模型 API：<https://datawhalechina.github.io/torch-rechub/api-reference/models/>
- Torch-RecHub ranking tutorial：<https://datawhalechina.github.io/torch-rechub/tutorials/ranking/>
- DLRM paper：<https://arxiv.org/abs/1906.00091>
- DCNv2 paper：<https://arxiv.org/abs/2008.13535>
- AutoInt paper：<https://arxiv.org/abs/1810.11921>
- FiBiNet paper：<https://arxiv.org/abs/1905.09433>
- NVIDIA Merlin DLRM block：<https://nvidia-merlin.github.io/models/stable/_modules/merlin/models/tf/blocks/dlrm.html>
- NVIDIA Merlin distributed embeddings：<https://developer.nvidia.com/blog/fast-terabyte-scale-recommender-training-made-easy-with-nvidia-merlin-distributed-embeddings/>
- NVIDIA recommender best practices：<https://docs.nvidia.com/deeplearning/performance/recsys-best-practices/index.html>

## 测试口径

临时脚本通过 here-doc 执行，未落盘为项目文件。

运行环境：

| 项目 | 值 |
| --- | --- |
| device | `cuda:0` |
| GPU | `Tesla V100-PCIE-32GB` |
| torch-rechub | `0.8.0` |
| batch | `2048` |
| 主要调参方向 | `sparse_count`, `dense_count`, `embed_dim`, `mlp_dims`, `layers` |
| batch 使用策略 | 默认固定，非主调参手段 |

本轮测试刻意不把 `100ms` 当上限，而是继续探索到单次推理接近或超过 `1s` 的配置。

## Benchmark 结果

以下结果均为 V100 32GB 上单 step 平均耗时。

### DCNv2

| 配置 | 参数量 | 参数 FP32 | train/step | inference/step | peak 显存 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse `24`, dense `12`, embed `128`, MLP `[2048,1024,512]`, layers `8`, low_rank `128` | 148.57M | 594 MB | 0.38s | 0.12s | 3.89 GB | large |
| sparse `24`, dense `12`, embed `192`, MLP `[3072,1536,768]`, layers `10`, low_rank `192` | 263.30M | 1.05 GB | 0.82s | 0.16s | 7.07 GB | large |
| sparse `24`, dense `12`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, low_rank `256` | 417.84M | 1.67 GB | 1.27s | 0.50s | 11.15 GB | xlarge |
| sparse `32`, dense `16`, embed `192`, MLP `[3072,1536,768]`, layers `10`, low_rank `192` | 397.03M | 1.59 GB | 0.97s | 0.59s | 9.90 GB | xlarge |
| sparse `32`, dense `16`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, low_rank `256` | 617.14M | 2.47 GB | 1.51s | 0.56s | 15.49 GB | xlarge |

建议：

- 主力使用 `24x192` 和 `32x192`。
- 长耗时样本使用 `24x256` 或 `32x256`。
- `vocab_size` 不作为主调参方向；只放大词表会增大参数和 ONNX 文件，但对固定 batch 的单步计算耗时提升不稳定。

### EDCN

| 配置 | 参数量 | 参数 FP32 | train/step | inference/step | peak 显存 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse `24`, dense `12`, embed `128`, MLP `[2048,1024,512]`, layers `8` | 266.17M | 1.06 GB | 0.74s | 0.32s | 5.49 GB | large |
| sparse `24`, dense `12`, embed `192`, MLP `[3072,1536,768]`, layers `10` | 597.95M | 2.39 GB | 1.77s | 0.45s | 12.06 GB | xlarge |
| sparse `24`, dense `12`, embed `256`, MLP `[4096,2048,1024]`, layers `12` | 1.14B | 4.55 GB | 3.39s | 1.05s | 22.79 GB | xlarge |
| sparse `32`, dense `16`, embed `192`, MLP `[3072,1536,768]`, layers `10` | 1.04B | 4.14 GB | 2.89s | 1.17s | 20.75 GB | xlarge |
| sparse `32`, dense `16`, embed `256`, MLP `[4096,2048,1024]`, layers `12` | - | - | OOM | OOM | - | boundary |

建议：

- `EDCN` 是最适合制造秒级推荐模型样本的簇。
- 主力使用 `24x128` 和 `24x192`。
- 少量使用 `24x256` 或 `32x192` 作为 xlarge。
- 不建议使用 `32x256`，V100 32GB 已触发 OOM。

### AutoInt

| 配置 | 参数量 | 参数 FP32 | train/step | inference/step | peak 显存 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse `24`, dense `12`, embed `128`, MLP `[2048,1024,512]`, layers `8`, heads `8` | 126.39M | 506 MB | 0.41s | 0.06s | 4.66 GB | large |
| sparse `24`, dense `12`, embed `192`, MLP `[3072,1536,768]`, layers `10`, heads `8` | 199.32M | 797 MB | 0.86s | 0.12s | 7.12 GB | large |
| sparse `24`, dense `12`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, heads `8` | 279.00M | 1.12 GB | 1.42s | 0.24s | 9.96 GB | xlarge |
| sparse `32`, dense `16`, embed `192`, MLP `[3072,1536,768]`, layers `10`, heads `8` | 311.73M | 1.25 GB | 0.55s | 0.38s | 10.89 GB | large |
| sparse `32`, dense `16`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, heads `8` | 432.03M | 1.73 GB | 1.40s | 0.15s | 15.01 GB | xlarge |

建议：

- `AutoInt` 适合作为中重型 attention 推荐模型。
- 它的训练耗时能稳定进入秒级附近，但推理不如 `EDCN/FiBiNet` 慢。
- 主力使用 `24x192` 和 `32x192`。
- xlarge 使用 `24x256` 或 `32x256`。

### FiBiNet

| 配置 | 参数量 | 参数 FP32 | train/step | inference/step | peak 显存 | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse `24`, embed `128`, MLP `[2048,1024,512]` | 265.65M | 1.06 GB | 0.88s | 0.22s | 5.64 GB | large |
| sparse `24`, embed `192`, MLP `[3072,1536,768]` | 512.35M | 2.05 GB | 1.47s | 0.46s | 10.28 GB | xlarge |
| sparse `24`, embed `256`, MLP `[4096,2048,1024]` | 834.99M | 3.34 GB | 1.64s | 0.84s | 16.73 GB | xlarge |
| sparse `32`, embed `192`, MLP `[3072,1536,768]` | 885.31M | 3.54 GB | 1.89s | 1.17s | 17.91 GB | xlarge |
| sparse `32`, embed `256`, MLP `[4096,2048,1024]` | - | - | OOM | OOM | - | boundary |

建议：

- `FiBiNet` 是除 `EDCN` 外最适合制造秒级推理样本的簇。
- 主力使用 `24x128`。
- xlarge 使用 `24x192`、`24x256`、`32x192`。
- 不建议使用 `32x256`，V100 32GB 已触发 OOM。

## 推荐配置矩阵

### Large 档

用于主力数据生成，兼顾稳定性和足够高的耗时。

| 模型 | 推荐配置 |
| --- | --- |
| `DCNv2` | sparse `24`, dense `12`, embed `192`, MLP `[3072,1536,768]`, layers `10`, low_rank `192`, experts `4` |
| `EDCN` | sparse `24`, dense `12`, embed `128`, MLP `[2048,1024,512]`, layers `8` |
| `AutoInt` | sparse `24`, dense `12`, embed `192`, MLP `[3072,1536,768]`, layers `10`, heads `8` |
| `FiBiNet` | sparse `24`, embed `128`, MLP `[2048,1024,512]` |

### Xlarge 档

用于补充长耗时和高显存样本。

| 模型 | 推荐配置 |
| --- | --- |
| `DCNv2` | sparse `32`, dense `16`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, low_rank `256`, experts `4` |
| `EDCN` | sparse `24`, dense `12`, embed `256`, MLP `[4096,2048,1024]`, layers `12` |
| `AutoInt` | sparse `24`, dense `12`, embed `256`, MLP `[4096,2048,1024]`, layers `12`, heads `8` |
| `FiBiNet` | sparse `32`, embed `192`, MLP `[3072,1536,768]` |

### Boundary 档

只用于少量上限验证，不进入常规批量生成。

| 模型 | 配置 | 原因 |
| --- | --- | --- |
| `EDCN` | sparse `32`, dense `16`, embed `256`, MLP `[4096,2048,1024]`, layers `12` | OOM |
| `FiBiNet` | sparse `32`, embed `256`, MLP `[4096,2048,1024]` | OOM |

## 变体生成建议

不要为每个模型生成过多轻量变体。建议每个模型首批控制在 `96 - 160` 个有效变体：

| 轴 | 建议值 |
| --- | --- |
| scale | `large`, `xlarge` |
| sparse_count | `24`, `32` |
| dense_count | `12`, `16`，`FiBiNet` 可不使用 dense |
| embed_dim | `128`, `192`, `256` |
| mlp_dims | `[2048,1024,512]`, `[3072,1536,768]`, `[4096,2048,1024]` |
| DCNv2 layers | `8`, `10`, `12` |
| DCNv2 low_rank | 与 `embed_dim` 对齐：`128`, `192`, `256` |
| DCNv2 experts | `4`，必要时少量加 `8` |
| EDCN layers | `8`, `10`, `12` |
| AutoInt layers | `8`, `10`, `12` |
| AutoInt heads | `8` |
| FiBiNet bilinear_type | `field_interaction` |
| batch | 默认 `2048` |

`vocab_size` 可以随业务语义放大，但不作为主要耗时调节轴。固定 batch 下，词表变大主要增加
embedding table 参数、显存和 ONNX 文件体积，对单步计算耗时的提升不如 `field count`、
`embed_dim`、交互层和 MLP 明确。

## 接入边界

首批仍保持当前推荐模型协议：

- 单任务 CTR ranking。
- 输入为 `{feature_name: Tensor}`。
- sparse 输入为 `LongTensor[B]`。
- dense 输入为 `FloatTensor[B]`。
- 输出为 `[B]` CTR score。
- 训练继续使用 `BCELoss`。
- `target_output_classes` 继续为 `1`。
- 暂不引入 sequence、matching、multi-task 协议。

需要新增的工程对象：

- `DCNv2ConfigOverride`
- `EDCNConfigOverride`
- `AutoIntConfigOverride`
- `FiBiNetConfigOverride`
- 对应 builder
- 对应 `config/arch/*_variants.yaml`
- 配置展开与 smoke tests

## 最终建议

首批只做这 4 个模型簇：

```text
DCNv2, EDCN, AutoInt, FiBiNet
```

默认不要再引入 tiny/small 档推荐变体。主力使用 large，少量使用 xlarge，并保留少量
boundary 档做 OOM 边界确认。

推荐生成比例：

| 档位 | 占比 |
| --- | ---: |
| large | 70% |
| xlarge | 25% |
| boundary | 5% |

这样可以让推荐模型数据从 `1ms` 级小模型分布，迁移到更符合推荐系统实际工程形态的
embedding-heavy、feature-interaction-heavy 分布，同时仍保持首批接入的训练协议和 ONNX
抽图协议可控。
