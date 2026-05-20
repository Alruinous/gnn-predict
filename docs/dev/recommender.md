# 推荐模型接入说明

`recommender` 是运行时模型大类。首批接入的是 ranking/CTR 场景下的
`DeepFM`、`DCN`、`DCNv2` 和 `EDCN`，依赖 `torch-rechub==0.8.0`。

已验证的上游接口：

- `SparseFeature(name, vocab_size, embed_dim)` 构建离散特征。
- `DenseFeature(name, embed_dim=1)` 构建连续特征。
- `DeepFM(deep_features, fm_features, mlp_params)` 前向输入为特征字典。
- `DCN(features, n_cross_layers, mlp_params)` 前向输入为特征字典。
- `DCNv2(features, n_cross_layers, mlp_params, ...)` 前向输入为特征字典。
- `EDCN(features, n_cross_layers, mlp_params, ...)` 前向输入为特征字典。

接入边界：

- 配置按具体模型拆分：`deepfm_config`、`dcn_config`、`dcnv2_config`、`edcn_config`。
- builder 按具体模型拆分：`deepfm_builder.py`、`dcn_builder.py`、`dcnv2_builder.py`、`edcn_builder.py`。
- 共享特征构造、fake batch、ONNX 输入名逻辑放在 `recommender/common.py`。
- `model_kind` 仍输出 `recommender`，用于 runner 元数据和数据协议分支。
- 大规模变体通过通用 `variant_config_grid` 展开，避免手写长 YAML。

运行时协议：

- fake batch 为 `{feature_name: Tensor[B]}`。
- sparse 输入为 `LongTensor[B]`。
- dense 输入为 `FloatTensor[B]` 或 `FloatTensor[B, D]`。
- 训练目标固定为 CTR 二分类，使用 `BCELoss` 和 float label `[B]`。
- ONNX 导出使用多 runtime input，输入名按 sparse 后 dense 的配置顺序排列。
- architecture-only ONNX 写入 `runtime_input_names` 元数据，供随机初始化工具过滤参数输入。

变体文件：

- `config/arch/deepfm_variants.yaml`：252 个 DeepFM 变体。
- `config/arch/dcn_variants.yaml`：252 个 DCN 变体。
- `config/arch/dcnv2_variants.yaml`：384 个 DCNv2 变体。
- `config/arch/edcn_variants.yaml`：252 个 EDCN 变体。

推荐配置共享特征名、词表、activation、dropout 和 batch 设计。DeepFM 扫
`fm_feature_names` 三档，DCN 扫 `n_cross_layers` 三档，DCNv2 扫 cross/rank、
`model_structure` 和 expert 数，EDCN 扫 `bridge_type` 和 regulation 开关。

重要约束：

- 同一 variant 只能设置一个模型专属 config。
- 推荐模型 `example_input_shape` 固定为 `[batch]`。
- 推荐模型 `target_output_classes` 固定为 `1`。
- DeepFM 的 `fm_feature_names` 只能引用 sparse feature。
- DeepFM 的 FM 分支 sparse `embed_dim` 必须一致。
- DCN 的 `n_cross_layers` 必须为正。
- DCNv2 的 `n_cross_layers`、`low_rank` 和 `num_experts` 必须为正。
- EDCN 的 `n_cross_layers` 和 `temperature` 必须为正。
- EDCN 不暴露 `mlp_dims` 轴；`torch-rechub` 构造时会按输入维度重设 MLP 维度。
- 首批只支持单任务 CTR score，不接入 matching、multi-task 或 generative 推荐模型。
