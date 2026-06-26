# 推荐模型接入说明

`recommender` 是运行时模型大类。当前只记录项目已经接入的 CTR/ranking 模型。

## 当前范围

当前支持：

- `DeepFM`
- `DCN`
- `DCNv2`
- `EDCN`

暂不把 matching、multi-task、sequence recommender 或 generative recommender 放进这条路径。

## 工程边界

- 配置按模型拆分：`deepfm_config`、`dcn_config`、`dcnv2_config`、`edcn_config`。
- builder 按模型拆分：`deepfm_builder.py`、`dcn_builder.py`、`dcnv2_builder.py`、`edcn_builder.py`。
- 共享特征构造、fake batch、ONNX 输入名逻辑放在 `recommender/common.py`。
- `model_kind` 输出 `recommender`，用于 runner 元数据和数据协议分支。
- 大规模变体通过通用 `variant_config_grid` 展开，不手写长 YAML。

## 运行协议

- fake batch 为 `{feature_name: Tensor[B]}`。
- sparse 输入为 `LongTensor[B]`。
- dense 输入为 `FloatTensor[B]` 或 `FloatTensor[B, D]`。
- 训练目标固定为 CTR 二分类，使用 `BCELoss` 和 float label `[B]`。
- ONNX 导出使用多 runtime input。
- architecture-only ONNX 写入 `runtime_input_names` 元数据。

## 约束

- 同一 variant 只能设置一个推荐模型专属 config。
- `example_input_shape` 固定为 `[batch]`。
- `target_output_classes` 固定为 `1`。
- DeepFM 的 `fm_feature_names` 只能引用 sparse feature。
- DeepFM 的 FM 分支 sparse `embed_dim` 必须一致。
- DCN 的 `n_cross_layers` 必须为正。
- DCNv2 的 `n_cross_layers`、`low_rank` 和 `num_experts` 必须为正。
- EDCN 的 `n_cross_layers` 和 `temperature` 必须为正。
- EDCN 不暴露 `mlp_dims` 轴；`torch-rechub` 构造时会按输入维度重设 MLP 维度。

## 规模口径

推荐模型不要长期停留在过轻配置。轻量 DeepFM/DCN 很容易产生毫秒级样本，对资源预测训练价值有限。

后续扩展推荐优先调大：

- sparse feature 数量。
- embedding 维度。
- MLP 宽度和深度。
- DCNv2 / EDCN 的 cross 层数和 rank。

这些建议只作为配置方向，不在文档中固定具体样本数。具体规模应以当前 GPU、CSV 覆盖和训练目标重新验证。
