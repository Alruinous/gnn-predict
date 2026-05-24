# GNN P0 特征工程实现设计

## 结论

P0 特征工程应该先补三类信息：

- ONNX 节点的 `op_type` 语义。
- 节点、边、图级的 tensor shape 摘要。
- 推荐模型的 sparse embedding 和 feature interaction 配置特征。

这三类特征都能从当前项目已有产物稳定获得，不需要改采集协议，也不需要先换 GNN
结构。推荐实现方式是继续使用 PyG `Data` 中的数值张量：

- `data.x` 追加节点级 one-hot 和 shape 摘要。
- `data.edge_attr` 追加边 tensor shape 摘要。
- `data.graph_features` 追加 op histogram、全局 shape 摘要和推荐模型专项特征。

不要把 `op_type` 作为单个整数 ID 塞进连续特征。整数 ID 会引入不存在的大小关系，应该用
one-hot、multi-hot 或显式 embedding。为了最小改动，P0 先用固定 one-hot。

## 背景

当前 `src/gnn_model/data/onnx_graph.py` 从 ONNX 和 `onnx_tool` profile
构造图样本：

- 节点特征：MACs、memory、params、输入输出数、attr 数、入度、出度。
- 边特征：tensor bytes、rank、element count、源出度、目标入度。
- 图级特征：phase、batch、sample count、GPU specs、参数输入统计、全图
  MACs/FLOPs/memory/params。

这个 schema 的问题是语义太轻。不同瓶颈会被压成相似的数字：

- `Conv`、`Gemm`、`MatMul`、`Gather` 可能有相近 bytes 或 MACs，但实际性能瓶颈不同。
- GPT-2/T5 的 attention 代价和 sequence length 强相关，当前只通过压缩后的
  element count 暗示。
- 推荐模型的 embedding lookup 和 feature interaction 主要是 sparse/memory 行为，
  FLOPs 对耗时解释力弱。

2026-05-19 训练报告也支持这个判断：

- `deepfm`、`dcn`、`edcn` 的显存误差很大，`deepfm` test 显存 MAE 达
  `5592.768555 MB`。
- `gpt2`、`t5` 的 mean target WAPE 仍高。
- 删除困难族群后，整体 MSE/RMSE 大幅下降，说明困难族群不是随机噪声，而是当前特征
  schema 难以表达的系统性分布。

调研文档中 PerfSeer、DNNPerf、DIPPM 等工作都强调 node/edge/global
features，但这些工作可信度不完全一致。P0 不应照搬任何单篇论文的模型结构或实验口径，
只采纳共识强、可由本项目误差现象直接解释的部分：op type、shape、domain-specific
static features。

## 目标

P0 实现需要满足：

- 保持当前 `Data.x`、`Data.edge_attr`、`Data.graph_features` 三张量接口。
- 特征维度固定，由 `constants.py` 中的 feature names 唯一决定。
- 所有新特征可以从 ONNX、`onnx_tool` profile、monitor CSV、result JSON 推导。
- 新 schema 下旧 prepared/scaled 数据必须显式失效，避免维度混用。
- 非推荐模型的推荐专项特征为确定的零向量，不引入缺失值。
- 所有大范围数值特征进入 scaler 前先做 `log1p` 或比例化，降低尾部支配。

P0 不做：

- 不重写 GNN backbone。
- 不引入 learnable op embedding。
- 不构造 backward graph。
- 不建模 runtime/compiler fusion。
- 不承诺跨 model-family 或跨硬件泛化，评估协议需要单独修。

## 当前链路

数据链路如下：

```text
monitor CSV
  -> gnn_model.data.extract.process_csv()
  -> build_model_record_info()
  -> extract_feature_target()
  -> build_graph_data_from_onnx()
  -> torch.save(train/val/test .pt)
  -> scaler.py 归一化
  -> runner.py 用常量维度构建模型
```

关键文件：

| 文件 | 当前职责 | P0 变化 |
| --- | --- | --- |
| `src/gnn_model/data/constants.py` | 定义 feature names 和维度 | 新增 op、shape、recommender feature names |
| `src/gnn_model/data/onnx_graph.py` | ONNX profile 到 PyG `Data` | 构造 op type、shape、histogram 特征 |
| `src/gnn_model/data/extract.py` | CSV 到 graph samples | 从 result JSON 解析变体上下文 |
| `src/gnn_model/data/dataset.py` | split 加载和维度校验 | 自动使用新维度，旧数据失败 |
| `src/gnn_model/data/scaler.py` | feature/target scaler | 后续只允许 train split fit |
| `src/gnn_model/runner.py` | 用常量维度建模 | 常量更新后自动跟随 |

## 数据源

### ONNX 和 onnx_tool

`build_graph_data_from_onnx()` 已经加载 ONNX，并通过 `onnx_tool` 执行：

```python
graph.shape_infer(runtime_inputs)
graph.profile()
```

可直接复用的字段：

- `graph.nodemap`: node name 到 node info。
- `node_info.op_type`: ONNX op 类型。
- `node_info.input`: 输入 tensor 名列表。
- `node_info.output`: 输出 tensor 名列表。
- `node_info.attr`: ONNX attribute。
- `graph.tensormap[tensor_name].get_shape()`: tensor shape。
- `graph.tensormap[tensor_name].dtype`: tensor dtype。
- `graph.tensormap[tensor_name].get_memsize()`: tensor bytes。

注意：如果 ONNX 导出时 batch 维是静态值，当前 `runtime_inputs` 不会覆盖这个静态 batch。
因此 shape 特征不要把 profiled dim0 误认为真实运行 batch。真实 batch 继续来自 CSV 的
`batch_size` graph feature。

### monitor CSV

CSV 已提供：

- `variant_name`
- `base_model_name`
- `phase`
- `batch_size`
- `sample_count`
- `gpu_node`
- `result_json`
- `config_path`

其中 `config_path` 是原始 YAML，不适合作为变体特征源，因为 grid 已经展开，直接读 YAML
还需要重新实现展开逻辑。

### result JSON

`src/gnn_archs/result.py` 的 `VariantResult` 已保存完全展开后的：

- `name`
- `base_model_name`
- `variant_config`
- `mutations`
- `metadata.model_kind`

P0 推荐从 `result_json` 读取 `variant_config`，按 `variant_name` 找到对应
`VariantResult`。这样可以直接拿到推荐模型的 `deepfm_config`、`dcn_config`、
`dcnv2_config`、`edcn_config`，避免从 YAML 反推。

## Feature Schema 版本

新增 schema 后应同步更新：

- `manifest["schema_version"]`: 建议改为 `3.0.0`。
- `manifest["feature_source"]`: 建议改为 `onnx_tool_profile_p0_features`。
- `manifest["node_feature_names"]`
- `manifest["edge_feature_names"]`
- `manifest["graph_feature_names"]`

当前 loader 只按维度校验。P0 建议把 feature names 写入 manifest，便于训练报告和
后续消融复现实验定位具体 schema。

旧的 `data/extracted`、`data/scaled`、`data/scalers` 不能复用。更新 constants 后，
旧 `.pt` 会因为维度不匹配在 `validate_graph_data()` 中失败，这是正确行为。

## P0-A: op_type 特征

### 设计

新增固定 op category one-hot，而不是单个整数 ID。

推荐先用 category 而不是完整 raw op vocabulary，原因是：

- ONNX op 种类多，长尾明显。
- 当前数据量对稀疏 raw one-hot 不一定稳定。
- category 能先解决“Conv/Gather/MatMul/Reshape 被压成相似数值”的主问题。

推荐类别：

| 特征名 | ONNX op 示例 | 语义 |
| --- | --- | --- |
| `op_conv` | `Conv`, `ConvTranspose` | 卷积 |
| `op_dense` | `Gemm`, `MatMul` | 矩阵乘/全连接 |
| `op_embedding` | `Gather`, `GatherElements`, `GatherND` | embedding/index lookup |
| `op_attention` | `Softmax`, `Einsum` | attention 常见核心算子 |
| `op_norm` | `BatchNormalization`, `LayerNormalization`, `InstanceNormalization` | 归一化 |
| `op_pool` | `MaxPool`, `AveragePool`, `GlobalAveragePool` | 池化 |
| `op_activation` | `Relu`, `Gelu`, `Sigmoid`, `Tanh`, `Softplus`, `Elu`, `Selu` | 激活 |
| `op_elementwise` | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Where` | 逐元素计算 |
| `op_reduce` | `ReduceMean`, `ReduceSum`, `ReduceMax` | reduce |
| `op_shape` | `Shape`, `Size`, `ConstantOfShape` | shape 计算 |
| `op_layout` | `Reshape`, `Transpose`, `Flatten`, `Squeeze`, `Unsqueeze` | layout 变换 |
| `op_join_split` | `Concat`, `Split`, `Slice`, `Tile`, `Expand` | 拼接/切分/广播 |
| `op_cast` | `Cast`, `CastLike` | dtype 转换 |
| `op_constant` | `Constant` | 常量节点 |
| `op_other` | 其他 | 长尾兜底 |

每个节点追加一个 one-hot category。所有 category 特征之和必须等于 `1.0`。

### 图级 histogram

`graph_features` 同时追加每类 op 的比例：

```text
graph_op_conv_ratio
graph_op_dense_ratio
...
graph_op_other_ratio
```

只用 ratio，不先加 raw count。当前图级已有总 MACs、params、memory；ratio 更适合表达
结构组成，减少大图节点数对 histogram 的重复支配。若后续需要 raw count，可加
`log1p(count)` 做 P1 消融。

### 实现位置

在 `constants.py` 中定义：

```python
OP_TYPE_FEATURE_NAMES = (
    "op_conv",
    "op_dense",
    ...
    "op_other",
)

GRAPH_OP_HISTOGRAM_FEATURE_NAMES = tuple(
    f"graph_{name}_ratio" for name in OP_TYPE_FEATURE_NAMES
)
```

在 `onnx_graph.py` 中新增：

- `resolve_op_type_category(op_type: str) -> str`
- `build_op_type_feature_vector(op_type: str) -> list[float]`
- `build_graph_op_histogram(nodes: Iterable[Any]) -> list[float]`

`build_node_feature_vector()` 接收 `node_info.op_type` 并 append one-hot。

`build_graph_feature_vector()` 接收 `node_infos` 或 `op_histogram`，append histogram。

### 验证

新增或扩展测试：

- tiny ConvNet 中至少出现 `op_conv`、`op_activation`、`op_dense`。
- 每个节点 op one-hot sum 为 `1.0`。
- 图级 op ratio sum 约等于 `1.0`。
- 未列入 op 进入 `op_other`。

## P0-B: tensor shape 摘要

### 设计约束

shape 特征必须固定长度。推荐 `MAX_SHAPE_RANK = 6`，超过 6 维时保留前 6 维，并额外用
`rank` 和 `element_count` 表达完整复杂度。

shape 中大数值统一使用 `log1p`：

- bytes
- elements
- dim size
- non-batch elements

原因是推荐模型 vocab、embedding 参数和 GPT/T5 sequence 相关张量会有明显长尾。

### batch 处理

当前 ONNX profile shape 的 dim0 不一定等于监控行 `batch_size`。P0 不直接改
`graph.shape_infer()` 的行为，避免同时改变既有 MACs/memory 统计口径。

因此 shape 摘要分两类：

- `profile_dim0_log`: profile 中看到的 dim0，仅作为 ONNX 图事实。
- `nonbatch_elements_log`: 除 dim0 之外的元素数，更接近结构形状。

真实 runtime batch 继续使用已有 graph feature `batch_size`。

后续若要让 profile MACs 跟 runtime batch 严格一致，应作为单独改动处理，并重跑所有基线。

### 通用 helper

在 `onnx_graph.py` 中新增：

```python
MAX_SHAPE_RANK = 6

def build_shape_summary(shape: tuple[int, ...]) -> list[float]:
    ...
```

推荐输出：

| 字段 | 说明 |
| --- | --- |
| `rank` | rank 原值 |
| `element_count_log` | `log1p(prod(shape))` |
| `nonbatch_element_count_log` | `log1p(prod(shape[1:]))`，scalar 为 `0` |
| `dim0_log` ... `dim5_log` | 固定 6 维，不足补 0 |
| `is_scalar` | `rank == 0` |
| `is_zero_sized` | 任一维为 0 |

`count_elements(())` 当前返回 `1`。shape summary 可以保留这个约定，但
`is_scalar` 必须单独暴露。

### 节点级 shape 特征

节点追加输入/输出 tensor 摘要，建议先放聚合统计，不为每个 input/output 展开：

| 特征名 | 说明 |
| --- | --- |
| `input_tensor_bytes_sum_log` | 输入动态/静态 tensor bytes 总和 |
| `output_tensor_bytes_sum_log` | 输出 tensor bytes 总和 |
| `input_tensor_elements_sum_log` | 输入元素总数 |
| `output_tensor_elements_sum_log` | 输出元素总数 |
| `input_tensor_rank_max` | 输入最大 rank |
| `output_tensor_rank_max` | 输出最大 rank |
| `output_tensor_nonbatch_elements_log` | 输出非 batch 元素总数 |
| `output_tensor_dim0_log` ... `output_tensor_dim5_log` | 主输出 shape，取第一个 output |
| `output_tensor_dtype_itemsize` | 主输出 dtype bytes |

这里的“主输出”取 `node_info.output[0]`。多输出节点的总量由 sum 特征表达。

### 边级 shape 特征

边表示一个 producer output 到 consumer input 的 tensor，应该追加完整 shape 摘要：

| 特征名 | 说明 |
| --- | --- |
| `tensor_dim0_log` ... `tensor_dim5_log` | 边 tensor shape |
| `tensor_nonbatch_element_count_log` | 非 batch 元素数 |
| `tensor_dtype_itemsize` | dtype bytes |
| `tensor_is_scalar` | scalar tensor |
| `tensor_is_zero_sized` | 任一维为 0 |

当前已有 `tensor_rank`、`tensor_element_count`、`tensor_bytes`，可以保留 raw 值，新增 log
摘要用于模型更稳定地学习尾部分布。

### 图级 shape 特征

图级追加全局 shape 摘要：

| 特征名 | 说明 |
| --- | --- |
| `node_count_log` | `log1p(num_nodes)` |
| `edge_count_log` | `log1p(num_edges)` |
| `activation_bytes_sum_log` | 所有节点输出 tensor bytes 去重总和 |
| `activation_elements_sum_log` | 所有节点输出 tensor elements 去重总和 |
| `max_tensor_rank` | 全图最大 tensor rank |
| `max_tensor_dim_log` | 全图最大 dim |
| `runtime_input_count` | runtime input 数 |
| `runtime_input_elements_sum_log` | runtime input 元素总数 |
| `runtime_input_nonbatch_elements_sum_log` | runtime input 非 batch 元素总数 |

输出 tensor 去重按 tensor name 去重，避免同一 tensor 被多个下游边重复统计。

### 验证

新增或扩展测试：

- scalar shape 和 zero-length shape 的 summary 不产生 NaN。
- rank 小于 6 时 dim padding 为 0。
- rank 大于 6 时不改变 feature length。
- tiny ConvNet 的边 shape 特征包含 `dim1=channels`、`dim2/3=height/width` 的
  `log1p` 值。
- graph-level `activation_bytes_sum_log` 为正。

## P0-C: 推荐模型专项 graph features

### 设计

推荐模型的关键瓶颈来自：

- embedding table size。
- 每 step lookup 数。
- sparse field 数。
- dense feature 维度。
- feature interaction 复杂度。
- cross network / expert / bridge 配置。

这些信息在 ONNX 图里可能被弱化或展开成普通 `Gather`、`Concat`、`MatMul`。P0 应从
result JSON 的 resolved `variant_config` 直接构造 graph-level 特征。

### 上下文对象

新增一个小的 module-local 数据结构，例如 `src/gnn_model/data/variant_context.py`：

```python
@dataclass(frozen=True)
class VariantFeatureContext:
    variant_name: str
    base_model_name: str
    model_kind: str
    variant_config: dict[str, object]
    mutations: list[dict[str, object]]
```

`extract.py` 中按 `result_json` 缓存解析结果：

```text
result_json -> {variant_name -> VariantFeatureContext}
```

`extract_feature_target(info)` 把 context 传给 `build_graph_data_from_onnx()`。

生产数据中 `result_json` 必须可读且包含当前 `variant_name`。测试 fixture 应写最小
result JSON，而不是让生产路径静默降级。对非推荐模型也保留 context，因为后续 text/domain
features 会复用。

### 推荐模型识别

按 `base_model_name` 或具体 config 字段识别：

- `deepfm_config`
- `dcn_config`
- `dcnv2_config`
- `edcn_config`

只允许一个 config 生效。这和 `VariantConfig` 的现有约束一致。

非推荐模型输出全零推荐特征，同时 `recommender_is_present = 0`。

### 特征列表

推荐新增 graph features：

| 特征名 | 说明 |
| --- | --- |
| `recommender_is_present` | 是否推荐模型 |
| `recommender_is_deepfm` | DeepFM |
| `recommender_is_dcn` | DCN |
| `recommender_is_dcnv2` | DCNv2 |
| `recommender_is_edcn` | EDCN |
| `recommender_sparse_feature_count` | sparse field 数 |
| `recommender_dense_feature_count` | dense field 数 |
| `recommender_dense_total_dim_log` | dense feature 总维度 |
| `recommender_vocab_sum_log` | sparse vocab 总和 |
| `recommender_vocab_max_log` | sparse vocab 最大值 |
| `recommender_embed_dim_sum_log` | sparse embed dim 总和 |
| `recommender_embed_dim_max_log` | sparse embed dim 最大值 |
| `recommender_embedding_param_count_log` | `sum(vocab_size * embed_dim)` |
| `recommender_embedding_table_bytes_log` | embedding 参数 bytes，默认 float32 |
| `recommender_step_sparse_index_count_log` | `batch_size * sparse_feature_count` |
| `recommender_step_embedding_output_elements_log` | `batch_size * sum(embed_dim)` |
| `recommender_mlp_layer_count` | MLP 层数，EDCN 用构造约定值 |
| `recommender_mlp_units_sum_log` | MLP units 总和 |
| `recommender_mlp_param_estimate_log` | 相邻 MLP dims 乘积估算 |
| `recommender_fm_feature_count` | DeepFM FM sparse field 数 |
| `recommender_fm_pair_count_log` | `n * (n - 1) / 2` |
| `recommender_cross_layer_count` | DCN/DCNv2/EDCN cross layers |
| `recommender_dcnv2_low_rank_log` | DCNv2 low rank |
| `recommender_dcnv2_num_experts_log` | DCNv2 expert 数 |
| `recommender_dcnv2_uses_low_rank_mixture` | DCNv2 mixture flag |
| `recommender_structure_crossnet_only` | DCNv2 structure one-hot |
| `recommender_structure_stacked` | DCNv2 structure one-hot |
| `recommender_structure_parallel` | DCNv2 structure one-hot |
| `recommender_edcn_uses_regulation` | EDCN regulation flag |
| `recommender_bridge_hadamard_product` | EDCN bridge one-hot |
| `recommender_bridge_pointwise_addition` | EDCN bridge one-hot |
| `recommender_bridge_concatenation` | EDCN bridge one-hot |
| `recommender_bridge_attention_pooling` | EDCN bridge one-hot |

`activation` 和 `dropout` P0 先不加。当前推荐配置 activation 基本固定为 `relu`，
dropout 多为 `0.0`，优先级低。

### MLP 参数估算

对 `mlp_dims = [d1, d2, ...]`：

```text
input_dim = sum(sparse embed_dim) + sum(dense embed_dim)
mlp_param_estimate =
  input_dim * d1 +
  d1 * d2 +
  ...
```

只做权重矩阵估算，不加 bias。这个特征不是参数精确值，而是 MLP 计算规模 proxy。

DeepFM 的 FM pair count：

```text
fm_pair_count = len(fm_feature_names) * (len(fm_feature_names) - 1) / 2
```

DCN/DCNv2/EDCN 的 cross/interactions 由对应层数、rank、experts 和 bridge 类型表达。

### 验证

新增测试：

- DeepFM minimal result JSON 能产生非零 embedding/vocab/FM features。
- DCN/DCNv2/EDCN 分别设置正确 one-hot。
- 非推荐模型推荐特征全零。
- 缺失 `result_json` 或找不到 `variant_name` 时失败。
- 只有一个 recommender config 生效，多 config 继续失败。

## 特征变换策略

P0 推荐在 extraction 阶段直接生成稳定尺度的特征：

- 类别/布尔：`0.0` 或 `1.0`。
- ratio：`0.0` 到 `1.0`。
- count/bytes/elements/dim/vocab/params：`log1p(float(value))`。
- 已有 raw profile features 暂时保留原口径，避免一次性改变所有历史特征含义。

这样做的好处：

- `RobustScaler` 不必承担所有重尾压缩。
- 新特征在 train-only scaler 修复前也不会过度放大尾部。
- 后续消融能区分“新增语义特征”和“重写历史特征变换”的效果。

## 实现顺序

推荐按以下顺序实施。

### 阶段 A: schema 和 helper

修改：

- `src/gnn_model/data/constants.py`
- `src/gnn_model/data/onnx_graph.py`

动作：

- 定义 `OP_TYPE_FEATURE_NAMES`。
- 定义 shape feature names。
- 定义 recommender graph feature names 的占位。
- 新增 `safe_log1p()`、`build_shape_summary()`。
- 新增 op category 映射和 one-hot helper。

验证：

```bash
PYTHONPATH=src uv run python -m pytest -q tests/test_gnn_model_data.py -k "shape or onnx"
```

### 阶段 B: ONNX op 和 shape 特征

修改：

- `src/gnn_model/data/onnx_graph.py`
- `tests/test_gnn_model_data.py`
- `tests/gnn_model_test_utils.py`

动作：

- 节点追加 op one-hot 和 node shape summaries。
- 边追加 tensor shape summaries。
- 图级追加 op histogram 和 global shape summaries。
- 调整 synthetic graph fixture 维度。

验证：

```bash
PYTHONPATH=src uv run python -m pytest -q tests/test_gnn_model_data.py tests/test_gnn_model_model.py
```

### 阶段 C: result JSON 上下文

修改：

- `src/gnn_model/data/extract.py`
- 新增 `src/gnn_model/data/variant_context.py`
- `tests/test_gnn_model_data.py`

动作：

- 在处理 CSV 时按 result JSON 建立 context cache。
- `ModelRecordInfo` 增加 `variant_context` 或在 `extract_feature_target()` 查询 context。
- 单测 fixture 写最小 result JSON。

验证：

```bash
PYTHONPATH=src uv run python -m pytest -q tests/test_gnn_model_data.py -k "process_csv or prepared"
```

### 阶段 D: 推荐专项特征

修改：

- `src/gnn_model/data/variant_context.py`
- `src/gnn_model/data/onnx_graph.py`
- `src/gnn_model/data/constants.py`
- `tests/test_gnn_model_data.py`

动作：

- 从 context 中提取推荐模型 config。
- 生成固定长度 recommender graph feature vector。
- 非推荐模型输出零向量。

验证：

```bash
PYTHONPATH=src uv run python -m pytest -q tests/test_gnn_model_data.py -k "recommender or process_csv"
```

### 阶段 E: manifest 和端到端

修改：

- `src/gnn_model/data/extract.py`
- `src/gnn_model/data/prepared_dataset.py`
- `docs/train` 后续报告脚本或手工流程

动作：

- manifest 写入 schema version、feature source、feature names。
- prepared loader 可选校验 manifest feature names 与 constants 一致。
- 重新生成 extracted/scaled/scalers。

验证：

```bash
PYTHONPATH=src uv run python -m pytest -q tests/test_gnn_model_data.py tests/test_gnn_model_pipeline.py tests/test_gnn_model_cli.py
```

完整检查：

```bash
uv run ruff check main.py src scripts tests
uv run ty check src main.py tests
uv run python -m pytest -q
```

## 评估协议

P0 特征的效果不能继续只看当前行级随机 split。至少需要两组对照：

| 实验 | 目的 |
| --- | --- |
| current schema + row split | 保持历史可比 |
| P0 schema + row split | 判断新特征是否改善当前分布拟合 |
| current schema + variant group split | 去掉同变体泄漏后的基线 |
| P0 schema + variant group split | 判断新特征是否改善未见变体 |

同时必须修 train-only scaler，否则 scaler 会从 val/test 分布获得信息。这个修复不属于 P0
特征本身，但属于 P0 特征评估的前置条件。

报告指标至少包含：

- overall WAPE/R2/RMSE。
- per target MAE/WAPE/RMSE。
- per family macro WAPE。
- per phase 指标。
- 推荐模型单独表。
- GPT-2/T5 单独表。

## 风险和处理

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| op category 太粗 | raw op 差异仍被合并 | P0 先 category，P1 再加 top-K raw op one-hot |
| ONNX 静态 batch 与 CSV batch 不一致 | shape/MACs 口径混杂 | P0 明确拆分 profile dim0 和 runtime batch，不改 profile |
| result JSON 缺失 | 无法拿推荐 config | 生产路径 fail fast，单测补最小 result JSON |
| 推荐特征过多 | 图级特征维度膨胀 | 只加结构强相关字段，不加 activation/dropout 等低变化字段 |
| 旧数据误用 | 训练维度错或隐性 schema 混用 | bump schema，loader 校验 feature names |
| 大数值尾部支配 | scaler 和 loss 不稳定 | 新增 count/bytes 全部 `log1p` |

## 不采用的方案

### 直接给模型加 op embedding

这是更干净的表示方式，但需要改 `Data` 字段、batch collate、`Predictor.forward()` 和模型结构。
P0 目标是先验证特征收益，one-hot 兼容当前三张量接口，风险更小。

### 从 YAML config 反推推荐特征

YAML 中可能是 grid/template/anchor，读取后还要重新展开。result JSON 已保存 resolved
variant_config，是更直接的事实源。

### 重新 profile 成真实 batch shape

这可能更准确，但会同时改变历史 `profile_total_macs`、`profile_memory_bytes` 等核心特征。
P0 先不动 profiler 口径，避免把收益归因混在一起。

## 预期收益

P0 完成后，模型应该更容易区分：

- CNN/YOLO 中 Conv-heavy 与 layout-heavy 变体。
- GPT-2/T5 中 MatMul/Softmax/Reshape-heavy 的 attention 图。
- 推荐模型中 embedding-heavy、FM interaction-heavy、cross-network-heavy 的差异。

最先应该观察的改善目标：

- `gpu_mem_used_mb_p95`，尤其 `deepfm/dcn/edcn/dcnv2`。
- `duration_sec_avg`，尤其推荐模型和 GPT/T5。
- `gpu_sm_occupancy_percent_p95`，尤其小分母但结构差异明显的 text inference。

若 row split 改善但 variant-group split 不改善，说明特征主要增强同分布拟合；若两者都改善，
才可以把 P0 视为泛化能力增强。
