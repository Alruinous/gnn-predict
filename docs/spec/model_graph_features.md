# 模型计算图与图特征规范

## 适用范围

本规范定义 GNN 预测器使用的模型计算图产物和 `torch_geometric.data.Data` 特征格式。计算图描述一次确定输入形状的推理执行，包括普通 inference、LLM prefill 和单步 cached decode。

当前版本：

- 计算图产物 schema：`1.0.0`
- 数据集 manifest schema：`5.0.0`
- `feature_source`：`pytorch_export_inference_ir_static_metrics_shape_topology_v1`

## 计算图产物

### 捕获和文件格式

模型必须处于 `eval()` 和 `torch.no_grad()` 状态，通过 `torch.export.export(strict=False)` 捕获，并使用空 decomposition table 生成功能化 inference IR。每个模型、phase 和静态输入形状组合独立生成一个图，不使用动态 shape 复用。

捕获结果必须先经过一次内存中的 `torch.export.save`/`load` 规范化，使直接构建和从文件读取时看到相同的 FX 图。产物使用 `torch.export.save` 保存为 `.pt2`。归档内必须包含名为 `gnn_predict_graph.json` 的附加文件，其字段如下：

| 字段 | 值或含义 |
| --- | --- |
| `format` | 固定为 `pt2` |
| `schema_version` | 固定为 `1.0.0` |
| `torch_version` | 生成产物的完整 PyTorch 版本 |
| `capture_mode` | 固定为 `static_inference` |
| `weights` | 固定为 `zero_stride_proxy` |
| `runtime_input_names` | 按 USER_INPUT graph signature 顺序排列的逻辑输入名 |

读取时必须校验产物 schema、运行环境 PyTorch major/minor 版本和 runtime input 数量。任一不匹配都应失败，不做兼容转换。

### 参数和常量

`.pt2` 是架构分析产物，不是可复现原模型输出的权重文件。参数、buffer 和 tensor constant 必须保持原 shape、dtype 与 `requires_grad` 属性，但其存储替换为 CPU 上的单元素零值和全零 stride。禁止在归档中保存真实模型权重。

参数和 buffer 仍作为 lifted graph input 出现在 ExportedProgram signature 中。它们参与输入 tensor shape 特征和参数输入统计，但不作为 PyG 节点。

### Phase 输入

- 普通模型：一个 `inputs`，或文本分类模型的 `input_ids`、`attention_mask`。
- T5 分类图：输入必须保证每个样本只有最后一个 token 为 EOS，分类表示固定取最后一个 token。
- prefill：`input_ids`、`attention_mask`，输出 logits 及每层 present key/value。
- decode：`input_ids`、`attention_mask` 及按层排列的 `past_{i}_key`、`past_{i}_value`。past 长度为 `sequence_length + decode_output_length - 1`，decode token 长度为 1。

## FX 图转换

### 节点

只有具有至少一个 tensor 输出的 FX `call_function` 节点进入 PyG 图。placeholder、output 以及只产生 Python scalar、`SymInt` 或控制元数据的节点不进入图。所有 tensor shape 必须能解析为非负整数；未约束的符号维度直接报错。

节点顺序沿用 FX graph 的拓扑顺序。tuple/list `getitem` 记为 identity；tensor indexing `getitem` 记为 embedding；lifted tensor literal 的物化节点记为 constant。一个多输出节点必须通过 `getitem` 消费，否则图转换失败。

算子必须显式归入以下 15 类之一：

`op_conv`、`op_dense`、`op_embedding`、`op_attention`、`op_norm`、`op_pool`、`op_activation`、`op_elementwise`、`op_reduce`、`op_shape`、`op_layout`、`op_join_split`、`op_cast`、`op_constant`、`op_identity`。

未知 tensor 算子不得降级到默认类别，必须报告具体 FX target。

### 边

目标节点的每次 tensor 节点依赖生成一条有向边，重复使用同一 tensor 会生成重复边。USER_INPUT、PARAMETER、BUFFER 等 placeholder 到算子节点的依赖不生成边。边顺序按目标节点顺序及其参数遍历顺序确定。

### 计算量和内存

- convolution MAC：`output_elements × product(weight_shape[1:])`，存在 bias 时再加 `output_elements`。
- linear MAC：`output_elements × input_features`，存在 bias 时再加 `output_elements`。
- matrix multiplication MAC：`output_elements × reduction_dimension`。
- scaled dot-product attention MAC：两次矩阵乘 MAC 加 attention score 上的 softmax 估算。
- normalization MAC：`input_elements × 5`。
- pooling MAC：输入元素数加输出字节数。
- elementwise、activation 和 reduce 使用固定单元素指令权重；layout、shape、join/split、cast、embedding、constant 和 identity 默认为零 MAC。

`node_memory_bytes` 是该节点全部 tensor 输出的字节数，`graph_memory_bytes` 是所有节点输出字节数之和。峰值 activation memory 按拓扑执行顺序和最后消费者释放位置计算，不追踪 view/alias 共享存储。

架构图不含真实权重，因此 `node_params` 和 `graph_params` 固定为零。参数规模由 graph signature 中 PARAMETER 与 BUFFER 输入的元素数和字节数表达。

## PyG Data 格式

### 节点特征 `x`

`x` 类型为 `torch.float32`，shape 为 `[N, 22]`。

| 索引 | 字段 | 含义 |
| ---: | --- | --- |
| 0 | `node_macs` | 节点 MAC |
| 1 | `node_memory_bytes` | 节点输出字节数 |
| 2 | `node_params` | 固定为 0 |
| 3 | `input_count` | 显式 FX Node 输入引用数 |
| 4 | `output_count` | tensor 输出数 |
| 5 | `attr_count` | 不含 FX Node 的显式 positional/keyword 参数数 |
| 6 | `in_degree` | PyG 入度 |
| 7 | `out_degree` | PyG 出度 |
| 8 | `input_tensor_bytes_sum_log` | `log1p` 输入 tensor 总字节数 |
| 9 | `output_tensor_bytes_sum_log` | `log1p` 输出 tensor 总字节数 |
| 10 | `input_tensor_elements_sum_log` | `log1p` 输入元素总数 |
| 11 | `output_tensor_elements_sum_log` | `log1p` 输出元素总数 |
| 12 | `input_tensor_rank_max` | 最大输入 rank |
| 13 | `output_tensor_rank_max` | 最大输出 rank |
| 14 | `output_tensor_nonbatch_elements_log` | `log1p` 输出非 batch 元素总数 |
| 15–20 | `output_tensor_dim0_log` … `output_tensor_dim5_log` | 首个输出前六维的 `log1p`，缺失维补 0 |
| 21 | `output_tensor_dtype_itemsize` | 首个输出的 dtype 字节数 |

`op_type_ids` 类型为 `torch.long`，shape 为 `[N]`，值为上述 15 类算子在固定类别表中的索引。

### 边特征

`edge_index` 类型为 `torch.long`，shape 为 `[2, E]`。`edge_attr` 类型为 `torch.float32`，shape 为 `[E, 15]`。

| 索引 | 字段 | 含义 |
| ---: | --- | --- |
| 0 | `tensor_bytes` | 依赖 tensor 字节数 |
| 1 | `tensor_rank` | tensor rank |
| 2 | `tensor_element_count` | tensor 元素数 |
| 3 | `source_out_degree` | 源节点出度 |
| 4 | `target_in_degree` | 目标节点入度 |
| 5–10 | `tensor_dim0_log` … `tensor_dim5_log` | 前六维的 `log1p`，缺失维补 0 |
| 11 | `tensor_nonbatch_element_count_log` | `log1p` 非 batch 元素数 |
| 12 | `tensor_dtype_itemsize` | dtype 字节数 |
| 13 | `tensor_is_scalar` | rank 0 时为 1 |
| 14 | `tensor_is_zero_sized` | 任一维为 0 时为 1 |

### 全图特征 `graph_features`

`graph_features` 类型为 `torch.float32`，shape 为 `[1, 30]`。

| 索引 | 字段 | 含义 |
| ---: | --- | --- |
| 0 | `phase_token_id` | training=0、inference=1、prefill=2、decode=3 |
| 1 | `batch_size` | workload batch size |
| 2 | `decode_output_length` | 非 decode 图为 0 |
| 3–13 | GPU 规格 | FP64、FP32、Tensor TFLOPS、显存、带宽、L2、SM、CUDA core、TDP、NVLink、PCIe lanes |
| 14 | `parameter_input_count` | PARAMETER 与 BUFFER 输入数量 |
| 15 | `parameter_input_element_count` | 参数输入元素总数 |
| 16 | `parameter_input_bytes` | 参数输入总字节数 |
| 17 | `graph_macs` | 节点 MAC 总和 |
| 18 | `graph_memory_bytes` | 节点输出字节数总和 |
| 19 | `graph_params` | 固定为 0 |
| 20 | `node_count_log` | `log1p(N)` |
| 21 | `edge_count_log` | `log1p(E)` |
| 22 | `activation_bytes_sum_log` | `log1p` activation 总字节数 |
| 23 | `peak_live_activation_bytes_log` | `log1p` 峰值 live activation 字节数 |
| 24 | `activation_elements_sum_log` | `log1p` activation 元素总数 |
| 25 | `max_tensor_rank` | 全图最大 tensor rank |
| 26 | `max_tensor_dim_log` | `log1p` 全图最大单维长度 |
| 27 | `runtime_input_count` | USER_INPUT tensor 数量 |
| 28 | `runtime_input_elements_sum_log` | `log1p` runtime input 元素总数 |
| 29 | `runtime_input_nonbatch_elements_sum_log` | `log1p` runtime input 非 batch 元素总数 |

### 附加属性和不变量

`graph_path` 保存 `.pt2` 路径；内存直接构建时为空字符串。数据集抽取阶段可附加 `y`、CSV 来源和 workload 元数据。

所有浮点特征必须有限；`edge_index` 必须位于 `[0, N)`；`x`、`edge_attr`、`graph_features` 和 `op_type_ids` 的首维必须分别与节点、边、图数量一致。违反这些约束的图不得进入训练数据。

传入的 workload `batch_size` 必须等于每个静态 USER_INPUT shape 的首维；不一致时不得用全图字段覆盖产物中的静态形状。
