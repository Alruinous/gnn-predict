# Workflow DAG 配置设计方案

## 目标

本模块定义一种 Workflow DAG 配置格式。

一个 Workflow 是一张 DAG：

- `nodes` 是节点名到节点属性的 map。
- `edges` 是有向边列表。

节点属性必须包含模型类型，并可包含预测器在图生成前就能知道的模型参数和推理超参数。边属性首版保留为空 map。

## 设计原则

参考 LangGraph 的最小图定义方式：节点由名字标识，节点之间通过边连接。本项目在这个基础上增加节点的模型描述，因为预测器需要知道节点内部模型的先验配置。

节点属性只保存“Workflow 计划阶段已经知道”的信息：

- 模型类型。
- 模型名。
- 模型构造参数。
- 推理阶段超参数。

节点属性不保存“必须构建模型或运行后才能得到”的信息：

- ONNX 图特征。
- 参数量、FLOPs、op 统计、图节点数。
- 真实运行耗时、显存、CPU、GPU 利用率。
- 训练开关、导出开关、监控窗口、冷却时间。

这些信息属于后续特征抽取或实验记录，不属于 Workflow 数据集配置。

## 支持范围依据

`csv_v2/v100` 用于确认当前主线数据集中实际保留的模型类型。

`config/arch` 和具体 builder 代码只用于理解已保留模型类型中哪些字段真正影响模型构造或推理形状。不能照抄其中所有字段，也不能把历史残留但已被当前主线抛弃的变体写入 Workflow schema。

应保留的字段含义：

- `batch_size`：推理批大小。
- `example_input_shape`：推理输入形状。
- `target_input_channels`：图像或检测模型输入通道数。
- `target_output_classes`：分类头或检测头输出类别数。
- Transformer 的层数、隐藏维度、注意力头数、FFN 中间维度、词表大小、最大位置长度。
- 生成式模型 decode 阶段的最大输出长度。

应删除的字段含义：

- 是否训练、是否推理、是否导出 ONNX。
- 监控测量时间。
- cooldown 时间。
- fake dataset 或 real dataset 开关。
- ONNX export mode。
- 输出目录、日志、结果文件路径。

## Workflow Schema

每个 Workflow 使用一个 YAML 文件表示。

```yaml
nodes:
  input:
    type: input

  planner:
    type: main_agent
    model:
      name: qwen3
      hidden_size: 1536
      intermediate_size: 5120
      num_hidden_layers: 28
      num_attention_heads: 24
      num_key_value_heads: 4
      vocab_size: 151936
      max_position_embeddings: 40960
    runtime:
      batch_size: 1
      sequence_length: 512
      phase: prefill

  detector:
    type: object_detection
    model:
      name: yolov5n
      input_channels: 3
      output_classes: 80
      image_size: [640, 640]
      depth_scale: 0.33
      width_scale: 0.25
    runtime:
      batch_size: 4
      input_shape: [4, 3, 640, 640]
      phase: inference

  summarizer:
    type: main_agent
    model:
      name: qwen3
      hidden_size: 1536
      intermediate_size: 5120
      num_hidden_layers: 28
      num_attention_heads: 24
      num_key_value_heads: 4
      vocab_size: 151936
      max_position_embeddings: 40960
    runtime:
      batch_size: 1
      sequence_length: 512
      decode_max_output_length: 128
      phase: decode

  output:
    type: output

edges:
  - source: input
    target: planner
    attributes: {}
  - source: planner
    target: detector
    attributes: {}
  - source: detector
    target: summarizer
    attributes: {}
  - source: summarizer
    target: output
    attributes: {}
```

## 字段定义

### nodes

`nodes` 是 map。

```yaml
nodes:
  <node_name>:
    type: <node_type>
    model: {}
    runtime: {}
```

约束：

- `node_name` 是当前 Workflow 内唯一的节点名。
- `type` 是节点类型。
- `model` 是模型构造前已知的模型参数。
- `runtime` 是推理前已知的运行参数。
- `input` 和 `output` 节点不需要 `model` 和 `runtime`。
- 其他节点必须包含 `model` 和 `runtime`。

### edges

`edges` 是 list。

```yaml
edges:
  - source: <source_node_name>
    target: <target_node_name>
    attributes: {}
```

约束：

- `source` 必须引用 `nodes` 中已定义的节点。
- `target` 必须引用 `nodes` 中已定义的节点。
- `attributes` 首版必须是空 map。
- 边方向表示 `target` 依赖 `source`。

## 节点类型

首版节点类型如下：

```text
input
output
main_agent
object_detection
image_classification
vision_transformer
text_encoder
text_generation
```

## 通用 runtime 字段

所有模型节点都应包含：

| 字段 | 含义 |
| --- | --- |
| `batch_size` | 推理批大小 |
| `phase` | `inference`、`prefill`、`decode` 等执行阶段 |

按模态补充：

| 字段 | 适用节点 | 含义 |
| --- | --- | --- |
| `input_shape` | 图像、检测 | 完整输入 shape |
| `sequence_length` | 文本、LLM | 输入 token 长度 |
| `decode_max_output_length` | 生成式模型 | 最大生成 token 数 |

`batch_size` 与 `input_shape` 首维应一致。文本节点可以用 `batch_size + sequence_length` 表达输入形状。

## 各模型类型的 model 字段

### main_agent

用于 Qwen/Gemma/LLaMA 一类主智能体。

```yaml
model:
  name: qwen3
  hidden_size: 1536
  intermediate_size: 5120
  num_hidden_layers: 28
  num_attention_heads: 24
  num_key_value_heads: 4
  head_dim: 128
  vocab_size: 151936
  max_position_embeddings: 40960
```

必要字段：

- `name`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `vocab_size`
- `max_position_embeddings`

可选字段：

- `num_key_value_heads`
- `head_dim`
- `tie_word_embeddings`
- `rope_theta`

### object_detection

用于 YOLO 检测模型。

```yaml
model:
  name: yolov5n
  input_channels: 3
  output_classes: 80
  image_size: [640, 640]
  depth_scale: 0.33
  width_scale: 0.25
  max_channels: 1024
  activation: silu
```

必要字段：

- `name`
- `input_channels`
- `output_classes`
- `image_size`

可选字段：

- `depth_scale`
- `width_scale`
- `max_channels`
- `activation`
- `backbone_module`
- `head_module`

这些字段对应检测模型 YAML 中真正影响结构和输入规模的部分。不要保存训练或导出配置。

### image_classification

用于 ResNet、VGG、MobileNet、EfficientNet、ConvNeXt、DenseNet 等图像分类模型。

```yaml
model:
  name: resnet18
  input_channels: 3
  output_classes: 1000
  image_size: [224, 224]
```

必要字段：

- `name`
- `input_channels`
- `output_classes`
- `image_size`

可选字段：

- `depth`
- `width_multiplier`
- `dropout`
- `activation`

如果已知该模型的结构参数，可以写入可选字段；如果未知，不要伪造。

### vision_transformer

用于 ViT、Swin、BEiT 等视觉 Transformer。

```yaml
model:
  name: vit_base_patch16_224
  input_channels: 3
  output_classes: 1000
  image_size: [224, 224]
  patch_size: 16
  embed_dim: 768
  num_layers: 12
  num_attention_heads: 12
  mlp_dim: 3072
```

必要字段：

- `name`
- `input_channels`
- `output_classes`
- `image_size`

可选字段：

- `patch_size`
- `embed_dim`
- `num_layers`
- `num_attention_heads`
- `mlp_dim`
- `dropout`

### text_encoder

用于 BERT/BERT-large 类文本编码模型。

```yaml
model:
  name: bert-base-uncased
  output_classes: 2
  vocab_size: 30522
  hidden_size: 768
  intermediate_size: 3072
  num_hidden_layers: 12
  num_attention_heads: 12
  max_position_embeddings: 512
```

必要字段：

- `name`
- `output_classes`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `max_position_embeddings`

可选字段：

- `vocab_size`
- `hidden_dropout_prob`
- `attention_probs_dropout_prob`

### text_generation

用于 GPT2/T5 类生成或序列模型。

GPT-like：

```yaml
model:
  name: gpt2
  output_classes: 2
  vocab_size: 50257
  hidden_size: 768
  intermediate_size: 3072
  num_hidden_layers: 12
  num_attention_heads: 12
  max_position_embeddings: 1024
```

T5-like：

```yaml
model:
  name: t5
  output_classes: 2
  vocab_size: 32128
  hidden_size: 768
  intermediate_size: 3072
  num_encoder_layers: 12
  num_decoder_layers: 12
  num_attention_heads: 12
  relative_attention_num_buckets: 32
```

必要字段按 `name` 对应的模型类别区分，但都必须包含足够表达层数、宽度、FFN 维度、头数、词表规模和位置/相对位置能力的字段。

## 合法性约束

每个 Workflow 必须满足：

- `nodes` 非空。
- 每个节点必须有 `type`。
- 每个节点 `type` 必须来自节点类型枚举。
- 除 `input` 和 `output` 外，每个节点必须有 `model` 和 `runtime`.
- 每条边必须有 `source`、`target`、`attributes`。
- 每条边的 `source` 和 `target` 必须引用已定义节点。
- `attributes` 首版必须是空 map。
- 图必须无环。

字段级约束：

- `batch_size` 必须为正整数。
- `input_shape` 中的维度必须为正整数。
- `sequence_length` 必须为正整数。
- `decode_max_output_length` 在 `phase: decode` 时必须为正整数。
- `hidden_size` 必须能被 `num_attention_heads` 整除。
- `num_attention_heads` 必须能被 `num_key_value_heads` 整除，如果后者存在。
- `sequence_length + decode_max_output_length` 不能超过 `max_position_embeddings`。
- `output_classes` 必须为正整数。

## 数据集组织

数据集就是一组 Workflow YAML 文件。

建议目录：

```text
data/workflows/
  workflow_000001.yaml
  workflow_000002.yaml
  workflow_000003.yaml
```

每个 YAML 文件自身就是完整样本。

## 模块设计

建议新建：

```text
src/workflow/
```

文件结构：

```text
src/workflow/
  __init__.py
  schema.py
  loader.py
  validation.py
```

职责：

| 文件 | 职责 |
| --- | --- |
| `schema.py` | 定义节点类型、节点配置、边配置和 Workflow 配置 |
| `loader.py` | 读取单个 YAML 或目录下所有 YAML |
| `validation.py` | 校验引用关系、节点类型、字段约束和 DAG 无环 |

## 测试计划

建议新增：

```text
tests/test_workflow_schema.py
tests/test_workflow_loader.py
tests/test_workflow_validation.py
```

测试重点：

- 能读取合法 Workflow YAML。
- `input` 和 `output` 节点可以没有 `model` 和 `runtime`。
- 模型节点必须包含 `model` 和 `runtime`。
- 边属性 `attributes` 必须为空 map。
- 未定义节点引用会失败。
- 非法节点类型会失败。
- 有环图会失败。
- LLM 节点的 head 数、KV head 数和位置长度约束能被校验。

## 最小交付

首版只需要完成：

- Workflow YAML schema。
- YAML loader。
- DAG validator。
- 节点模型参数和 runtime 参数校验。
- 基础测试。

成功标准：

- Workflow 配置仍然是 `nodes + edges`。
- 节点包含预测所需的已知模型先验和推理超参数。
- 边只描述连接。
- 不包含 ONNX 派生特征、资源测量字段、训练字段或运行时结果。
