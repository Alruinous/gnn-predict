# Qwen模型变体引入方案

## 背景

阅读 docs 目录中文档，理解项目背景。当前项目数据为多种模型类型的变体数据，数据记录了变体在训练和推理过程稿中各项硬件指标。项目使用一个基于 GNN 实现的预测器，以模型 ONNX 计算图、实验过程硬件信息、实验超参数等作为输入，预测变体模型在实验时的各项硬件性能指标。

## 我的简易探索

我在 tmp/qwen.ipynb 中尝试使用 transformers 库预训练 qwen3.5 模型，其中一些使用样例，可能可以供你参考。

## 目标

我打算为当前项目使用的数据集引入现代大模型变体（如 qwen3.5），以丰富数据集的多样性和代表性。这将有助于提升预测器的泛化能力和准确性。

## 问题

### 现代大模型训推区别

首批 Qwen 数据按 causal LM 任务处理，不按 GPT-2 当前的 sequence classification 口径处理。训练阶段对应 SFT 或 next-token loss：模型读入一批 token，输出每个位置的 logits，并基于下一个 token 计算 loss 和反向传播。

后训练涉及强化学习、偏好对齐等场景，暂时不考虑。

推理阶段只引入 Prefill。Prefill 阶段读入完整 prompt，并执行一次 full forward-only。Decode 阶段会逐 token 生成并维护 KV cache，资源模式与当前项目的普通推理阶段差异较大，首批暂不引入。

首批 full-flow Qwen 参数组合应生成 training 和 prefill 两个阶段的数据样本。SFT 阶段复用现有 training 语义；Prefill 新增独立 phase id，不复用现有 inference phase。显存压力较大的型号可只生成 prefill 样本。

### 变体可调范围

由于现代大模型的结构相对固定，一个系列的大模型大多只有参数规模和 dense 或 moe 的区别。比如 Qwen3.5 模型，其在单卡可训推的基础上有 0.8B、2B 等参数规模的变体。

在当前项目中，Qwen 首批变体不暴露 Transformers 模型结构参数。更合适的做法是固定一个可加载的 Qwen checkpoint，只调整项目数据生成层能够稳定表达的输入张量形状。

首批建议只保留以下变体轴：

- `example_input_shape`：固定为二维输入形状，用于生成固定形状的 `input_ids` 和 `attention_mask`。
- `max_sequence_length`：与 `example_input_shape[1]` 保持一致，用于校验输入 token 长度不超过模型上限。
- `training_batch_sizes`：与 `example_input_shape[0]` 保持一致，避免训练阶段和导出/Prefill 阶段的 batch 口径不一致。

`base_model.name` 只记录具体使用的 Qwen checkpoint 名称，例如 `Qwen3.5-0.8B`，不作为同一模型内部的变体轴。

首批使用 fake token，不使用真实 tokenizer 样本。因此输入 token 长度由 `example_input_shape[1]` 直接决定，不依赖 `SFTConfig.max_length` 或 tokenizer padding。padding 前文本真实长度不作为首批变体参数。

以下运行条件第一批建议固定，不作为变体轴：

- `max_new_tokens`：只对完整生成或 Decode 有意义，不适合混入 Prefill-only 样本。
- prompt 文本内容：首批使用 fake token，真实 prompt 内容不进入数据生成口径。
- `dtype`：会影响显存和算力路径，但当前主线特征中没有稳定表达，不适合作为首批变体轴。
- `gradient_accumulation_steps`：影响 optimizer update 频率，但不应改变单次 forward/backward 的资源口径。
- `gradient_checkpointing`：训练阶段显存和时间的权衡项，当前不宜与基础训练口径混用。
- `use_cache`：Prefill 固定为 false，避免引入 KV cache；完整 Decode 暂不进入首批数据。
- hidden size、层数、head 数、中间层宽度：这些是模型结构参数，不适合直接修改已有 Qwen checkpoint。
- LoRA 或 full fine-tune 开关：这属于训练方式变化，不是纯模型结构变体；如果纳入，需要单独作为训练协议特征。

首批运行固定使用 fp16。ONNX 导出默认使用 `architecture_only`，避免把大模型完整权重写入 ONNX 文件。

### 首批模型型号

Qwen2 架构族包括 Qwen2 和 Qwen2.5。Qwen2 公开模型包含 0.5B、1.5B、7B、57B-A14B、72B；Qwen2.5 公开模型包含 0.5B、1.5B、3B、7B、14B、32B、72B。首批只考虑 dense causal LM，不纳入 Coder、Math、VL、Omni、量化或 MoE 派生模型。

Qwen3 公开模型包含 dense 的 0.6B、1.7B、4B、8B、14B、32B，以及 MoE 的 30B-A3B、235B-A22B。首批只考虑 dense causal LM，不纳入 Embedding、Reranker、Guard、Coder、Next、量化或 MoE 派生模型。

Qwen3.5 公开模型包含 dense 的 0.8B、2B、4B、9B、27B，以及 MoE 的 35B-A3B、122B-A10B、397B-A17B。首批按 text-only causal LM 输入使用，不使用图像、视频或工具调用相关输入。

首批数据生成要同时覆盖 training 和 prefill。由于当前训练口径是全参数反向传播，下面型号作为默认可顺利完成实验的范围：

- V100-32GB 和 A100 通用：`Qwen2-0.5B`、`Qwen2-1.5B`、`Qwen2.5-0.5B`、`Qwen2.5-1.5B`、`Qwen3-0.6B`、`Qwen3-1.7B`、`Qwen3.5-0.8B`。
- A100-80GB 可追加：`Qwen2.5-3B`、`Qwen3-4B`、`Qwen3.5-2B`、`Qwen3.5-4B`。

上述清单默认 `training_batch_sizes=[1]`，`sequence_length` 从 128、256、512 这类较短输入开始取值。增大 batch size 或 sequence length 时需要重新验证显存，不改变首批型号清单。

默认不把 7B 及以上 dense 模型、任何 MoE 模型、任何量化模型列入首批清单。这些模型可以单独做 prefill 或 LoRA 类实验，但不适合作为当前 training + prefill 全流程数据的默认型号。

## 细节

### 变体实验记录数据生成

参考现有变体配置，一个变体应该生成多个阶段的数据样本。对于现代大模型变体，一种参数组合应该生成 training 和 prefill 两个阶段的数据样本。

### 变体模型输入的数据

参考现有 NLP 类变体模型的输入数据，首批 Qwen 数据使用 fake token，通过构造固定形状的张量模拟模型输入。对于现代大模型，构造时需要考虑 batch size 和 sequence length 两个维度，确保生成的输入张量形状与变体配置中 `example_input_shape` 定义的形状一致。

首批输入数据口径：

- `input_ids`：随机 token id，形状为 `[batch_size, sequence_length]`。
- `attention_mask`：全 1 张量，形状为 `[batch_size, sequence_length]`。
- `labels`：训练阶段使用的 causal LM 标签，形状为 `[batch_size, sequence_length]`。

由于首批使用 fake token 且不做 padding，所有 token 都是有效输入，因此 `attention_mask` 可以固定为全 1。后续如果引入真实 tokenizer 样本，再按 padding 位置生成对应的 `attention_mask`。

### batch size 和 sequence length 扩充

首批 Qwen 变体通过 `batch_size` 和 `sequence_length` 组合扩充。对于同一个 Qwen checkpoint，一组组合对应一个固定输入形状：

```text
[batch_size, sequence_length]
```

`sequence_length` 扩充时，构造目标长度的 fake token 张量，并同步设置：

```yaml
example_input_shape: [1, 512]
max_sequence_length: 512
training_batch_sizes: [1]
```

`batch_size` 扩充时，只改变二维输入形状的第一个维度，并同步设置训练 batch：

```yaml
example_input_shape: [4, 512]
max_sequence_length: 512
training_batch_sizes: [4]
```

组合数量按下面方式计算：

```text
len(sequence_lengths) * len(batch_sizes)
```

如果 `sequence_lengths=[128, 256, 512]`，`batch_sizes=[1, 2, 4]`，则单个 Qwen checkpoint 有 9 种输入形状组合。每种组合生成 training 和 prefill 两个阶段的数据样本。

fake batch 构造示例：

```python
batch_size = 4
sequence_length = 512
vocab_size = model.config.vocab_size

input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
labels = input_ids.clone()
```

训练阶段使用 `input_ids`、`attention_mask`、`labels`；Prefill 阶段只使用 `input_ids` 和 `attention_mask`。`sequence_length` 必须不超过模型上下文上限，增大 `batch_size` 或 `sequence_length` 后需要重新验证显存。

首批 full-flow 变体同时开启 training 和 prefill。显存压力较大的 Qwen 型号可以只开启 prefill，不单独引入新的运行框架。

### ONNX 输出口径

Qwen 首批 ONNX 导出保留 causal LM 完整 logits，不裁剪最后一个 token。模型 forward 输出应保持原始 logits 形状：

```text
[batch_size, sequence_length, vocab_size]
```

训练和 Prefill 运行阶段使用原始模型。ONNX legacy exporter 直接导出 Qwen 模型时会得到 last-token logits，因此导出阶段只使用一个 logits adapter 返回 `model(...).logits`，不引入 `logits[:, -1, :]` 这类 next-token-only 包装。

Qwen3.5 的 linear attention 默认可能使用 FLA/Triton 自定义 kernel。训练和 Prefill 阶段不改动模型运行路径；ONNX 导出阶段临时切换到 transformers 中的 torch fallback，避免 Triton kernel 进入 ONNX trace。这个处理只用于导出兼容，不作为变体轴，也不改变训练或 Prefill 的实验口径。

Qwen3.5 在当前 PyTorch legacy ONNX exporter 下需要关闭 constant folding，否则可能在导出阶段触发 `ComplexDouble` 类型错误。这个问题发生在 PyTorch 导出流程中，不是 `onnx` Python 包版本过旧导致。
