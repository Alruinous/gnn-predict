# Causal LM 接入口径

本文记录现代 decoder-only LLM 在项目中的当前口径。旧的 checkpoint 接入方案不再作为主文档保留。

## 当前主线

当前稳定口径是用配置构造随机初始化的 causal LM 变体，而不是依赖本地 checkpoint 作为结构来源。

以 Qwen3 为例：

- `base_model.name` 使用 `qwen3`。
- `pretrained` 必须为 `false`。
- 结构参数来自 `variant_config.qwen3_config`。
- 输入形状使用 `[batch_size, sequence_length]`。
- `batch_size` 必须与 `example_input_shape[0]` 一致。
- prompt length 和 decode output length 不能超过上下文上限。

这种口径更适合本项目的结构变体生成，也避免把本机 checkpoint 路径写进长期配置。

## 阶段

Causal LM 目前主要覆盖三类阶段：

- `training`：fake token next-token loss，执行 forward/backward/optimizer step。
- `prefill`：完整 prompt forward-only。
- `decode`：基于 prompt 调用 `generate`，最多生成配置里的新增 token 数。

`decode_output_length` 是 decode 阶段的重要运行特征。非 decode 阶段该值为 `0`。

## Workflow 使用口径

当前 `decode` 样本不是纯 KV cache decode step，而是从 prompt 调用
`generate(max_new_tokens=decode_output_length)` 的端到端生成请求。

该运行目标包含 prompt prefill、逐 token decode 和 generate 框架开销。workflow
中应把它理解为一次 LLM 节点请求的整体生成成本，不应再与 `prefill` 样本相加作为总耗时。

如果后续需要分阶段预测，应新增纯 KV cache decode 数据口径，而不是复用当前
`decode` 样本。

## 输入与标签

首版使用 fake token，不引入真实 tokenizer 样本。

运行约束：

- `input_ids` 形状为 `[batch_size, sequence_length]`。
- `attention_mask` 形状为 `[batch_size, sequence_length]`。
- `labels` 使用 `input_ids.clone()`。
- fake token 不表达 padding、prompt 语义或真实任务难度。

如果后续引入真实文本样本，应单独写数据口径，不混入当前结构变体文档。

## ONNX

ONNX 导出保留完整 logits，不裁剪最后一个 token。

导出边界：

- 输入名固定为 `input_ids` 和 `attention_mask`。
- 输出名固定为 `logits`。
- 首选 `architecture_only`，避免把大模型权重写入 ONNX。
- 导出兼容补丁只能影响导出阶段，不应改变训练和运行阶段口径。

## LLaMA / Gemma

LLaMA 和 Gemma 可以复用 causal LM 的阶段与 fake token 口径，但不要把它们当成 Qwen 的简单别名。

需要额外注意：

- 配置字段、attention 实现、词表规模和 RoPE 细节不同。
- Gemma 的大词表会放大 logits 和导出图压力。
- 多模态 Gemma 不属于当前纯文本 causal LM 路径。

是否纳入新 family，应以当前 builder、配置和 ONNX 导出验证为准，不以旧方案文档为准。
