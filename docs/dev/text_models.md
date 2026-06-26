# 文本模型接入口径

本文只记录当前 `gnn_archs` 中 GPT-2 和 T5 的接入边界。更细的历史探索不再单独保留。

## GPT-2

GPT-2 走独立 builder，不复用 BERT 的通用 text 分支。

当前边界：

- 构建入口是 `src/gnn_archs/gpt2_builder.py`。
- 配置入口是 `variant_config.gpt2_config`。
- 任务口径是 sequence classification。
- 不支持 mutation。
- `target_output_classes` 必须显式设置。
- `example_input_shape` 使用 `[batch_size, sequence_length]`。
- `n_positions` 必须覆盖 `sequence_length`。
- `n_embd` 必须能被 `n_head` 整除。
- ONNX 导出使用 eager attention，并把二维 attention mask 转成 GPT-2 可导出的 4D additive mask。

GPT-2 的主要价值是补齐 decoder-only text classification 图结构，不是模拟现代 causal LM 生成任务。

## T5

T5 也走独立 builder，不复用 BERT 或 GPT-2 的构建逻辑。

当前边界：

- 构建入口是 `src/gnn_archs/t5_builder.py`。
- 配置入口是 `variant_config.t5_config`。
- 任务口径是 sequence classification。
- 不支持 mutation。
- `target_output_classes` 必须显式设置。
- `example_input_shape` 使用 `[batch_size, sequence_length]`。
- fake text batch 必须保证每条输入有一致数量的 EOS。
- `feed_forward_proj` 只保留 `relu` 和 `gated-gelu`。
- `d_kv` 可省略；省略时由 `d_model // num_heads` 推出。

T5 的主要价值是补齐 encoder-decoder、cross-attention、relative attention bias 和 shared embedding 这类图形态。

## 文档边界

本文不记录完整变体数量、历史临时脚本、一次性 benchmark 结果。那些信息很容易随数据集和配置变动失效。
