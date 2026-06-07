# LLaMA 和 Gemma 模型变体引入方案

## 背景

`docs/dev/qwen.md` 已经把现代大语言模型接入当前项目的基本口径确定为 causal LM，而不是 GPT-2 旧路径里的 sequence classification。LLaMA 和 Gemma 与 Qwen 一样属于 decoder-only 语言模型，适合复用 fake token、next-token loss、Prefill-only 推理和 architecture-only ONNX 导出的总体思路。

但它们不能按“结构相似即可直接复用”处理。LLaMA 3.2、Gemma 2、Gemma 3 在配置字段、attention 实现、词表规模、layernorm 布局、RoPE 参数、sliding window 以及 Transformers 模型类上都有差异。首版应把它们归入通用 causal LM 分支，同时保留 Qwen3.5 自己的 ONNX 导出补丁。

## 目标

为当前项目增加 LLaMA 和 Gemma 两类现代开源 LLM 变体，扩充现代大模型类别的数据覆盖面，并保持与 Qwen 数据的实验口径一致：

- 训练阶段使用 full forward/backward 的 causal LM loss。
- Prefill 阶段使用一次完整 prompt forward-only。
- 首批只调整 batch size 和 sequence length，不修改 checkpoint 内部结构参数。
- 首批使用 fake token，不引入真实 tokenizer、padding、LoRA、gradient checkpointing、KV cache decode 或量化模型。
- ONNX 导出使用 architecture-only，输出保留完整 logits。

## 资料来源

- Hugging Face Transformers loading docs: https://huggingface.co/docs/transformers/models
- Hugging Face LLaMA model docs: https://huggingface.co/docs/transformers/model_doc/llama
- Hugging Face Gemma 2 model docs: https://huggingface.co/docs/transformers/model_doc/gemma2
- Hugging Face Gemma 3 model docs: https://huggingface.co/docs/transformers/model_doc/gemma3
- Hugging Face repos: `unsloth/Llama-3.2-1B`, `unsloth/Llama-3.2-3B`, `unsloth/gemma-2-2b`, `unsloth/gemma-3-1b-pt`
- ModelScope repos:
  - https://modelscope.ai/models/unsloth/Llama-3.2-1B
  - https://modelscope.ai/models/unsloth/Llama-3.2-3B
  - https://modelscope.ai/models/unsloth/gemma-2-2b
  - https://modelscope.ai/models/unsloth/gemma-3-1b-pt

## 本地模型状态

当前 `/data/Models/unsloth` 下有四个可用 checkpoint：

| 本地目录 | HF 任务 | HF 模型类 | 架构 | 参数量 |
| --- | --- | --- | --- | ---: |
| `/data/Models/unsloth/Llama-3.2-1B` | text-generation | `AutoModelForCausalLM` | `llama` | 1,235.8M |
| `/data/Models/unsloth/Llama-3.2-3B` | text-generation | `AutoModelForCausalLM` | `llama` | 3,212.7M |
| `/data/Models/unsloth/gemma-2-2b` | text-generation | `AutoModelForCausalLM` | `gemma2` | 2,614.3M |
| `/data/Models/unsloth/gemma-3-1b-pt` | text-generation | `AutoModelForCausalLM` | `gemma3_text` | 999.9M |

`/data/Models/meta-llama` 和 `/data/Models/google` 当前不是本次目标模型的有效根目录。配置里也不建议使用 `unsloth/...` 这种带斜杠的 `base_model.name`，因为现有 `build_variant_config_grid_name()` 会把 `base_model.name` 拼进 `spec.name`，而 `spec.name` 会进入 ONNX 文件名。首版配置应使用本地 checkpoint 目录名，例如 `Llama-3.2-1B`。

`unsloth/gemma-3-4b-it` 暂不纳入首批。它在 HF 元数据里是 `image-text-to-text`，模型类为 `AutoModelForImageTextToText`，对应 `Gemma3ForConditionalGeneration`，不是当前 fake token causal LM 路径可直接覆盖的纯文本 checkpoint。

## 临时验证

临时验证均通过 here-doc Python 脚本运行，ONNX 文件放在 `TemporaryDirectory` 中，脚本退出后自动删除。

### 配置扫描

| checkpoint | model_type | architectures | vocab | max_position_embeddings | hidden | layers | heads | kv heads | sliding_window |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-3.2-1B` | `llama` | `LlamaForCausalLM` | 128256 | 131072 | 2048 | 16 | 32 | 8 | - |
| `Llama-3.2-3B` | `llama` | `LlamaForCausalLM` | 128256 | 131072 | 3072 | 28 | 24 | 8 | - |
| `gemma-2-2b` | `gemma2` | `Gemma2ForCausalLM` | 256000 | 8192 | 2304 | 26 | 8 | 4 | 4096 |
| `gemma-3-1b-pt` | `gemma3_text` | `Gemma3ForCausalLM` | 262144 | 32768 | 1152 | 26 | 4 | 1 | 512 |

补充字段：

- LLaMA 3.2 使用 `rope_scaling={"rope_type": "llama3", "factor": 32.0, ...}`，原始上下文为 8192，本地 config 上限为 131072。
- Gemma 2 有 `query_pre_attn_scalar=256`、`final_logit_softcapping=30.0`、`attn_logit_softcapping=50.0`。
- Gemma 3 1B 的 `rope_scaling` 区分 sliding attention 和 full attention。
- 四个本地 config 的 `torch_dtype` 都声明为 bf16；V100 首版应按 Qwen 口径强制加载 fp16。

### 参数命名扫描

四个模型的 safetensors key 都以 `model.*` 为主：

- LLaMA: `model.embed_tokens.weight`、`model.layers.0.self_attn.q_proj.weight`、`model.layers.0.mlp.gate_proj.weight`。
- Gemma 2: 同样有 `q_proj/k_proj/v_proj/o_proj` 和 gated MLP，但额外存在 `pre_feedforward_layernorm`、`post_feedforward_layernorm`。
- Gemma 3: 在 Gemma 2 风格之外，还出现 `model.layers.0.self_attn.q_norm.weight` 和 `k_norm.weight`。

因此首版不应写任何按层名改结构的逻辑。只加载 checkpoint，变体轴只放在输入形状层。

### Forward smoke

运行条件：V100 32GB，`dtype=torch.float16`，`attn_implementation="eager"`，`batch=1`，`seq=8`，fake token，`labels=input_ids.clone()`。

| checkpoint | class | logits shape | loss finite | peak allocated |
| --- | --- | --- | --- | ---: |
| `Llama-3.2-1B` | `LlamaForCausalLM` | `[1, 8, 128256]` | true | 2.320 GB |
| `Llama-3.2-3B` | `LlamaForCausalLM` | `[1, 8, 128256]` | true | 6.002 GB |
| `gemma-2-2b` | `Gemma2ForCausalLM` | `[1, 8, 256000]` | true | 4.898 GB |
| `gemma-3-1b-pt` | `Gemma3ForCausalLM` | `[1, 8, 262144]` | true | 1.940 GB |

这说明四个本地模型都能复用 Qwen 的 fake token causal LM loss 口径。Gemma 的词表明显更大，完整 logits 输出会比同规模 LLaMA 更重。

补充验证当前项目的完整训练口径，即 full-parameter AdamW `optimizer.step()`：

| checkpoint | shape | step | peak allocated |
| --- | --- | --- | ---: |
| `Llama-3.2-3B` | `[1, 8]` | pass | 29.941 GB |
| `Llama-3.2-3B` | `[1, 128]` | pass | 29.971 GB |
| `Llama-3.2-3B` | `[4, 256]` | pass | 30.184 GB |
| `gemma-2-2b` | `[1, 8]` | pass | 24.373 GB |
| `gemma-2-2b` | `[1, 128]` | pass | 24.433 GB |
| `gemma-2-2b` | `[4, 256]` | pass | 24.858 GB |

因此四个目标模型都可以进入 full-flow，但 `Llama-3.2-3B` 在 V100 32GB 上余量很小，首批不外推到 1024 序列或更大的 batch×sequence 组合。

### ONNX 和 onnx_tool smoke

运行条件：V100 32GB，`dtype=torch.float16`，`attn_implementation="eager"`，`batch=1`，`seq=4`，architecture-only，opset 14，导出后调用 `build_graph_data_from_onnx()`。

| checkpoint | ONNX nodes | output shape | initializer count | PyG nodes | PyG edges | file size |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `Llama-3.2-1B` | 1852 | `[1, 4, 128256]` | 0 | 1852 | 2193 | 0.34 MB |
| `Llama-3.2-3B` | 3184 | `[1, 4, 128256]` | 0 | 3184 | 3777 | - |
| `gemma-3-1b-pt` | 4542 | `[1, 4, 262144]` | 0 | 4542 | 5282 | 0.84 MB |
| `gemma-2-2b` | 4026 | `[1, 4, 256000]` | 0 | 4026 | 4712 | - |

四个目标模型导出验证均通过，未复现 Qwen3.5 的 BFLOAT16 `Constant` 解析问题，也未遇到 `onnx_tool` 无法 shape infer 的新算子。这个结论覆盖短序列 architecture-only 图；正式生成仍应以完整配置的首批结果为准。

### Attention backend

默认加载时，`Llama-3.2-1B` 和 `gemma-3-1b-pt` 的 `config._attn_implementation` 都是 `sdpa`。临时把 `model.config._attn_implementation` 改为 `eager` 后，首层 attention 模块读取到的也是 `eager`，说明可以在 ONNX 导出期间临时切换并恢复。

首版不建议为了导出兼容而让训练和 Prefill 全程使用 eager。训练和 Prefill 应保留默认 SDPA，ONNX 导出阶段用上下文管理器临时切换到 eager。

## 与 Qwen 接入的共同口径

LLaMA/Gemma 首版与 Qwen 保持这些一致点：

- `example_input_shape` 固定为 `[batch_size, sequence_length]`。
- `max_sequence_length` 与 `example_input_shape[1]` 同步。
- `training_batch_sizes` 与 `example_input_shape[0]` 同步。
- `input_ids` 使用 `torch.randint(0, vocab_size, [B, S])`。
- `attention_mask` 固定为全 1，形状 `[B, S]`。
- `labels` 使用 `input_ids.clone()`，形状 `[B, S]`。
- ONNX adapter 返回 `model(...).logits`，不裁剪最后一个 token。
- `use_cache` 固定关闭，不进入 decode/KV cache 场景。

## 与 Qwen 接入的差异

| 项目 | Qwen | LLaMA/Gemma 首版处理 |
| --- | --- | --- |
| 模型根目录 | `/data/Models/Qwen` | `/data/Models/unsloth` |
| 模型识别 | `is_qwen_model_name()` | 新增 `is_causal_lm_model_name()` 和 family resolver |
| 默认 attention | 按具体 Qwen 实现 | 训练/Prefill 保留默认 SDPA，导出临时 eager |
| Qwen3.5 linear attention | 需要专门 fallback | 不适用，不能复用到 LLaMA/Gemma |
| constant folding | Qwen3.5 关闭 | LLaMA/Gemma smoke 可保持开启 |
| 词表规模 | Qwen 系列差异较大 | Gemma 2/3 词表约 256K，需要更保守估计 logits 内存 |
| layernorm 布局 | Qwen 特有命名 | Gemma 2/3 有额外 layernorm，不做结构 mutation |

## 实现方案

### 通用 causal LM 识别

在 `src/gnn_archs/config.py` 中新增 causal LM family 判断：

- `CAUSAL_LM_MODEL_PREFIXES = ("qwen", "llama", "gemma")`
- `is_causal_lm_model_name(model_name: str) -> bool`
- `get_causal_lm_family(model_name: str) -> str`

`TEXT_MODEL_PREFIXES` 可以加入 `llama` 和 `gemma`，但所有会影响训练标签、模型构造、ONNX 导出和 Prefill 的地方必须优先判断 causal LM，避免落入 BERT sequence classification 分支。

`BaseModelGroup.validate_single_variant_targets()` 中的 Qwen 特例应改为 causal LM 特例：causal LM 变体必须省略 `target_input_channels` 和 `target_output_classes`，且不支持 mutation。

### Builder 抽象

新增 `src/gnn_archs/causal_lm_builder.py`，把 Qwen 当前可复用逻辑上移：

- `CausalLMOnnxLogitsExport`
- `build_causal_lm_variant_model(spec)`
- `resolve_causal_lm_model_path(model_name)`
- `disable_causal_lm_cache(model)`

路径解析建议：

| family | 配置名示例 | 本地解析 |
| --- | --- | --- |
| qwen | `Qwen3-1.7B` | `/data/Models/Qwen/Qwen3-1.7B` |
| llama | `Llama-3.2-1B` | `/data/Models/unsloth/Llama-3.2-1B` |
| gemma | `gemma-3-1b-pt` | `/data/Models/unsloth/gemma-3-1b-pt` |

`src/gnn_archs/qwen_builder.py` 可以保留薄 wrapper，减少一次性 diff；后续稳定后再考虑删除 Qwen 专用文件。首版不用引入可配置模型根目录，避免把本机目录习惯扩散成配置 schema。

加载建议：

- `AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, local_files_only=True)`
- 不设置 `trust_remote_code=True`，当前四个模型都由 Transformers 内置类支持。
- 不在 builder 里全局设置 `attn_implementation="eager"`。
- 关闭 `config.use_cache`、`config.text_config.use_cache` 和 `generation_config.use_cache`。

### Runner 改造点

`src/gnn_archs/variant_runner.py` 的 Qwen 专用路径应收敛为 causal LM 路径：

- `build_variant_model()`：causal LM 先于 generic text 分支处理。
- `validate_model()`：仍按 text batch 构造，但 causal LM 输出是 `[B, S, V]`。
- `build_training_batch()`：causal LM 标签为 `input_ids.clone()`。
- `build_example_batch()`：causal LM 使用二维 fake token batch，并校验 `max_position_embeddings` 或 `text_config.max_position_embeddings`。
- `run_prefill()`：从仅支持 Qwen 改成支持 causal LM。
- `export_onnx_model()`：causal LM 使用 `export_causal_lm_onnx_model()`。
- metadata：`pretrained_weights_loaded` 对所有 causal LM 为 true；`model_kind` 可继续写具体 family，如 `qwen`、`llama`、`gemma`。

### ONNX 导出

新增通用 `export_causal_lm_onnx_model()`：

- 输入名固定为 `input_ids`、`attention_mask`。
- 输出名固定为 `logits`。
- adapter 返回完整 logits。
- `export_params` 仍由 `onnx_export_mode == "full"` 决定，首版配置使用 `architecture_only`。
- 默认 opset 14。
- 导出期间临时把 `config._attn_implementation` 和 `text_config._attn_implementation` 切到 `eager`，结束后恢复。
- 如果 family 是 Qwen3.5，继续执行当前 linear attention fallback，并关闭 constant folding。
- LLaMA/Gemma 不复用 Qwen3.5 fallback，constant folding 保持默认开启。

导出期间切换 attention 的原因是：默认 SDPA 更接近训练/Prefill 运行路径，但 legacy ONNX exporter 对 SDPA 路径更敏感；代表性 smoke 已验证 eager 导出和 onnx_tool 解析可行。

## 首批变体配置

首批不要追求数量最大化。Qwen 复训记录显示，Qwen test 子集只有 29 条时表现明显难于 non-Qwen，且 Qwen training 比 prefill 难很多。扩充 LLaMA/Gemma 的目的应是增加现代 LLM family 的多样性，而不是只在单个 family 上插值更多相邻形状。

最终实现按模型 family 拆成两个配置文件：

- `config/arch/llama_variants.yaml`
- `config/arch/gemma_variants.yaml`

四个本地 checkpoint 都启用 training + prefill：

- `Llama-3.2-1B`
- `Llama-3.2-3B`
- `gemma-3-1b-pt`
- `gemma-2-2b`

统一形状：

```yaml
- name: bs1_s128
  overrides: {example_input_shape: [1, 128], max_sequence_length: 128, training_batch_sizes: [1]}
- name: bs1_s256
  overrides: {example_input_shape: [1, 256], max_sequence_length: 256, training_batch_sizes: [1]}
- name: bs1_s384
  overrides: {example_input_shape: [1, 384], max_sequence_length: 384, training_batch_sizes: [1]}
- name: bs1_s512
  overrides: {example_input_shape: [1, 512], max_sequence_length: 512, training_batch_sizes: [1]}
- name: bs2_s128
  overrides: {example_input_shape: [2, 128], max_sequence_length: 128, training_batch_sizes: [2]}
- name: bs2_s256
  overrides: {example_input_shape: [2, 256], max_sequence_length: 256, training_batch_sizes: [2]}
- name: bs2_s384
  overrides: {example_input_shape: [2, 384], max_sequence_length: 384, training_batch_sizes: [2]}
- name: bs3_s128
  overrides: {example_input_shape: [3, 128], max_sequence_length: 128, training_batch_sizes: [3]}
- name: bs3_s256
  overrides: {example_input_shape: [3, 256], max_sequence_length: 256, training_batch_sizes: [3]}
- name: bs4_s128
  overrides: {example_input_shape: [4, 128], max_sequence_length: 128, training_batch_sizes: [4]}
- name: bs4_s256
  overrides: {example_input_shape: [4, 256], max_sequence_length: 256, training_batch_sizes: [4]}
```

每个模型 11 个 variant，每个 variant 生成 training 和 prefill 两条 phase 记录。`llama_variants.yaml` 和 `gemma_variants.yaml` 各 22 个 variant；若后续全量生成，两个文件合计 88 条 phase 记录。

### 暂不纳入

- `gemma-3-4b-it`：多模态条件生成模型，不走纯文本 `AutoModelForCausalLM`。
- 7B 及以上 dense 模型：V100 上全参数训练风险高，首批不放默认 full-flow。
- instruct/chat 版本：聊天模板和特殊 token 更依赖真实 tokenizer，fake token 首版不需要。
- GGUF、AWQ、GPTQ、bnb 4bit：量化会改变算子和显存路径，应单独建训练协议。

## 验证计划

代码实现后按下面顺序验证：

- 单元测试：新增 causal LM predicate、路径解析、target 字段校验、fake batch labels、Prefill 支持和 ONNX logits shape。
- 配置测试：新增 `config/arch/llama_variants.yaml` 和 `config/arch/gemma_variants.yaml` 的展开数量和 shape 同步校验。
- 本地 smoke：对四个目标模型跑 `batch=1, seq=8` 的 forward loss。
- ONNX smoke：对四个目标模型各跑一个 `batch=1, seq=4` 的 architecture-only 导出和 `build_graph_data_from_onnx()`。
- 小规模监控：先跑每个模型 1 个 shape，确认 Prometheus 抽取、quality filter、extract、scale 全链路。
- 数据集纳入：确认 LLaMA/Gemma 样本按 family、phase、base model 分布进入 train/val/test。
- GNN 复训评估：至少报告 overall、modern LLM 子集、family 子集、phase 子集和 per-target 的 WAPE、MAE、RMSE、R2；同时区分 single GNN、direct stacker、residual stacker。

## 风险

- LLaMA/Gemma 原始 dtype 为 bf16，V100 首版必须按 fp16 跑，避免引入 BF16 ONNX/onnx_tool 兼容问题。
- Gemma 2/3 vocab 约 256K，完整 logits 会增加显存和图输出维度；训练形状不应直接照搬 Qwen 最大组合。
- 3B/2B 模型短 forward 可跑，不代表 full training 的 512/1024 长序列组合可跑。
- 默认 SDPA 与导出 eager 存在路径差异；该差异只应影响 ONNX 结构特征，不应污染训练/Prefill 性能测量。
- 现有 Qwen3.5 fallback 不能泛化到 LLaMA/Gemma；任何新导出失败都应按具体 op 或 config 处理。
- 新增现代 LLM 样本数量仍小，复训结果不能只看 overall；必须单独看 LLaMA/Gemma/Qwen 子集。

## 阶段性结论

LLaMA 3.2 1B/3B、Gemma 2 2B、Gemma 3 1B 都适合引入当前项目，但首版应以通用 causal LM 分支接入，而不是复制 Qwen 特例或走 generic text sequence classification。根据补充 smoke，四个模型都进入 full-flow；其中 `Llama-3.2-3B` 显存余量最小，因此首批只使用 11 个保守 shape。

这批模型能增加现代 LLM family 多样性，但不会单靠 88 条 phase 样本彻底解决 Qwen 子集泛化问题。更合理的推进方式是先完成稳定接入和监控链路，再基于复训后的 family/phase/per-target 误差决定是否扩展形状或追加更多模型家族。
