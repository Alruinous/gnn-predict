# Qwen3Config 变体实现记录

## 实现口径

当前 Qwen3 变体使用 `transformers.Qwen3Config` 构造随机初始化 causal LM，不再依赖
本地 Qwen checkpoint。配置入口是 `config/arch/qwen3_variants.yaml`。

旧的 `config/arch/qwen_variants.yaml` 和 `src/gnn_archs/qwen_builder.py` 已移除。
Qwen3 变体使用 `base_model.name: qwen3`，要求 `pretrained: false`，并且必须提供
`variant_config.qwen3_config`。

## 变体轴

结构轴覆盖 `hidden_size`、`intermediate_size`、`num_hidden_layers`、
`num_attention_heads`、`num_key_value_heads`。固定项包括 Qwen3 词表规模
`151936`、`head_dim: 128`、`max_position_embeddings: 40960`、RoPE theta `1000000`。

运行形状轴覆盖 `batch_size`、prompt length 和 decode 输出长度。`batch_size`
必须与 `example_input_shape[0]` 保持一致。decode 使用
`generate(max_new_tokens=decode_max_output_length, use_cache=True)`；配置校验要求
`prompt length + decode_max_output_length <= max_position_embeddings`。

`decode_max_output_length` 表示最多新增 token 数，不替代模型上下文上限。公开真实对话数据
中 response 平均长度可以大于 prompt，但长上下文、摘要、RAG 等场景也常见 output 小于
prompt。因此 decode-only 轴同时保留 output 小于、等于、大于 prompt 的组合。

长 decode 组使用较小的 `h512_l12` 静态结构，把显存留给更大的 prompt 和 KV cache。
该组变体名使用 `qwen3_token_bs..._o...`，覆盖 12 个 prompt 长度、12 个输出长度和 2 个 batch size。

## 阶段

Qwen3 变体支持三类阶段：

- `training`：fake token next-token loss，执行完整 forward/backward/optimizer step。
- `prefill`：完整 prompt forward-only。
- `decode`：基于 prompt 调用 `generate`，最多生成 `decode_max_output_length` 个新 token。

监控 CSV 和 GNN 图级特征新增 `decode_output_length`。非 decode 阶段该值为 `0`。

## V100 上界验证

验证设备：Tesla V100-PCIE-32GB。

只验证当前配置中的上界组合，不逐个扫描变体：

| 阶段 | 变体 | 验证内容 | 峰值 allocated | 峰值 reserved | 结果 |
| --- | --- | --- | ---: | ---: | --- |
| training | `qwen3_h2048_l28_bs1_s1024` | 单次完整训练 step | 16724.3 MB | 未记录 | 通过 |
| prefill | `qwen3_h2560_l36_bs1_s512` | 单次 forward | 7885.4 MB | 8028.0 MB | 通过 |
| decode | `qwen3_h2560_l36_decode_bs1_s256_out32` | 单次 generate | 7812.7 MB | 7838.0 MB | 通过 |
| decode | `qwen3_h2048_l28_decode_bs2_s512_out128` | 单次 generate | 3531.9 MB | 未记录 | 通过 |
| decode | `qwen3_h2048_l28_decode_bs2_s512_out256` | 单次 generate | 3531.9 MB | 3570.0 MB | 通过 |
| decode | `qwen3_token_bs2_s4096_o2048` | 单次 generate | 3012.0 MB | 3200.0 MB | 通过 |

这些组合分别覆盖训练结构上界、prefill 静态参数和动态激活上界、decode 静态参数上界，
以及 decode 更大的 batch/prompt/output 动态组合。实测均未 OOM。

## 配置规模

当前 `config/arch/qwen3_variants.yaml` 展开后共 `524` 个变体：

- `176` 个 training + prefill + decode 组合。
- `20` 个 prefill-only 组合。
- `328` 个 decode-only 组合，其中 `288` 个来自长 decode 组。
