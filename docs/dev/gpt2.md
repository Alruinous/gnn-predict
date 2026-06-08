# GPT-2 变体生成接入说明

## 结论

GPT-2 可以有机接入现有 `gnn_archs` 变体数据集生成流程，但当前代码不能直接接入。

已验证可复用的现有步骤：

- `ArchConfig` / `expand_arch_config` 可以接受 `base_model.name: gpt2`，并把它识别为 text 模型。
- `validate_model()` 可以验证正确初始化后的 GPT-2 分类模型。
- `export_onnx_model()` 可以导出 GPT-2 的 `full` 和 `architecture_only` ONNX。
- `write_randomized_onnx_model()` 可以把 GPT-2 architecture-only ONNX 随机初始化为可运行 full ONNX。
- `train_model()` 可以用 fake text batch 对 GPT-2 跑训练负载。
- `run_inference()` 可以用 fake text batch 对 GPT-2 跑推理负载。

当前缺口：

- `build_variant_model()` 对所有 text 模型都构建 `BertForSequenceClassification`，所以 `gpt2` YAML 现在会静默生成 BERT。
- 原生 `GPT2ForSequenceClassification` 接收二维 `attention_mask` 时，在当前 `torch.onnx.export(..., dynamo=False)` 和 Transformers 4.57.1 下会触发 `unordered_map::at`。
- GPT-2 需要一个很薄的模型包装层，把现有 text 路径的二维 `attention_mask` 转成 GPT-2 可导出的 4D additive mask。

## 已验证脚本

临时验证脚本：

```bash
PYTHONPATH=src uv run python tmp/verify_gpt2_gnn_archs_pipeline.py --device cpu
PYTHONPATH=src uv run python tmp/verify_gpt2_gnn_archs_pipeline.py --device cuda
```

验证覆盖：

| variant | params | layers | hidden | heads | inner | arch nodes | full initializers | ORT | train | infer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `gpt2_tiny` | 102,016 | 2 | 64 | 4 | 128 | 268 | 15 | pass | pass | pass |
| `gpt2_deep_narrow` | 678,624 | 8 | 96 | 4 | 192 | 772 | 39 | pass | pass | pass |
| `gpt2_mlp_narrow` | 1,863,168 | 4 | 256 | 4 | 256 | 421 | 22 | pass | pass | pass |
| `gpt2_mlp_wide` | 5,540,352 | 4 | 256 | 8 | 2048 | 424 | 23 | pass | pass | pass |

脚本同时验证了当前 builder 缺口：

```text
current_build_variant_model=BertForSequenceClassification gpt2_builder_gap=True
```

## GPT2Config 可变参数

推荐先纳入这些字段：

| 字段 | 作用 | 结构影响 |
| --- | --- | --- |
| `vocab_size` | token embedding 和分类输入词表 | 是 |
| `n_positions` | position embedding 和最大序列长度 | 是 |
| `n_embd` | hidden size | 是 |
| `n_layer` | Transformer block 数 | 是 |
| `n_head` | attention head 数 | 是 |
| `n_inner` | MLP 中间层宽度 | 是 |
| `activation_function` | MLP 激活函数 | 是 |
| `resid_pdrop` | residual/dropout 概率 | 否 |
| `embd_pdrop` | embedding dropout 概率 | 否 |
| `attn_pdrop` | attention dropout 概率 | 否 |
| `layer_norm_epsilon` | LayerNorm epsilon | 否 |
| `initializer_range` | 初始化尺度 | 否 |
| `scale_attn_by_inverse_layer_idx` | 分层 attention 缩放 | 是 |
| `reorder_and_upcast_attn` | attention 计算路径 | 是 |

硬约束：

- `n_embd % n_head == 0`
- `n_positions >= example_input_shape[1]`
- `n_layer > 0`
- `n_head > 0`
- `n_embd > 0`
- `n_inner is None or n_inner > 0`
- `vocab_size > max(bos_token_id, eos_token_id, pad_token_id)`
- `pad_token_id` 必须设置，否则 `batch_size > 1` 的 GPT-2 sequence classification 会失败
- `resid_pdrop`、`embd_pdrop`、`attn_pdrop` 必须在 `[0, 1)` 内
- ONNX 导出使用 `attn_implementation="eager"`，并通过包装层传入 4D mask

## 数量规模探索

临时探索脚本：

```bash
uv run python tmp/explore_gpt2_variant_scale.py
```

探索空间：

- hidden sizes: `64, 96, 128, 192, 256, 384, 512, 768`
- layers: `2, 4, 6, 8, 12`
- heads: `2, 4, 6, 8, 12`
- MLP ratio: `1, 2, 4, 8`
- token profiles: `compact, short, standard, long`
- output classes: `2, 10, 100`
- 过滤条件：`hidden % heads == 0` 且参数量不超过 `120M`

安全候选空间统计：

| 指标 | 数值 |
| --- | ---: |
| variants | 7,620 |
| unique shapes | 635 |
| min params | 85,504 |
| median params | 3,505,536 |
| p90 params | 31,724,544 |
| max params | 97,896,960 |

参数量分布：

| bucket | variants |
| --- | ---: |
| `0-0.25M` | 234 |
| `0.25-1M` | 1,648 |
| `1-5M` | 2,438 |
| `5-20M` | 1,962 |
| `20-80M` | 1,218 |
| `80-120M` | 120 |

全量 7,620 不适合第一批接入：它会明显超过当前 BERT 文本变体规模，也会让 GPT-2 在新增数据中占比过高。推荐首批使用 `216` 个变体：

```text
24 architecture shapes × 3 token profiles × 3 output class counts = 216 variants
```

若训练和推理都开启，最终监控 CSV 约产生 `432` 行阶段记录。

推荐规模分层：

| tier | 公式 | variants | 用途 |
| --- | --- | ---: | --- |
| smoke | `4 shapes × 2 token profiles × 2 output classes` | 16 | CI / 接入烟测 |
| pilot | `12 shapes × 3 token profiles × 3 output classes` | 108 | 首次 GPU 小批量验证 |
| recommended | `24 shapes × 3 token profiles × 3 output classes` | 216 | 首批正式数据生成 |
| broad | `48 shapes × 3 token profiles × 3 output classes` | 432 | 需要更密文本覆盖时使用 |

推荐首批 `216` 的覆盖情况：

| 指标 | 数值 |
| --- | ---: |
| unique shapes | 24 |
| min params | 102,016 |
| median params | 2,770,432 |
| p90 params | 11,103,744 |
| max params | 30,100,992 |

推荐 `216` 的参数量分布：

| bucket | variants |
| --- | ---: |
| `0-0.25M` | 12 |
| `0.25-1M` | 44 |
| `1-5M` | 91 |
| `5-20M` | 57 |
| `20-80M` | 12 |

推荐 `216` 的结构类型分布：

| shape tag | unique shapes |
| --- | ---: |
| balanced | 13 |
| deep_narrow | 4 |
| shallow_wide | 3 |
| mlp_narrow | 3 |
| mlp_wide | 1 |

推荐 token profiles：

| name | vocab | positions | sequence |
| --- | ---: | ---: | ---: |
| compact | 512 | 32 | 24 |
| short | 1024 | 64 | 40 |
| standard | 2048 | 128 | 64 |

推荐 output classes：

```text
2, 10, 100
```

## 最小接入内容

推荐新增一个 GPT-2 专用构建分支，不复用 BERT mutation 逻辑。

### 配置模型

在 `src/gnn_archs/config.py` 增加 GPT-2 配置块，并挂到 `VariantConfig`：

```python
class Gpt2ConfigOverride(StrictModel):
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


class VariantConfig(StrictModel):
    ...
    gpt2_config: Gpt2ConfigOverride | None = None
```

校验放在 GPT-2 builder 内即可，避免污染 image / YOLO 路径。

### 模型包装

在 `src/gnn_archs/variant_runner.py` 或独立 `src/gnn_archs/gpt2_builder.py` 中加入：

```python
class Gpt2ForGnnArchsSequenceClassification(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.model = GPT2ForSequenceClassification(config)
        self.config = self.model.config
        self.transformer = self.model.transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        prepared_mask = build_gpt2_export_attention_mask(input_ids, attention_mask)
        return self.model(
            input_ids=input_ids,
            attention_mask=prepared_mask,
            labels=labels,
        )
```

mask 构造：

```python
def build_gpt2_export_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = input_ids.shape
    mask_value = torch.finfo(torch.float32).min
    causal = torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=input_ids.device,
        )
    )
    causal_mask = torch.zeros(
        (sequence_length, sequence_length),
        dtype=torch.float32,
        device=input_ids.device,
    ).masked_fill(~causal, mask_value)
    causal_mask = causal_mask.view(
        1,
        1,
        sequence_length,
        sequence_length,
    ).expand(batch_size, 1, sequence_length, sequence_length)
    padding_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, sequence_length)
    return causal_mask.masked_fill(~padding_mask, mask_value)
```

### 构建分支

`build_variant_model()` 中在通用 text/BERT 分支前加 GPT-2 分支：

```python
if normalize_model_identifier(spec.base_model.name) == "gpt2":
    return build_gpt2_variant_model(spec)
```

`build_gpt2_variant_model()`：

```python
def build_gpt2_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    gpt2_config = spec.variant_config.gpt2_config
    if gpt2_config is None:
        raise ValueError("gpt2 variants require variant_config.gpt2_config")
    if spec.variant_config.target_output_classes is None:
        raise ValueError("gpt2 variants require target_output_classes")

    config = GPT2Config(
        num_labels=spec.variant_config.target_output_classes,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attn_implementation="eager",
        **gpt2_config.model_dump(),
    )
    validate_gpt2_config(config, spec.variant_config.example_input_shape[1])
    return Gpt2ForGnnArchsSequenceClassification(config)
```

### YAML 形态

不改 `variant_expander.py`，用 `single_variant_define` 显式列出 GPT-2 配置变体。

```yaml
base_model_groups:
- base_model:
    name: gpt2
    pretrained: false
  single_variant_define:
  - name: gpt2_tiny
    variant_config:
      target_input_channels: 1
      target_output_classes: 2
      example_input_shape: [2, 24]
      export_onnx: true
      onnx_export_mode: architecture_only
      run_training: true
      batch_size: 2
      training_measurement_min_seconds: 30.0
      use_fake_text_dataset: true
      run_inference: true
      pre_inference_cooldown_seconds: 5.0
      inference_measurement_min_seconds: 40.0
      gpt2_config:
        vocab_size: 512
        n_positions: 32
        n_embd: 64
        n_layer: 2
        n_head: 4
        n_inner: 128
        activation_function: gelu_new
        resid_pdrop: 0.0
        embd_pdrop: 0.0
        attn_pdrop: 0.0
    mutations: []
```

## 建议的结构集合

首批 `216` 使用下列 `24` 个 architecture shapes，再与 `3` 个 token profiles 和 `3` 个 output classes 做笛卡尔积：

| hidden | layers | heads | inner | tag |
| ---: | ---: | ---: | ---: | --- |
| 64 | 2 | 4 | 128 | balanced |
| 64 | 4 | 4 | 256 | balanced |
| 96 | 4 | 4 | 192 | balanced |
| 96 | 8 | 4 | 192 | deep_narrow |
| 128 | 2 | 4 | 512 | balanced |
| 128 | 4 | 4 | 128 | mlp_narrow |
| 128 | 6 | 4 | 512 | balanced |
| 128 | 8 | 8 | 256 | deep_narrow |
| 192 | 2 | 6 | 768 | balanced |
| 192 | 4 | 6 | 384 | balanced |
| 192 | 8 | 6 | 192 | deep_narrow |
| 192 | 12 | 6 | 384 | deep_narrow |
| 256 | 2 | 4 | 2048 | shallow_wide |
| 256 | 4 | 4 | 256 | mlp_narrow |
| 256 | 4 | 8 | 2048 | mlp_wide |
| 256 | 8 | 8 | 512 | balanced |
| 384 | 2 | 8 | 1536 | shallow_wide |
| 384 | 4 | 8 | 768 | balanced |
| 384 | 6 | 12 | 1536 | balanced |
| 384 | 8 | 12 | 384 | mlp_narrow |
| 512 | 2 | 8 | 2048 | shallow_wide |
| 512 | 4 | 8 | 1024 | balanced |
| 512 | 6 | 8 | 2048 | balanced |
| 768 | 4 | 12 | 3072 | balanced |

这些组合覆盖：

- 小参数模型
- 深窄模型
- 浅宽模型
- MLP 窄/宽对照
- 中型 balanced
- 约 `30M` 参数以内的较大文本模型

## 接入后验证清单

```bash
uv run ruff check src/gnn_archs tests/test_variant_runner.py tests/test_arch_configs.py
uv run ty check src/gnn_archs tests/test_variant_runner.py tests/test_arch_configs.py
uv run python -m pytest -q tests/test_variant_runner.py -k "gpt2 or text"
uv run python -m pytest -q tests/test_arch_configs.py
```

端到端小配置：

```bash
uv run python main.py \
  --config config/arch/gpt2_variants.yaml \
  --output_dir output \
  --gpu_node node0
```

重点检查：

- `metadata.model_kind == "text"`
- `metadata.parameter_count` 随 GPT2Config 变化
- `onnx_export.graph_info.runtime_input_names == ["input_ids", "attention_mask"]`
- `architecture_only` 导出无 initializers，且存在 parameter inputs
- 随机初始化后的 ONNX 只暴露 `input_ids` 和 `attention_mask`
- training / inference 阶段都有时间窗口和轮数
