# YOLO 检测模型变体生成说明

## 适用范围

本文档说明 gnn-predict 项目中 YOLO 系列检测模型的变体生成机制，包括：

- 支持的模型族和 scale 范围
- 变体展开逻辑和数量
- YAML 级 pre-build mutation 类型
- ONNX 导出的关键修复（`m.export = True`）
- 与非 YOLO 模型的隔离机制

## 涉及的代码和配置文件

| 文件 | 职责 |
| --- | --- |
| `src/gnn_archs/config.py` | `DETECTION_MODEL_PREFIXES`、`is_detection_model_name()` |
| `src/gnn_archs/yolo_builder.py` | 模型构建、ONNX 导出、训练、推理的 YOLO 专用实现 |
| `src/gnn_archs/variant_runner.py` | 四处 detection 分派入口（build / export / train / infer） |
| `config/arch/yolo11_variants.yaml` | YOLO11 变体配置（23 个 scale × 5 nc × 40 mutations = 4,600 变体） |
| `config/arch/yolov8_variants.yaml` | YOLOv8 变体配置（13 × 5 × 40 = 2,600 变体） |
| `config/arch/yolov5_variants.yaml` | YOLOv5 变体配置（10 × 4 × 30 = 1,200 变体） |
| `config/arch/yolov9_variants.yaml` | YOLOv9 变体配置（10 × 3 × 20 = 600 变体） |
| `config/arch/yolov10_variants.yaml` | YOLOv10 变体配置（10 × 3 × 15 = 450 变体） |
| `tests/test_arch_configs.py` | YOLO 变体的验证和展开测试 |

## 支持的模型族

通过 `config.py` 中的 `DETECTION_MODEL_PREFIXES` 识别：

```
yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, yoloe
```

识别逻辑：对模型名做 `normalize_model_identifier()` 后检查前缀匹配。

## Scale 体系

### 原生 scale

直接使用 ultralytics 原生支持的 scale 字母后缀：

| 模型族 | 原生 scale |
| --- | --- |
| YOLO11 | n, s, m, l, x |
| YOLOv8 | n, s, m, l, x |
| YOLOv5 | n, s, m, l, x |
| YOLOv9 | t, s, m, c, e |
| YOLOv10 | n, s, m, b, l, x |

### 自定义 scale

通过下划线命名（如 `yolo11_pico`）触发自定义 scale 路径。
自定义 scale 在 `yolo_builder.py` 的 `CUSTOM_SCALES` 字典中注册，
每个条目定义 `[depth_multiple, width_multiple, max_channels]` 三元组。

当前注册的自定义 scale（18 个）：

| scale 名 | depth | width | max_ch | 设计意图 |
| --- | --- | --- | --- | --- |
| pico | 0.33 | 0.15 | 512 | 极小模型 |
| micro | 0.33 | 0.20 | 768 | 微型模型 |
| ns_1 | 0.50 | 0.31 | 1024 | nano-small 插值 1 |
| ns_2 | 0.50 | 0.375 | 1024 | nano-small 插值 2 |
| sm_1 | 0.50 | 0.625 | 768 | small-medium 插值 1 |
| sm_2 | 0.50 | 0.75 | 768 | small-medium 插值 2 |
| sm_3 | 0.50 | 0.875 | 512 | small-medium 插值 3 |
| ml_1 | 0.625 | 1.00 | 512 | medium-large 插值 1 |
| ml_2 | 0.75 | 1.00 | 512 | medium-large 插值 2 |
| lx_1 | 1.00 | 1.125 | 512 | large-xlarge 插值 1 |
| lx_2 | 1.00 | 1.25 | 512 | large-xlarge 插值 2 |
| huge | 1.50 | 2.00 | 1024 | 超大模型 |
| mega | 2.00 | 2.00 | 1024 | 极大模型 |
| tiny_wide | 0.33 | 0.50 | 512 | 浅层宽通道 |
| shallow_wide | 0.33 | 1.00 | 512 | 极浅宽通道 |
| deep_narrow | 1.50 | 0.25 | 1024 | 深层窄通道 |
| deep_mid | 1.50 | 0.50 | 768 | 深层中等通道 |
| ultra_deep | 2.00 | 0.50 | 512 | 极深窄通道 |

实现原理：向 ultralytics YAML dict 的 `scales` 字段注入自定义条目，
然后设 `scale` 键为该条目名称，`parse_model()` 原生处理所有缩放。

## 变体展开逻辑

变体展开复用项目已有的 `variant_expander.py`，遵循与 image/text 模型完全一致的
组合笛卡尔积展开机制：

```
total_variants = Σ (每个 base_model_group 的 |output_classes| × |mutation_sets|)
```

每个 YAML 配置使用 YAML 锚点 (`&anchor`) 和别名 (`*alias`) 实现 DRY：
第一个 `base_model_group` 定义完整的 `combinatorial_variant_grid`，
后续 group 直接引用。

### 变体总量

| 配置文件 | 模型组数 | 输出类别数 | Mutation 集数 | 变体数 |
| --- | --- | --- | --- | --- |
| yolo11_variants.yaml | 23 | 5 | 40 | 4,600 |
| yolov8_variants.yaml | 13 | 5 | 40 | 2,600 |
| yolov5_variants.yaml | 10 | 4 | 30 | 1,200 |
| yolov9_variants.yaml | 10 | 3 | 20 | 600 |
| yolov10_variants.yaml | 10 | 3 | 15 | 450 |
| **合计** | **66** | | | **9,450** |

## YAML 级 Pre-build Mutation 类型

YOLO 的 mutation 与 image/text 模型不同：它们作用于 **YAML dict 层面**
（即架构定义），而非已构建的 `nn.Module`。这些 mutation 在 `build_detection_model()`
调用 `parse_model()` **之前**执行。

已注册的 6 种 mutation 类型（`YOLO_YAML_MUTATION_TYPES`）：

| 类型 | 参数 | 作用 |
| --- | --- | --- |
| `BackboneModuleReplace` | `from_module`, `to_module`, `layers`(可选) | 替换 backbone 中的模块类型 |
| `HeadModuleReplace` | `from_module`, `to_module`, `layers`(可选) | 替换 head 中的模块类型 |
| `ActivationOverride` | `activation` | 覆盖全局激活函数 |
| `SPPFKernelReplace` | `kernel_size` | 替换 SPPF 层的 kernel 大小 |
| `BackboneDepthOverride` | `repeat_value` | 覆盖 backbone 重复次数 |
| `ChannelScale` | `scale_factor` | 缩放通道数 |

各模型族的核心模块差异：

| 模型族 | 核心模块 | 常见替换对 |
| --- | --- | --- |
| YOLO11 | C3k2 | C3k2 → C2f, C3k2 → C3 |
| YOLOv8 | C2f | C2f → C3, C2f → C3k2 |
| YOLOv5 | C3 | C3 → C2f, C3 → C3k2 |
| YOLOv9 | RepNCSPELAN4 | RepNCSPELAN4 → C2f |
| YOLOv10 | C2fCIB | C2fCIB → C2f |

## ONNX 导出关键修复：`m.export = True`

### 问题背景

YOLO 的 `Detect` head 在正常推理模式下包含 grid/anchor 计算和 NMS 后处理操作。
这些操作使用了部分 ONNX 不支持的算子（如动态 shape 操作、自定义 grid 生成），
导致 `torch.onnx.export()` 失败。

ultralytics 的解决方案是在 `Detect`/`Segment`/`Pose` 等 head 模块中设置
`export` 属性：当 `self.export = True` 时，head 的 `forward()` 会跳过
不可导出的后处理，只输出原始预测 tensor。

### 修复实现

在 `yolo_builder.py` 的 `export_detection_onnx()` 函数中，
在 `torch.onnx.export()` **之前**遍历所有子模块设置 `export = True`：

```python
# yolo_builder.py → export_detection_onnx()
model.eval()

# 关键修复：设置 Detect head 导出模式
for m in model.modules():
    if hasattr(m, "export"):
        m.export = True

# 之后才调用 torch.onnx.export(...)
```

使用 `hasattr(m, "export")` 而非检查特定类名，这样可以兼容 ultralytics
后续新增的带 `export` 属性的 head 模块。

### 与非 YOLO 模型的隔离

**此修复不会影响非 YOLO 模型**。隔离机制如下：

```
variant_runner.py: export_onnx_model()
  │
  ├── if is_detection_model_name(...)  → yolo_builder.export_detection_onnx()
  │                                      └── m.export = True（仅在此路径执行）
  │
  └── else → 原有的 OnnxExportWrapper 路径（image/text 模型）
              └── 不涉及 m.export
```

四个分派函数（`build_variant_model`、`export_onnx_model`、`train_model`、
`run_inference`）的入口都通过 `is_detection_model_name()` 做前缀匹配判断。
只有模型名以 `DETECTION_MODEL_PREFIXES` 中的前缀开头时才进入 YOLO 路径，
其他模型走原有的 image/text 路径，代码完全不交叉。

## 构建流程总结

```
YAML 配置
  │
  ▼
variant_expander.py  →  展开为 ResolvedVariantSpec 列表
  │
  ▼
variant_runner.py: build_variant_model()
  │
  ├── is_detection_model_name() = True
  │   └── yolo_builder.build_detection_model()
  │       ├── 标准 scale → YOLO(name.yaml).model
  │       └── 自定义 scale / mutation
  │           → load YAML dict → apply scale/mutations → 写临时 YAML → YOLO(tmp.yaml).model
  │
  └── is_detection_model_name() = False
      └── 原有 timm/transformers 路径

variant_runner.py: export_onnx_model()
  │
  ├── detection → yolo_builder.export_detection_onnx()
  │               ├── m.export = True（关键修复）
  │               └── torch.onnx.export(opset=14)
  │
  └── non-detection → OnnxExportWrapper 原有路径

variant_runner.py: train_model()
  │
  ├── detection → yolo_builder.train_detection_model()
  │               └── forward + sum reduction + backward
  │
  └── non-detection → 原有 image/text 训练路径

variant_runner.py: run_inference()
  │
  ├── detection → yolo_builder.run_detection_inference()
  │               └── warmup + 计时推理循环
  │
  └── non-detection → 原有 image/text 推理路径
```

## 测试覆盖

YOLO 变体相关的测试在 `tests/test_arch_configs.py` 中，
通过参数化覆盖 5 个 YAML 配置文件：

- `test_arch_configs_validate_and_expand` — 验证 YAML 合法、展开数量正确
- `test_arch_config_mutations_match_model_kind` — 验证 mutation 类型在 `YOLO_YAML_MUTATION_TYPES` 范围内
- `test_arch_configs_define_phase_isolation_for_training_inference_pairs` — 验证训练/推理阶段隔离参数

当前 15 个 YOLO 相关测试全部通过。
