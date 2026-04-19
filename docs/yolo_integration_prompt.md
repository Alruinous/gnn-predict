# YOLO 变体生成集成实施指南

> **目标**：将 YOLO 检测模型的变体生成逻辑集成到 `gnn-predict` 项目的 `gnn_archs` 子系统中，用于为 GNN 性能预测器生成大量结构多样的检测模型计算图。
>
> 分支 `wjh-test`
>
> **核心原则**：直接复用 ultralytics 原生的 `scales` dict + `parse_model()` compound scaling 机制，不重新实现任何缩放逻辑。所有 YOLO 相关代码封装在一个新文件 `yolo_builder.py` 中，现有文件只加最小化的分派入口。

***

## 一、现有项目结构与约定（必读）

### 1.1 目录结构

```
gnn-predict/
├── main.py                              # 入口：--config, --output_dir, --gpu_node
├── config/arch/                         # YAML 配置文件（15 个）
│   ├── resnet50_variants.yaml
│   ├── bert_variants.yaml
│   └── ...
├── src/gnn_archs/                       # 变体生成 + 数据采集子系统
│   ├── config.py                        # Pydantic v2 配置类
│   ├── variant_runner.py                # 核心执行引擎
│   ├── mutations.py                     # image/text post-build 变异（不动）
│   ├── result.py                        # VariantResult 数据结构（不动）
│   ├── util/
│   │   ├── variant_expander.py          # 笛卡尔积展开（不动）
│   │   └── onnx_initializer.py          # ONNX metadata 工具（不动）
│   └── monitoring/                      # Prometheus 监控（不动）
└── src/gnn_model/                       # GNN 预测模型子系统（不动）
```

### 1.2 核心约定

1. **Pydantic v2 StrictModel**：所有配置类继承 `StrictModel`，使用 `ConfigDict(extra="forbid")`，多余字段直接报错
2. **YAML 声明式配置**：通过 `base_model_groups` → `combinatorial_variant_grid` → `mutation_sets` 的笛卡尔积展开变体
3. **二路分派**：`is_text_model_name()` 判断 text/image，现在扩展为三路（+detection）
4. **Architecture-only ONNX**：`export_params=False` 只导出计算图结构，不含权重
5. **Fake dataset 训练**：用随机数据训练，目的是产生 GPU 负载供 Prometheus 采集

### 1.3 数据流

```
main.py --config xxx.yaml
    → ArchConfig.model_validate(yaml)        # config.py
    → expand_arch_config(config)             # variant_expander.py（笛卡尔积展开）
    → list[ResolvedVariantSpec]
    → for spec in specs:
        run_variant(spec, context)           # variant_runner.py
            → build_variant_model(spec)      # 二路分派: timm / transformers
            → validate_model()
            → export_onnx_model()            # → onnx_models/{name}.onnx
            → train_model()                  # fake dataset
            → run_inference()                # warmup + timed
            → VariantResult                  # → results/result.json
```

### 1.4 变体展开公式

```
每个 base_model_group 的变体数 = |input_channels| × |output_classes| × |mutation_sets|
```

每个 YOLO scale（如 yolo11n、yolo11s）就是一个独立的 `base_model_group`，与 resnet18/resnet50 各是一个 group 同理。

***

## 二、YOLO 原生变体生成机制（必须理解）

ultralytics 的 YOLO 变体生成完全由以下原生机制处理：

### 2.1 YAML 中的 scales dict

以 `yolo11.yaml` 为例：

```yaml
scales:
  # [depth_multiple, width_multiple, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  ...

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  ...
  - [[16, 19, 22], 1, Detect, [nc]]
```

### 2.2 模型名解析（ultralytics 内部）

```python
# yaml_model_load("yolo11n.yaml") 内部:
# 1. guess_model_scale("yolo11n") → "n"（正则提取 scale 字母）
# 2. "yolo11n.yaml" → "yolo11.yaml"（去掉 scale 字母，加载基础 YAML）
# 3. d["scale"] = "n"

# parse_model(d, ch=3) 内部:
# depth, width, max_channels = scales["n"] = [0.50, 0.25, 1024]
# 每层: n = max(round(n * depth), 1)  # 深度缩放
# 每层: c2 = make_divisible(min(c2, max_channels) * width, 8)  # 宽度缩放
```

### 2.3 关键结论

**只要传入不同的 model name 字符串（如** **`"yolo11n"`、`"yolo11x"`），ultralytics 自动生成结构完全不同的计算图。** 我们不需要实现任何缩放逻辑，只需要调用 `YOLO("yolo11n.yaml").model`。

对于自定义 scales（非 n/s/m/l/x），需要加载 YAML dict，注入新的 scale 条目到 `scales` dict，设置 `d["scale"]`，然后写临时 YAML 交给 ultralytics 构建。

***

## 三、需要修改/新建的文件清单

| 文件                                  |   操作   |   改动量   |
| ----------------------------------- | :----: | :-----: |
| `src/gnn_archs/config.py`           | **修改** |  +15 行  |
| `src/gnn_archs/yolo_builder.py`     | **新建** | \~250 行 |
| `src/gnn_archs/variant_runner.py`   | **修改** |  +20 行  |
| `config/arch/yolo11_variants.yaml`  | **新建** |   配置文件  |
| `config/arch/yolov8_variants.yaml`  | **新建** |   配置文件  |
| `config/arch/yolov5_variants.yaml`  | **新建** |   配置文件  |
| `config/arch/yolov9_variants.yaml`  | **新建** |   配置文件  |
| `config/arch/yolov10_variants.yaml` | **新建** |   配置文件  |

**绝对不能修改的文件**：`mutations.py`、`variant_expander.py`、`result.py`、`monitoring/`下所有文件、`main.py`、`util/onnx_initializer.py`

***

## 四、实施步骤

### 步骤 1：修改 `src/gnn_archs/config.py`

在文件中找到 `TEXT_MODEL_PREFIXES` 和 `is_text_model_name()` 函数，在其**后面**追加以下代码。不要修改任何已有代码。

```python
# ============== 新增：Detection 模型识别 ==============

DETECTION_MODEL_PREFIXES = (
    "yolov3",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolo11",
    "yoloe",
)


def is_detection_model_name(model_name: str) -> bool:
    """判断是否为 YOLO 检测模型。"""
    normalized = normalize_model_identifier(model_name)
    return any(normalized.startswith(prefix) for prefix in DETECTION_MODEL_PREFIXES)
```

**验证**：确保 `is_detection_model_name("yolo11n")` 返回 `True`，`is_detection_model_name("resnet50")` 返回 `False`。

***

### 步骤 2：新建 `src/gnn_archs/yolo_builder.py`

这是**最核心的新文件**，封装全部 YOLO 相关逻辑。创建文件 `src/gnn_archs/yolo_builder.py`，内容如下：

```python
"""
YOLO 检测模型构建器 — 封装 ultralytics 原生变体生成机制。

核心思想：
    不重新实现任何缩放逻辑。
    利用 ultralytics 的 yaml_model_load() + parse_model() 原生处理
    depth/width/max_channels 缩放，生成结构不同的计算图。

    gnn_archs 只负责：
    1. 原生 scale 模型：直接 YOLO(name).model
    2. 自定义 scale 模型：加载 YAML dict → 注入 scale → 写临时 YAML → YOLO 构建
    3. YAML 级 pre-build mutation：在 dict 层面修改 backbone/head 定义
    4. ONNX 导出、训练、推理的适配

对外接口（供 variant_runner.py 调用）：
    - build_detection_model(spec) -> nn.Module
    - export_detection_onnx(spec, model, context) -> OnnxExportResult
    - train_detection_model(spec, model, device) -> TrainingResult
    - run_detection_inference(spec, model, device) -> InferenceResult
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from gnn_archs.result import InferenceResult, OnnxExportResult, TrainingResult, TimeWindow
from gnn_archs.util.onnx_initializer import (
    ONNX_EXPORT_MODE_METADATA_KEY,
    RUNTIME_INPUT_NAMES_METADATA_KEY,
    set_model_metadata_value,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec
    from gnn_archs.variant_runner import RunContext


# ============== YOLO YAML Mutation 类型注册表 ==============

YOLO_YAML_MUTATION_TYPES = frozenset({
    "BackboneModuleReplace",
    "HeadModuleReplace",
    "ActivationOverride",
    "SPPFKernelReplace",
    "BackboneDepthOverride",
    "ChannelScale",
})


# ============== 模型构建 ==============

def build_detection_model(spec: ResolvedVariantSpec) -> nn.Module:
    """
    构建 YOLO 检测模型。

    根据 spec.mutations 判断是否需要 YAML 级变异：
    - 无 mutation 且是标准 scale（n/s/m/l/x/t/b/c/e）：直接 YOLO(name.yaml).model
    - 有 mutation 或自定义 scale：加载 YAML dict → 应用变异 → 写临时 YAML → 构建
    """
    from ultralytics import YOLO

    model_name = spec.base_model.name  # e.g. "yolo11n", "yolov8x", "yolo11_pico"
    has_mutations = len(spec.mutations) > 0 and any(
        m.type in YOLO_YAML_MUTATION_TYPES for m in spec.mutations
    )

    # 判断是否为自定义 scale（名字里有下划线，如 yolo11_pico）
    is_custom_scale = "_" in model_name

    if not has_mutations and not is_custom_scale:
        # 标准路径：直接利用 ultralytics 原生 scale 解析
        yolo = YOLO(f"{model_name}.yaml", task="detect")
        return yolo.model

    # 高级路径：需要修改 YAML dict
    d = _load_yolo_yaml_dict(model_name)

    if is_custom_scale:
        d = _apply_custom_scale(d, model_name)

    if has_mutations:
        d = _apply_yaml_mutations(d, spec.mutations)

    return _build_model_from_dict(d, task="detect")


def _load_yolo_yaml_dict(model_name: str) -> dict[str, Any]:
    """
    加载 YOLO YAML dict。

    对于标准 model name（如 yolo11n），使用 ultralytics 原生加载。
    对于自定义 name（如 yolo11_pico），提取基础版本名加载。
    """
    from ultralytics.nn.tasks import yaml_model_load

    if "_" in model_name:
        # yolo11_pico → yolo11
        base_name = model_name.split("_")[0]
        # 去掉 base_name 末尾可能的 scale 字母，得到纯版本名
        clean_base = re.sub(r"[nslmxbtce]$", "", base_name)
        return yaml_model_load(f"{clean_base}.yaml")
    else:
        return yaml_model_load(f"{model_name}.yaml")


# ---- 自定义 Scale 注入 ----

# 预定义的自定义 scale 参数：[depth_multiple, width_multiple, max_channels]
CUSTOM_SCALES = {
    # YOLO11 插值 scales
    "pico":   [0.33, 0.15, 512],
    "micro":  [0.33, 0.20, 768],
    "ns_1":   [0.50, 0.31, 1024],
    "ns_2":   [0.50, 0.375, 1024],
    "sm_1":   [0.50, 0.625, 768],
    "sm_2":   [0.50, 0.75, 768],
    "sm_3":   [0.50, 0.875, 512],
    "ml_1":   [0.625, 1.00, 512],
    "ml_2":   [0.75, 1.00, 512],
    "lx_1":   [1.00, 1.125, 512],
    "lx_2":   [1.00, 1.25, 512],
    "huge":   [1.50, 2.00, 1024],
    "mega":   [2.00, 2.00, 1024],
    # 额外的极端值（增加多样性）
    "tiny_wide":   [0.33, 0.50, 512],
    "shallow_wide": [0.33, 1.00, 512],
    "deep_narrow":  [1.50, 0.25, 1024],
    "deep_mid":     [1.50, 0.50, 768],
    "ultra_deep":   [2.00, 0.50, 512],
}


def _apply_custom_scale(d: dict[str, Any], model_name: str) -> dict[str, Any]:
    """
    对自定义 scale 名称（如 yolo11_pico），注入 scale 参数到 YAML dict。

    原理：向 d["scales"] 注入自定义条目，设置 d["scale"] 为该条目的 key，
    然后 parse_model() 原生处理缩放。
    """
    d = copy.deepcopy(d)
    # yolo11_pico → scale_name = "pico"
    parts = model_name.split("_", 1)
    scale_name = parts[1] if len(parts) > 1 else "n"

    if scale_name not in CUSTOM_SCALES:
        raise ValueError(
            f"Unknown custom scale '{scale_name}'. "
            f"Available: {sorted(CUSTOM_SCALES.keys())}"
        )

    if "scales" not in d:
        d["scales"] = {}
    d["scales"][scale_name] = CUSTOM_SCALES[scale_name]
    d["scale"] = scale_name
    return d


# ---- YAML 级 Pre-build Mutations ----

def _apply_yaml_mutations(
    d: dict[str, Any], mutations: list[Any]
) -> dict[str, Any]:
    """
    在 YAML dict 层面执行 pre-build 变异。

    这些变异修改的是架构定义（backbone/head 列表），
    而非已构建的 nn.Module。变异后的 dict 仍是合法的 YOLO YAML dict，
    可直接传给 parse_model()。
    """
    d = copy.deepcopy(d)

    for mutation in mutations:
        mt = mutation.type
        params = mutation.params

        if mt not in YOLO_YAML_MUTATION_TYPES:
            # 跳过非 YOLO mutation（兼容性：不报错，只跳过）
            continue

        if mt == "BackboneModuleReplace":
            # 替换 backbone 中的模块类型
            # params: {"from_module": "C3k2", "to_module": "C2f"}
            # 可选 params: {"layers": [2, 4, 6]}（指定层索引，不填则替换全部匹配层）
            _replace_module_in_section(d, "backbone", params)

        elif mt == "HeadModuleReplace":
            # 替换 head 中的模块类型
            _replace_module_in_section(d, "head", params)

        elif mt == "ActivationOverride":
            # 全局替换激活函数
            # params: {"activation": "nn.LeakyReLU"}
            # ultralytics 的 parse_model() 会读取 d["activation"] 并执行
            # Conv.default_act = eval(activation)
            d["activation"] = params["activation"]

        elif mt == "SPPFKernelReplace":
            # 替换 SPPF 模块的 kernel size
            # params: {"kernel_size": 3}
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if len(layer) >= 3 and layer[2] == "SPPF":
                        if len(layer[3]) > 1:
                            layer[3][1] = params["kernel_size"]

        elif mt == "BackboneDepthOverride":
            # 覆盖 backbone 中所有 repeat>1 的层的 repeat 数
            # params: {"repeat_value": 1} 或 {"repeat_value": 4}
            repeat_val = params["repeat_value"]
            for layer in d.get("backbone", []):
                if layer[1] > 1:
                    layer[1] = repeat_val

        elif mt == "ChannelScale":
            # 缩放所有通道参数
            # params: {"scale_factor": 0.5}
            factor = params["scale_factor"]
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if len(layer) >= 4 and isinstance(layer[3], list) and len(layer[3]) > 0:
                        if isinstance(layer[3][0], (int, float)) and layer[3][0] > 1:
                            layer[3][0] = max(8, int(layer[3][0] * factor))

    return d


def _replace_module_in_section(
    d: dict[str, Any], section: str, params: dict[str, Any]
) -> None:
    """替换指定 section 中的模块类型。"""
    from_mod = params["from_module"]
    to_mod = params["to_module"]
    target_layers = params.get("layers")  # None = 替换所有匹配层

    for i, layer in enumerate(d.get(section, [])):
        if len(layer) >= 3 and layer[2] == from_mod:
            if target_layers is None or i in target_layers:
                layer[2] = to_mod


def _build_model_from_dict(d: dict[str, Any], task: str = "detect") -> nn.Module:
    """
    从修改后的 YAML dict 构建模型。

    写入临时 YAML 文件，使用 YOLO(tmp.yaml) 构建，
    ultralytics 的 parse_model() 原生处理所有缩放。
    """
    from ultralytics import YOLO

    # 清理非标准字段
    build_d = copy.deepcopy(d)
    build_d.pop("yaml_file", None)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="yolo_variant_"
    ) as f:
        yaml.dump(build_d, f, default_flow_style=False)
        tmp_path = f.name

    try:
        yolo = YOLO(tmp_path, task=task)
        return yolo.model
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ============== ONNX 导出 ==============

def export_detection_onnx(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
) -> OnnxExportResult:
    """
    YOLO 检测模型的 ONNX 导出。

    YOLO DetectionModel 的 forward() 在 eval 模式下返回预测 tensor，
    可直接 torch.onnx.export。使用 [1, 3, 640, 640] 标准检测输入。
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(
        *spec.variant_config.example_input_shape, device=device
    )
    export_path = context.output_layout.onnx_models_dir / f"{spec.name}.onnx"

    export_mode = spec.variant_config.onnx_export_mode
    input_names = ["images"]

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        input_names=input_names,
        output_names=["output"],
        opset_version=14,
        dynamo=False,
        export_params=(export_mode == "full"),
    )

    # 加载并设置 metadata（复用现有的 onnx_initializer 工具）
    onnx_model = onnx.load(str(export_path))
    set_model_metadata_value(
        onnx_model, RUNTIME_INPUT_NAMES_METADATA_KEY, json.dumps(input_names)
    )
    set_model_metadata_value(
        onnx_model, ONNX_EXPORT_MODE_METADATA_KEY, export_mode
    )
    onnx.save(onnx_model, str(export_path))
    onnx.checker.check_model(onnx_model)

    graph_input_names = [v.name for v in onnx_model.graph.input]
    parameter_input_names = [
        name for name in graph_input_names if name not in input_names
    ]
    initializer_names = [v.name for v in onnx_model.graph.initializer]

    graph_info = {
        "node_count": len(onnx_model.graph.node),
        "input_names": graph_input_names,
        "output_names": [v.name for v in onnx_model.graph.output],
        "op_types": sorted({node.op_type for node in onnx_model.graph.node}),
        "runtime_input_names": input_names,
        "parameter_input_names": parameter_input_names,
        "initializer_names": initializer_names,
        "initializer_count": len(initializer_names),
    }

    return OnnxExportResult(
        path=str(export_path),
        opset_version=14,
        file_size_bytes=export_path.stat().st_size,
        graph_info=graph_info,
    )


# ============== 训练 ==============

def train_detection_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> TrainingResult:
    """
    检测模型的 fake 训练。

    目的是产生真实的 GPU 计算负载，使 Prometheus 能采集到有意义的 DCGM 指标。
    使用 forward + sum reduction + backward 来模拟训练过程。
    """
    batch_size = spec.variant_config.training_batch_sizes[0]
    _, channels, height, width = spec.variant_config.example_input_shape
    dataset_size = spec.variant_config.fake_dataset_size
    generator = torch.Generator().manual_seed(42)

    images = torch.randn(
        (dataset_size, channels, height, width), generator=generator
    )
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    total_steps = 0
    last_loss = 0.0
    training_started_at = time.time()

    model.train()
    for _epoch in range(spec.variant_config.training_epochs):
        for (img_batch,) in dataloader:
            optimizer.zero_grad(set_to_none=True)
            img_batch = img_batch.to(device)
            outputs = model(img_batch)

            # 将检测模型的输出统一归约为标量 loss
            loss = _reduce_detection_output(outputs)
            loss.backward()
            optimizer.step()

            total_steps += 1
            last_loss = float(loss.detach().item())

    training_ended_at = time.time()

    return TrainingResult(
        hyperparameters={
            "batch_size": batch_size,
            "epochs": spec.variant_config.training_epochs,
            "dataset_size": dataset_size,
        },
        optimizer={"name": "AdamW", "lr": 1e-3},
        metrics={"final_loss": last_loss, "total_steps": total_steps},
        timings=_build_time_window(training_started_at, training_ended_at),
    )


# ============== 推理 ==============

def run_detection_inference(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> InferenceResult:
    """
    检测模型的推理计时。

    与现有 image 推理逻辑一致：warmup 2 次，然后持续推理直到达到最低测量时间。
    """
    generator = torch.Generator().manual_seed(42)
    dummy_input = torch.randn(
        *spec.variant_config.example_input_shape,
        generator=generator,
        device=device,
    )

    measurement_min_seconds = spec.variant_config.inference_measurement_min_seconds
    iterations = 0

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(2):
            _ = model(dummy_input)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        started_at = time.time()
        ended_at = started_at
        while ended_at - started_at < measurement_min_seconds:
            outputs = model(dummy_input)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            iterations += 1
            ended_at = time.time()

    total_duration = ended_at - started_at
    logits = _extract_detection_logits(outputs)

    return InferenceResult(
        metrics={
            "iterations": iterations,
            "batch_size": spec.variant_config.example_input_shape[0],
            "avg_latency_ms": round(total_duration / iterations * 1000, 4),
            "num_outputs": int(logits.shape[-1]) if logits is not None else 0,
        },
        timings=_build_time_window(started_at, ended_at),
    )


# ============== 内部工具函数 ==============

def _reduce_detection_output(outputs: Any) -> torch.Tensor:
    """将检测模型的多种输出格式统一为标量 loss。"""
    if isinstance(outputs, torch.Tensor):
        return outputs.sum()
    if isinstance(outputs, dict):
        # 部分 YOLO 模式返回 loss dict
        losses = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
        if losses:
            return sum(l.sum() for l in losses)
    if isinstance(outputs, (tuple, list)):
        tensors = [o for o in outputs if isinstance(o, torch.Tensor)]
        if tensors:
            return sum(t.sum() for t in tensors)
    if hasattr(outputs, "logits"):
        return outputs.logits.sum()
    raise TypeError(f"unexpected detection model output type: {type(outputs)}")


def _extract_detection_logits(outputs: Any) -> torch.Tensor | None:
    """从检测模型输出中提取用于统计的 tensor。"""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                return o
    if hasattr(outputs, "logits"):
        return outputs.logits
    return None


def _build_time_window(started_at: float, ended_at: float) -> TimeWindow:
    """构建 TimeWindow（与 variant_runner 中的格式一致）。"""
    from datetime import UTC, datetime

    return TimeWindow(
        started_at_ts=started_at,
        ended_at_ts=ended_at,
        started_at_text=datetime.fromtimestamp(started_at, tz=UTC).isoformat(),
        ended_at_text=datetime.fromtimestamp(ended_at, tz=UTC).isoformat(),
    )
```

***

### 步骤 3：修改 `src/gnn_archs/variant_runner.py`

在 variant\_runner.py 中做**最小化的分派改动**。以下是每个函数需要加的代码。

#### 3.1 顶部 import 区域追加

在文件顶部的 `from gnn_archs.config import is_text_model_name` 后面追加：

```python
from gnn_archs.config import is_detection_model_name
```

#### 3.2 `build_variant_model()` 函数

在函数开头、`if is_text_model_name(...)` 之前，插入：

```python
    # ---- detection (YOLO) 路径 ----
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import build_detection_model
        return build_detection_model(spec)
```

#### 3.3 `export_onnx_model()` 函数

在函数开头插入：

```python
    # ---- detection (YOLO) 路径 ----
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import export_detection_onnx
        return export_detection_onnx(spec, model, context)
```

#### 3.4 `train_model()` 函数

在函数开头插入：

```python
    # ---- detection (YOLO) 路径 ----
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import train_detection_model
        return train_detection_model(spec, model, device)
```

#### 3.5 `run_inference()` 函数

在函数开头插入：

```python
    # ---- detection (YOLO) 路径 ----
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import run_detection_inference
        return run_detection_inference(spec, model, device)
```

#### 3.6 `run_variant()` 函数

找到 `is_text_model = is_text_model_name(spec.base_model.name)` 这一行，在下面追加：

```python
    is_detection_model = is_detection_model_name(spec.base_model.name)
```

找到 metadata 中的 `"model_kind"` 行，修改为：

```python
    "model_kind": "detection" if is_detection_model else ("text" if is_text_model else "image"),
```

***

### 步骤 4：创建 YAML 配置文件

#### 4.1 `config/arch/yolo11_variants.yaml`

设计原则：

- 每个 YOLO scale（含自定义 scale）是一个独立的 `base_model_group`
- 用 YAML anchor `&` 和 alias `*` 避免配置重复
- `output_classes` 设置多个值来改变 Detect head 结构
- `mutation_sets` 包含 40 组 YAML 级变异

```yaml
# YOLO11 变体生成配置
# 使用 ultralytics 原生 scales 机制 + YAML 级 pre-build mutations
# 变体数 = 20 groups × 5 nc × 40 mutation_sets = 4,000

base_model_groups:

  # ===== 原生 5 个 scale =====
  - base_model: { name: "yolo11n", pretrained: false }
    combinatorial_variant_grid: &yolo11_grid
      input_channels: [3]
      output_classes: [10, 20, 50, 80, 200]
      base_variant_config_template: &yolo11_tpl
        example_input_shape: [1, 3, 640, 640]
        export_onnx: true
        onnx_export_mode: "architecture_only"
        run_training: true
        training_epochs: 3
        training_batch_sizes: [16]
        fake_dataset_size: 500
        use_fake_imagenet: true
        run_inference: true
        inference_measurement_min_seconds: 30.0
        pre_inference_cooldown_seconds: 5.0
      mutation_sets: &yolo11_mutations
        # ── 基线 ──
        - name: "baseline"
          mutations: []

        # ── backbone 模块替换（5 组）──
        - name: "bb_all_c2f"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
        - name: "bb_all_c3"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
        - name: "bb_front_c2f"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f", layers: [2, 4] }
        - name: "bb_rear_c3"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3", layers: [6, 8] }
        - name: "bb_mixed"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f", layers: [2, 4] }
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3", layers: [6, 8] }

        # ── head 模块替换（3 组）──
        - name: "hd_all_c2f"
          mutations:
            - type: "HeadModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
        - name: "hd_all_c3"
          mutations:
            - type: "HeadModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
        - name: "hd_mixed"
          mutations:
            - type: "HeadModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f", layers: [2, 5] }

        # ── 激活函数替换（6 组）──
        - name: "act_relu"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.ReLU" }
        - name: "act_leaky_relu"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.LeakyReLU" }
        - name: "act_gelu"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.GELU" }
        - name: "act_hardswish"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.Hardswish" }
        - name: "act_mish"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.Mish" }
        - name: "act_elu"
          mutations:
            - type: "ActivationOverride"
              params: { activation: "nn.ELU" }

        # ── SPPF kernel 替换（3 组）──
        - name: "sppf_k3"
          mutations:
            - type: "SPPFKernelReplace"
              params: { kernel_size: 3 }
        - name: "sppf_k7"
          mutations:
            - type: "SPPFKernelReplace"
              params: { kernel_size: 7 }
        - name: "sppf_k9"
          mutations:
            - type: "SPPFKernelReplace"
              params: { kernel_size: 9 }

        # ── 深度覆盖（5 组）──
        - name: "depth_min"
          mutations:
            - type: "BackboneDepthOverride"
              params: { repeat_value: 1 }
        - name: "depth_3"
          mutations:
            - type: "BackboneDepthOverride"
              params: { repeat_value: 3 }
        - name: "depth_4"
          mutations:
            - type: "BackboneDepthOverride"
              params: { repeat_value: 4 }
        - name: "depth_6"
          mutations:
            - type: "BackboneDepthOverride"
              params: { repeat_value: 6 }
        - name: "depth_8"
          mutations:
            - type: "BackboneDepthOverride"
              params: { repeat_value: 8 }

        # ── 通道缩放（5 组）──
        - name: "ch_x0.5"
          mutations:
            - type: "ChannelScale"
              params: { scale_factor: 0.5 }
        - name: "ch_x0.75"
          mutations:
            - type: "ChannelScale"
              params: { scale_factor: 0.75 }
        - name: "ch_x1.25"
          mutations:
            - type: "ChannelScale"
              params: { scale_factor: 1.25 }
        - name: "ch_x1.5"
          mutations:
            - type: "ChannelScale"
              params: { scale_factor: 1.5 }
        - name: "ch_x2.0"
          mutations:
            - type: "ChannelScale"
              params: { scale_factor: 2.0 }

        # ── 二元组合（6 组）──
        - name: "c2f_relu"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.ReLU" }
        - name: "c2f_gelu"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.GELU" }
        - name: "c3_leaky"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
            - type: "ActivationOverride"
              params: { activation: "nn.LeakyReLU" }
        - name: "c2f_sppf_k3"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "SPPFKernelReplace"
              params: { kernel_size: 3 }
        - name: "c2f_depth_4"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "BackboneDepthOverride"
              params: { repeat_value: 4 }
        - name: "c3_ch_x0.75"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
            - type: "ChannelScale"
              params: { scale_factor: 0.75 }

        # ── 三元组合（6 组）──
        - name: "c2f_gelu_sppf_k3"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.GELU" }
            - type: "SPPFKernelReplace"
              params: { kernel_size: 3 }
        - name: "c3_mish_depth_4"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
            - type: "ActivationOverride"
              params: { activation: "nn.Mish" }
            - type: "BackboneDepthOverride"
              params: { repeat_value: 4 }
        - name: "c2f_relu_ch_x0.5"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.ReLU" }
            - type: "ChannelScale"
              params: { scale_factor: 0.5 }
        - name: "c3_hardswish_sppf_k7"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C3" }
            - type: "ActivationOverride"
              params: { activation: "nn.Hardswish" }
            - type: "SPPFKernelReplace"
              params: { kernel_size: 7 }
        - name: "c2f_leaky_depth_6"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.LeakyReLU" }
            - type: "BackboneDepthOverride"
              params: { repeat_value: 6 }
        - name: "c2f_elu_ch_x1.5"
          mutations:
            - type: "BackboneModuleReplace"
              params: { from_module: "C3k2", to_module: "C2f" }
            - type: "ActivationOverride"
              params: { activation: "nn.ELU" }
            - type: "ChannelScale"
              params: { scale_factor: 1.5 }

  - base_model: { name: "yolo11s", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11m", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11l", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11x", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  # ===== 自定义 15 个插值 scale =====
  - base_model: { name: "yolo11_pico", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_micro", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_ns_1", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_ns_2", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_sm_1", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_sm_2", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_sm_3", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_ml_1", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_ml_2", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_lx_1", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_lx_2", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_huge", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_mega", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_tiny_wide", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_shallow_wide", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_deep_narrow", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_deep_mid", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid

  - base_model: { name: "yolo11_ultra_deep", pretrained: false }
    combinatorial_variant_grid: *yolo11_grid
```

**变体数 = 23 groups × 5 nc × 40 mutation\_sets = 4,600**

#### 4.2 `config/arch/yolov8_variants.yaml`

结构与 yolo11 完全相同，只是：

- `base_model.name` 改为 `yolov8n`/`yolov8s`/`yolov8m`/`yolov8l`/`yolov8x` + 自定义 scales
- mutation\_sets 中 `from_module` 改为 `"C2f"`（YOLOv8 backbone 用 C2f 而非 C3k2），`to_module` 改为 `"C3"`/`"C3k2"` 等
- 自定义 scales 可以比 YOLO11 少一些（8 个），因为 YOLOv8 架构相对简单

**变体数 = 13 groups × 5 nc × 40 mutation\_sets = 2,600**

#### 4.3 `config/arch/yolov5_variants.yaml`

- `base_model.name`: `yolov5n`/`yolov5s`/`yolov5m`/`yolov5l`/`yolov5x` + 5 个自定义 scales
- mutation\_sets 中 `from_module` = `"C3"`（YOLOv5 backbone 用 C3）
- output\_classes: \[10, 20, 50, 80]（4 个值）
- mutation\_sets 可以缩减到 30 组（去掉部分三元组合）

**变体数 = 10 groups × 4 nc × 30 mutation\_sets = 1,200**

#### 4.4 `config/arch/yolov9_variants.yaml`

- `base_model.name`: `yolov9t`/`yolov9s`/`yolov9m`/`yolov9c`/`yolov9e` + 5 个自定义 scales
- mutation\_sets 中 `from_module` = `"RepNCSPELAN4"`（YOLOv9 特有模块）
- output\_classes: \[10, 50, 80]（3 个值）
- mutation\_sets: 20 组

**变体数 = 10 groups × 3 nc × 20 mutation\_sets = 600**

#### 4.5 `config/arch/yolov10_variants.yaml`

- `base_model.name`: `yolov10n`/`yolov10s`/`yolov10m`/`yolov10b`/`yolov10l`/`yolov10x` + 4 个自定义 scales
- mutation\_sets 中 `from_module` = `"C2fCIB"` / `"SCDown"`（YOLOv10 特有模块）
- output\_classes: \[10, 50, 80]（3 个值）
- mutation\_sets: 15 组

**变体数 = 10 groups × 3 nc × 15 mutation\_sets = 450**

***

## 五、变体总量汇总

| YAML 配置文件              | Base Model Groups |  × nc  | × mutation\_sets |    变体数    |
| ---------------------- | :---------------: | :----: | :--------------: | :-------: |
| yolo11\_variants.yaml  |         23        |    5   |        40        | **4,600** |
| yolov8\_variants.yaml  |         13        |    5   |        40        | **2,600** |
| yolov5\_variants.yaml  |         10        |    4   |        30        | **1,200** |
| yolov9\_variants.yaml  |         10        |    3   |        20        |  **600**  |
| yolov10\_variants.yaml |         10        |    3   |        15        |  **450**  |
| **Detection 总计**       |       **66**      | <br /> |      <br />      | **9,450** |

加上现有：

| 模型家族                 |     变体数    |
| -------------------- | :--------: |
| Image (timm)         |    4,372   |
| Text (transformers)  |     498    |
| **Detection (YOLO)** |  **9,450** |
| **全部总计**             | **14,320** |

***

## 六、输出数据存储结构

```
{output_dir}/
├── yolo11/                              # 从 yolo11_variants.yaml 推导
│   ├── logs/
│   ├── results/
│   │   └── result.json                  # 4,600 个 VariantResult
│   └── onnx_models/
│       ├── yolo11n_baseline.onnx
│       ├── yolo11n_bb_all_c2f.onnx
│       ├── yolo11n_act_relu.onnx
│       ├── yolo11_pico_baseline.onnx
│       └── ...                          # 4,600 个 ONNX 文件
├── yolov8/
│   ├── results/result.json              # 2,600 个
│   └── onnx_models/                     # 2,600 个
├── yolov5/
├── yolov9/
└── yolov10/
```

每个 ONNX 文件（architecture\_only 模式）约 10-100KB，总存储约 \~700MB。

***

## 七、验证清单

实施完成后，按以下步骤验证：

### 7.1 单元验证

```bash
cd /Users/bytedance/study/paper/gnn-pre/gnn-predict

# 1. 验证 config.py 的类型判断
python -c "
from gnn_archs.config import is_detection_model_name, is_text_model_name
assert is_detection_model_name('yolo11n') == True
assert is_detection_model_name('yolov8x') == True
assert is_detection_model_name('yolo11_pico') == True
assert is_detection_model_name('resnet50') == False
assert is_text_model_name('bert-base-uncased') == True
print('config.py OK')
"

# 2. 验证 yolo_builder.py 的模型构建
python -c "
from gnn_archs.yolo_builder import build_detection_model, CUSTOM_SCALES
# 测试需要 mock spec，或直接测试内部函数
from gnn_archs.yolo_builder import _load_yolo_yaml_dict, _apply_custom_scale
d = _load_yolo_yaml_dict('yolo11n')
print(f'Loaded YAML with {len(d.get(\"backbone\", []))} backbone layers')
d2 = _apply_custom_scale(_load_yolo_yaml_dict('yolo11_pico'), 'yolo11_pico')
print(f'Custom scale applied: {d2[\"scale\"]} = {d2[\"scales\"][\"pico\"]}')
print('yolo_builder.py OK')
"

# 3. 验证 YAML 配置加载
python -c "
from gnn_archs.config import ArchConfig
import yaml
with open('config/arch/yolo11_variants.yaml') as f:
    raw = yaml.safe_load(f)
config = ArchConfig.model_validate(raw)
print(f'Groups: {len(config.base_model_groups)}')
for g in config.base_model_groups[:3]:
    grid = g.combinatorial_variant_grid
    n = len(grid.input_channels) * len(grid.output_classes) * len(grid.mutation_sets)
    print(f'  {g.base_model.name}: {n} variants')
print('YAML config OK')
"

# 4. 验证变体展开（不需要 GPU）
python -c "
from gnn_archs.config import ArchConfig
from gnn_archs.util.variant_expander import expand_arch_config
import yaml
with open('config/arch/yolo11_variants.yaml') as f:
    raw = yaml.safe_load(f)
config = ArchConfig.model_validate(raw)
specs = expand_arch_config(config)
print(f'Total variants expanded: {len(specs)}')
print(f'First 5: {[s.name for s in specs[:5]]}')
print('variant_expander OK')
"
```

### 7.2 端到端验证（需要 GPU）

```bash
# 用小配置做端到端测试
python main.py \
    --config config/arch/yolo11_variants.yaml \
    --output_dir /tmp/yolo_test \
    --gpu_node test-node

# 验证输出
ls /tmp/yolo_test/yolo11/onnx_models/ | wc -l    # 应为 4,600
python -c "
import json
with open('/tmp/yolo_test/yolo11/results/result.json') as f:
    data = json.load(f)
print(f'Results: {len(data[\"variant_results\"])} variants')
print(f'First result: {data[\"variant_results\"][0][\"name\"]}')
print(f'Model kind: {data[\"variant_results\"][0][\"metadata\"][\"model_kind\"]}')
"
```

### 7.3 回归验证（确保不影响已有模型）

```bash
# 运行一个已有的 image 配置，确认输出不变
python main.py \
    --config config/arch/resnet50_variants.yaml \
    --output_dir /tmp/regression_test \
    --gpu_node test-node
```

***

## 八、注意事项

1. **依赖安装**：需要在环境中安装 `ultralytics` 包，使用uv创建虚拟环境
   ```bash
   pip install ultralytics
   ```
2. **YAML anchor 兼容性**：Pydantic `model_validate()` 接收的是 Python dict（由 PyYAML 解析后），YAML anchor 在 `yaml.safe_load()` 阶段就已经展开为独立对象，不会影响 Pydantic 校验。
3. **自定义 scale 命名约定**：使用 `{版本}_{scale名}` 格式（如 `yolo11_pico`），`yolo_builder.py` 中的 `_apply_custom_scale()` 通过下划线分割提取 scale 名。
4. **YOLO mutation 与 image/text mutation 的隔离**：YOLO mutation 的 `type` 值（如 `BackboneModuleReplace`）与 `IMAGE_MUTATION_TYPES`/`TEXT_MUTATION_TYPES` 完全不重叠。`yolo_builder.py` 内部通过 `YOLO_YAML_MUTATION_TYPES` 集合过滤，不匹配的 mutation 会被跳过而非报错。
5. **ONNX 导出注意**：部分 YOLO 模型在 `torch.onnx.export` 时可能需要设置 `model.model[-1].export = True`（Detect head 导出模式），如果遇到导出错误可以在 `export_detection_onnx` 中加上：
   ```python
   # 在 torch.onnx.export 之前
   for m in model.modules():
       if hasattr(m, 'export'):
           m.export = True
   ```
6. **内存管理**：大 scale 模型（如 yolo11\_mega, yolo11\_huge）可能需要较多 GPU 显存。如果 OOM，考虑减小 `training_batch_sizes` 或在配置中将大 scale 的 `run_training` 设为 `false`。

