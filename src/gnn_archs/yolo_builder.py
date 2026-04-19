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

from gnn_archs.result import InferenceResult, OnnxExportResult, TimeWindow, TrainingResult
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
            _replace_module_in_section(d, "backbone", params)

        elif mt == "HeadModuleReplace":
            _replace_module_in_section(d, "head", params)

        elif mt == "ActivationOverride":
            d["activation"] = params["activation"]

        elif mt == "SPPFKernelReplace":
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if len(layer) >= 3 and layer[2] == "SPPF":
                        if len(layer[3]) > 1:
                            layer[3][1] = params["kernel_size"]

        elif mt == "BackboneDepthOverride":
            repeat_val = params["repeat_value"]
            for layer in d.get("backbone", []):
                if layer[1] > 1:
                    layer[1] = repeat_val

        elif mt == "ChannelScale":
            factor = params["scale_factor"]
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if (
                        len(layer) >= 4
                        and isinstance(layer[3], list)
                        and len(layer[3]) > 0
                        and isinstance(layer[3][0], (int, float))
                        and layer[3][0] > 1
                    ):
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

    注意：必须在 export 前设置所有含 export 属性的模块（如 Detect head）
    的 export = True，否则部分 YOLO 模型会包含不支持的算子导致导出失败。
    """
    model.eval()

    # 设置 Detect head 导出模式，避免不支持的算子导致 export 失败
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True

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
            return sum(loss.sum() for loss in losses)
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
