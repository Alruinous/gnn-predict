from __future__ import annotations

import json
import logging
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

GRAPH_ARTIFACT_METADATA_FILE = "gnn_predict_graph.json"
GRAPH_ARTIFACT_SCHEMA_VERSION = "1.0.0"
_PT2_ARCHIVE_LOGGER_NAME = "torch.export.pt2_archive._package"
_PT2_EMPTY_TENSOR_WARNING = "Cannot call torch.frombuffer() on empty bytes."


class _Pt2ArchiveLoadFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_PT2_EMPTY_TENSOR_WARNING)


class GraphArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["pt2"]
    schema_version: str
    torch_version: str
    capture_mode: Literal["static_inference"]
    weights: Literal["zero_stride_proxy"]
    runtime_input_names: list[str]


def capture_inference_graph(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> ExportedProgram:
    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, args, kwargs or {}, strict=False)
        inference_graph = exported.run_decompositions(decomp_table={})
    architecture_graph = replace_state_with_zero_storage(inference_graph)
    canonical_program = canonicalize_exported_program(architecture_graph)
    del architecture_graph, inference_graph, exported
    return canonical_program


def canonicalize_exported_program(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    with BytesIO() as buffer:
        torch.export.save(exported_program, buffer)
        buffer.seek(0)
        canonical_program = _load_exported_program(buffer)
    validate_zero_storage_state(canonical_program)
    return canonical_program


def clear_graph_capture_caches() -> None:
    torch.compiler.reset()
    FakeTensorMode.cache_clear()


def _load_exported_program(
    source: BytesIO | Path,
    *,
    extra_files: dict[str, str] | None = None,
) -> ExportedProgram:
    archive_logger = logging.getLogger(_PT2_ARCHIVE_LOGGER_NAME)
    log_filter = _Pt2ArchiveLoadFilter()
    archive_logger.addFilter(log_filter)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        try:
            return torch.export.load(source, extra_files=extra_files)
        finally:
            archive_logger.removeFilter(log_filter)


def replace_state_with_zero_storage(
    exported_program: ExportedProgram,
) -> ExportedProgram:
    state_dict = {
        name: build_zero_storage_tensor(value)
        for name, value in exported_program.state_dict.items()
    }
    constants = {
        name: build_zero_storage_tensor(value)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in exported_program.constants.items()
    }
    return ExportedProgram(
        exported_program.graph_module,
        exported_program.graph,
        exported_program.graph_signature,
        state_dict,
        exported_program.range_constraints,
        exported_program.module_call_graph,
        example_inputs=None,
        constants=constants,
        **_exported_program_verifier_kwargs(exported_program),
    )


def _exported_program_verifier_kwargs(exported_program: ExportedProgram) -> dict[str, Any]:
    # PyTorch 2.4 exposes one `verifier`; PyTorch 2.9 exposes `verifiers`.
    if hasattr(exported_program, "verifiers"):
        return {"verifiers": exported_program.verifiers}
    if hasattr(exported_program, "verifier"):
        return {"verifier": exported_program.verifier}
    raise ValueError("ExportedProgram has no supported verifier attribute")


def build_zero_storage_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.layout is not torch.strided:
        raise ValueError(f"graph state tensor must use strided layout: {value.layout}")
    base = torch.zeros((), dtype=value.dtype, device="cpu")
    proxy = torch.as_strided(
        base,
        size=tuple(value.shape),
        stride=(0,) * value.dim(),
    )
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(proxy, requires_grad=value.requires_grad)
    if value.requires_grad and (value.is_floating_point() or value.is_complex()):
        proxy.requires_grad_(True)
    return proxy


def save_graph_artifact(
    exported_program: ExportedProgram,
    path: str | Path,
    *,
    runtime_input_names: list[str],
) -> GraphArtifactMetadata:
    validate_runtime_input_names(exported_program, runtime_input_names)
    validate_zero_storage_state(exported_program)
    metadata = GraphArtifactMetadata(
        format="pt2",
        schema_version=GRAPH_ARTIFACT_SCHEMA_VERSION,
        torch_version=torch.__version__,
        capture_mode="static_inference",
        weights="zero_stride_proxy",
        runtime_input_names=runtime_input_names,
    )
    artifact_path = Path(path)
    validate_graph_artifact_path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(
        exported_program,
        artifact_path,
        extra_files={
            GRAPH_ARTIFACT_METADATA_FILE: metadata.model_dump_json(),
        },
    )
    return metadata


def load_graph_artifact(
    path: str | Path,
) -> tuple[ExportedProgram, GraphArtifactMetadata]:
    validate_graph_artifact_path(Path(path))
    extra_files = {GRAPH_ARTIFACT_METADATA_FILE: ""}
    exported_program = _load_exported_program(Path(path), extra_files=extra_files)
    raw_metadata = extra_files[GRAPH_ARTIFACT_METADATA_FILE]
    if not raw_metadata:
        raise ValueError(f"graph artifact metadata is missing: {path}")
    metadata = GraphArtifactMetadata.model_validate(json.loads(raw_metadata))
    validate_graph_artifact_metadata(metadata)
    validate_runtime_input_names(exported_program, metadata.runtime_input_names)
    validate_zero_storage_state(exported_program)
    return exported_program, metadata


def validate_graph_artifact_metadata(metadata: GraphArtifactMetadata) -> None:
    if metadata.schema_version != GRAPH_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported graph artifact schema: "
            f"{metadata.schema_version} != {GRAPH_ARTIFACT_SCHEMA_VERSION}"
        )
    if torch_major_minor(metadata.torch_version) != torch_major_minor(
        torch.__version__
    ):
        raise ValueError(
            "graph artifact PyTorch version mismatch: "
            f"{metadata.torch_version} != {torch.__version__}"
        )


def validate_runtime_input_names(
    exported_program: ExportedProgram,
    runtime_input_names: list[str],
) -> None:
    user_input_count = sum(
        spec.kind == InputKind.USER_INPUT
        for spec in exported_program.graph_signature.input_specs
    )
    if len(runtime_input_names) != user_input_count:
        raise ValueError(
            "runtime input name count does not match exported graph inputs: "
            f"{len(runtime_input_names)} != {user_input_count}"
        )
    if len(runtime_input_names) != len(set(runtime_input_names)):
        raise ValueError("runtime input names must be unique")
    if any(not name.strip() for name in runtime_input_names):
        raise ValueError("runtime input names must not be empty")


def validate_zero_storage_state(exported_program: ExportedProgram) -> None:
    tensors = [
        *exported_program.state_dict.values(),
        *(
            value
            for value in exported_program.constants.values()
            if isinstance(value, torch.Tensor)
        ),
    ]
    for value in tensors:
        if value.device.type != "cpu":
            raise ValueError(f"graph state proxy must be on CPU: {value.device}")
        if any(stride != 0 for stride in value.stride()):
            raise ValueError("graph state proxy strides must all be zero")
        if value.untyped_storage().nbytes() > value.element_size():
            raise ValueError(
                "graph artifact contains tensor storage larger than one element"
            )


def validate_graph_artifact_path(path: Path) -> None:
    if path.suffix != ".pt2":
        raise ValueError(f"graph artifact path must use .pt2: {path}")


def build_graph_info(
    exported_program: ExportedProgram,
    *,
    runtime_input_names: list[str],
) -> dict[str, int | float | str | list[str]]:
    signature = exported_program.graph_signature
    input_names = [
        str(getattr(spec.arg, "name", spec.arg)) for spec in signature.input_specs
    ]
    parameter_names = [
        str(spec.target)
        for spec in signature.input_specs
        if spec.kind == InputKind.PARAMETER
    ]
    buffer_names = [
        str(spec.target)
        for spec in signature.input_specs
        if spec.kind == InputKind.BUFFER
    ]
    constant_names = [
        str(spec.target)
        for spec in signature.input_specs
        if spec.kind == InputKind.CONSTANT_TENSOR
    ]
    output_names = [
        str(getattr(spec.arg, "name", spec.arg)) for spec in signature.output_specs
    ]
    nodes = [
        node for node in exported_program.graph.nodes if node.op == "call_function"
    ]
    return {
        "node_count": len(nodes),
        "input_names": input_names,
        "output_names": output_names,
        "op_types": sorted({str(node.target) for node in nodes}),
        "runtime_input_names": runtime_input_names,
        "parameter_input_names": parameter_names,
        "buffer_input_names": buffer_names,
        "constant_input_names": constant_names,
        "parameter_input_count": len(parameter_names) + len(buffer_names),
    }


def torch_major_minor(version: str) -> tuple[int, int]:
    release = version.split("+", maxsplit=1)[0]
    parts = release.split(".")
    if len(parts) < 2:
        raise ValueError(f"invalid PyTorch version: {version}")
    return int(parts[0]), int(parts[1])
