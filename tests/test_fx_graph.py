from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn
from torch.export import ExportedProgram
from gnn_model_test_utils import TinyConvNet

import common.graph_artifact as graph_artifact_module
from common.graph_artifact import (
    GRAPH_ARTIFACT_METADATA_FILE,
    GRAPH_ARTIFACT_SCHEMA_VERSION,
    _exported_program_verifier_kwargs,
    capture_inference_graph,
    clear_graph_capture_caches,
    load_graph_artifact,
    replace_state_with_zero_storage,
    save_graph_artifact,
)
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    GRAPH_FEATURE_NAMES,
    NODE_FEATURE_DIM,
    OP_TYPE_TO_INDEX,
)
from gnn_model.data.fx_graph import (
    build_graph_data_from_exported_program,
    build_graph_data_from_fx,
)


def test_clear_graph_capture_caches_resets_compiler_and_fake_tensor_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        graph_artifact_module.torch.compiler,
        "reset",
        lambda: calls.append("compiler"),
    )
    monkeypatch.setattr(
        graph_artifact_module.FakeTensorMode,
        "cache_clear",
        lambda: calls.append("fake_tensor"),
    )

    clear_graph_capture_caches()

    assert calls == ["compiler", "fake_tensor"]


def test_exported_program_verifier_kwargs_supports_old_and_new_torch() -> None:
    old_verifier = object()
    new_verifiers = [object()]
    old_program = cast(ExportedProgram, SimpleNamespace(verifier=old_verifier))
    new_program = cast(ExportedProgram, SimpleNamespace(verifiers=new_verifiers))

    assert _exported_program_verifier_kwargs(old_program) == {
        "verifier": old_verifier
    }
    assert _exported_program_verifier_kwargs(new_program) == {
        "verifiers": new_verifiers
    }


def test_graph_artifact_preserves_state_shape_without_weight_storage(
    tmp_path: Path,
) -> None:
    model = TinyConvNet(width=4, output_dim=3).eval()
    exported_program = capture_inference_graph(
        model,
        (torch.randn(1, 3, 32, 32),),
    )
    graph_path = tmp_path / "tiny.pt2"

    metadata = save_graph_artifact(
        exported_program,
        graph_path,
        runtime_input_names=["inputs"],
    )
    loaded, loaded_metadata = load_graph_artifact(graph_path)

    assert metadata == loaded_metadata
    assert loaded_metadata.runtime_input_names == ["inputs"]
    assert graph_path.stat().st_size < 100_000
    assert all(
        value.untyped_storage().nbytes() <= value.element_size()
        for value in loaded.state_dict.values()
    )
    assert tuple(loaded.state_dict["features.0.weight"].shape) == (4, 3, 3, 3)


def test_build_graph_data_matches_static_feature_contract(tmp_path: Path) -> None:
    model = TinyConvNet(width=4, output_dim=3).eval()
    exported_program = capture_inference_graph(
        model,
        (torch.randn(1, 3, 32, 32),),
    )
    graph_path = tmp_path / "tiny.pt2"
    save_graph_artifact(
        exported_program,
        graph_path,
        runtime_input_names=["inputs"],
    )

    data = build_graph_data_from_fx(graph_path)
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    assert isinstance(x, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)
    assert isinstance(edge_attr, torch.Tensor)

    assert x.shape == (5, NODE_FEATURE_DIM)
    assert edge_index.shape == (2, 4)
    assert edge_attr.shape == (4, EDGE_FEATURE_DIM)
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert x[:, 0].tolist() == [114688.0, 4096.0, 4112.0, 0.0, 15.0]
    assert x[:, 1].tolist() == [16384.0, 16384.0, 16.0, 16.0, 12.0]
    graph_values = dict(
        zip(GRAPH_FEATURE_NAMES, data.graph_features[0].tolist(), strict=True)
    )
    assert graph_values["parameter_input_count"] == 4.0
    assert graph_values["parameter_input_element_count"] == 127.0
    assert graph_values["parameter_input_bytes"] == 508.0
    assert graph_values["graph_macs"] == 122911.0
    assert graph_values["graph_memory_bytes"] == 32812.0
    assert graph_values["graph_params"] == 0.0
    assert data.graph_path == str(graph_path)


class BFloat16Output(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(torch.bfloat16)


class BatchNormGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs)


def test_build_graph_data_handles_bfloat16_tensor_metadata() -> None:
    exported_program = capture_inference_graph(
        BFloat16Output(),
        (torch.randn(2),),
    )

    data = build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=["inputs"],
        batch_size=2,
    )
    x = data.x
    assert isinstance(x, torch.Tensor)

    assert x.shape == (1, NODE_FEATURE_DIM)
    assert x[0, -1].item() == 2.0


def test_in_memory_and_saved_graph_features_are_identical(tmp_path: Path) -> None:
    exported_program = capture_inference_graph(
        BatchNormGraph(),
        (torch.randn(1, 4, 8, 8),),
    )
    graph_path = tmp_path / "batch-norm.pt2"
    in_memory = build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=["inputs"],
    )
    save_graph_artifact(
        exported_program,
        graph_path,
        runtime_input_names=["inputs"],
    )
    saved = build_graph_data_from_fx(graph_path)
    assert isinstance(in_memory.x, torch.Tensor)
    assert isinstance(saved.x, torch.Tensor)
    assert isinstance(in_memory.edge_index, torch.Tensor)
    assert isinstance(saved.edge_index, torch.Tensor)
    assert isinstance(in_memory.edge_attr, torch.Tensor)
    assert isinstance(saved.edge_attr, torch.Tensor)

    assert torch.equal(in_memory.x, saved.x)
    assert torch.equal(in_memory.edge_index, saved.edge_index)
    assert torch.equal(in_memory.edge_attr, saved.edge_attr)
    assert torch.equal(in_memory.graph_features, saved.graph_features)
    assert torch.equal(in_memory.op_type_ids, saved.op_type_ids)


class UnsupportedTensorOperator(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(inputs, dim=-1)


def test_build_graph_data_rejects_unknown_tensor_operator() -> None:
    exported_program = capture_inference_graph(
        UnsupportedTensorOperator(),
        (torch.randn(2, 3),),
    )

    with pytest.raises(ValueError, match=r"aten\.linalg_vector_norm\.default"):
        build_graph_data_from_exported_program(
            exported_program,
            runtime_input_names=["inputs"],
        )


class TensorConstantGraph(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + torch.tensor([1.0, 2.0])


def test_build_graph_data_classifies_lifted_tensor_constants() -> None:
    exported_program = capture_inference_graph(
        TensorConstantGraph(),
        (torch.randn(2),),
    )

    data = build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=["inputs"],
        batch_size=2,
    )

    assert OP_TYPE_TO_INDEX["op_constant"] in data.op_type_ids.tolist()


class RollGraph(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.roll(inputs, shifts=1, dims=1)


def test_build_graph_data_classifies_roll_as_layout() -> None:
    exported_program = capture_inference_graph(
        RollGraph(),
        (torch.randn(2, 3),),
    )

    data = build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=["inputs"],
        batch_size=2,
    )

    assert data.op_type_ids.tolist() == [OP_TYPE_TO_INDEX["op_layout"]]


def test_graph_artifact_rejects_pytorch_version_mismatch(tmp_path: Path) -> None:
    exported_program = capture_inference_graph(
        BFloat16Output(),
        (torch.randn(2),),
    )
    graph_path = tmp_path / "wrong-version.pt2"
    metadata = {
        "format": "pt2",
        "schema_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
        "torch_version": "1.0.0",
        "capture_mode": "static_inference",
        "weights": "zero_stride_proxy",
        "runtime_input_names": ["inputs"],
    }
    torch.export.save(
        exported_program,
        graph_path,
        extra_files={GRAPH_ARTIFACT_METADATA_FILE: json.dumps(metadata)},
    )

    with pytest.raises(ValueError, match="PyTorch version mismatch"):
        load_graph_artifact(graph_path)


def test_graph_artifact_rejects_runtime_input_name_mismatch(tmp_path: Path) -> None:
    exported_program = capture_inference_graph(
        BFloat16Output(),
        (torch.randn(2),),
    )

    with pytest.raises(ValueError, match="runtime input name count"):
        save_graph_artifact(
            exported_program,
            tmp_path / "invalid.pt2",
            runtime_input_names=[],
        )


def test_graph_artifact_rejects_incomplete_metadata(tmp_path: Path) -> None:
    exported_program = capture_inference_graph(
        BFloat16Output(),
        (torch.randn(2),),
    )
    graph_path = tmp_path / "incomplete.pt2"
    torch.export.save(
        exported_program,
        graph_path,
        extra_files={
            GRAPH_ARTIFACT_METADATA_FILE: json.dumps(
                {
                    "torch_version": torch.__version__,
                    "runtime_input_names": ["inputs"],
                }
            )
        },
    )

    with pytest.raises(ValueError, match="validation errors"):
        load_graph_artifact(graph_path)


def test_build_graph_data_rejects_symbolic_shapes() -> None:
    exported_program = torch.export.export(
        BFloat16Output(),
        (torch.randn(2),),
        dynamic_shapes={"inputs": {0: torch.export.Dim("batch")}},
        strict=False,
    ).run_decompositions(decomp_table={})
    exported_program = replace_state_with_zero_storage(exported_program)

    with pytest.raises(ValueError, match="symbolic dimension"):
        build_graph_data_from_exported_program(
            exported_program,
            runtime_input_names=["inputs"],
        )


def test_build_graph_data_rejects_batch_size_mismatch() -> None:
    exported_program = capture_inference_graph(
        BFloat16Output(),
        (torch.randn(2),),
    )

    with pytest.raises(ValueError, match="batch_size does not match"):
        build_graph_data_from_exported_program(
            exported_program,
            runtime_input_names=["inputs"],
            batch_size=1,
        )
