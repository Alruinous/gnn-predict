from __future__ import annotations

import pytest
import torch

from gnn_archs.device_runtime import resolve_runtime


def test_resolve_runtime_recognizes_corex_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _device: "Iluvatar BI-V150"
    )

    runtime = resolve_runtime()

    assert runtime.device == torch.device("cuda:0")
    assert runtime.backend == "corex"
    assert runtime.device_name == "Iluvatar BI-V150"


def test_resolve_runtime_keeps_nvidia_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA A100")

    assert resolve_runtime().backend == "nvidia"
    with pytest.raises(RuntimeError, match="corex backend requested"):
        resolve_runtime("corex")


def test_resolve_runtime_rejects_missing_requested_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_runtime().backend == "cpu"
    with pytest.raises(RuntimeError, match="no GPU is available"):
        resolve_runtime("corex")


def test_resolve_runtime_does_not_mislabel_other_cuda_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Other X1")

    with pytest.raises(RuntimeError, match="unrecognized CUDA-compatible"):
        resolve_runtime()
