from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceRuntime:
    device: torch.device
    backend: str
    device_name: str


def resolve_runtime(requested_backend: str = "auto") -> DeviceRuntime:
    """Identify the accelerator separately from PyTorch's device type.

    CoreX exposes BI-V150 through the CUDA-compatible torch.cuda API, so
    ``device.type == 'cuda'`` alone cannot identify the hardware vendor.
    """
    if requested_backend not in {"auto", "cpu", "nvidia", "corex"}:
        raise ValueError(f"unsupported device backend: {requested_backend}")
    if requested_backend == "cpu":
        return DeviceRuntime(torch.device("cpu"), "cpu", "CPU")
    if not torch.cuda.is_available():
        if requested_backend != "auto":
            raise RuntimeError(
                f"{requested_backend} backend requested but no GPU is available"
            )
        return DeviceRuntime(torch.device("cpu"), "cpu", "CPU")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    normalized_name = device_name.lower()
    if "iluvatar" in normalized_name:
        detected_backend = "corex"
    elif requested_backend == "nvidia" or any(
        marker in normalized_name
        for marker in ("nvidia", "tesla", "geforce", "quadro", "v100", "a100")
    ):
        detected_backend = "nvidia"
    else:
        raise RuntimeError(
            f"unrecognized CUDA-compatible accelerator: {device_name}; "
            "add a device backend before collecting data"
        )
    if requested_backend != "auto" and requested_backend != detected_backend:
        raise RuntimeError(
            f"{requested_backend} backend requested, but cuda:0 is {device_name} "
            f"({detected_backend})"
        )
    return DeviceRuntime(device, detected_backend, device_name)
