from __future__ import annotations

from pathlib import Path

from scripts.workflow import profile_workflow_cache as profile


def test_parser_accepts_generic_gpu_kind() -> None:
    args = profile.build_parser().parse_args(["--gpu-kind", " P100 ", "--dry-run"])

    assert args.gpu_kind == "p100"


def test_discover_idle_gpus_matches_requested_kind(monkeypatch) -> None:
    names = ["Tesla P100-PCIE-16GB", "Tesla V100-SXM2-32GB"]
    total_memory = [16_384 * profile.MIB, 32_768 * profile.MIB]
    monkeypatch.setattr(profile.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(profile.pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(profile.pynvml, "nvmlDeviceGetCount", lambda: len(names))
    monkeypatch.setattr(profile.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(profile.pynvml, "nvmlDeviceGetName", names.__getitem__)
    monkeypatch.setattr(
        profile.pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda index: type("Memory", (), {"total": total_memory[index]})(),
    )
    monkeypatch.setattr(
        profile.pynvml,
        "nvmlDeviceGetUUID",
        lambda index: f"GPU-{index}",
    )
    monkeypatch.setattr(profile, "running_gpu_pids", lambda handle: set())

    assert profile.discover_idle_gpus("p100") == [
        profile.GpuDevice(
            index=0,
            name="Tesla P100-PCIE-16GB",
            uuid="GPU-0",
            total_memory_mb=16_384,
        )
    ]


def test_prediction_cache_contains_only_measured_gpu_kind(tmp_path: Path) -> None:
    spec = profile.ProfileSpec(
        group_name="model_group",
        model_name="test-model",
        model_path="/models/test-model",
        dtype="float16",
        phase="decode",
        batch_size=1,
        sequence_length=128,
        decode_output_length=32,
    )
    task = profile.GroupTask(
        name="model_group",
        model_name="test-model",
        model_path="/models/test-model",
        dtype="float16",
        weight_bytes=0,
        specs=(spec,),
    )
    gpu = profile.GpuDevice(
        index=0,
        name="Tesla P100-PCIE-16GB",
        uuid="GPU-0",
        total_memory_mb=16_384,
    )
    profile.write_json(
        profile.load_state_path(tmp_path, task.name),
        {
            "status": "success",
            "predicted_load_sec": 2.0,
            "sample_count": 3,
        },
    )
    profile.save_group_records(
        profile.group_state_path(tmp_path, task.name),
        task.name,
        [
            {
                "spec_id": spec.spec_id,
                "status": "success",
                "duration_sec": 1.0,
                "peak_vram_mb": 1024.0,
                "power_watts_avg": 120.0,
                "gpu": profile.asdict(gpu),
            }
        ],
    )
    manifest = {
        "gpu_kind": "p100",
        "cache_yaml": "/config/workflow/cache.yaml",
        "cache_yaml_sha256": "cache-hash",
        "selection_sha256": "selection-hash",
        "gpus": [profile.asdict(gpu)],
        "environment": {},
        "measurement": {},
    }

    cache = profile.build_prediction_cache(
        tmp_path,
        [task],
        [spec],
        manifest,
        complete=True,
    )

    assert len(cache.entries) == 1
    assert cache.entries[0].key.gpu_name == "p100"
    assert cache.entries[0].predictor_metadata["source_gpu_kind"] == "p100"
