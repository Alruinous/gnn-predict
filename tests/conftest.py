from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

# ray 2.55 的 uv run 集成会把整个项目 (含 .venv) 上传为 working_dir.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def ray_session():
    import ray

    # ray worker 进程不继承 driver 的 sys.path, src 和 tests 必须显式进 PYTHONPATH.
    pythonpath = ":".join([str(SRC), str(ROOT / "tests")])
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
    )
    yield
    ray.shutdown()


@pytest.fixture()
def wait_until() -> Callable[..., bool]:
    def _wait(
        predicate: Callable[[], bool],
        timeout_sec: float = 60.0,
        interval_sec: float = 0.2,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval_sec)
        return False

    return _wait
