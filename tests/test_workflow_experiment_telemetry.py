from __future__ import annotations

import json
import time
from pathlib import Path

from experiment.workflow.telemetry import PeriodicRecorder


class FakeSampler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.index = 0

    def start(self) -> None:
        self.started = True

    def sample(self) -> list[dict[str, object]]:
        self.index += 1
        return [{"ts": float(self.index), "value": self.index}]

    def stop(self) -> None:
        self.stopped = True


def test_periodic_recorder_writes_raw_samples_and_stops(tmp_path: Path) -> None:
    sampler = FakeSampler()
    path = tmp_path / "telemetry.jsonl"
    recorder = PeriodicRecorder(path, sampler, interval_sec=0.01)

    recorder.start()
    rows_at_start = [json.loads(line) for line in path.read_text().splitlines()]
    time.sleep(0.035)
    recorder.stop()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sampler.started
    assert sampler.stopped
    assert rows_at_start == [{"ts": 1.0, "value": 1}]
    assert len(rows) >= 2
    assert [row["value"] for row in rows] == list(range(1, len(rows) + 1))
