from __future__ import annotations

import json
from pathlib import Path

from dataset.__main__ import main


def test_dataset_cli_writes_normalized_jsonl(tmp_path: Path) -> None:
    gsm8k_dir = tmp_path / "gsm8k"
    gsm8k_dir.mkdir()
    (gsm8k_dir / "train.jsonl").write_text(
        '{"question": "Train?", "answer": "Work\\n#### 1"}\n',
        encoding="utf-8",
    )
    (gsm8k_dir / "test.jsonl").write_text(
        '{"question": "Test?", "answer": "Work\\n#### 2"}\n',
        encoding="utf-8",
    )
    mbpp_path = tmp_path / "sanitized-mbpp.json"
    mbpp_path.write_text(
        json.dumps(
            [
                {
                    "task_id": 11,
                    "prompt": "Write a function.",
                    "code": "def f():\n    return 1",
                    "test_imports": [],
                    "test_list": ["assert f() == 1"],
                    "source_file": "fixture",
                }
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--gsm8k_dir",
            str(gsm8k_dir),
            "--mbpp_path",
            str(mbpp_path),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    gsm8k_test = read_jsonl(output_dir / "gsm8k_test.jsonl")
    mbpp = read_jsonl(output_dir / "mbpp_sanitized.jsonl")
    assert gsm8k_test[0]["sample_id"] == "test_00000"
    assert gsm8k_test[0]["gold_answer"] == "2"
    assert mbpp[0]["source_dataset"] == "mbpp_sanitized"
    assert mbpp[0]["split"] == "test"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
