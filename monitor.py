from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fail-fast Prometheus metrics for result documents.",
    )
    parser.add_argument(
        "--config",
        default="config/monitor/monitor.yaml",
        help="Monitor YAML config file to process.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from gnn_archs.monitoring import load_monitor_settings, run_monitoring

    args = parse_args(argv)
    settings = load_monitor_settings(Path(args.config))
    written_paths = run_monitoring(settings)
    for path in written_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
