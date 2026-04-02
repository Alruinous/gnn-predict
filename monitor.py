from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    from gnn_archs.monitoring.cli import main as monitor_main

    return monitor_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
