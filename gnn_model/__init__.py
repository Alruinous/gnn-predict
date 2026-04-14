from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "gnn_model"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))
