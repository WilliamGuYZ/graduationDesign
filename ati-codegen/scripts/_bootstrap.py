from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src_path() -> None:
    """
    允许在未 `pip install -e .` 时直接运行 scripts/*.py。
    """
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # ati-codegen/
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

