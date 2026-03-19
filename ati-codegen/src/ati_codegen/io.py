from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import Problem


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def problem_from_record(r: dict) -> Problem:
    return Problem(
        task_id=str(r["task_id"]),
        language=r["language"],
        instruction=r.get("instruction", ""),
        input_spec=r.get("input_spec", ""),
        output_spec=r.get("output_spec", ""),
        prompt=r["prompt"],
        tests=r.get("tests", ""),
        meta=r.get("meta"),
    )

