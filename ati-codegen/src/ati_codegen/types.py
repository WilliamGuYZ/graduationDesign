from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Language = Literal["python", "java", "cpp", "javascript"]


@dataclass(frozen=True)
class Problem:
    """统一表示一条代码生成样本（题面 + 约束 + I/O 格式）。"""

    task_id: str
    language: Language
    instruction: str
    input_spec: str
    output_spec: str
    prompt: str
    tests: str
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class Generation:
    task_id: str
    language: Language
    code: str
    meta: dict[str, Any] | None = None

