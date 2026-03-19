from __future__ import annotations

from ..prompts import PromptBundle
from ..types import Generation, Problem
from .base import CodeGenModel


class DummyModel(CodeGenModel):
    """用于跑通流程的占位模型：总是返回一段可执行的最小代码。"""

    def generate(self, problem: Problem, prompt: PromptBundle) -> Generation:
        if problem.language != "python":
            return Generation(
                task_id=problem.task_id,
                language=problem.language,
                code="",
                meta={"error": "DummyModel only supports python in mock flow."},
            )

        code = "\n".join(
            [
                "def solve(x):",
                "    # mock implementation",
                "    return x",
                "",
            ]
        )
        return Generation(task_id=problem.task_id, language=problem.language, code=code, meta={"model": "dummy"})

