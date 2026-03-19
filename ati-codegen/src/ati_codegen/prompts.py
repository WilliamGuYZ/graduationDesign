from __future__ import annotations

from dataclasses import dataclass

from .types import Problem


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


DEFAULT_SYSTEM = "你是一位代码生成领域的专家，擅长编写正确、可读、可通过测试的代码。"


def build_plain_prompt(p: Problem) -> PromptBundle:
    user = "\n\n".join(
        [
            p.instruction or "请根据题目描述生成满足输入输出要求的代码。",
            "【题目】",
            p.prompt,
            "【输出要求】",
            "只输出代码，不要解释。",
        ]
    )
    return PromptBundle(system=DEFAULT_SYSTEM, user=user)


def build_cot_prompt(p: Problem) -> PromptBundle:
    user = "\n\n".join(
        [
            p.instruction or "请根据题目描述生成满足输入输出要求的代码。",
            "请先给出简短的解题思路（不要展开长篇推理），再给出最终代码。",
            "【题目】",
            p.prompt,
            "【输出格式】",
            "先输出：思路：<一句话到三句话>\n再输出：代码：<代码正文>",
        ]
    )
    return PromptBundle(system=DEFAULT_SYSTEM, user=user)

