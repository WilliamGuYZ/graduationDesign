from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .text_clean import html_to_text


@dataclass(frozen=True)
class BuildOptions:
    prefer_translated: bool = True
    languages: tuple[str, ...] = ("python", "java")
    include_paid: bool = False


def _pick_statement(q: dict[str, Any], prefer_translated: bool) -> str:
    if prefer_translated and q.get("translatedContent"):
        return html_to_text(q["translatedContent"])
    return html_to_text(q.get("content") or "")


def _normalize_lang(lang_slug: str) -> str | None:
    m = {
        "python3": "python",
        "python": "python",
        "java": "java",
        "cpp": "cpp",
        "javascript": "javascript",
    }
    return m.get(lang_slug)


def _snippets_by_lang(q: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for snip in q.get("codeSnippets") or []:
        lang = _normalize_lang(snip.get("langSlug", ""))
        if not lang:
            continue
        code = (snip.get("code") or "").strip()
        if code:
            out[lang] = code
    return out


def build_records_from_question(q: dict[str, Any], opts: BuildOptions) -> Iterable[dict]:
    """
    将单道题的详情转换成多语言 JSONL 记录。

    说明：
    - LeetCode 公开接口无法直接拿到完整隐藏测试；这里先落盘 sampleTestCase/exampleTestcases，
      以及各语言函数骨架（codeSnippets），方便你后续构造自建评测（或改用开源数据集）。
    """
    meta = {
        "source": "leetcode.cn",
        "questionFrontendId": q.get("questionFrontendId"),
        "title": q.get("title"),
        "titleSlug": q.get("titleSlug"),
        "difficulty": q.get("difficulty"),
        "topicTags": q.get("topicTags") or [],
        "sampleTestCase": q.get("sampleTestCase"),
        "exampleTestcases": q.get("exampleTestcases"),
    }

    statement = _pick_statement(q, prefer_translated=opts.prefer_translated)
    snippets = _snippets_by_lang(q)

    for lang in opts.languages:
        if lang not in snippets:
            continue
        prompt = "\n\n".join(
            [
                f"题目：{q.get('title','')}",
                statement,
                "",
                "请根据下方函数签名补全实现：",
                snippets[lang],
            ]
        ).strip()

        yield {
            "task_id": f"leetcode-cn/{lang}/{q.get('questionFrontendId') or q.get('questionId')}",
            "language": lang,
            "instruction": "根据题目描述与函数签名生成可通过测试的代码。",
            "input_spec": "",
            "output_spec": "",
            "prompt": prompt,
            "tests": "",  # 真实 pass@k 需要可执行 tests；你后续可用开源数据集或自建评测补齐
            "meta": meta,
        }

