#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data/raw/Total.json 生成 Alpaca JSONL：
  instruction = content.problem + content.examples + content.constraints（合并为完整任务描述）
  input = ""
  output = 从 python 字段的 Markdown 代码块中提取的纯代码（无 ``` 围栏）
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def extract_python_code(text):
    # type: (Optional[str]) -> str
    """从 python 字段中提取第一个 ```python / ```py / ``` 代码块内的纯代码。"""
    if not text:
        return ""
    s = text.strip()
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.DOTALL)
        if m:
            return m.group(1).strip()
    return s.strip()


def build_instruction(content):
    # type: (Dict[str, Any]) -> str
    parts = []
    problem = content.get("problem")
    if problem:
        parts.append(str(problem).strip())

    examples = content.get("examples")
    if examples:
        ex_text = "\n\n".join(str(e).strip() for e in examples if e)
        if ex_text:
            parts.append(ex_text)

    cons = content.get("constraints")
    if cons:
        parts.append(str(cons).strip())

    return "\n\n".join(parts).strip()


def iter_records(data):
    # type: (List[Dict[str, Any]])
    for item in data:
        content = item.get("content") or {}
        instruction = build_instruction(content)
        raw_py = item.get("python")
        output = extract_python_code(raw_py if raw_py else "")
        if not instruction or not output:
            continue
        yield {
            "instruction": instruction,
            "input": "",
            "output": output,
        }


def convert(input_path, output_path):
    # type: (str, str) -> int
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 应为数组")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in iter_records(data):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Total.json → Alpaca（合并题面 + 纯 Python 代码）")
    p.add_argument(
        "-i",
        "--input",
        default=os.path.join(root, "data", "raw", "Total.json"),
    )
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(root, "data", "processed", "train_code.jsonl"),
    )
    args = p.parse_args()
    n = convert(args.input, args.output)
    print("已写入 {}，共 {} 条".format(args.output, n))


if __name__ == "__main__":
    main()
