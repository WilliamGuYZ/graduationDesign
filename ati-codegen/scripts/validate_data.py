from __future__ import annotations

"""
检查 LoRA 训练用 jsonl 是否与 scripts/train_lora.py 中 build_text 的字段约定一致。

每条非空行须为 JSON 对象，且包含字符串字段 instruction、input、output；
空 input 合法（训练时会替换为「无」）。
"""

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

REQUIRED_KEYS = ("instruction", "input", "output")


def validate_record(obj: object, line_no: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"第 {line_no} 行：根元素必须是 JSON 对象"]

    for key in REQUIRED_KEYS:
        if key not in obj:
            errors.append(f"第 {line_no} 行：缺少必填字段 {key!r}")

    if errors:
        return errors

    for key in REQUIRED_KEYS:
        val = obj[key]
        if not isinstance(val, str):
            errors.append(
                f"第 {line_no} 行：字段 {key!r} 须为 string，实际为 {type(val).__name__}"
            )

    if isinstance(obj.get("instruction"), str) and not obj["instruction"].strip():
        errors.append(f"第 {line_no} 行：instruction 不能为空或仅空白")
    if isinstance(obj.get("output"), str) and not obj["output"].strip():
        errors.append(
            f"第 {line_no} 行：output 不能为空或仅空白（监督微调需要目标文本）"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description="检查 jsonl 是否满足 train_lora.py 的 load_dataset + build_text 要求。"
    )
    ap.add_argument("dataset_path", type=Path, help="jsonl 文件路径")
    ap.add_argument(
        "--max-report",
        type=int,
        default=50,
        help="最多打印的错误条数（0 表示全部打印）",
    )
    args = ap.parse_args()

    path = args.dataset_path
    if not path.is_file():
        print(f"错误：文件不存在 {path}", file=sys.stderr)
        return 1

    total_lines = 0
    all_errors: list[str] = []

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"第 {i} 行：JSON 解析失败 — {e}")
                continue
            all_errors.extend(validate_record(obj, i))

    limit = args.max_report
    to_show = all_errors if limit == 0 else all_errors[:limit]
    for err in to_show:
        print(err, file=sys.stderr)
    if limit and len(all_errors) > limit:
        print(f"... 另有 {len(all_errors) - limit} 条错误未显示", file=sys.stderr)

    print(f"共检查 {total_lines} 条记录（已跳过空行）。")
    if all_errors:
        print(f"发现 {len(all_errors)} 处问题。", file=sys.stderr)
        return 1

    print("格式检查通过：instruction / input / output 均为字符串，且 instruction、output 非空。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
