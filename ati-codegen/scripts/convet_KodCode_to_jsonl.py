"""
从 KodCode Parquet 导出精简 JSONL，仅保留 question / solution / test / test_info。
每条在跑测试前须满足：
  question、solution（代码）、test 均为非空字符串；
  test_info 为非空 list，且 test_info[0] 中 function_declaration、function_name、parameter_list 均非空。
再按顺序用 validate_code_with_test.test_solution_code 校验，直到凑满目标条数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validate_code_with_test import test_solution_code  # noqa: E402

# region 可调参数
HEAD_N = 10000

INPUT_PARQUET = _REPO_ROOT / "data" / "raw" / "KodCode.parquet"
OUTPUT_JSONL = _REPO_ROOT / "data" / "raw" / "KodCode.jsonl"

_FIELDS = ("question", "solution", "test", "test_info")

# 每处理多少条打印一次进度（按扫描行数）
PROGRESS_EVERY = 500
# endregion


def _nonempty_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _normalize_test_info(raw) -> list | None:
    """返回可用的 test_info 列表；不满足结构要求时返回 None。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ti = raw
    if hasattr(ti, "tolist"):
        try:
            ti = ti.tolist()
        except (TypeError, ValueError):
            return None
    if not isinstance(ti, list) or len(ti) == 0:
        return None
    first = ti[0]
    if not isinstance(first, dict):
        return None
    for key in ("function_declaration", "function_name", "parameter_list"):
        if not _nonempty_str(first.get(key)):
            return None
    return ti


def _row_to_record(row: pd.Series) -> dict | None:
    """从 DataFrame 行构造记录；字段不全或 test_info 不合规则返回 None。"""
    record: dict = {}
    for field in _FIELDS:
        v = row[field]
        try:
            if pd.isna(v):
                v = None
        except (TypeError, ValueError):
            pass
        record[field] = v

    if not _nonempty_str(record.get("question")):
        return None
    if not _nonempty_str(record.get("solution")):
        return None
    if not _nonempty_str(record.get("test")):
        return None

    ti = _normalize_test_info(record.get("test_info"))
    if ti is None:
        return None
    record["test_info"] = ti
    return record


def main() -> None:
    if not INPUT_PARQUET.is_file():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PARQUET}")

    print(f"正在读取 Parquet: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")

    missing = [c for c in _FIELDS if c not in df.columns]
    if missing:
        raise KeyError(f"Parquet 缺少字段: {missing}，实际列: {list(df.columns)}")

    target = int(HEAD_N) if HEAD_N is not None else None
    if target is not None and target <= 0:
        raise ValueError("HEAD_N 必须为正整数或 None")

    selected: list[dict] = []
    scanned = 0
    skipped_bad_row = 0

    print(f"原始行数: {len(df)}，目标通过验证条数: {target}")
    print("使用 validate_code_with_test.test_solution_code 进行校验…")

    for _, row in df.iterrows():
        if target is not None and len(selected) >= target:
            break

        scanned += 1
        record = _row_to_record(row)
        if record is None:
            skipped_bad_row += 1
            if PROGRESS_EVERY and scanned % PROGRESS_EVERY == 0:
                print(
                    f"  已扫描 {scanned} 行，已通过 {len(selected)}/{target}，"
                    f"缺字段跳过 {skipped_bad_row}"
                )
            continue

        ok, _err = test_solution_code(
            record["solution"],
            record["test"],
            record.get("test_info") or [],
        )
        if ok:
            selected.append(record)

        if PROGRESS_EVERY and scanned % PROGRESS_EVERY == 0:
            print(
                f"  已扫描 {scanned} 行，已通过 {len(selected)}/{target if target else '∞'}"
            )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"完成: 扫描 {scanned} 行，写入 {len(selected)} 条 -> {OUTPUT_JSONL}"
    )
    if target is not None and len(selected) < target:
        print(
            f"警告: 仅凑齐 {len(selected)} 条（不足目标 {target}），"
            "请扩大 Parquet 数据源或放宽校验。"
        )


if __name__ == "__main__":
    main()
