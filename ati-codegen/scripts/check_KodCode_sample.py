"""
从 KodCode Parquet 读取全部列，
按行规范化后用 test_solution_code 校验；
按扫描顺序取前 SAMPLE_COUNT 条**通过校验**的记录，写入缩进 JSON（顶层为数组，元素为各条样例）。
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
INPUT_PARQUET = _REPO_ROOT / "data" / "raw" / "KodCode.parquet"
OUTPUT_JSON = _REPO_ROOT / "data" / "raw" / "KodCode_sample.json"

# test_solution_code 至少需要这两列（列名须与 Parquet 一致）
_REQUIRED_TEST_COLS = ("solution", "test")

# 通过校验后写入 JSON 的样例条数（按 Parquet 行顺序凑满即停）
SAMPLE_COUNT = 3

# endregion


def _cell_value(v):
    """单元格转 Python 原生值；NaN -> None。不因空字符串丢弃行。"""
    if pd.api.types.is_scalar(v):
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(v, "item") and not isinstance(v, (bytes, str)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _coerce_test_info(raw):
    """尽量转为可 JSON 序列化的 list；不校验子字段是否非空。"""
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    ti = raw
    if hasattr(ti, "tolist"):
        try:
            ti = ti.tolist()
        except (TypeError, ValueError):
            return []
    if isinstance(ti, dict):
        return [{k: _cell_value(ti[k]) for k in ti}]
    if not isinstance(ti, list):
        return []
    out = []
    for item in ti:
        if isinstance(item, dict):
            out.append({k: _cell_value(item[k]) for k in item})
        else:
            out.append(_cell_value(item))
    return out


def _normalize_for_json(v):
    """递归将单元格（含嵌套 dict/list、numpy/pandas）转为可 JSON 序列化的 Python 对象。"""
    if v is None:
        return None
    if pd.api.types.is_scalar(v):
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(v, "tolist") and not isinstance(v, (dict, str, bytes)):
        try:
            return _normalize_for_json(v.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(v, "item") and not isinstance(v, (bytes, str, dict, list, tuple)):
        try:
            out = v.item()
            if type(out).__module__ == "numpy" and hasattr(out, "item"):
                return out.item()
            return out
        except (ValueError, AttributeError):
            pass
    if isinstance(v, dict):
        return {str(k): _normalize_for_json(v[k]) for k in v}
    if isinstance(v, (list, tuple)):
        return [_normalize_for_json(x) for x in v]
    return v


def _row_to_record(row: pd.Series) -> dict:
    """从 DataFrame 行构造记录：包含 Parquet 全部列；不因字段为空返回 None。"""
    record: dict = {}
    for col in row.index:
        raw = row[col]
        if col == "test_info":
            record[col] = _coerce_test_info(raw)
        else:
            record[col] = _normalize_for_json(raw)
    return record


def main() -> None:
    if not INPUT_PARQUET.is_file():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PARQUET}")

    print(f"正在读取 Parquet: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")

    missing_test = [c for c in _REQUIRED_TEST_COLS if c not in df.columns]
    if missing_test:
        raise KeyError(
            f"校验需要列 {missing_test}，实际列: {list(df.columns)}"
        )

    n = int(SAMPLE_COUNT)
    if n < 1:
        raise ValueError("SAMPLE_COUNT 须为 >= 1 的整数")

    picked: list[dict] = []
    scanned = 0
    print(
        f"原始行数: {len(df)}，目标输出 {n} 条通过校验的样例 -> {OUTPUT_JSON}"
    )
    print("使用 validate_code_with_test.test_solution_code 进行校验…")

    for _, row in df.iterrows():
        if len(picked) >= n:
            break
        scanned += 1
        record = _row_to_record(row)

        ok, _err = test_solution_code(
            record.get("solution"),
            record.get("test"),
            record.get("test_info") or [],
        )
        if ok:
            picked.append(record)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not picked:
        print(
            f"警告: 扫描 {scanned} 行后仍无通过校验的记录，已写入 [] -> {OUTPUT_JSON}"
        )
    elif len(picked) < n:
        print(
            f"警告: 扫描 {scanned} 行仅凑齐 {len(picked)}/{n} 条 -> {OUTPUT_JSON}"
        )
    else:
        print(
            f"完成: 扫描 {scanned} 行，已写入 {len(picked)} 条 -> {OUTPUT_JSON}"
        )


if __name__ == "__main__":
    main()
