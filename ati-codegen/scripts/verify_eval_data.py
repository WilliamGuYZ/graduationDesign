#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检验 eval_data.jsonl：在独立命名空间中 exec canonical_solution，再 exec test，
利用参考实现自带的 assert 检查数据一致性。

用法：
  python scripts/verify_eval_data.py -i data/processed/eval_data.jsonl
  python scripts/verify_eval_data.py -i data/processed/eval_data.jsonl --limit 50 --verbose
  python scripts/verify_eval_data.py -i data/processed/eval_data.jsonl --failures failures.txt
"""
import argparse
import ast
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_executable_reference(record: Dict[str, Any]) -> str:
    """
    HumanEval：prompt（代码前缀）+ canonical_solution（通常为缩进的函数体）拼成完整模块；
    旧版 eval_data：prompt 为题面 Markdown、canonical 为完整 def，则仅执行 canonical。
    """
    # 禁止对 canonical 使用 strip()：会去掉首行缩进，导致 HumanEval 体无法拼接。
    p = (record.get("prompt") or "").rstrip()
    # 仅去掉末尾空白，勿用 strip() 以免破坏体代码首行缩进
    c = (record.get("canonical_solution") or "").rstrip("\n\r")
    if not c:
        return ""
    if not p:
        return c
    merged = p + "\n" + c
    try:
        ast.parse(merged)
        return merged
    except SyntaxError:
        pass
    try:
        ast.parse(c)
        return c
    except SyntaxError:
        return merged


def run_one(record: Dict[str, Any], verbose: bool = False) -> Tuple[bool, str]:
    """
    返回 (是否通过, 说明)。
    通过：参考代码与 test 中 assert 均执行成功。
    """
    task_id = record.get("task_id", "?")
    test = (record.get("test") or "").strip()

    # 勿对整段 reference 使用 strip()：无 prompt 且 canonical 以缩进体开头时会破坏语法
    ref_src = build_executable_reference(record).rstrip("\n\r")
    if not ref_src:
        return False, "canonical_solution 为空或无法拼接"
    if not test:
        return False, "test 为空"

    g: Dict[str, Any] = {"__name__": "__main__"}

    try:
        exec(compile(ref_src, "<reference>", "exec"), g, g)
    except SyntaxError as e:
        return False, f"canonical 语法错误: {e}"
    except Exception as e:
        return False, f"canonical 执行异常: {type(e).__name__}: {e}"

    g["__name__"] = "__main__"
    try:
        exec(compile(test, "<test>", "exec"), g, g)
    except AssertionError as e:
        msg = f"断言失败: {e}"
        if verbose:
            msg += "\n" + traceback.format_exc()
        return False, msg
    except Exception as e:
        msg = f"test 执行异常: {type(e).__name__}: {e}"
        if verbose:
            msg += "\n" + traceback.format_exc()
        return False, msg

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="检验 eval_data.jsonl 中参考解能否通过自带 test")
    ap.add_argument(
        "-i",
        "--input",
        default="data/processed/eval_data.jsonl",
        help="JSONL 路径（每行含 canonical_solution、test）",
    )
    ap.add_argument("--limit", type=int, default=0, help="只检验前 N 条（0=全部）")
    ap.add_argument("-v", "--verbose", action="store_true", help="失败时打印完整 traceback")
    ap.add_argument(
        "--failures",
        type=str,
        default="",
        help="将失败的 task_id 逐行写入该文件",
    )
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"错误：找不到文件 {path}", file=sys.stderr)
        sys.exit(1)

    ok_n = 0
    fail_n = 0
    failed_ids: List[Any] = []
    n_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.limit and n_lines >= args.limit:
                break
            n_lines += 1
            rec = json.loads(line)
            task_id = rec.get("task_id")
            passed, msg = run_one(rec, verbose=args.verbose)
            if passed:
                ok_n += 1
            else:
                fail_n += 1
                failed_ids.append(task_id)
                print(f"[FAIL] task_id={task_id}  {msg.splitlines()[0]}")
                if args.verbose and "\n" in msg:
                    print(msg)

    total = ok_n + fail_n
    print()
    print("========== 汇总 ==========")
    print(f"检验条数: {total}")
    print(f"通过: {ok_n}")
    print(f"失败: {fail_n}")
    if total:
        print(f"通过率: {100.0 * ok_n / total:.2f}%")

    if args.failures and failed_ids:
        out = Path(args.failures)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fp:
            for tid in failed_ids:
                fp.write(f"{tid}\n")
        print(f"失败 task_id 已写入: {out.resolve()}")

    sys.exit(0 if fail_n == 0 else 1)


if __name__ == "__main__":
    main()
