#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data/raw/Total.json（LeetCode 风格）清洗为 HumanEval 测试集风格的 JSONL。

每行核心字段（与 OpenAI HumanEval / HuggingFace openai_humaneval 一致）：
  - task_id:   "Total/{题目 id}"
  - prompt:    import / 辅助类 / 其它顶层定义 + 目标函数签名与题面 docstring（无 pass，供与 canonical 拼接成完整 def）
  - canonical_solution: 目标函数实现部分（仅函数体语句，缩进与源码一致）
  - entry_point: 模块中最后一个顶层 def 的函数名
  - test:      METADATA + def check(candidate): ... + if __name__ == "__main__": check(entry_point)

说明：仅当存在至少一个可解析的顶层 def、且能根据 example 生成合法 assert 时保留；无顶层 def、
仅 class 内方法、或测例无法解析时会跳过。

用法：
  python scripts/convert_eval_data.py
  python scripts/convert_eval_data.py -i data/raw/Total.json -o data/processed/eval_data.jsonl --limit 100
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def extract_python_code(text: Optional[str]) -> str:
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


def build_instruction(content: Dict[str, Any]) -> str:
    parts: List[str] = []
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


def split_top_level_commas(s: str) -> List[str]:
    """按括号深度为 0 的逗号切分参数（用于 LeetCode example.input）。"""
    depth = 0
    parts: List[str] = []
    cur: List[str] = []
    for c in s:
        if c in "[{(":
            depth += 1
        elif c in "]})":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def literal_eval_loose(s: str) -> Any:
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        s2 = s.replace("true", "True").replace("false", "False").replace("null", "None")
        return ast.literal_eval(s2)


def code_has_listnode(code: str) -> bool:
    return bool(re.search(r"^\s*class\s+ListNode\b", code, re.MULTILINE))


def code_has_treenode(code: str) -> bool:
    return bool(re.search(r"^\s*class\s+TreeNode\b", code, re.MULTILINE))


def all_examples_two_list_args(examples: Sequence[Dict[str, Any]]) -> bool:
    """仅当每条 example 的 input 都可拆成两个 list 参数时，才用 ListNode 辅助测例。"""
    for ex in examples:
        inp = ex.get("input")
        if inp is None:
            return False
        parts = split_top_level_commas(str(inp))
        if len(parts) != 2:
            return False
        try:
            a = literal_eval_loose(parts[0])
            b = literal_eval_loose(parts[1])
        except Exception:
            return False
        if not isinstance(a, list) or not isinstance(b, list):
            return False
    return True


def needs_typing_import(code: str) -> bool:
    if re.search(r"^from\s+typing\s+import", code, re.MULTILINE):
        return False
    return bool(
        re.search(r"\bList\s*\[", code)
        or re.search(r"\bOptional\s*\[", code)
        or re.search(r"\bDict\s*\[", code)
        or re.search(r"\bSet\s*\[", code)
        or re.search(r"\bTuple\s*\[", code)
    )


def typing_import_preamble() -> str:
    return "from typing import Any, Dict, List, Optional, Set, Tuple\n\n"


def parse_example_io(
    inp: str, out: str
) -> Optional[Tuple[List[Any], Any]]:
    try:
        arg_strs = split_top_level_commas(inp)
        args = [literal_eval_loose(a) for a in arg_strs]
        expected = literal_eval_loose(out)
        return args, expected
    except Exception:
        return None


def build_test_block(
    entry_point: str,
    examples: Sequence[Dict[str, Any]],
    code: str,
) -> Optional[str]:
    """生成 HumanEval 风格 test 字符串；失败返回 None。"""
    if not examples:
        return None

    # 仅在「双链表入参」典型题（两数相加等）下用 ListNode 辅助；含 TreeNode 或多参/非双 list 走普通字面量调用
    uses_ll = (
        code_has_listnode(code)
        and not code_has_treenode(code)
        and all_examples_two_list_args(examples)
    )
    lines: List[str] = [
        'METADATA = {"language": "python", "dataset": "Total"}',
        "",
    ]

    check_lines: List[str] = ["def check(candidate):"]

    if uses_ll:
        check_lines.append("    def _as_list(node):")
        check_lines.append("        if node is None:")
        check_lines.append("            return None")
        check_lines.append("        if isinstance(node, list):")
        check_lines.append("            return node")
        check_lines.append("        r = []")
        check_lines.append("        while node:")
        check_lines.append("            r.append(node.val)")
        check_lines.append("            node = node.next")
        check_lines.append("        return r")
        check_lines.append("    def _from_list(arr):")
        check_lines.append("        h = ListNode(0)")
        check_lines.append("        t = h")
        check_lines.append("        for x in arr:")
        check_lines.append("            t.next = ListNode(x)")
        check_lines.append("            t = t.next")
        check_lines.append("        return h.next")

    for ex in examples:
        inp = ex.get("input")
        out = ex.get("output")
        if inp is None or out is None:
            return None
        parsed = parse_example_io(str(inp), str(out))
        if parsed is None:
            return None
        args, expected = parsed

        if uses_ll:
            if len(args) < 2:
                return None
            arg_parts = ["_from_list(%s)" % repr(a) for a in args]
            call = "candidate(%s)" % ", ".join(arg_parts)
            lhs = "_as_list(%s)" % call
            lines_assert = "    assert %s == %s" % (lhs, repr(expected))
        else:
            parts_repr = ", ".join(repr(a) for a in args)
            lines_assert = "    assert candidate(%s) == %s" % (parts_repr, repr(expected))

        check_lines.append(lines_assert)

    check_lines.append("")
    check_lines.append('if __name__ == "__main__":')
    check_lines.append("    check(%s)" % entry_point)

    lines.extend(check_lines)
    return "\n".join(lines) + "\n"


def _last_top_level_function(tree: ast.Module) -> Optional[ast.FunctionDef]:
    last: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            last = node
    return last


def _stmt_docstring(stmt: ast.stmt) -> Optional[str]:
    if isinstance(stmt, ast.Expr):
        v = stmt.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
        if isinstance(v, ast.Str):  # py35-37
            return v.s
    return None


def _implementation_statements(body: List[ast.stmt]) -> List[ast.stmt]:
    if not body:
        return []
    if _stmt_docstring(body[0]) is not None:
        return body[1:]
    return body


def _max_lineno(node: ast.AST) -> int:
    m = getattr(node, "lineno", None) or 0
    for ch in ast.iter_child_nodes(node):
        m = max(m, _max_lineno(ch))
    return m


def _node_source(code: str, node: ast.AST) -> Optional[str]:
    lines = code.splitlines()
    a = node.lineno - 1
    b = _max_lineno(node) - 1
    if a < 0 or b >= len(lines):
        return None
    return "\n".join(lines[a : b + 1])


def _function_header_source(code: str, target: ast.FunctionDef) -> Optional[str]:
    """从 decorator 或 def 起，到第一个函数体语句之前（不含 docstring/实现）。"""
    lines = code.splitlines()
    if target.decorator_list:
        start = min(d.lineno for d in target.decorator_list) - 1
    else:
        start = target.lineno - 1
    if not target.body:
        return None
    body0 = target.body[0].lineno - 1
    if body0 < start:
        return None
    return "\n".join(lines[start:body0])


def _format_docstring_for_stub(text: str) -> str:
    """生成可嵌入函数体的 docstring 字面量（题面中若含 \"\"\" 须用单引号 docstring）。"""
    if '"""' not in text:
        return '"""%s"""' % text
    if "'''" not in text:
        return "'''%s'''" % text
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return '"""%s"""' % escaped


def _stmt_source(code: str, st: ast.stmt) -> Optional[str]:
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(code, st)
        if seg is not None:
            return seg
    return _node_source(code, st)


def build_humaneval_record(
    item: Dict[str, Any], code: str, instruction: str
) -> Optional[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    target = _last_top_level_function(tree)
    if target is None:
        return None
    name = target.name
    if name == "__init__":
        return None

    try:
        idx = tree.body.index(target)
    except ValueError:
        return None

    preamble_nodes = tree.body[:idx]

    preamble_parts: List[str] = []
    for n in preamble_nodes:
        src = _node_source(code, n)
        if src is None:
            return None
        preamble_parts.append(src)
    preamble = "\n\n".join(preamble_parts)

    header = _function_header_source(code, target)
    if header is None:
        return None

    doc_tok = _format_docstring_for_stub(instruction)
    # 不以 pass 结尾，便于与 canonical（缩进函数体）拼接为合法语法
    stub_lines = "    %s" % doc_tok
    if preamble:
        prompt = preamble + "\n\n" + header + "\n" + stub_lines + "\n"
    else:
        prompt = header + "\n" + stub_lines + "\n"

    if needs_typing_import(code):
        prompt = typing_import_preamble() + prompt

    impl = _implementation_statements(target.body)
    if not impl:
        return None

    segments: List[str] = []
    for st in impl:
        seg = _stmt_source(code, st)
        if seg is None:
            return None
        segments.append(seg)
    canonical_solution = "\n".join(segments).rstrip() + "\n"

    tid = item.get("id")
    task_id = "Total/%s" % tid if tid is not None else "Total/unknown"

    examples = item.get("example") or []
    if not isinstance(examples, list):
        examples = []

    test = build_test_block(name, examples, code)
    if test is None:
        return None

    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "entry_point": name,
        "test": test,
    }


def convert_total_to_humaneval_jsonl(
    input_path: Path,
    output_path: Path,
    limit: int = 0,
    include_meta: bool = True,
) -> Tuple[int, int]:
    with open(input_path, "r", encoding="utf-8") as f:
        if input_path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入应为 JSON 数组，或 JSONL 多行对象")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            if limit and n_ok >= limit:
                break

            content = item.get("content") or {}
            instruction = build_instruction(content)
            py_raw = item.get("python")
            code = extract_python_code(py_raw if py_raw else "")

            if not instruction or not code:
                n_skip += 1
                continue

            rec = build_humaneval_record(item, code, instruction)
            if rec is None:
                n_skip += 1
                continue

            if include_meta:
                prob = content.get("problem")
                if prob:
                    rec["description"] = str(prob).strip()[:8000]
                if item.get("title") is not None:
                    rec["title"] = item.get("title")
                if item.get("difficulty") is not None:
                    rec["difficulty"] = item.get("difficulty")

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    return n_ok, n_skip


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Total.json → HumanEval 风格 JSONL（eval 集）"
    )
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=root / "data" / "raw" / "Total.json",
        help="输入：Total.json 或 JSONL 列表",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "data" / "processed" / "eval_data.jsonl",
        help="输出 JSONL",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多写入 N 条成功样本（0=不限制）",
    )
    ap.add_argument(
        "--no-meta",
        action="store_true",
        help="不写入 description/title/difficulty，仅核心字段",
    )

    args = ap.parse_args()

    if not args.input.is_file():
        print("错误：找不到输入文件 %s" % args.input, file=sys.stderr)
        sys.exit(1)

    n_ok, n_skip = convert_total_to_humaneval_jsonl(
        args.input,
        args.output,
        limit=args.limit,
        include_meta=not args.no_meta,
    )

    print("输出: %s" % args.output.resolve())
    print("成功: %s 条，跳过: %s 条" % (n_ok, n_skip))


if __name__ == "__main__":
    main()
