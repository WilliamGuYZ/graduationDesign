from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from ati_codegen.io import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    args = ap.parse_args()

    # 最小样例：要求实现 solve(x)=x（DummyModel 也会生成这个）
    record = {
        "task_id": "mock/python/0",
        "language": "python",
        "instruction": "实现函数 solve(x)，返回 x 本身。",
        "input_spec": "x: int",
        "output_spec": "int",
        "prompt": "Write a Python function solve(x) that returns x.",
        "tests": "assert ns.solve(1)==1\nassert ns.solve(-3)==-3\n",
        "meta": {"source": "mock"},
    }
    write_jsonl(args.out, [record])
    print(f"Wrote 1 record to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

