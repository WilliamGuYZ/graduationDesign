from __future__ import annotations

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

try:
    from rich.console import Console  # type: ignore

    def _printer():
        c = Console()
        return c.print

except Exception:  # noqa: BLE001

    def _printer():
        return print

from ati_codegen.eval.passk import passk_from_bools
from ati_codegen.eval.python_runner import run_python_tests
from ati_codegen.io import problem_from_record, read_jsonl
from ati_codegen.models.dummy import DummyModel
from ati_codegen.prompts import build_plain_prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="jsonl 数据集路径")
    ap.add_argument("--k", type=int, default=1, help="pass@k 的 k")
    args = ap.parse_args()

    pr = _printer()
    model = DummyModel()

    records = read_jsonl(args.dataset)
    passed_flags: list[bool] = []

    for r in records:
        p = problem_from_record(r)
        prompt = build_plain_prompt(p)
        gen = model.generate(p, prompt)
        if p.language != "python":
            passed_flags.append(False)
            pr(f"{p.task_id} skip (only python supported in mock)")
            continue
        rr = run_python_tests(gen.code, p.tests)
        passed_flags.append(rr.ok)
        pr(f"{p.task_id} ok={rr.ok} err={rr.error}")

    res = passk_from_bools(passed_flags, k=args.k)
    pr(f"\npass@{res.k} = {res.pass_at_k:.4f} (n={res.n}, c={res.c})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

