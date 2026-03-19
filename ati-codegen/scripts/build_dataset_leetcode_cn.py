from __future__ import annotations

import argparse
import time
from pathlib import Path

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from tqdm import tqdm  # type: ignore

from ati_codegen.datasets.leetcode_cn import LeetCodeCnClient, LeetCodeCnConfig
from ati_codegen.datasets.leetcode_to_jsonl import BuildOptions, build_records_from_question
from ati_codegen.io import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="输出 jsonl 路径")
    ap.add_argument("--limit", type=int, default=200, help="最多抓取多少题（按列表顺序）")
    ap.add_argument("--difficulty", default="", help="EASY/MEDIUM/HARD 或留空")
    ap.add_argument("--languages", default="python,java", help="输出语言（逗号分隔）")
    ap.add_argument("--sleep", type=float, default=0.3, help="请求间隔秒数")
    ap.add_argument("--cookie", default="", help="可选：浏览器复制的 Cookie（遇到403/风控再填）")
    ap.add_argument("--csrf", default="", help="可选：x-csrftoken")
    ap.add_argument("--prefer-translated", action="store_true", help="优先使用中文译文题面")
    ap.add_argument("--include-paid", action="store_true", help="是否包含付费题")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = LeetCodeCnConfig(
        cookie=args.cookie or None,
        csrf_token=args.csrf or None,
        sleep_s=float(args.sleep),
    )
    cli = LeetCodeCnClient(cfg)

    difficulty = args.difficulty.strip().upper() or None
    languages = tuple(x.strip().lower() for x in args.languages.split(",") if x.strip())
    opts = BuildOptions(prefer_translated=bool(args.prefer_translated), languages=languages, include_paid=bool(args.include_paid))

    records: list[dict] = []

    # 分页抓题目列表，再逐题拉详情
    page_size = 50
    got = 0
    skip = 0
    pbar = tqdm(total=args.limit, desc="fetch leetcode.cn")
    while got < args.limit:
        batch = cli.list_questions(skip=skip, limit=page_size, difficulty=difficulty)
        if not batch:
            break
        skip += page_size
        for it in batch:
            if got >= args.limit:
                break
            if it.get("paidOnly") and not opts.include_paid:
                continue
            slug = it.get("titleSlug")
            if not slug:
                continue
            q = cli.fetch_question_detail(slug)
            for r in build_records_from_question(q, opts):
                records.append(r)
            got += 1
            pbar.update(1)
            time.sleep(cfg.sleep_s)
    pbar.close()

    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

