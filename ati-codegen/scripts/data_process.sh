#!/usr/bin/env bash
# TACO 数据处理流水线（在 ati-codegen 仓库根目录执行）：
#   1) taco_arrow_to_jsonl.py — HF Arrow 分片 → JSONL
#   2) filter_passed.py — 仅保留 solution 能通过测例的样本
#
# 默认路径相对仓库根目录，可用环境变量覆盖，例如：
#   TRAIN_ARROW=data/TACO/train RAW_TRAIN=out/raw_train.jsonl bash scripts/taco_process.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

TACO_ARROW="${TRAIN_ARROW:-data/TACO/train}"
TACO_JSONL="${RAW_TRAIN:-data/raw/taco.jsonl}"
TACO_PASSED="${PASSED_TRAIN:-data/raw/taco_passed.jsonl}"
FILTER_TIMEOUT="${FILTER_TIMEOUT:-2.0}"
FILTER_MAX_CASES="${FILTER_MAX_CASES:-1}"


echo "==> Arrow → JSONL"
"$PY" scripts/taco_arrow_to_jsonl.py --arrow_dir "$TACO_ARROW" --out "$TACO_JSONL"

echo "==> filter_passed"
"$PY" scripts/filter_passed.py \
  --input "$TACO_JSONL" \
  --output "$TACO_PASSED" \
  --timeout "$FILTER_TIMEOUT" \
  --max-cases "$FILTER_MAX_CASES"

