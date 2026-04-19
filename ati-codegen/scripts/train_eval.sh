#!/usr/bin/env bash
# 在任意目录执行均可：自动 cd 到 ati-codegen 根目录。默认划分 9:1 → 训练 → 基座评测 → LoRA 评测。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">>> 划分 KodCode_train / KodCode_eval"
python3 scripts/split_train_eval.py --ratio 0.9 --no-shuffle

echo ">>> LoRA 训练"
python3 train/train.py

echo ">>> PASS@K 基座（评测集）"
python3 evaluation/eval_base_passk.py

echo ">>> PASS@K LoRA（评测集，adapter 来自上一步）"
python3 evaluation/eval_lora_passk.py

echo ">>> CoT LoRA 训练"
python3 train/train_cot.py

echo ">>> PASS@K CoT LoRA（评测集，adapter 来自上一步）"
python3 evaluation/eval_lora_cot_passk.py

echo ">>> 全部完成"