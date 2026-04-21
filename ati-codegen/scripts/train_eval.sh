#!/usr/bin/env bash
# 从“训练阶段”开始的一键实验脚本（不含数据准备/标注/划分）。
# 运行前请确保：
#   1) data/processed/KodCode_train.jsonl 与 KodCode_eval.jsonl 已准备好
#   2) models/Qwen2.5-Coder-7B-Instruct 已就绪
#
# 对照组顺序：
#   A. Base（不训练）             -> eval_base_passk.py
#   B. LoRA（代码生成）           -> train.py + eval_lora_passk.py
#   C. LoRA-CoT（两阶段推理）     -> train_cot.py + eval_lora_cot_passk.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "实验 A：Base 基座模型（无微调）"
echo "============================================================"
python3 evaluation/eval_base_passk.py

echo "============================================================"
echo "实验 B：LoRA（仅代码）"
echo "============================================================"
echo ">>> 训练 LoRA（train.py）"
python3 train/train.py
echo ">>> 评测 LoRA（eval_lora_passk.py）"
python3 evaluation/eval_lora_passk.py

echo "============================================================"
echo "实验 C：LoRA-CoT（先推理再代码）"
echo "============================================================"
echo ">>> 训练 LoRA-CoT（train_cot.py）"
python3 train/train_cot.py
echo ">>> 评测 LoRA-CoT（eval_lora_cot_passk.py）"
python3 evaluation/eval_lora_cot_passk.py

echo "============================================================"
echo "全部实验完成：A(Base) vs B(LoRA) vs C(LoRA-CoT)"
echo "============================================================"