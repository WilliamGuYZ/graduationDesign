#!/usr/bin/env bash
# 从“训练阶段”开始的一键实验脚本（不含数据准备/标注/划分）。
# 运行前请确保：
#   1) data/processed/KodCode_train.jsonl 与 KodCode_eval.jsonl 已准备好
#   2) models/Qwen2.5-Coder-7B-Instruct 已就绪
#
# 对照组顺序：
#   A. Base（不训练）                              -> eval_base_passk.py
#   B. LoRA（代码生成）                            -> train.py + eval_lora_passk.py
#   C. CoT zero-shot（shots=0）                    -> train_cot.py + eval_lora_cot_passk.py --few-shot-k 0
#   D. CoT few-shot（shots=${COT_FEW_SHOTS}）      -> train_cot.py + eval_lora_cot_passk.py --few-shot-k ${COT_FEW_SHOTS}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COT_FEW_SHOTS="${COT_FEW_SHOTS:-2}"

#echo "============================================================"
#echo "实验 A：Base 基座模型（无微调）"
#echo "============================================================"
#python3 evaluation/eval_base_passk.py

#echo "============================================================"
#echo "实验 B：LoRA（仅代码）"
#echo "============================================================"
#echo ">>> 训练 LoRA（train.py）"
#python3 train/train.py
echo ">>> 评测 LoRA（eval_lora_passk.py）"
python3 evaluation/eval_lora_passk.py

echo "============================================================"
echo "实验 C：CoT zero-shot（shots=0）"
echo "============================================================"
echo ">>> 训练 CoT（train_cot.py --num-few-shots 0）"
python3 train/train_cot.py --num-few-shots 0
echo ">>> 评测 CoT zero-shot（eval_lora_cot_passk.py --few-shot-k 0）"
python3 evaluation/eval_lora_cot_passk.py --few-shot-k 0 --result-suffix cot_zs

echo "============================================================"
echo "实验 D：CoT few-shot（shots=${COT_FEW_SHOTS}）"
echo "============================================================"
echo ">>> 训练 CoT（train_cot.py --num-few-shots ${COT_FEW_SHOTS}）"
python3 train/train_cot.py --num-few-shots "${COT_FEW_SHOTS}"
echo ">>> 评测 CoT few-shot（eval_lora_cot_passk.py --few-shot-k ${COT_FEW_SHOTS}）"
python3 evaluation/eval_lora_cot_passk.py --few-shot-k "${COT_FEW_SHOTS}" --result-suffix "cot_fs${COT_FEW_SHOTS}"

echo "============================================================"
echo "全部实验完成：A(Base) vs B(LoRA) vs C(CoT-ZS) vs D(CoT-FS)"
echo "============================================================"