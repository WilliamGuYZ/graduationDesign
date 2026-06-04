#!/usr/bin/env bash
# 2 因素析因实验（10 组）一键脚本：H1 ~ H10
#
# 设计：
#   训练侧 ∈ {Base, LoRA_code, LoRA_cot_zs, LoRA_cot_fs(k)}
#   推理侧 ∈ {Direct, CoT-ZS, CoT-FS(k)}
#   合法性约束：若训练侧与推理侧均为 CoT，则两侧示例数 k 必须相等
#              （保证 train/test 上下文分布一致），据此从 4×3=12 组中排除 2 组非法组合，
#              剩 10 组有效实验。详见 README.md「实验矩阵（H1 ~ H10）」。
#
# 使用方式：在本文件顶部打开/关闭各组开关，执行 `bash scripts/train_eval.sh`

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ==================== 手动配置区 ====================
# 建议：首次跑通时仅开 RUN_H1=1、全部 RUN_TRAIN_*=0（需已具备 data/processed 与 models/ 基座）。
# 全量十组实验再按需打开 RUN_TRAIN_* 与 RUN_H2~H10。
# ---- few-shot 示例数（H3/H6/H9/H10 统一使用） ----
FS_K=2

# ---- 训练开关（只在首次或需要重训时打开） ----
RUN_TRAIN_CODE=0     # train/train.py                       → latest_lora_adapter.txt
RUN_TRAIN_COT_ZS=0   # train/train_cot.py --num-few-shots 0 → latest_lora_cot_zs_adapter.txt
RUN_TRAIN_COT_FS=0   # train/train_cot.py --num-few-shots k → latest_lora_cot_fs${FS_K}_adapter.txt

# ---- 评测开关：10 组（H1 ~ H10） ----
# 训练侧 Base
RUN_H1=0    # Base         + Direct
RUN_H2=0    # Base         + CoT-ZS
RUN_H3=0    # Base         + CoT-FS(k)
# 训练侧 LoRA_code（train.py 产物：仅监督 solution）
RUN_H4=0    # LoRA_code    + Direct
RUN_H5=0    # LoRA_code    + CoT-ZS
RUN_H6=0    # LoRA_code    + CoT-FS(k)
# 训练侧 LoRA_cot_zs（train_cot.py --num-few-shots 0）
RUN_H7=0    # LoRA_cot_zs  + Direct
RUN_H8=0    # LoRA_cot_zs  + CoT-ZS
# 训练侧 LoRA_cot_fs(k)（train_cot.py --num-few-shots k）
RUN_H9=0    # LoRA_cot_fs  + Direct
RUN_H10=0   # LoRA_cot_fs  + CoT-FS(k)
# ====================================================

echo "============================================================"
echo "一键实验启动   ROOT=${ROOT}"
echo "FS_K=${FS_K}"
echo "TRAIN:  CODE=${RUN_TRAIN_CODE}  COT_ZS=${RUN_TRAIN_COT_ZS}  COT_FS=${RUN_TRAIN_COT_FS}"
echo "H1=${RUN_H1}  H2=${RUN_H2}  H3=${RUN_H3}"
echo "H4=${RUN_H4}  H5=${RUN_H5}  H6=${RUN_H6}"
echo "H7=${RUN_H7}  H8=${RUN_H8}"
echo "H9=${RUN_H9}  H10=${RUN_H10}"
echo "============================================================"

# -------- 训练阶段 --------
if [ "${RUN_TRAIN_CODE}" = "1" ]; then
  echo ">>> 训练 LoRA_code（train/train.py）"
  python3 train/train.py
fi
if [ "${RUN_TRAIN_COT_ZS}" = "1" ]; then
  echo ">>> 训练 LoRA_cot_zs（train/train_cot.py --num-few-shots 0）"
  python3 train/train_cot.py --num-few-shots 0
fi
if [ "${RUN_TRAIN_COT_FS}" = "1" ]; then
  echo ">>> 训练 LoRA_cot_fs（train/train_cot.py --num-few-shots ${FS_K}）"
  python3 train/train_cot.py --num-few-shots "${FS_K}"
fi

# -------- adapter 指针与路径 --------
CODE_POINTER="${ROOT}/train/latest_lora_adapter.txt"
COT_ZS_POINTER="${ROOT}/train/latest_lora_cot_zs_adapter.txt"
COT_FS_POINTER="${ROOT}/train/latest_lora_cot_fs${FS_K}_adapter.txt"

require_pointer() {
  local label="$1"; local path="$2"; local hint="$3"
  if [ ! -f "${path}" ]; then
    echo "[error] ${label} 缺少指针文件 ${path}" >&2
    echo "        请先 ${hint}" >&2
    exit 1
  fi
}

# -------- 评测阶段 --------

if [ "${RUN_H1}" = "1" ]; then
  echo "============================================================"
  echo "H1：Base + Direct"
  echo "============================================================"
  python3 evaluation/eval_base_passk.py \
    --result-suffix h1_base_direct
fi

if [ "${RUN_H2}" = "1" ]; then
  echo "============================================================"
  echo "H2：Base + CoT-ZS"
  echo "============================================================"
  python3 evaluation/eval_lora_cot_passk.py \
    --lora-path NONE \
    --few-shot-k 0 \
    --result-suffix h2_base_cot_zs
fi

if [ "${RUN_H3}" = "1" ]; then
  echo "============================================================"
  echo "H3：Base + CoT-FS(k=${FS_K})"
  echo "============================================================"
  python3 evaluation/eval_lora_cot_passk.py \
    --lora-path NONE \
    --few-shot-k "${FS_K}" \
    --result-suffix "h3_base_cot_fs${FS_K}"
fi

if [ "${RUN_H4}" = "1" ]; then
  echo "============================================================"
  echo "H4：LoRA_code + Direct"
  echo "============================================================"
  require_pointer "LoRA_code" "${CODE_POINTER}" "打开 RUN_TRAIN_CODE=1"
  python3 evaluation/eval_lora_passk.py \
    --result-suffix h4_lora_code_direct
fi

if [ "${RUN_H5}" = "1" ]; then
  echo "============================================================"
  echo "H5：LoRA_code + CoT-ZS"
  echo "============================================================"
  require_pointer "LoRA_code" "${CODE_POINTER}" "打开 RUN_TRAIN_CODE=1"
  CODE_ADAPTER="$(head -n1 "${CODE_POINTER}")"
  python3 evaluation/eval_lora_cot_passk.py \
    --lora-path "${CODE_ADAPTER}" \
    --few-shot-k 0 \
    --result-suffix h5_lora_code_cot_zs
fi

if [ "${RUN_H6}" = "1" ]; then
  echo "============================================================"
  echo "H6：LoRA_code + CoT-FS(k=${FS_K})"
  echo "============================================================"
  require_pointer "LoRA_code" "${CODE_POINTER}" "打开 RUN_TRAIN_CODE=1"
  CODE_ADAPTER="$(head -n1 "${CODE_POINTER}")"
  python3 evaluation/eval_lora_cot_passk.py \
    --lora-path "${CODE_ADAPTER}" \
    --few-shot-k "${FS_K}" \
    --result-suffix "h6_lora_code_cot_fs${FS_K}"
fi

if [ "${RUN_H7}" = "1" ]; then
  echo "============================================================"
  echo "H7：LoRA_cot_zs + Direct"
  echo "============================================================"
  require_pointer "LoRA_cot_zs" "${COT_ZS_POINTER}" "打开 RUN_TRAIN_COT_ZS=1"
  COT_ZS_ADAPTER="$(head -n1 "${COT_ZS_POINTER}")"
  python3 evaluation/eval_lora_passk.py \
    --lora-path "${COT_ZS_ADAPTER}" \
    --result-suffix h7_lora_cot_zs_direct
fi

if [ "${RUN_H8}" = "1" ]; then
  echo "============================================================"
  echo "H8：LoRA_cot_zs + CoT-ZS"
  echo "============================================================"
  require_pointer "LoRA_cot_zs" "${COT_ZS_POINTER}" "打开 RUN_TRAIN_COT_ZS=1"
  python3 evaluation/eval_lora_cot_passk.py \
    --few-shot-k 0 \
    --result-suffix h8_lora_cot_zs_cot_zs
fi

if [ "${RUN_H9}" = "1" ]; then
  echo "============================================================"
  echo "H9：LoRA_cot_fs(k=${FS_K}) + Direct"
  echo "============================================================"
  require_pointer "LoRA_cot_fs${FS_K}" "${COT_FS_POINTER}" "打开 RUN_TRAIN_COT_FS=1 且 FS_K=${FS_K}"
  COT_FS_ADAPTER="$(head -n1 "${COT_FS_POINTER}")"
  python3 evaluation/eval_lora_passk.py \
    --lora-path "${COT_FS_ADAPTER}" \
    --result-suffix "h9_lora_cot_fs${FS_K}_direct"
fi

if [ "${RUN_H10}" = "1" ]; then
  echo "============================================================"
  echo "H10：LoRA_cot_fs(k=${FS_K}) + CoT-FS(k=${FS_K})"
  echo "============================================================"
  require_pointer "LoRA_cot_fs${FS_K}" "${COT_FS_POINTER}" "打开 RUN_TRAIN_COT_FS=1 且 FS_K=${FS_K}"
  python3 evaluation/eval_lora_cot_passk.py \
    --few-shot-k "${FS_K}" \
    --result-suffix "h10_lora_cot_fs${FS_K}_cot_fs${FS_K}"
fi

echo "============================================================"
echo "全部选中实验完成。结果见 evaluation/results/（H1: eval_base_passk_<suffix>.json；Direct+LoRA: eval_lora_passk_<suffix>.json；CoT: eval_lora_cot_passk_<suffix>.json）"
echo "============================================================"