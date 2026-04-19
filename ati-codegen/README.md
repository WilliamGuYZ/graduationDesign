# ati-codegen（毕设：算法代码生成）

面向选题「**基于大语言模型的算法代码生成**」的实验代码，主要包括：

- **数据**：从 KodCode 类数据得到可执行单测校验过的 JSONL，可选地用 API 补全 **思维链（thought_step）** 与 **算法标签（tags）**
- **微调**：对 **Qwen2.5-Coder-7B-Instruct** 做 **LoRA**（`train/train.py` 仅代码，`train/train_cot.py` 为 CoT 格式）
- **评测**：用 **vLLM** 批量生成，按题目自带 `test` 跑 **pass@k**，结果在 `evaluation/results/`

环境安装与依赖说明见 **[SETUP.md](SETUP.md)**。

---

## 目录结构

```text
ati-codegen/
  data/
    raw/                  # KodCode.parquet、KodCode.jsonl、标注中间文件等
    processed/            # KodCode_train.jsonl、KodCode_eval.jsonl
  scripts/
    convet_KodCode_to_jsonl.py   # Parquet → JSONL，并做单测校验
    validate_code_with_test.py   # 执行 solution + test，数据与评测共用
    annotate_thought_steps.py     # 可选：生成 thought_step
    annotate_algorithm_tags.py   # 可选：打 tags
    split_train_eval.py          # 划分训练/评测集（默认读 KodCode_annotate.jsonl）
    train_eval.sh                # Bash：划分 → 训练 → 基座与 LoRA 评测
  train/
    train.py                     # LoRA；最新 adapter 路径写入 latest_lora_adapter.txt
    train_cot.py                 # CoT LoRA；指针为 latest_lora_cot_adapter.txt
    outputs/                     # 训练产出（不入库）
  evaluation/
    eval_base_passk.py           # 基座 pass@k
    eval_lora_passk.py           # LoRA pass@k
    eval_lora_cot_passk.py       # CoT 评测
    results/
  models/
    Qwen2.5-Coder-7B-Instruct/   # 基座权重，需自行下载放置（见 SETUP.md）
```

训练与评测的超参在各 `train/*.py`、`evaluation/*.py` 顶部配置，**没有**单独的 `configs/` 或 `src/` 包目录。

---

## 流程说明

**1. 数据**

- `data/raw/KodCode.parquet` → 运行 `scripts/convet_KodCode_to_jsonl.py` → `data/raw/KodCode.jsonl`（字段与单测校验见脚本内说明）。
- 可选：`annotate_thought_steps.py` 得到 `KodCode_with_thought.jsonl`，再 `annotate_algorithm_tags.py` 得到 `KodCode_annotate.jsonl`。
- `split_train_eval.py` 默认读 `KodCode_annotate.jsonl`，输出 `data/processed/KodCode_train.jsonl` 与 `KodCode_eval.jsonl`（比例、是否打乱见脚本参数）。`train_eval.sh` 里用的是**顺序划分**（`--no-shuffle`）。

**2. 训练**

- 训练集路径在 `train/train.py`、`train_cot.py` 里指向 `data/processed/KodCode_train.jsonl`。
- 跑完后会在 `train/outputs/` 下生成带时间戳目录，并把当前 adapter 目录写入 `latest_lora_adapter.txt` 或 `latest_lora_cot_adapter.txt`，供评测脚本读取。

**3. 评测**

- 评测集默认 `data/processed/KodCode_eval.jsonl`，用 vLLM 多次采样生成，再通过 `validate_code_with_test` 统计 pass@k，结果写入 `evaluation/results/`。

**4. 一键跑通（Linux / Git Bash / WSL）**

```bash
bash scripts/train_eval.sh
```

内容为：划分 → `python train/train.py` → `eval_base_passk.py` → `eval_lora_passk.py`。若做 **CoT** 实验，需自行改用 `train_cot.py` 与 `eval_lora_cot_passk.py`。

---

## 依赖

`requirements.txt` 为某台机器上的完整 **`pip freeze`**（含 PyTorch CUDA 版、vLLM 等），换电脑安装可能需按 PyTorch / vLLM 文档调整索引或版本，见 **SETUP.md**。
