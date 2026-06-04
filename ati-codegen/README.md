# ati-codegen（毕设：算法代码生成）

面向选题「**基于 CodeGeeX4-ALL-9B 的算法代码生成**」的实验代码，主要包括：

- **数据**：从 KodCode 类数据得到可执行单测校验过的 JSONL，可选地用 API 补全 **思维链（thought_step）** 与 **算法标签（tags）**
- **微调**：对 **CodeGeeX4-ALL-9B** 做 **LoRA**（`train/train.py` 仅代码，`train/train_cot.py` 为 CoT 格式）
- **评测**：用 **vLLM** 批量生成，按题目自带 `test` 跑 **pass@k**，结果在 `evaluation/results/`
- **实验设计**：按 *训练侧 × 推理侧* 二因素析因展开，共 **H1 ~ H10** 十组实验，详见下文「实验矩阵（H1 ~ H10）」章节

环境安装与依赖说明见 **[SETUP.md](SETUP.md)**。

---

## 目录结构

```text
ati-codegen/
  data/
    raw/                  # KodCode.jsonl、标注中间文件等（Git LFS）
    processed/            # KodCode_train.jsonl、KodCode_eval.jsonl
  scripts/
    convet_KodCode_to_jsonl.py   # Parquet → JSONL，并做单测校验
    validate_code_with_test.py   # 执行 solution + test，数据与评测共用
    annotate_thought_steps.py    # 可选：生成 thought_step
    annotate_algorithm_tags.py   # 可选：打 tags
    split_train_eval.py          # 划分训练/评测集（默认读 KodCode_annotate.jsonl）
    check_max_seq_length.py      # Token 长度统计（Direct 模板）
    check_cot_seq_length.py      # Token 长度统计（CoT 模板）
    check_KodCode_sample.py      # 数据集样本检查
    train_eval.sh                # Bash：一键 H1~H10（训练 → 评测）
  train/
    train.py                     # LoRA_code；指针 latest_lora_adapter.txt
    train_cot.py                 # LoRA_cot_zs / LoRA_cot_fs(k)；指针见 README
    outputs/                     # 训练产出（不入库）
  evaluation/
    eval_base_passk.py           # 基座 pass@k（H1）
    eval_lora_passk.py           # LoRA + Direct pass@k（H4/H7/H9）
    eval_lora_cot_passk.py       # CoT 两阶段 pass@k（H2/H3/H5/H6/H8/H10）
    results/
  models/
    CodeGeeX4-ALL-9B/            # 基座权重，需自行下载放置（见 SETUP.md）
```

训练与评测的超参在各 `train/*.py`、`evaluation/*.py` 顶部配置，**没有**单独的 `configs/` 或 `src/` 包目录。

---

## 流程说明

**1. 数据**

- `data/raw/KodCode.parquet` → 运行 `scripts/convet_KodCode_to_jsonl.py` → `data/raw/KodCode.jsonl`（字段与单测校验见脚本内说明）。
- 可选：`annotate_thought_steps.py` 得到 `KodCode_with_thought.jsonl`，再 `annotate_algorithm_tags.py` 得到 `KodCode_annotate.jsonl`。
- `split_train_eval.py` 默认读 `KodCode_annotate.jsonl`，输出 `data/processed/KodCode_train.jsonl` 与 `KodCode_eval.jsonl`（比例、是否打乱见脚本参数）。`train_eval.sh` 里用的是**顺序划分**（`--no-shuffle`）。

**2. 训练**

- 默认训练集：`data/processed/KodCode_train.jsonl`（可用 `--data-path` 覆盖）。
- `train/train.py`：仅监督 `solution`（Direct 模板）；默认单条序列截断 `MAX_LENGTH=1024`，可用 `--max-length` 在显存紧张时下调（不宜低于 256）。
- `train/train_cot.py`：CoT 格式 + 可选 few-shot；默认 `MAX_LENGTH=2048`（兼顾 CoT 目标段保留与 32GB 单卡上限），仍 OOM 时可降至 `--max-length 1536`（≥512）。会先过滤掉 `thought_step` 为空的样本。
- 日志：`TrainingArguments.report_to="none"`，无需安装 TensorBoard 即可训练。
- 产出：`train/outputs/<时间戳>/` 下保存 `lora_adapter/`，并写入 `latest_lora_*.txt` 指针供评测读取。

**3. 评测**

- 评测集默认 `data/processed/KodCode_eval.jsonl`，用 vLLM 多次采样生成，再通过 `validate_code_with_test` 统计 pass@k，结果写入 `evaluation/results/`。

**4. 一键跑通（Linux / Git Bash / WSL）**

```bash
bash scripts/train_eval.sh
```

在 `scripts/train_eval.sh` 顶部按需打开 `RUN_H1 ~ RUN_H10`，以及三个训练开关 `RUN_TRAIN_CODE / RUN_TRAIN_COT_ZS / RUN_TRAIN_COT_FS`；`FS_K` 控制 few-shot 示例数；其余超参数在各训练/评测脚本顶部设置。

---

## 实验矩阵（H1 ~ H10）

### 1. 设计框架

整体采用 **训练侧 × 推理侧** 两因素析因设计：

- **训练侧**（LoRA 适配器类型）：`Base`、`LoRA_code`、`LoRA_cot_zs`、`LoRA_cot_fs(k)`
- **推理侧**（Prompt 结构）：`Direct`、`CoT-ZS`、`CoT-FS(k)`

理论组合数为 4 × 3 = 12。其中加入一条**合法性约束**：

> 若训练侧与推理侧均为 CoT（即训练侧 ∈ {LoRA_cot_zs, LoRA_cot_fs(k)}，推理侧 ∈ {CoT-ZS, CoT-FS(k)}），
> 则两侧的示例数 k 必须相等。

该约束的目的是保持 train/test 上下文分布一致，避免"训练见过 0 例、推理给 k 例"或反之带来的分布漂移干扰。据此排除 2 组非法组合，剩余 **10 组有效实验（H1 ~ H10）**。

### 2. 术语约定

| 符号 | 含义 | 产生方式 |
|---|---|---|
| `Base` | 不加 LoRA 的原始 CodeGeeX4-ALL-9B | — |
| `LoRA_code` | 仅监督 `solution`（纯代码）的 LoRA | `train/train.py` → `latest_lora_adapter.txt` |
| `LoRA_cot_zs` | 监督 `thought_step + solution` 且训练 prompt **不含** few-shot 示例 | `train/train_cot.py --num-few-shots 0` → `latest_lora_cot_zs_adapter.txt` |
| `LoRA_cot_fs(k)` | 监督 `thought_step + solution` 且训练 prompt **含 k 条** few-shot 示例 | `train/train_cot.py --num-few-shots k` → `latest_lora_cot_fs{k}_adapter.txt` |
| `Direct` | 单阶段 prompt：题目 → 代码 | `eval_lora_passk.py` / `eval_base_passk.py` |
| `CoT-ZS` | 两阶段 prompt：先生成推理再生成代码，0 示例 | `eval_lora_cot_passk.py --few-shot-k 0` |
| `CoT-FS(k)` | 在 CoT-ZS 基础上前置 k 条 `(题目, 推理, 代码)` 示例（按 `strong_tags` 检索） | `eval_lora_cot_passk.py --few-shot-k k` |

### 3. 十组实验明细

| 编号 | 训练侧 | 推理侧 | LoRA adapter 指针 | 评测命令（摘要） |
|---|---|---|---|---|
| **H1** | Base | Direct | — | `eval_base_passk.py` |
| **H2** | Base | CoT-ZS | — | `eval_lora_cot_passk.py --lora-path NONE --few-shot-k 0` |
| **H3** | Base | CoT-FS(k) | — | `eval_lora_cot_passk.py --lora-path NONE --few-shot-k k` |
| **H4** | LoRA_code | Direct | `latest_lora_adapter.txt` | `eval_lora_passk.py` |
| **H5** | LoRA_code | CoT-ZS | `latest_lora_adapter.txt` | `eval_lora_cot_passk.py --lora-path <code adapter> --few-shot-k 0` |
| **H6** | LoRA_code | CoT-FS(k) | `latest_lora_adapter.txt` | `eval_lora_cot_passk.py --lora-path <code adapter> --few-shot-k k` |
| **H7** | LoRA_cot_zs | Direct | `latest_lora_cot_zs_adapter.txt` | `eval_lora_passk.py --lora-path <cot_zs adapter>` |
| **H8** | LoRA_cot_zs | CoT-ZS | `latest_lora_cot_zs_adapter.txt` | `eval_lora_cot_passk.py --few-shot-k 0` |
| **H9** | LoRA_cot_fs(k) | Direct | `latest_lora_cot_fs{k}_adapter.txt` | `eval_lora_passk.py --lora-path <cot_fs adapter>` |
| **H10** | LoRA_cot_fs(k) | CoT-FS(k) | `latest_lora_cot_fs{k}_adapter.txt` | `eval_lora_cot_passk.py --few-shot-k k` |

**被约束排除的非法组合**：

- ~~LoRA_cot_zs + CoT-FS(k>0)~~ ：训练 k=0 与推理 k>0 不一致
- ~~LoRA_cot_fs(k) + CoT-ZS~~    ：训练 k>0 与推理 k=0 不一致

### 4. 关键对照与要回答的问题

**(a) 训练侧主效应**（推理侧固定为 Direct）

| 对比 | 结论 |
|---|---|
| H4 − H1 | 纯代码 LoRA 的训练侧主效应 |
| H7 − H1 | CoT 格式监督（无 few-shot）的训练侧主效应 |
| H9 − H1 | CoT 格式监督（含 k 例 few-shot）的训练侧主效应 |
| H7 − H4 | "加入思维链监督" vs "仅代码监督" 的增益 |
| H9 − H7 | **"训练时加入 few-shot 上下文"** 的独立增益（本设计相对 8 组版本新增的核心对照） |

**(b) 推理侧主效应**（训练侧固定为 Base）

| 对比 | 结论 |
|---|---|
| H2 − H1 | CoT-ZS 推理侧主效应（不依赖任何微调） |
| H3 − H1 | CoT-FS(k) 推理侧主效应 |
| H3 − H2 | Few-shot 示例在基座上的边际增益 |

**(c) 训练 × 推理 交互效应（"正交收益"判据）**

| 对比 | 结论 |
|---|---|
| (H8 − H7) − (H2 − H1) | LoRA_cot_zs 与 CoT-ZS 推理是否**正交可加** |
| (H10 − H9) − (H3 − H1) | LoRA_cot_fs(k) 与 CoT-FS(k) 推理是否**正交可加**（匹配 k） |
| (H5 − H4) − (H2 − H1) | CoT-ZS 推理在 LoRA_code 上的边际 vs 在 Base 上的边际 |
| (H6 − H4) − (H3 − H1) | CoT-FS 推理在 LoRA_code 上的边际 vs 在 Base 上的边际 |

**(d) Few-shot 边际效应的一致性**（本文段落核心设问）

| 对比 | 结论 |
|---|---|
| H3 − H2（基座）vs H6 − H5（LoRA_code） | few-shot 在**未监督 CoT** 的两种训练侧是否一致 |
| H10 − H8（匹配 k 的 LoRA_cot） | 训练推理 k 同步放大时 CoT 的净收益 |

**(e) 协同设计的核心结论（本文段落要回答）**

- **正交性**：(H8 − H7) − (H2 − H1) ≈ 0 → 正交可加；显著 > 0 → LoRA 放大 CoT；显著 < 0 → LoRA 吸收了 CoT 的部分收益。
- **Few-shot 一致性**：`H3 − H2` 与 `H6 − H5` 两条边际若符号与量级一致，说明 few-shot 边际不依赖训练侧；否则说明训练侧会调节 few-shot 的有效性。
- **训练/推理的 CoT 等价性**：若 `H9 ≳ H10` 且 `H7 ≳ H8`，说明"把 CoT 监督进参数"已经足以替代"推理时显式走 CoT 两阶段"；反之则两条路径互补。

### 5. 结果文件命名

各组结果统一写入 `evaluation/results/`：

| 编号 | 文件名 |
|---|---|
| H1 | `eval_base_passk_h1_base_direct.json`（`--result-suffix h1_base_direct`；省略后缀时为 `eval_base_passk.json`） |
| H2 | `eval_lora_cot_passk_h2_base_cot_zs.json` |
| H3 | `eval_lora_cot_passk_h3_base_cot_fs{k}.json` |
| H4 | `eval_lora_passk_h4_lora_code_direct.json` |
| H5 | `eval_lora_cot_passk_h5_lora_code_cot_zs.json` |
| H6 | `eval_lora_cot_passk_h6_lora_code_cot_fs{k}.json` |
| H7 | `eval_lora_passk_h7_lora_cot_zs_direct.json` |
| H8 | `eval_lora_cot_passk_h8_lora_cot_zs_cot_zs.json` |
| H9 | `eval_lora_passk_h9_lora_cot_fs{k}_direct.json` |
| H10 | `eval_lora_cot_passk_h10_lora_cot_fs{k}_cot_fs{k}.json` |

每个 CoT 评测的 JSON 额外记录 `mode / use_lora / lora_path / few_shot_k`，方便后续追溯。

### 6. 运行建议

**训练阶段**（先跑一次，之后可复用）：

1. `RUN_TRAIN_CODE=1`    → 产出 `LoRA_code`（H4/H5/H6 依赖）
2. `RUN_TRAIN_COT_ZS=1`  → 产出 `LoRA_cot_zs`（H7/H8 依赖）
3. `RUN_TRAIN_COT_FS=1`  → 产出 `LoRA_cot_fs(k)`（H9/H10 依赖，需与 `FS_K` 一致）

**评测阶段**（按设问重要性排序）：

1. 主干闭合：H1 / H4 / H7 / H2 / H8（最少 5 组即可得到 LoRA 主效应、CoT 主效应、LoRA×CoT 交互项）。
2. Few-shot 边际：补 H3 / H5 / H6 / H9 / H10（得到 few-shot 一致性与 k 匹配时的协同）。
3. 若算力充足，全部 10 组一次性跑完；若紧张，按上述两步分批。

**统计稳健性**：评测脚本现已支持 `--num-samples` 可调，建议固定 ≥2 个随机 seed 复跑，结果报均值 ± std。

---

## 依赖

`requirements.txt` 为某台机器上的完整 **`pip freeze`**（含 PyTorch CUDA 版、vLLM 等），换电脑安装可能需按 PyTorch / vLLM 文档调整索引或版本，见 **SETUP.md**。
