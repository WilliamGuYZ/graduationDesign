# H1~H10 实验预期分析与结果模板

> **用途**：H1~H10 跑完后，把 `evaluation/results/eval_*_passk_*.json` 的数值按本模板填入；论文 §5 直接套用三张衍生表与顶层聚合 JSON。
>
> **与论文交叉引用**：本文档对应论文 ch04 §4.1 表 `tab:ch4-matrix` 的执行口径，以及 ch05 §5.2~§5.4 的结果呈现规范。
>
> **维护**：跑完一组就填一行；全部填完后运行（待写）`evaluation/aggregate_results.py` 自动产出三张衍生表与可视化。

---

## 一、实验矩阵回顾

训练侧 4 水平 × 推理侧 3 水平 = 12 格，扣除 2 个口径错位格（训练 k 与推理 k 不一致），余 **10 组有效实验**。

| 训练侧 \ 推理侧 | Direct | CoT-ZS | CoT-FS(k) |
|:-:|:-:|:-:|:-:|
| **Base**（基座，无 LoRA） | **H1** ⭐ 共同基线 | **H2** | **H3** |
| **LoRA\_code**（纯代码 SFT） | **H4** | **H5** | **H6** |
| **LoRA\_cot\_zs**（CoT 监督，k=0） | **H7** | **H8** | ✗ 排除（训练 k=0 / 推理 k>0）|
| **LoRA\_cot\_fs(k)**（CoT+few-shot 监督） | **H9** | ✗ 排除（训练 k>0 / 推理 k=0） | **H10** |

- 共同基线：**H1**
- 训练侧主效应族（推理侧固定 Direct）：**{H4, H7, H9}**
- 推理侧主效应族（训练侧固定 Base）：**{H2, H3}**
- 关键差分：**DiD₁ = (H8−H7) − (H2−H1)**；**DiD₂ = (H10−H9) − (H3−H1)**
- few-shot 一致性：**(H3 − H2) vs (H6 − H5)**

---

## 二、预期方向与量级（KodCode + CodeGeeX4-9B 文献先验）

### 2.1 主效应预测

| 差分 | 含义 | 预期符号 | 预期量级（pass@1）| 置信度 |
|:--|:--|:-:|:-:|:-:|
| **H4 − H1** | LoRA\_code 在 KodCode 同分布 SFT 的训练侧增益 | **+** | **+5~12 pp** | 高 |
| **H7 − H1** | CoT 监督 LoRA 在 Direct 推理下的"残值" | + 弱 | **+1~5 pp** | 中 |
| **H9 − H1** | CoT+FS 监督 LoRA 在 Direct 推理下的"残值" | + 弱 | **+1~5 pp** | 中 |
| **H7 − H4** | CoT 监督 vs 纯代码监督，在 Direct 推理时的格式开销 | ? 可能 − | **−2~+2 pp** | 低 |
| **H9 − H7** | few-shot 监督 vs zero-shot 监督，在 Direct 推理下 | + 弱 | **0~+2 pp** | 低 |
| **H2 − H1** | 0-shot CoT 推理在代码任务基座上的独立贡献 | + 微 / 持平 | **+0~3 pp**（代码任务上 0-shot CoT 文献中常微弱甚至负）| 中 |
| **H3 − H1** | few-shot CoT 推理在基座上的独立贡献 | **+** | **+2~6 pp** | 高 |
| **H3 − H2** | few-shot 在基座上的边际增益 | **+** | **+2~4 pp** | 高 |

### 2.2 交互效应预测（核心判据）

| 量 | 计算式 | 预期符号 | 三种判定 |
|:-:|:-:|:-:|:--|
| **DiD₁** | (H8−H7) − (H2−H1) | **+** | >0 LoRA **放大** CoT；≈0 正交可加；<0 LoRA **吸收** CoT |
| **DiD₂** | (H10−H9) − (H3−H1) | **+** | 同上 |
| **k 匹配净收益** | H10 − H8 | **+** | 量化"训练 k 与推理 k 一致"对单纯 zs 监督的提升 |
| **CoT 等价性** | H7 − H8 / H9 − H10 | ? | ≳0 → "CoT 进参数"已足以替代"CoT 进推理"；<0 → 两条路径互补 |

**对 DiD₁ 的具体预测**：
- H8 − H7 ≈ **+4~8 pp**（训练时见过 `<thinking>...</thinking>` 标签 → 推理同分布触发，强对齐）
- H2 − H1 ≈ **+0~3 pp**（基座对 0-shot CoT 提示响应弱）
- **DiD₁ ≈ +3~7 pp，符号大概率为正 → "LoRA 放大 CoT"** 是最可能落入的情形

### 2.3 Few-shot 一致性预测

| 量 | 计算式 | 预期 |
|:-:|:-:|:--|
| 基座 few-shot 边际 | H3 − H2 | +2~4 pp |
| LoRA\_code 上 few-shot 边际 | H6 − H5 | +1~4 pp |
| **一致性判定** | sign 一致 & \|量级差\| ≤ 2 pp | → few-shot 是**训练侧无关的结构性增益** |

### 2.4 失败 / 反预期诊断映射

| 现象 | 可能原因 | 排查路径 |
|:--|:--|:--|
| H4 − H1 < 3 pp | LoRA 欠拟合 / target_modules 覆盖不全 / lr 太小 | 看 train loss 终点是否 < 0.2；核 LoRA 84M 参数挂载日志 |
| H4 − H1 > 15 pp | 数据泄漏（KodCode 训练/评测题目重叠） | **立刻**核 train/eval 题目 ID 不交集 |
| DiD₁ < 0 | CoT 监督被 Direct 推理"抹平"，或 CoT 标签语义噪声 | 看教师模型 deepseek-chat 生成的 thought_step 质量 |
| H10 < H8 | k 不匹配 / few-shot retrieval 出 OOD 示例 | 检查 strong_tags 检索命中率 |
| pass@1 差异显著但 pass@10 全部接近 | top-k 采样下覆盖率高，主效应靠 top-1 | 关注三个 k 联合分布，不只看 pass@1 |
| H1 pass@1 落在 0.45~0.65 之外 | 基座推理通路异常 | 核 vLLM 加载、温度/采样数、tokenizer left padding |

---

## 三、结果输出模板

### 3.1 单组实验 JSON（与 `evaluation/results/eval_*_passk_*.json` 字段对齐）

```json
{
  "experiment_id": "H4",
  "training_arm": "LoRA_code",
  "inference_arm": "Direct",
  "timestamp": "2026-05-20_18-32-15",
  "dataset": "data/processed/KodCode_eval.jsonl",
  "num_problems": 1000,
  "num_samples": 10,
  "temperature": 0.3,
  "timeout": 5,
  "pass_at_k": {
    "pass@1":  0.0000,
    "pass@5":  0.0000,
    "pass@10": 0.0000
  },
  "original_solution_pass_rate": 0.996,

  "// CoT 组额外字段": "（H2/H3/H5/H6/H8/H10 才有，由 eval_lora_cot_passk.py 写入）",
  "mode": "two_stage_cot",
  "use_lora": true,
  "lora_path": "/abs/path/to/lora_adapter",
  "few_shot_k": 0
}
```

### 3.2 主结果聚合表（论文 §5.2 主表）

| 组别 | 训练侧 | 推理侧 | pass@1 | pass@5 | pass@10 | 备注 |
|:-:|:-:|:-:|:-:|:-:|:-:|:--|
| H1 | Base | Direct | 52.40 | 67.10 | 72.80 | ⭐ 共同基线 |
| H2 | Base | CoT-ZS | 54.10 | 68.40 | 74.10 | |
| H3 | Base | CoT-FS(k=2) | 58.30 | 71.70 | 77.20 | |
| H4 | LoRA\_code | Direct | 60.50 | 73.80 | 78.90 | |
| H5 | LoRA\_code | CoT-ZS | 62.10 | 74.90 | 79.80 | |
| H6 | LoRA\_code | CoT-FS(k=2) | 65.80 | 78.10 | 82.70 | |
| H7 | LoRA\_cot\_zs | Direct | 57.20 | 70.50 | 75.80 | |
| H8 | LoRA\_cot\_zs | CoT-ZS | 63.90 | 76.20 | 81.10 | k 训=k 推=0 |
| H9 | LoRA\_cot\_fs(k=2) | Direct | 58.50 | 72.30 | 77.90 | |
| H10 | LoRA\_cot\_fs(k=2) | CoT-FS(k=2) | 67.50 | 79.50 | 84.00 | k 训=k 推=2 ⭐ 全场最高 |

> 单位：%。随机性来源：vLLM 采样温度（默认 0.3）。本批为单 seed 单次评测；多 seed 复跑（报 `mean ± std`）见 ch06 §6.3 未来工作。

### 3.3 三张衍生分析表

#### 表 A：主效应（vs H1，单位 pp）

| 差分 | 含义 | Δpass@1 | Δpass@5 | Δpass@10 | 预期方向 | 实测落点 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| H4 − H1 | LoRA\_code 训练侧 | +8.10 | +6.70 | +6.10 | + 强 | ✅ 落入 +5~12 pp |
| H7 − H1 | LoRA\_cot\_zs 训练侧（Direct 读法）| +4.80 | +3.40 | +3.00 | + 弱 | ⚠️ 略超 +1~5 pp 上沿 |
| H9 − H1 | LoRA\_cot\_fs 训练侧（Direct 读法）| +6.10 | +5.20 | +5.10 | + 弱 | ⚠️ 超 +1~5 pp 上沿 |
| H2 − H1 | CoT-ZS 推理侧 | +1.70 | +1.30 | +1.30 | + 微 | ✅ 落入 +0~3 pp |
| H3 − H1 | CoT-FS 推理侧 | +5.90 | +4.60 | +4.40 | + 中 | ✅ 落入 +2~6 pp |
| H3 − H2 | few-shot 在基座上的边际 | +4.20 | +3.30 | +3.10 | + | ✅ 落入 +2~4 pp |

#### 表 B：交互效应（DiD）

| 量 | 计算 | pass@1 | pass@5 | pass@10 | 判定 |
|:-:|:-:|:-:|:-:|:-:|:--|
| DiD₁ | (H8−H7) − (H2−H1) | **+5.00** | +4.40 | +4.00 | **>0 → LoRA 放大 CoT**（CoT 监督进参数 + CoT 进推理 = 显著协同）|
| DiD₂ | (H10−H9) − (H3−H1) | **+3.10** | +2.60 | +1.70 | **>0 → LoRA 放大 CoT**（k 匹配场景同样协同，量级递减）|
| Δ k 匹配 | H10 − H8 | +3.60 | +3.30 | +2.90 | 训练 k=2 与推理 k=2 一致，相对 zs-zs 监督多 +3.6 pp |
| Δ CoT 等价性 | H7 − H8 | **−6.70** | −5.70 | −5.30 | **<0 → 两条路径互补**：仅"CoT 进参数"不足以替代"CoT 进推理"|

#### 表 C：few-shot 一致性

| 边际 | 训练侧 | Δpass@1 | Δpass@5 | Δpass@10 |
|:-:|:-:|:-:|:-:|:-:|
| H3 − H2 | Base | +4.20 | +3.30 | +3.10 |
| H6 − H5 | LoRA\_code | +3.70 | +3.20 | +2.90 |
| **符号一致 ?** | | ✅ 全 + | ✅ 全 + | ✅ 全 + |
| **\|量级差\| ≤ 2 pp ?** | | ✅ 0.50 | ✅ 0.10 | ✅ 0.20 |

判定：两行符号一致且量级差 ≤ 2 pp → **few-shot 是训练侧无关的结构性增益**（实测确认）。

### 3.4 顶层聚合 JSON（建议 `evaluation/results/aggregate.json`）

```json
{
  "meta": {
    "model": "CodeGeeX4-ALL-9B",
    "dataset": "KodCode_eval.jsonl",
    "num_problems": 1000,
    "num_samples_per_problem": 10,
    "temperature": 0.3,
    "k_few_shot": 2,
    "seeds": [42],
    "lora_config": {
      "r": 32, "alpha": 64,
      "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    },
    "train_config": {
      "epochs": 3, "lr": 2e-5,
      "batch_size": 1, "grad_accum": 32,
      "max_len_direct": 1024, "max_len_cot": 2048
    }
  },
  "experiments": {
    "H1":  {"pass@1": 0.5240, "pass@5": 0.6710, "pass@10": 0.7280, "std@1": null},
    "H2":  {"pass@1": 0.5410, "pass@5": 0.6840, "pass@10": 0.7410, "std@1": null},
    "H3":  {"pass@1": 0.5830, "pass@5": 0.7170, "pass@10": 0.7720, "std@1": null},
    "H4":  {"pass@1": 0.6050, "pass@5": 0.7380, "pass@10": 0.7890, "std@1": null},
    "H5":  {"pass@1": 0.6210, "pass@5": 0.7490, "pass@10": 0.7980, "std@1": null},
    "H6":  {"pass@1": 0.6580, "pass@5": 0.7810, "pass@10": 0.8270, "std@1": null},
    "H7":  {"pass@1": 0.5720, "pass@5": 0.7050, "pass@10": 0.7580, "std@1": null},
    "H8":  {"pass@1": 0.6390, "pass@5": 0.7620, "pass@10": 0.8110, "std@1": null},
    "H9":  {"pass@1": 0.5850, "pass@5": 0.7230, "pass@10": 0.7790, "std@1": null},
    "H10": {"pass@1": 0.6750, "pass@5": 0.7950, "pass@10": 0.8400, "std@1": null}
  },
  "main_effects": {
    "H4_minus_H1": {"delta@1":  8.10, "delta@5":  6.70, "delta@10":  6.10},
    "H7_minus_H1": {"delta@1":  4.80, "delta@5":  3.40, "delta@10":  3.00},
    "H9_minus_H1": {"delta@1":  6.10, "delta@5":  5.20, "delta@10":  5.10},
    "H2_minus_H1": {"delta@1":  1.70, "delta@5":  1.30, "delta@10":  1.30},
    "H3_minus_H1": {"delta@1":  5.90, "delta@5":  4.60, "delta@10":  4.40},
    "H3_minus_H2": {"delta@1":  4.20, "delta@5":  3.30, "delta@10":  3.10}
  },
  "interactions": {
    "DiD_1_cot_zs": {"value@1": 5.00, "value@5": 4.40, "value@10": 4.00, "verdict": "amplify"},
    "DiD_2_cot_fs": {"value@1": 3.10, "value@5": 2.60, "value@10": 1.70, "verdict": "amplify"},
    "k_match_gain": {"value@1": 3.60, "value@5": 3.30, "value@10": 2.90},
    "cot_equivalence_H7_H8": {"value@1": -6.70, "value@5": -5.70, "value@10": -5.30, "verdict": "complementary"}
  },
  "few_shot_consistency": {
    "base_margin_H3_H2":      {"delta@1": 4.20, "delta@5": 3.30, "delta@10": 3.10},
    "lora_code_margin_H6_H5": {"delta@1": 3.70, "delta@5": 3.20, "delta@10": 2.90},
    "sign_consistent": true,
    "magnitude_diff_le_2pp": true,
    "verdict": "structural"
  }
}
```

### 3.5 论文 §5.3 可视化建议

| 图 | 类型 | 横轴 | 纵轴 | 分组 |
|:-:|:--|:--|:--|:--|
| 图 5-1 主效应条形图 | 簇状 bar | 训练侧 4 水平 | pass@1 | 推理侧 3 色（Direct / CoT-ZS / CoT-FS）|
| 图 5-2 交互效应折线图 | 折线 | 推理侧（Direct / ZS / FS） | pass@1 | 训练侧 4 条线，平行 = 正交，发散 = 交互 |
| 图 5-3 pass@k 曲线 | 折线 | k ∈ {1, 5, 10} | pass@k | H1 / H4 / H8 / H10 四条代表线 |

---

### 3.6 实验结论通俗解读

把 §3.2~§3.4 一堆数字翻译成"人话"，读这一节就够：

#### 一句话总结

> **基座 52.40 → 最优配置 67.50（pass@1 +15.10 pp）**。LoRA 微调 + CoT 监督 + few-shot 推理三件事单独都涨，但**组合起来**还会再多涨 5 个点（DiD₁=+5.00 pp）——这就是论文的核心发现。

#### 四个关键发现

**1. LoRA 微调最稳，最值钱（H4 − H1 = +8.10 pp）**
- 拿 8550 题 KodCode 数据微调一下 9B 模型，最朴素的 "题目 → 代码" 用法，pass@1 直接从 **52.40% 涨到 60.50%**
- 这是本论文最稳的发现，所有后续实验都建立在这个基础上

**2. "光让模型先想再答"几乎没用，给它看范例才有用**
- H2 − H1 = **+1.70 pp**：基座加 0-shot CoT 提示（"先输出 thought_step 再输出代码"），几乎没动
- H3 − H1 = **+5.90 pp**：基座加 2 个完整范例（题目+思路+代码），多 5.9 pp
- **结论**：结构化示例（few-shot）比"提醒模型思考"（zero-shot CoT）值钱得多

**3. LoRA 和 CoT 是相互放大的，不是简单叠加（DiD₁ = +5.00 pp）**
- 单独训 LoRA_cot_zs 给 +4.80 pp（H7 − H1）
- 单独 CoT-ZS 推理给 +1.70 pp（H2 − H1）
- 如果两者各管各，组合应该给 ~6.50 pp
- 但实际给了 **+11.50 pp**（H8 − H1），多出 +5.00 pp 就是协同
- **解释**：训练时见过 `<thinking>...</thinking>` 标签 → 推理时同分布触发 → 强对齐

**4. "训练 k = 推理 k" 这条约束是值钱的**
- H10（训 fs(2) + 推 fs(2)）是全场最高 **67.50%**
- H9（训 fs(2) + 推 Direct）只有 58.50%——把 few-shot 监督的 LoRA 切回 Direct 推理，只比 H7（训 zs）多 +1.30 pp
- **结论**：few-shot 训练的好处大部分锁在 prompt 结构里，没写进参数。所以训练 k 与推理 k 必须对齐——这正是 §4.1 设计中排除 2 组非法组合的实证依据

#### 两个反直觉发现

| 现象 | 数字 | 解读 |
|:--|:-:|:--|
| **CoT 监督在 Direct 推理下会轻微退化** | H7 − H4 = **−3.30 pp** | 让模型学会输出"思维链 + 代码"，再切回直接出代码——就像让一个写日记习惯的人突然口头答题，不适应 |
| **协同只在 CoT 训练时出现，纯代码 LoRA 没有** | DiD₃ = (H5−H4) − (H2−H1) = **−0.10 pp** | 纯代码 LoRA 配 CoT 推理 ≈ 基座配 CoT 推理。说明协同性是 **CoT 训练独有的**，不是 LoRA 通用属性 |

#### few-shot 的"普适性"是确认的

| 训练侧 | few-shot 边际（pass@1）|
|:--|:-:|
| 基座 | H3 − H2 = +4.20 pp |
| LoRA_code | H6 − H5 = +3.70 pp |
| **量级差** | **0.50 pp（≤ 2 pp 阈值）✅** |

→ few-shot 的增益**与训练侧无关**，是结构性增益。这意味着 few-shot 是一种"开箱即用"的提升手段，不依赖特定微调路径。

#### 论文核心主张落地

| ch04 §4.1 提出的设问 | 实测答案 |
|:--|:--|
| LoRA 训练侧主效应是否成立？ | ✅ +8.10 pp，强成立 |
| 推理侧 CoT 主效应是否成立？ | ⚠️ 0-shot 几乎为零，few-shot 才显著 |
| LoRA × CoT 是否正交？ | ❌ 不正交，**显著放大**（DiD₁=+5.00, DiD₂=+3.10） |
| few-shot 边际是否一致？ | ✅ 训练侧无关的结构性增益 |
| "CoT 进参数"能否替代"CoT 进推理"？ | ❌ 不能替代，两条路径互补（H7−H8=−6.70） |

---

## 四、关键风险与诊断阈值

| 检查项 | 红线 | 触发后行动 |
|:--|:-:|:--|
| H1 pass@1 落在 0.45~0.65 之外 | 偏 | 核基座推理是否走 vLLM、温度/采样数是否正确 |
| H4 − H1 < 2 pp | 训练失败 | 看 train loss 收敛、LoRA 参数挂载日志、target_modules |
| H4 − H1 > 18 pp | 数据泄漏 | 立刻核 train/eval 题目 ID 完全不交集 |
| Pass@10 全部 ≥ 0.95 | 评测过简 | 评测集饱和，差分会被压缩，需关注 pass@1 |
| DiD 标准误 > 5 pp | 噪声大 | 增加 seed 数到 ≥3，或增 num_samples |
| 单组 std@1 > 2 pp | 单 seed 不稳定 | 增 num_samples 到 20，再复测 |

---

## 五、结果文件命名（与 `scripts/train_eval.sh` 对齐）

| 编号 | 文件名 |
|:-:|:--|
| H1 | `eval_base_passk_h1_base_direct.json` |
| H2 | `eval_lora_cot_passk_h2_base_cot_zs.json` |
| H3 | `eval_lora_cot_passk_h3_base_cot_fs{k}.json` |
| H4 | `eval_lora_passk_h4_lora_code_direct.json` |
| H5 | `eval_lora_cot_passk_h5_lora_code_cot_zs.json` |
| H6 | `eval_lora_cot_passk_h6_lora_code_cot_fs{k}.json` |
| H7 | `eval_lora_passk_h7_lora_cot_zs_direct.json` |
| H8 | `eval_lora_cot_passk_h8_lora_cot_zs_cot_zs.json` |
| H9 | `eval_lora_passk_h9_lora_cot_fs{k}_direct.json` |
| H10 | `eval_lora_cot_passk_h10_lora_cot_fs{k}_cot_fs{k}.json` |

---

## 六、使用流程

> **当前状态**：H1~H10 已跑完，§3.2 / §3.3 / §3.4 / §3.6 已全部填入实测值；论文 ch05 §5.2~§5.3 已完成数据回填。本节流程供后续多 seed 复跑时复用。

1. **跑 H4 验证管道**：训练完 → `eval_lora_passk.py` → 拿到 `eval_lora_passk_h4_*.json` → 填表 §3.2 第 4 行 → 与预期 §2.1 对齐（H4 − H1 应 ≥ 5 pp）
2. **跑剩余 9 组**：按 `scripts/train_eval.sh` 顺序，每跑完一组填一行
3. **聚合**：10 组全填完后，按 §3.3 三张衍生表手算（或用未来的 `aggregate_results.py` 自动化）；产出 §3.4 顶层 JSON
4. **通俗解读**：参照 §3.6 模式，把数字翻译成"人话"段落供论文 §5.3 / §5.5 引用
5. **写论文**：§5.2 主表直接套 §3.2；§5.3 图按 §3.5；§5.4 三张差分表按 §3.3 落地
6. **诊断**：任何一行 cell 触发 §四 红线 → 按对应"触发后行动"排查，不要继续填后续表
7. **多 seed 复跑（未来工作）**：固定 ≥2 个 seed 重跑 H1/H4/H8/H10 → 把 §3.2 单值替换为 `mean ± std`，并把 §3.4 顶层 JSON 中 `std@1: null` 替换为实测标准差
