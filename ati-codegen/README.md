## ati-codegen（毕设代码框架）

面向你的选题“**基于大语言模型的算法代码生成**”的研究型工程骨架，覆盖：

- **数据集构建**：LeetCode/自建题库 → 指令数据（Instruction/Input/Output）JSONL
- **微调**：LoRA（PEFT）+ Transformers
- **推理增强**：CoT（可扩展到 RAG / ATI：问题解析 + 模板检索 + 实例化）
- **评测**：按测试用例执行并计算 **pass@k**

### 目录结构

```text
ati-codegen/
  configs/                # 配置（训练/推理/评测）
  data/
    raw/                  # 原始数据（题面、样例等）
    processed/            # 处理后的数据集（jsonl）
  scripts/                # 命令行入口脚本
  src/ati_codegen/        # 主包：数据/模板/模型/评测
  templates/              # 算法模板库（可人工或自动扩展）
  tests/                  # 单元测试（最小覆盖）
```

### 快速开始（最小闭环）

1) 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) 生成一份示例数据集（用于跑通流程）

```bash
python scripts/make_mock_dataset.py --out data/processed/mock.jsonl
```

3) 用“占位模型”跑一次生成与评测（不需要 GPU）

```bash
python scripts/eval_passk.py --dataset data/processed/mock.jsonl --k 1
```
### 第3-4周：LeetCode 数据采集与预处理（JSONL）

我们提供了 `leetcode.cn` 的 GraphQL 抓取脚本，把题面清洗成纯文本，并按语言输出函数签名（`codeSnippets`），最终落盘 JSONL。

```bash
python scripts/build_dataset_leetcode_cn.py ^
  --out data/processed/leetcode-cn.jsonl ^
  --limit 200 ^
  --difficulty MEDIUM ^
  --languages python,java ^
  --prefer-translated
```

如果遇到 403/风控，把浏览器里 `leetcode.cn` 的 Cookie 复制出来传入：

```bash
python scripts/build_dataset_leetcode_cn.py --out data/processed/leetcode-cn.jsonl --limit 50 --cookie "<YOUR_COOKIE>"
```

注意：LeetCode 公开接口拿不到隐藏测试用例，因此导出的 `tests` 字段默认为空。你后续可以：
- 用开源数据集（如 HumanEval-X）做可执行评测
- 或按你的开题方案自建 `TemplateLeetcode`：自己维护题目测试（推荐）

### 下一步你需要接入的真实内容

- **数据**：把你抓取/整理的 LeetCode 数据落到 `data/raw/`，再写/补 `scripts/build_dataset.py`
- **模型**：在 `configs/` 填你的基座模型（如 Qwen2.5-Coder / CodeGeeX / Llama）
- **LoRA 微调**：跑 `scripts/train_lora.py`（骨架已给出参数与入口）
- **ATI/RAG**：在 `src/ati_codegen/retrieval/` 与 `src/ati_codegen/templates/` 扩展检索与模板实例化策略

