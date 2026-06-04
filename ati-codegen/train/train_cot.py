"""
CodeGeeX4-ALL-9B LoRA 微调脚本 —— Few-Shot CoT 版
数据格式: {"question": "...", "solution": "...", "thought_step": "...", "test": "...", "test_info": [...], "tags": [...]}

Few-Shot CoT 训练策略
    - 每条训练样本的 prompt 中随机插入 NUM_FEW_SHOTS 条示范样例（question → thought + code）
    - Loss 只计算在最后一个 assistant 回复（即当前样本的 thought + code）上
    - 模型由此同时学习「先推理后编码」的格式，以及具体的算法知识
"""

import argparse
import functools
import inspect
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


# ==================== 配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "CodeGeeX4-ALL-9B")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "KodCode_train.jsonl")
OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "train", "outputs")

# few-shot 示范数量（实验 C=0，实验 D>0）
NUM_FEW_SHOTS = 0

MAX_LENGTH = 2048  # 序列上限；CoT zero/few-shot 在 KodCode 上 p95 < 2048；32GB 卡使用 4096 会反向 OOM；如需要可 --max-length 显式覆盖

# LoRA 配置
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "query_key_value",                 # ChatGLM 融合 QKV
    "dense",                           # attention 输出
    "dense_h_to_4h", "dense_4h_to_h",  # MLP 两侧
]

# 训练配置
BATCH_SIZE = 1                 # 单卡 batch；CoT 序列更长（默认 MAX_LENGTH=2048），32GB 卡须保持 1
GRADIENT_ACCUMULATION = 32     # 等效 batch = 1 * 32 = 32（与 train.py 对齐，可复现性不变）
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50

SEED = 42
# =============================================


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"加载了 {len(data)} 条数据")
    return data


def _first_test_info(example: Dict[str, Any]) -> Dict[str, Any]:
    ti = example.get("test_info")
    if not ti or not isinstance(ti, list) or not isinstance(ti[0], dict):
        raise ValueError("样本缺少有效的 test_info[0]（需含 function_name、parameter_list）")
    return ti[0]


def build_instruction(example):
    """构建用户侧 instruction（CoT 版：末尾要求逐步思考再输出代码；与 eval_lora_cot_passk 中题目指令一致）。"""
    question = example["question"].strip()
    ti0 = _first_test_info(example)
    func_name = ti0.get("function_name") or "solution"
    params = ti0.get("parameter_list", "")
    return (
        f"Implement the function `{func_name}` that takes {params} to solve:\n\n"
        f"{question}\n\n"
        f"Think step by step, then output the Python code."
    )


def build_cot_response(example):
    """将 thought_step 和 solution 拼接为 assistant 回复。"""
    thought = example.get("thought_step", "").strip()
    code    = example["solution"].strip()
    if thought:
        return f"{thought}\n\n```python\n{code}\n```"
    return f"```python\n{code}\n```"


def _as_str_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def _few_shot_score(target: Dict[str, Any], cand: Dict[str, Any]) -> int:
    """与 evaluation/eval_lora_cot_passk.py 共享评分：强标签权重 2，普通标签权重 1。"""
    t_strong = set(_as_str_list(target.get("strong_tags")))
    c_strong = set(_as_str_list(cand.get("strong_tags")))
    t_tags   = set(_as_str_list(target.get("tags")))
    c_tags   = set(_as_str_list(cand.get("tags")))
    return 2 * len(t_strong & c_strong) + len(t_tags & c_tags)


def _select_few_shots_train(example: Dict[str, Any], pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    训练侧 few-shot 选池，与 4.3.1 (4) / 4.4.2 (1) 一致：
      1. 候选 = pool 扣除目标题；
      2. 按 score 降序；
      3. 取前 max(k, 4k) 作候选池；
      4. 池内 random.sample(k) 做随机性。
    """
    if k <= 0:
        return []
    candidates = [e for e in pool if e.get("question") != example.get("question")]
    if not candidates:
        return []
    scored = sorted(candidates, key=lambda x: _few_shot_score(example, x), reverse=True)
    pool_size = min(len(scored), max(k, 4 * k))
    top_pool  = scored[:pool_size]
    if len(top_pool) <= k:
        return top_pool
    return random.sample(top_pool, k)


def build_few_shot_messages(example, pool, num_few_shots: int):
    """
    构建 few-shot CoT 对话。

    消息结构：
        system
        user  (shot_1)  →  assistant (thought + code)
        user  (shot_2)  →  assistant (thought + code)
        ...
        user  (target)  →  assistant (thought + code)   ← 只对此段计算 loss
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. "
                "Before writing code, think through the problem step by step. "
                "Format your response as numbered reasoning steps followed by the Python solution."
            ),
        }
    ]

    # 按强标签评分选池后再随机抽取（与评测侧 _few_shot_score 共享评分函数）
    shots = _select_few_shots_train(example, pool, num_few_shots)

    for shot in shots:
        messages.append({"role": "user",      "content": build_instruction(shot)})
        messages.append({"role": "assistant", "content": build_cot_response(shot)})

    # 目标样本
    messages.append({"role": "user",      "content": build_instruction(example)})
    messages.append({"role": "assistant", "content": build_cot_response(example)})

    return {"messages": messages}


def tokenize_function(examples, tokenizer, max_length: int):
    """
    Tokenize 并设置 labels：仅对最后一条 assistant 回复段计算 loss。
    前缀（system + few-shot turns + user query）全部标记为 -100。
    """
    messages = examples["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages 须以 assistant 结尾")

    prefix_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=max_length,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
    )
    # ChatGLM4Tokenizer.apply_chat_template 即使单条输入也返回 batch 格式 [[...]]，扁平化为 flat list
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]

    # 定位 assistant 回复的起点
    i, lim = 0, min(len(prefix_ids), len(full_ids))
    while i < lim and prefix_ids[i] == full_ids[i]:
        i += 1
    assistant_start = i

    labels = (
        [-100] * len(full_ids)
        if assistant_start >= len(full_ids)
        else [-100] * assistant_start + full_ids[assistant_start:]
    )

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def _lora_pointer_file(num_few_shots: int) -> str:
    if num_few_shots == 0:
        filename = "latest_lora_cot_zs_adapter.txt"
    else:
        filename = f"latest_lora_cot_fs{num_few_shots}_adapter.txt"
    return os.path.join(PROJECT_ROOT, "train", filename)


def save_latest_adapter_path(adapter_path: str, num_few_shots: int) -> str:
    pointer_file = _lora_pointer_file(num_few_shots)
    os.makedirs(os.path.dirname(pointer_file), exist_ok=True)
    with open(pointer_file, "w", encoding="utf-8") as f:
        f.write(adapter_path + "\n")
    print(f"已保存最新适配器路径: {pointer_file}")
    print(f"路径内容: {adapter_path}")
    return pointer_file


def patch_accelerate_unwrap_model():
    """
    兼容旧版 accelerate：
    transformers 新版本会调用 unwrap_model(..., keep_torch_compile=False)，
    而旧 accelerate 的 unwrap_model 不接受该参数。
    """
    try:
        from accelerate import Accelerator
    except Exception:
        return

    unwrap = Accelerator.unwrap_model
    if "keep_torch_compile" in inspect.signature(unwrap).parameters:
        return

    def _unwrap_model_compat(self, model, *args, keep_torch_compile=None, **kwargs):
        return unwrap(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _unwrap_model_compat
    print("检测到旧版 accelerate，已应用 unwrap_model 兼容补丁。")


def main():
    patch_accelerate_unwrap_model()

    parser = argparse.ArgumentParser(description="训练 CoT LoRA（支持 few-shot 示例）")
    parser.add_argument(
        "--num-few-shots",
        type=int,
        default=NUM_FEW_SHOTS,
        help=f"每条样本前插入 few-shot 示例数（默认: {NUM_FEW_SHOTS}）",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help=f"训练 JSONL 路径；默认 {DATA_PATH}",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help=f"单条样本 tokenizer 截断上限（默认 {MAX_LENGTH}）；显存不足可试 2048",
    )
    args = parser.parse_args()
    if args.num_few_shots < 0:
        raise ValueError("--num-few-shots 不能为负数")
    num_few_shots = args.num_few_shots

    data_path = os.path.abspath(args.data_path.strip()) if args.data_path.strip() else DATA_PATH
    max_length = args.max_length if args.max_length > 0 else MAX_LENGTH
    if max_length < 512:
        raise ValueError("--max-length 过小，few-shot CoT 易被截断为无效 labels")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"未找到训练数据: {data_path}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"未找到基座模型目录: {MODEL_PATH}")
    mode_name = "CoT zero-shot (shots=0)" if num_few_shots == 0 else f"CoT few-shot (shots={num_few_shots})"
    training_mode = "cot_zero_shot" if num_few_shots == 0 else "cot_few_shot"

    random.seed(SEED)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"codegeex4_lora_{training_mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    raw_data = load_jsonl(data_path)

    # 过滤掉 thought_step 为空的条目（标注失败）
    before = len(raw_data)
    raw_data = [d for d in raw_data if d.get("thought_step", "").strip()]
    print(f"过滤空 thought_step：{before} → {len(raw_data)} 条")

    # 2. 构建 few-shot 消息
    print(f"[2/5] 构建对话格式（{mode_name}）...")
    formatted_data = [build_few_shot_messages(item, raw_data, num_few_shots) for item in raw_data]
    dataset = Dataset.from_list(formatted_data)

    # 3. 划分数据集
    print("[3/5] 划分训练/验证集...")
    split         = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(eval_dataset)} 条")

    # 4. 加载模型和分词器
    print("[4/5] 加载模型...")
    # ChatGLM4Tokenizer._pad 内部硬断言 padding_side == "left"（vendored 代码只实现了 left padding）。
    # 训练侧用 left padding 与 right padding 在数学上等价：loss 仅在 labels != -100 处计算，
    # PAD 同时被 attention_mask 屏蔽与 labels=-100 屏蔽，方向不影响梯度。
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 5. 配置 LoRA
    print("配置 LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    inner = getattr(getattr(model, "base_model", None), "model", None)
    if inner is not None and hasattr(inner, "config"):
        inner.config.use_cache = False
    model.print_trainable_parameters()

    # 6. Tokenize
    print("[5/5] Tokenizing...")
    tokenize_with_config = functools.partial(
        tokenize_function, tokenizer=tokenizer, max_length=max_length
    )

    train_dataset = train_dataset.map(
        tokenize_with_config,
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    eval_dataset = eval_dataset.map(
        tokenize_with_config,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
    )

    print(f"\n示例序列长度: {len(train_dataset[0]['input_ids'])}")
    train_ratio = 1 - train_dataset[0]["labels"].count(-100) / len(train_dataset[0]["labels"])
    print(f"有效训练 token 比例（最后 assistant 段）: {train_ratio:.1%}")

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
        logging_dir=f"{output_dir}/logs",
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        ),
    )

    # 8. 训练
    print(f"\n开始训练（{mode_name}）...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    # 9. 保存模型
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA 适配器保存至: {adapter_path}")

    # 10. 保存 latest_lora_adapter.txt
    pointer_file = save_latest_adapter_path(adapter_path, num_few_shots)

    # 11. 保存配置
    config = {
        "training_mode": training_mode,
        "num_few_shots": num_few_shots,
        "model_path": MODEL_PATH,
        "data_path": data_path,
        "max_length": max_length,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_time_seconds": train_time,
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 12. 打印总结
    print("\n" + "=" * 60)
    print(f"训练完成！（{mode_name}）")
    print("=" * 60)
    print(f"Few-Shot 示范数: {num_few_shots}")
    print(f"训练耗时: {train_time / 60:.2f} 分钟")
    print(f"输出目录: {output_dir}")
    print(f"LoRA 适配器: {adapter_path}")
    print(f"最新适配器指针: {pointer_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
