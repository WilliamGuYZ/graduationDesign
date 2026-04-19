"""
Qwen2.5-Coder-7B-Instruct LoRA 微调脚本 —— Few-Shot CoT 版
数据格式: {"question": "...", "solution": "...", "thought_step": "...", "test": "...", "test_info": [...], "tags": [...]}

Few-Shot CoT 训练策略
    - 每条训练样本的 prompt 中随机插入 NUM_FEW_SHOTS 条示范样例（question → thought + code）
    - Loss 只计算在最后一个 assistant 回复（即当前样本的 thought + code）上
    - 模型由此同时学习「先推理后编码」的格式，以及具体的算法知识
"""

import os
import json
import random
import time
from datetime import datetime

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

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-Coder-7B-Instruct")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "KodCode_train.jsonl")
LATEST_LORA_POINTER = os.path.join(PROJECT_ROOT, "train", "latest_lora_cot_adapter.txt")
OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "train", "outputs")

# few-shot 示范数量
NUM_FEW_SHOTS = 0

MAX_LENGTH = 1024

# LoRA 配置
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练配置
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
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


def build_instruction(example):
    """构建用户侧 instruction（与 train.py 保持一致）。"""
    question  = example["question"].strip()
    func_name = example["test_info"][0]["function_name"]
    params    = example["test_info"][0]["parameter_list"]
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


def build_few_shot_messages(example, pool):
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

    # 从 pool 中随机抽取 few-shot 示范（排除自身）
    candidates = [e for e in pool if e["question"] != example["question"]]
    shots = random.sample(candidates, min(NUM_FEW_SHOTS, len(candidates)))

    for shot in shots:
        messages.append({"role": "user",      "content": build_instruction(shot)})
        messages.append({"role": "assistant", "content": build_cot_response(shot)})

    # 目标样本
    messages.append({"role": "user",      "content": build_instruction(example)})
    messages.append({"role": "assistant", "content": build_cot_response(example)})

    return {"messages": messages}


def tokenize_function(examples, tokenizer):
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
        max_length=MAX_LENGTH,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # 定位 assistant 回复的起点
    i, lim = 0, min(len(prefix_ids), len(full_ids))
    while i < lim and prefix_ids[i] == full_ids[i]:
        i += 1
    assistant_start = i

    labels = (
        list(full_ids)
        if assistant_start >= len(full_ids)
        else [-100] * assistant_start + full_ids[assistant_start:]
    )

    # #region agent log - B: 检测序列截断
    import json as _j, time as _t, os as _o
    _DBG_LOG_T = "/home/ubuntu/yejunyin/graduationDesign/.cursor/debug-fc880c.log"
    _is_truncated = len(full_ids) >= MAX_LENGTH
    _train_token_ratio = (len(full_ids) - assistant_start) / len(full_ids) if full_ids else 0
    if not hasattr(tokenize_function, "_log_count"):
        tokenize_function._log_count = 0
    if tokenize_function._log_count < 10:
        tokenize_function._log_count += 1
        _o.makedirs(_o.path.dirname(_DBG_LOG_T), exist_ok=True)
        _entry = _j.dumps({"sessionId":"fc880c","timestamp":int(_t.time()*1000),"location":"train_cot.py:tokenize","message":"tokenize_sample","data":{"full_len":len(full_ids),"max_length":MAX_LENGTH,"is_truncated":_is_truncated,"assistant_start":assistant_start,"train_tokens":len(full_ids)-assistant_start,"train_ratio":round(_train_token_ratio,3)},"hypothesisId":"B"})
        with open(_DBG_LOG_T, "a") as _f:
            _f.write(_entry + "\n")
    # #endregion

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def save_latest_adapter_path(adapter_path):
    os.makedirs(os.path.dirname(LATEST_LORA_POINTER), exist_ok=True)
    with open(LATEST_LORA_POINTER, "w", encoding="utf-8") as f:
        f.write(adapter_path + "\n")
    print(f"已保存最新适配器路径: {LATEST_LORA_POINTER}")
    print(f"路径内容: {adapter_path}")


def main():
    random.seed(SEED)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"qwen_lora_cot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    raw_data = load_jsonl(DATA_PATH)

    # 过滤掉 thought_step 为空的条目（标注失败）
    before = len(raw_data)
    raw_data = [d for d in raw_data if d.get("thought_step", "").strip()]
    print(f"过滤空 thought_step：{before} → {len(raw_data)} 条")

    # 2. 构建 few-shot 消息
    print("[2/5] 构建 Few-Shot CoT 对话格式...")
    formatted_data = [build_few_shot_messages(item, raw_data) for item in raw_data]
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
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
    tokenize_with_config = lambda x: tokenize_function(x, tokenizer)

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
        report_to="tensorboard",
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
    print("\n开始训练（Few-Shot CoT）...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    # 9. 保存模型
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA 适配器保存至: {adapter_path}")

    # 10. 保存 latest_lora_adapter.txt
    save_latest_adapter_path(adapter_path)

    # 11. 保存配置
    config = {
        "training_mode": "few_shot_cot",
        "num_few_shots": NUM_FEW_SHOTS,
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "max_length": MAX_LENGTH,
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
    print("训练完成！（Few-Shot CoT）")
    print("=" * 60)
    print(f"Few-Shot 示范数: {NUM_FEW_SHOTS}")
    print(f"训练耗时: {train_time / 60:.2f} 分钟")
    print(f"输出目录: {output_dir}")
    print(f"LoRA 适配器: {adapter_path}")
    print(f"最新适配器指针: {LATEST_LORA_POINTER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
