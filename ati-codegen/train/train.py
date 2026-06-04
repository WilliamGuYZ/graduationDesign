"""
CodeGeeX4-ALL-9B LoRA 微调脚本
数据格式: {"question": "...", "solution": "...", "thought_step": "...", "test": "...", "test_info": [...], "tags": [...]}
所有字段均已确认不为空
"""

import argparse
import functools
import inspect
import json
import os
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

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "CodeGeeX4-ALL-9B")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "KodCode_train.jsonl")
LATEST_LORA_POINTER = os.path.join(PROJECT_ROOT, "train", "latest_lora_adapter.txt")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "train", "outputs")

MAX_LENGTH = 1024

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
# 等效 batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 32
# 32GB 单卡（V100 等）：9B + LoRA + bf16 + grad-ckpt 后 BATCH_SIZE 须 ≤ 1，否则激活 OOM
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 32
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
    """加载 JSONL 数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"加载了 {len(data)} 条数据")
    return data


def _first_test_info(example: dict) -> dict:
    """取出 test_info[0]；缺失时抛出明确错误，避免 KeyError。"""
    ti = example.get("test_info")
    if not ti or not isinstance(ti, list) or not isinstance(ti[0], dict):
        raise ValueError("样本缺少有效的 test_info[0]（需含 function_name、parameter_list）")
    return ti[0]


def build_instruction(example):
    """
    构建 instruction（Direct / 仅代码 LoRA；与 evaluation 中 Direct 模板一致）。
    """
    question = example["question"].strip()
    ti0 = _first_test_info(example)
    func_name = ti0.get("function_name") or "solution"
    params = ti0.get("parameter_list", "")
    
    return f"""Implement the function `{func_name}` that takes {params} to solve:

{question}

Only output the Python code, no explanation."""


def build_messages(example):
    """构建 chat messages 列表（由 tokenizer.apply_chat_template 转换为模型所需格式）"""
    instruction = build_instruction(example)
    code = example["solution"].strip()
    
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write correct Python code to solve the given problem."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": code}
    ]
    
    return {"messages": messages}


def tokenize_function(examples, tokenizer, max_length: int):
    """
    Tokenize 并设置 labels：仅对 assistant 回复段计算 loss。
    多数基座 chat template 不含 `{% generation %}`，不能直接使用 return_assistant_tokens_mask；
    通过「除最后一条 assistant 外的对话 + add_generation_prompt=True」得到前缀 token 边界。
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

    # 前缀与完整序列的公共前缀长度即 assistant 起点（两侧截断策略一致时与 len(prefix_ids) 一致）
    i = 0
    lim = min(len(prefix_ids), len(full_ids))
    while i < lim and prefix_ids[i] == full_ids[i]:
        i += 1
    assistant_start = i
    if assistant_start >= len(full_ids):
        labels = list(full_ids)
    else:
        labels = [-100] * assistant_start + full_ids[assistant_start:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def save_latest_adapter_path(adapter_path):
    """保存最新适配器路径到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(LATEST_LORA_POINTER), exist_ok=True)
    
    with open(LATEST_LORA_POINTER, "w", encoding="utf-8") as f:
        f.write(adapter_path + "\n")
    
    print(f"已保存最新适配器路径: {LATEST_LORA_POINTER}")
    print(f"路径内容: {adapter_path}")


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

    parser = argparse.ArgumentParser(description="CodeGeeX4 LoRA：仅监督 solution（Direct 模板）")
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
        help=f"单条样本 tokenizer 截断上限（默认 {MAX_LENGTH}）；显存紧张可试 768",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path.strip()) if args.data_path.strip() else DATA_PATH
    max_length = args.max_length if args.max_length > 0 else MAX_LENGTH
    if max_length < 256:
        raise ValueError("--max-length 过小，assistant 段易被完全截断")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"未找到训练数据: {data_path}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"未找到基座模型目录: {MODEL_PATH}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"codegeex4_lora_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    raw_data = load_jsonl(data_path)
    
    # 2. 构建消息
    print("[2/5] 构建对话格式...")
    formatted_data = [build_messages(item) for item in raw_data]
    dataset = Dataset.from_list(formatted_data)
    
    # 3. 划分数据集
    print("[3/5] 划分训练/验证集...")
    split = dataset.train_test_split(test_size=0.05, seed=SEED)
    train_dataset = split["train"]
    eval_dataset = split["test"]
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
    # 梯度检查点与推理 KV cache 互斥；显式关闭可避免反复打印「Setting use_cache=False」
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
    
    # 打印示例信息
    print(f"\n示例序列长度: {len(train_dataset[0]['input_ids'])}")
    train_ratio = 1 - train_dataset[0]['labels'].count(-100) / len(train_dataset[0]['labels'])
    print(f"训练 token 比例: {train_ratio:.1%}")
    
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
    print("\n开始训练...")
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
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"训练耗时: {train_time/60:.2f} 分钟")
    print(f"输出目录: {output_dir}")
    print(f"LoRA 适配器: {adapter_path}")
    print(f"最新适配器指针: {LATEST_LORA_POINTER}")
    print("="*60)


if __name__ == "__main__":
    main()