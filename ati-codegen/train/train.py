import os
import json
import time
import torch
import warnings
from functools import partial

from datetime import datetime, timedelta
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# ================= 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_code.jsonl")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "train")

# ================= 工具函数 =================

def create_timestamp_dir(base_dir, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path, timestamp

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_training_text(example, tokenizer):
    """
    使用与评估完全一致的纯文本格式（不使用chat template）
    这样可以避免训练和评估时的格式不匹配问题
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    code = example.get("output", "")

    # 构建问题（与评估保持一致）
    if input_text:
        problem = instruction + "\n" + input_text
    else:
        problem = instruction

    # 使用与评估完全相同的prompt格式
    prompt = f"""Problem: {problem}

Write a Python function to solve the problem above.

Implementation:"""
    
    # 完整文本 = prompt + 代码（确保代码前有换行）
    full = prompt + "\n" + code
    
    return {
        "text": full,
        "prompt": prompt,
        "response": code,
    }

def tokenize_with_mask(example, tokenizer, max_len):
    """
    改进版tokenization：正确处理截断和对齐
    """
    prompt = example["prompt"]
    full_text = example["text"]
    
    # 先tokenize完整文本，启用截断
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_overflowing_tokens=False,
    )
    
    # tokenize prompt（同样截断，确保长度一致）
    prompt_tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    
    labels = tokenized["input_ids"].copy()
    prompt_len = len(prompt_tokenized["input_ids"])
    
    # 安全处理：如果prompt比完整文本还长，全部mask
    if prompt_len >= len(labels):
        labels = [-100] * len(labels)
    else:
        # 只对prompt部分设置-100（不计算损失）
        labels[:prompt_len] = [-100] * prompt_len
    
    tokenized["labels"] = labels
    return tokenized

def print_training_example(dataset, tokenizer, num_examples=2):
    """打印训练样本示例，用于调试"""
    print("\n>>> 训练数据示例:")
    for i in range(min(num_examples, len(dataset))):
        example = dataset[i]
        print(f"\n--- 示例 {i+1} ---")
        print(f"Prompt (前200字符): {example['prompt'][:200]}...")
        print(f"Response (前100字符): {example['response'][:100]}...")
        print(f"完整文本长度: {len(example['text'])} 字符")
        
        # 检查tokenization结果
        tokenized = tokenize_with_mask(example, tokenizer, 2048)
        print(f"Tokenized长度: {len(tokenized['input_ids'])}")
        print(f"Labels中-100的数量: {tokenized['labels'].count(-100)}")
        print(f"实际训练token数: {len([l for l in tokenized['labels'] if l != -100])}")

# ================= 主函数 =================

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    train_dir, timestamp = create_timestamp_dir(OUTPUT_ROOT, "qwen_coder_lora_train")
    print("训练目录:", train_dir)
    
    # ===== Tokenizer =====
    print(">>> 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== 模型加载（添加4-bit量化以提高稳定性）=====
    print(">>> 加载模型 (4-bit量化 + LoRA)")
    
    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache=False,
        device_map="auto",
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # ===== LoRA配置（优化后的参数）=====
    lora_config = LoraConfig(
        r=16,                      # 降低rank减少过拟合
        lora_alpha=32,             # alpha = 2 * r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,          # 增加dropout提高泛化
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ===== 数据加载和划分 =====
    print(">>> 加载数据")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    print(f"原始数据: {len(dataset)} 条")
    
    # 构建训练文本
    dataset = dataset.map(
        partial(build_training_text, tokenizer=tokenizer),
        num_proc=4,
    )
    
    # 打印训练示例（调试用）
    print_training_example(dataset, tokenizer)
    
    MAX_SEQ_LENGTH = 2048
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_with_mask(x, tokenizer, MAX_SEQ_LENGTH),
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    # 过滤掉过短的样本（可选）
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    
    print(f"总样本: {len(dataset)} 条")
    
    # 检查序列长度分布
    lengths = [len(x["input_ids"]) for x in dataset.select(range(min(500, len(dataset))))]
    print(f"序列长度统计: 平均={sum(lengths)/len(lengths):.1f}, "
          f"最大={max(lengths)}, 最小={min(lengths)}, "
          f"P95={sorted(lengths)[int(len(lengths)*0.95)]}")
    
    # ===== 划分训练集和验证集 =====
    print("\n>>> 划分训练集和验证集")
    if len(dataset) < 200:
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    else:
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"训练样本: {len(train_dataset)} 条")
    print(f"验证样本: {len(eval_dataset)} 条")
    
    # ===== 优化后的训练参数 =====
    batch_size = 4                  # 增加到4（量化后显存更充裕）
    grad_acc = 4                    # 降低到4，有效batch=16
    epochs = 3

    training_args = TrainingArguments(
        output_dir=train_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=5e-5,         # 降低学习率（从2e-4降到5e-5）
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,          # 增加warmup
        bf16=False,
        fp16=True,
        optim="adamw_8bit",
        weight_decay=0.05,          # 增加权重衰减
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        # 验证相关参数
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 其他参数
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        tf32=False,
        remove_unused_columns=True,
        # 添加训练稳定性参数
        max_grad_norm=1.0,          # 梯度裁剪
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    
    # ===== 创建 Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print(f"\n>>> 开始训练")
    print(f"有效 batch size: {batch_size * grad_acc}")
    print(f"学习率: {training_args.learning_rate}")
    print(f"LoRA rank: {lora_config.r}")
    print(f"LoRA alpha: {lora_config.lora_alpha}")
    print(f"LoRA dropout: {lora_config.lora_dropout}")
    print(f"权重衰减: {training_args.weight_decay}")
    print(f"最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"早停耐心值: 3 次评估")
    print(f"梯度裁剪: {training_args.max_grad_norm}")
    
    # 开始训练
    trainer.train()
    
    # ===== 保存最佳模型 =====
    adapter_dir, _ = create_timestamp_dir(
        os.path.join(PROJECT_ROOT, "models"),
        "qwen_coder_lora_adapter",
    )
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("LoRA 保存:", adapter_dir)
    
    # 同时保存一份配置信息供评估使用
    config_info = {
        "prompt_format": "plain_text",  # 标记使用的格式
        "prompt_template": """Problem: {problem}

Write a Python function to solve the problem above.

Implementation:""",
        "notes": "训练和评估使用相同的纯文本格式"
    }
    save_json(os.path.join(adapter_dir, "training_config.json"), config_info)
    
    # ===== 日志 =====
    total_time = time.time() - start_time
    
    best_metrics = {}
    if trainer.state.best_metric is not None:
        best_metrics = {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
    
    summary = {
        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "total_samples": len(dataset),
        "total_time": str(timedelta(seconds=int(total_time))),
        "adapter_dir": adapter_dir,
        "best_metrics": best_metrics,
        "config": {
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": batch_size,
            "grad_accum": grad_acc,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "epochs": epochs,
            "early_stopping_patience": 3,
            "quantization": "4bit",
            "prompt_format": "plain_text",
        }
    }
    save_json(os.path.join(train_dir, "training_log.json"), summary)
    
    print(f"\n训练完成，总时间: {summary['total_time']}")
    if best_metrics:
        print(f"最佳验证损失: {best_metrics['best_metric']:.4f}")
    

if __name__ == "__main__":
    main()