# train_lora.py
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
    使用 tokenizer 内建的 Qwen2.5 Instruct chat_template（默认 system、<|redacted_im_end|> 等），
    与 HuggingFace / 官方推理一致。problem = instruction + "\\n" + input（与 eval 侧一致）。
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    code = example.get("output", "")

    if input_text:
        problem = instruction + "\n" + input_text
    else:
        problem = instruction

    user_msg = {"role": "user", "content": problem}
    prompt = tokenizer.apply_chat_template(
        [user_msg],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [user_msg, {"role": "assistant", "content": code}],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": full,
        "prompt": prompt,
        "response": code,
    }

def tokenize_with_mask(example, tokenizer, max_len):
    """Tokenize 并设置 mask，只对 assistant 部分计算损失"""
    prompt = example["prompt"]
    full_text = example["text"]
    
    # Tokenize prompt 确定 mask 位置
    prompt_tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=False
    )
    
    # Tokenize 完整文本
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=False
    )
    
    labels = tokenized["input_ids"].copy()
    prompt_len = len(prompt_tokenized["input_ids"])
    
    # prompt 部分不计算损失
    if prompt_len < len(labels):
        labels[:prompt_len] = [-100] * prompt_len
    
    tokenized["labels"] = labels
    return tokenized

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
    
    # ===== 模型 - 不使用量化 =====
    print(">>> 加载模型 (FP16 LoRA)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    model.gradient_checkpointing_enable()
    
    # ===== LoRA 配置（折中方案，平衡学习和泛化）=====
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
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
    
    MAX_SEQ_LENGTH = 2048
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_with_mask(x, tokenizer, MAX_SEQ_LENGTH),
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    print(f"总样本: {len(dataset)} 条")
    
    # 检查序列长度
    lengths = [len(x["input_ids"]) for x in dataset.select(range(min(100, len(dataset))))]
    print(f"平均长度: {sum(lengths)/len(lengths):.1f}, 最大: {max(lengths)}, 最小: {min(lengths)}")
    
    # ===== 划分训练集和验证集 =====
    print("\n>>> 划分训练集和验证集")
    if len(dataset) < 100:
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    else:
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"训练样本: {len(train_dataset)} 条")
    print(f"验证样本: {len(eval_dataset)} 条")
    
    # ===== 训练参数（折中配置）=====
    batch_size = 2
    grad_acc = 8
    epochs = 3

    training_args = TrainingArguments(
        output_dir=train_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,
        optim="adamw_8bit",
        weight_decay=0.03,
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
        }
    }
    save_json(os.path.join(train_dir, "training_log.json"), summary)
    
    print(f"\n训练完成，总时间: {summary['total_time']}")
    if best_metrics:
        print(f"最佳验证损失: {best_metrics['best_metric']:.4f}")

if __name__ == "__main__":
    main()