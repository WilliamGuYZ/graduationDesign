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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# ================= 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-Coder-7B-Instruct")
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
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== 模型 =====
    print(">>> 加载模型 (QLoRA)")
    # Tesla V100（Volta）：QLoRA 计算与加载用 fp16；bf16/tf16 非硬件最优
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        # 双卡：在各 GPU 间均衡放置层（需可见 2 张卡，如 CUDA_VISIBLE_DEVICES=0,1）
        device_map="balanced",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # ===== LoRA 配置 =====
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ===== 数据 =====
    print(">>> 加载数据")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    print(f"原始数据: {len(dataset)} 条")
    
    dataset = dataset.map(
        partial(build_training_text, tokenizer=tokenizer),
        num_proc=4,
    )
    
    MAX_SEQ_LENGTH = 2048
    
    dataset = dataset.map(
        lambda x: tokenize_with_mask(x, tokenizer, MAX_SEQ_LENGTH),
        remove_columns=dataset.column_names,
        num_proc=4
    )
    
    print(f"训练样本: {len(dataset)} 条")
    
    # 检查序列长度
    lengths = [len(x["input_ids"]) for x in dataset.select(range(min(100, len(dataset))))]
    print(f"平均长度: {sum(lengths)/len(lengths):.1f}, 最大: {max(lengths)}, 最小: {min(lengths)}")
    
    # ===== 训练参数（固定：Tesla V100-SXM2-32GB ×2，Volta 架构）=====
    batch_size = 2
    grad_acc = 8
    epochs = 3

    training_args = TrainingArguments(
        output_dir=train_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print(f"\n>>> 开始训练")
    print(f"有效 batch size: {batch_size * grad_acc}")
    print(f"学习率: {training_args.learning_rate}")
    print(f"LoRA rank: {lora_config.r}")
    print(f"最大序列长度: {MAX_SEQ_LENGTH}")
    
    trainer.train()
    
    # ===== 保存 =====
    adapter_dir, _ = create_timestamp_dir(
        os.path.join(PROJECT_ROOT, "models"),
        "qwen_coder_lora_adapter",
    )
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("LoRA 保存:", adapter_dir)
    
    # ===== 日志 =====
    total_time = time.time() - start_time
    summary = {
        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(dataset),
        "total_time": str(timedelta(seconds=int(total_time))),
        "adapter_dir": adapter_dir,
        "config": {
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": batch_size,
            "grad_accum": grad_acc,
            "learning_rate": training_args.learning_rate,
            "epochs": epochs,
        }
    }
    save_json(os.path.join(train_dir, "training_log.json"), summary)
    
    print(f"\n训练完成，总时间: {summary['total_time']}")

if __name__ == "__main__":
    main()