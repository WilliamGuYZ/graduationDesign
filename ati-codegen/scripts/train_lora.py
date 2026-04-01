from __future__ import annotations

"""
LoRA 训练入口（骨架）。

你后续需要做的事：
- 把 `dataset_path` 指向你构建好的指令数据（jsonl / datasets）
- 设置 `base_model` 为你选的代码模型（如 Qwen2.5-Coder / CodeGeeX / Llama）
- 按你的显卡调整 batch/seq_len/gradient_accumulation
"""

import argparse

from _bootstrap import bootstrap_src_path

bootstrap_src_path()


def build_text(ex: dict) -> str:
    """从 instruction/input/output 构建训练文本"""
    input_text = ex["input"] if ex["input"] else "无"
    return "\n\n".join([
        f"### Instruction\n{ex['instruction']}",
        f"### Input\n{input_text}",
        f"### Output\n{ex['output']}",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF 模型名或本地路径")
    ap.add_argument("--dataset_path", required=True, help="训练数据路径")
    ap.add_argument("--output_dir", required=True, help="LoRA 输出目录")
    args = ap.parse_args()

    # 延迟导入，避免没有 torch 环境也能跑其它脚本
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None,  # 你可以按模型结构指定（例如 q_proj,v_proj,...）
    )
    model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files={"train": args.dataset_path})

    def tokenize(ex):
        text = build_text(ex)
        return tokenizer(text, truncation=True, max_length=1024)

    ds_tok = ds.map(tokenize, remove_columns=ds["train"].column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())