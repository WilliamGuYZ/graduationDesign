"""
检查数据集的序列长度分布，帮助确定合适的 MAX_LENGTH
"""

import os
import json
import random
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# ==================== 配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-Coder-7B-Instruct")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "KodCode_train.jsonl")

# 测试不同的 few-shot 数量
NUM_SHOTS_LIST = [0, 1, 2]
# =============================================


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"加载了 {len(data)} 条数据\n")
    return data


def build_instruction(example):
    question = example["question"].strip()
    func_name = example["test_info"][0]["function_name"]
    params = example["test_info"][0]["parameter_list"]
    return (
        f"Implement the function `{func_name}` that takes {params} to solve:\n\n"
        f"{question}\n\n"
        f"Think step by step, then output the Python code."
    )


def build_cot_response(example):
    thought = example.get("thought_step", "").strip()
    code = example["solution"].strip()
    if thought:
        return f"{thought}\n\n```python\n{code}\n```"
    return f"```python\n{code}\n```"


def build_few_shot_messages(example, pool, num_shots):
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
    
    if num_shots > 0:
        candidates = [e for e in pool if e["question"] != example["question"]]
        shots = random.sample(candidates, min(num_shots, len(candidates)))
        for shot in shots:
            messages.append({"role": "user", "content": build_instruction(shot)})
            messages.append({"role": "assistant", "content": build_cot_response(shot)})
    
    messages.append({"role": "user", "content": build_instruction(example)})
    messages.append({"role": "assistant", "content": build_cot_response(example)})
    
    return messages


def analyze(data, tokenizer, num_shots, max_samples=500):
    """分析序列长度，返回统计信息"""
    random.seed(42)
    sample_data = random.sample(data, min(max_samples, len(data)))
    
    lengths = []
    assistant_ratios = []
    
    for item in tqdm(sample_data, desc=f"  Shots={num_shots}"):
        messages = build_few_shot_messages(item, data, num_shots)
        
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prefix_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        
        full_len = len(full_ids)
        prefix_len = len(prefix_ids)
        assistant_len = full_len - prefix_len
        
        lengths.append(full_len)
        assistant_ratios.append(assistant_len / full_len if full_len > 0 else 0)
    
    return {
        "mean": np.mean(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "median": np.median(lengths),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99),
        "train_ratio": np.mean(assistant_ratios),
    }


def main():
    print("加载数据...")
    raw_data = load_jsonl(DATA_PATH)
    raw_data = [d for d in raw_data if d.get("thought_step", "").strip()]
    print(f"有效数据: {len(raw_data)} 条\n")
    
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("完成\n")
    
    print("=" * 70)
    print("序列长度统计 (基于 500 条随机样本)")
    print("=" * 70)
    
    for num_shots in NUM_SHOTS_LIST:
        stats = analyze(raw_data, tokenizer, num_shots)
        
        print(f"\nNUM_FEW_SHOTS = {num_shots}")
        print(f"  平均长度:     {stats['mean']:.0f} ± {stats['std']:.0f} tokens")
        print(f"  中位数:       {stats['median']:.0f} tokens")
        print(f"  95% 分位数:   {stats['p95']:.0f} tokens")
        print(f"  99% 分位数:   {stats['p99']:.0f} tokens")
        print(f"  最大长度:     {stats['max']:.0f} tokens")
        print(f"  训练 token 比: {stats['train_ratio']:.1%}")
        
        # 建议 MAX_LENGTH
        recommended = int(stats['p95'] + 100)
        print(f"  建议 MAX_LENGTH: {recommended} (覆盖95%样本)")


if __name__ == "__main__":
    main()