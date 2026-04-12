import os
import json
import re
import subprocess
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import Counter

# =========================
# CONFIG
# =========================

MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "./models/qwen_coder_lora_adapter_fp16_20260411_021749"
DATASET_PATH = "./data/processed/mbpp.jsonl"

TIMEOUT = 3
MAX_TOKENS = 2048
NUM_SAMPLES = 10 
TEMPERATURE = 0.8 

# =========================
# LOAD MODEL WITH LORA
# =========================

print("Loading model with LoRA adaptor...")
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    enable_lora=True,  # 启用LoRA支持
    max_lora_rank=32, 
)

# 创建LoRA请求（只需创建一次）
from vllm.lora.request import LoRARequest

lora_request = LoRARequest(
    lora_name="my_adapter",  # 可以任意命名
    lora_int_id=1,           # 唯一ID
    lora_path=LORA_PATH      # LoRA文件路径
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    top_p=0.95,
    top_k=50,
    stop=None
)

# =========================
# PROMPT
# =========================

def build_prompt(problem, test_list=None):
    """改进的prompt，包含测试用例"""
    prompt = f"""Problem: {problem}

Write a Python function to solve the problem above."""
    
    if test_list and len(test_list) > 0:
        prompt += f"\n\nThe function must pass this test:\n{test_list[0]}"
    
    prompt += "\n\nImplementation:"
    
    return prompt

# =========================
# CODE EXTRACTION AND FIXING
# =========================
def add_missing_imports(code):
    """自动添加代码中缺失的常见imports"""
    imports_to_add = []
    
    # 常见库的检测规则
    import_rules = [
        ('import heapq', ['heapq.']),
        ('import re', ['re.']),
        ('import math', ['math.']),
        ('import random', ['random.']),
        ('import json', ['json.']),
        ('import itertools', ['itertools.']),
        ('import collections', ['collections.']),
        ('import functools', ['functools.']),
        ('import typing', ['typing.']),
        ('import os', ['os.']),
        ('import sys', ['sys.']),
        ('import copy', ['copy.']),
        ('import time', ['time.']),
        ('import datetime', ['datetime.']),
        ('import hashlib', ['hashlib.']),
        ('import base64', ['base64.']),
        ('import csv', ['csv.']),
        ('import string', ['string.']),
    ]
    
    for import_stmt, patterns in import_rules:
        if import_stmt not in code:
            for pattern in patterns:
                if pattern in code:
                    imports_to_add.append(import_stmt)
                    break
    
    # 添加所有缺失的imports
    if imports_to_add:
        code = '\n'.join(imports_to_add) + '\n' + code
    
    return code

def extract_code(text):
    """提取并修复生成的代码"""
    code_block = re.findall(r"```python(.*?)```", text, re.S)
    if code_block:
        code = code_block[0]
    else:
        code_block = re.findall(r"```(.*?)```", text, re.S)
        if code_block:
            code = code_block[0]
        else:
            code = text
    
    lines = code.split('\n')
    function_lines = []
    in_function = False
    
    for line in lines:
        if 'def ' in line and not in_function:
            in_function = True
            function_lines.append(line)
        elif in_function:
            function_lines.append(line)
            if line.strip() and not line.startswith((' ', '\t')) and line.strip():
                if len(function_lines) > 1:
                    function_lines.pop()
                    break
    
    if function_lines:
        code = '\n'.join(function_lines)
    
    # 自动添加缺失的imports
    code = add_missing_imports(code)
    
    return code.strip()

# =========================
# RUN TESTS
# =========================

def run_test(code, tests):
    """运行测试用例"""
    if not code:
        return False
    
    program = code + "\n\n"
    for t in tests:
        program += t + "\n"
    
    try:
        result = subprocess.run(
            ["python3", "-c", program],
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

# =========================
# LOAD DATASET
# =========================

dataset = []
with open(DATASET_PATH) as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Dataset size: {len(dataset)}")

# =========================
# GENERATION WITH LORA
# =========================

# 为每个问题生成 NUM_SAMPLES 个独立的 prompt
all_prompts = []
prompt_to_problem_idx = []

for idx, data in enumerate(dataset):
    prompt = build_prompt(data["text"], data.get("test_list", []))
    for _ in range(NUM_SAMPLES):
        all_prompts.append(prompt)
        prompt_to_problem_idx.append(idx)

print(f"Generating {len(all_prompts)} code samples ({NUM_SAMPLES} per problem) with LoRA...")

# 使用LoRA进行生成（只需添加lora_request参数）
outputs = llm.generate(
    all_prompts, 
    sampling_params,
    lora_request=lora_request  # 使用你的LoRA适配器
)

# =========================
# EVALUATION
# =========================

# 存储每个问题的结果
problem_results = [[] for _ in range(len(dataset))]

for idx, (data_idx, output) in enumerate(tqdm(zip(prompt_to_problem_idx, outputs), total=len(all_prompts))):
    generated = output.outputs[0].text
    code = extract_code(generated)
    
    if not code or '# Your code here' in code:
        passed = False
    else:
        passed = run_test(code, dataset[data_idx]["test_list"])
    
    problem_results[data_idx].append(passed)

# =========================
# CALCULATE PASS@1, PASS@5, PASS@10
# =========================

total_correct_pass1 = 0
total_correct_pass5 = 0
total_correct_pass10 = 0

# 存储每个问题的详细统计
problem_stats = []

for samples in problem_results:
    # pass@1: 第一个样本通过
    pass1_correct = samples[0] if samples else False
    
    # pass@5: 前5个样本中至少有一个通过
    pass5_correct = any(samples[:5]) if len(samples) >= 5 else any(samples)
    
    # pass@10: 所有样本中至少有一个通过
    pass10_correct = any(samples)
    
    # 统计正确数量
    num_correct = sum(samples)
    
    if pass1_correct:
        total_correct_pass1 += 1
    if pass5_correct:
        total_correct_pass5 += 1
    if pass10_correct:
        total_correct_pass10 += 1
    
    problem_stats.append({
        "num_correct": num_correct,
        "pass1": pass1_correct,
        "pass5": pass5_correct,
        "pass10": pass10_correct,
        "samples": samples
    })

total_problems = len(dataset)
pass1 = total_correct_pass1 / total_problems if total_problems > 0 else 0
pass5 = total_correct_pass5 / total_problems if total_problems > 0 else 0
pass10 = total_correct_pass10 / total_problems if total_problems > 0 else 0

# =========================
# PRINT RESULTS
# =========================

print("\n" + "=" * 60)
print("MBPP PASS@K EVALUATION RESULTS (WITH LoRA)")
print("=" * 60)
print(f"Total problems:        {total_problems}")
print(f"Samples per problem:   {NUM_SAMPLES}")
print(f"Temperature:           {TEMPERATURE}")
print(f"LoRA Path:             {LORA_PATH}")
print("-" * 60)
print(f"PASS@1:                {pass1:.4f} ({pass1*100:.2f}%)")
print(f"  Correct:             {total_correct_pass1}/{total_problems}")
print("-" * 60)
print(f"PASS@5:                {pass5:.4f} ({pass5*100:.2f}%)")
print(f"  Correct:             {total_correct_pass5}/{total_problems}")
print("-" * 60)
print(f"PASS@10:               {pass10:.4f} ({pass10*100:.2f}%)")
print(f"  Correct:             {total_correct_pass10}/{total_problems}")
print("=" * 60)
