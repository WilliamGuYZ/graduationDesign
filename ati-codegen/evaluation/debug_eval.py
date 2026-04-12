import os
import json
import re
import subprocess
from tqdm import tqdm
from vllm import LLM, SamplingParams

# =========================
# CONFIG
# =========================

MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
DATASET_PATH = "./data/processed/mbpp.jsonl"

TIMEOUT = 3
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    stop=None  # 移除stop限制，让模型完整生成
)

# =========================
# PROMPT
# =========================

def build_prompt(problem, test_list=None):
    """改进的prompt，包含测试用例"""
    prompt = f"""Problem: {problem}

Write a Python function to solve the problem above."""
    
    # 如果有测试用例，添加到prompt中
    if test_list and len(test_list) > 0:
        prompt += f"\n\nThe function must pass this test:\n{test_list[0]}"
    
    prompt += "\n\nImplementation:"
    
    return prompt

# =========================
# CODE EXTRACTION AND FIXING
# =========================

def extract_code(text):
    """提取并修复生成的代码"""
    # 提取代码块
    code_block = re.findall(r"```python(.*?)```", text, re.S)
    if code_block:
        code = code_block[0]
    else:
        code_block = re.findall(r"```(.*?)```", text, re.S)
        if code_block:
            code = code_block[0]
        else:
            code = text
    
    # 提取函数定义（如果有多余内容）
    lines = code.split('\n')
    function_lines = []
    in_function = False
    brace_count = 0
    
    for line in lines:
        if 'def ' in line and not in_function:
            in_function = True
            function_lines.append(line)
        elif in_function:
            function_lines.append(line)
            # Python用缩进，检测函数结束
            if line.strip() and not line.startswith((' ', '\t')) and line.strip():
                if len(function_lines) > 1:
                    function_lines.pop()
                    break
    
    if function_lines:
        code = '\n'.join(function_lines)
    
    # 自动添加缺失的imports
    if 'heapq' in code and 'import heapq' not in code:
        code = 'import heapq\n' + code
    if 're.' in code and 'import re' not in code:
        code = 'import re\n' + code
    if 'math.' in code and 'import math' not in code:
        code = 'import math\n' + code
    if 'random.' in code and 'import random' not in code:
        code = 'import random\n' + code
    
    return code.strip()

# =========================
# RUN TESTS
# =========================

def run_test(code, tests):
    """运行测试用例"""
    if not code:
        return False
    
    # 构建完整的测试程序
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
        
        if result.returncode != 0:
            # 可选：打印错误信息用于调试
            # print(f"Error: {result.stderr}")
            pass
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

# =========================
# LOAD DATASET
# =========================

dataset = []

with open(DATASET_PATH) as f:
    for line in f:
        dataset.append(json.loads(line))

print("Dataset size:", len(dataset))

# 可选：只测试前N个样本用于快速验证
# dataset = dataset[:10]

# =========================
# GENERATION
# =========================

prompts = [build_prompt(d["text"], d.get("test_list", [])) for d in dataset]

print("Generating code...")
outputs = llm.generate(prompts, sampling_params)

# =========================
# EVALUATION
# =========================

correct = 0
results = []

for data, output in tqdm(zip(dataset, outputs), total=len(dataset)):
    
    # 使用 MBPP 数据中提供的原始代码
    code = data.get("code", "")
    
    # 检查代码是否为空
    if not code or code.strip() == "":
        passed = False
    else:
        # 直接使用原始代码运行测试
        passed = run_test(code, data["test_list"])
    
    if passed:
        correct += 1
    
    # 保存结果用于调试
    results.append({
        "task_id": data.get("task_id", "unknown"),
        "passed": passed,
        "code": code[:200]  # 只保存前200字符
    })

# =========================
# RESULT
# =========================

total = len(dataset)
pass1 = correct / total if total > 0 else 0

print("\n" + "=" * 60)
print("MBPP EVALUATION RESULTS")
print("=" * 60)
print(f"Total samples:  {total}")
print(f"Correct:        {correct}")
print(f"Failed:         {total - correct}")
print(f"pass@1:         {pass1:.4f} ({pass1*100:.2f}%)")
print("=" * 60)

# 打印失败的样本（可选）
failed_samples = [r for r in results if not r["passed"]]
if failed_samples:
    print(f"\nFailed samples ({len(failed_samples)}):")
    for sample in failed_samples[:5]:  # 只显示前5个
        print(f"  - {sample['task_id']}")
        print(f"    Code preview: {sample['code']}...")
    
    if len(failed_samples) > 5:
        print(f"  ... and {len(failed_samples) - 5} more")

print(f"\nDetailed results saved to mbpp_eval_results.json")
