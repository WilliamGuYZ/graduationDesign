import os

# ===== 必须环境变量 =====
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import sys
import multiprocessing
import subprocess
import tempfile
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


# ======================
# CONFIG
# ======================

MODEL_PATH = "../models/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "../models/qwen_coder_lora_adapter_20260409_001400"
DATASET_PATH = "../data/processed/eval_code_fixed.jsonl"

K = 1
MAX_NEW_TOKENS = 512
BATCH_SIZE = 128
TEST_WORKERS = 8  # 减少并发数
TIMEOUT = 5
DATASET_LIMIT = 100


# ======================
# LOAD DATASET
# ======================

def load_dataset(path, limit=None):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            dataset.append(json.loads(line))
    
    print(f"Loaded dataset: {len(dataset)} samples")
    return dataset


# ======================
# PROMPT BUILDING（与 train.py 一致：Qwen2.5 Instruct 官方 chat_template）
# ======================

def build_prompt(tokenizer, problem):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )

# ======================
# CODE EXTRACTION
# ======================

def extract_code(text):
    text = text.replace("<|im_start|>assistant", "").strip()
    text = text.split("<|im_end|>")[0].strip()
    
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return text.strip()


def clean_code(code):
    if not code:
        return ""
    
    code = re.sub(r'<\|im_[a-z]+\|>', '', code)
    code = code.replace("```", "")
    
    # 修复全局 return 错误
    lines = code.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("return ") and "def " not in line:
            # 尝试将 return 转换为 print
            return_value = stripped[7:].strip()
            fixed_lines.append(f"print({return_value})  # Fixed: return -> print")
        else:
            fixed_lines.append(line)
    code = "\n".join(fixed_lines)
    
    # 移除开头空行
    lines = code.split("\n")
    start = 0
    for i, l in enumerate(lines):
        if l.strip():
            start = i
            break
    
    code = "\n".join(lines[start:])
    return code.strip()


# ======================
# OUTPUT NORMALIZATION
# ======================

def normalize_output(s):
    if not s:
        return ""
    return " ".join(s.strip().split())


# ======================
# SAFE EXECUTION WITH SUBPROCESS
# ======================

def run_single_test(code, test):
    """使用子进程执行代码，避免多进程环境中的 stdout 重定向问题"""
    input_data = test["input"]
    expected = normalize_output(test["output"])
    
    if not code.strip():
        return False
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # 运行代码
        result = subprocess.run(
            ['python', temp_file],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        
        # 清理临时文件
        os.unlink(temp_file)
        
        if result.returncode == 0:
            output = normalize_output(result.stdout)
            return output == expected
        else:
            return False
            
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_sample(args):
    """测试单个样本的所有测试用例"""
    code, tests = args
    
    if not code.strip():
        return False
    
    # 运行所有测试
    for test in tests:
        if not run_single_test(code, test):
            return False
    
    return True


# ======================
# PASS@K
# ======================

def compute_pass_at_k(results):
    if not results:
        return 0
    return sum(results) / len(results)


# ======================
# MAIN
# ======================

def main():
    print("=" * 60)
    print("Loading tokenizer")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizer loaded")
    
    print("\nLoading vLLM model")
    
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
        trust_remote_code=True,
        enable_lora=(LORA_PATH is not None),
        max_lora_rank=32

    )
    
    lora_request = None
    if LORA_PATH:
        print("Using LoRA:", LORA_PATH)
        lora_request = LoRARequest(
            "qwen_lora",
            1,
            LORA_PATH
        )
    
    dataset = load_dataset(DATASET_PATH, DATASET_LIMIT)
    
    prompts = []
    for sample in dataset:
        if "instruction" in sample:
            inst = sample["instruction"]
            inp = sample.get("input") or ""
            problem = inst + "\n" + inp if inp else inst
        else:
            problem = sample["input"]
        prompts.append(build_prompt(tokenizer, problem))
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        n=1,
        stop=["<|im_end|>", "</s>", "<|im_start|>"]  # 添加更多停止符
    )
    
    print("\nGenerating solutions...")
    
    # 一次性生成所有输出
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request
    )
    
    print(f"Generated {len(outputs)} solutions")
    print("\nPreparing test data...")
    
    # 准备测试数据
    test_args = []
    for out, sample in zip(outputs, dataset):
        tests = sample.get("tests", [])
        if not tests:
            test_args.append(("", []))
        else:
            code = clean_code(extract_code(out.outputs[0].text))
            test_args.append((code, tests))
    
    print("\nRunning tests in parallel...")
    
    # 使用进程池并行测试
    with multiprocessing.Pool(processes=TEST_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(test_sample, test_args),
            total=len(test_args),
            desc="Testing"
        ))
    
    score = compute_pass_at_k(results)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"pass@{K}: {score:.4f}")
    print(f"Correct problems: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()