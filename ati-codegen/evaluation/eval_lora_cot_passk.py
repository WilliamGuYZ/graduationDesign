"""
Qwen2.5-Coder-7B-Instruct + LoRA CoT 两阶段推理评估脚本
"""

import gc
import json
import os
import random
import re
import subprocess
import sys
import warnings
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 屏蔽日志
# =========================

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_LOGGING_PREFIX"] = ""
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"

import logging

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("vllm.engine").setLevel(logging.ERROR)
logging.getLogger("vllm.worker").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)  
_SCRIPTS = os.path.join(_PROJECT_ROOT, "scripts")
LORA_POINTER_FILE = os.path.join(_PROJECT_ROOT, "train", "latest_lora_cot_adapter.txt")

MODEL_PATH   = os.path.join(_PROJECT_ROOT, "models", "Qwen2.5-Coder-7B-Instruct")
DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "KodCode_eval.jsonl")
RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "evaluation", "results")

# 评估参数
TIMEOUT = 5
MAX_TOKENS_REASONING = 1024
MAX_TOKENS_CODE = 1024
NUM_SAMPLES = 10
TEMPERATURE = 0.3
BATCH_SIZE = 32

# vLLM 配置
VLLM_MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85
SAMPLE_LIMIT: Optional[int] = None

_SUBPROC_SNIPPET = (
    "import json,sys;sys.path.insert(0,sys.argv[1]);"
    "from validate_code_with_test import test_solution_code;"
    "d=json.load(sys.stdin);"
    "r=test_solution_code(d['solution'],d['test'],[]);"
    "json.dump(list(r),sys.stdout)"
)


def _read_lora_adapter_dir() -> str:
    with open(LORA_POINTER_FILE, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    return os.path.abspath(line)


def clean_question(text: Any) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_code(text: str) -> str:
    if not text:
        return ""
    code_blocks = re.findall(r"```python(.*?)```", text, re.S)
    if not code_blocks:
        code_blocks = re.findall(r"```(.*?)```", text, re.S)
    if code_blocks:
        return code_blocks[-1].strip()
    return text.strip()


def get_function_name_from_test_info(test_info: List[Dict]) -> str:
    if test_info and len(test_info) > 0:
        return test_info[0].get("function_name", "solution")
    return "solution"


def get_parameter_list(test_info: List[Dict]) -> str:
    if test_info and len(test_info) > 0:
        return test_info[0].get("parameter_list", "")
    return ""


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def test_solution_code_with_timeout(
    solution_code: str, test_code: str, timeout: int = 5
) -> Tuple[bool, Optional[str]]:
    """返回 (是否通过, 错误信息)"""
    if not solution_code or not test_code:
        return False, "代码或测试为空"
    
    payload = json.dumps({"solution": solution_code, "test": test_code}, ensure_ascii=False)
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROC_SNIPPET, _SCRIPTS],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"执行超时 (>{timeout}秒)"
    
    out = (result.stdout or "").strip()
    if result.returncode == 0 and out:
        try:
            pair = json.loads(out)
            return bool(pair[0]), pair[1] if len(pair) > 1 else None
        except json.JSONDecodeError:
            return False, f"输出解析失败: {out[:200]}"
    
    err = (result.stderr or result.stdout or f"退出码 {result.returncode}")[-500:]
    return False, err


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lora_path = _read_lora_adapter_dir()
    
    print("=" * 70)
    print("两阶段推理评估")
    print("=" * 70)
    print(f"LoRA 路径:           {lora_path}")
    print(f"每问题样本数:        {NUM_SAMPLES}")
    print(f"批处理大小:          {BATCH_SIZE}")
    
    # 加载 Tokenizer
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载 vLLM 模型
    _llm_kwargs: Dict[str, Any] = dict(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
        enable_lora=True,
        max_lora_rank=32,
    )
    
    _tok_cfg = os.path.join(lora_path, "tokenizer_config.json")
    _tok_json = os.path.join(lora_path, "tokenizer.json")
    if os.path.isdir(lora_path) and (os.path.isfile(_tok_cfg) or os.path.isfile(_tok_json)):
        _llm_kwargs["tokenizer"] = lora_path
    
    llm = LLM(**_llm_kwargs)
    
    lora_request = LoRARequest(
        lora_name="eval_adapter",
        lora_int_id=1,
        lora_path=lora_path,
    )
    
    # 加载数据集
    dataset = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = clean_question(row.get("question"))
            test_code = row.get("test", "")
            test_info = row.get("test_info", [])
            
            if not question or not test_code:
                continue
            
            dataset.append({
                "question": question,
                "test_code": test_code,
                "func_name": get_function_name_from_test_info(test_info),
                "params": get_parameter_list(test_info),
            })
    
    print(f"数据集: {len(dataset)} 条")
    
    if SAMPLE_LIMIT:
        dataset = dataset[:SAMPLE_LIMIT]
    
    total_samples = len(dataset) * NUM_SAMPLES
    
    # 构建 prompts
    print("构建 prompts...")
    all_reasoning_prompts = []
    all_metadata = []
    
    for problem_idx, data in enumerate(dataset):
        for sample_idx in range(NUM_SAMPLES):
            reasoning_messages = [
                {"role": "system", "content": "You are an expert algorithm teacher. Generate step-by-step reasoning."},
                {"role": "user", "content": f"Problem:\n{data['question']}\n\nGenerate step-by-step reasoning."}
            ]
            reasoning_prompt = tokenizer.apply_chat_template(reasoning_messages, tokenize=False, add_generation_prompt=True)
            
            all_reasoning_prompts.append(reasoning_prompt)
            all_metadata.append({
                "problem_idx": problem_idx,
                "sample_idx": sample_idx,
                "func_name": data["func_name"],
                "params": data["params"],
                "question": data["question"],
                "test_code": data["test_code"],
            })
    
    # ========== 阶段1：生成推理 ==========
    print(f"\n[1/2] 生成推理 (共 {total_samples} 个)...")
    all_reasonings = [None] * len(all_reasoning_prompts)
    
    with tqdm(total=len(all_reasoning_prompts), desc="推理进度", unit="个") as pbar:
        for i in range(0, len(all_reasoning_prompts), BATCH_SIZE):
            batch = all_reasoning_prompts[i:i+BATCH_SIZE]
            
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_REASONING,
                top_p=0.95,
                top_k=50,
            )
            outputs = llm.generate(batch, sampling_params, lora_request=lora_request, use_tqdm=False)
            
            for j, output in enumerate(outputs):
                all_reasonings[i+j] = output.outputs[0].text.strip()
            
            pbar.update(len(batch))
    
    # ========== 阶段2：生成代码 ==========
    print(f"\n[2/2] 生成代码 (共 {total_samples} 个)...")
    all_codes = [None] * len(all_reasonings)
    
    code_prompts = []
    code_indices = []
    
    for idx, (meta, reasoning) in enumerate(zip(all_metadata, all_reasonings)):
        if not reasoning:
            continue
        
        code_messages = [
            {"role": "system", "content": "You are an expert Python programmer. Based on the reasoning, write the code."},
            {"role": "user", "content": f"Reasoning:\n{reasoning}\n\nNow implement the function `{meta['func_name']}` that takes {meta['params']} to solve:\n\n{meta['question']}\n\nWrite only the Python code in ```python block."}
        ]
        code_prompt = tokenizer.apply_chat_template(code_messages, tokenize=False, add_generation_prompt=True)
        code_prompts.append(code_prompt)
        code_indices.append(idx)
    
    with tqdm(total=len(code_prompts), desc="代码进度", unit="个") as pbar:
        code_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_CODE,
            top_p=0.95,
            top_k=50,
        )
        
        for i in range(0, len(code_prompts), BATCH_SIZE):
            batch = code_prompts[i:i+BATCH_SIZE]
            outputs = llm.generate(batch, code_params, lora_request=lora_request, use_tqdm=False)
            
            for j, output in enumerate(outputs):
                idx = code_indices[i+j]
                all_codes[idx] = extract_code(output.outputs[0].text)
            
            pbar.update(len(batch))
    
    # ========== 评估 ==========
    print("\n评估代码...")
    problem_results = [[] for _ in range(len(dataset))]
    timeout_count = 0
    empty_code_count = 0
    
    with tqdm(total=len(all_metadata), desc="测试进度", unit="个") as pbar:
        for idx, (meta, code) in enumerate(zip(all_metadata, all_codes)):
            if not code:
                empty_code_count += 1
                passed = False
            else:
                passed, error = test_solution_code_with_timeout(code, meta["test_code"], TIMEOUT)
                if error and "超时" in str(error):
                    timeout_count += 1
            
            problem_results[meta["problem_idx"]].append(passed)
            pbar.update(1)
    
    # 统计信息
    total_generations = len(all_metadata)
    print(f"\n统计: 空代码={empty_code_count}/{total_generations} ({empty_code_count/total_generations*100:.1f}%), 超时={timeout_count}")
    
    # 计算 pass@k
    K_LIST = [1, 5, 10]
    results_passk = {}
    for k in K_LIST:
        scores = [pass_at_k(len(s), sum(s), k) for s in problem_results if len(s) == NUM_SAMPLES]
        results_passk[k] = sum(scores) / len(scores) if scores else float("nan")
    
    print("\n" + "=" * 70)
    print("两阶段推理评估结果")
    print("=" * 70)
    for k in K_LIST:
        v = results_passk[k]
        print(f"PASS@{k:<2}:             {v:.4f} ({v*100:.2f}%)")
    print("=" * 70)
    
    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, "eval_lora_cot_passk.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_problems": len(dataset),
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "pass_at_k": {f"pass@{k}": round(results_passk[k], 4) for k in K_LIST},
            "generation_stats": {
                "empty_code": empty_code_count,
                "timeouts": timeout_count,
            }
        }, f, indent=2)
    
    print(f"\n结果保存至: {output_file}")
    
    del llm
    gc.collect()