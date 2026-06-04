"""
CodeGeeX4-ALL-9B 评估脚本
数据格式: {"question": "...", "solution": "...", "thought_step": "...", "test": "...", "test_info": [...], "tags": [...]}
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 屏蔽 INFO 日志
# =========================
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_LOGGING_PREFIX"] = ""

import logging
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_PROJECT_ROOT, "scripts")

MODEL_PATH   = os.path.join(_PROJECT_ROOT, "models", "CodeGeeX4-ALL-9B")
DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "KodCode_eval.jsonl")
RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "evaluation", "results")

TIMEOUT     = 5
MAX_TOKENS  = 1024
NUM_SAMPLES = 10
TEMPERATURE = 0.3

VLLM_MAX_MODEL_LEN     = 2048
GPU_MEMORY_UTILIZATION = 0.9

# SAMPLE_LIMIT 设为整数可快速跑通流程（None = 全量）
SAMPLE_LIMIT: Optional[int] = None

# 子进程内复用 scripts/validate_code_with_test.py
_SUBPROC_SNIPPET = (
    "import json,sys;sys.path.insert(0,sys.argv[1]);"
    "from validate_code_with_test import test_solution_code;"
    "d=json.load(sys.stdin);"
    "r=test_solution_code(d['solution'],d['test'],[]);"
    "json.dump(list(r),sys.stdout)"
)


# =========================
# PASS@K（HumanEval 无偏估计）
# =========================


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏 pass@k 估计量（Chen et al., 2021）。"""
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# =========================
# BASIC HELPERS
# =========================

def clean_question(text: Any) -> str:
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_code(text: str) -> str:
    code_blocks = re.findall(r"```python(.*?)```", text, re.S)
    if not code_blocks:
        code_blocks = re.findall(r"```(.*?)```", text, re.S)
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


def get_function_name_from_test_info(test_info: List[Dict]) -> str:
    if test_info and len(test_info) > 0:
        return test_info[0].get("function_name", "solution")
    return "solution"


def get_parameter_list(test_info: List[Dict]) -> str:
    if test_info and len(test_info) > 0:
        return test_info[0].get("parameter_list", "")
    return ""


# =========================
# PROMPT BUILDING
# =========================

SYSTEM_PROMPT = "You are an expert Python programmer. Write correct Python code to solve the given problem."


def build_instruction(question: str, func_name: str, params: str) -> str:
    return f"""Implement the function `{func_name}` that takes {params} to solve:

{question}

Only output the Python code, no explanation."""


def build_chat_prompt(tokenizer: AutoTokenizer, question: str, func_name: str, params: str) -> str:
    instruction = build_instruction(question, func_name, params)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =========================
# TEST EXECUTION（子进程 + 超时）
# =========================

def test_solution_code_with_timeout(solution_code: str, test_code: str, timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """子进程运行 test_solution_code，超时返回失败。"""
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
            return bool(pair[0]), pair[1]
        except json.JSONDecodeError:
            return False, f"输出解析失败: {out[:200]}"
    err = (result.stderr or result.stdout or f"退出码 {result.returncode}")[-500:]
    return False, err


def execute_test_with_debug(
    code: str, test_code: str, func_name: str, problem_idx: int = -1
) -> Tuple[bool, Optional[str], str, str, str]:
    """验证 / 调试用：返回 (passed, error, stdout, stderr, preview)。"""
    if not code:
        return False, "代码为空", "", "", ""
    if not test_code:
        return False, "测试代码为空", "", "", ""
    preview = (code[:400] + "\n# --- test ---\n" + test_code[:400]).strip()
    if len(code) + len(test_code) > 800:
        preview = preview[:800] + "..."
    ok, err = test_solution_code_with_timeout(code, test_code, TIMEOUT)
    if ok:
        return True, None, "", "", preview
    err = err or "测试未通过"
    return False, err, "", err, preview


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Base 模型 pass@k 评测（H1）")
    parser.add_argument(
        "--result-suffix",
        type=str,
        default="",
        help="结果文件名后缀，写入 eval_base_passk_<suffix>.json；不传则仍为 eval_base_passk.json",
    )
    args = parser.parse_args()

    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATASET_PATH}")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"未找到基座模型目录: {MODEL_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n" + "=" * 60)
    print("加载 vLLM 模型...")
    print("=" * 60)
    
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
    )
    
    sampling_params = SamplingParams(
        n=NUM_SAMPLES,              # 一个 prompt 内部采样 NUM_SAMPLES 次，相比外层复制 prompt 效率高 2-4×
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=0.95,
        top_k=50,
        stop=None,
    )
    
    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)
    
    dataset = []
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATASET_PATH}")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = clean_question(row.get("question"))
            test_code = row.get("test", "")
            test_info = row.get("test_info", [])
            solution = row.get("solution", "")
            
            if not question or not test_code:
                continue
            dataset.append({
                "question": question,
                "test_code": test_code,
                "solution": solution,
                "func_name": get_function_name_from_test_info(test_info),
                "params": get_parameter_list(test_info),
            })
    
    print(f"原始数据集大小: {len(dataset)} 条")

    if SAMPLE_LIMIT:
        dataset = dataset[:SAMPLE_LIMIT]
        print(f"SAMPLE_LIMIT={SAMPLE_LIMIT}，仅评估前 {len(dataset)} 条")

    # =========================================================
    # 验证原始 solution 是否能通过测试
    # =========================================================
    print("\n" + "=" * 60)
    print("验证原始 solution 代码...")
    print("=" * 60)
    
    solution_pass_count = 0
    solution_fail_details = []
    
    for idx, data in enumerate(dataset):
        passed, error, stdout, stderr, program_preview = execute_test_with_debug(
            data["solution"], data["test_code"], data["func_name"], idx
        )
        if passed:
            solution_pass_count += 1
        else:
            solution_fail_details.append(
                {
                    "index": idx,
                    "func_name": data["func_name"],
                    "error": error,
                    "preview": program_preview[:200] if program_preview else "",
                }
            )

    n_ds = len(dataset)
    pct = (solution_pass_count / n_ds * 100) if n_ds else 0.0
    print(f"\n原始 solution 通过率: {solution_pass_count}/{n_ds} ({pct:.1f}%)")
    
    # =========================================================
    # 模型生成
    # =========================================================
    print("\n" + "=" * 60)
    print("构建 Prompts...")
    print("=" * 60)
    
    all_prompts = []
    prompt_to_problem_idx = []
    
    for idx, data in enumerate(dataset):
        prompt = build_chat_prompt(tokenizer, data["question"], data["func_name"], data["params"])
        all_prompts.append(prompt)
        prompt_to_problem_idx.append(idx)
    
    print(f"总生成请求: {len(all_prompts)} 个 × n={NUM_SAMPLES} samples/prompt")
    
    print("\n" + "=" * 60)
    print("开始生成...")
    print("=" * 60)
    
    outputs = llm.generate(all_prompts, sampling_params)
    
    # =========================================================
    # 评估生成的代码
    # =========================================================
    print("\n" + "=" * 60)
    print("评估中...")
    print("=" * 60)
    
    problem_results: List[List[bool]] = [[] for _ in range(len(dataset))]
    timeout_count = 0

    total_samples = len(all_prompts) * NUM_SAMPLES
    pbar = tqdm(total=total_samples)
    for data_idx, output in zip(prompt_to_problem_idx, outputs):
        for completion in output.outputs:      # n=NUM_SAMPLES 时每个 output 带 NUM_SAMPLES 条 completion
            generated = completion.text
            code = extract_code(generated)
            passed, error, _, _, _ = execute_test_with_debug(
                code, dataset[data_idx]["test_code"], dataset[data_idx]["func_name"], data_idx
            )
            if error and "超时" in str(error):
                timeout_count += 1
            problem_results[data_idx].append(passed)
            pbar.update(1)
    pbar.close()

    if timeout_count:
        print(f"\n⚠️ 超时: {timeout_count}/{total_samples} 次 ({timeout_count/total_samples*100:.1f}%)")

    # ── pass@k（无偏估计）────────────────────────────────────────────────────
    K_LIST         = [1, 5, 10]
    total_problems = len(dataset)
    results_passk: Dict[int, float] = {}
    for k in K_LIST:
        scores = [
            pass_at_k(len(s), sum(s), k)
            for s in problem_results
            if len(s) == NUM_SAMPLES
        ]
        results_passk[k] = sum(scores) / len(scores) if scores else float("nan")

    print("\n" + "=" * 60)
    print("PASS@K 评估结果（Base）")
    print("=" * 60)
    print(f"时间戳:              {timestamp}")
    print(f"总问题数:            {total_problems}")
    print(f"每问题样本数:        {NUM_SAMPLES}")
    print(f"温度参数:            {TEMPERATURE}")
    print(f"测试超时:            {TIMEOUT} 秒")
    print("-" * 60)
    for k in K_LIST:
        v = results_passk[k]
        if v != v:
            print(f"PASS@{k:<2}:             N/A")
        else:
            print(f"PASS@{k:<2}:             {v:.4f} ({v*100:.2f}%)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix = f"_{args.result_suffix.strip()}" if args.result_suffix.strip() else ""
    output_file = os.path.join(RESULTS_DIR, f"eval_base_passk{suffix}.json")

    # 构建结果字典（只包含 summary）
    results = {
        "timestamp": timestamp,
        "dataset": DATASET_PATH,
        "num_problems": total_problems,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "timeout": TIMEOUT,
        "pass_at_k": {f"pass@{k}": round(results_passk[k], 4) for k in K_LIST},
        "original_solution_pass_rate": round(solution_pass_count / n_ds, 4) if n_ds else 0,
    }

    # 保存为格式化的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_file}")
    
    # 清理
    del llm
    gc.collect()


if __name__ == "__main__":
    main()