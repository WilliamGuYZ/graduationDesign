"""
Qwen2.5-Coder-7B-Instruct + LoRA 评估脚本
数据格式与 train.py / eval_base_passk.py 一致:
{"question": "...", "solution": "...", "test": "...", "test_info": [...]}
"""

import gc
import json
import os
import re
import subprocess
import sys
import warnings
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
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_PROJECT_ROOT, "scripts")
LORA_POINTER_FILE = os.path.join(_PROJECT_ROOT, "train", "latest_lora_adapter.txt")

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen2.5-Coder-7B-Instruct")
DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "KodCode_eval.jsonl")
RESULT_PATH = os.path.join(_PROJECT_ROOT, "evaluation", "results", "eval_lora_passk.txt")
DEBUG_PATH = os.path.join(_PROJECT_ROOT, "evaluation", "results", "debug_lora_samples.json")

TIMEOUT = 5
MAX_TOKENS = 1024
NUM_SAMPLES = 10
TEMPERATURE = 0.7

VLLM_MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.9

DEBUG = False
DEBUG_SAMPLE_COUNT = None

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
# PROMPT（与 train.py 一致）
# =========================

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write correct Python code to solve the given problem."
)


def build_instruction(question: str, func_name: str, params: str) -> str:
    return f"""Implement the function `{func_name}` that takes {params} to solve:

{question}

Only output the Python code, no explanation."""


def build_chat_prompt(
    tokenizer: AutoTokenizer, question: str, func_name: str, params: str
) -> str:
    instruction = build_instruction(question, func_name, params)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =========================
# TEST EXECUTION（子进程 + 超时，与 eval_base_passk 一致）
# =========================


def test_solution_code_with_timeout(
    solution_code: str, test_code: str, timeout: int = 5
) -> Tuple[bool, Optional[str]]:
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
    lora_path = _read_lora_adapter_dir()

    print("=" * 60)
    print("加载 Tokenizer...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 60)
    print("加载 vLLM + LoRA...")
    print("=" * 60)
    print("LoRA 路径:", lora_path)

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
    if os.path.isdir(lora_path) and (
        os.path.isfile(_tok_cfg) or os.path.isfile(_tok_json)
    ):
        _llm_kwargs["tokenizer"] = lora_path

    llm = LLM(**_llm_kwargs)

    lora_request = LoRARequest(
        lora_name="eval_adapter",
        lora_int_id=1,
        lora_path=lora_path,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=0.95,
        top_k=50,
        stop=None,
    )

    print("\n" + "=" * 60)
    print("加载数据集...")
    print("=" * 60)

    dataset: List[Dict[str, Any]] = []
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
            dataset.append(
                {
                    "question": question,
                    "test_code": test_code,
                    "solution": solution,
                    "func_name": get_function_name_from_test_info(test_info),
                    "params": get_parameter_list(test_info),
                }
            )

    print(f"原始数据集大小: {len(dataset)} 条")

    if DEBUG and DEBUG_SAMPLE_COUNT is not None:
        dataset = dataset[:DEBUG_SAMPLE_COUNT]
        print(f"调试模式: 只测试前 {len(dataset)} 条")

    print("\n" + "=" * 60)
    print("验证原始 solution 代码...")
    print("=" * 60)

    solution_pass_count = 0
    solution_fail_details: List[Dict[str, Any]] = []

    for idx, data in enumerate(dataset):
        passed, error, _stdout, _stderr, program_preview = execute_test_with_debug(
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

    print("\n" + "=" * 60)
    print("构建 Prompts...")
    print("=" * 60)

    all_prompts: List[str] = []
    prompt_to_problem_idx: List[int] = []

    for idx, data in enumerate(dataset):
        prompt = build_chat_prompt(
            tokenizer, data["question"], data["func_name"], data["params"]
        )
        for _ in range(NUM_SAMPLES):
            all_prompts.append(prompt)
            prompt_to_problem_idx.append(idx)

    print(f"总生成请求: {len(all_prompts)} 个")

    print("\n" + "=" * 60)
    print("开始生成（LoRA）...")
    print("=" * 60)

    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=lora_request,
    )

    print("\n" + "=" * 60)
    print("评估中...")
    print("=" * 60)

    problem_results: List[List[bool]] = [[] for _ in range(len(dataset))]
    debug_samples: List[Dict[str, Any]] = []
    timeout_count = 0

    for data_idx, output in tqdm(
        zip(prompt_to_problem_idx, outputs), total=len(all_prompts)
    ):
        generated = output.outputs[0].text
        code = extract_code(generated)

        passed, error, _o, _e, _p = execute_test_with_debug(
            code,
            dataset[data_idx]["test_code"],
            dataset[data_idx]["func_name"],
            data_idx,
        )
        if error and "超时" in str(error):
            timeout_count += 1

        problem_results[data_idx].append(passed)

        if DEBUG and len([s for s in debug_samples if s["index"] == data_idx]) == 0:
            debug_samples.append(
                {
                    "index": data_idx,
                    "func_name": dataset[data_idx]["func_name"],
                    "question_preview": dataset[data_idx]["question"][:300],
                    "generated_code_preview": code[:500] if code else "空",
                    "passed": passed,
                    "error": error,
                }
            )

    if DEBUG:
        os.makedirs(os.path.dirname(DEBUG_PATH), exist_ok=True)
        with open(DEBUG_PATH, "w", encoding="utf-8") as f:
            json.dump(debug_samples, f, indent=2, ensure_ascii=False)
        print(f"\n调试信息已保存至: {DEBUG_PATH}")

    if timeout_count > 0:
        print(
            f"\n⚠️ 超时统计: {timeout_count}/{len(all_prompts)} 次 "
            f"({timeout_count/len(all_prompts)*100:.1f}%)"
        )

    total_correct_pass1 = sum(
        1 for samples in problem_results if samples and samples[0]
    )
    total_correct_pass5 = sum(
        1 for samples in problem_results if len(samples) >= 5 and any(samples[:5])
    )
    total_correct_pass10 = sum(1 for samples in problem_results if any(samples))

    total_problems = len(dataset)
    pass1 = total_correct_pass1 / total_problems if total_problems > 0 else 0
    pass5 = total_correct_pass5 / total_problems if total_problems > 0 else 0
    pass10 = total_correct_pass10 / total_problems if total_problems > 0 else 0

    print("\n" + "=" * 60)
    print("PASS@K 评估结果（LoRA）")
    print("=" * 60)
    print(f"总问题数:            {total_problems}")
    print(f"每问题样本数:        {NUM_SAMPLES}")
    print(f"温度参数:            {TEMPERATURE}")
    print(f"测试超时:            {TIMEOUT} 秒")
    print(f"LoRA 路径:           {lora_path}")
    print("-" * 60)
    print(f"PASS@1:              {pass1:.4f} ({pass1 * 100:.2f}%)")
    print(f"  正确数:            {total_correct_pass1}/{total_problems}")
    print("-" * 60)
    print(f"PASS@5:              {pass5:.4f} ({pass5 * 100:.2f}%)")
    print(f"  正确数:            {total_correct_pass5}/{total_problems}")
    print("-" * 60)
    print(f"PASS@10:             {pass10:.4f} ({pass10 * 100:.2f}%)")
    print(f"  正确数:            {total_correct_pass10}/{total_problems}")
    print("=" * 60)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(f"总问题数:            {total_problems}\n")
        f.write(f"每问题样本数:        {NUM_SAMPLES}\n")
        f.write(f"温度参数:            {TEMPERATURE}\n")
        f.write(f"测试超时:            {TIMEOUT} 秒\n")
        f.write(f"LoRA 路径:           {lora_path}\n")
        f.write(f"PASS@1:              {pass1:.4f} ({pass1 * 100:.2f}%)\n")
        f.write(f"  正确数:            {total_correct_pass1}/{total_problems}\n")
        f.write(f"PASS@5:              {pass5:.4f} ({pass5 * 100:.2f}%)\n")
        f.write(f"  正确数:            {total_correct_pass5}/{total_problems}\n")
        f.write(f"PASS@10:             {pass10:.4f} ({pass10 * 100:.2f}%)\n")
        f.write(f"  正确数:            {total_correct_pass10}/{total_problems}\n")

    print(f"\n结果已保存至: {RESULT_PATH}")

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
