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
DATASET_PATH = "./data/processed/eval_data.jsonl"

TIMEOUT = 3
MAX_TOKENS = 512
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
    stop=["```"]
)

# =========================
# PROMPT
# =========================

def build_prompt(problem):

    return f"""You are a Python expert programmer.

Write a Python function to solve the following problem.

Problem:
{problem}

Return only Python code.
"""

# =========================
# CODE EXTRACTION
# =========================

def extract_code(text):

    code_block = re.findall(r"```python(.*?)```", text, re.S)

    if code_block:
        return code_block[0]

    code_block = re.findall(r"```(.*?)```", text, re.S)

    if code_block:
        return code_block[0]

    return text


# =========================
# RUN TESTS
# =========================

def run_test(code, tests):

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


# =========================
# LOAD DATASET
# =========================

dataset = []

with open(DATASET_PATH) as f:
    for line in f:
        dataset.append(json.loads(line))

print("Dataset size:", len(dataset))

# =========================
# GENERATION
# =========================

prompts = [build_prompt(d["text"]) for d in dataset]

outputs = llm.generate(prompts, sampling_params)

# =========================
# EVALUATION
# =========================

correct = 0

for data, output in tqdm(zip(dataset, outputs), total=len(dataset)):

    generated = output.outputs[0].text

    code = extract_code(generated)

    passed = run_test(code, data["test_list"])

    if passed:
        correct += 1


# =========================
# RESULT
# =========================

total = len(dataset)

pass1 = correct / total

print("\n====================")
print("MBPP RESULT")
print("====================")
print(f"Correct: {correct}/{total}")
print(f"pass@1: {pass1:.4f}")