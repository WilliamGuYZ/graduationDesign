import os

# ===== 必须环境变量 =====
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import sys
from io import StringIO
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from math import comb

from vllm import LLM, SamplingParams


# =============================
# CONFIG
# =============================

MODEL_PATH = "../models/Qwen2.5-Coder-7B"
DATASET_PATH = "../data/processed/eval_code.jsonl"

K = 10
MAX_NEW_TOKENS = 768

BATCH_SIZE = 32
TEST_WORKERS = 32
TIMEOUT = 3


# =============================
# DATASET
# =============================

def load_dataset(path):

    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
            
    dataset = dataset[:5]
    return dataset


# =============================
# PROMPT
# =============================

def build_prompt(problem):

    return f"""
You are an expert competitive programmer.

Solve the following problem.

Write a correct and efficient Python program.

The program must read from standard input
and write to standard output.

Do not include explanation.

Problem:
{problem}

Python code:
"""


# =============================
# CODE EXTRACTION
# =============================

def extract_code(text):

    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)

    pattern = r"```\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)

    return text


def clean_code(code):

    lines = code.split("\n")

    start = 0

    for i, l in enumerate(lines):

        if (
            l.startswith("import")
            or l.startswith("from")
            or l.startswith("def")
        ):
            start = i
            break

    return "\n".join(lines[start:])


# =============================
# OUTPUT NORMALIZATION
# =============================

def normalize_output(s):

    return " ".join(s.strip().split())


# =============================
# SANDBOX EXECUTION
# =============================

def run_single_test(code, test):

    try:

        input_data = test["input"]
        expected = normalize_output(test["output"])

        stdin_backup = sys.stdin
        stdout_backup = sys.stdout

        sys.stdin = StringIO(input_data)
        sys.stdout = StringIO()

        exec(code, {"__name__": "__main__"})

        output = sys.stdout.getvalue()

        sys.stdin = stdin_backup
        sys.stdout = stdout_backup

        output = normalize_output(output)

        return output == expected

    except Exception:

        return False


def run_code_tests(code, tests):

    for test in tests:

        queue = mp.Queue()

        p = mp.Process(
            target=lambda q: q.put(run_single_test(code, test)),
            args=(queue,)
        )

        p.start()
        p.join(TIMEOUT)

        if p.is_alive():

            p.terminate()
            p.join()

            return False

        if queue.empty():

            return False

        if not queue.get():

            return False

    return True


# =============================
# PASS@K
# =============================

def pass_at_k(n, c, k):

    if n - c < k:
        return 1.0

    return 1 - comb(n - c, k) / comb(n, k)


# =============================
# MAIN
# =============================

def main():

    print("Loading model...")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
        enable_prefix_caching=True,
        trust_remote_code=True,
        disable_log_stats=True
    )

    dataset = load_dataset(DATASET_PATH)

    print("Dataset size:", len(dataset))

    prompts = []

    for sample in dataset:

        if "instruction" in sample:
            problem = sample["instruction"] + "\n" + sample["input"]
        else:
            problem = sample["input"]

        prompts.append(build_prompt(problem))

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        n=K
    )

    # =============================
    # INFERENCE
    # =============================

    print("Generating solutions...")

    outputs = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Inference"):

        batch = prompts[i:i+BATCH_SIZE]

        result = llm.generate(batch, sampling_params)

        outputs.extend(result)

    # =============================
    # PREPARE TEST TASKS
    # =============================

    print("Preparing test cases...")

    tasks = []

    for out, sample in zip(outputs, dataset):

        tests = sample["tests"]

        for o in out.outputs:

            code = extract_code(o.text)
            code = clean_code(code)

            tasks.append((code, tests))

    # =============================
    # RUN TESTS
    # =============================

    print("Running tests...")

    results = []

    with ProcessPoolExecutor(max_workers=TEST_WORKERS) as executor:

        futures = []

        for code, tests in tasks:
            futures.append(executor.submit(run_code_tests, code, tests))

        for f in tqdm(futures, desc="Testing"):
            results.append(f.result())

    # =============================
    # COMPUTE PASS@K
    # =============================

    scores = []

    idx = 0

    for _ in dataset:

        problem_results = results[idx:idx+K]

        c = sum(problem_results)

        scores.append(pass_at_k(K, c, K))

        idx += K

    score = sum(scores) / len(scores)

    print("\n===== RESULT =====")

    print(f"pass@{K}: {score:.4f}")

    correct = sum([any(results[i:i+K]) for i in range(0, len(results), K)])

    print(f"Solved problems: {correct}/{len(dataset)}")


# =============================
# ENTRY
# =============================

if __name__ == "__main__":

    main()