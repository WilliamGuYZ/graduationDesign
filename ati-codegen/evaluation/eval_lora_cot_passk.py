"""
CodeGeeX4-ALL-9B + LoRA CoT 两阶段推理评估脚本

实验矩阵（H1 ~ H10）中承担推理侧为 CoT 的 6 组：
  - H2 ：Base          + CoT-ZS       （--lora-path NONE --few-shot-k 0）
  - H3 ：Base          + CoT-FS(k)    （--lora-path NONE --few-shot-k k）
  - H5 ：LoRA_code     + CoT-ZS       （--lora-path <code adapter>  --few-shot-k 0）
  - H6 ：LoRA_code     + CoT-FS(k)    （--lora-path <code adapter>  --few-shot-k k）
  - H8 ：LoRA_cot_zs   + CoT-ZS       （默认读 latest_lora_cot_zs_adapter.txt，k=0）
  - H10：LoRA_cot_fs(k)+ CoT-FS(k)    （默认读 latest_lora_cot_fs{k}_adapter.txt，k 相等）

约束：若训练侧与推理侧同时为 CoT，则两侧示例数 k 必须相等（否则违反 train/test 分布一致性）。
对应在本脚本里：k>0 时不再回退到 zs adapter，必须先训好 LoRA_cot_fs(k)。
"""

import gc
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
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

MODEL_PATH   = os.path.join(_PROJECT_ROOT, "models", "CodeGeeX4-ALL-9B")
DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "KodCode_eval.jsonl")
RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "evaluation", "results")

# 评估参数
TIMEOUT = 5                         # 每次单测执行超时(秒)
MAX_TOKENS_REASONING = 1024         # 第一阶段“推理”最大生成长度
MAX_TOKENS_CODE = 1024              # 第二阶段“代码”最大生成长度
NUM_SAMPLES = 10                    # 每题采样次数（影响 pass@k 稳定性与耗时）
TEMPERATURE = 0.3                   # 采样温度
BATCH_SIZE = 4                      # vLLM 批大小（卡顿优先先调小）
MAX_REASONING_CHARS = 1200          # 推理文本字符截断上限（防过长 prompt）
MAX_REASONING_PROMPT_TOKENS = 256   # 推理文本 token 截断上限（防过长 prompt）

# vLLM 配置
VLLM_MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.9
SAMPLE_LIMIT: Optional[int] = None   # 调试用：仅评测前 N 题；正式评测保持 None
FEW_SHOT_K_DEFAULT = 0
FEW_SHOT_POOL_PATH = os.path.join(_PROJECT_ROOT, "data", "processed", "KodCode_train.jsonl")

_SUBPROC_SNIPPET = (
    "import json,sys;sys.path.insert(0,sys.argv[1]);"
    "from validate_code_with_test import test_solution_code;"
    "d=json.load(sys.stdin);"
    "r=test_solution_code(d['solution'],d['test'],[]);"
    "json.dump(list(r),sys.stdout)"
)


def _lora_pointer_file(few_shot_k: int) -> str:
    if few_shot_k == 0:
        filename = "latest_lora_cot_zs_adapter.txt"
    else:
        filename = f"latest_lora_cot_fs{few_shot_k}_adapter.txt"
    return os.path.join(_PROJECT_ROOT, "train", filename)


def _read_lora_adapter_dir(pointer_file: str) -> str:
    with open(pointer_file, "r", encoding="utf-8") as f:
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


def _clip_reasoning_before_code(text: str) -> str:
    """
    Stage-1 输出若提前滑出 ```python（或任何 ``` 代码块），立即截断保留前置自然语言段。
    对应 4.4.1 (1) 「避免推理阶段偷跑代码污染第二阶段上下文」。
    """
    if not text:
        return ""
    idx_py = text.find("```python")
    idx_any = text.find("```")
    cut_idx = min(i for i in [idx_py, idx_any] if i >= 0) if (idx_py >= 0 or idx_any >= 0) else -1
    if cut_idx >= 0:
        return text[:cut_idx].rstrip()
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


def _as_str_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def _few_shot_score(target: Dict[str, Any], cand: Dict[str, Any]) -> int:
    t_strong = set(_as_str_list(target.get("strong_tags")))
    c_strong = set(_as_str_list(cand.get("strong_tags")))
    t_tags = set(_as_str_list(target.get("tags")))
    c_tags = set(_as_str_list(cand.get("tags")))
    strong_overlap = len(t_strong & c_strong)
    tag_overlap = len(t_tags & c_tags)
    # 强标签权重更高，提升“算法骨架”一致性
    return strong_overlap * 2 + tag_overlap


def _build_instruction(question: str, func_name: str, params: str) -> str:
    return (
        f"Implement the function `{func_name}` that takes {params} to solve:\n\n"
        f"{question}\n\n"
        "Think step by step, then output the Python code."
    )


def _build_cot_response(item: Dict[str, Any]) -> str:
    thought = str(item.get("thought_step", "")).strip()
    code = str(item.get("solution", "")).strip()
    if thought:
        return f"{thought}\n\n```python\n{code}\n```"
    return f"```python\n{code}\n```"


def _select_few_shots(
    target: Dict[str, Any],
    pool: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    candidates = [
        p for p in pool
        if p.get("question") != target.get("question")
        and str(p.get("thought_step", "")).strip()
        and str(p.get("solution", "")).strip()
    ]
    if not candidates:
        return []
    scored = sorted(
        candidates,
        key=lambda x: _few_shot_score(target, x),
        reverse=True,
    )
    return scored


def _truncate_reasoning(
    reasoning_text: str,
    tokenizer: AutoTokenizer,
    max_reasoning_prompt_tokens: int,
    max_reasoning_chars: int,
) -> str:
    text = reasoning_text[:max_reasoning_chars]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_reasoning_prompt_tokens:
        return text
    ids = ids[:max_reasoning_prompt_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


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
    parser = argparse.ArgumentParser(description="CoT LoRA 两阶段评测（支持 few-shot）")
    parser.add_argument("--few-shot-k", type=int, default=FEW_SHOT_K_DEFAULT, help="每道题 few-shot 示例数量")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="",
        help="显式指定 LoRA 适配器路径；不传时按 few-shot-k 自动读取对应 latest 指针文件",
    )
    parser.add_argument("--few-shot-pool", type=str, default=FEW_SHOT_POOL_PATH, help="few-shot 示例库 JSONL 路径")
    parser.add_argument("--result-suffix", type=str, default="", help="结果文件后缀（如 cot_zs / cot_fs2）")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help=f"每题生成样本数（默认: {NUM_SAMPLES}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"vLLM 批大小（默认: {BATCH_SIZE}）")
    parser.add_argument(
        "--max-tokens-reasoning",
        type=int,
        default=MAX_TOKENS_REASONING,
        help=f"推理阶段最大生成 token（默认: {MAX_TOKENS_REASONING}）",
    )
    parser.add_argument(
        "--max-tokens-code",
        type=int,
        default=MAX_TOKENS_CODE,
        help=f"代码阶段最大生成 token（默认: {MAX_TOKENS_CODE}）",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=SAMPLE_LIMIT,
        help="仅评测前 N 题（默认: 全量）",
    )
    parser.add_argument(
        "--max-reasoning-chars",
        type=int,
        default=MAX_REASONING_CHARS,
        help=f"第二阶段拼接时最多保留的推理字符数（默认: {MAX_REASONING_CHARS}）",
    )
    parser.add_argument(
        "--max-reasoning-prompt-tokens",
        type=int,
        default=MAX_REASONING_PROMPT_TOKENS,
        help=f"第二阶段拼接时最多保留的推理 token 数（默认: {MAX_REASONING_PROMPT_TOKENS}）",
    )
    args = parser.parse_args()
    if args.few_shot_k < 0:
        raise ValueError("--few-shot-k 不能为负数")
    if args.num_samples <= 0:
        raise ValueError("--num-samples 必须 > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须 > 0")
    if args.max_tokens_reasoning <= 0 or args.max_tokens_code <= 0:
        raise ValueError("--max-tokens-reasoning / --max-tokens-code 必须 > 0")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit 需为正整数或不传")
    if args.max_reasoning_chars <= 0:
        raise ValueError("--max-reasoning-chars 必须 > 0")
    if args.max_reasoning_prompt_tokens <= 0:
        raise ValueError("--max-reasoning-prompt-tokens 必须 > 0")

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"未找到基座模型目录: {MODEL_PATH}")
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"未找到数据文件: {DATASET_PATH}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --lora-path NONE（不区分大小写）= 不加载任何 LoRA，在 Base 上跑 CoT（H2 / H3）
    use_lora = args.lora_path.strip().upper() != "NONE"

    pointer_file = _lora_pointer_file(args.few_shot_k)
    if not use_lora:
        lora_path = ""
    elif args.lora_path.strip():
        # 显式指定：用于 H5 / H6（指向 LoRA_code adapter）
        lora_path = os.path.abspath(args.lora_path.strip())
    else:
        # 默认走 train/ 下对应 k 的指针文件：H8 用 zs、H10 用 fs{k}。
        # 10 组设计要求 k 严格匹配，此处不再回退到 zs adapter。
        if not os.path.isfile(pointer_file):
            raise FileNotFoundError(
                f"未找到 LoRA 指针文件: {pointer_file}。\n"
                f"若要跑 few-shot-k={args.few_shot_k}，请先运行 "
                f"`python3 train/train_cot.py --num-few-shots {args.few_shot_k}` 训练对应 adapter；\n"
                "或用 --lora-path 显式指定其他 adapter（如 LoRA_code，对应 H5 / H6）；\n"
                "或用 --lora-path NONE 在基座上运行（对应 H2 / H3）。"
            )
        lora_path = _read_lora_adapter_dir(pointer_file)

    if use_lora and lora_path and not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA 适配器路径不是有效目录: {lora_path}")

    # ------- 非法组合硬拒绝（4.4.4 (3)）-------
    # 若挂载的 LoRA 是 CoT 适配器（训练时由 train_cot.py 写入 training_mode / num_few_shots），
    # 则要求训练 k* 与评测 --few-shot-k 严格一致：
    #   - LoRA_cot_zs  (k*=0) + --few-shot-k > 0  → 退出
    #   - LoRA_cot_fs(k*) + --few-shot-k ≠ k*      → 退出
    # LoRA_code（train.py 写入，无 training_mode 键）不触发该校验，允许 H5/H6 等跨脚本挂载。
    if use_lora and lora_path:
        _cfg_path = os.path.join(os.path.dirname(lora_path), "config.json")
        if os.path.isfile(_cfg_path):
            try:
                with open(_cfg_path, "r", encoding="utf-8") as _cf:
                    _adapter_cfg = json.load(_cf)
            except Exception:
                _adapter_cfg = {}
            _tm = _adapter_cfg.get("training_mode")
            _ak = _adapter_cfg.get("num_few_shots")
            if _tm == "cot_zero_shot" and args.few_shot_k > 0:
                raise SystemExit(
                    f"[非法组合] 挂载 LoRA_cot_zs (k*=0) 与 --few-shot-k={args.few_shot_k} 不匹配，\n"
                    f"            必须改用 --few-shot-k 0 或改挂 LoRA_cot_fs({args.few_shot_k}) 适配器。"
                )
            if _tm == "cot_few_shot" and isinstance(_ak, int) and _ak != args.few_shot_k:
                raise SystemExit(
                    f"[非法组合] 挂载 LoRA_cot_fs(k*={_ak}) 与 --few-shot-k={args.few_shot_k} 不匹配，\n"
                    f"            4.1 / 4.4.4 约束要求 train/test few-shot k 严格相等。"
                )

    if not use_lora:
        mode_tag = "Base"
    else:
        mode_tag = "LoRA"
    if args.few_shot_k == 0:
        mode_name = f"{mode_tag} + CoT zero-shot"
    else:
        mode_name = f"{mode_tag} + CoT few-shot (k={args.few_shot_k})"

    print("=" * 60)
    print(f"两阶段推理评估（{mode_name}）")
    print("=" * 60)
    if not use_lora:
        print("LoRA 指针文件:       不使用 LoRA（--lora-path NONE）")
        print("LoRA 路径:           <Base, 无 LoRA>")
    elif args.lora_path.strip():
        print("LoRA 指针文件:       手动覆盖（--lora-path）")
        print(f"LoRA 路径:           {lora_path}")
    else:
        print(f"LoRA 指针文件:       {pointer_file}")
        print(f"LoRA 路径:           {lora_path}")
    print(f"每问题样本数:        {args.num_samples}")
    print(f"批处理大小:          {args.batch_size}")
    print(f"Few-shot K:          {args.few_shot_k}")
    print(f"推理最大 token:       {args.max_tokens_reasoning}")
    print(f"代码最大 token:       {args.max_tokens_code}")
    print(f"推理拼接最大 token:    {args.max_reasoning_prompt_tokens}")
    print(f"推理截断字符数:       {args.max_reasoning_chars}")
    print(f"SAMPLE_LIMIT:        {args.sample_limit if args.sample_limit is not None else 'None'}")
    if args.num_samples < 10:
        print(f"注意: num_samples={args.num_samples}，PASS@10 将不会计算。")

    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n加载 vLLM" + (" + LoRA..." if use_lora else "（Base，无 LoRA）..."))
    _llm_kwargs: Dict[str, Any] = dict(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
        enable_lora=use_lora,
    )
    if use_lora:
        _llm_kwargs["max_lora_rank"] = 32
        _tok_cfg = os.path.join(lora_path, "tokenizer_config.json")
        _tok_json = os.path.join(lora_path, "tokenizer.json")
        if os.path.isdir(lora_path) and (os.path.isfile(_tok_cfg) or os.path.isfile(_tok_json)):
            _llm_kwargs["tokenizer"] = lora_path
    llm = LLM(**_llm_kwargs)
    lora_request = (
        LoRARequest(lora_name="eval_adapter", lora_int_id=1, lora_path=lora_path)
        if use_lora
        else None
    )

    sampling_reasoning = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=args.max_tokens_reasoning,
        top_p=0.95,
        top_k=50,
    )
    sampling_code = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=args.max_tokens_code,
        top_p=0.95,
        top_k=50,
    )

    print("\n加载数据集...")
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
            
            if not question or not test_code:
                continue
            
            dataset.append({
                "question": question,
                "test_code": test_code,
                "func_name": get_function_name_from_test_info(test_info),
                "params": get_parameter_list(test_info),
                "tags": row.get("tags", []),
                "strong_tags": row.get("strong_tags", []),
            })
    
    print(f"原始数据集大小: {len(dataset)} 条")
    
    if args.sample_limit:
        dataset = dataset[:args.sample_limit]
        print(f"SAMPLE_LIMIT={args.sample_limit}，仅评估前 {len(dataset)} 条")
    
    total_samples = len(dataset) * args.num_samples

    print("\n准备 few-shot 示例...")
    few_shot_pool: List[Dict[str, Any]] = []
    if args.few_shot_k > 0:
        if not os.path.isfile(args.few_shot_pool):
            raise FileNotFoundError(f"few-shot 示例库不存在: {args.few_shot_pool}")
        with open(args.few_shot_pool, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = clean_question(row.get("question"))
                ti = row.get("test_info", [])
                few_shot_pool.append({
                    "question": q,
                    "func_name": get_function_name_from_test_info(ti),
                    "params": get_parameter_list(ti),
                    "thought_step": row.get("thought_step", ""),
                    "solution": row.get("solution", ""),
                    "tags": row.get("tags", []),
                    "strong_tags": row.get("strong_tags", []),
                })
        print(f"few-shot 示例库:      {len(few_shot_pool)} 条（来自 {args.few_shot_pool}）")
    else:
        print("few-shot 示例库:      不使用（zero-shot）")

    print("\n构建 Prompts...")
    all_reasoning_prompts = []
    all_metadata = []
    
    for problem_idx, data in enumerate(dataset):
        ranked_shots = _select_few_shots(data, few_shot_pool, args.few_shot_k) if args.few_shot_k > 0 else []
        top_pool_size = min(len(ranked_shots), max(args.few_shot_k, args.few_shot_k * 4))
        top_ranked_pool = ranked_shots[:top_pool_size]

        for sample_idx in range(args.num_samples):
            if args.few_shot_k > 0 and top_ranked_pool:
                rng = random.Random(2026 + problem_idx * 1000 + sample_idx)
                if len(top_ranked_pool) <= args.few_shot_k:
                    selected_shots = top_ranked_pool
                else:
                    selected_shots = rng.sample(top_ranked_pool, args.few_shot_k)
            else:
                selected_shots = []
            reasoning_messages = [
                {"role": "system", "content": "You are an expert algorithm teacher. Generate step-by-step reasoning."},
            ]
            # Stage-1 shots：按 4.4.2 (3)「题目→推理→代码块」注入（assistant = thought + code block）
            for shot in selected_shots:
                reasoning_messages.append({
                    "role": "user",
                    "content": _build_instruction(shot["question"], shot["func_name"], shot["params"]),
                })
                reasoning_messages.append({
                    "role": "assistant",
                    "content": _build_cot_response(shot),
                })
            # 目标题：与 Direct 一致的 instruction + "Think step by step..." 后缀（已在 _build_instruction 内）
            reasoning_messages.append({
                "role": "user",
                "content": _build_instruction(data["question"], data["func_name"], data["params"]),
            })
            reasoning_prompt = tokenizer.apply_chat_template(reasoning_messages, tokenize=False, add_generation_prompt=True)
            
            all_reasoning_prompts.append(reasoning_prompt)
            all_metadata.append({
                "problem_idx": problem_idx,
                "sample_idx": sample_idx,
                "func_name": data["func_name"],
                "params": data["params"],
                "question": data["question"],
                "test_code": data["test_code"],
                "few_shots": selected_shots,
            })
    
    print(f"\n[1/2] 生成推理 (共 {total_samples} 个)...")
    all_reasonings = [None] * len(all_reasoning_prompts)

    reasoning_t0 = time.perf_counter()
    with tqdm(total=len(all_reasoning_prompts), desc="推理进度", unit="个") as pbar:
        for i in range(0, len(all_reasoning_prompts), args.batch_size):
            batch = all_reasoning_prompts[i:i+args.batch_size]
            outputs = llm.generate(batch, sampling_reasoning, lora_request=lora_request, use_tqdm=False)
            for j, output in enumerate(outputs):
                # 在 ```python 之前截断，阻止 Stage-1 偷跑代码污染 Stage-2 上下文（4.4.1 (1)）
                all_reasonings[i+j] = _clip_reasoning_before_code(output.outputs[0].text)
            pbar.update(len(batch))
            elapsed = max(time.perf_counter() - reasoning_t0, 1e-6)
            rate = pbar.n / elapsed
            pbar.set_postfix_str(f"{rate:.2f}个/秒", refresh=False)

    print(f"\n[2/2] 生成代码 (共 {total_samples} 个)...")
    all_codes = [None] * len(all_reasonings)
    empty_reasoning_count = 0
    
    code_prompts = []
    code_indices = []
    
    for idx, (meta, reasoning) in enumerate(zip(all_metadata, all_reasonings)):
        if not reasoning:
            empty_reasoning_count += 1
        reasoning_text = reasoning if reasoning else "No reasoning available. Solve directly from the problem statement."
        
        code_messages = [
            {"role": "system", "content": "You are an expert Python programmer. Based on the reasoning, write the code."},
        ]
        for shot in meta.get("few_shots", []):
            shot_instruction = _build_instruction(shot["question"], shot["func_name"], shot["params"])
            code_messages.append({"role": "user", "content": shot_instruction})
            code_messages.append({"role": "assistant", "content": _build_cot_response(shot)})
        short_reasoning = _truncate_reasoning(
            reasoning_text=reasoning_text,
            tokenizer=tokenizer,
            max_reasoning_prompt_tokens=args.max_reasoning_prompt_tokens,
            max_reasoning_chars=args.max_reasoning_chars,
        )
        code_messages.append(
            {
                "role": "user",
                "content": (
                    f"Reasoning:\n{short_reasoning}\n\n"
                    f"Now implement the function `{meta['func_name']}` that takes {meta['params']} to solve:\n\n"
                    f"{meta['question']}\n\n"
                    "Write only the Python code in ```python block."
                ),
            }
        )
        code_prompt = tokenizer.apply_chat_template(code_messages, tokenize=False, add_generation_prompt=True)
        code_prompts.append(code_prompt)
        code_indices.append(idx)
    
    code_t0 = time.perf_counter()
    with tqdm(total=len(code_prompts), desc="代码进度", unit="个") as pbar:
        for i in range(0, len(code_prompts), args.batch_size):
            batch = code_prompts[i:i+args.batch_size]
            outputs = llm.generate(batch, sampling_code, lora_request=lora_request, use_tqdm=False)
            for j, output in enumerate(outputs):
                idx = code_indices[i+j]
                all_codes[idx] = extract_code(output.outputs[0].text)
            pbar.update(len(batch))
            elapsed = max(time.perf_counter() - code_t0, 1e-6)
            rate = pbar.n / elapsed
            pbar.set_postfix_str(f"{rate:.2f}个/秒", refresh=False)

    print("\n评估代码...")
    problem_results = [[] for _ in range(len(dataset))]
    timeout_count = 0
    empty_code_count = 0
    
    test_t0 = time.perf_counter()
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
            elapsed = max(time.perf_counter() - test_t0, 1e-6)
            rate = pbar.n / elapsed
            pbar.set_postfix_str(f"{rate:.2f}个/秒", refresh=False)
    
    # 统计信息
    total_generations = len(all_metadata)
    print(
        f"\n统计: 空推理={empty_reasoning_count}/{total_generations} ({empty_reasoning_count/total_generations*100:.1f}%), "
        f"空代码={empty_code_count}/{total_generations} ({empty_code_count/total_generations*100:.1f}%), "
        f"超时={timeout_count}"
    )
    
    # 计算 pass@k
    K_LIST = [k for k in [1, 5, 10] if k <= args.num_samples]
    results_passk = {}
    for k in K_LIST:
        scores = [pass_at_k(len(s), sum(s), k) for s in problem_results if len(s) == args.num_samples]
        results_passk[k] = sum(scores) / len(scores) if scores else float("nan")
    
    print("\n" + "=" * 60)
    print(f"两阶段推理评估结果（{mode_name}）")
    print("=" * 60)
    for k in K_LIST:
        v = results_passk[k]
        print(f"PASS@{k:<2}:             {v:.4f} ({v*100:.2f}%)")
    print("=" * 60)
    
    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.result_suffix.strip()}" if args.result_suffix.strip() else ""
    output_file = os.path.join(RESULTS_DIR, f"eval_lora_cot_passk{suffix}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "mode": mode_name,
            "use_lora": use_lora,
            "lora_path": lora_path if use_lora else "",
            "num_problems": len(dataset),
            "num_samples": args.num_samples,
            "few_shot_k": args.few_shot_k,
            "batch_size": args.batch_size,
            "pass_at_k": {f"pass@{k}": round(results_passk[k], 4) for k in K_LIST},
            "generation_stats": {
                "empty_reasoning": empty_reasoning_count,
                "empty_code": empty_code_count,
                "timeouts": timeout_count,
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存至: {output_file}")
    
    del llm
    gc.collect()


if __name__ == "__main__":
    main()