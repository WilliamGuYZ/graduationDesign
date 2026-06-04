"""
批量为 KodCode.jsonl 中的每条 solution 生成 thought_step。

使用方式：
    python scripts/annotate_thought_steps.py

机制：
    - 路径、模型、并发数等均在文件顶部配置区修改。
    - 已写入 output 的条目（按 question 去重）自动跳过，支持断点续标。
    - 生成失败时重试最多 MAX_RETRY 次，仍失败则 thought_step 为空字符串并记录日志。
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# ======================== 配置（直接在此修改） ========================
OPENAI_API_KEY  = "sk-91de952175a247ffba72f005df1e6711"   # 填入你的 API Key
OPENAI_BASE_URL = "https://api.deepseek.com"              # DeepSeek 官方地址

DEFAULT_MODEL   = "deepseek-chat"
DEFAULT_INPUT   = "data/raw/KodCode.jsonl"
DEFAULT_OUTPUT  = "data/raw/KodCode_with_thought.jsonl"
DEFAULT_WORKERS = 32           

MAX_RETRY          = 3  
RETRY_DELAY        = 2 
MAX_THOUGHT_TOKENS = 512
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert algorithm teacher. "
    "Given a coding problem and its correct Python solution, "
    "generate a clear, concise step-by-step thought process that naturally leads to that solution. "
    "Focus on the reasoning and algorithmic insight, not on restating the code."
)

USER_PROMPT_TEMPLATE = """\
Problem: {question}

Solution:
```python
{solution}
```

Output 3-5 numbered steps (Step N: [Tag] - explanation). Tags: Understand/Approach/Edge Cases/Implement/Verify. Steps only, no extra text. Do NOT output any code, code blocks, or ```python fences — describe logic in prose only."""


def load_done_set(output_path: str) -> set[str]:
    """读取已标注文件，返回已完成条目的 question 集合（用于断点续标）。"""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj["question"])
                except (json.JSONDecodeError, KeyError):
                    pass
    logger.info("断点续标：已完成 %d 条，自动跳过", len(done))
    return done


def call_openai(client: OpenAI, model: str, question: str, solution: str) -> str:
    """调用 OpenAI API 生成 thought_step，带指数退避重试。"""
    user_content = USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        solution=solution.strip(),
    )
    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=MAX_THOUGHT_TOKENS,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate limit" in str(exc).lower()
            wait = 15 if is_rate_limit else RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRY:
                logger.warning("第 %d 次调用失败（%s），%.0fs 后重试...", attempt, exc, wait)
                time.sleep(wait)
            else:
                logger.error("第 %d 次调用仍失败，跳过该条目：%s", attempt, exc)
                return ""


def process_one(args_tuple):
    """线程工作函数，返回 (原始 item dict, thought_step str)。"""
    client, model, item = args_tuple
    thought = call_openai(client, model, item["question"], item["solution"])
    return item, thought


def main():
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key or api_key.startswith("sk-xxx"):
        raise ValueError("请在文件顶部配置区填入有效的 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)

    # 读取输入
    with open(DEFAULT_INPUT, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    logger.info("读取输入：共 %d 条", len(all_items))

    # 断点续标
    done_set = load_done_set(DEFAULT_OUTPUT)
    pending = [item for item in all_items if item["question"] not in done_set]
    logger.info("待处理：%d 条（已跳过 %d 条）", len(pending), len(all_items) - len(pending))

    if not pending:
        logger.info("所有条目已标注完成，无需处理。")
        return

    # 并发生成
    out_file = open(DEFAULT_OUTPUT, "a", encoding="utf-8")
    success = 0
    fail = 0

    task_args = [(client, DEFAULT_MODEL, item) for item in pending]

    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        futures = {executor.submit(process_one, t): t for t in task_args}
        with tqdm(total=len(pending), unit="条",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  ✓{postfix}") as pbar:
            for future in as_completed(futures):
                item, thought = future.result()
                enriched = dict(item)
                enriched["thought_step"] = thought
                out_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                out_file.flush()

                if thought:
                    success += 1
                else:
                    fail += 1

                pbar.set_postfix(成功=success, 失败=fail)
                pbar.update(1)

    out_file.close()
    logger.info("完成！输出文件：%s", DEFAULT_OUTPUT)
    logger.info("成功 %d 条，失败（空 thought_step）%d 条", success, fail)


if __name__ == "__main__":
    main()
