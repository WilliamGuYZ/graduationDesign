"""
Batch annotate algorithm tags for each solution in KodCode.jsonl.

Usage:
    python scripts/annotate_algorithm_tags.py

"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# ======================== 配置（直接在此修改） ========================
OPENAI_API_KEY  = "sk-91de952175a247ffba72f005df1e6711"
OPENAI_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODEL   = "deepseek-chat"
DEFAULT_INPUT   = "data/raw/KodCode_with_thought.jsonl"
DEFAULT_OUTPUT  = "data/raw/KodCode_annotate.jsonl"
DEFAULT_WORKERS = 32

MAX_RETRY   = 3
RETRY_DELAY = 2
MAX_TOKENS  = 96
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Tag vocabulary: STRONG vs WEAK
#
# STRONG  — high discriminability; captures the algorithmic "skeleton"
#           that makes two problems genuinely similar
# WEAK    — low discriminability; almost every problem uses them, so
#           they add little signal to similarity matching
#
# Rule: every annotated entry MUST contain at least one STRONG tag.
#       If the LLM returns only WEAK tags, the call is retried.
# ──────────────────────────────────────────────────────────────────────
STRONG_TAGS: set[str] = {
    # Core paradigms
    "Dynamic Programming", "Greedy", "Backtracking", "Divide and Conquer",
    "Memoization",
    # Search / graph algorithms
    "BFS", "DFS", "Binary Search", "Topological Sort", "Shortest Path",
    "Union Find",
    # Array techniques
    "Two Pointers", "Sliding Window",
    "Prefix Sum", "Prefix Product", "Suffix Product", "Difference Array",
    # Hash-based lookup / dedup / frequency — core technique, not generic container
    "Hash Table", "Hash Set",
    # Specialized structures
    "Heap", "Trie", "Monotonic Stack", "Monotonic Queue",
    # Bit / math algorithms
    "Bit Manipulation", "Number Theory", "Combinatorics",
    # Sort variants with specific use-cases
    "Counting Sort", "Merge Sort",
    # String algorithms
    "String Matching",
}

WEAK_TAGS: set[str] = {
    # Generic data types (almost always present, low signal)
    "Array", "String", "Math", "Matrix", "Linked List",
    "Stack", "Queue", "Deque",
    "Tree", "Binary Tree", "BST", "Graph",
    # Broad-brush paradigms
    "Recursion", "Sorting", "Simulation", "Brute Force",
    # Other
    "Regular Expression",
}

CANONICAL_TAGS: list[str] = sorted(STRONG_TAGS | WEAK_TAGS)
_CANONICAL_SET: set[str] = set(CANONICAL_TAGS)

# ──────────────────────────────────────────────────────────────────────
# Non-canonical → canonical normalization table
# ──────────────────────────────────────────────────────────────────────
_TAG_NORMALIZE: dict[str, str] = {
    # Descriptive pseudo-tags → structural/algorithmic tags
    "Uniqueness Check":        "Hash Set",
    "Frequency Counting":      "Hash Table",
    "Array Transformation":    "Array",
    "In-place Modification":   "Array",
    "Counting":                "Hash Table",
    "Lookup Table":            "Hash Table",
    "Set":                     "Hash Set",
    "Dictionary":              "Hash Table",
    # Vague → precise
    "Prefix":                  "Prefix Sum",
    "Suffix":                  "Suffix Product",
    "Product Array":           "Prefix Product",
    # Synonyms / aliases
    "Depth First Search":      "DFS",
    "Breadth First Search":    "BFS",
    "Binary Search Tree":      "BST",
    "Priority Queue":          "Heap",
    "Disjoint Set":            "Union Find",
    "Subset Sum":              "Dynamic Programming",
    "Partition Problem":       "Dynamic Programming",
    "Knapsack":                "Dynamic Programming",
    "Fibonacci":               "Dynamic Programming",
    "Kadane":                  "Dynamic Programming",
    "Dijkstra":                "Shortest Path",
    "Bellman-Ford":            "Shortest Path",
    "Floyd-Warshall":          "Shortest Path",
    "Prim":                    "Graph",
    "Kruskal":                 "Union Find",
    "Segment Tree":            "Tree",
    "Fenwick Tree":            "Tree",
    "Binary Indexed Tree":     "Tree",
    "Monotone Stack":          "Monotonic Stack",
}

# Pre-built strings for prompt injection
_STRONG_TAG_LIST   = ", ".join(f'"{t}"' for t in sorted(STRONG_TAGS))
_CANONICAL_TAG_LIST = ", ".join(f'"{t}"' for t in CANONICAL_TAGS)

# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert competitive programmer acting as a problem classifier.

Your task: given a coding problem description and a reference solution, identify the \
MINIMUM SET of algorithm/data-structure tags that captures what the problem \
fundamentally requires — NOT what the solution happens to implement.

STRICT RULES:
1. Return 1 to 3 tags (choose as few as are sufficient).
   - If the problem has ONE core technique (e.g. pure Greedy), return ["Greedy"].
   - Only add a second/third tag when it genuinely adds algorithmic information.
   - NEVER pad with generic tags ("Array", "String", "Math") just to reach 3.
2. Prefer STRONG (high-discriminability) tags over WEAK ones.
   Strong tags: {strong_list}
3. Choose tags ONLY from the canonical vocabulary:
   [{canonical_list}]
4. Return ONLY a compact JSON array of tag strings. No explanation, no extra text.\
""".format(
    strong_list=_STRONG_TAG_LIST,
    canonical_list=_CANONICAL_TAG_LIST,
)

# Used when the first call returns only-weak tags
SYSTEM_PROMPT_STRICT = """\
You are an expert competitive programmer acting as a problem classifier.

The previous answer contained only generic/weak tags, which provide no useful \
discriminability for retrieval. Try again with the following constraints:

MANDATORY: Your answer MUST include at least one STRONG algorithmic tag from:
{strong_list}

Return 1 to 3 tags total. No padding with "Array", "String", "Math" unless \
they are the ONLY possible description. Return a JSON array only.\
""".format(strong_list=_STRONG_TAG_LIST)

USER_PROMPT_TEMPLATE = """\
Problem:
{question}

Reference solution:
```python
{solution}
```

Focus on what the PROBLEM requires, not just what the code does.
Return 1-3 canonical tags as a JSON array."""


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _normalize_tag(tag: str) -> str:
    """Map a raw LLM tag string to the nearest canonical tag.

    Returns an empty string if no canonical mapping can be found — the caller
    must filter out empty strings.  Non-canonical tags are DROPPED rather than
    kept, so they cannot pollute downstream tag_similarity retrieval.
    """
    tag = tag.strip()
    if tag in _CANONICAL_SET:
        return tag
    if tag in _TAG_NORMALIZE:
        mapped = _TAG_NORMALIZE[tag]
        return mapped if mapped in _CANONICAL_SET else ""
    tag_lower = tag.lower()
    for ct in CANONICAL_TAGS:
        if ct.lower() == tag_lower:
            return ct
    for k, v in _TAG_NORMALIZE.items():
        if k.lower() in tag_lower or tag_lower in k.lower():
            return v if v in _CANONICAL_SET else ""
    logger.debug("No canonical mapping found, dropping raw tag: %r", tag)
    return ""


def _parse_raw(raw: str) -> list[str]:
    """Parse LLM output into a list of normalized, non-empty tag strings (up to 3)."""
    raw = raw.strip()
    try:
        tags = json.loads(raw)
        if isinstance(tags, list):
            return [t for t in (_normalize_tag(str(x)) for x in tags[:3]) if t]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[([^\[\]]+)\]', raw)
    if match:
        try:
            tags = json.loads(f"[{match.group(1)}]")
            if isinstance(tags, list):
                return [t for t in (_normalize_tag(str(x)) for x in tags[:3]) if t]
        except json.JSONDecodeError:
            pass

    parts = re.split(r'[\n,]+', raw)
    tags = []
    for p in parts:
        p = re.sub(r'^[\d\.\-\*\s]+', '', p).strip().strip('"\'')
        if p:
            norm = _normalize_tag(p)
            if norm:
                tags.append(norm)
        if len(tags) == 3:
            break
    return tags


def _has_strong_tag(tags: list[str]) -> bool:
    return any(t in STRONG_TAGS for t in tags)


def _dedup(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def load_done_set(output_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["question"])
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info("断点续标：已完成 %d 条，自动跳过", len(done))
    return done


# ──────────────────────────────────────────────────────────────────────
# API call
# ──────────────────────────────────────────────────────────────────────

def call_api(client: OpenAI, model: str, question: str, solution: str) -> list[str]:
    """
    Call the API and return a validated, normalized tag list.

    Strategy:
      - First call: normal prompt.
      - If result is all-weak, retry once with a stricter prompt that mandates
        at least one STRONG tag.
      - Exponential back-off on network/rate-limit errors.
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        solution=solution.strip(),
    )

    def _call(sys_prompt: str) -> list[str]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        return _dedup(_parse_raw(raw))

    for attempt in range(1, MAX_RETRY + 1):
        try:
            tags = _call(SYSTEM_PROMPT)

            # If all-weak, retry once with a stricter prompt
            if tags and not _has_strong_tag(tags):
                logger.debug("All-weak result %s — retrying with strict prompt", tags)
                tags = _call(SYSTEM_PROMPT_STRICT)

            # Final hardening: drop any non-canonical stragglers, dedup
            if tags:
                tags = _dedup([t for t in tags if t in _CANONICAL_SET])

            # If still all-weak after strict retry, keep only the single most
            # informative weak tag to avoid over-weighting noise in retrieval
            if tags and not _has_strong_tag(tags):
                _FALLBACK_PREFERRED = ["Hash Table", "Hash Set", "Sorting", "Simulation", "Array", "String"]
                for preferred in _FALLBACK_PREFERRED:
                    if preferred in tags:
                        return [preferred]
                return tags[:1]

            return tags

        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "rate limit" in str(exc).lower()
            wait = 15 if is_rate_limit else RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRY:
                logger.warning(
                    "Attempt %d failed (%s), retrying in %.0fs...", attempt, exc, wait
                )
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed, skipping: %s", attempt, exc)
                return []


def process_one(args_tuple):
    client, model, item = args_tuple
    tags = call_api(client, model, item["question"], item["solution"])
    strong_tags = [t for t in tags if t in STRONG_TAGS]
    return item, tags, strong_tags


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key or api_key.startswith("sk-xxx"):
        raise ValueError("请在文件顶部配置区填入有效的 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)

    with open(DEFAULT_INPUT, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    logger.info("读取输入：共 %d 条", len(all_items))

    done_set = load_done_set(DEFAULT_OUTPUT)
    pending = [item for item in all_items if item["question"] not in done_set]
    logger.info("待处理：%d 条（已跳过 %d 条）", len(pending), len(all_items) - len(pending))

    if not pending:
        logger.info("所有条目已标注完成，无需处理。")
        return

    out_file = open(DEFAULT_OUTPUT, "a", encoding="utf-8")
    success = fail = all_weak = 0
    tag_count_dist = {1: 0, 2: 0, 3: 0}

    task_args = [(client, DEFAULT_MODEL, item) for item in pending]

    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        futures = {executor.submit(process_one, t): t for t in task_args}
        with tqdm(
            total=len(pending),
            unit="条",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  ✓{postfix}",
        ) as pbar:
            for future in as_completed(futures):
                item, tags, strong_tags = future.result()
                enriched = dict(item)
                enriched["tags"] = tags
                enriched["strong_tags"] = strong_tags  # for weighted retrieval
                out_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                out_file.flush()

                if tags:
                    success += 1
                    n = min(len(tags), 3)
                    tag_count_dist[n] = tag_count_dist.get(n, 0) + 1
                    if not strong_tags:
                        all_weak += 1
                else:
                    fail += 1

                pbar.set_postfix(成功=success, 失败=fail, 纯弱标签=all_weak)
                pbar.update(1)

    out_file.close()
    logger.info("完成！输出文件：%s", DEFAULT_OUTPUT)
    logger.info(
        "成功 %d 条 | 失败 %d 条 | 纯弱标签（无强标签）%d 条",
        success, fail, all_weak,
    )
    logger.info(
        "标签数量分布：1个=%d  2个=%d  3个=%d",
        tag_count_dist.get(1, 0),
        tag_count_dist.get(2, 0),
        tag_count_dist.get(3, 0),
    )


if __name__ == "__main__":
    main()
