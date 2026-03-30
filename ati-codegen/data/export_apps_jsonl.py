import argparse
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_code(code: str) -> str:
    # Keep original code content, but normalize whitespace for dedup + line endings.
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code.strip()


def normalize_io(io: str) -> str:
    # input/output strings may contain trailing newlines; remove only trailing whitespace/newlines.
    # Some dataset files may store each IO example as a list of lines; handle both str and list.
    if io is None:
        return ""
    if isinstance(io, list):
        return "\n".join(str(x) for x in io).rstrip("\r\n")
    return str(io).rstrip("\r\n")


def build_input(question: str, sample_input: str, sample_output: str) -> str:
    return (
        "题目：\n"
        + question.strip()
        + "\n\n"
        + "样例输入：\n"
        + normalize_io(sample_input)
        + "\n\n"
        + "样例输出：\n"
        + normalize_io(sample_output)
        + "\n"
    )


def iter_problem_dirs(split_dir: str) -> Iterable[str]:
    # e.g. .../train/0000, .../train/0001 ...
    if not os.path.isdir(split_dir):
        return
    for name in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, name)
        if os.path.isdir(p):
            yield p


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_one_solution(
    solutions: List[str],
    max_chars: int,
) -> Optional[str]:
    """
    Cleaning rules (heuristic, no execution):
    - strip + normalize line endings
    - drop empty solutions
    - drop too long solutions
    - deduplicate by sha256
    - choose the shortest remaining solution
    """
    seen = set()
    candidates: List[str] = []
    for sol in solutions:
        code = normalize_code(sol)
        if not code:
            continue
        if len(code) > max_chars:
            continue
        h = sha256_text(code)
        if h in seen:
            continue
        seen.add(h)
        candidates.append(code)

    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def export_split(
    split_name: str,
    split_dir: str,
    out_dir: str,
    instruction: str,
    max_solution_chars: int,
    max_file_bytes: int,
):
    os.makedirs(out_dir, exist_ok=True)

    part_idx = 0
    bytes_in_file = 0
    out_path = os.path.join(out_dir, f"{split_name}_part_{part_idx:04d}.jsonl")
    f = open(out_path, "w", encoding="utf-8")

    total = 0
    kept = 0
    skipped = 0

    try:
        for prob_dir in iter_problem_dirs(split_dir):
            total += 1
            q_path = os.path.join(prob_dir, "question.txt")
            io_path = os.path.join(prob_dir, "input_output.json")
            sol_path = os.path.join(prob_dir, "solutions.json")
            # Some problems might miss files; skip them.
            if not (os.path.isfile(q_path) and os.path.isfile(io_path) and os.path.isfile(sol_path)):
                skipped += 1
                continue

            try:
                question = read_text(q_path)
                io_obj = read_json(io_path)
                solutions_obj = read_json(sol_path)
            except Exception:
                skipped += 1
                continue

            inputs = io_obj.get("inputs", []) if isinstance(io_obj, dict) else []
            outputs = io_obj.get("outputs", []) if isinstance(io_obj, dict) else []
            solutions = solutions_obj if isinstance(solutions_obj, list) else []

            if not inputs or not outputs:
                skipped += 1
                continue
            if len(inputs) != len(outputs):
                # Data seems usually aligned, but be strict for training.
                skipped += 1
                continue
            if not solutions:
                skipped += 1
                continue

            sample_input = inputs[0]
            sample_output = outputs[0]

            chosen_code = choose_one_solution(solutions, max_solution_chars)
            if not chosen_code:
                skipped += 1
                continue

            record: Dict[str, str] = {
                "instruction": instruction,
                "input": build_input(question, sample_input, sample_output),
                "output": chosen_code,
            }
            line = json.dumps(record, ensure_ascii=False)
            line_bytes = len((line + "\n").encode("utf-8"))

            # Split by approximate file size.
            if bytes_in_file > 0 and bytes_in_file + line_bytes > max_file_bytes:
                f.close()
                part_idx += 1
                bytes_in_file = 0
                out_path = os.path.join(out_dir, f"{split_name}_part_{part_idx:04d}.jsonl")
                f = open(out_path, "w", encoding="utf-8")

            f.write(line + "\n")
            bytes_in_file += line_bytes
            kept += 1

    finally:
        f.close()

    print(
        f"[{split_name}] total_dirs={total}, kept={kept}, skipped={skipped}, out_dir={out_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="APPS root folder that contains train/ and test/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for jsonl files. Default: ./exports/jsonl",
    )
    parser.add_argument(
        "--max_solution_chars",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--max_file_mb",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--include_splits",
        type=str,
        default="train_and_test",
        choices=["train_only", "train_and_test"],
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="请根据题目编写解题代码（通过样例）。",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if args.out_dir is None:
        out_dir = os.path.join(dataset_root, "exports", "jsonl")
    else:
        out_dir = os.path.abspath(args.out_dir)

    max_file_bytes = args.max_file_mb * 1024 * 1024

    if args.include_splits in ("train_only",):
        export_split(
            "train",
            os.path.join(dataset_root, "train"),
            out_dir,
            args.instruction,
            args.max_solution_chars,
            max_file_bytes,
        )
    else:
        export_split(
            "train",
            os.path.join(dataset_root, "train"),
            out_dir,
            args.instruction,
            args.max_solution_chars,
            max_file_bytes,
        )
        export_split(
            "test",
            os.path.join(dataset_root, "test"),
            out_dir,
            args.instruction,
            args.max_solution_chars,
            max_file_bytes,
        )


if __name__ == "__main__":
    main()

