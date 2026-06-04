"""Patch CodeGeeX4 / ChatGLM4Tokenizer._pad to accept new transformers kwargs.

Background
----------
transformers >= 4.45 calls ``self._pad(..., padding_side=...)`` from
``PreTrainedTokenizerBase.pad``. The vendored ``tokenization_chatglm.py``
shipped with CodeGeeX4-ALL-9B was written against an older transformers
API and its ``_pad`` signature does **not** accept ``padding_side``,
which raises::

    TypeError: ChatGLM4Tokenizer._pad() got an unexpected keyword
               argument 'padding_side'

This script patches ``_pad`` by appending ``**kwargs`` to its signature.
It is idempotent: running twice is a no-op.

Targets
-------
1. ``models/CodeGeeX4-ALL-9B/tokenization_chatglm.py`` (source under repo)
2. ``~/.cache/huggingface/modules/transformers_modules/*/tokenization_chatglm.py``
   (all copies that ``trust_remote_code`` may have generated)

Usage
-----
    python3 scripts/patch_chatglm_tokenizer.py            # apply patch
    python3 scripts/patch_chatglm_tokenizer.py --check    # dry-run, exit 1 if any file still needs patching
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Match the multi-line `def _pad(...) -> dict:` signature and capture the
# closing `)` line so we can insert `**kwargs,` right before it.
#
# Example matched block:
#     def _pad(
#         self,
#         encoded_inputs: ...,
#         max_length: ...,
#         padding_strategy: ...,
#         pad_to_multiple_of: ...,
#         return_attention_mask: ...,
#     ) -> dict:
_PAD_SIGNATURE = re.compile(
    r"(def\s+_pad\s*\(\s*\n"            # `def _pad(\n`
    r"(?:[^)]*?\n)+?)"                  # parameter lines (non-greedy)
    r"(\s*\)\s*->\s*dict\s*:)",         # closing `) -> dict:`
    re.MULTILINE,
)

PATCH_LINE = "        **kwargs,\n"


def find_targets(repo_root: pathlib.Path) -> list[pathlib.Path]:
    targets: list[pathlib.Path] = []

    # 1) repo-local model directory
    local = repo_root / "models" / "CodeGeeX4-ALL-9B" / "tokenization_chatglm.py"
    if local.is_file():
        targets.append(local)

    # 2) HF trust_remote_code cache (any matching dir)
    cache_root = pathlib.Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if cache_root.is_dir():
        for p in cache_root.glob("*/tokenization_chatglm.py"):
            if p.is_file():
                targets.append(p)

    return targets


def needs_patch(src: str) -> bool:
    m = _PAD_SIGNATURE.search(src)
    if not m:
        return False  # no _pad found → nothing to do
    params_block = m.group(1)
    return "**kwargs" not in params_block


def apply_patch(src: str) -> tuple[str, int]:
    def _inject(match: re.Match[str]) -> str:
        params, closer = match.group(1), match.group(2)
        if "**kwargs" in params:
            return match.group(0)
        return params + PATCH_LINE + closer

    return _PAD_SIGNATURE.subn(_inject, src)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="dry-run: exit 1 if any file still needs patching")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    targets = find_targets(repo_root)

    if not targets:
        print("[patch_chatglm_tokenizer] no tokenization_chatglm.py found.")
        print("  expected:")
        print(f"    - {repo_root / 'models/CodeGeeX4-ALL-9B/tokenization_chatglm.py'}")
        print("    - ~/.cache/huggingface/modules/transformers_modules/*/tokenization_chatglm.py")
        print("  did you finish `modelscope download` and load the model at least once?")
        return 1

    rc = 0
    for path in targets:
        src = path.read_text(encoding="utf-8")
        if not _PAD_SIGNATURE.search(src):
            print(f"[skip ] {path}  (no `_pad` signature found, file may be already modified)")
            continue
        if not needs_patch(src):
            print(f"[ok   ] {path}  (already patched)")
            continue

        if args.check:
            print(f"[NEED ] {path}  (would be patched)")
            rc = 1
            continue

        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())

        new_src, n = apply_patch(src)
        if n == 0:
            print(f"[fail ] {path}  (regex matched but substitution returned 0; file untouched)")
            rc = 1
            continue
        path.write_text(new_src, encoding="utf-8")
        print(f"[patch] {path}  (+**kwargs, backup -> {backup.name})")

    return rc


if __name__ == "__main__":
    sys.exit(main())
