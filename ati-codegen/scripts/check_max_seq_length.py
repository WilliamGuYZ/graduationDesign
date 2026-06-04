#!/usr/bin/env python
"""
检查数据集中 instruction + output 的最大 token 长度
用于确定 LoRA 微调时的 MAX_SEQ_LENGTH 参数

模板格式:
{
    "instruction": "Implement the function `{function_name}` that takes {parameter_list} to solve:\n\n{question}\n\nOnly output the Python code, no explanation.",
    "input": "",
    "output": "{solution}"
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer

# ==================== 配置参数 ====================
# 本地模型路径
LOCAL_MODEL_PATH = "models/CodeGeeX4-ALL-9B"

# 输入数据文件路径
INPUT_JSONL = "data/raw/KodCode.jsonl"

# 限制处理的数据条数（None 表示全部，设为数字如 100 表示只处理前100条）
LIMIT = None

# 显示最长的 N 个样本（设为 0 则不显示）
SHOW_LONG = 10

# ==================== 固定模板 ====================
INSTRUCTION_TEMPLATE = """Implement the function `{function_name}` that takes {parameter_list} to solve:

{question}

Only output the Python code, no explanation."""


def load_data(input_path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """加载 JSONL 数据"""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"加载了 {len(data)} 条数据")
    return data


def build_full_text(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    根据模板构建完整的训练文本
    
    Returns:
        (instruction, output, full_text)
    """
    question = item.get("question", "")
    solution = item.get("solution", "")
    
    # 获取函数信息（部分样本 test_info 为空列表 []）
    test_info = item.get("test_info") or []
    if (
        isinstance(test_info, list)
        and len(test_info) > 0
        and isinstance(test_info[0], dict)
    ):
        function_name = test_info[0].get("function_name", "solution")
        parameter_list = test_info[0].get("parameter_list", "")
    else:
        function_name = "solution"
        parameter_list = ""
    
    # 构建 instruction
    instruction = INSTRUCTION_TEMPLATE.format(
        function_name=function_name,
        parameter_list=parameter_list,
        question=question,
    )
    
    # 完整输入 = instruction + output
    full_text = f"{instruction}\n\n{solution}"
    
    return instruction, solution, full_text


def compute_token_lengths(
    tokenizer: AutoTokenizer,
    data: List[Dict[str, Any]],
) -> Tuple[List[int], Dict[str, Any], List[Dict[str, Any]]]:
    """
    计算每条数据的 token 长度
    
    Returns:
        token_lengths: 每条数据的 token 长度列表
        stats: 统计信息
        details: 每条数据的详细信息
    """
    token_lengths = []
    details = []
    
    for idx, item in enumerate(data):
        instruction, output, full_text = build_full_text(item)
        
        # 计算 token 长度
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        token_len = len(tokens)
        token_lengths.append(token_len)
        
        ti = item.get("test_info") or []
        fn = (
            ti[0].get("function_name", "unknown")
            if ti and isinstance(ti[0], dict)
            else "unknown"
        )
        details.append({
            "index": idx,
            "question_preview": item.get("question", "")[:80],
            "function_name": fn,
            "token_length": token_len,
        })
    
    # 统计信息
    if token_lengths:
        sorted_lengths = sorted(token_lengths)
        stats = {
            "total_samples": len(token_lengths),
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "p50": sorted_lengths[len(sorted_lengths) // 2],
            "p90": sorted_lengths[int(len(sorted_lengths) * 0.9)],
            "p95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
            "p99": sorted_lengths[int(len(sorted_lengths) * 0.99)],
            "total_tokens": sum(token_lengths),
        }
    else:
        stats = {}
    
    return token_lengths, stats, details


def get_recommended_max_length(stats: Dict[str, Any]) -> int:
    """推荐 MAX_SEQ_LENGTH 值"""
    p99 = stats["p99"]
    max_val = stats["max"]
    
    # 推荐值：取 P99 和最大值的较大者，加 64 缓冲
    recommended = max(p99, max_val) + 64
    
    # 向上取整到 64 的倍数
    recommended = ((recommended + 63) // 64) * 64
    
    # 常见长度选项
    common_lengths = [512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 5120, 6144, 8192]
    
    for cl in common_lengths:
        if recommended <= cl:
            return cl
    
    return recommended


def print_stats(stats: Dict[str, Any]) -> None:
    """打印统计信息"""
    print(f"\n{'='*70}")
    print("序列长度统计 (instruction + output)")
    print(f"{'='*70}")
    print(f"总样本数:          {stats['total_samples']:,}")
    print(f"总 Token 数:       {stats['total_tokens']:,}")
    print(f"最小长度:          {stats['min']:,} tokens")
    print(f"最大长度:          {stats['max']:,} tokens")
    print(f"平均长度:          {stats['mean']:.1f} tokens")
    print(f"中位数 (P50):      {stats['p50']:,} tokens")
    print(f"P90:               {stats['p90']:,} tokens")
    print(f"P95:               {stats['p95']:,} tokens")
    print(f"P99:               {stats['p99']:,} tokens")
    
    recommended = get_recommended_max_length(stats)
    print(f"\n推荐 MAX_SEQ_LENGTH: {recommended}")
    print(f"P99 占推荐值的比例: {stats['p99']/recommended*100:.1f}%")
    print(f"{'='*70}\n")


def print_long_samples(details: List[Dict[str, Any]], top_n: int = 10) -> None:
    """打印最长的 N 个样本"""
    if top_n <= 0:
        return
    
    sorted_details = sorted(details, key=lambda x: x["token_length"], reverse=True)
    
    print(f"最长的 {top_n} 个样本:")
    print(f"{'索引':<6} {'Token数':<10} {'函数名':<20} {'问题预览'}")
    print("-" * 80)
    for detail in sorted_details[:top_n]:
        print(f"{detail['index']:<6} {detail['token_length']:<10} {detail['function_name']:<20} {detail['question_preview']}...")


def main():
    """主函数"""
    # 转换为 Path 对象
    input_path = Path(INPUT_JSONL)
    model_path = Path(LOCAL_MODEL_PATH)
    
    # 检查输入文件
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        return
    
    # 检查本地模型路径
    if not model_path.exists():
        print(f"错误: 本地模型路径不存在 {model_path}")
        print("请修改脚本顶部的 LOCAL_MODEL_PATH 配置")
        return
    
    # 加载 tokenizer
    print(f"加载本地 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=True,
    )
    print("Tokenizer 加载成功")
    
    # 加载数据
    data = load_data(input_path, LIMIT)
    
    if not data:
        print("错误: 没有加载到数据")
        return
    
    # 计算 token 长度
    print("正在计算 token 长度...")
    token_lengths, stats, details = compute_token_lengths(tokenizer, data)
    
    # 打印统计
    print_stats(stats)
    
    # 显示最长样本
    if SHOW_LONG > 0:
        print_long_samples(details, SHOW_LONG)

if __name__ == "__main__":
    main()