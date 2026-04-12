"""
划分数据集为训练集和验证集
默认比例 9:1
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

# ==================== 配置 ====================
# 默认路径
DEFAULT_INPUT = "data/raw/KodCode.jsonl"
DEFAULT_TRAIN_OUTPUT = "data/processed/KodCode_train.jsonl"
DEFAULT_EVAL_OUTPUT = "data/processed/KodCode_eval.jsonl"

# 默认比例
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_SEED = 42
# ==============================================


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """保存为 JSONL 文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple:
    """
    划分数据集
    
    Args:
        data: 原始数据列表
        train_ratio: 训练集比例 (0-1)
        seed: 随机种子
    
    Returns:
        (train_data, eval_data)
    """
    # 设置随机种子
    random.seed(seed)
    
    # 随机打乱数据
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # 计算分割点
    split_idx = int(len(shuffled) * train_ratio)
    
    # 划分
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]
    
    return train_data, eval_data


def print_stats(train_data: List[Dict], eval_data: List[Dict], input_path: Path) -> None:
    """打印统计信息"""
    total = len(train_data) + len(eval_data)
    
    print("\n" + "=" * 60)
    print("数据集划分统计")
    print("=" * 60)
    print(f"输入文件:        {input_path}")
    print(f"原始数据总量:    {total:,} 条")
    print("-" * 60)
    print(f"训练集:          {len(train_data):,} 条 ({len(train_data)/total*100:.1f}%)")
    print(f"验证集:          {len(eval_data):,} 条 ({len(eval_data)/total*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="划分数据集为训练集和验证集")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 JSONL 文件路径 (默认: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--train-output", "-t",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help=f"训练集输出路径 (默认: {DEFAULT_TRAIN_OUTPUT})"
    )
    parser.add_argument(
        "--eval-output", "-e",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
        help=f"验证集输出路径 (默认: {DEFAULT_EVAL_OUTPUT})"
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"训练集比例 (默认: {DEFAULT_TRAIN_RATIO})"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子 (默认: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="不随机打乱数据，直接按顺序划分"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在 {input_path}")
        return
    
    # 加载数据
    print(f"加载数据: {input_path}")
    data = load_jsonl(input_path)
    print(f"共加载 {len(data):,} 条数据")
    
    # 划分数据集
    if args.no_shuffle:
        # 不随机打乱，直接按顺序划分
        split_idx = int(len(data) * args.ratio)
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        print("使用顺序划分（未随机打乱）")
    else:
        train_data, eval_data = split_dataset(data, args.ratio, args.seed)
        print(f"使用随机划分 (seed={args.seed})")
    
    # 保存文件
    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)
    
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)
    
    # 打印统计
    print_stats(train_data, eval_data, input_path)
    
    print(f"\n训练集已保存: {train_path}")
    print(f"验证集已保存: {eval_path}")


if __name__ == "__main__":
    main()