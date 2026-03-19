from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    无偏 pass@k 估计（Chen et al. 2021 常用公式）：
      pass@k = 1 - C(n-c, k) / C(n, k)
    其中：
      n: 采样总数
      c: 通过测试的样本数
      k: 采样预算（k<=n）
    """
    if k <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        k = n

    # 用对数组合数避免溢出
    def log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return float("-inf")
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    log_num = log_comb(n - c, k)
    log_den = log_comb(n, k)
    ratio = math.exp(log_num - log_den)
    return float(1.0 - ratio)


@dataclass(frozen=True)
class PassKResult:
    n: int
    c: int
    k: int
    pass_at_k: float


def passk_from_bools(passed: Iterable[bool], k: int) -> PassKResult:
    passed_list = list(passed)
    n = len(passed_list)
    c = sum(1 for x in passed_list if x)
    return PassKResult(n=n, c=c, k=k, pass_at_k=estimate_pass_at_k(n=n, c=c, k=k))

