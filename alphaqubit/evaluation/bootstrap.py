"""
Bootstrap误差估计模块

这个模块实现了LER的Bootstrap误差估计。

为什么需要Bootstrap？
- LER是通过拟合得到的，需要估计其不确定性
- Bootstrap提供非参数的置信区间估计
- 可以评估结果的统计显著性

Bootstrap方法：
1. 对原始数据进行有放回重采样
2. 对重采样数据计算LER
3. 重复B次得到LER的分布
4. 从分布中计算均值和标准差

流程图：
┌─────────────────────────────────────────────────────────────┐
│                    Bootstrap LER                             │
│                                                              │
│  原始数据: {n: (predictions, labels)}                        │
│                                                              │
│  for b in range(B):                                         │
│      │                                                       │
│      ▼                                                       │
│    ┌────────────────────────────────┐                       │
│    │  对每个rounds的数据重采样      │                       │
│    │  (with replacement)            │                       │
│    └───────────────┬────────────────┘                       │
│                    │                                         │
│                    ▼                                         │
│    ┌────────────────────────────────┐                       │
│    │  计算重采样数据的LER           │                       │
│    └───────────────┬────────────────┘                       │
│                    │                                         │
│                    ▼                                         │
│    保存 LER_b                                               │
│                                                              │
│  结果:                                                       │
│    - LER_mean = mean({LER_b})                               │
│    - LER_std = std({LER_b})                                 │
│    - 置信区间                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

推荐参数：
- B = 100：一般用途
- B = 1000：需要精确置信区间时
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_error_rate, compute_fidelity, fit_ler, LERResult


@dataclass
class BootstrapResult:
    """Bootstrap结果

    包含LER的Bootstrap估计和置信区间。
    """
    ler_mean: float             # LER均值
    ler_std: float              # LER标准差
    ler_median: float           # LER中位数
    ci_lower: float             # 置信区间下界
    ci_upper: float             # 置信区间上界
    ci_level: float             # 置信水平 (如0.95)
    num_bootstrap: int          # Bootstrap次数
    valid_rate: float           # 有效拟合的比例
    all_lers: List[float]       # 所有Bootstrap的LER


def bootstrap_ler(
    predictions_by_rounds: Dict[int, np.ndarray],
    labels_by_rounds: Dict[int, np.ndarray],
    num_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Bootstrap估计LER

    使用Bootstrap方法估计LER的不确定性。

    Args:
        predictions_by_rounds: {rounds: predictions} 字典
        labels_by_rounds: {rounds: labels} 字典
        num_bootstrap: Bootstrap重采样次数
        confidence_level: 置信水平 (如0.95表示95%置信区间)
        seed: 随机种子

    Returns:
        BootstrapResult对象

    使用示例：
        result = bootstrap_ler(
            predictions_by_rounds={3: pred_3, 6: pred_6, 12: pred_12},
            labels_by_rounds={3: lab_3, 6: lab_6, 12: lab_12},
            num_bootstrap=100,
        )
        print(f"LER = {result.ler_mean:.6f} ± {result.ler_std:.6f}")
    """
    if seed is not None:
        np.random.seed(seed)

    rounds_list = sorted(predictions_by_rounds.keys())

    # 收集所有Bootstrap的LER
    all_lers = []
    valid_count = 0

    for b in range(num_bootstrap):
        # ==================== 重采样每个rounds的数据 ====================
        resampled_preds = {}
        resampled_labels = {}

        for n in rounds_list:
            preds = predictions_by_rounds[n]
            labs = labels_by_rounds[n]
            n_samples = len(preds)

            # 有放回重采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            resampled_preds[n] = preds[indices]
            resampled_labels[n] = labs[indices]

        # ==================== 计算重采样数据的LER ====================
        fidelities = []
        for n in rounds_list:
            e = compute_error_rate(resampled_preds[n], resampled_labels[n])
            f = compute_fidelity(e)
            fidelities.append(f)

        ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fidelities)

        if is_valid and ler > 0:
            all_lers.append(ler)
            valid_count += 1

    # ==================== 计算统计量 ====================
    if len(all_lers) == 0:
        # 所有拟合都失败了
        return BootstrapResult(
            ler_mean=0.0,
            ler_std=0.0,
            ler_median=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_level=confidence_level,
            num_bootstrap=num_bootstrap,
            valid_rate=0.0,
            all_lers=[],
        )

    all_lers = np.array(all_lers)

    # 均值和标准差
    ler_mean = np.mean(all_lers)
    ler_std = np.std(all_lers)
    ler_median = np.median(all_lers)

    # 置信区间（使用百分位数方法）
    alpha = 1 - confidence_level
    ci_lower = np.percentile(all_lers, 100 * alpha / 2)
    ci_upper = np.percentile(all_lers, 100 * (1 - alpha / 2))

    # 有效率
    valid_rate = valid_count / num_bootstrap

    return BootstrapResult(
        ler_mean=ler_mean,
        ler_std=ler_std,
        ler_median=ler_median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=confidence_level,
        num_bootstrap=num_bootstrap,
        valid_rate=valid_rate,
        all_lers=all_lers.tolist(),
    )


def bootstrap_compare(
    ler1: BootstrapResult,
    ler2: BootstrapResult,
) -> Tuple[float, float]:
    """比较两个LER结果

    使用Bootstrap样本比较两个LER是否有显著差异。

    Args:
        ler1: 第一个LER的Bootstrap结果
        ler2: 第二个LER的Bootstrap结果

    Returns:
        Tuple of:
            - 差异均值 (ler1 - ler2)
            - 差异大于0的概率 (P(ler1 > ler2))
    """
    if len(ler1.all_lers) == 0 or len(ler2.all_lers) == 0:
        return 0.0, 0.5

    # 配对比较（如果样本数相同）
    n = min(len(ler1.all_lers), len(ler2.all_lers))
    diffs = np.array(ler1.all_lers[:n]) - np.array(ler2.all_lers[:n])

    diff_mean = np.mean(diffs)
    prob_greater = np.mean(diffs > 0)

    return float(diff_mean), float(prob_greater)


def bootstrap_lambda(
    ler_small: BootstrapResult,
    ler_large: BootstrapResult,
) -> Tuple[float, float, float]:
    """Bootstrap估计抑错因子Λ

    使用Bootstrap样本计算Λ的分布。

    Args:
        ler_small: 小码距的Bootstrap结果
        ler_large: 大码距的Bootstrap结果

    Returns:
        Tuple of:
            - Λ均值
            - Λ标准差
            - P(Λ > 1)：Λ大于1的概率
    """
    if len(ler_small.all_lers) == 0 or len(ler_large.all_lers) == 0:
        return 0.0, 0.0, 0.0

    # 计算每对Bootstrap样本的Λ
    n = min(len(ler_small.all_lers), len(ler_large.all_lers))
    lambdas = []

    for i in range(n):
        if ler_large.all_lers[i] > 0:
            lam = ler_small.all_lers[i] / ler_large.all_lers[i]
            lambdas.append(lam)

    if len(lambdas) == 0:
        return 0.0, 0.0, 0.0

    lambdas = np.array(lambdas)

    lambda_mean = np.mean(lambdas)
    lambda_std = np.std(lambdas)
    prob_greater_one = np.mean(lambdas > 1)

    return float(lambda_mean), float(lambda_std), float(prob_greater_one)


def print_bootstrap_summary(
    results: Dict[int, BootstrapResult],
) -> None:
    """打印Bootstrap结果摘要

    Args:
        results: {distance: BootstrapResult} 字典
    """
    print("\n" + "="*60)
    print("Bootstrap LER Summary")
    print("="*60)

    for d, result in sorted(results.items()):
        valid_pct = result.valid_rate * 100
        status = "✓" if result.valid_rate >= 0.9 else "⚠" if result.valid_rate >= 0.5 else "✗"

        print(f"\nDistance d={d} [{status}]")
        print(f"  LER: {result.ler_mean:.6f} ± {result.ler_std:.6f}")
        print(f"  Median: {result.ler_median:.6f}")
        print(f"  {result.ci_level*100:.0f}% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"  Valid rate: {valid_pct:.1f}% ({int(result.valid_rate * result.num_bootstrap)}/{result.num_bootstrap})")

    # 计算Λ
    distances = sorted(results.keys())
    if len(distances) >= 2:
        print("\n" + "-"*40)
        print("Error Suppression Factor (Λ) with Bootstrap:")
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i+1]
            lam_mean, lam_std, prob = bootstrap_lambda(results[d1], results[d2])
            print(f"  Λ_{d1}→{d2} = {lam_mean:.4f} ± {lam_std:.4f}")
            print(f"    P(Λ > 1) = {prob*100:.1f}%")

    print("="*60 + "\n")
