"""
评估指标模块 - LER计算和相关指标

这个模块实现了量子纠错解码器的核心评估指标：

LER (Logical Error Rate per Round)：
- 定义：每轮纠错中逻辑错误发生的概率
- 计算方法：通过拟合多个rounds的错误率得到

计算步骤：
1. 对于每个rounds数n，计算错误率 E(n)
2. 计算fidelity F(n) = 1 - 2×E(n)
3. 线性拟合 log F(n) vs n
   - log F(n) = log F_0 + n × log(1-2ε)
   - 从斜率得到 LER ε

物理意义：
- E(n)：n轮后解码失败的概率
- F(n)：n轮后保持正确的fidelity
- ε (LER)：每轮的逻辑错误概率

质量检查：
- R² ≥ 0.9：拟合质量好
- |log F_0| ≤ 0.2：初始fidelity接近1
- F(n) > 0：确保log有效
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class LERResult:
    """LER计算结果

    包含LER值和相关的统计信息。
    """
    ler: float                        # Logical Error Rate per Round
    ler_std: Optional[float] = None   # LER标准差（来自Bootstrap）
    r_squared: float = 0.0            # 拟合R²
    log_f0: float = 0.0               # log(F_0)，应接近0
    slope: float = 0.0                # 拟合斜率
    intercept: float = 0.0            # 拟合截距
    is_valid: bool = True             # 拟合是否有效
    error_rates: Optional[Dict[int, float]] = None  # 每个rounds的错误率
    fidelities: Optional[Dict[int, float]] = None   # 每个rounds的fidelity


def compute_error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """计算错误率

    错误率 = (预测错误的样本数) / (总样本数)

    Args:
        predictions: 预测标签 [N]，0或1
        labels: 真实标签 [N]，0或1

    Returns:
        错误率 (0到1之间)
    """
    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten()

    # 计算预测错误的比例
    error_rate = np.mean(predictions != labels)

    return float(error_rate)


def compute_fidelity(error_rate: float) -> float:
    """计算fidelity

    Fidelity定义：
        F = 1 - 2×E

    其中E是错误率。

    物理意义：
    - F = 1：完全正确
    - F = 0：随机猜测
    - F < 0：比随机更差

    Args:
        error_rate: 错误率 (0到1之间)

    Returns:
        fidelity
    """
    return 1.0 - 2.0 * error_rate


def fit_ler(
    rounds_list: List[int],
    fidelities: List[float],
    min_fidelity: float = 0.1,
) -> Tuple[float, float, float, float, bool]:
    """拟合LER

    使用线性回归拟合：
        log F(n) = log F_0 + n × log(1-2ε)

    其中ε是LER。

    Args:
        rounds_list: rounds数列表 [n1, n2, ...]
        fidelities: 对应的fidelity列表 [F(n1), F(n2), ...]
        min_fidelity: 最小有效fidelity阈值

    Returns:
        Tuple of:
            - ler: Logical Error Rate
            - r_squared: R²值
            - log_f0: log(F_0)
            - slope: 拟合斜率
            - is_valid: 拟合是否有效
    """
    rounds_arr = np.array(rounds_list)
    fid_arr = np.array(fidelities)

    # ==================== 过滤无效数据 ====================
    # F ≤ 0 时log无定义
    # F太小时拟合不可靠
    valid_mask = fid_arr > min_fidelity
    if valid_mask.sum() < 2:
        # 有效点太少，无法拟合
        return 0.0, 0.0, 0.0, 0.0, False

    rounds_valid = rounds_arr[valid_mask]
    fid_valid = fid_arr[valid_mask]

    # ==================== 计算log(F) ====================
    log_fid = np.log(fid_valid)

    # ==================== 线性回归 ====================
    # log F(n) = intercept + slope × n
    # slope = log(1-2ε) → ε = (1 - exp(slope)) / 2
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        rounds_valid, log_fid
    )

    r_squared = r_value ** 2

    # ==================== 计算LER ====================
    # slope = log(1-2ε)
    # 1-2ε = exp(slope)
    # ε = (1 - exp(slope)) / 2
    if slope < 0:
        # 正常情况：斜率为负，fidelity随rounds下降
        ler = (1.0 - np.exp(slope)) / 2.0
    else:
        # 异常情况：斜率为正，fidelity不应该上升
        ler = 0.0

    # ==================== 验证拟合质量 ====================
    log_f0 = intercept
    is_valid = (r_squared >= 0.9) and (abs(log_f0) <= 0.2) and (ler > 0)

    return ler, r_squared, log_f0, slope, is_valid


def compute_ler(
    predictions_by_rounds: Dict[int, np.ndarray],
    labels_by_rounds: Dict[int, np.ndarray],
) -> LERResult:
    """计算LER的完整流程

    Args:
        predictions_by_rounds: {rounds: predictions} 字典
        labels_by_rounds: {rounds: labels} 字典

    Returns:
        LERResult对象
    """
    rounds_list = sorted(predictions_by_rounds.keys())

    # ==================== 计算每个rounds的错误率和fidelity ====================
    error_rates = {}
    fidelities = {}

    for n in rounds_list:
        preds = predictions_by_rounds[n]
        labs = labels_by_rounds[n]

        e = compute_error_rate(preds, labs)
        f = compute_fidelity(e)

        error_rates[n] = e
        fidelities[n] = f

    # ==================== 拟合LER ====================
    fid_list = [fidelities[n] for n in rounds_list]
    ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fid_list)

    return LERResult(
        ler=ler,
        r_squared=r_squared,
        log_f0=log_f0,
        slope=slope,
        intercept=log_f0,
        is_valid=is_valid,
        error_rates=error_rates,
        fidelities=fidelities,
    )


def compute_lambda(
    ler_small: float,
    ler_large: float,
) -> float:
    """计算抑错因子Λ

    Λ = LER_small / LER_large

    物理意义：
    - Λ > 1：增大码距可以降低错误率（正确行为）
    - Λ ≈ 1：码距增大没有效果
    - Λ < 1：码距增大反而增加错误率（异常）

    Args:
        ler_small: 小码距的LER
        ler_large: 大码距的LER

    Returns:
        抑错因子Λ
    """
    if ler_large <= 0:
        return float('inf')

    return ler_small / ler_large


def compute_lambda_with_error(
    ler_small: float,
    ler_small_std: float,
    ler_large: float,
    ler_large_std: float,
) -> Tuple[float, float]:
    """计算抑错因子Λ及其误差

    使用误差传播公式计算Λ的标准差。

    Args:
        ler_small: 小码距LER
        ler_small_std: 小码距LER标准差
        ler_large: 大码距LER
        ler_large_std: 大码距LER标准差

    Returns:
        Tuple of (Λ, Λ的标准差)
    """
    if ler_large <= 0:
        return float('inf'), float('inf')

    lambda_val = ler_small / ler_large

    # 误差传播：σ_λ/λ = sqrt((σ_s/s)² + (σ_l/l)²)
    rel_err_small = ler_small_std / ler_small if ler_small > 0 else 0
    rel_err_large = ler_large_std / ler_large if ler_large > 0 else 0

    rel_err_lambda = np.sqrt(rel_err_small**2 + rel_err_large**2)
    lambda_std = lambda_val * rel_err_lambda

    return lambda_val, lambda_std


def print_ler_summary(results: Dict[int, LERResult]) -> None:
    """打印LER结果摘要

    Args:
        results: {distance: LERResult} 字典
    """
    print("\n" + "="*60)
    print("LER Summary")
    print("="*60)

    for d, result in sorted(results.items()):
        status = "✓" if result.is_valid else "✗"
        ler_str = f"{result.ler:.6f}" if result.ler > 0 else "N/A"
        std_str = f"±{result.ler_std:.6f}" if result.ler_std else ""

        print(f"\nDistance d={d} [{status}]")
        print(f"  LER: {ler_str} {std_str}")
        print(f"  R²:  {result.r_squared:.4f}")
        print(f"  log(F₀): {result.log_f0:.4f}")

        if result.error_rates:
            print("  Error rates by rounds:")
            for n, e in sorted(result.error_rates.items()):
                print(f"    n={n:2d}: E={e:.4f}, F={result.fidelities[n]:.4f}")

    # 计算Λ
    distances = sorted(results.keys())
    if len(distances) >= 2:
        print("\n" + "-"*40)
        print("Error Suppression Factor (Λ):")
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i+1]
            if results[d1].is_valid and results[d2].is_valid:
                lam = compute_lambda(results[d1].ler, results[d2].ler)
                print(f"  Λ_{d1}→{d2} = {lam:.4f}")

    print("="*60 + "\n")
