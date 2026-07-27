"""
校准分析模块 - 预测概率的可靠性评估

这个模块实现了模型校准(Calibration)分析。

什么是校准？
- 校准衡量预测概率的可靠性
- 如果模型预测"错误概率70%"，那么在所有这类预测中应该有约70%确实是错误
- 良好校准的模型给出的概率是可信赖的

校准评估方法：
1. 将预测概率分成若干bins（如10个）
2. 对每个bin计算：
   - 平均预测概率（confidence）
   - 实际错误率（accuracy）
3. 绘制reliability diagram
4. 计算ECE（Expected Calibration Error）

Reliability Diagram示意：
    实际错误率
    1.0 │                    ╱
        │                  ╱
        │                ╱   理想校准线
        │              ╱
        │         ●  ╱
    0.5 │      ●   ╱ ●
        │    ●   ╱
        │  ●   ╱  ●
        │●   ╱
    0.0 └──┴──┴──┴──┴──┴──► 预测概率
        0.0      0.5     1.0

        ● = 模型实际表现
        / = 理想校准线

ECE计算：
    ECE = Σ (|bin_i| / N) × |acc_i - conf_i|
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算校准曲线

    将预测概率分成bins，计算每个bin的平均confidence和accuracy。

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]，0或1
        num_bins: bin数量

    Returns:
        Tuple of:
            - bin_confidences: 每个bin的平均预测概率 [num_bins]
            - bin_accuracies: 每个bin的实际正确率 [num_bins]
            - bin_counts: 每个bin的样本数 [num_bins]
            - bin_edges: bin边界 [num_bins + 1]
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    # 创建bin边界
    bin_edges = np.linspace(0, 1, num_bins + 1)

    # 初始化输出
    bin_confidences = np.zeros(num_bins)
    bin_accuracies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    # 对每个bin计算统计
    for i in range(num_bins):
        # 找到落在这个bin的样本
        if i == num_bins - 1:
            # 最后一个bin包含右边界
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        else:
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])

        if mask.sum() > 0:
            bin_probs = probs[mask]
            bin_labels = labels[mask]

            # 平均预测概率（confidence）
            bin_confidences[i] = np.mean(bin_probs)

            # 实际正确率（注意：这里我们预测的是错误概率）
            # 如果probs是P(error)，那么"正确"是指预测与实际一致
            # accuracy = mean(labels == (probs > 0.5))
            # 但这里我们关心的是概率与实际频率的匹配
            # 所以用：实际错误率 = mean(labels == 1)
            bin_accuracies[i] = np.mean(bin_labels)

            bin_counts[i] = mask.sum()
        else:
            bin_confidences[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_accuracies[i] = np.nan

    return bin_confidences, bin_accuracies, bin_counts, bin_edges


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """计算期望校准误差（ECE）

    ECE = Σ (|bin_i| / N) × |acc_i - conf_i|

    ECE越小表示校准越好：
    - ECE = 0: 完美校准
    - ECE > 0.1: 校准较差

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: bin数量

    Returns:
        ECE值
    """
    confidences, accuracies, counts, _ = compute_calibration_curve(
        probs, labels, num_bins
    )

    # 计算ECE
    total_samples = np.sum(counts)
    if total_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(num_bins):
        if counts[i] > 0 and not np.isnan(accuracies[i]):
            weight = counts[i] / total_samples
            ece += weight * abs(accuracies[i] - confidences[i])

    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """计算最大校准误差（MCE）

    MCE = max_i |acc_i - conf_i|

    MCE关注最差情况的校准。

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: bin数量

    Returns:
        MCE值
    """
    confidences, accuracies, counts, _ = compute_calibration_curve(
        probs, labels, num_bins
    )

    mce = 0.0
    for i in range(num_bins):
        if counts[i] > 0 and not np.isnan(accuracies[i]):
            error = abs(accuracies[i] - confidences[i])
            mce = max(mce, error)

    return float(mce)


def plot_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    ax=None,
    show_histogram: bool = True,
    title: str = "Calibration Plot",
) -> None:
    """绘制校准图（Reliability Diagram）

    包含：
    1. 校准曲线（预测概率 vs 实际频率）
    2. 理想对角线
    3. 可选：预测概率的直方图

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: bin数量
        ax: matplotlib axes（如果为None则创建新图）
        show_histogram: 是否显示直方图
        title: 图标题
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要matplotlib来绘制校准图")
        print("请运行: pip install matplotlib")
        return

    confidences, accuracies, counts, bin_edges = compute_calibration_curve(
        probs, labels, num_bins
    )

    if ax is None:
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                           gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            ax2 = None
    else:
        ax1 = ax
        ax2 = None
        show_histogram = False

    # ==================== 绘制校准曲线 ====================
    # 理想对角线
    ax1.plot([0, 1], [0, 1], 'k--', label='理想校准', alpha=0.7)

    # 实际校准曲线
    valid_mask = counts > 0
    ax1.bar(
        confidences[valid_mask],
        accuracies[valid_mask],
        width=0.1,
        alpha=0.7,
        edgecolor='black',
        label='模型校准',
    )

    # 计算ECE
    ece = expected_calibration_error(probs, labels, num_bins)

    ax1.set_xlabel('预测概率 (Confidence)', fontsize=12)
    ax1.set_ylabel('实际频率 (Accuracy)', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ==================== 绘制直方图 ====================
    if show_histogram and ax2 is not None:
        ax2.hist(probs, bins=bin_edges, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('预测概率', fontsize=12)
        ax2.set_ylabel('样本数', fontsize=12)
        ax2.set_xlim([0, 1])

    if ax is None:
        plt.tight_layout()
        plt.show()


def get_calibration_summary(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """获取校准摘要统计

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: bin数量

    Returns:
        包含各种校准指标的字典
    """
    ece = expected_calibration_error(probs, labels, num_bins)
    mce = maximum_calibration_error(probs, labels, num_bins)

    # 计算其他统计
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()

    # Brier Score
    brier = np.mean((probs - labels) ** 2)

    # 平均预测概率
    mean_pred = np.mean(probs)
    mean_true = np.mean(labels)

    return {
        'ece': ece,
        'mce': mce,
        'brier_score': float(brier),
        'mean_predicted_prob': float(mean_pred),
        'mean_true_prob': float(mean_true),
        'num_samples': len(probs),
    }


def print_calibration_summary(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> None:
    """打印校准摘要

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: bin数量
    """
    summary = get_calibration_summary(probs, labels, num_bins)

    print("\n" + "="*50)
    print("Calibration Summary")
    print("="*50)
    print(f"  样本数:          {summary['num_samples']:,}")
    print(f"  ECE:             {summary['ece']:.4f}")
    print(f"  MCE:             {summary['mce']:.4f}")
    print(f"  Brier Score:     {summary['brier_score']:.4f}")
    print(f"  平均预测概率:     {summary['mean_predicted_prob']:.4f}")
    print(f"  实际正例比例:     {summary['mean_true_prob']:.4f}")

    # 校准质量评估
    if summary['ece'] < 0.05:
        quality = "优秀 ✓"
    elif summary['ece'] < 0.1:
        quality = "良好"
    elif summary['ece'] < 0.2:
        quality = "一般 ⚠"
    else:
        quality = "较差 ✗"

    print(f"\n  校准质量:        {quality}")
    print("="*50 + "\n")
