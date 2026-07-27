"""
评估模块 - AlphaQubit模型评估工具

这个模块提供模型性能评估所需的所有工具：

1. LER计算 (Logical Error Rate per Round)
   - 核心评估指标
   - 通过拟合F(n) = F_0 × (1-2ε)^n 得到

2. Bootstrap误差估计
   - 计算LER的置信区间
   - 评估结果的统计显著性

3. 抑错因子Λ
   - Λ = LER_d1 / LER_d2
   - Λ > 1 表示码距增大有效

4. Calibration分析
   - 检查预测概率的可靠性
   - 绘制reliability diagram

评估流程：
┌─────────────────────────────────────────────────────────────┐
│                     Evaluation Pipeline                      │
│                                                              │
│  1. 在多个rounds上采样                                       │
│     rounds = [3, 5, 7, 9, 11, 12]                           │
│                                                              │
│  2. 计算每个rounds的错误率E(n)                              │
│     E(n) = (预测错误数) / (总样本数)                         │
│                                                              │
│  3. 计算fidelity F(n)                                       │
│     F(n) = 1 - 2×E(n)                                       │
│                                                              │
│  4. 线性拟合log(F(n))                                       │
│     log F(n) = log F_0 + n × log(1-2ε)                      │
│     从斜率得到 LER ε                                        │
│                                                              │
│  5. Bootstrap误差估计                                        │
│     重复采样50-100次，计算LER的分布                         │
│                                                              │
│  6. 计算抑错因子Λ                                           │
│     Λ = LER_small / LER_large                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
"""

from .metrics import (
    compute_error_rate,
    compute_fidelity,
    fit_ler,
    compute_ler,
    compute_lambda,
    print_ler_summary,
    LERResult,
)
from .bootstrap import bootstrap_ler, BootstrapResult
from .calibration import (
    compute_calibration_curve,
    plot_calibration,
    expected_calibration_error,
)

__all__ = [
    # LER计算
    "compute_error_rate",
    "compute_fidelity",
    "fit_ler",
    "compute_ler",
    "compute_lambda",
    "print_ler_summary",
    "LERResult",
    # Bootstrap
    "bootstrap_ler",
    "BootstrapResult",
    # Calibration
    "compute_calibration_curve",
    "plot_calibration",
    "expected_calibration_error",
]
