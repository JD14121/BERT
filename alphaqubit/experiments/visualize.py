"""
可视化模块 - 评估结果的可视化

这个模块提供AlphaQubit评估结果的可视化工具。

可视化类型：
┌─────────────────────────────────────────────────────────────────┐
│                     Visualization Types                          │
│                                                                   │
│  1. LER vs 噪声强度曲线                                          │
│     ┌────────────────────────────┐                               │
│     │ LER                         │                               │
│     │  │     d=3                 │                               │
│     │  │  ●─●──●──●              │                               │
│     │  │      d=5                │                               │
│     │  │    ●──●──●──●           │                               │
│     │  │        d=7              │                               │
│     │  │      ●──●──●──●         │                               │
│     │  └───────────────────► p   │                               │
│     └────────────────────────────┘                               │
│                                                                   │
│  2. LER vs 码距曲线（对数坐标）                                   │
│     ┌────────────────────────────┐                               │
│     │ log(LER)                    │                               │
│     │  │  ●                       │                               │
│     │  │    ●                     │                               │
│     │  │      ●   阈值以下区域    │                               │
│     │  │        ●   (指数下降)   │                               │
│     │  │          ●              │                               │
│     │  └───────────────────► d   │                               │
│     └────────────────────────────┘                               │
│                                                                   │
│  3. Λ因子热力图                                                  │
│     ┌────────────────────────────┐                               │
│     │      d=3  d=5  d=7  d=9    │                               │
│     │ p1   1.5  1.8  2.1         │                               │
│     │ p2   1.4  1.7  1.9         │                               │
│     │ p3   1.2  1.5  1.7         │                               │
│     └────────────────────────────┘                               │
│                                                                   │
│  4. 与MWPM基线对比                                               │
│     ┌────────────────────────────┐                               │
│     │ LER                         │                               │
│     │  │  ●──── MWPM             │                               │
│     │  │                          │                               │
│     │  │  ○──── AlphaQubit       │                               │
│     │  │      (低于MWPM)         │                               │
│     │  └───────────────────► p   │                               │
│     └────────────────────────────┘                               │
│                                                                   │
│  5. 校准图 (Reliability Diagram)                                 │
│     ┌────────────────────────────┐                               │
│     │ Accuracy                    │                               │
│     │  │         ╱               │                               │
│     │  │   ●  ╱ ●                │                               │
│     │  │ ●  ╱                    │                               │
│     │  └╱───────────────► Conf   │                               │
│     └────────────────────────────┘                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

使用示例：
    from alphaqubit.experiments.visualize import (
        plot_ler_vs_noise,
        plot_comparison,
        create_summary_report,
    )

    # 绘制LER曲线
    plot_ler_vs_noise(results, save_path="ler_vs_noise.png")

    # 与基线对比
    plot_comparison(model_results, baseline_results, save_path="comparison.png")

    # 生成完整报告
    create_summary_report(results, output_dir="reports/")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json

import numpy as np

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装。可视化功能将不可用。")
    print("安装方法: pip install matplotlib")

from .evaluator import EvaluationResult
from .baselines import BaselineResult


def _check_matplotlib():
    """检查matplotlib是否可用"""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib未安装。请运行: pip install matplotlib"
        )


def plot_ler_vs_noise(
    results: Dict[Tuple[int, float], EvaluationResult],
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True,
    show_error_bars: bool = True,
    save_path: Optional[str] = None,
    title: str = "LER vs Noise Strength",
) -> None:
    """绘制LER随噪声强度变化的曲线

    对每个码距绘制一条曲线，展示LER如何随噪声强度p变化。

    Args:
        results: {(distance, noise_p): EvaluationResult} 字典
        figsize: 图像大小
        log_scale: 是否使用对数坐标
        show_error_bars: 是否显示误差棒
        save_path: 保存路径（可选）
        title: 图标题
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # 提取码距列表
    distances = sorted(set(d for d, p in results.keys()))

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(distances)))

    for i, d in enumerate(distances):
        # 提取该码距的数据
        d_results = [
            (p, r) for (dist, p), r in results.items()
            if dist == d and r.is_valid
        ]
        d_results.sort(key=lambda x: x[0])

        if not d_results:
            continue

        noise_levels = [p for p, r in d_results]
        lers = [r.ler for p, r in d_results]

        # 绘制曲线
        if show_error_bars and d_results[0][1].ler_std is not None:
            ler_stds = [r.ler_std for p, r in d_results]
            ax.errorbar(
                noise_levels, lers,
                yerr=ler_stds,
                fmt='o-',
                color=colors[i],
                label=f'd={d}',
                capsize=3,
                markersize=6,
            )
        else:
            ax.plot(
                noise_levels, lers,
                'o-',
                color=colors[i],
                label=f'd={d}',
                markersize=6,
            )

    ax.set_xlabel('物理噪声强度 p', fontsize=12)
    ax.set_ylabel('逻辑错误率 LER', fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def plot_ler_vs_distance(
    results: Dict[Tuple[int, float], EvaluationResult],
    noise_levels: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True,
    show_error_bars: bool = True,
    save_path: Optional[str] = None,
    title: str = "LER vs Code Distance",
) -> None:
    """绘制LER随码距变化的曲线

    对每个噪声水平绘制一条曲线，展示LER如何随码距d变化。

    Args:
        results: {(distance, noise_p): EvaluationResult} 字典
        noise_levels: 要绘制的噪声水平列表（None表示全部）
        figsize: 图像大小
        log_scale: 是否使用对数坐标
        show_error_bars: 是否显示误差棒
        save_path: 保存路径（可选）
        title: 图标题
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # 提取噪声水平列表
    if noise_levels is None:
        noise_levels = sorted(set(p for d, p in results.keys()))

    # 颜色映射
    colors = plt.cm.plasma(np.linspace(0, 1, len(noise_levels)))

    for i, p in enumerate(noise_levels):
        # 提取该噪声水平的数据
        p_results = [
            (d, r) for (d, noise), r in results.items()
            if noise == p and r.is_valid
        ]
        p_results.sort(key=lambda x: x[0])

        if not p_results:
            continue

        distances = [d for d, r in p_results]
        lers = [r.ler for d, r in p_results]

        # 绘制曲线
        if show_error_bars and p_results[0][1].ler_std is not None:
            ler_stds = [r.ler_std for d, r in p_results]
            ax.errorbar(
                distances, lers,
                yerr=ler_stds,
                fmt='o-',
                color=colors[i],
                label=f'p={p:.4f}',
                capsize=3,
                markersize=6,
            )
        else:
            ax.plot(
                distances, lers,
                'o-',
                color=colors[i],
                label=f'p={p:.4f}',
                markersize=6,
            )

    ax.set_xlabel('码距 d', fontsize=12)
    ax.set_ylabel('逻辑错误率 LER', fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 设置x轴刻度为整数
    ax.set_xticks(sorted(set(d for d, p in results.keys())))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def plot_lambda_factor(
    results: Dict[Tuple[int, float], EvaluationResult],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    title: str = "Error Suppression Factor (Λ)",
) -> None:
    """绘制Λ因子

    展示相邻码距之间的错误抑制因子。

    Args:
        results: {(distance, noise_p): EvaluationResult} 字典
        figsize: 图像大小
        save_path: 保存路径（可选）
        title: 图标题
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # 计算Λ因子
    distances = sorted(set(d for d, p in results.keys()))
    noise_levels = sorted(set(p for d, p in results.keys()))

    # 对每对相邻码距绘制
    colors = plt.cm.Set1(np.linspace(0, 1, len(distances) - 1))

    for i in range(len(distances) - 1):
        d1, d2 = distances[i], distances[i + 1]
        lambdas = []
        lambda_stds = []
        valid_noise = []

        for p in noise_levels:
            if (d1, p) in results and (d2, p) in results:
                r1 = results[(d1, p)]
                r2 = results[(d2, p)]

                if r1.is_valid and r2.is_valid and r2.ler > 0:
                    lam = r1.ler / r2.ler
                    lambdas.append(lam)
                    valid_noise.append(p)

                    # 误差传播
                    if r1.ler_std and r2.ler_std:
                        rel_err = np.sqrt(
                            (r1.ler_std / r1.ler) ** 2 +
                            (r2.ler_std / r2.ler) ** 2
                        )
                        lambda_stds.append(lam * rel_err)
                    else:
                        lambda_stds.append(0)

        if lambdas:
            ax.errorbar(
                valid_noise, lambdas,
                yerr=lambda_stds,
                fmt='o-',
                color=colors[i],
                label=f'Λ_{d1}→{d2}',
                capsize=3,
                markersize=6,
            )

    # 绘制Λ=1参考线
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Λ=1')

    ax.set_xlabel('物理噪声强度 p', fontsize=12)
    ax.set_ylabel('抑错因子 Λ', fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def plot_comparison(
    model_results: Dict[Tuple[int, float], EvaluationResult],
    baseline_results: Dict[Tuple[int, float], BaselineResult],
    distance: int = 5,
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """绘制模型与基线的对比

    Args:
        model_results: 模型评估结果
        baseline_results: 基线评估结果
        distance: 要比较的码距
        figsize: 图像大小
        log_scale: 是否使用对数坐标
        save_path: 保存路径
        title: 图标题
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # 提取指定码距的数据
    model_data = [
        (p, r) for (d, p), r in model_results.items()
        if d == distance and r.is_valid
    ]
    model_data.sort(key=lambda x: x[0])

    baseline_data = [
        (p, r) for (d, p), r in baseline_results.items()
        if d == distance
    ]
    baseline_data.sort(key=lambda x: x[0])

    # 绘制模型结果
    if model_data:
        noise_levels = [p for p, r in model_data]
        lers = [r.ler for p, r in model_data]

        if model_data[0][1].ler_std is not None:
            ler_stds = [r.ler_std for p, r in model_data]
            ax.errorbar(
                noise_levels, lers,
                yerr=ler_stds,
                fmt='o-',
                color='blue',
                label='AlphaQubit',
                capsize=3,
                markersize=8,
                linewidth=2,
            )
        else:
            ax.plot(
                noise_levels, lers,
                'o-',
                color='blue',
                label='AlphaQubit',
                markersize=8,
                linewidth=2,
            )

    # 绘制基线结果
    if baseline_data:
        noise_levels = [p for p, r in baseline_data]
        error_rates = [r.error_rate for p, r in baseline_data]

        ax.plot(
            noise_levels, error_rates,
            's--',
            color='red',
            label=f'MWPM (Baseline)',
            markersize=8,
            linewidth=2,
            alpha=0.7,
        )

    ax.set_xlabel('物理噪声强度 p', fontsize=12)
    ax.set_ylabel('逻辑错误率', fontsize=12)

    if title is None:
        title = f'AlphaQubit vs MWPM (d={distance})'
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """绘制训练历史

    Args:
        history: 训练历史字典
        figsize: 图像大小
        save_path: 保存路径
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 损失曲线
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train', alpha=0.7)
    if 'val_loss' in history:
        # val_loss可能是嵌套字典或列表
        if isinstance(history['val_loss'], dict):
            for d, losses in history['val_loss'].items():
                ax.plot(losses, label=f'Val d={d}', alpha=0.7)
        else:
            ax.plot(history['val_loss'], label='Val', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率曲线
    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(history['train_acc'], label='Train', alpha=0.7)
    if 'val_acc' in history:
        ax.plot(history['val_acc'], label='Val', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 学习率曲线
    ax = axes[2]
    if 'learning_rate' in history:
        ax.plot(history['learning_rate'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def create_summary_report(
    results: Dict[Tuple[int, float], EvaluationResult],
    baseline_results: Optional[Dict[Tuple[int, float], BaselineResult]] = None,
    output_dir: str = "reports",
) -> None:
    """创建完整的摘要报告

    生成多个图表和JSON报告。

    Args:
        results: 模型评估结果
        baseline_results: 基线评估结果（可选）
        output_dir: 输出目录
    """
    _check_matplotlib()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n生成评估报告到: {output_path}")

    # 1. LER vs 噪声强度
    print("  生成 LER vs Noise 图...")
    plot_ler_vs_noise(
        results,
        save_path=str(output_path / "ler_vs_noise.png"),
    )

    # 2. LER vs 码距
    print("  生成 LER vs Distance 图...")
    plot_ler_vs_distance(
        results,
        save_path=str(output_path / "ler_vs_distance.png"),
    )

    # 3. Λ因子
    print("  生成 Lambda Factor 图...")
    plot_lambda_factor(
        results,
        save_path=str(output_path / "lambda_factor.png"),
    )

    # 4. 与基线对比（如果有）
    if baseline_results:
        distances = sorted(set(d for d, p in results.keys()))
        for d in distances:
            print(f"  生成 Comparison (d={d}) 图...")
            plot_comparison(
                results,
                baseline_results,
                distance=d,
                save_path=str(output_path / f"comparison_d{d}.png"),
            )

    # 5. 保存JSON数据
    print("  保存 JSON 数据...")
    json_data = {
        "model_results": {
            f"d{d}_p{p}": r.to_dict()
            for (d, p), r in results.items()
        }
    }
    if baseline_results:
        json_data["baseline_results"] = {
            f"d{r.distance}_p{r.noise_p}": {
                "error_rate": r.error_rate,
                "method": r.method,
            }
            for (d, p), r in baseline_results.items()
        }

    with open(output_path / "results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n报告生成完成！所有文件保存在: {output_path}")


def plot_calibration_comparison(
    results: Dict[Tuple[int, float], EvaluationResult],
    distances: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """绘制不同码距的校准对比

    Args:
        results: 评估结果
        distances: 要比较的码距列表
        figsize: 图像大小
        save_path: 保存路径
    """
    _check_matplotlib()

    if distances is None:
        distances = sorted(set(d for d, p in results.keys()))[:3]

    fig, axes = plt.subplots(1, len(distances), figsize=figsize)
    if len(distances) == 1:
        axes = [axes]

    # 选择中间噪声水平
    noise_levels = sorted(set(p for d, p in results.keys()))
    mid_noise = noise_levels[len(noise_levels) // 2]

    for ax, d in zip(axes, distances):
        if (d, mid_noise) in results:
            r = results[(d, mid_noise)]

            # 绘制ECE条形图
            ax.bar(['ECE', 'MCE', 'Brier/10'], [r.ece, r.mce, r.brier_score/10])
            ax.set_ylabel('Score')
            ax.set_title(f'd={d}, p={mid_noise:.4f}')
            ax.set_ylim(0, 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()
