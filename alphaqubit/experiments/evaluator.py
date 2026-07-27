"""
完整评估流程模块 - 系统化的模型评估

这个模块实现了AlphaQubit模型的完整评估流程。

评估流程：
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Pipeline                          │
│                                                                   │
│  输入:                                                            │
│    - 训练好的模型                                                 │
│    - 评估配置（码距列表、噪声水平、rounds等）                      │
│                                                                   │
│  Step 1: 数据生成                                                │
│    ┌────────────────────────────────────────┐                    │
│    │ 为每个(distance, noise_p, rounds)组合   │                    │
│    │ 生成评估数据集                          │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 2: 模型推理                                                │
│    ┌────────────────────────────────────────┐                    │
│    │ 批量推理，记录预测概率和标签            │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 3: 计算错误率                                               │
│    ┌────────────────────────────────────────┐                    │
│    │ 对每个配置计算 E(n) = P(预测错误)       │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 4: 计算LER                                                 │
│    ┌────────────────────────────────────────┐                    │
│    │ 拟合 log F(n) = log F_0 + n×log(1-2ε)  │                    │
│    │ 从斜率得到 LER ε                        │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 5: 计算Λ因子                                               │
│    ┌────────────────────────────────────────┐                    │
│    │ Λ = LER_{d1} / LER_{d2}                │                    │
│    │ Λ > 1 表示码距增大有效                  │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 6: Bootstrap置信区间                                        │
│    ┌────────────────────────────────────────┐                    │
│    │ 重采样计算LER的不确定性                 │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  Step 7: 校准分析                                                │
│    ┌────────────────────────────────────────┐                    │
│    │ 检查预测概率的可靠性（ECE, MCE）        │                    │
│    └────────────────────────────────────────┘                    │
│                         │                                         │
│                         ▼                                         │
│  输出:                                                            │
│    - LER结果（包含Bootstrap误差）                                 │
│    - Λ因子                                                        │
│    - 校准指标                                                     │
│    - 完整评估报告                                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import SurfaceCodeDataset
from ..data.coordinates import CoordinateSystem
from ..evaluation.metrics import (
    compute_error_rate,
    compute_fidelity,
    fit_ler,
    compute_ler,
    compute_lambda,
    LERResult,
)
from ..evaluation.bootstrap import (
    bootstrap_ler,
    BootstrapResult,
)
from ..evaluation.calibration import (
    expected_calibration_error,
    get_calibration_summary,
)


@dataclass
class EvaluationConfig:
    """评估配置

    定义评估的范围和参数。

    Attributes:
        distances: 要评估的码距列表
        noise_levels: 噪声水平列表
        rounds_list: 用于计算LER的rounds列表
        num_samples: 每个配置的评估样本数
        num_bootstrap: Bootstrap重采样次数
        batch_size: 推理批大小
        device: 计算设备
        seed: 随机种子
    """
    # ==================== 评估范围 ====================
    distances: List[int] = field(default_factory=lambda: [3, 5, 7])
    noise_levels: List[float] = field(
        default_factory=lambda: [0.001, 0.002, 0.005, 0.007, 0.01]
    )
    rounds_list: List[int] = field(
        default_factory=lambda: [3, 6, 9, 12]
    )

    # ==================== 采样配置 ====================
    num_samples: int = 10000       # 每个配置的样本数
    num_bootstrap: int = 100        # Bootstrap次数

    # ==================== 计算配置 ====================
    batch_size: int = 512
    device: str = "cuda"
    seed: int = 42

    # ==================== 软读出配置 ====================
    use_soft_readout: bool = True
    snr: float = 10.0


@dataclass
class EvaluationResult:
    """评估结果

    存储单个(distance, noise_p)配置的完整评估结果。

    包含：
    - 错误率（按rounds分解）
    - LER及其置信区间
    - 校准指标
    - 推理时间统计
    """
    distance: int
    noise_p: float
    error_rates: Dict[int, float]     # {rounds: error_rate}
    fidelities: Dict[int, float]      # {rounds: fidelity}
    ler: float                         # LER
    ler_std: Optional[float]           # LER标准差（来自Bootstrap）
    ler_ci: Optional[Tuple[float, float]]  # 95%置信区间
    r_squared: float                   # 拟合R²
    is_valid: bool                     # 拟合是否有效
    ece: float                         # 期望校准误差
    mce: float                         # 最大校准误差
    brier_score: float                 # Brier分数
    num_samples: int                   # 总样本数
    inference_time: float              # 推理总时间
    time_per_sample: float             # 每样本推理时间

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "distance": self.distance,
            "noise_p": self.noise_p,
            "error_rates": self.error_rates,
            "fidelities": self.fidelities,
            "ler": self.ler,
            "ler_std": self.ler_std,
            "ler_ci": list(self.ler_ci) if self.ler_ci else None,
            "r_squared": self.r_squared,
            "is_valid": self.is_valid,
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "num_samples": self.num_samples,
            "inference_time": self.inference_time,
            "time_per_sample": self.time_per_sample,
        }

    def __str__(self) -> str:
        ler_str = f"{self.ler:.6f}"
        if self.ler_std is not None:
            ler_str += f" ± {self.ler_std:.6f}"
        return (
            f"d={self.distance}, p={self.noise_p:.4f} | "
            f"LER={ler_str} | R²={self.r_squared:.4f} | "
            f"ECE={self.ece:.4f}"
        )


class Evaluator:
    """完整评估器

    执行AlphaQubit模型的完整评估流程。

    使用示例：
        # 加载模型
        model = AlphaQubitDecoder(coord_system)
        model.load_state_dict(torch.load("best.pt"))

        # 创建评估器
        config = EvaluationConfig(
            distances=[3, 5, 7],
            noise_levels=[0.001, 0.005, 0.01],
        )
        evaluator = Evaluator(model, config)

        # 执行评估
        results = evaluator.evaluate_all()

        # 生成报告
        evaluator.generate_report(results, "eval_report.json")
    """

    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig,
        coord_systems: Optional[Dict[int, CoordinateSystem]] = None,
    ):
        """初始化评估器

        Args:
            model: 训练好的模型
            config: 评估配置
            coord_systems: {distance: CoordinateSystem} 字典
                          如果不提供，会自动创建
        """
        self.model = model
        self.config = config

        # 设置设备
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # 坐标系统
        if coord_systems is None:
            self.coord_systems = {}
            for d in config.distances:
                self.coord_systems[d] = CoordinateSystem(d)
        else:
            self.coord_systems = coord_systems

    def evaluate_all(self) -> Dict[Tuple[int, float], EvaluationResult]:
        """评估所有配置

        Returns:
            {(distance, noise_p): EvaluationResult} 字典
        """
        results = {}

        total_configs = len(self.config.distances) * len(self.config.noise_levels)
        current = 0

        for d in self.config.distances:
            for p in self.config.noise_levels:
                current += 1
                print(f"\n[{current}/{total_configs}] 评估 d={d}, p={p:.4f}...")

                result = self.evaluate_single(d, p)
                results[(d, p)] = result

                print(f"  {result}")

        return results

    def evaluate_single(
        self,
        distance: int,
        noise_p: float,
    ) -> EvaluationResult:
        """评估单个(distance, noise_p)配置

        Args:
            distance: 码距
            noise_p: 噪声强度

        Returns:
            EvaluationResult对象
        """
        error_rates = {}
        fidelities = {}
        all_probs = []
        all_labels = []
        all_preds_by_rounds = {}
        all_labels_by_rounds = {}

        total_inference_time = 0.0
        total_samples = 0

        # ==================== Step 1 & 2: 数据生成和推理 ====================
        for rounds in self.config.rounds_list:
            print(f"    rounds={rounds}...", end=" ")

            # 创建数据集
            dataset = SurfaceCodeDataset(
                distance=distance,
                rounds=rounds,
                p=noise_p,
                use_soft_readout=self.config.use_soft_readout,
                snr=self.config.snr,
                num_samples=self.config.num_samples,
                seed=self.config.seed + rounds,
            )

            # 创建DataLoader
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

            # 收集预测和标签
            round_probs = []
            round_labels = []
            round_preds = []

            start_time = time.time()

            with torch.no_grad():
                for batch in loader:
                    measurement = batch["measurement"].to(self.device)
                    event = batch["event"].to(self.device)
                    leakage = batch["leakage"].to(self.device)
                    event_leakage = batch["event_leakage"].to(self.device)
                    final_soft = batch["final_soft"].to(self.device)
                    labels = batch["label"].to(self.device)

                    # 推理
                    logits = self.model(measurement, event, leakage, event_leakage, final_soft)
                    probs = torch.sigmoid(logits).squeeze(-1)
                    preds = (probs > 0.5).long()

                    round_probs.extend(probs.cpu().numpy().tolist())
                    round_labels.extend(labels.cpu().numpy().flatten().tolist())
                    round_preds.extend(preds.cpu().numpy().tolist())

            inference_time = time.time() - start_time
            total_inference_time += inference_time
            total_samples += len(dataset)

            # 转换为numpy数组
            round_probs = np.array(round_probs)
            round_labels = np.array(round_labels)
            round_preds = np.array(round_preds)

            # 计算错误率
            e = compute_error_rate(round_preds, round_labels)
            f = compute_fidelity(e)

            error_rates[rounds] = e
            fidelities[rounds] = f

            # 保存用于Bootstrap
            all_preds_by_rounds[rounds] = round_preds
            all_labels_by_rounds[rounds] = round_labels

            # 保存用于校准分析（取最长rounds）
            if rounds == max(self.config.rounds_list):
                all_probs = round_probs
                all_labels = round_labels

            print(f"E={e:.4f}, F={f:.4f}")

        # ==================== Step 3: 计算LER ====================
        rounds_list = sorted(error_rates.keys())
        fid_list = [fidelities[r] for r in rounds_list]
        ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fid_list)

        # ==================== Step 4: Bootstrap置信区间 ====================
        print(f"    Bootstrap ({self.config.num_bootstrap} iterations)...", end=" ")

        bootstrap_result = bootstrap_ler(
            predictions_by_rounds=all_preds_by_rounds,
            labels_by_rounds=all_labels_by_rounds,
            num_bootstrap=self.config.num_bootstrap,
            seed=self.config.seed,
        )

        ler_std = bootstrap_result.ler_std
        ler_ci = (bootstrap_result.ci_lower, bootstrap_result.ci_upper)
        print(f"done")

        # ==================== Step 5: 校准分析 ====================
        calibration_summary = get_calibration_summary(
            probs=all_probs,
            labels=all_labels,
        )

        return EvaluationResult(
            distance=distance,
            noise_p=noise_p,
            error_rates=error_rates,
            fidelities=fidelities,
            ler=ler,
            ler_std=ler_std,
            ler_ci=ler_ci,
            r_squared=r_squared,
            is_valid=is_valid,
            ece=calibration_summary["ece"],
            mce=calibration_summary["mce"],
            brier_score=calibration_summary["brier_score"],
            num_samples=total_samples,
            inference_time=total_inference_time,
            time_per_sample=total_inference_time / total_samples,
        )

    def compute_lambda_factors(
        self,
        results: Dict[Tuple[int, float], EvaluationResult],
    ) -> Dict[Tuple[int, int, float], Tuple[float, float]]:
        """计算相邻码距之间的Λ因子

        Args:
            results: 评估结果字典

        Returns:
            {(d1, d2, noise_p): (lambda, lambda_std)} 字典
        """
        lambda_factors = {}

        distances = sorted(self.config.distances)

        for p in self.config.noise_levels:
            for i in range(len(distances) - 1):
                d1, d2 = distances[i], distances[i + 1]

                if (d1, p) in results and (d2, p) in results:
                    r1 = results[(d1, p)]
                    r2 = results[(d2, p)]

                    if r1.is_valid and r2.is_valid and r2.ler > 0:
                        lam = r1.ler / r2.ler

                        # 误差传播
                        if r1.ler_std and r2.ler_std:
                            rel_err_1 = r1.ler_std / r1.ler if r1.ler > 0 else 0
                            rel_err_2 = r2.ler_std / r2.ler if r2.ler > 0 else 0
                            lam_std = lam * np.sqrt(rel_err_1**2 + rel_err_2**2)
                        else:
                            lam_std = 0.0

                        lambda_factors[(d1, d2, p)] = (lam, lam_std)

        return lambda_factors

    def generate_report(
        self,
        results: Dict[Tuple[int, float], EvaluationResult],
        output_path: Optional[str] = None,
    ) -> Dict:
        """生成完整评估报告

        Args:
            results: 评估结果字典
            output_path: 输出JSON文件路径（可选）

        Returns:
            报告字典
        """
        # 计算Λ因子
        lambda_factors = self.compute_lambda_factors(results)

        # 组装报告
        report = {
            "config": {
                "distances": self.config.distances,
                "noise_levels": self.config.noise_levels,
                "rounds_list": self.config.rounds_list,
                "num_samples": self.config.num_samples,
                "num_bootstrap": self.config.num_bootstrap,
            },
            "results": {
                f"d{d}_p{p}": results[(d, p)].to_dict()
                for (d, p) in results
            },
            "lambda_factors": {
                f"d{d1}_d{d2}_p{p}": {"lambda": lam, "lambda_std": lam_std}
                for (d1, d2, p), (lam, lam_std) in lambda_factors.items()
            },
            "summary": self._generate_summary(results, lambda_factors),
        }

        # 保存到文件
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n报告已保存到: {output_path}")

        return report

    def _generate_summary(
        self,
        results: Dict[Tuple[int, float], EvaluationResult],
        lambda_factors: Dict,
    ) -> Dict:
        """生成摘要统计"""
        summary = {
            "best_ler_per_distance": {},
            "average_ece": 0.0,
            "valid_fits_percentage": 0.0,
            "average_lambda": {},
        }

        # 每个码距的最佳LER
        for d in self.config.distances:
            d_results = [
                r for (dist, p), r in results.items()
                if dist == d and r.is_valid
            ]
            if d_results:
                best = min(d_results, key=lambda x: x.ler)
                summary["best_ler_per_distance"][str(d)] = {
                    "ler": best.ler,
                    "noise_p": best.noise_p,
                }

        # 平均ECE
        eces = [r.ece for r in results.values()]
        summary["average_ece"] = float(np.mean(eces))

        # 有效拟合比例
        valid_count = sum(1 for r in results.values() if r.is_valid)
        summary["valid_fits_percentage"] = valid_count / len(results) * 100

        # 平均Λ因子
        distances = sorted(self.config.distances)
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i + 1]
            lambdas = [
                lam for (dd1, dd2, p), (lam, std) in lambda_factors.items()
                if dd1 == d1 and dd2 == d2
            ]
            if lambdas:
                summary["average_lambda"][f"{d1}_{d2}"] = float(np.mean(lambdas))

        return summary

    def print_summary(
        self,
        results: Dict[Tuple[int, float], EvaluationResult],
    ):
        """打印评估摘要

        Args:
            results: 评估结果字典
        """
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)

        # 按码距分组打印
        for d in sorted(self.config.distances):
            print(f"\n--- Distance d={d} ---")
            print(f"{'Noise p':<10} {'Error Rate':<12} {'LER':<20} {'R²':<8} {'ECE':<8}")
            print("-" * 60)

            for p in sorted(self.config.noise_levels):
                if (d, p) in results:
                    r = results[(d, p)]
                    er = r.error_rates[max(r.error_rates.keys())]
                    ler_str = f"{r.ler:.6f}"
                    if r.ler_std:
                        ler_str += f"±{r.ler_std:.6f}"
                    status = "✓" if r.is_valid else "✗"
                    print(f"{p:<10.4f} {er:<12.4f} {ler_str:<20} {r.r_squared:<8.4f} {r.ece:<8.4f} {status}")

        # Λ因子
        lambda_factors = self.compute_lambda_factors(results)
        if lambda_factors:
            print("\n--- Error Suppression Factor (Λ) ---")
            distances = sorted(self.config.distances)
            for i in range(len(distances) - 1):
                d1, d2 = distances[i], distances[i + 1]
                print(f"\nΛ_{d1}→{d2}:")
                for p in sorted(self.config.noise_levels):
                    if (d1, d2, p) in lambda_factors:
                        lam, lam_std = lambda_factors[(d1, d2, p)]
                        print(f"  p={p:.4f}: Λ={lam:.4f}±{lam_std:.4f}")

        print("=" * 70)


def quick_evaluate(
    model: nn.Module,
    distance: int = 5,
    noise_p: float = 0.005,
    rounds_list: List[int] = [3, 6, 9, 12],
    num_samples: int = 5000,
) -> EvaluationResult:
    """快速评估函数

    Args:
        model: 模型
        distance: 码距
        noise_p: 噪声强度
        rounds_list: rounds列表
        num_samples: 样本数

    Returns:
        EvaluationResult对象

    使用示例：
        result = quick_evaluate(model, distance=5, noise_p=0.005)
        print(result)
    """
    config = EvaluationConfig(
        distances=[distance],
        noise_levels=[noise_p],
        rounds_list=rounds_list,
        num_samples=num_samples,
        num_bootstrap=50,  # 快速评估使用较少的bootstrap
    )

    evaluator = Evaluator(model, config)
    return evaluator.evaluate_single(distance, noise_p)
