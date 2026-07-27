"""
基线比较模块 - MWPM和其他传统解码器

这个模块实现了与神经网络解码器对比的基线方法。

主要基线：
1. MWPM (Minimum Weight Perfect Matching)
   - 使用PyMatching库实现
   - 量子纠错的经典解码方法
   - 作为神经网络解码器的benchmark

2. 简单基线
   - 随机猜测
   - 多数投票
   - 阈值分类

MWPM原理：
┌─────────────────────────────────────────────────────────────────┐
│                  Minimum Weight Perfect Matching                 │
│                                                                   │
│  1. 构建匹配图                                                    │
│     - 节点：detection events（综合征变化）                        │
│     - 边权：错误链的长度/概率                                     │
│                                                                   │
│  2. 最小权完美匹配                                                 │
│     - 找到总权重最小的完美匹配                                    │
│     - 使用Blossom算法（O(n³)）                                    │
│                                                                   │
│  3. 错误链推断                                                    │
│     - 从匹配结果推断错误位置                                      │
│     - 计算逻辑运算符的奇偶性                                      │
│                                                                   │
│  优点：                                                           │
│    - 理论最优（在独立噪声假设下）                                 │
│    - 计算效率高                                                   │
│    - 无需训练                                                     │
│                                                                   │
│  缺点：                                                           │
│    - 假设噪声独立                                                 │
│    - 不利用soft readout信息                                       │
│    - 不考虑时间相关性                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

PyMatching使用：
- 需要安装: pip install pymatching
- 输入: Stim电路 + detection events
- 输出: observable预测
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np

# 尝试导入PyMatching
try:
    import pymatching
    PYMATCHING_AVAILABLE = True
except ImportError:
    PYMATCHING_AVAILABLE = False
    print("警告: PyMatching未安装。MWPM基线将不可用。")
    print("安装方法: pip install pymatching")

try:
    import stim
    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False


@dataclass
class BaselineResult:
    """基线评估结果

    存储基线方法在特定配置下的评估结果。

    Attributes:
        method: 方法名称
        distance: 码距
        rounds: 纠错轮数
        noise_p: 噪声强度
        error_rate: 错误率
        num_samples: 样本数
        time_per_sample: 每样本处理时间
        additional_info: 其他信息
    """
    method: str
    distance: int
    rounds: int
    noise_p: float
    error_rate: float
    num_samples: int
    time_per_sample: float
    additional_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return (
            f"{self.method} | d={self.distance}, r={self.rounds}, "
            f"p={self.noise_p:.4f} | "
            f"Error Rate: {self.error_rate:.4f} "
            f"({self.time_per_sample*1000:.2f}ms/sample)"
        )


class MWPMBaseline:
    """MWPM基线解码器

    使用PyMatching实现最小权完美匹配解码。

    使用示例：
        baseline = MWPMBaseline()

        # 单次评估
        result = baseline.evaluate_single(
            distance=5,
            rounds=12,
            noise_p=0.005,
            num_samples=10000,
        )
        print(result)

        # 多配置评估
        results = baseline.evaluate_grid(
            distances=[3, 5, 7],
            noise_levels=[0.001, 0.005, 0.01],
        )
    """

    def __init__(self, seed: int = 42):
        """初始化MWPM基线

        Args:
            seed: 随机种子
        """
        if not PYMATCHING_AVAILABLE:
            raise ImportError(
                "PyMatching未安装。请运行: pip install pymatching"
            )
        if not STIM_AVAILABLE:
            raise ImportError(
                "Stim未安装。请运行: pip install stim"
            )

        self.seed = seed
        np.random.seed(seed)

    def _create_circuit_and_matcher(
        self,
        distance: int,
        rounds: int,
        noise_p: float,
    ) -> Tuple[Any, Any]:
        """创建Stim电路和PyMatching匹配器

        Args:
            distance: 码距
            rounds: 纠错轮数
            noise_p: 噪声强度

        Returns:
            Tuple of (circuit, matcher)
        """
        # 生成Surface Code电路（SI1000噪声模型）
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=noise_p,
            after_reset_flip_probability=noise_p,
            before_measure_flip_probability=noise_p,
            before_round_data_depolarization=noise_p,
        )

        # 创建PyMatching匹配器
        # PyMatching需要detector error model
        detector_error_model = circuit.detector_error_model(
            decompose_errors=True
        )
        matcher = pymatching.Matching.from_detector_error_model(
            detector_error_model
        )

        return circuit, matcher

    def evaluate_single(
        self,
        distance: int,
        rounds: int,
        noise_p: float,
        num_samples: int = 10000,
    ) -> BaselineResult:
        """在单个配置上评估MWPM

        Args:
            distance: 码距
            rounds: 纠错轮数
            noise_p: 噪声强度
            num_samples: 样本数

        Returns:
            BaselineResult对象
        """
        # 创建电路和匹配器
        circuit, matcher = self._create_circuit_and_matcher(
            distance, rounds, noise_p
        )

        # 创建采样器
        sampler = circuit.compile_detector_sampler()

        # 采样detection events和observable
        start_time = time.time()

        detection_events, observable_flips = sampler.sample(
            shots=num_samples,
            separate_observables=True,
        )

        # MWPM解码
        predictions = matcher.decode_batch(detection_events)

        elapsed_time = time.time() - start_time

        # 计算错误率
        # predictions是预测的observable翻转
        # observable_flips是真实的observable翻转
        # 错误 = 预测 XOR 真实
        errors = np.logical_xor(
            predictions.flatten(),
            observable_flips.flatten()
        )
        error_rate = np.mean(errors)

        return BaselineResult(
            method="MWPM",
            distance=distance,
            rounds=rounds,
            noise_p=noise_p,
            error_rate=float(error_rate),
            num_samples=num_samples,
            time_per_sample=elapsed_time / num_samples,
            additional_info={
                "total_time": elapsed_time,
                "num_errors": int(np.sum(errors)),
            }
        )

    def evaluate_grid(
        self,
        distances: List[int],
        noise_levels: List[float],
        rounds_per_distance: Optional[Dict[int, int]] = None,
        num_samples: int = 10000,
    ) -> Dict[Tuple[int, float], BaselineResult]:
        """在网格配置上评估MWPM

        Args:
            distances: 码距列表
            noise_levels: 噪声水平列表
            rounds_per_distance: 每个码距的rounds数
            num_samples: 每个配置的样本数

        Returns:
            {(distance, noise_p): BaselineResult} 字典
        """
        if rounds_per_distance is None:
            rounds_per_distance = {d: 12 for d in distances}

        results = {}

        for d in distances:
            rounds = rounds_per_distance.get(d, 12)
            for p in noise_levels:
                print(f"评估MWPM: d={d}, rounds={rounds}, p={p:.4f}...")

                result = self.evaluate_single(
                    distance=d,
                    rounds=rounds,
                    noise_p=p,
                    num_samples=num_samples,
                )
                results[(d, p)] = result
                print(f"  错误率: {result.error_rate:.4f}")

        return results

    def compute_ler(
        self,
        distance: int,
        noise_p: float,
        rounds_list: List[int] = [3, 6, 9, 12],
        num_samples: int = 10000,
    ) -> Tuple[float, Dict[int, float]]:
        """计算MWPM的LER (Logical Error Rate per Round)

        通过在多个rounds上测量错误率，拟合得到LER。

        Args:
            distance: 码距
            noise_p: 噪声强度
            rounds_list: rounds列表
            num_samples: 每个rounds的样本数

        Returns:
            Tuple of (ler, {rounds: error_rate})
        """
        error_rates = {}

        for rounds in rounds_list:
            result = self.evaluate_single(
                distance=distance,
                rounds=rounds,
                noise_p=noise_p,
                num_samples=num_samples,
            )
            error_rates[rounds] = result.error_rate

        # 使用与evaluation/metrics.py相同的方法计算LER
        from ..evaluation.metrics import compute_fidelity, fit_ler

        fidelities = [compute_fidelity(error_rates[r]) for r in rounds_list]
        ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fidelities)

        return ler, error_rates


class RandomBaseline:
    """随机猜测基线

    始终随机预测0或1，预期错误率为50%。
    用于验证其他方法确实学到了有用的东西。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def evaluate_single(
        self,
        distance: int,
        rounds: int,
        noise_p: float,
        num_samples: int = 10000,
    ) -> BaselineResult:
        """评估随机基线"""
        start_time = time.time()

        # 随机预测
        predictions = np.random.randint(0, 2, size=num_samples)

        # 生成真实标签（使用Stim）
        if STIM_AVAILABLE:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=noise_p,
            )
            sampler = circuit.compile_detector_sampler()
            _, observable_flips = sampler.sample(
                shots=num_samples,
                separate_observables=True,
            )
            labels = observable_flips.flatten()
        else:
            # 没有Stim时假设标签均匀分布
            labels = np.random.randint(0, 2, size=num_samples)

        elapsed_time = time.time() - start_time

        # 计算错误率
        errors = predictions != labels
        error_rate = np.mean(errors)

        return BaselineResult(
            method="Random",
            distance=distance,
            rounds=rounds,
            noise_p=noise_p,
            error_rate=float(error_rate),
            num_samples=num_samples,
            time_per_sample=elapsed_time / num_samples,
        )


class MajorityVoteBaseline:
    """多数投票基线

    基于detection events的总数进行投票。
    如果events数量为奇数，预测为1（有错误）。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def evaluate_single(
        self,
        distance: int,
        rounds: int,
        noise_p: float,
        num_samples: int = 10000,
    ) -> BaselineResult:
        """评估多数投票基线"""
        if not STIM_AVAILABLE:
            raise ImportError("Stim未安装")

        start_time = time.time()

        # 生成电路和采样
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=rounds,
            distance=distance,
            after_clifford_depolarization=noise_p,
        )
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots=num_samples,
            separate_observables=True,
        )

        # 多数投票：event数量的奇偶性
        event_counts = np.sum(detection_events, axis=1)
        predictions = (event_counts % 2 == 1).astype(int)

        elapsed_time = time.time() - start_time

        # 计算错误率
        labels = observable_flips.flatten()
        errors = predictions != labels
        error_rate = np.mean(errors)

        return BaselineResult(
            method="MajorityVote",
            distance=distance,
            rounds=rounds,
            noise_p=noise_p,
            error_rate=float(error_rate),
            num_samples=num_samples,
            time_per_sample=elapsed_time / num_samples,
        )


def compare_all_baselines(
    distance: int = 5,
    rounds: int = 12,
    noise_p: float = 0.005,
    num_samples: int = 10000,
) -> Dict[str, BaselineResult]:
    """比较所有基线方法

    Args:
        distance: 码距
        rounds: 纠错轮数
        noise_p: 噪声强度
        num_samples: 样本数

    Returns:
        {method_name: BaselineResult} 字典
    """
    results = {}

    # 随机基线
    random_baseline = RandomBaseline()
    results["Random"] = random_baseline.evaluate_single(
        distance, rounds, noise_p, num_samples
    )

    # 多数投票基线
    if STIM_AVAILABLE:
        majority_baseline = MajorityVoteBaseline()
        results["MajorityVote"] = majority_baseline.evaluate_single(
            distance, rounds, noise_p, num_samples
        )

    # MWPM基线
    if PYMATCHING_AVAILABLE and STIM_AVAILABLE:
        mwpm_baseline = MWPMBaseline()
        results["MWPM"] = mwpm_baseline.evaluate_single(
            distance, rounds, noise_p, num_samples
        )

    return results


def print_baseline_comparison(results: Dict[str, BaselineResult]):
    """打印基线比较结果

    Args:
        results: {method_name: BaselineResult} 字典
    """
    print("\n" + "="*60)
    print("Baseline Comparison")
    print("="*60)

    # 按错误率排序
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].error_rate
    )

    for method, result in sorted_results:
        print(f"\n{result}")

    # 计算相对改进
    if "Random" in results and len(results) > 1:
        random_rate = results["Random"].error_rate
        print("\n" + "-"*40)
        print("相对随机基线的改进：")
        for method, result in sorted_results:
            if method != "Random":
                improvement = (random_rate - result.error_rate) / random_rate * 100
                print(f"  {method}: {improvement:.1f}%")

    print("="*60 + "\n")
