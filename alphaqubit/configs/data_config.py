"""
数据配置模块 - AlphaQubit数据生成的配置类

这个模块定义了数据生成所需的所有配置参数，包括：
- 量子纠错码的参数（距离、轮数）
- 噪声模型参数（物理错误率）
- 软读出参数（信噪比、测量时间）
- 训练参数（批大小、样本数）
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """数据生成配置类

    Surface Code的关键参数对照表：
    | 距离(d) | 稳定子数(n_stab) | 数据比特数(n_data) | 网格大小 |
    |---------|------------------|--------------------| ---------|
    | 3       | 8                | 9                  | 4×4      |
    | 5       | 24               | 25                 | 6×6      |
    | 7       | 48               | 49                 | 8×8      |
    | 9       | 80               | 81                 | 10×10    |
    | 11      | 120              | 121                | 12×12    |

    公式关系：
    - n_stab = d² - 1  （稳定子/辅助比特数量）
    - n_data = d²      （数据比特数量）
    - grid_size = d + 1（嵌入网格的边长）

    Attributes:
        distance: 码距离，决定了纠错能力，必须是>=3的奇数
                  码距离d可以纠正最多(d-1)/2个错误
        rounds: 纠错轮数，即重复测量稳定子的次数
                更多轮数可以更好地追踪错误演化
        p: 物理错误率，用于SI1000噪声模型
           典型值范围：0.001 ~ 0.01
        snr: 信噪比(Signal-to-Noise Ratio)，用于软读出模拟
             SNR越高，测量结果越接近二值(0或1)
        t: 归一化测量时间 t_meas/T1，控制幅度衰减
           t越大，|1⟩态的信号衰减越严重
        use_soft_readout: 是否使用软读出（连续概率值）
                          False时使用硬读出（二值0/1）
        batch_size: 训练时的批大小
        num_samples: 数据集总样本数
        seed: 随机种子，用于可复现性
        num_workers: DataLoader的工作进程数
    """

    # ==================== 量子纠错码参数 ====================
    distance: int = 3           # 码距离，决定纠错能力
    rounds: int = 12            # 纠错轮数（时间步数）

    # ==================== 噪声模型参数 ====================
    p: float = 0.005            # 物理错误率（SI1000噪声模型）

    # ==================== 软读出参数 ====================
    snr: float = 10.0           # 信噪比
    t: float = 0.01             # 归一化测量时间（t_meas/T1）
    use_soft_readout: bool = True  # 是否启用软读出

    # ==================== 训练参数 ====================
    batch_size: int = 256       # 批大小
    num_samples: int = 100000   # 数据集样本总数
    seed: int = 42              # 随机种子
    num_workers: int = 0        # DataLoader工作进程数

    def __post_init__(self):
        """初始化后的参数验证

        确保所有参数都在合理范围内，防止无效配置导致的错误。
        """
        # 码距离必须是>=3的奇数（Surface Code的要求）
        assert self.distance >= 3 and self.distance % 2 == 1, \
            f"码距离必须是>=3的奇数，当前值: {self.distance}"

        # 至少需要1轮纠错
        assert self.rounds >= 1, \
            f"纠错轮数必须>=1，当前值: {self.rounds}"

        # 物理错误率必须在(0,1)范围内
        assert 0 < self.p < 1, \
            f"物理错误率p必须在(0,1)范围内，当前值: {self.p}"

        # 信噪比必须为正
        assert self.snr > 0, \
            f"信噪比SNR必须为正，当前值: {self.snr}"

        # 归一化时间必须在(0,1)范围内
        assert 0 < self.t < 1, \
            f"归一化时间t必须在(0,1)范围内，当前值: {self.t}"

    @property
    def n_stab(self) -> int:
        """稳定子（辅助量子比特）数量

        对于距离为d的Rotated Surface Code：
        - X稳定子数量：(d²-1)/2
        - Z稳定子数量：(d²-1)/2
        - 总稳定子数量：d² - 1

        Returns:
            稳定子总数
        """
        return self.distance ** 2 - 1

    @property
    def n_data(self) -> int:
        """数据量子比特数量

        对于距离为d的Surface Code，数据比特排列成d×d的阵列。

        Returns:
            数据比特总数
        """
        return self.distance ** 2

    @property
    def grid_size(self) -> int:
        """嵌入网格的边长

        将稳定子嵌入到(d+1)×(d+1)的网格中，便于神经网络处理。
        网格中的空位用padding填充。

        Returns:
            网格边长
        """
        return self.distance + 1
