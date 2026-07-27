r"""
软读出模拟模块 - 量子测量的连续概率表示

这个模块模拟超导量子比特的"软读出"（soft readout）过程，
将二值测量结果转换为连续的概率值。

什么是软读出？
- 传统（硬）读出：测量结果是离散的0或1
- 软读出：测量结果是连续的概率值P(1|信号)
- 软读出保留了更多信息，有助于提高解码性能

物理模型：
超导量子比特的读出通过测量谐振腔的I/Q信号实现。
不同的量子态对应不同的信号分布：

    |0⟩态：信号 z ~ N(μ₀, σ²)，μ₀ = 0
    |1⟩态：信号 z ~ N(μ₁, σ²)，μ₁ = 1 - t/2（考虑幅度衰减）

其中：
- σ = 1/√(2·SNR) 是噪声标准差
- t = t_meas/T1 是归一化测量时间
- SNR 是信噪比

                    P(z|0)           P(z|1)
                      │                 │
                      ▼                 ▼
    概率密度    ┌─────────┐       ┌─────────┐
                │    ∧    │       │    ∧    │
                │   / \   │       │   / \   │
                │  /   \  │       │  /   \  │
                └─/─────\─┴───────┴/─────\─┴────► z
                 0              μ₁=1-t/2

后验概率计算（贝叶斯公式）：
    P(1|z) = P(z|1) / (P(z|0) + P(z|1))
"""

import numpy as np


class SoftReadoutSimulator:
    """软读出模拟器

    将二值量子测量结果转换为连续概率值，
    模拟真实超导量子比特的I/Q信号读出过程。

    物理参数：
        snr: 信噪比 (Signal-to-Noise Ratio)
             - SNR越高，|0⟩和|1⟩的信号分离越好
             - 典型值：5-20
        t: 归一化测量时间 (t_meas/T1)
           - 控制|1⟩态的幅度衰减
           - t越大，μ₁越接近0.5
           - 典型值：0.01-0.1
        sigma: 噪声标准差 = 1/√(2·SNR)
        mu_1: |1⟩态的信号均值 = 1 - t/2

    主要方法：
        simulate(): 将二值测量转换为软概率
        simulate_detection_events(): 生成软detection events
    """

    def __init__(self, snr: float = 10.0, t: float = 0.01):
        """初始化软读出模拟器

        Args:
            snr: 信噪比，控制测量的精确度
                 - 高SNR（>15）：软读出接近硬读出（接近0或1）
                 - 低SNR（<5）：软读出更分散，更接近0.5
            t: 归一化测量时间（t_meas/T1）
               - 控制|1⟩态的amplitude damping（幅度衰减）
               - t=0：无衰减，μ₁=1
               - t→1：完全衰减，μ₁→0.5
        """
        assert snr > 0, "信噪比SNR必须为正"
        assert 0 < t < 1, "归一化时间t必须在(0,1)范围内"

        self.snr = snr
        self.t = t

        # 计算噪声标准差
        # σ = 1/√(2·SNR)
        # 这来自于高斯分布的方差定义
        self.sigma = 1.0 / np.sqrt(2.0 * snr)

        # |1⟩态的信号均值（考虑amplitude damping）
        # 在测量时间t内，|1⟩态会部分衰减到|0⟩态
        # 期望信号 = 1 - t/2（近似）
        self.mu_1 = 1.0 - t / 2.0

    def sample_iq_signal(self, binary_meas: np.ndarray) -> np.ndarray:
        """采样I/Q信号

        给定真实的量子态（0或1），生成对应的I/Q信号。
        信号服从高斯分布，均值取决于量子态。

        信号模型：
            |0⟩: z ~ N(0, σ²)
            |1⟩: z ~ N(μ₁, σ²)，μ₁ = 1 - t/2

        Args:
            binary_meas: 二值测量结果（True/False 或 0/1）

        Returns:
            采样的I/Q信号值，与输入形状相同
        """
        shape = binary_meas.shape

        # 采样高斯噪声
        noise = np.random.normal(0, self.sigma, shape)

        # 根据量子态确定信号均值
        # |0⟩: mean = 0
        # |1⟩: mean = μ₁ = 1 - t/2
        signal = np.where(binary_meas, self.mu_1, 0.0)

        # 最终信号 = 均值 + 噪声
        z = signal + noise

        return z

    def compute_posterior(self, z: np.ndarray) -> np.ndarray:
        """计算后验概率 P(1|z)

        使用贝叶斯公式从观测到的I/Q信号计算后验概率：
            P(1|z) = P(z|1) / (P(z|0) + P(z|1))

        对于高斯似然：
            P(z|0) ∝ exp(-SNR · z²)
            P(z|1) ∝ exp(-SNR · (z - μ₁)²)

        为了数值稳定性，我们计算对数似然比：
            log(P(z|1)/P(z|0)) = -SNR · [(z - μ₁)² - z²]
                               = SNR · μ₁ · (2z - μ₁)

        然后使用sigmoid函数：
            P(1|z) = σ(log_ratio) = 1 / (1 + exp(-log_ratio))

        Args:
            z: 观测到的I/Q信号值

        Returns:
            后验概率 P(1|z)，范围[0, 1]
        """
        # 计算对数似然比（数值稳定的方式）
        # log(P(z|1)/P(z|0)) = SNR · μ₁ · (2z - μ₁)
        log_ratio = self.snr * self.mu_1 * (2 * z - self.mu_1)

        # 裁剪以避免数值溢出
        log_ratio = np.clip(log_ratio, -50, 50)

        # Sigmoid函数：P(1|z) = 1 / (1 + exp(-log_ratio))
        posterior = 1.0 / (1.0 + np.exp(-log_ratio))

        return posterior

    def simulate(self, binary_meas: np.ndarray) -> np.ndarray:
        """将二值测量转换为软概率

        这是主要的接口方法。完整的软读出流程：
        1. 采样I/Q信号（添加测量噪声）
        2. 计算后验概率

        Args:
            binary_meas: 二值测量结果，任意形状

        Returns:
            软概率值，范围[0, 1]，与输入形状相同
        """
        # 步骤1：采样I/Q信号
        z = self.sample_iq_signal(binary_meas.astype(bool))

        # 步骤2：计算后验概率
        soft = self.compute_posterior(z)

        return soft.astype(np.float32)

    @staticmethod
    def compute_soft_event(
        soft_curr: np.ndarray,
        soft_prev: np.ndarray,
    ) -> np.ndarray:
        """计算软detection event

        对于硬测量：event = curr XOR prev
        对于软测量，我们需要计算"软XOR"。

        概率计算：
        P(event=1) = P(curr=1, prev=0) + P(curr=0, prev=1)
                   = P(curr=1)·P(prev=0) + P(curr=0)·P(prev=1)
                   = p·(1-q) + (1-p)·q
                   = p + q - 2pq

        其中 p = P(curr=1), q = P(prev=1)

        这就是"软XOR"操作。

        Args:
            soft_curr: 当前时刻的软概率
            soft_prev: 前一时刻的软概率

        Returns:
            软detection event概率
        """
        # P(XOR=1) = p + q - 2pq
        # 这是概率论中两个独立事件恰好有一个发生的概率
        soft_event = soft_curr + soft_prev - 2 * soft_curr * soft_prev

        return soft_event.astype(np.float32)

    def simulate_detection_events(
        self,
        ancilla_meas: np.ndarray,
    ) -> np.ndarray:
        """将二值ancilla测量转换为软detection events

        完整流程：
        1. 将每轮的二值测量转换为软概率
        2. 计算相邻轮次之间的软XOR

        时间序列示意：
        t=0: soft_meas[0] → soft_event[0] = soft_meas[0] (与0做XOR)
        t=1: soft_meas[1] → soft_event[1] = soft_XOR(soft_meas[1], soft_meas[0])
        t=2: soft_meas[2] → soft_event[2] = soft_XOR(soft_meas[2], soft_meas[1])
        ...

        Args:
            ancilla_meas: [shots, T, n_stab] 二值ancilla测量

        Returns:
            [shots, T, n_stab] 软detection events
        """
        shots, T, n_stab = ancilla_meas.shape

        # 步骤1：将所有二值测量转换为软概率
        soft_meas = self.simulate(ancilla_meas)

        # 步骤2：计算软detection events
        soft_events = np.zeros_like(soft_meas)

        # 第一轮：与假设的初始值0做XOR
        # soft_XOR(p, 0) = p + 0 - 2·p·0 = p
        soft_events[:, 0, :] = soft_meas[:, 0, :]

        # 后续轮次：与前一轮做软XOR
        for t in range(1, T):
            soft_events[:, t, :] = self.compute_soft_event(
                soft_meas[:, t, :],
                soft_meas[:, t - 1, :],
            )

        return soft_events

    def get_distribution_stats(self, num_samples: int = 10000) -> dict:
        """分析软读出值的分布统计

        用于验证软读出模拟的正确性。
        应该看到双峰分布：
        - 给定|0⟩：软值集中在接近0处
        - 给定|1⟩：软值集中在接近1处

        SNR影响：
        - 高SNR：两个峰分离更好，软值更接近0/1
        - 低SNR：两个峰重叠更多，软值更分散

        Args:
            num_samples: 用于统计的样本数

        Returns:
            包含分布统计信息的字典
        """
        # 生成等量的0和1
        binary = np.random.randint(0, 2, num_samples).astype(bool)
        soft = self.simulate(binary)

        # 按真实标签分组
        soft_0 = soft[~binary]  # 给定|0⟩的软读出
        soft_1 = soft[binary]   # 给定|1⟩的软读出

        return {
            "soft_given_0_mean": np.mean(soft_0),   # |0⟩的软读出均值（应接近0）
            "soft_given_0_std": np.std(soft_0),     # |0⟩的软读出标准差
            "soft_given_1_mean": np.mean(soft_1),   # |1⟩的软读出均值（应接近1）
            "soft_given_1_std": np.std(soft_1),     # |1⟩的软读出标准差
            "separation": np.mean(soft_1) - np.mean(soft_0),  # 两峰分离度
            "snr": self.snr,                        # 当前SNR
            "sigma": self.sigma,                    # 噪声标准差
        }
