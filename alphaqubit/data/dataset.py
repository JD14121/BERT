"""
PyTorch Dataset模块 - Surface Code训练数据集

这个模块提供PyTorch兼容的数据集类，用于训练AlphaQubit神经网络解码器。

数据生成流程：
┌─────────────────┐
│  StimDataGenerator │  ← 使用Stim生成带噪声的测量数据
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SoftReadoutSimulator │  ← 将二值测量转换为软概率（可选）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SurfaceCodeDataset │  ← PyTorch Dataset接口
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DataLoader     │  ← 批量加载，多进程支持
└─────────────────┘

输出数据格式：
{
    'events':      [T, n_stab]    - detection events（float32）
    'final_soft':  [n_data]       - 最终数据测量（float32）
    'label':       [1]            - observable标签（float32，0或1）
    'stab_pos_idx': [n_stab]      - 稳定子在网格中的位置索引（int64）
}

使用示例：
    # 创建数据集
    dataset = SurfaceCodeDataset(
        distance=3,
        rounds=12,
        p=0.005,
        use_soft_readout=True,
        snr=10.0
    )

    # 获取单个样本
    sample = dataset[0]

    # 使用DataLoader
    loader = dataset.get_dataloader(batch_size=256)
    for batch in loader:
        events = batch['events']  # [B, T, n_stab]
        labels = batch['label']   # [B, 1]
        # 训练...
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .coordinates import CoordinateSystem
from .soft_readout import SoftReadoutSimulator
from .stim_generator import StimDataGenerator


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker初始化函数

    每个worker需要有独立的随机状态，否则所有worker会生成相同的数据。
    这个函数在每个worker启动时被调用。

    工作原理：
    1. 获取worker信息和基础随机种子
    2. 为该worker设置独立的随机种子
    3. 重新初始化Stim generator（因为它有内部状态）

    Args:
        worker_id: DataLoader分配的worker ID（0, 1, 2, ...）
    """
    # 获取当前worker的信息
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is not None:
        # 计算该worker的唯一种子
        base_seed = worker_info.dataset.seed
        worker_seed = base_seed + worker_id

        # 设置NumPy随机种子
        np.random.seed(worker_seed)

        # 重新初始化Stim generator
        # 这是必要的，因为Stim的sampler有内部状态
        dataset = worker_info.dataset
        dataset.generator = StimDataGenerator(
            distance=dataset.distance,
            rounds=dataset.rounds,
            p=dataset.p,
            seed=worker_seed,
        )


class SurfaceCodeDataset(Dataset):
    """Surface Code PyTorch数据集

    在线生成训练数据的Dataset实现。
    数据在__getitem__调用时实时生成，而非预先存储。

    主要特性：
    1. 在线数据生成：使用Stim实时采样
    2. 可选软读出：将二值测量转为连续概率
    3. 批量缓存：提高数据生成效率
    4. 多worker支持：DataLoader并行加载

    属性：
        distance: 码距离
        rounds: 纠错轮数
        p: 物理错误率
        use_soft_readout: 是否使用软读出
        snr: 信噪比（软读出用）
        t: 归一化测量时间（软读出用）
        num_samples: 数据集大小
        seed: 随机种子

        n_stab: 稳定子数量 = d² - 1
        n_data: 数据比特数量 = d²
        generator: Stim数据生成器
        coord_system: 坐标系统
        soft_simulator: 软读出模拟器（可选）
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        p: float = 0.005,
        use_soft_readout: bool = True,
        snr: float = 10.0,
        t: float = 0.01,
        num_samples: int = 100000,
        seed: int = 42,
        batch_generation: bool = True,
        batch_cache_size: int = 4096,  # 缓存大小（平衡启动时间和效率）
    ):
        """初始化数据集

        Args:
            distance: 码距离（必须是>=3的奇数）
            rounds: 纠错轮数
            p: 物理错误率，用于SI1000噪声模型
            use_soft_readout: 是否使用软读出
                              True: detection events是连续概率值
                              False: detection events是0/1二值
            snr: 软读出的信噪比
            t: 软读出的归一化测量时间
            num_samples: 数据集的虚拟大小
                         实际数据是在线生成的，这个值决定了一个epoch的样本数
            seed: 随机种子，用于可复现性
            batch_generation: 是否使用批量生成（推荐True，更高效）
            batch_cache_size: 内部缓存批次大小
        """
        # 保存配置
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.use_soft_readout = use_soft_readout
        self.snr = snr
        self.t = t
        self.num_samples = num_samples
        self.seed = seed
        self.batch_generation = batch_generation
        self.batch_cache_size = batch_cache_size

        # 计算派生量
        self.n_stab = distance ** 2 - 1   # 稳定子数量
        self.n_data = distance ** 2       # 数据比特数量

        # 初始化Stim数据生成器
        self.generator = StimDataGenerator(
            distance=distance,
            rounds=rounds,
            p=p,
            seed=seed,
        )

        # 获取坐标系统（用于scatter/gather操作）
        self.coord_system = self.generator.coord_system

        # 初始化软读出模拟器（如果启用）
        self.soft_simulator: Optional[SoftReadoutSimulator] = None
        if use_soft_readout:
            self.soft_simulator = SoftReadoutSimulator(snr=snr, t=t)

        # 批量缓存（用于提高效率）
        # 一次生成多个样本，然后从缓存中返回
        self._cache: Optional[Dict[str, np.ndarray]] = None
        self._cache_start_idx: int = 0
        self._cache_size: int = 0

    def __len__(self) -> int:
        """返回数据集大小

        注意：这是一个虚拟大小，实际数据是在线生成的。
        这个值决定了一个epoch包含多少样本。
        """
        return self.num_samples

    def _fill_cache(self, start_idx: int) -> None:
        """填充内部缓存

        批量生成数据比单个生成效率高得多。
        这个方法生成一批数据并存入缓存。

        Args:
            start_idx: 缓存开始的虚拟索引
        """
        if not self.batch_generation:
            return

        # 一次性采样所有需要的数据（避免双重采样！）
        raw_measurements, stim_observables = self.generator.sample(self.batch_cache_size)

        # 提取各部分
        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        final_data = self.generator.extract_final_data(raw_measurements)

        # 计算detection events
        events = self.generator.compute_detection_events(ancilla_meas)

        # 使用Stim的observable作为labels
        labels = stim_observables

        # 如果启用软读出，进行转换
        if self.soft_simulator is not None:
            # 软测量值 (soft ancilla measurements)
            # 重要：只采样一次I/Q噪声，确保measurement和event的一致性
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            # 软detection events (soft XOR)
            # 从已有的soft_meas计算，不重新采样！
            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]  # 第一轮

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            # 转换最终数据为软概率
            final_data = self.soft_simulator.simulate(final_data.astype(bool))

            measurement = soft_meas
            event = soft_events
        else:
            # 硬读出：直接转换为float32
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)
            final_data = final_data.astype(np.float32)

        # leakage 在模拟环境中为0
        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        # 更新缓存
        self._cache = {
            "measurement": measurement,
            "event": event,
            "leakage": leakage,
            "event_leakage": event_leakage,
            "final_soft": final_data,
            "labels": labels.astype(np.float32),
        }
        self._cache_start_idx = start_idx
        self._cache_size = self.batch_cache_size

    def _get_from_cache_or_generate(self, idx: int) -> Dict[str, np.ndarray]:
        """从缓存获取样本，或生成新数据

        如果请求的索引在缓存范围内，直接返回。
        否则，重新填充缓存。

        Args:
            idx: 样本索引

        Returns:
            包含单个样本数据的字典
        """
        if self.batch_generation:
            # 检查idx是否在缓存范围内
            if (
                self._cache is None
                or idx < self._cache_start_idx
                or idx >= self._cache_start_idx + self._cache_size
            ):
                # 缓存失效，重新填充
                self._fill_cache(idx)

            # 从缓存返回
            cache_idx = idx - self._cache_start_idx
            return {
                "measurement": self._cache["measurement"][cache_idx],
                "event": self._cache["event"][cache_idx],
                "leakage": self._cache["leakage"][cache_idx],
                "event_leakage": self._cache["event_leakage"][cache_idx],
                "final_soft": self._cache["final_soft"][cache_idx],
                "label": self._cache["labels"][cache_idx],
            }
        else:
            # 不使用缓存，直接生成单个样本
            return self._generate_single()

    def _generate_single(self) -> Dict[str, np.ndarray]:
        """生成单个样本

        当不使用批量缓存时调用。
        效率较低，仅用于调试。

        Returns:
            包含单个样本数据的字典
        """
        # ✅ 修复：只采样一次，确保measurement和label来自同一次采样
        raw_measurements, stim_observables = self.generator.sample(1)

        # 从同一次采样提取所有数据
        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        final_data = self.generator.extract_final_data(raw_measurements)

        # 计算detection events
        events = self.generator.compute_detection_events(ancilla_meas)

        # 使用Stim的observable作为labels（来自同一次采样）
        labels = stim_observables

        # 软读出转换（如果启用）
        if self.soft_simulator is not None:
            # 只采样一次I/Q噪声
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            # 从soft_meas计算soft events（不重新采样）
            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            final_data = self.soft_simulator.simulate(final_data.astype(bool))

            measurement = soft_meas
            event = soft_events
        else:
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)
            final_data = final_data.astype(np.float32)

        # leakage 在模拟环境中为0
        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        return {
            "measurement": measurement[0],
            "event": event[0],
            "leakage": leakage[0],
            "event_leakage": event_leakage[0],
            "final_soft": final_data[0],
            "label": labels[0].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """获取单个样本（PyTorch Dataset接口）

        这是Dataset的核心方法，被DataLoader调用。

        Args:
            idx: 样本索引
                 注意：由于数据是在线生成的，idx主要用于缓存管理，
                 而非真正的索引。同一个idx在不同epoch可能返回不同数据。

        Returns:
            包含以下键的字典：
            - 'measurement': [T, n_stab] float32张量，软测量值
            - 'event': [T, n_stab] float32张量，软detection events
            - 'leakage': [T, n_stab] float32张量，泄漏概率（模拟中为0）
            - 'event_leakage': [T, n_stab] float32张量，泄漏事件（模拟中为0）
            - 'final_soft': [n_data] float32张量，最终的数据比特测量
            - 'label': [1] float32张量，observable标签（0或1）
            - 'stab_pos_idx': [n_stab] int64张量，稳定子位置索引
        """
        # 获取数据（从缓存或新生成）
        data = self._get_from_cache_or_generate(idx)

        # 转换为PyTorch张量并返回
        return {
            "measurement": torch.from_numpy(data["measurement"]),
            "event": torch.from_numpy(data["event"]),
            "leakage": torch.from_numpy(data["leakage"]),
            "event_leakage": torch.from_numpy(data["event_leakage"]),
            "final_soft": torch.from_numpy(data["final_soft"]),
            "label": torch.tensor([data["label"]], dtype=torch.float32),
            "stab_pos_idx": self.coord_system.scatter_idx.clone(),
        }

    def get_batch(self, batch_size: int) -> Dict[str, Tensor]:
        """直接获取一个batch的数据

        这是一个便捷方法，跳过DataLoader直接生成batch。
        主要用于测试和调试。

        Args:
            batch_size: batch大小

        Returns:
            包含批量数据的字典：
            - 'measurement': [B, T, n_stab] 张量，软测量值
            - 'event': [B, T, n_stab] 张量，软detection events
            - 'leakage': [B, T, n_stab] 张量，泄漏概率
            - 'event_leakage': [B, T, n_stab] 张量，泄漏事件
            - 'final_soft': [B, n_data] 张量
            - 'label': [B, 1] 张量
            - 'stab_pos_idx': [n_stab] 张量
        """
        # ✅ 修复：只采样一次，确保measurement和label来自同一次采样
        raw_measurements, stim_observables = self.generator.sample(batch_size)

        # 从同一次采样提取所有数据
        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        final_data = self.generator.extract_final_data(raw_measurements)

        # 计算detection events
        events = self.generator.compute_detection_events(ancilla_meas)

        # 使用Stim的observable作为labels（来自同一次采样）
        labels = stim_observables

        # 软读出转换
        if self.soft_simulator is not None:
            # 只采样一次I/Q噪声
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            # 从soft_meas计算soft events（不重新采样）
            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            final_data = self.soft_simulator.simulate(final_data.astype(bool))

            measurement = soft_meas
            event = soft_events
        else:
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)
            final_data = final_data.astype(np.float32)

        # leakage 在模拟环境中为0
        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        return {
            "measurement": torch.from_numpy(measurement),
            "event": torch.from_numpy(event),
            "leakage": torch.from_numpy(leakage),
            "event_leakage": torch.from_numpy(event_leakage),
            "final_soft": torch.from_numpy(final_data),
            "label": torch.from_numpy(labels.astype(np.float32)).unsqueeze(-1),
            "stab_pos_idx": self.coord_system.scatter_idx.clone(),
        }

    def get_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
    ) -> "torch.utils.data.DataLoader":
        """创建DataLoader

        这是推荐的数据加载方式，支持：
        - 批量加载
        - 多进程并行
        - 自动shuffle

        Args:
            batch_size: 每个batch的样本数
            num_workers: 数据加载的worker进程数
                         0表示在主进程加载
                         >0表示使用多进程
            shuffle: 是否打乱数据顺序
            pin_memory: 是否将数据固定在内存中（GPU训练时推荐True）

        Returns:
            PyTorch DataLoader对象

        使用示例：
            loader = dataset.get_dataloader(batch_size=256, num_workers=4)
            for batch in loader:
                events = batch['events']
                labels = batch['label']
                # 训练...
        """
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # 多进程时需要worker初始化函数
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

    def validate(self) -> Dict[str, float]:
        """运行数据集验证检查

        生成一批数据并计算各种统计指标，用于验证数据生成是否正确。

        验证项目：
        1. event density（事件密度）：应在1-10%范围
        2. label flip rate（标签翻转率）：应合理
        3. scatter/gather round-trip：应该是恒等变换

        Returns:
            包含验证指标的字典：
            - 'event_density': 事件密度（event > 0.5的比例）
            - 'event_mean': 事件均值
            - 'final_soft_mean': 最终测量均值
            - 'label_flip_rate': 标签为1的比例
            - 'scatter_gather_valid': scatter/gather验证是否通过
        """
        # 生成测试batch
        batch = self.get_batch(1000)

        events = batch["event"].numpy()
        final_soft = batch["final_soft"].numpy()
        labels = batch["label"].numpy()

        # 计算统计指标
        metrics = {
            "event_density": float(np.mean(events > 0.5)),      # 硬化后的事件密度
            "event_mean": float(np.mean(events)),               # 原始事件均值
            "final_soft_mean": float(np.mean(final_soft)),      # 最终测量均值
            "label_flip_rate": float(np.mean(labels)),          # 标签翻转率
        }

        # 验证scatter/gather round-trip
        metrics["scatter_gather_valid"] = self.coord_system.validate()

        return metrics
