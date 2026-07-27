"""
预训练数据集模块 - PretrainDataset

这个模块实现了预训练阶段的数据集：
1. 不需要 label（只生成 syndrome 序列）
2. 可选：同时生成 mask 后的输入和 mask 索引
3. 复用现有的 StimDataGenerator 和 SoftReadoutSimulator

与 SurfaceCodeDataset 的区别：
- 不需要 label 和 final_soft（预训练阶段不使用）
- 可选：在 __getitem__ 中直接返回 mask 后的输入
- 数据生成更快（不需要 Pauli Frame 追踪）
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .coordinates import CoordinateSystem
from .soft_readout import SoftReadoutSimulator
from .stim_generator import StimDataGenerator


class PretrainDataset(Dataset):
    """预训练数据集

    在线生成 syndrome 序列，不需要 label。
    可选在采样时直接应用 masking。

    属性：
        distance: 码距离
        rounds: 纠错轮数
        p: 物理错误率
        use_soft_readout: 是否使用软读出
        snr: 信噪比
        t: 归一化测量时间
        num_samples: 数据集大小
        seed: 随机种子
        apply_masking: 是否在 __getitem__ 中应用 masking
        mask_ratio: masking 比例

    输出格式（无 masking）：
        {
            'measurement': [T, n_stab] float32,
            'event': [T, n_stab] float32,
            'leakage': [T, n_stab] float32,
            'event_leakage': [T, n_stab] float32,
            'stab_pos_idx': [n_stab] int64,
        }

    输出格式（有 masking）：
        {
            'measurement': [T, n_stab] float32,
            'event': [T, n_stab] float32,
            'leakage': [T, n_stab] float32,
            'event_leakage': [T, n_stab] float32,
            'stab_pos_idx': [n_stab] int64,
            'masked_measurement': [T, n_stab] float32,
            'masked_event': [T, n_stab] float32,
            'mask_indices': [T, n_stab] bool,
        }
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
        batch_cache_size: int = 4096,
        apply_masking: bool = False,
        mask_ratio: float = 0.15,
    ):
        """初始化预训练数据集

        Args:
            distance: 码距离
            rounds: 纠错轮数
            p: 物理错误率
            use_soft_readout: 是否使用软读出
            snr: 信噪比
            t: 归一化测量时间
            num_samples: 数据集虚拟大小
            seed: 随机种子
            batch_generation: 是否使用批量生成
            batch_cache_size: 缓存批次大小
            apply_masking: 是否在 __getitem__ 中应用 masking
            mask_ratio: masking 比例
        """
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
        self.apply_masking = apply_masking
        self.mask_ratio = mask_ratio

        self.n_stab = distance ** 2 - 1
        self.n_data = distance ** 2

        # 初始化 Stim 数据生成器
        self.generator = StimDataGenerator(
            distance=distance,
            rounds=rounds,
            p=p,
            seed=seed,
        )

        self.coord_system = self.generator.coord_system

        # 初始化软读出模拟器
        self.soft_simulator: Optional[SoftReadoutSimulator] = None
        if use_soft_readout:
            self.soft_simulator = SoftReadoutSimulator(snr=snr, t=t)

        # 批量缓存
        self._cache: Optional[Dict[str, np.ndarray]] = None
        self._cache_start_idx: int = 0
        self._cache_size: int = 0

    def __len__(self) -> int:
        return self.num_samples

    def _fill_cache(self, start_idx: int) -> None:
        """填充内部缓存"""
        if not self.batch_generation:
            return

        # 采样原始测量（不需要 observables）
        raw_measurements, _ = self.generator.sample(self.batch_cache_size)

        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        final_data = self.generator.extract_final_data(raw_measurements)

        events = self.generator.compute_detection_events(ancilla_meas)

        # 软读出转换
        if self.soft_simulator is not None:
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            measurement = soft_meas
            event = soft_events
        else:
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)

        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        self._cache = {
            "measurement": measurement,
            "event": event,
            "leakage": leakage,
            "event_leakage": event_leakage,
        }
        self._cache_start_idx = start_idx
        self._cache_size = self.batch_cache_size

    def _get_from_cache_or_generate(self, idx: int) -> Dict[str, np.ndarray]:
        """从缓存获取样本，或生成新数据"""
        if self.batch_generation:
            if (
                self._cache is None
                or idx < self._cache_start_idx
                or idx >= self._cache_start_idx + self._cache_size
            ):
                self._fill_cache(idx)

            cache_idx = idx - self._cache_start_idx
            return {
                "measurement": self._cache["measurement"][cache_idx],
                "event": self._cache["event"][cache_idx],
                "leakage": self._cache["leakage"][cache_idx],
                "event_leakage": self._cache["event_leakage"][cache_idx],
            }
        else:
            return self._generate_single()

    def _generate_single(self) -> Dict[str, np.ndarray]:
        """生成单个样本"""
        raw_measurements, _ = self.generator.sample(1)

        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        events = self.generator.compute_detection_events(ancilla_meas)

        if self.soft_simulator is not None:
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            measurement = soft_meas
            event = soft_events
        else:
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)

        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        return {
            "measurement": measurement[0],
            "event": event[0],
            "leakage": leakage[0],
            "event_leakage": event_leakage[0],
        }

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """获取单个样本

        Returns:
            如果 apply_masking=False:
                {
                    'measurement': [T, n_stab],
                    'event': [T, n_stab],
                    'leakage': [T, n_stab],
                    'event_leakage': [T, n_stab],
                    'stab_pos_idx': [n_stab],
                }
            如果 apply_masking=True:
                {
                    'measurement': [T, n_stab],          # 原始值（作为 target）
                    'event': [T, n_stab],                  # 原始值（作为 target）
                    'leakage': [T, n_stab],
                    'event_leakage': [T, n_stab],
                    'stab_pos_idx': [n_stab],
                    'masked_measurement': [T, n_stab],     # mask 后的输入
                    'masked_event': [T, n_stab],           # mask 后的输入
                    'mask_indices': [T, n_stab],           # bool
                }
        """
        data = self._get_from_cache_or_generate(idx)

        result = {
            "measurement": torch.from_numpy(data["measurement"]),
            "event": torch.from_numpy(data["event"]),
            "leakage": torch.from_numpy(data["leakage"]),
            "event_leakage": torch.from_numpy(data["event_leakage"]),
            "stab_pos_idx": self.coord_system.scatter_idx.clone(),
        }

        # 可选：应用 masking
        if self.apply_masking:
            T, n_stab = result["measurement"].shape
            mask_indices = self._generate_mask_indices(T, n_stab)

            masked_measurement = self._apply_mask(result["measurement"], mask_indices)
            masked_event = self._apply_mask(result["event"], mask_indices)

            result["masked_measurement"] = masked_measurement
            result["masked_event"] = masked_event
            result["mask_indices"] = mask_indices

        return result

    def _generate_mask_indices(self, T: int, n_stab: int) -> Tensor:
        """生成 mask 位置"""
        rand = torch.rand(T, n_stab)
        return rand < self.mask_ratio

    def _apply_mask(self, tensor: Tensor, mask_indices: Tensor) -> Tensor:
        """应用 masking"""
        masked = tensor.clone()
        rand = torch.rand_like(tensor)

        mask_token_mask = mask_indices & (rand < 0.8)
        masked = torch.where(mask_token_mask, 0.5, masked)

        random_mask = mask_indices & (rand >= 0.8) & (rand < 0.9)
        random_values = (torch.rand_like(tensor) > 0.5).float()
        masked = torch.where(random_mask, random_values, masked)

        return masked

    def get_batch(self, batch_size: int) -> Dict[str, Tensor]:
        """直接获取一个 batch 的数据"""
        raw_measurements, _ = self.generator.sample(batch_size)

        ancilla_meas = self.generator.extract_ancilla_measurements(raw_measurements)
        events = self.generator.compute_detection_events(ancilla_meas)

        if self.soft_simulator is not None:
            soft_meas = self.soft_simulator.simulate(ancilla_meas)

            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            measurement = soft_meas
            event = soft_events
        else:
            measurement = ancilla_meas.astype(np.float32)
            event = events.astype(np.float32)

        leakage = np.zeros_like(measurement)
        event_leakage = np.zeros_like(event)

        return {
            "measurement": torch.from_numpy(measurement),
            "event": torch.from_numpy(event),
            "leakage": torch.from_numpy(leakage),
            "event_leakage": torch.from_numpy(event_leakage),
            "stab_pos_idx": self.coord_system.scatter_idx.clone(),
        }
