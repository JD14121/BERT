"""
多码距训练模块 - 联合训练多个码距的解码器

这个模块实现了AlphaQubit论文中的多码距训练策略。

为什么需要多码距训练？
1. 单一模型处理多个码距，减少部署复杂度
2. 不同码距之间的知识迁移，提高泛化能力
3. 小码距数据更容易生成，可以帮助大码距训练

训练策略：
┌─────────────────────────────────────────────────────────────────┐
│                   Multi-Distance Training                        │
│                                                                   │
│  策略1: 联合训练 (Joint Training)                                │
│    - 每个batch混合多个码距的样本                                  │
│    - 共享大部分参数，码距特定的位置编码                           │
│    - 优点：参数高效，泛化好                                       │
│                                                                   │
│  策略2: 课程学习 (Curriculum Learning)                           │
│    - 先训练小码距，逐渐增加大码距                                 │
│    - d=3 → d=5 → d=7 → d=9                                       │
│    - 优点：稳定训练，加速收敛                                     │
│                                                                   │
│  策略3: 渐进式训练 (Progressive Training)                         │
│    - 从小模型开始，逐渐增加容量                                   │
│    - 使用知识蒸馏传递知识                                         │
│    - 优点：适合超大码距                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

关键挑战：
1. 不同码距的grid大小不同 → 使用相对位置编码
2. 不同码距的难度不同 → 动态调整采样权重
3. 大码距样本稀少 → 重采样平衡

实现细节：
- 为每个码距创建独立的CoordinateSystem
- 模型核心参数共享，仅位置相关参数独立
- 使用DataLoader的ConcatDataset或自定义采样器
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from ..data.coordinates import CoordinateSystem
from ..data.dataset import SurfaceCodeDataset
from ..models.decoder import AlphaQubitDecoder
from ..training.trainer import TrainingConfig
from ..training.scheduler import WarmupCosineScheduler
from ..training.losses import DecoderLoss


@dataclass
class MultiDistanceConfig:
    """多码距训练配置

    继承自单码距训练配置，添加多码距特有的选项。

    Attributes:
        distances: 要训练的码距列表，如[3, 5, 7, 9]
        strategy: 训练策略，joint(联合)/curriculum(课程)/progressive(渐进)
        distance_weights: 各码距采样权重，None表示均匀采样
        curriculum_schedule: 课程学习进度表，[(step, [distances])]
        share_position_embedding: 是否共享位置嵌入
        separate_readout: 是否为每个码距使用独立的readout head
    """
    # ==================== 码距配置 ====================
    distances: List[int] = field(default_factory=lambda: [3, 5, 7])
    strategy: str = "joint"  # joint, curriculum, progressive

    # 采样权重（None表示均匀）
    distance_weights: Optional[Dict[int, float]] = None

    # 课程学习进度表：[(step, [可用码距列表])]
    # 例如：[(0, [3]), (5000, [3, 5]), (15000, [3, 5, 7])]
    curriculum_schedule: List[Tuple[int, List[int]]] = field(
        default_factory=lambda: [(0, [3]), (5000, [3, 5]), (15000, [3, 5, 7])]
    )

    # ==================== 模型配置 ====================
    share_position_embedding: bool = True  # 使用相对位置编码，可共享
    separate_readout: bool = False         # 通常共享readout head

    # ==================== 基础训练配置 ====================
    total_steps: int = 50000
    batch_size: int = 256                  # 每个码距的batch size
    eval_batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # ==================== 轮数配置 ====================
    rounds_per_distance: Dict[int, int] = field(
        default_factory=lambda: {3: 12, 5: 12, 7: 12, 9: 12}
    )
    noise_p: float = 0.005

    # ==================== 验证和保存 ====================
    eval_interval: int = 1000
    log_interval: int = 100
    save_interval: int = 2000
    early_stopping_patience: int = 10

    # ==================== 其他 ====================
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0


class DistanceCurriculum:
    """码距课程学习调度器

    根据训练步数返回当前可用的码距列表。
    随着训练进行，逐渐引入更大的码距。

    使用示例：
        curriculum = DistanceCurriculum([
            (0, [3]),        # 步骤0-4999：只训练d=3
            (5000, [3, 5]),  # 步骤5000-14999：训练d=3和d=5
            (15000, [3, 5, 7])  # 步骤15000+：训练所有码距
        ])

        distances = curriculum.get_distances(step=10000)  # 返回 [3, 5]
    """

    def __init__(self, schedule: List[Tuple[int, List[int]]]):
        """初始化课程调度器

        Args:
            schedule: 进度表，格式为[(step, [distances])]
                     必须按step升序排列
        """
        # 验证schedule按step升序
        for i in range(1, len(schedule)):
            if schedule[i][0] <= schedule[i-1][0]:
                raise ValueError("课程进度表必须按step升序排列")

        self.schedule = sorted(schedule, key=lambda x: x[0])

    def get_distances(self, step: int) -> List[int]:
        """获取当前步数对应的可用码距

        Args:
            step: 当前训练步数

        Returns:
            当前可用的码距列表
        """
        # 找到不超过当前step的最新进度
        available_distances = self.schedule[0][1]  # 默认第一个

        for milestone_step, distances in self.schedule:
            if step >= milestone_step:
                available_distances = distances
            else:
                break

        return available_distances

    def get_progress_info(self, step: int) -> Dict[str, Any]:
        """获取课程进度信息

        Args:
            step: 当前步数

        Returns:
            包含当前阶段信息的字典
        """
        current_distances = self.get_distances(step)

        # 找到下一个里程碑
        next_milestone = None
        next_distances = None
        for milestone_step, distances in self.schedule:
            if step < milestone_step:
                next_milestone = milestone_step
                next_distances = distances
                break

        return {
            "current_step": step,
            "current_distances": current_distances,
            "next_milestone": next_milestone,
            "next_distances": next_distances,
            "progress": self._compute_progress(step),
        }

    def _compute_progress(self, step: int) -> float:
        """计算课程进度(0-1)"""
        if len(self.schedule) <= 1:
            return 1.0

        final_step = self.schedule[-1][0]
        return min(1.0, step / final_step) if final_step > 0 else 1.0


class MultiDistanceDataset:
    """多码距数据集包装器

    管理多个码距的数据集，支持动态采样权重调整。

    功能：
    1. 为每个码距创建独立的SurfaceCodeDataset
    2. 根据权重动态采样
    3. 支持课程学习（动态启用/禁用某些码距）
    """

    def __init__(
        self,
        distances: List[int],
        rounds_per_distance: Dict[int, int],
        noise_p: float = 0.005,
        num_samples_per_distance: int = 100000,
        seed: int = 42,
        use_soft_readout: bool = True,
        snr: float = 10.0,
    ):
        """初始化多码距数据集

        Args:
            distances: 码距列表
            rounds_per_distance: 每个码距的rounds数
            noise_p: 噪声强度
            num_samples_per_distance: 每个码距的样本数
            seed: 随机种子
            use_soft_readout: 是否使用软读出
            snr: 信噪比
        """
        self.distances = distances
        self.datasets: Dict[int, SurfaceCodeDataset] = {}
        self.coord_systems: Dict[int, CoordinateSystem] = {}

        # 为每个码距创建数据集
        for i, d in enumerate(distances):
            rounds = rounds_per_distance.get(d, 12)
            ds = SurfaceCodeDataset(
                distance=d,
                rounds=rounds,
                p=noise_p,
                use_soft_readout=use_soft_readout,
                snr=snr,
                num_samples=num_samples_per_distance,
                seed=seed + i * 1000,  # 不同码距使用不同seed
            )
            self.datasets[d] = ds
            self.coord_systems[d] = ds.coord_system

    def get_dataset(self, distance: int) -> SurfaceCodeDataset:
        """获取指定码距的数据集"""
        return self.datasets[distance]

    def get_coord_system(self, distance: int) -> CoordinateSystem:
        """获取指定码距的坐标系统"""
        return self.coord_systems[distance]

    def get_max_distance(self) -> int:
        """获取最大码距"""
        return max(self.distances)

    def get_combined_loader(
        self,
        batch_size: int,
        active_distances: Optional[List[int]] = None,
        weights: Optional[Dict[int, float]] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, Dict[int, int]]:
        """获取组合的DataLoader

        Args:
            batch_size: 总batch大小
            active_distances: 当前激活的码距列表（用于课程学习）
            weights: 采样权重
            shuffle: 是否打乱
            num_workers: 工作进程数

        Returns:
            Tuple of (DataLoader, distance_indices)
            distance_indices记录每个样本属于哪个码距
        """
        if active_distances is None:
            active_distances = self.distances

        # 过滤出激活的数据集
        active_datasets = [self.datasets[d] for d in active_distances]

        # 合并数据集
        combined_dataset = ConcatDataset(active_datasets)

        # 计算采样权重
        if weights is not None and shuffle:
            # 创建加权采样器
            sample_weights = []
            for d in active_distances:
                ds = self.datasets[d]
                w = weights.get(d, 1.0)
                sample_weights.extend([w] * len(ds))

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(combined_dataset),
                replacement=True,
            )
            shuffle = False  # 使用sampler时不能shuffle
        else:
            sampler = None

        loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 记录每个数据集的起始索引
        distance_indices = {}
        offset = 0
        for d in active_distances:
            ds_len = len(self.datasets[d])
            distance_indices[d] = (offset, offset + ds_len)
            offset += ds_len

        return loader, distance_indices


class MultiDistanceModel(nn.Module):
    """多码距共享模型

    核心思想：大部分参数共享，仅码距相关的参数独立。

    共享的参数：
    - 事件嵌入（使用相对位置编码）
    - RNN参数
    - 卷积参数（使用相对坐标）
    - Readout head

    独立的参数：
    - 绝对位置嵌入（如果不使用相对位置）
    - Late Fusion中的stab-to-data映射
    """

    def __init__(
        self,
        coord_systems: Dict[int, CoordinateSystem],
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_rnn_layers: int = 2,
        rnn_type: str = "gru",
        use_conv: bool = True,
        num_conv_layers: int = 2,
        soft_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        """初始化多码距模型

        Args:
            coord_systems: {distance: CoordinateSystem} 字典
            其他参数与AlphaQubitDecoder相同
        """
        super().__init__()

        self.coord_systems = coord_systems
        self.distances = sorted(coord_systems.keys())
        self.max_distance = max(self.distances)

        # 为最大码距创建基础模型
        # 小码距通过padding适配
        max_coord = coord_systems[self.max_distance]

        self.decoder = AlphaQubitDecoder(
            coord_system=max_coord,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            use_conv=use_conv,
            num_conv_layers=num_conv_layers,
            soft_embed_dim=soft_embed_dim,
            dropout=dropout,
        )

        # 保存维度信息
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        events: torch.Tensor,
        final_soft: torch.Tensor,
        distance: Optional[int] = None,
    ) -> torch.Tensor:
        """前向传播

        对于非最大码距的输入，需要进行padding到最大码距的尺寸。

        Args:
            events: [B, T, n_stab]
            final_soft: [B, n_data]
            distance: 当前样本的码距（如果不指定，假设是最大码距）

        Returns:
            logit: [B, 1]
        """
        if distance is None or distance == self.max_distance:
            # 最大码距直接处理
            return self.decoder(events, final_soft)

        # 小码距需要padding
        B, T, n_stab = events.shape
        n_data = final_soft.shape[1]

        max_n_stab = self.coord_systems[self.max_distance].n_stab
        max_n_data = self.coord_systems[self.max_distance].n_data

        # Padding events
        if n_stab < max_n_stab:
            pad_stab = torch.zeros(B, T, max_n_stab - n_stab, device=events.device)
            events_padded = torch.cat([events, pad_stab], dim=2)
        else:
            events_padded = events

        # Padding final_soft
        if n_data < max_n_data:
            pad_data = torch.zeros(B, max_n_data - n_data, device=final_soft.device)
            final_soft_padded = torch.cat([final_soft, pad_data], dim=1)
        else:
            final_soft_padded = final_soft

        return self.decoder(events_padded, final_soft_padded)

    def get_num_parameters(self) -> Dict[str, int]:
        """获取参数数量"""
        return self.decoder.get_num_parameters()


class MultiDistanceTrainer:
    """多码距训练器

    管理多个码距的联合训练流程。

    主要功能：
    1. 联合训练：在每个batch中混合多个码距的样本
    2. 课程学习：逐渐引入更大的码距
    3. 动态权重：根据训练进度调整采样权重
    4. 分别评估：在每个码距上独立评估性能

    使用示例：
        config = MultiDistanceConfig(
            distances=[3, 5, 7],
            strategy="curriculum",
            total_steps=50000,
        )
        trainer = MultiDistanceTrainer(config)
        history = trainer.train()
    """

    def __init__(
        self,
        config: MultiDistanceConfig,
        save_dir: Optional[str] = None,
        logger: Optional[Callable] = None,
    ):
        """初始化多码距训练器

        Args:
            config: 训练配置
            save_dir: 检查点保存目录
            logger: 日志记录函数
        """
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints_multi")
        self.logger = logger or self._default_logger

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # ==================== 创建数据集 ====================
        self.multi_dataset = MultiDistanceDataset(
            distances=config.distances,
            rounds_per_distance=config.rounds_per_distance,
            noise_p=config.noise_p,
            seed=config.seed,
        )

        # 验证集
        self.val_datasets: Dict[int, SurfaceCodeDataset] = {}
        for d in config.distances:
            rounds = config.rounds_per_distance.get(d, 12)
            self.val_datasets[d] = SurfaceCodeDataset(
                distance=d,
                rounds=rounds,
                p=config.noise_p,
                num_samples=10000,
                seed=config.seed + 10000,
            )

        # ==================== 创建模型 ====================
        self.model = MultiDistanceModel(
            coord_systems=self.multi_dataset.coord_systems,
            embed_dim=64,
            hidden_dim=128,
            num_rnn_layers=2,
        ).to(self.device)

        # ==================== 创建优化器 ====================
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ==================== 创建调度器 ====================
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # ==================== 课程学习 ====================
        if config.strategy == "curriculum":
            self.curriculum = DistanceCurriculum(config.curriculum_schedule)
        else:
            self.curriculum = None

        # ==================== 损失函数 ====================
        self.loss_fn = DecoderLoss()

        # ==================== 训练状态 ====================
        self.global_step = 0
        self.best_val_loss = {d: float('inf') for d in config.distances}
        self.history = {
            "train_loss": [],
            "val_loss": {d: [] for d in config.distances},
            "learning_rate": [],
        }

    def _default_logger(self, metrics: Dict):
        """默认日志记录器"""
        msg = " | ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        print(msg)

    def train(self) -> Dict:
        """执行完整训练

        Returns:
            训练历史字典
        """
        self.logger({
            "message": f"开始多码距训练",
            "distances": self.config.distances,
            "strategy": self.config.strategy,
            "device": str(self.device),
        })

        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger({"message": f"模型参数量: {param_count:,}"})

        start_time = time.time()

        for step in range(self.config.total_steps):
            self.global_step = step

            # ==================== 获取当前激活的码距 ====================
            if self.curriculum is not None:
                active_distances = self.curriculum.get_distances(step)
            else:
                active_distances = self.config.distances

            # ==================== 训练步 ====================
            train_metrics = self._train_step(active_distances)

            # ==================== 日志 ====================
            if step % self.config.log_interval == 0:
                train_metrics["step"] = step
                train_metrics["lr"] = self.scheduler.get_last_lr()[0]
                train_metrics["active_distances"] = active_distances
                train_metrics["elapsed"] = time.time() - start_time
                self.logger(train_metrics)

            # ==================== 验证 ====================
            if step > 0 and step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                val_metrics["step"] = step
                self.logger({"validation": val_metrics})

                # 保存最佳模型
                for d in self.config.distances:
                    if val_metrics[f"loss_d{d}"] < self.best_val_loss[d]:
                        self.best_val_loss[d] = val_metrics[f"loss_d{d}"]
                        self._save_checkpoint(f"best_d{d}.pt")

            # ==================== 定期保存 ====================
            if step > 0 and step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{step}.pt")

        # 保存最终检查点
        self._save_checkpoint("final.pt")
        self.logger({"message": "训练完成"})

        return self.history

    def _train_step(self, active_distances: List[int]) -> Dict:
        """执行单个训练步

        Args:
            active_distances: 当前激活的码距列表

        Returns:
            训练指标字典
        """
        self.model.train()

        total_loss = 0.0
        num_samples = 0

        # 对每个激活的码距采样
        for d in active_distances:
            dataset = self.multi_dataset.get_dataset(d)

            # 随机采样一个batch
            indices = np.random.choice(
                len(dataset),
                size=min(self.config.batch_size, len(dataset)),
                replace=False,
            )

            batch_measurement = []
            batch_event = []
            batch_leakage = []
            batch_event_leakage = []
            batch_final_soft = []
            batch_labels = []

            for idx in indices:
                sample = dataset[idx]
                batch_measurement.append(sample["measurement"])
                batch_event.append(sample["event"])
                batch_leakage.append(sample["leakage"])
                batch_event_leakage.append(sample["event_leakage"])
                batch_final_soft.append(sample["final_soft"])
                batch_labels.append(sample["label"])

            measurement = torch.stack(batch_measurement).to(self.device)
            event = torch.stack(batch_event).to(self.device)
            leakage = torch.stack(batch_leakage).to(self.device)
            event_leakage = torch.stack(batch_event_leakage).to(self.device)
            final_soft = torch.stack(batch_final_soft).to(self.device)
            labels = torch.stack(batch_labels).to(self.device)

            # 前向传播
            logits = self.model(measurement, event, leakage, event_leakage, final_soft, distance=d)

            # 计算损失
            loss, _ = self.loss_fn(logits, labels)
            total_loss += loss * len(indices)
            num_samples += len(indices)

        # 平均损失
        avg_loss = total_loss / num_samples

        # 反向传播
        self.optimizer.zero_grad()
        avg_loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # 优化器步进
        self.optimizer.step()
        self.scheduler.step()

        # 记录历史
        self.history["train_loss"].append(avg_loss.item())
        self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])

        return {
            "loss": avg_loss.item(),
            "grad_norm": grad_norm.item(),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """在所有码距的验证集上评估

        Returns:
            各码距的验证指标
        """
        self.model.eval()

        results = {}

        for d in self.config.distances:
            val_dataset = self.val_datasets[d]

            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            # 批量评估
            batch_size = self.config.eval_batch_size
            num_batches = (len(val_dataset) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(val_dataset))

                batch_measurement = []
                batch_event = []
                batch_leakage = []
                batch_event_leakage = []
                batch_final_soft = []
                batch_labels = []

                for idx in range(start_idx, end_idx):
                    sample = val_dataset[idx]
                    batch_measurement.append(sample["measurement"])
                    batch_event.append(sample["event"])
                    batch_leakage.append(sample["leakage"])
                    batch_event_leakage.append(sample["event_leakage"])
                    batch_final_soft.append(sample["final_soft"])
                    batch_labels.append(sample["label"])

                measurement = torch.stack(batch_measurement).to(self.device)
                event = torch.stack(batch_event).to(self.device)
                leakage = torch.stack(batch_leakage).to(self.device)
                event_leakage = torch.stack(batch_event_leakage).to(self.device)
                final_soft = torch.stack(batch_final_soft).to(self.device)
                labels = torch.stack(batch_labels).to(self.device)

                logits = self.model(measurement, event, leakage, event_leakage, final_soft, distance=d)
                loss, metrics = self.loss_fn(logits, labels)

                batch_samples = end_idx - start_idx
                total_loss += loss.item() * batch_samples
                total_correct += metrics["accuracy"] * batch_samples
                total_samples += batch_samples

            results[f"loss_d{d}"] = total_loss / total_samples
            results[f"acc_d{d}"] = total_correct / total_samples

            # 记录历史
            self.history["val_loss"][d].append(total_loss / total_samples)

        return results

    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        self.logger({"message": f"从 {path} 加载检查点，step: {self.global_step}"})


def train_multi_distance(
    distances: List[int] = [3, 5, 7],
    strategy: str = "joint",
    total_steps: int = 50000,
    save_dir: str = "checkpoints_multi",
) -> Dict:
    """便捷的多码距训练函数

    Args:
        distances: 码距列表
        strategy: 训练策略 (joint/curriculum)
        total_steps: 总训练步数
        save_dir: 保存目录

    Returns:
        训练历史

    使用示例：
        history = train_multi_distance(
            distances=[3, 5, 7],
            strategy="curriculum",
            total_steps=50000,
        )
    """
    config = MultiDistanceConfig(
        distances=distances,
        strategy=strategy,
        total_steps=total_steps,
    )

    trainer = MultiDistanceTrainer(config, save_dir=save_dir)
    return trainer.train()
