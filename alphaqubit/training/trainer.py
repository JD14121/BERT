"""
训练器模块 - 完整的训练循环管理

这个模块实现了AlphaQubit模型的完整训练流程。

训练器功能：
1. 训练循环管理
2. 梯度裁剪
3. 学习率调度
4. 验证和评估
5. 检查点保存/加载
6. 日志记录
7. 早停机制

训练流程：
┌─────────────────────────────────────────────────────────────┐
│                      Trainer.train()                         │
│                                                              │
│  初始化:                                                     │
│    - 创建优化器 (AdamW)                                      │
│    - 创建学习率调度器 (WarmupCosine)                         │
│    - 创建损失函数 (BCE)                                      │
│                                                              │
│  训练循环:                                                   │
│    for step in range(total_steps):                          │
│        1. 获取batch                                          │
│        2. 前向传播                                           │
│        3. 计算损失                                           │
│        4. 反向传播                                           │
│        5. 梯度裁剪 (max_norm=1.0)                           │
│        6. 优化器步进                                         │
│        7. 学习率调度器步进                                   │
│                                                              │
│        if step % log_interval == 0:                         │
│            记录训练指标                                      │
│                                                              │
│        if step % eval_interval == 0:                        │
│            在验证集上评估                                    │
│            如果是最佳模型，保存检查点                        │
│                                                              │
│  结束:                                                       │
│    - 保存最终检查点                                          │
│    - 返回训练历史                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘

推荐配置：
| 参数 | d=3 | d=5 | d=9 |
|------|-----|-----|-----|
| batch_size | 512 | 512 | 256 |
| learning_rate | 2e-4 | 2e-4 | 1e-4 |
| total_steps | 30k | 30k | 50k |
| warmup_steps | 1000 | 1000 | 1000 |
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .scheduler import WarmupCosineScheduler, RoundsCurriculum
from .losses import DecoderLoss


@dataclass
class TrainingConfig:
    """训练配置

    包含训练所需的所有超参数。
    使用dataclass便于配置管理和序列化。
    """
    # ==================== 基本配置 ====================
    total_steps: int = 30000           # 总训练步数
    batch_size: int = 512              # 批大小
    eval_batch_size: int = 1024        # 验证批大小

    # ==================== 优化器配置 ====================
    learning_rate: float = 2e-4        # 学习率
    weight_decay: float = 0.01         # 权重衰减（L2正则化）
    max_grad_norm: float = 1.0         # 梯度裁剪阈值

    # ==================== 学习率调度 ====================
    warmup_steps: int = 1000           # Warmup步数
    min_lr_ratio: float = 0.1          # 最终学习率比例

    # ==================== 轮数课程 ====================
    use_rounds_curriculum: bool = True # 是否使用轮数课程
    rounds_schedule: List[tuple] = field(
        default_factory=lambda: [(0, 3), (5000, 6), (15000, 12)]
    )

    # ==================== 验证和保存 ====================
    eval_interval: int = 500           # 验证间隔（步数）
    log_interval: int = 100            # 日志间隔（步数）
    save_interval: int = 1000          # 保存间隔（步数）
    early_stopping_patience: int = 10  # 早停耐心值（eval次数）

    # ==================== 损失函数 ====================
    label_smoothing: float = 0.0       # 标签平滑
    focal_gamma: float = 0.0           # Focal loss gamma

    # ==================== 其他 ====================
    seed: int = 42                     # 随机种子
    device: str = "cuda"               # 训练设备
    # 注意: num_workers>0 在 Windows 上会因 Stim sampler 无法 pickle 而失败
    # 解决方案: 使用预生成数据集或保持 num_workers=0
    num_workers: int = 0               # DataLoader工作进程数

    # ==================== 性能优化 ====================
    use_amp: bool = True               # 使用混合精度训练（大幅加速）
    use_compile: bool = False          # 使用torch.compile（PyTorch 2.0+）


class Trainer:
    """AlphaQubit训练器

    管理完整的训练流程，包括：
    - 训练循环
    - 验证评估
    - 检查点管理
    - 日志记录
    - 早停机制

    使用示例：
        config = TrainingConfig(
            total_steps=30000,
            batch_size=512,
        )
        trainer = Trainer(model, train_dataset, val_dataset, config)
        history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: TrainingConfig,
        save_dir: Optional[str] = None,
        logger: Optional[Callable] = None,
    ):
        """初始化训练器

        Args:
            model: AlphaQubit模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 训练配置
            save_dir: 检查点保存目录
            logger: 日志记录函数（接收dict）
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.logger = logger or self._default_logger

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 创建数据加载器
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

        # 创建优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 创建学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # 创建轮数课程
        if config.use_rounds_curriculum:
            self.rounds_curriculum = RoundsCurriculum(config.rounds_schedule)
        else:
            self.rounds_curriculum = None

        # 创建损失函数
        self.loss_fn = DecoderLoss(
            label_smoothing=config.label_smoothing,
            focal_gamma=config.focal_gamma,
        )

        # ==================== 性能优化 ====================
        # 混合精度训练
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger({"message": "启用混合精度训练 (AMP)"})
        else:
            self.scaler = None

        # torch.compile加速（PyTorch 2.0+）
        if config.use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            self.logger({"message": "启用 torch.compile 加速"})

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
        }

    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        batch_size = self.config.batch_size if shuffle else self.config.eval_batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

    def _default_logger(self, metrics: Dict):
        """默认日志记录器"""
        msg = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                          for k, v in metrics.items()])
        print(msg)

    def _format_metrics(self, metrics: Dict) -> str:
        """格式化指标为字符串"""
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                if k == 'accuracy' or k.endswith('_rate'):
                    parts.append(f"{k}: {v:.2%}")
                else:
                    parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def train(self) -> Dict:
        """执行完整训练

        Returns:
            训练历史字典
        """
        self.logger({"message": f"开始训练，设备: {self.device}"})
        self.logger({"message": f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}"})

        # 训练数据迭代器
        train_iter = iter(self.train_loader)
        start_time = time.time()

        # 创建进度条
        if HAS_TQDM:
            pbar = tqdm(total=self.config.total_steps, desc="Training", unit="step",
                       dynamic_ncols=True, leave=True)
        else:
            pbar = None

        for step in range(self.config.total_steps):
            self.global_step = step

            # ==================== 获取batch ====================
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # ==================== 训练步 ====================
            train_metrics = self._train_step(batch)

            # ==================== 更新进度条 ====================
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{train_metrics['loss']:.4f}",
                    'acc': f"{train_metrics['accuracy']:.2%}",
                })

            # ==================== 日志 ====================
            if step % self.config.log_interval == 0:
                train_metrics['step'] = step
                train_metrics['lr'] = self.scheduler.get_last_lr()[0]
                train_metrics['elapsed'] = time.time() - start_time
                if pbar is not None:
                    # 使用 tqdm.write 避免干扰进度条
                    tqdm.write(self._format_metrics(train_metrics))
                else:
                    self.logger(train_metrics)

            # ==================== 验证 ====================
            if step > 0 and step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                val_metrics['step'] = step

                # 记录历史
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])

                val_msg = f"[Eval] step: {step} | val_loss: {val_metrics['loss']:.4f} | val_acc: {val_metrics['accuracy']:.2%}"
                if pbar is not None:
                    tqdm.write(val_msg)
                else:
                    self.logger({"validation": val_metrics})

                # 检查是否是最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint('best.pt')
                    msg = f"[Save] Best model saved, val_loss: {val_metrics['loss']:.4f}"
                    if pbar is not None:
                        tqdm.write(msg)
                    else:
                        self.logger({"message": msg})
                else:
                    self.patience_counter += 1

                # 早停检查
                if self.patience_counter >= self.config.early_stopping_patience:
                    msg = f"[Early Stop] patience: {self.patience_counter}"
                    if pbar is not None:
                        tqdm.write(msg)
                    else:
                        self.logger({"message": msg})
                    break

            # ==================== 定期保存 ====================
            if step > 0 and step % self.config.save_interval == 0:
                self._save_checkpoint(f'step_{step}.pt')

        # 关闭进度条
        if pbar is not None:
            pbar.close()

        # 保存最终检查点
        self._save_checkpoint('final.pt')
        self.logger({"message": "训练完成"})

        return self.history

    def _train_step(self, batch: Dict) -> Dict:
        """执行单个训练步

        Args:
            batch: 数据batch

        Returns:
            训练指标字典
        """
        self.model.train()

        # 移动数据到设备（5输入模型接口），使用non_blocking加速
        measurement = batch['measurement'].to(self.device, non_blocking=True)
        event = batch['event'].to(self.device, non_blocking=True)
        leakage = batch['leakage'].to(self.device, non_blocking=True)
        event_leakage = batch['event_leakage'].to(self.device, non_blocking=True)
        final_soft = batch['final_soft'].to(self.device, non_blocking=True)
        labels = batch['label'].to(self.device, non_blocking=True)

        # 使用混合精度训练
        if self.use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # 前向传播
                logits = self.model(measurement, event, leakage, event_leakage, final_soft)
                # 计算损失
                loss, metrics = self.loss_fn(logits, labels)

            # 反向传播（使用scaler）
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 前向传播
            logits = self.model(measurement, event, leakage, event_leakage, final_soft)

            # 计算损失
            loss, metrics = self.loss_fn(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            # 优化器步进
            self.optimizer.step()

        # 学习率调度器步进
        self.scheduler.step()

        # 记录历史
        self.history['train_loss'].append(loss.item())
        self.history['train_acc'].append(metrics['accuracy'])
        self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """在验证集上评估

        Returns:
            验证指标字典
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            # 5输入模型接口
            measurement = batch['measurement'].to(self.device, non_blocking=True)
            event = batch['event'].to(self.device, non_blocking=True)
            leakage = batch['leakage'].to(self.device, non_blocking=True)
            event_leakage = batch['event_leakage'].to(self.device, non_blocking=True)
            final_soft = batch['final_soft'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # 使用混合精度
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(measurement, event, leakage, event_leakage, final_soft)
                    loss, metrics = self.loss_fn(logits, labels)
            else:
                logits = self.model(measurement, event, leakage, event_leakage, final_soft)
                loss, metrics = self.loss_fn(logits, labels)

            batch_size = measurement.size(0)
            total_loss += loss.item() * batch_size
            total_correct += metrics['accuracy'] * batch_size
            total_samples += batch_size

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
        }

    def _save_checkpoint(self, filename: str):
        """保存检查点

        Args:
            filename: 检查点文件名
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def load_checkpoint(self, path: str):
        """加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.logger({"message": f"从 {path} 加载检查点，step: {self.global_step}"})


def train_decoder(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: Optional[TrainingConfig] = None,
    save_dir: str = "checkpoints",
) -> Dict:
    """便捷的训练函数

    Args:
        model: AlphaQubit模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        config: 训练配置（可选）
        save_dir: 保存目录

    Returns:
        训练历史

    使用示例：
        from alphaqubit.data import SurfaceCodeDataset
        from alphaqubit.models import AlphaQubitDecoder

        train_ds = SurfaceCodeDataset(distance=3, rounds=12)
        val_ds = SurfaceCodeDataset(distance=3, rounds=12, seed=123)

        model = AlphaQubitDecoder(train_ds.coord_system)

        history = train_decoder(model, train_ds, val_ds)
    """
    if config is None:
        config = TrainingConfig()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=save_dir,
    )

    return trainer.train()
