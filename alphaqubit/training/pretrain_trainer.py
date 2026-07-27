"""
预训练 Trainer 模块 - PretrainTrainer

这个模块实现了预训练阶段的训练循环：
1. 使用 PretrainDataset 生成无标注数据
2. 应用 MaskedSyndromeModeling
3. 通过 PretrainDecoder 预测 mask 位置
4. 仅计算 mask 位置的 BCE loss

与原始 Trainer 的区别：
- 不需要 label 和 final_soft
- 使用 PretrainLoss 而非 DecoderLoss
- 监控 mask 预测准确率而非逻辑错误率
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

from .scheduler import WarmupCosineScheduler
from ..models.pretrain import PretrainLoss, MaskedSyndromeModeling


@dataclass
class PretrainConfig:
    """预训练配置"""

    # ==================== 基本配置 ====================
    total_steps: int = 100000          # 总训练步数
    batch_size: int = 512              # 批大小
    eval_batch_size: int = 1024        # 验证批大小

    # ==================== 优化器配置 ====================
    learning_rate: float = 2e-4        # 学习率
    weight_decay: float = 0.01         # 权重衰减
    max_grad_norm: float = 1.0         # 梯度裁剪阈值

    # ==================== 学习率调度 ====================
    warmup_steps: int = 5000           # Warmup 步数
    min_lr_ratio: float = 0.1          # 最终学习率比例

    # ==================== Masking 配置 ====================
    mask_ratio: float = 0.15           # Masking 比例
    mask_strategy: str = "random"      # Masking 策略

    # ==================== 验证和保存 ====================
    eval_interval: int = 1000          # 验证间隔
    log_interval: int = 100            # 日志间隔
    save_interval: int = 5000          # 保存间隔
    early_stopping_patience: int = 20  # 早停耐心值

    # ==================== 损失函数 ====================
    use_spatial_consistency: bool = False
    use_temporal_consistency: bool = False
    consistency_weight: float = 0.1

    # ==================== 其他 ====================
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0

    # ==================== 性能优化 ====================
    use_amp: bool = True
    use_compile: bool = False


class PretrainTrainer:
    """预训练 Trainer

    管理预训练的完整流程：
    - 训练循环
    - Masking 应用
    - 验证评估
    - 检查点管理
    - 日志记录

    使用示例：
        ```python
        config = PretrainConfig(total_steps=100000, batch_size=512)
        model = PretrainDecoder(coord_system)
        dataset = PretrainDataset(distance=3, rounds=25)

        trainer = PretrainTrainer(model, dataset, config)
        history = trainer.train()
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: PretrainConfig,
        save_dir: Optional[str] = None,
        logger: Optional[Callable] = None,
    ):
        """初始化预训练 Trainer

        Args:
            model: PretrainDecoder 模型
            train_dataset: 预训练数据集
            val_dataset: 验证数据集
            config: 预训练配置
            save_dir: 检查点保存目录
            logger: 日志记录函数
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints/pretrain")
        self.logger = logger or self._default_logger

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        self.loss_fn = PretrainLoss(
            use_spatial_consistency=config.use_spatial_consistency,
            use_temporal_consistency=config.use_temporal_consistency,
            consistency_weight=config.consistency_weight,
        )

        self.masking = MaskedSyndromeModeling(mask_ratio=config.mask_ratio)

        self.use_amp = config.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mask_acc': [],
            'val_mask_acc': [],
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
        """格式化指标"""
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                if k == 'mask_accuracy':
                    parts.append(f"{k}: {v:.2%}")
                else:
                    parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def train(self) -> Dict:
        """执行完整预训练

        Returns:
            训练历史字典
        """
        self.logger({"message": f"开始预训练，设备: {self.device}"})
        self.logger({"message": f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}"})

        train_iter = iter(self.train_loader)
        start_time = time.time()

        if HAS_TQDM:
            pbar = tqdm(total=self.config.total_steps, desc="Pretraining", unit="step",
                       dynamic_ncols=True, leave=True)
        else:
            pbar = None

        for step in range(self.config.total_steps):
            self.global_step = step

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            train_metrics = self._train_step(batch)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{train_metrics['loss']:.4f}",
                    'mask_acc': f"{train_metrics['mask_accuracy']:.2%}",
                })

            if step % self.config.log_interval == 0:
                train_metrics['step'] = step
                train_metrics['lr'] = self.scheduler.get_last_lr()[0]
                train_metrics['elapsed'] = time.time() - start_time
                if pbar is not None:
                    tqdm.write(self._format_metrics(train_metrics))
                else:
                    self.logger(train_metrics)

            if step > 0 and step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                val_metrics['step'] = step

                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_mask_acc'].append(val_metrics['mask_accuracy'])

                val_msg = f"[Eval] step: {step} | val_loss: {val_metrics['loss']:.4f} | val_mask_acc: {val_metrics['mask_accuracy']:.2%}"
                if pbar is not None:
                    tqdm.write(val_msg)
                else:
                    self.logger({"validation": val_metrics})

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

                if self.patience_counter >= self.config.early_stopping_patience:
                    msg = f"[Early Stop] patience: {self.patience_counter}"
                    if pbar is not None:
                        tqdm.write(msg)
                    else:
                        self.logger({"message": msg})
                    break

            if step > 0 and step % self.config.save_interval == 0:
                self._save_checkpoint(f'step_{step}.pt')

        if pbar is not None:
            pbar.close()

        self._save_checkpoint('final.pt')
        self.logger({"message": "预训练完成"})

        return self.history

    def _train_step(self, batch: Dict) -> Dict:
        """执行单个训练步"""
        self.model.train()

        measurement = batch['measurement'].to(self.device, non_blocking=True)
        event = batch['event'].to(self.device, non_blocking=True)
        leakage = batch['leakage'].to(self.device, non_blocking=True)
        event_leakage = batch['event_leakage'].to(self.device, non_blocking=True)

        # 应用 masking
        masked_inputs, mask_indices = self.masking.mask_sequence(
            measurement, event, leakage, event_leakage
        )

        masked_measurement = masked_inputs['measurement']
        masked_event = masked_inputs['event']
        masked_leakage = masked_inputs.get('leakage', leakage)
        masked_event_leakage = masked_inputs.get('event_leakage', event_leakage)

        if self.use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                pred = self.model(
                    masked_measurement,
                    masked_event,
                    masked_leakage,
                    masked_event_leakage,
                )
                loss, metrics = self.loss_fn(pred, measurement, mask_indices)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred = self.model(
                masked_measurement,
                masked_event,
                masked_leakage,
                masked_event_leakage,
            )
            loss, metrics = self.loss_fn(pred, measurement, mask_indices)

            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            self.optimizer.step()

        self.scheduler.step()

        self.history['train_loss'].append(loss.item())
        self.history['train_mask_acc'].append(metrics['mask_accuracy'])
        self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """在验证集上评估"""
        self.model.eval()

        total_loss = 0.0
        total_mask_acc = 0.0
        total_samples = 0

        for batch in self.val_loader:
            measurement = batch['measurement'].to(self.device, non_blocking=True)
            event = batch['event'].to(self.device, non_blocking=True)
            leakage = batch['leakage'].to(self.device, non_blocking=True)
            event_leakage = batch['event_leakage'].to(self.device, non_blocking=True)

            masked_inputs, mask_indices = self.masking.mask_sequence(
                measurement, event, leakage, event_leakage
            )

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(
                        masked_inputs['measurement'],
                        masked_inputs['event'],
                        masked_inputs.get('leakage', leakage),
                        masked_inputs.get('event_leakage', event_leakage),
                    )
                    loss, metrics = self.loss_fn(pred, measurement, mask_indices)
            else:
                pred = self.model(
                    masked_inputs['measurement'],
                    masked_inputs['event'],
                    masked_inputs.get('leakage', leakage),
                    masked_inputs.get('event_leakage', event_leakage),
                )
                loss, metrics = self.loss_fn(pred, measurement, mask_indices)

            batch_size = measurement.size(0)
            total_loss += loss.item() * batch_size
            total_mask_acc += metrics['mask_accuracy'] * batch_size
            total_samples += batch_size

        return {
            'loss': total_loss / total_samples,
            'mask_accuracy': total_mask_acc / total_samples,
            'num_samples': total_samples,
        }

    def _save_checkpoint(self, filename: str):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        self.logger({"message": f"从 {path} 加载检查点，step: {self.global_step}"})
