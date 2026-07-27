"""
训练模块 - AlphaQubit模型训练工具

这个模块提供完整的训练工具链：

组件：
- Trainer: 训练器，管理完整的训练循环
- LRScheduler: 学习率调度器（warmup + cosine decay）
- LossFunctions: 损失函数（BCE + 可选的辅助损失）

训练流程：
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│                                                              │
│  for epoch in epochs:                                        │
│      for batch in train_loader:                              │
│          │                                                   │
│          ▼                                                   │
│      ┌─────────────────────┐                                │
│      │   Forward Pass      │  model(events, final_soft)     │
│      └──────────┬──────────┘                                │
│                 │                                            │
│                 ▼                                            │
│      ┌─────────────────────┐                                │
│      │   Compute Loss      │  BCEWithLogitsLoss             │
│      └──────────┬──────────┘                                │
│                 │                                            │
│                 ▼                                            │
│      ┌─────────────────────┐                                │
│      │   Backward Pass     │  loss.backward()               │
│      └──────────┬──────────┘                                │
│                 │                                            │
│                 ▼                                            │
│      ┌─────────────────────┐                                │
│      │  Gradient Clipping  │  clip_grad_norm_(max=1.0)      │
│      └──────────┬──────────┘                                │
│                 │                                            │
│                 ▼                                            │
│      ┌─────────────────────┐                                │
│      │   Optimizer Step    │  AdamW                          │
│      └──────────┬──────────┘                                │
│                 │                                            │
│                 ▼                                            │
│      ┌─────────────────────┐                                │
│      │   LR Scheduler      │  warmup + cosine decay          │
│      └─────────────────────┘                                │
│                                                              │
│      if step % eval_interval == 0:                          │
│          evaluate(dev_loader)                                │
│          save_checkpoint_if_best()                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

关键训练技巧：
1. Gradient Clipping: RNN必须，防止梯度爆炸
2. Warmup: 前1000步逐步增加学习率
3. Cosine Decay: 平滑降低学习率
4. Early Stopping: 基于验证集BCE
"""

from .trainer import Trainer, TrainingConfig, train_decoder
from .scheduler import (
    WarmupCosineScheduler,
    RoundsCurriculum,
    NoiseCurriculum,
    LinearWarmupScheduler,
    CurriculumManager,
)
from .losses import DecoderLoss

__all__ = [
    # 训练器
    "Trainer",
    "TrainingConfig",
    "train_decoder",
    # 学习率调度
    "WarmupCosineScheduler",
    "LinearWarmupScheduler",
    # 课程学习
    "RoundsCurriculum",
    "NoiseCurriculum",
    "CurriculumManager",
    # 损失函数
    "DecoderLoss",
]
