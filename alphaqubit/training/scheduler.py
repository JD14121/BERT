"""
学习率调度器模块

这个模块实现了训练过程中的各种调度策略：

1. WarmupCosineScheduler: 学习率调度
   - Warmup阶段：从0线性增加到base_lr
   - Cosine Decay阶段：余弦衰减到min_lr

2. RoundsCurriculum: 轮数课程学习
   - 逐步增加训练时的rounds数
   - 帮助模型从简单到复杂逐步学习

学习率曲线示意：
    lr
    │
    │         ╭──────╮
    │        ╱        ╲
    │       ╱          ╲
    │      ╱            ╲
    │     ╱              ╲
    │    ╱                ╲___
    │   ╱
    │  ╱
    │_╱
    └─────────────────────────────► steps
      warmup    cosine decay

Rounds课程学习：
    rounds
    │
    │                    ┌─────── 12
    │           ┌───────┘
    │    ┌──────┘          6
    │────┘
    │                 3
    └─────────────────────────────► steps
      0-5k    5k-15k    15k+
"""

import math
from typing import Optional, List, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Warmup + Cosine Decay学习率调度器

    学习率调度分为两个阶段：
    1. Warmup（预热）：从0线性增加到base_lr
    2. Cosine Decay（余弦衰减）：从base_lr余弦衰减到min_lr

    为什么需要Warmup？
    - 训练初期模型参数随机，梯度方向不稳定
    - 直接使用大学习率可能导致训练不稳定
    - 逐步增加学习率让模型适应

    为什么用Cosine Decay？
    - 比阶梯衰减更平滑
    - 在训练后期提供更精细的调整
    - 实践中效果通常更好
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        """初始化调度器

        Args:
            optimizer: PyTorch优化器
            warmup_steps: Warmup步数（推荐1000）
            total_steps: 总训练步数
            min_lr_ratio: 最终学习率与初始学习率的比值
                         min_lr = base_lr * min_lr_ratio
            last_epoch: 上次epoch（用于恢复训练）
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """计算当前步的学习率

        Returns:
            每个参数组的学习率列表
        """
        step = self.last_epoch

        if step < self.warmup_steps:
            # ==================== Warmup阶段 ====================
            # 从10%的base_lr线性增加到100%的base_lr
            # 这样初始学习率就是 0.1 * base_lr，足够大能学习
            min_factor = 0.1  # 初始学习率为base_lr的10%
            warmup_factor = min_factor + (1.0 - min_factor) * (step / self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # ==================== Cosine Decay阶段 ====================
            # 从base_lr余弦衰减到min_lr
            decay_steps = self.total_steps - self.warmup_steps
            current_step = step - self.warmup_steps

            # 边界保护：避免 total_steps <= warmup_steps 时 decay_steps=0 导致除零
            if decay_steps <= 0:
                return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

            current_step = min(current_step, decay_steps)

            # 余弦衰减公式
            # lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t / T))
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))

            lrs = []
            for base_lr in self.base_lrs:
                min_lr = base_lr * self.min_lr_ratio
                lr = min_lr + (base_lr - min_lr) * cosine_factor
                lrs.append(lr)

            return lrs


class RoundsCurriculum:
    """轮数课程学习调度器

    逐步增加训练时的rounds数，帮助模型从简单到复杂学习。

    课程设计原理：
    - 少轮数时，错误模式更简单
    - 多轮数时，错误可能累积和传播
    - 先学简单模式，再学复杂模式

    默认课程：
    | 步数范围 | Rounds |
    |----------|--------|
    | 0 - 5k   | 3      |
    | 5k - 15k | 6      |
    | 15k+     | 12     |
    """

    def __init__(
        self,
        schedule: Optional[List[Tuple[int, int]]] = None,
        final_rounds: int = 12,
    ):
        """初始化轮数课程

        Args:
            schedule: 课程表，格式为 [(step_threshold, rounds), ...]
                     默认为 [(0, 3), (5000, 6), (15000, 12)]
            final_rounds: 最终轮数
        """
        if schedule is None:
            # 默认课程表
            self.schedule = [
                (0, 3),       # 步数0开始，使用3轮
                (5000, 6),    # 步数5000开始，使用6轮
                (15000, 12),  # 步数15000开始，使用12轮
            ]
        else:
            self.schedule = sorted(schedule, key=lambda x: x[0])

        self.final_rounds = final_rounds

    def get_rounds(self, step: int) -> int:
        """获取当前步数应使用的轮数

        Args:
            step: 当前训练步数

        Returns:
            当前应使用的轮数
        """
        rounds = self.schedule[0][1]  # 默认第一个值

        for threshold, r in self.schedule:
            if step >= threshold:
                rounds = r
            else:
                break

        return rounds

    def __repr__(self) -> str:
        """返回课程表的字符串表示"""
        lines = ["RoundsCurriculum:"]
        for threshold, rounds in self.schedule:
            lines.append(f"  step >= {threshold}: rounds = {rounds}")
        return "\n".join(lines)


class NoiseCurriculum:
    """噪声课程学习调度器

    逐步增加噪声强度，类似于轮数课程。

    设计原理：
    - 低噪声时，错误更稀疏，模式更清晰
    - 高噪声时，错误更密集，模式更复杂
    - 从低噪声开始训练可能帮助模型更好地学习

    论文参考 (alphaqubit_plan.md 4.3节):
    - 推荐范围：p ≈ 0.005 ~ 0.01
    - 可选增强：从 0.6p 线性爬到 p
    """

    def __init__(
        self,
        base_p: float = 0.005,
        start_ratio: float = 0.6,
        warmup_steps: int = 10000,
        schedule_type: str = "linear",
    ):
        """初始化噪声课程

        Args:
            base_p: 目标噪声强度
            start_ratio: 起始噪声与目标噪声的比值
            warmup_steps: 从start到base的过渡步数
            schedule_type: 调度类型 ("linear", "cosine", "step")
        """
        self.base_p = base_p
        self.start_p = base_p * start_ratio
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

    def get_p(self, step: int) -> float:
        """获取当前步数应使用的噪声强度

        Args:
            step: 当前训练步数

        Returns:
            当前噪声强度p
        """
        if step >= self.warmup_steps:
            return self.base_p

        ratio = step / self.warmup_steps

        if self.schedule_type == "linear":
            # 线性插值
            return self.start_p + (self.base_p - self.start_p) * ratio
        elif self.schedule_type == "cosine":
            # 余弦插值（更平滑的过渡）
            cosine_ratio = 0.5 * (1 - math.cos(math.pi * ratio))
            return self.start_p + (self.base_p - self.start_p) * cosine_ratio
        elif self.schedule_type == "step":
            # 阶梯式（3个阶段）
            if ratio < 0.33:
                return self.start_p
            elif ratio < 0.66:
                return (self.start_p + self.base_p) / 2
            else:
                return self.base_p
        else:
            return self.base_p

    def __repr__(self) -> str:
        return (f"NoiseCurriculum(base_p={self.base_p}, "
                f"start_p={self.start_p}, warmup_steps={self.warmup_steps})")


class LinearWarmupScheduler(_LRScheduler):
    """简单的线性Warmup调度器

    只有warmup阶段，之后保持恒定学习率。
    适合与其他调度器组合使用。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """初始化

        Args:
            optimizer: 优化器
            warmup_steps: Warmup步数
            last_epoch: 上次epoch
        """
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        step = self.last_epoch

        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class CurriculumManager:
    """综合课程学习管理器

    管理所有课程学习策略：
    - 轮数课程 (Rounds Curriculum)
    - 噪声课程 (Noise Curriculum)
    - 可扩展支持其他课程

    使用示例：
        curriculum = CurriculumManager(
            rounds_schedule=[(0, 3), (5000, 6), (15000, 12)],
            noise_base_p=0.005,
            noise_start_ratio=0.6,
            noise_warmup_steps=10000,
        )

        for step in range(total_steps):
            rounds = curriculum.get_rounds(step)
            p = curriculum.get_noise(step)
            # 使用rounds和p生成数据
    """

    def __init__(
        self,
        # 轮数课程参数
        use_rounds_curriculum: bool = True,
        rounds_schedule: Optional[List[Tuple[int, int]]] = None,
        final_rounds: int = 12,
        # 噪声课程参数
        use_noise_curriculum: bool = False,
        noise_base_p: float = 0.005,
        noise_start_ratio: float = 0.6,
        noise_warmup_steps: int = 10000,
        noise_schedule_type: str = "linear",
    ):
        """初始化课程管理器

        Args:
            use_rounds_curriculum: 是否使用轮数课程
            rounds_schedule: 轮数课程表
            final_rounds: 最终轮数
            use_noise_curriculum: 是否使用噪声课程
            noise_base_p: 目标噪声强度
            noise_start_ratio: 起始噪声比例
            noise_warmup_steps: 噪声warmup步数
            noise_schedule_type: 噪声调度类型
        """
        # 轮数课程
        self.use_rounds_curriculum = use_rounds_curriculum
        if use_rounds_curriculum:
            self.rounds_curriculum = RoundsCurriculum(
                schedule=rounds_schedule,
                final_rounds=final_rounds,
            )
        else:
            self.rounds_curriculum = None
            self.fixed_rounds = final_rounds

        # 噪声课程
        self.use_noise_curriculum = use_noise_curriculum
        if use_noise_curriculum:
            self.noise_curriculum = NoiseCurriculum(
                base_p=noise_base_p,
                start_ratio=noise_start_ratio,
                warmup_steps=noise_warmup_steps,
                schedule_type=noise_schedule_type,
            )
        else:
            self.noise_curriculum = None
            self.fixed_p = noise_base_p

    def get_rounds(self, step: int) -> int:
        """获取当前轮数"""
        if self.use_rounds_curriculum and self.rounds_curriculum is not None:
            return self.rounds_curriculum.get_rounds(step)
        return self.fixed_rounds

    def get_noise(self, step: int) -> float:
        """获取当前噪声强度"""
        if self.use_noise_curriculum and self.noise_curriculum is not None:
            return self.noise_curriculum.get_p(step)
        return self.fixed_p

    def get_config(self, step: int) -> dict:
        """获取当前步的完整配置

        Args:
            step: 当前训练步数

        Returns:
            包含rounds和p的配置字典
        """
        return {
            "rounds": self.get_rounds(step),
            "p": self.get_noise(step),
            "step": step,
        }

    def __repr__(self) -> str:
        lines = ["CurriculumManager:"]
        if self.use_rounds_curriculum:
            lines.append(f"  Rounds: {self.rounds_curriculum}")
        else:
            lines.append(f"  Rounds: fixed at {self.fixed_rounds}")
        if self.use_noise_curriculum:
            lines.append(f"  Noise: {self.noise_curriculum}")
        else:
            lines.append(f"  Noise: fixed at {self.fixed_p}")
        return "\n".join(lines)
