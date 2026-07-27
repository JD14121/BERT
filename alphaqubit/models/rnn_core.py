"""
RNN Core模块 - 论文中的循环核心处理

这个模块实现了AlphaQubit中的RNN Core，
包含了0.7缩放因子和SyndromeTransformer集成。

RNN Core架构（参考alphaqubit_design.md 4.5节）：
┌─────────────────────────────────────────────────────────────┐
│                        RNN Core                              │
│                                                              │
│  每个时间步 t:                                               │
│                                                              │
│      Decoder State_{t-1}     S_t (Syndrome Embedding)       │
│            │                        │                        │
│            └──────────┬─────────────┘                        │
│                       │                                      │
│                      (+)                                     │
│                       │                                      │
│                      (× 0.7)  ← 缩放因子 ≈ 1/√2             │
│                       │                                      │
│                       ▼                                      │
│             ┌─────────────────┐                             │
│             │SyndromeTransformer│ × N层 (串行)              │
│             └────────┬────────┘                             │
│                      │                                       │
│                      ▼                                       │
│              Decoder State_t                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

0.7缩放因子的原因：
- 假设 state 和 embed 都是零均值单位方差
- 相加后方差变为 2，标准差变为 √2
- 乘以 1/√2 ≈ 0.7 使方差回到 1，保持信号幅度稳定
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .transformer import SyndromeTransformer, SyndromeTransformerStack


class RNNCore(nn.Module):
    """RNN Core - 带0.7缩放因子的循环处理

    实现论文中的核心循环架构：
    1. 状态累加
    2. 0.7缩放
    3. SyndromeTransformer处理

    初始状态为全零。
    """

    def __init__(
        self,
        embed_dim: int,
        n_stab: int,
        n_heads: int = 4,
        num_transformer_layers: int = 3,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        scale_factor: float = 0.7,
    ):
        """初始化RNN Core

        Args:
            embed_dim: 嵌入维度
            n_stab: 稳定子数量
            n_heads: 注意力头数
            num_transformer_layers: SyndromeTransformer层数
            expansion_factor: GeGLU扩展因子
            num_conv_layers: 每层的卷积层数
            dropout: Dropout比率
            scale_factor: 缩放因子，默认0.7 ≈ 1/√2
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_stab = n_stab
        self.scale_factor = scale_factor

        # SyndromeTransformer堆叠
        self.transformer = SyndromeTransformerStack(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_stab=n_stab,
            num_layers=num_transformer_layers,
            expansion_factor=expansion_factor,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
        )

    def forward(
        self,
        syndrome_embeddings: Tensor,
        coord_system=None,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播

        Args:
            syndrome_embeddings: 综合征嵌入 [B, T, n_stab, D]
            coord_system: 坐标系统对象
            stab_positions: 稳定子位置

        Returns:
            最终状态 [B, n_stab, D]
        """
        B, T, n_stab, D = syndrome_embeddings.shape
        device = syndrome_embeddings.device

        # 初始化状态为零
        state = torch.zeros(B, n_stab, D, device=device, dtype=syndrome_embeddings.dtype)

        # 逐时间步处理
        for t in range(T):
            # 获取当前时间步的嵌入
            s_t = syndrome_embeddings[:, t, :, :]  # [B, n_stab, D]

            # 状态累加
            state = state + s_t

            # 0.7缩放因子
            state = state * self.scale_factor

            # 通过SyndromeTransformer处理
            state = self.transformer(
                state,
                coord_system=coord_system,
                stab_positions=stab_positions,
            )

        return state

    def forward_with_all_states(
        self,
        syndrome_embeddings: Tensor,
        coord_system=None,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播，返回所有时间步的状态

        Args:
            syndrome_embeddings: 综合征嵌入 [B, T, n_stab, D]
            coord_system: 坐标系统对象
            stab_positions: 稳定子位置

        Returns:
            所有状态 [B, T, n_stab, D]
        """
        B, T, n_stab, D = syndrome_embeddings.shape
        device = syndrome_embeddings.device

        # 收集所有状态
        all_states = []

        # 初始化状态
        state = torch.zeros(B, n_stab, D, device=device, dtype=syndrome_embeddings.dtype)

        for t in range(T):
            s_t = syndrome_embeddings[:, t, :, :]
            state = state + s_t
            state = state * self.scale_factor
            state = self.transformer(
                state,
                coord_system=coord_system,
                stab_positions=stab_positions,
            )
            all_states.append(state)

        # 堆叠所有状态
        all_states = torch.stack(all_states, dim=1)  # [B, T, n_stab, D]
        return all_states
