"""
卷积块模块 - Scatter-Conv-Gather操作

这个模块实现了AlphaQubit中的空间卷积处理。

为什么需要卷积？
- 量子错误具有空间局部性
- 相邻稳定子的错误可能相关
- 卷积可以捕捉这种空间结构

Scatter-Conv-Gather流程：
┌─────────────────────────────────────────────────────────────┐
│   Stabilizer states: [B, n_stab, D]                         │
│           │                                                  │
│           ▼  Scatter（分散到网格）                           │
│   Lattice flat: [B, (d+1)², D]                              │
│           │                                                  │
│           ▼  Reshape                                         │
│   Lattice 2D: [B, D, d+1, d+1]                              │
│           │                                                  │
│           ▼  Conv2d（空间卷积）                              │
│   Lattice 2D: [B, D, d+1, d+1]                              │
│           │                                                  │
│           ▼  Flatten                                         │
│   Lattice flat: [B, (d+1)², D]                              │
│           │                                                  │
│           ▼  Reset Padding（重置空位）                       │
│   Lattice flat: [B, (d+1)², D]                              │
│           │                                                  │
│           ▼  Gather（收集稳定子位置）                        │
│   Stabilizer states: [B, n_stab, D]                         │
└─────────────────────────────────────────────────────────────┘

关键约束：
1. Padding Reset: 每次卷积后必须重置空位，防止卷积污染
2. Gather精确性: Gather必须是Scatter的逆操作
3. 边界处理: 使用适当的padding mode处理边界
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.coordinates import CoordinateSystem


class ConvBlock(nn.Module):
    """2D卷积块

    包含卷积、归一化、激活函数和可选的残差连接。

    架构：
    input → Conv2d → BatchNorm/LayerNorm → Activation → [+ residual] → output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        activation: str = "gelu",
    ):
        """初始化卷积块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            use_batch_norm: 是否使用BatchNorm
            use_residual: 是否使用残差连接
            activation: 激活函数类型 ("gelu", "relu", "silu")
        """
        super().__init__()

        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)

        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm,  # 有BatchNorm时不需要bias
        )

        # 归一化层
        if use_batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C', H', W']
        """
        identity = x

        out = self.conv(x)
        out = self.norm(out)

        if self.use_residual:
            out = out + identity

        out = self.activation(out)

        return out


class ScatterConvGather(nn.Module):
    """Scatter-Conv-Gather模块

    完整的空间卷积处理流程：
    1. Scatter: 将稀疏稳定子状态分散到稠密网格
    2. Conv: 在2D网格上进行卷积
    3. Reset Padding: 重置空位的值
    4. Gather: 收集稳定子位置的值

    这个模块封装了坐标系统，处理所有的形状变换。
    """

    def __init__(
        self,
        coord_system: CoordinateSystem,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        use_learned_padding: bool = True,
        dropout: float = 0.1,
    ):
        """初始化Scatter-Conv-Gather模块

        Args:
            coord_system: 坐标系统对象
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_conv_layers: 卷积层数量
            kernel_size: 卷积核大小
            use_learned_padding: 是否使用可学习的padding向量
            dropout: Dropout比率
        """
        super().__init__()

        self.coord_system = coord_system
        self.grid_size = coord_system.grid_size
        self.n_stab = coord_system.n_stab

        # ==================== Padding向量 ====================
        if use_learned_padding:
            # 可学习的padding向量
            self.padding_vec = nn.Parameter(torch.zeros(out_channels))
        else:
            # 使用固定的零padding
            self.register_buffer("padding_vec", torch.zeros(out_channels))

        # ==================== 卷积层 ====================
        conv_layers = []
        for i in range(num_conv_layers):
            layer_in = in_channels if i == 0 else out_channels
            conv_layers.append(
                ConvBlock(
                    in_channels=layer_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                    use_residual=(i > 0),  # 第一层后才用残差
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)

        # ==================== 输出处理 ====================
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 稳定子状态 [B, n_stab, D]

        Returns:
            处理后的稳定子状态 [B, n_stab, D']
        """
        B, n_stab, D = x.shape
        device = x.device

        # ==================== Scatter: 稀疏 → 稠密 ====================
        # [B, n_stab, D] → [B, (d+1)², D]
        lattice_flat = self.coord_system.scatter(x)

        # ==================== 重塑为2D网格 ====================
        # [B, (d+1)², D] → [B, D, d+1, d+1]
        lattice_2d = self.coord_system.to_2d(lattice_flat)

        # ==================== 卷积处理 ====================
        for i, conv in enumerate(self.conv_layers):
            lattice_2d = conv(lattice_2d)

            # 每层卷积后重置padding（防止卷积污染空位）
            if i < len(self.conv_layers) - 1:
                # 转换回flat格式以重置padding
                lattice_flat = self.coord_system.from_2d(lattice_2d)
                lattice_flat = self._reset_padding(lattice_flat, device)
                lattice_2d = self.coord_system.to_2d(lattice_flat)

        # ==================== 最终padding重置 ====================
        lattice_flat = self.coord_system.from_2d(lattice_2d)
        lattice_flat = self._reset_padding(lattice_flat, device)

        # ==================== Gather: 稠密 → 稀疏 ====================
        # [B, (d+1)², D] → [B, n_stab, D]
        output = self.coord_system.gather(lattice_flat)

        output = self.dropout(output)

        return output

    def _reset_padding(self, lattice_flat: Tensor, device: torch.device) -> Tensor:
        """重置padding位置的值

        Args:
            lattice_flat: [B, (d+1)², D]
            device: 设备

        Returns:
            重置后的lattice
        """
        # 获取grid mask
        grid_mask = self.coord_system.grid_mask.to(device)
        padding_mask = ~grid_mask  # 空位mask

        # 扩展维度以便广播
        # padding_mask: [(d+1)²] → [1, (d+1)², 1]
        padding_mask = padding_mask.unsqueeze(0).unsqueeze(-1)

        # padding_vec: [D] → [1, 1, D]
        pad_vec = self.padding_vec.unsqueeze(0).unsqueeze(0)

        # 使用where操作：空位用padding_vec，其他保持原值
        output = torch.where(
            padding_mask.expand_as(lattice_flat),
            pad_vec.expand_as(lattice_flat),
            lattice_flat,
        )

        return output


class MultiScaleConv(nn.Module):
    """多尺度卷积模块

    使用不同大小的卷积核捕捉不同尺度的空间特征。
    类似于Inception模块的思想。

    架构：
              input
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
      Conv1   Conv3   Conv5
        │       │       │
        └───────┼───────┘
                │
              concat
                │
              output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (1, 3, 5),
    ):
        """初始化多尺度卷积

        Args:
            in_channels: 输入通道数
            out_channels: 每个分支的输出通道数
            kernel_sizes: 卷积核大小元组
        """
        super().__init__()

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
            )

        # 融合层：将所有分支concat后降维
        self.fusion = nn.Conv2d(
            out_channels * len(kernel_sizes),
            out_channels,
            kernel_size=1,
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, C, H, W]

        Returns:
            输出 [B, C', H, W]
        """
        # 并行处理各分支
        branch_outputs = [branch(x) for branch in self.branches]

        # Concat所有分支
        concat = torch.cat(branch_outputs, dim=1)

        # 融合
        output = self.fusion(concat)
        output = self.norm(output)
        output = self.activation(output)

        return output
