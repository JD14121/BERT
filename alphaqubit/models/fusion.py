"""
融合模块 - Late Fusion机制

这个模块实现了AlphaQubit中的Late Fusion机制，
用于融合主路径（stabilizer RNN）和软路径（final data soft readout）。

为什么需要Late Fusion？
- 如果final data qubit的软读出过早进入RNN，模型可能"偷看"逻辑态
- 这会导致label leakage，模型学到的是trivial的解决方案
- Late Fusion将软读出延迟到最后阶段融合，避免泄露

Late Fusion架构：
┌─────────────────────────────────────────────────────────────┐
│  Main Path (from RNN):                                       │
│      H_T [B, n_stab, D]                                     │
│          │                                                   │
│          ▼  Scatter to 2D                                    │
│      [B, D, d+1, d+1]                                       │
│          │                                                   │
│          ▼  StabToDataConv (2×2 Conv)                       │
│      [B, D', d, d]                                          │
│          │                                                   │
│          ▼  Flatten                                          │
│      F_stab_to_data [B, n_data, D']                         │
│                                                              │
│  Soft Path (from Final Data):                               │
│      final_soft [B, n_data]                                 │
│          │                                                   │
│          ▼  SoftDataEmbedder                                │
│      F_soft [B, n_data, D']                                 │
│                                                              │
│  Fusion:                                                     │
│      F_final = Concat(F_stab_to_data, F_soft)               │
│          或                                                  │
│      F_final = F_stab_to_data + F_soft                      │
│          │                                                   │
│          ▼  ReadoutHead                                     │
│      logit [B, 1]                                           │
└─────────────────────────────────────────────────────────────┘

Stab-to-Data映射原理：
- 在rotated surface code中，每个data qubit被4个相邻stabilizer包围
- 使用2×2 Conv将4个stabilizer的特征融合为1个data qubit特征
- 这保持了物理上的对应关系
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..data.coordinates import CoordinateSystem
from .embeddings import SoftDataEmbedder


class StabToDataConv(nn.Module):
    """稳定子到数据比特的卷积映射

    使用2×2卷积将stabilizer特征映射到data qubit特征。

    物理意义：
    - Rotated surface code中，每个data qubit位于4个stabilizer的中心
    - 2×2 Conv正好捕捉这种几何关系
    - 输入: (d+1)×(d+1) stabilizer grid
    - 输出: d×d data qubit grid

    维度变化：
        [B, D, d+1, d+1] → Conv2d(kernel=2, stride=1) → [B, D', d, d]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coord_system: CoordinateSystem,
    ):
        """初始化Stab-to-Data卷积

        Args:
            in_channels: 输入通道数（stabilizer特征维度）
            out_channels: 输出通道数（data qubit特征维度）
            coord_system: 坐标系统对象
        """
        super().__init__()

        self.coord_system = coord_system
        self.distance = coord_system.distance
        self.n_data = coord_system.n_data

        # 2×2卷积：将4个相邻stabilizer映射到1个data qubit
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            padding=0,  # 无padding，输出尺寸会减1
        )

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, stab_features: Tensor) -> Tensor:
        """前向传播

        Args:
            stab_features: 稳定子特征 [B, n_stab, D]

        Returns:
            数据比特特征 [B, n_data, D']
        """
        B = stab_features.shape[0]

        # ==================== Scatter到2D网格 ====================
        # [B, n_stab, D] → [B, (d+1)², D]
        lattice_flat = self.coord_system.scatter(stab_features)

        # [B, (d+1)², D] → [B, D, d+1, d+1]
        lattice_2d = self.coord_system.to_2d(lattice_flat)

        # ==================== 2×2卷积 ====================
        # [B, D, d+1, d+1] → [B, D', d, d]
        data_2d = self.conv(lattice_2d)

        # ==================== Flatten到data qubit特征 ====================
        # [B, D', d, d] → [B, D', d²] → [B, d², D']
        D_out = data_2d.shape[1]
        data_flat = data_2d.view(B, D_out, -1).permute(0, 2, 1)

        # 归一化和激活
        data_flat = self.norm(data_flat)
        data_flat = self.activation(data_flat)

        return data_flat


class LateFusion(nn.Module):
    """Late Fusion模块

    论文设计原理：
    1. Stabilizer特征（从RNN）已经包含了threshold控制的信息（d² bits）
    2. Final data soft readout提供额外信息，但通过Late Fusion避免label leakage
    3. 在readout阶段融合，而不是在RNN阶段，防止模型"偷看"逻辑态

    信息流：
    - Main Path: Stabilizer (threshold控制) → RNN → StabToDataConv
    - Soft Path: Final data soft readout → SoftDataEmbedder
    - Fusion: 在readout阶段融合两路信息
    """

    def __init__(
        self,
        stab_dim: int,
        soft_dim: int,
        output_dim: int,
        coord_system: CoordinateSystem,
        fusion_type: Literal["concat", "add", "attention"] = "concat",
        dropout: float = 0.1,
    ):
        """初始化Late Fusion模块

        Args:
            stab_dim: 稳定子特征维度（RNN输出维度）
            soft_dim: 软读出嵌入维度
            output_dim: 融合后的输出维度
            coord_system: 坐标系统
            fusion_type: 融合类型 ("concat", "add", "attention")
            dropout: Dropout比率
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.coord_system = coord_system
        self.n_data = coord_system.n_data

        # ==================== Stab-to-Data映射 ====================
        self.stab_to_data = StabToDataConv(
            in_channels=stab_dim,
            out_channels=output_dim,
            coord_system=coord_system,
        )

        # ==================== 软读出嵌入 ====================
        grid_positions = (coord_system.distance + 1) ** 2
        self.soft_embedder = SoftDataEmbedder(
            embed_dim=soft_dim,
            n_positions=grid_positions,
            dropout=dropout,
        )

        # 注册数据比特的位置索引为buffer
        # 为n_data个数据比特创建0到n_data-1的索引
        data_position_idx = coord_system.data_grid_indices.clone().long()
        self.register_buffer('data_position_idx', data_position_idx)

        # ==================== 融合层 ====================
        if fusion_type == "concat":
            # Concat后需要投影到统一维度
            self.fusion_proj = nn.Linear(output_dim + soft_dim, output_dim)
        elif fusion_type == "add":
            # Add需要维度相同
            if soft_dim != output_dim:
                self.soft_proj = nn.Linear(soft_dim, output_dim)
            else:
                self.soft_proj = nn.Identity()
            self.fusion_proj = nn.Identity()
        elif fusion_type == "attention":
            # 注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            if soft_dim != output_dim:
                self.soft_proj = nn.Linear(soft_dim, output_dim)
            else:
                self.soft_proj = nn.Identity()
            self.fusion_proj = nn.Identity()
        else:
            raise ValueError(f"未知的融合类型: {fusion_type}")

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        stab_features: Tensor,
        final_soft: Tensor,
    ) -> Tensor:
        """前向传播

        Args:
            stab_features: 稳定子特征（RNN输出）[B, n_stab, D]
            final_soft: 最终数据比特软读出 [B, n_data]

        Returns:
            融合后的特征 [B, n_data, D']
        """
        # ==================== 处理主路径 ====================
        # [B, n_stab, D] → [B, n_data, D']
        stab_data_features = self.stab_to_data(stab_features)

        # ==================== 处理软路径 ====================
        # [B, n_data] → [B, n_data, D_soft]
        soft_features = self.soft_embedder(final_soft, self.data_position_idx)

        # ==================== 融合 ====================
        if self.fusion_type == "concat":
            # 拼接后投影
            fused = torch.cat([stab_data_features, soft_features], dim=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_type == "add":
            # 投影后相加
            soft_proj = self.soft_proj(soft_features)
            fused = stab_data_features + soft_proj

        elif self.fusion_type == "attention":
            # 使用注意力机制
            # stab_data_features作为query，soft_features作为key/value
            soft_proj = self.soft_proj(soft_features)
            fused, _ = self.attention(
                query=stab_data_features,
                key=soft_proj,
                value=soft_proj,
            )
            fused = fused + stab_data_features  # 残差连接

        # ==================== 输出处理 ====================
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class GlobalPooling(nn.Module):
    """全局池化模块

    将数据比特特征聚合为全局表示。

    支持的池化方式：
    - mean: 平均池化
    - max: 最大池化
    - attention: 注意力加权池化
    """

    def __init__(
        self,
        input_dim: int,
        pooling_type: Literal["mean", "max", "attention"] = "mean",
    ):
        """初始化全局池化

        Args:
            input_dim: 输入特征维度
            pooling_type: 池化类型
        """
        super().__init__()

        self.pooling_type = pooling_type

        if pooling_type == "attention":
            # 注意力池化：学习每个位置的权重
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.Tanh(),
                nn.Linear(input_dim // 4, 1),
            )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入特征 [B, N, D]

        Returns:
            池化后的特征 [B, D]
        """
        if self.pooling_type == "mean":
            return x.mean(dim=1)

        elif self.pooling_type == "max":
            return x.max(dim=1)[0]

        elif self.pooling_type == "attention":
            # 计算注意力权重
            weights = self.attention_weights(x)  # [B, N, 1]
            weights = torch.softmax(weights, dim=1)
            # 加权求和
            return (x * weights).sum(dim=1)

        else:
            raise ValueError(f"未知的池化类型: {self.pooling_type}")
