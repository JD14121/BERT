"""
SyndromeTransformer模块 - 论文核心组件

这个模块实现了AlphaQubit中的SyndromeTransformer层，
是论文的核心创新点之一。

SyndromeTransformer架构（参考alphaqubit_design.md 4.6节）：
┌─────────────────────────────────────────────────────────────┐
│                    SyndromeTransformer                       │
│                                                              │
│  Input [B, n_stab, D]                                       │
│      │                                                       │
│      ├───────────────┐ (残差)                               │
│      ▼               │                                       │
│  Self-Attention      │  ← 捕捉长程依赖 + Attention Bias     │
│  (+ Spatial Bias)    │                                       │
│      │               │                                       │
│     (+) ◄────────────┘                                       │
│      │                                                       │
│      ├───────────────┐ (残差)                               │
│      ▼               │                                       │
│  Gated Dense Block   │  ← GLU风格，筛选特征                 │
│  (GeGLU)             │                                       │
│      │               │                                       │
│     (+) ◄────────────┘                                       │
│      │                                                       │
│      ├───────────────┐ (残差)                               │
│      ▼               │                                       │
│  Scatter to 2D       │                                       │
│      ▼               │                                       │
│  Dilated Convs ×3    │  ← 捕捉局部空间特征                  │
│      ▼               │                                       │
│  Gather from 2D      │                                       │
│      │               │                                       │
│     (+) ◄────────────┘                                       │
│      │                                                       │
│      ▼                                                       │
│  Output [B, n_stab, D]                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialAttentionBias(nn.Module):
    """空间注意力偏置

    基于稳定子之间的空间距离生成注意力偏置。
    近距离的稳定子应该有更强的关注。

    最终偏置 = 基于距离的先验 + 可学习的修正项
                ↑                  ↑
            "合理的初始假设"    "学习特殊情况"
    
    论文的 Attention Bias 构造：
    1. 为每对 (i,j) 构建特征向量，包括：
    - stabilizer i 的空间坐标
    - stabilizer j 的空间坐标  
    - i 到 j 的有符号偏移
    - Manhattan距离
    - 类型是否相同（X-X, Z-Z, X-Z）

    2. 这些特征通过一个 ResNet 生成 48 维 embedding

    3. 每层 Transformer 将 48 维投影到 n_heads 维

    4. 还有动态特征：当前轮和上一轮的 event 相关性

    公式：
        Attention(Q, K, V) = softmax(QK^T / √d + B) V
        其中 B 是基于空间距离的偏置矩阵
    """

    def __init__(
        self,
        n_stab: int,
        n_heads: int,
        max_distance: float = 10.0,
    ):
        """初始化空间注意力偏置

        Args:
            n_stab: 稳定子数量
            n_heads: 注意力头数
            max_distance: 最大距离（用于归一化）
        """
        super().__init__()

        self.n_stab = n_stab
        self.n_heads = n_heads
        self.max_distance = max_distance

        # 可学习的距离到偏置的映射
        # 使用多项式基函数
        self.distance_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, n_heads),
        )

        # 可选：直接学习的偏置矩阵（当位置固定时）
        self.learned_bias = nn.Parameter(
            torch.zeros(n_heads, n_stab, n_stab)
        )

    def forward(
        self,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """计算注意力偏置

        Args:
            stab_positions: 稳定子位置 [n_stab, 2]
                           如果为None，使用学习的偏置

        Returns:
            注意力偏置 [n_heads, n_stab, n_stab]
        """
        if stab_positions is None:
            # 使用学习的偏置
            return self.learned_bias

        # 计算成对距离
        # [n_stab, 2] → [n_stab, 1, 2] - [1, n_stab, 2] → [n_stab, n_stab]
        diff = stab_positions.unsqueeze(1) - stab_positions.unsqueeze(0)
        distances = torch.norm(diff, dim=-1)

        # 归一化距离
        norm_distances = distances / self.max_distance
        norm_distances = norm_distances.unsqueeze(-1)  # [n_stab, n_stab, 1]

        # 通过MLP生成偏置
        bias = self.distance_embed(norm_distances)  # [n_stab, n_stab, n_heads]
        bias = bias.permute(2, 0, 1)  # [n_heads, n_stab, n_stab]

        # 加上学习的偏置
        bias = bias + self.learned_bias

        return bias


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（带空间偏置）

    实现论文中的Self-Attention with Bias。

    公式：
        Attention(Q, K, V) = softmax(QK^T / √d + B) V
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_stab: int,
        dropout: float = 0.1,
        use_spatial_bias: bool = True,
    ):
        """初始化多头自注意力

        Args:
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            n_stab: 稳定子数量
            dropout: Dropout比率
            use_spatial_bias: 是否使用空间偏置
        """
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim必须能被n_heads整除"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 空间偏置
        self.use_spatial_bias = use_spatial_bias
        if use_spatial_bias:
            self.spatial_bias = SpatialAttentionBias(n_stab, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, n_stab, D]
            stab_positions: 稳定子位置 [n_stab, 2]

        Returns:
            输出 [B, n_stab, D]
        """
        B, n_stab, D = x.shape

        # Pre-LN
        x_norm = self.layer_norm(x)

        # 计算 Q, K, V
        q = self.q_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim)

        # 转置为 [B, n_heads, n_stab, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 计算注意力分数
        # [B, n_heads, n_stab, n_stab]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 添加空间偏置
        if self.use_spatial_bias:
            bias = self.spatial_bias(stab_positions)  # [n_heads, n_stab, n_stab]
            attn_scores = attn_scores + bias.unsqueeze(0)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        # [B, n_heads, n_stab, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # 重塑和投影
        # [B, n_stab, n_heads, head_dim] → [B, n_stab, D]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, n_stab, D)
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # 残差连接
        return x + output


class GatedDenseBlock(nn.Module):
    """门控稠密块 (GeGLU)

    实现论文中的Gated Dense Block，使用GeGLU激活。

    公式：
        Output = (x @ W1) ⊙ GELU(x @ W_gate) @ W2
    
    假设某个特征维度 i：
    - 如果这个维度包含"噪声"（与错误无关的随机信号）
        → gate_i 学习到趋近 0
        → value_i * gate_i ≈ 0（噪声被抑制）

    - 如果这个维度包含"关键错误信号"
        → gate_i 学习到趋近 1
        → value_i * gate_i ≈ value_i（信号被保留）

    门控机制可以过滤噪声信号，放大关键错误特征。
    """

    def __init__(
        self,
        embed_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        """初始化门控稠密块

        Args:
            embed_dim: 嵌入维度
            expansion_factor: 扩展因子
            dropout: Dropout比率
        """
        super().__init__()

        hidden_dim = embed_dim * expansion_factor

        self.layer_norm = nn.LayerNorm(embed_dim)

        # 两个投影：一个用于值，一个用于门
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w_gate = nn.Linear(embed_dim, hidden_dim)

        # 输出投影
        self.w2 = nn.Linear(hidden_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # 使用小值初始化，允许梯度流动
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, n_stab, D]

        Returns:
            输出 [B, n_stab, D]
        """
        # Pre-LN
        x_norm = self.layer_norm(x)

        # GeGLU: value * gate
        value = self.w1(x_norm)
        gate = F.gelu(self.w_gate(x_norm))
        hidden = value * gate

        # 输出投影
        output = self.w2(hidden)
        output = self.dropout(output)

        # 残差连接
        return x + output


class DilatedConvBlock(nn.Module):
    """空洞卷积块

    实现论文中的Dilated 2D Convolutions，用于时序RNN中的stabilizer特征交互（捕捉错误关联）。

    流程：
    1. Scatter: [B, n_stab, D] → [B, D, H, W]
    2. Dilated Conv2D × N
    3. Gather: [B, D, H, W] → [B, n_stab, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_conv_layers: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        """初始化空洞卷积块

        Args:
            embed_dim: 嵌入维度
            num_conv_layers: 卷积层数
            dilations: 各层的空洞率，默认[1, 2, 4]
            dropout: Dropout比率
        """
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4][:num_conv_layers]

        self.layer_norm = nn.LayerNorm(embed_dim)

        # 卷积层
        conv_layers = []
        for i, dilation in enumerate(dilations):
            # 3×3空洞卷积，保持尺寸
            padding = dilation  # same padding
            conv_layers.append(
                nn.Conv2d(
                    embed_dim, embed_dim,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
            conv_layers.append(nn.GELU())
            if i < len(dilations) - 1:
                conv_layers.append(nn.Dropout2d(dropout))

        self.convs = nn.Sequential(*conv_layers)

        # 输出投影（小值初始化，允许梯度流动）
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        scatter_fn,
        gather_fn,
        reset_padding_fn=None,
        pad_vec: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, n_stab, D]
            scatter_fn: Scatter函数
            gather_fn: Gather函数
            reset_padding_fn: 重置padding函数
            pad_vec: Padding向量

        Returns:
            输出 [B, n_stab, D]
        """
        B = x.shape[0]

        # Pre-LN
        x_norm = self.layer_norm(x)

        # Scatter到2D网格
        lattice_flat = scatter_fn(x_norm)  # [B, grid_size², D]

        # 重置padding
        if reset_padding_fn is not None and pad_vec is not None:
            lattice_flat = reset_padding_fn(lattice_flat, pad_vec)

        # 获取grid_size
        grid_size = int(math.sqrt(lattice_flat.shape[1]))
        D = lattice_flat.shape[2]

        # 转为2D: [B, grid_size², D] → [B, D, grid_size, grid_size]
        lattice_2d = lattice_flat.view(B, grid_size, grid_size, D)
        lattice_2d = lattice_2d.permute(0, 3, 1, 2)

        # 应用卷积
        conv_out = self.convs(lattice_2d)

        # 转回flat: [B, D, H, W] → [B, H*W, D]
        conv_out = conv_out.permute(0, 2, 3, 1).reshape(B, -1, D)

        # Gather回稳定子位置
        output = gather_fn(conv_out)  # [B, n_stab, D]

        # 输出投影
        output = self.out_proj(output)
        output = self.dropout(output)

        # 残差连接
        return x + output


class SyndromeTransformer(nn.Module):
    """SyndromeTransformer层

    论文的核心组件，结合了：
    1. Self-Attention with Spatial Bias
    2. Gated Dense Block (GeGLU)
    3. Dilated 2D Convolutions

    每个子层都有Pre-LN和残差连接。
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_stab: int,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_spatial_bias: bool = True,
    ):
        """初始化SyndromeTransformer

        Args:
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            n_stab: 稳定子数量
            expansion_factor: GeGLU扩展因子
            num_conv_layers: 卷积层数
            dilations: 卷积空洞率
            dropout: Dropout比率
            use_spatial_bias: 是否使用空间注意力偏置
        """
        super().__init__()

        self.embed_dim = embed_dim

        # 1. Self-Attention with Spatial Bias
        self.self_attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_stab=n_stab,
            dropout=dropout,
            use_spatial_bias=use_spatial_bias,
        )

        # 2. Gated Dense Block
        self.gated_dense = GatedDenseBlock(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # 3. Dilated Convolutions
        self.dilated_conv = DilatedConvBlock(
            embed_dim=embed_dim,
            num_conv_layers=num_conv_layers,
            dilations=dilations,
            dropout=dropout,
        )

        # 可学习的padding向量
        self.pad_vec = nn.Parameter(torch.zeros(embed_dim))

    def forward(
        self,
        x: Tensor,
        coord_system=None,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, n_stab, D]
            coord_system: 坐标系统对象（用于scatter/gather）
            stab_positions: 稳定子位置 [n_stab, 2]

        Returns:
            输出 [B, n_stab, D]
        """
        # 1. Self-Attention
        x = self.self_attention(x, stab_positions)

        # 2. Gated Dense
        x = self.gated_dense(x)

        # 3. Dilated Convolutions (需要坐标系统)
        if coord_system is not None:
            x = self.dilated_conv(
                x,
                scatter_fn=coord_system.scatter,
                gather_fn=coord_system.gather,
                reset_padding_fn=coord_system.reset_padding,
                pad_vec=self.pad_vec,
            )

        return x


class SyndromeTransformerStack(nn.Module):
    """SyndromeTransformer堆叠

    多层SyndromeTransformer串联。
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_stab: int,
        num_layers: int = 3,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
    ):
        """初始化SyndromeTransformer堆叠

        Args:
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            n_stab: 稳定子数量
            num_layers: Transformer层数
            expansion_factor: GeGLU扩展因子
            num_conv_layers: 每层的卷积层数
            dropout: Dropout比率
        """
        super().__init__()

        self.layers = nn.ModuleList([
            SyndromeTransformer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_stab=n_stab,
                expansion_factor=expansion_factor,
                num_conv_layers=num_conv_layers,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        coord_system=None,
        stab_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, n_stab, D]
            coord_system: 坐标系统对象
            stab_positions: 稳定子位置

        Returns:
            输出 [B, n_stab, D]
        """
        for layer in self.layers:
            x = layer(x, coord_system, stab_positions)

        x = self.final_norm(x)
        return x
