"""
读出头模块 - 输出最终预测

这个模块实现了AlphaQubit的输出层，将融合后的特征转换为逻辑错误预测。

论文中的Readout Network架构（alphaqubit_design.md 4.7节）：
┌─────────────────────────────────────────────────────────────────┐
│                    Deep Readout Network                          │
│                                                                  │
│  Fused Features (batch, n_data, d_model)                        │
│          │                                                       │
│          ▼  Project to readout_dim                               │
│  (batch, n_data, d_readout)                                     │
│          │                                                       │
│          ▼  Mean Pool over data qubits                           │
│  (batch, d_readout)                                             │
│          │                                                       │
│   ┌──────┴──────┐                                               │
│   │             │                                                │
│   ▼             ▼                                                │
│ Pooled      Cycle Embed                                          │
│ Feature     (轮次编码)                                           │
│   │             │                                                │
│   └──────┬──────┘                                               │
│          │ (+)                                                   │
│          ▼                                                       │
│  Deep ResNet (16层)                                              │
│          │                                                       │
│          ▼  Linear + Sigmoid                                     │
│  P(Logical Error)                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

输出解释：
- logit: 未经sigmoid的原始输出
- 训练时使用BCEWithLogitsLoss（内含sigmoid）
- 推理时：P(error) = sigmoid(logit)
- 预测：ŷ = 1 if sigmoid(logit) > 0.5 else 0

注：LateFusion已经完成了StabToDataConv，所以DeepResNetReadout接收的是
     融合后的data qubit特征[B, n_data, D]，而不是stabilizer特征。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResNetBlock(nn.Module):
    """Pre-LN ResNet块

    论文使用的残差块设计（alphaqubit_design.md 4.4节）：
    - Pre-LayerNorm（归一化在残差分支前）
    - 等宽设计（不使用瓶颈）
    - ReLU激活函数（论文明确要求，不是GELU）
    - 零初始化输出层

    公式: y = x + Linear2(Dropout(ReLU(Dropout(Linear1(LayerNorm(x))))))
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
    ):
        """初始化ResNet块

        Args:
            dim: 特征维度
            dropout: Dropout比率
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 使用小值初始化第二个线性层，保持残差分支输出较小但非零
        nn.init.normal_(self.linear2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入 [B, D] 或 [B, N, D]

        Returns:
            输出 [B, D] 或 [B, N, D]
        """
        residual = x

        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.relu(x)  # 论文使用ReLU，不是GELU
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return residual + x


class CycleEmbedding(nn.Module):
    """轮次编码（v2: 正弦位置编码，可外推到任意 n_rounds）。

    旧版用 nn.Embedding(max_rounds) 查找 + n>=50 MLP 回退；但训练仅在 r=10 ->
    仅 index 10 有梯度，r=1/13/30/50 全用未训练随机索引，污染所有 OOD 轮次预测。
    正弦编码对任意 n_rounds 可外推，无未训练路径。审查必改#5。
    """

    def __init__(
        self,
        embed_dim: int,
        max_rounds: int = 50,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_rounds = max_rounds  # 保留以兼容调用方（正弦编码不依赖它）
        half = max(embed_dim // 2, 1)
        # 频率分母 10000^(2i/d) == exp(-ln(10000) * i / half)
        div = torch.exp(torch.arange(half, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / half))
        self.register_buffer('div_freq', div)  # [half]，随 model.to() 移动

    def forward(self, n_rounds: int, batch_size: int, device: torch.device) -> Tensor:
        pos = torch.tensor(float(n_rounds), device=self.div_freq.device)
        arg = pos * self.div_freq                            # [half]
        pe = torch.cat([torch.sin(arg), torch.cos(arg)])    # [2*half]
        if pe.shape[0] < self.embed_dim:                    # embed_dim 奇数补零
            pe = torch.cat([pe, torch.zeros(self.embed_dim - pe.shape[0], device=pe.device)])
        return pe[:self.embed_dim].unsqueeze(0).expand(batch_size, -1)  # [B, D]


class LineMeanPool(nn.Module):
    """Line Mean Pool - 沿logical observable方向池化

    论文发现沿特定方向（通常是垂直方向）池化效果更好。
    这与Surface Code的logical observable结构有关。
    """

    def __init__(
        self,
        pool_direction: str = "vertical",
    ):
        """初始化Line Mean Pool

        Args:
            pool_direction: 池化方向 ("vertical", "horizontal", "both")
        """
        super().__init__()
        self.pool_direction = pool_direction

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 2D特征图 [B, D, H, W]

        Returns:
            池化后的特征 [B, D]
        """
        if self.pool_direction == "vertical":
            # 沿垂直方向（H）池化，然后沿水平方向
            x = x.mean(dim=2)  # [B, D, W]
            x = x.mean(dim=2)  # [B, D]
        elif self.pool_direction == "horizontal":
            # 沿水平方向（W）池化，然后沿垂直方向
            x = x.mean(dim=3)  # [B, D, H]
            x = x.mean(dim=2)  # [B, D]
        elif self.pool_direction == "both":
            # 同时沿两个方向池化
            x = x.mean(dim=[2, 3])  # [B, D]
        else:
            raise ValueError(f"未知的池化方向: {self.pool_direction}")

        return x


class FullReadoutNetwork(nn.Module):
    """完整读出网络 - 论文完整实现

    论文架构（alphaqubit_design.md 4.7节）：
    1. Scatter to 2D stabilizer grid
    2. Conv2D (2×2 kernel): Stabilizer grid → Data qubit grid
    3. Project to readout dimension
    4. Line Mean Pool (沿logical observable方向)
    5. Add Cycle Embedding
    6. Deep ResNet (16层)
    7. Linear + Sigmoid → P(logical error)

    关键设计：
    - 2×2卷积将stab grid (d+1)×(d+1) 转换为data grid d×d
    - Line Mean Pool沿垂直方向池化效果更好（论文发现）
    - Cycle Embedding帮助模型理解时间范围
    - 16层Deep ResNet提供强大的表达能力
    """

    def __init__(
        self,
        stab_dim: int,
        coord_system,
        readout_dim: int = 64,
        num_resnet_layers: int = 16,
        max_rounds: int = 50,
        pool_direction: str = "vertical",
        dropout: float = 0.1,
    ):
        """初始化完整读出网络

        Args:
            stab_dim: 稳定子特征维度 (d_model)
            coord_system: 坐标系统对象
            readout_dim: 读出维度（ResNet隐藏维度，论文用64）
            num_resnet_layers: ResNet层数（论文用16层）
            max_rounds: 最大轮数
            pool_direction: 池化方向 ("vertical", "horizontal", "both")
            dropout: Dropout比率
        """
        super().__init__()

        self.coord_system = coord_system
        self.stab_dim = stab_dim
        self.readout_dim = readout_dim

        # Step 1: 2×2卷积 - Stabilizer grid → Data qubit grid
        # 输入: [B, d_model, d+1, d+1] (stabilizer grid)
        # 输出: [B, d_model, d, d] (data qubit grid)
        # kernel_size=2, stride=1 将(d+1)×(d+1) → d×d
        self.stab_to_data_conv = nn.Conv2d(
            in_channels=stab_dim,
            out_channels=stab_dim,
            kernel_size=2,
            stride=1,
            padding=0,
        )

        # Step 2: 投影到readout维度
        # [B, d_model, d, d] → [B, readout_dim, d, d]
        # 使用1×1卷积实现逐位置投影
        self.proj_conv = nn.Conv2d(
            in_channels=stab_dim,
            out_channels=readout_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Step 3: Line Mean Pool
        # [B, readout_dim, d, d] → [B, readout_dim]
        self.line_pool = LineMeanPool(pool_direction)

        # Step 4: Cycle Embedding
        # 添加轮次信息到pooled特征
        self.cycle_embed = CycleEmbedding(readout_dim, max_rounds)

        # Step 5: Deep ResNet (16层)
        # [B, readout_dim] → [B, readout_dim]
        self.resnet_layers = nn.ModuleList([
            ResNetBlock(readout_dim, dropout)
            for _ in range(num_resnet_layers)
        ])

        # Step 6: 最终归一化和输出
        self.final_norm = nn.LayerNorm(readout_dim)
        self.output_layer = nn.Linear(readout_dim, 1)

        # 使用小值初始化输出层，初始logit接近0但允许梯度流动
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        stab_features: Tensor,
        n_rounds: Optional[int] = None,
    ) -> Tensor:
        """前向传播

        Args:
            stab_features: 稳定子特征 [B, n_stab, D]
                来自RNN Core的最终状态
            n_rounds: 当前轮数（用于cycle embedding）

        Returns:
            logit [B, 1] - 未经sigmoid的原始输出

        使用方式：
            - 训练: BCEWithLogitsLoss(logit, target)
            - 推理: prob = sigmoid(logit), pred = (prob > 0.5)
        """
        B = stab_features.shape[0]
        device = stab_features.device

        # Step 1: Scatter到2D stabilizer grid
        # [B, n_stab, D] → [B, (d+1)², D]
        lattice_flat = self.coord_system.scatter(stab_features)
        # [B, (d+1)², D] → [B, D, d+1, d+1]
        lattice_2d = self.coord_system.to_2d(lattice_flat)

        # Step 2: 2×2卷积 - Stabilizer grid → Data qubit grid
        # [B, D, d+1, d+1] → [B, D, d, d]
        data_2d = self.stab_to_data_conv(lattice_2d)

        # Step 3: 投影到readout维度
        # [B, D, d, d] → [B, readout_dim, d, d]
        data_2d = self.proj_conv(data_2d)

        # Step 4: Line Mean Pool
        # [B, readout_dim, d, d] → [B, readout_dim]
        pooled = self.line_pool(data_2d)

        # Step 5: 添加Cycle Embedding
        if n_rounds is not None:
            cycle_embed = self.cycle_embed(n_rounds, B, device)
            x = pooled + cycle_embed
        else:
            x = pooled

        # Step 6: Deep ResNet (16层)
        for layer in self.resnet_layers:
            x = layer(x)

        # Step 7: 最终归一化和输出
        x = self.final_norm(x)
        logit = self.output_layer(x)

        return logit


