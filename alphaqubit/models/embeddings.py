"""
嵌入层模块 - 将输入特征映射到高维表示

这个模块实现了AlphaQubit论文中的嵌入层（完整版）。

## 主要组件

### 1. SyndromeEmbedder: 综合征嵌入器（论文完整版）
   按照论文设计接收5个独立输入：
   - measurement: 稳定子原始测量值 [B, n_stab]
   - event: 检测事件（软XOR） [B, n_stab]
   - leakage: 泄漏概率 [B, n_stab]（模拟环境为0）
   - event_leakage: 泄漏事件变化 [B, n_stab]（模拟环境为0）
   - position_idx: 位置索引 [n_stab]

   输出: 嵌入表示 [B, n_stab, D]

### 2. FinalDataEmbedder: 最终数据比特嵌入器
   - 输入: data_meas [B, n_data], position_idx [n_data]
   - 输出: 嵌入表示 [B, n_data, D]
   - 用于Late Fusion阶段，避免label leakage

### 3. PositionalEmbedding: 相对位置编码
   - 用于多距离联合训练
   - 将归一化(x,y)坐标映射到嵌入空间

## 使用示例

```python
import torch
from alphaqubit.models.embeddings import SyndromeEmbedder, FinalDataEmbedder
from alphaqubit.data.stim_generator import StimDataGenerator
from alphaqubit.data.soft_readout import SoftReadoutSimulator

# 1. 初始化嵌入器
d_model = 256
distance = 5
n_positions = (distance + 1) ** 2  # 6×6 = 36

syndrome_embedder = SyndromeEmbedder(
    embed_dim=d_model,
    n_positions=n_positions,
)

final_data_embedder = FinalDataEmbedder(
    embed_dim=d_model,
    n_positions=n_positions,
)

# 2. 生成数据
generator = StimDataGenerator(distance=5, rounds=25, p=0.005)
soft_sim = SoftReadoutSimulator(snr=10.0, t=0.01)

measurements, labels = generator.sample(num_shots=32)
ancilla_meas = generator.extract_ancilla_measurements(measurements)
final_data = generator.extract_final_data(measurements)

# 3. 准备5个输入
soft_meas = soft_sim.simulate(ancilla_meas)  # [B, T, n_stab]
soft_event = soft_sim.simulate_detection_events(ancilla_meas)  # [B, T, n_stab]
soft_final = soft_sim.simulate(final_data)  # [B, n_data]

# 模拟环境：leakage为0
leakage = torch.zeros_like(soft_meas)
event_leakage = torch.zeros_like(soft_meas)

# 位置索引
stab_pos_idx = torch.tensor(generator.coord_system.stab_grid_indices)
data_pos_idx = torch.tensor(generator.coord_system.data_grid_indices)

# 4. 逐轮嵌入syndrome
for t in range(25):
    syndrome_embed = syndrome_embedder(
        measurement=soft_meas[:, t],      # [B, n_stab]
        event=soft_event[:, t],           # [B, n_stab]
        leakage=leakage[:, t],            # [B, n_stab]
        event_leakage=event_leakage[:, t],# [B, n_stab]
        position_idx=stab_pos_idx,        # [n_stab]
    )  # 输出: [B, n_stab, D]

# 5. 嵌入最终数据
final_embed = final_data_embedder(
    data_meas=soft_final,          # [B, n_data]
    position_idx=data_pos_idx,     # [n_data]
)  # 输出: [B, n_data, D]
```

## 设计考虑

- **论文完整实现**: 严格按照论文架构，支持5个独立输入
- **模拟环境简化**: leakage和event_leakage在Stim模拟中为0，但保留接口
- **位置编码**: 使用可学习的Embedding表，每个网格位置一个向量
- **ResNet融合**: 通过ResNetBlock进行特征混合和非线性变换
- **避免label leakage**: 最终数据嵌入只在readout阶段使用
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SyndromeEmbedder(nn.Module):
    """综合征（Syndrome）嵌入器 - 论文完整版

    按照论文设计，接收5个独立输入并融合为高维嵌入表示。

    论文架构（alphaqubit_design.md 4.2节）：
    ┌─────────────────────────────────────────────────────────────┐
    │  measurement ──→ Linear(1, d_model) ──┐                    │
    │  event       ──→ Linear(1, d_model) ──┼──→ (+) ──→ ResNet  │
    │  leakage     ──→ Linear(1, d_model) ──┤           Block    │
    │  event_leak  ──→ Linear(1, d_model) ──┤                    │
    │  position    ──→ Embedding(n_pos, d_model)─┘               │
    └─────────────────────────────────────────────────────────────┘

    5个输入说明：
    1. measurement: 稳定子原始测量值（soft readout概率）
    2. event: 检测事件 = meas[t] XOR meas[t-1]（软XOR）
    3. leakage: 泄漏概率（模拟环境为0）
    4. event_leakage: 泄漏事件变化（模拟环境为0）
    5. position_idx: 稳定子在2D网格中的位置索引

    输入/输出：
        输入: 5个独有的张量
        输出: [B, n_stab, D] 嵌入表示
    """

    def __init__(
        self,
        embed_dim: int,
        n_positions: int,   # 网格位置总数 (d+1)²
        dropout: float = 0.0,
    ):
        """初始化综合征嵌入器

        Args:
            embed_dim: 嵌入维度 d_model
            n_positions: 位置总数，等于 (d+1)²
            dropout: Dropout比率（论文默认0.0）
        """
        super().__init__()

        self.embed_dim = embed_dim

        # ==================== 5个独立的线性投影层 ====================
        # 每个输入都有自己的投影层，将标量映射到d_model维
        self.measurement_proj = nn.Linear(1, embed_dim)
        self.event_proj = nn.Linear(1, embed_dim)
        self.leakage_proj = nn.Linear(1, embed_dim)
        self.event_leakage_proj = nn.Linear(1, embed_dim)

        # ==================== 位置嵌入 ====================
        # 使用可学习的嵌入表，每个网格位置一个向量
        self.position_embed = nn.Embedding(n_positions, embed_dim)

        # ==================== ResNet融合块 ====================
        # 导入ResNetBlock（从readout.py）
        from .readout import ResNetBlock
        self.resnet_block = ResNetBlock(dim=embed_dim, dropout=dropout)

    def forward(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Tensor,
        event_leakage: Tensor,
        position_idx: Tensor,
    ) -> Tensor:
        """前向传播

        Args:
            measurement: 原始测量值 [B, n_stab]
                        值范围[0,1]，soft readout概率
            event: 检测事件 [B, n_stab]
                  值范围[0,1]，软XOR概率
            leakage: 泄漏概率 [B, n_stab]
                    值范围[0,1]，模拟环境通常为0
            event_leakage: 泄漏事件 [B, n_stab]
                          值范围[-1,1]，模拟环境通常为0
            position_idx: 位置索引 [n_stab]
                         整数，范围[0, n_positions-1]

        Returns:
            嵌入表示 [B, n_stab, D]
        """
        # ==================== 投影各个输入 ====================
        # 将每个标量特征投影到d_model维
        # [B, n_stab] → [B, n_stab, 1] → [B, n_stab, D]
        meas_embed = self.measurement_proj(measurement.unsqueeze(-1))
        event_embed = self.event_proj(event.unsqueeze(-1))
        leak_embed = self.leakage_proj(leakage.unsqueeze(-1))
        event_leak_embed = self.event_leakage_proj(event_leakage.unsqueeze(-1))

        # ==================== 位置嵌入 ====================
        # [n_stab] → [n_stab, D] → [1, n_stab, D]
        pos_embed = self.position_embed(position_idx).unsqueeze(0)

        # ==================== 融合所有特征 ====================
        # 按照论文：直接相加5个嵌入
        embedded = (
            meas_embed +
            event_embed +
            leak_embed +
            event_leak_embed +
            pos_embed
        )  # [B, n_stab, D]

        # ==================== ResNet融合 ====================
        # 通过ResNet块进行非线性变换和特征混合
        output = self.resnet_block(embedded)  # [B, n_stab, D]

        return output


class PositionalEmbedding(nn.Module):
    """相对位置编码

    将归一化的(x, y)坐标映射到嵌入空间，使模型能**泛化到不同码距离**。
    适用于多距离联合训练，因为不同距离的码共享相同的位置编码器。

    映射方式：
    1. 输入坐标归一化到[-1, 1]²
    2. 使用MLP将2D坐标映射到D维嵌入

    归一化公式：
        x_norm = 2 * (col / (d+1)) - 1
        y_norm = 2 * (row / (d+1)) - 1

    为什么用相对位置？
    - d=3的stabilizer在grid中的相对位置与d=5的类似
    - 模型可以学习"边缘"、"角落"等位置概念
    - 便于迁移学习和联合训练
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        """初始化相对位置编码器

        Args:
            embed_dim: 输出嵌入维度
            hidden_dim: 隐藏层维度，默认为embed_dim
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        # MLP: 2D坐标 → D维嵌入
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, positions: Tensor) -> Tensor:
        """前向传播

        Args:
            positions: 归一化坐标 [n_stab, 2] 或 [B, n_stab, 2]
                      值范围应为[-1, 1]

        Returns:
            位置嵌入 [n_stab, D] 或 [B, n_stab, D]
        """
        return self.mlp(positions)

    @staticmethod
    def normalize_positions(
        row: Tensor,
        col: Tensor,
        grid_size: int,
    ) -> Tensor:
        """归一化位置坐标到[-1, 1]

        Args:
            row: 行坐标 [n_stab]
            col: 列坐标 [n_stab]
            grid_size: 网格大小 (d+1)

        Returns:
            归一化坐标 [n_stab, 2]
        """
        # 归一化到[-1, 1]
        x_norm = 2.0 * (col.float() / (grid_size - 1)) - 1.0
        y_norm = 2.0 * (row.float() / (grid_size - 1)) - 1.0

        return torch.stack([x_norm, y_norm], dim=-1)


class FinalDataEmbedder(nn.Module):
    """最终数据比特嵌入器

    按照论文设计（alphaqubit_design.md 4.3节），将最终轮的数据比特
    soft readout转换为嵌入表示。

    论文架构：
    ┌─────────────────────────────────────────────────────────────┐
    │  soft_meas ──→ MLP(1 → hidden → d_model) ──┐               │
    │                                             ├──→ (+) ──→    │
    │  position  ──→ Embedding(n_pos, d_model) ──┘      ResNet   │
    └─────────────────────────────────────────────────────────────┘

    设计考虑：
    - 输入是[0,1]的软概率值
    - 使用MLP而非简单Linear（更强表达力）
    - 添加位置信息
    - 通过ResNet融合

    注意：这个嵌入只在Late Fusion阶段使用，避免label leakage。
    """

    def __init__(
        self,
        embed_dim: int,
        n_positions: int,   # 网格位置总数 (d+1)²
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        """初始化最终数据嵌入器

        Args:
            embed_dim: 嵌入维度 d_model
            n_positions: 位置总数，等于 (d+1)²
            hidden_dim: MLP隐藏层维度（论文默认64）
            dropout: Dropout比率（论文默认0.0）
        """
        super().__init__()

        self.embed_dim = embed_dim

        # ==================== MLP值嵌入 ====================
        # 论文使用: Linear(1, 64) → ReLU → Linear(64, d_model)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # ==================== 位置嵌入 ====================
        self.position_embed = nn.Embedding(n_positions, embed_dim)

        # ==================== ResNet融合块 ====================
        from .readout import ResNetBlock
        self.resnet_block = ResNetBlock(dim=embed_dim, dropout=dropout)

    def forward(self, data_meas: Tensor, position_idx: Tensor) -> Tensor:
        """前向传播

        Args:
            data_meas: 最终数据比特软读出 [B, n_data]
                      值范围[0,1]
            position_idx: 位置索引 [n_data]
                         整数，范围[0, n_positions-1]

        Returns:
            嵌入表示 [B, n_data, D]
        """
        # ==================== MLP值嵌入 ====================
        # [B, n_data] → [B, n_data, 1] → [B, n_data, D]
        value_embed = self.value_mlp(data_meas.unsqueeze(-1))

        # ==================== 位置嵌入 ====================
        # [n_data] → [n_data, D] → [1, n_data, D]
        pos_embed = self.position_embed(position_idx).unsqueeze(0)

        # ==================== 融合 ====================
        embedded = value_embed + pos_embed  # [B, n_data, D]

        # ==================== ResNet融合 ====================
        output = self.resnet_block(embedded)  # [B, n_data, D]

        return output


# ==================== 保留旧名称以保持兼容性 ====================
SoftDataEmbedder = FinalDataEmbedder  # 别名


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码

    使用正弦/余弦函数生成位置编码，类似于Transformer。
    这种编码方式不需要学习，可以泛化到任意长度。

    编码公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    为**时间维度**提供位置信息，告诉网络"这是第几轮纠错"。
    """

    def __init__(self, embed_dim: int, max_len: int = 100):
        """初始化正弦位置编码

        Args:
            embed_dim: 嵌入维度
            max_len: 最大序列长度
        """
        super().__init__()

        # 预计算位置编码
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为buffer（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """获取位置编码

        Args:
            x: 输入张量 [B, T, ...] 或位置索引 [T]

        Returns:
            位置编码 [T, D] 或 [1, T, D]
        """
        if x.dim() == 1:
            # x是位置索引
            return self.pe[x]
        else:
            # x是序列，返回前T个位置的编码
            T = x.size(1)
            return self.pe[:T].unsqueeze(0)
