"""
模型模块 - AlphaQubit神经网络解码器

这个模块包含AlphaQubit解码器的所有神经网络组件：

组件层级结构：
┌─────────────────────────────────────────────────────────────┐
│                    AlphaQubitDecoder                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Main Path                          │   │
│  │  events → SyndromeEmbedder → RNNCore → H_T          │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Soft Path                          │   │
│  │  final_soft → SoftDataEmbedder → F_soft             │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Late Fusion                        │   │
│  │  H_T → StabToDataConv → concat(F_soft) → Readout    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

各组件说明：
- SyndromeEmbedder: 将detection events嵌入到高维空间（5输入版本）
- RNNCore: 处理时间序列的RNN核心（带0.7缩放+SyndromeTransformer）
- ConvBlock: Scatter-Conv-Gather操作
- StabToDataConv: 稳定子特征到数据比特特征的映射
- LateFusion: 融合主路径和软读出路径
- FullReadoutNetwork: 完整读出网络（LineMeanPool + CycleEmbedding + Deep ResNet）
- AlphaQubitDecoder: 完整的端到端解码器
"""

from .embeddings import SyndromeEmbedder, PositionalEmbedding, SoftDataEmbedder
from .conv_block import ConvBlock, ScatterConvGather
from .fusion import StabToDataConv, LateFusion
from .readout import (
    FullReadoutNetwork,
    ResNetBlock,
    CycleEmbedding,
    LineMeanPool,
)
from .decoder import AlphaQubitDecoder, AlphaQubitDecoderConfig

# 新增组件
from .transformer import (
    SyndromeTransformer,
    SyndromeTransformerStack,
    MultiHeadSelfAttention,
    GatedDenseBlock,
    DilatedConvBlock,
    SpatialAttentionBias,
)
from .rnn_core import RNNCore

__all__ = [
    # 嵌入层
    "SyndromeEmbedder",
    "PositionalEmbedding",
    "SoftDataEmbedder",
    # Transformer组件
    "SyndromeTransformer",
    "SyndromeTransformerStack",
    "MultiHeadSelfAttention",
    "GatedDenseBlock",
    "DilatedConvBlock",
    "SpatialAttentionBias",
    # RNN Core
    "RNNCore",
    # 卷积层
    "ConvBlock",
    "ScatterConvGather",
    # 融合层
    "StabToDataConv",
    "LateFusion",
    # 输出层（论文实现）
    "FullReadoutNetwork",
    "ResNetBlock",
    "CycleEmbedding",
    "LineMeanPool",
    # 完整模型
    "AlphaQubitDecoder",
    "AlphaQubitDecoderConfig",
]
