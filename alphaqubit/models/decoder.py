"""
AlphaQubit解码器 - 完整论文实现

这个模块实现了完整的AlphaQubit神经网络解码器，
包含所有论文中描述的组件：

1. SyndromeEmbedder (with position + time embedding)
2. RNN Core with 0.7 scaling factor
3. SyndromeTransformer (Self-Attention + GLU + Dilated Conv)
4. Late Fusion (StabToData Conv + Soft Embedding)
5. Deep ResNet Readout (16 layers + Cycle Embed + Line Mean Pool)

完整架构：
┌─────────────────────────────────────────────────────────────────────────┐
│                        AlphaQubitDecoder                                 │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         Main Path                                  │  │
│  │                                                                    │  │
│  │  events [B, T, n_stab]                                            │  │
│  │      │                                                             │  │
│  │      ▼  SyndromeEmbedder (+ Position + Time)                      │  │
│  │  [B, T, n_stab, D]                                                │  │
│  │      │                                                             │  │
│  │      ▼  RNN Core (0.7 scaling + SyndromeTransformer × 3)          │  │
│  │  [B, n_stab, D]                                                   │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        Late Fusion                                 │  │
│  │                                                                    │  │
│  │  H_T ─────────────────────┐                                       │  │
│  │                           │                                        │  │
│  │  final_soft ───► SoftEmbed│                                       │  │
│  │                           ▼                                        │  │
│  │               StabToDataConv + Concat                             │  │
│  │                           │                                        │  │
│  │                           ▼                                        │  │
│  │                F_fused [B, n_data, D']                            │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Deep Readout Network                            │  │
│  │                                                                    │  │
│  │  F_fused → LineMeanPool → (+CycleEmbed) → ResNet(16) → logit     │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Output: logit [B, 1], P(error) = sigmoid(logit)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..data.coordinates import CoordinateSystem
from .embeddings import SyndromeEmbedder
from .rnn_core import RNNCore
from .readout import FullReadoutNetwork, ResNetBlock, CycleEmbedding
from .fusion import LateFusion


class AlphaQubitDecoder(nn.Module):
    """AlphaQubit神经网络解码器V2 - 完整论文实现

    包含所有论文描述的组件：
    - SyndromeTransformer
    - 0.7缩放因子
    - Deep ResNet Readout (16层)
    - Cycle Embedding
    - Line Mean Pool

    主要改进：
    1. RNN Core使用SyndromeTransformer替代普通GRU
    2. 添加0.7缩放因子保持训练稳定
    3. Readout使用16层Deep ResNet
    4. 添加Cycle Embedding编码轮次信息
    """

    def __init__(
        self,
        coord_system: CoordinateSystem,
        # 主要维度参数
        embed_dim: int = 256,
        readout_dim: int = 64,
        # Transformer参数
        n_heads: int = 4,
        num_transformer_layers: int = 3,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        # Readout参数
        num_readout_layers: int = 16,
        pool_direction: str = "vertical",
        # 通用参数
        max_rounds: int = 50,
        dropout: float = 0.1,
        scale_factor: float = 0.7,
        # Fusion参数
        use_late_fusion: bool = True,
        fusion_type: str = "concat",
        soft_embed_dim: int = 64,
    ):
        """初始化AlphaQubit解码器V2

        Args:
            coord_system: 坐标系统对象
            embed_dim: 主嵌入维度（论文中d_model=256/320）
            readout_dim: 读出维度（论文中d_readout=64）
            n_heads: 注意力头数
            num_transformer_layers: SyndromeTransformer层数
            expansion_factor: GeGLU扩展因子
            num_conv_layers: 每层Transformer中的卷积层数
            num_readout_layers: Deep ResNet层数（论文用16）
            pool_direction: Line Mean Pool方向
            max_rounds: 最大轮数
            dropout: Dropout比率
            scale_factor: RNN Core缩放因子（论文用0.7）
            use_late_fusion: 是否使用Late Fusion（强烈推荐True）
            fusion_type: Late Fusion类型 ("concat", "add", "attention")
            soft_embed_dim: 软读出嵌入维度
        """
        super().__init__()

        # 保存配置
        self.coord_system = coord_system
        self.n_stab = coord_system.n_stab
        self.n_data = coord_system.n_data
        self.embed_dim = embed_dim
        self.use_late_fusion = use_late_fusion
        self.readout_dim = readout_dim
        self.max_rounds = max_rounds

        # ==================== 1. 综合征嵌入器 ====================
        # SyndromeEmbedder使用网格总位置数 (d+1)²
        grid_positions = (coord_system.distance + 1) ** 2
        self.syndrome_embedder = SyndromeEmbedder(
            embed_dim=embed_dim,
            n_positions=grid_positions,
            dropout=dropout,
        )

        # ==================== 2. RNN Core ====================
        # 使用完整的SyndromeTransformer实现
        self.rnn_core = RNNCore(
            embed_dim=embed_dim,
            n_stab=self.n_stab,
            n_heads=n_heads,
            num_transformer_layers=num_transformer_layers,
            expansion_factor=expansion_factor,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            scale_factor=scale_factor,
        )

        # ==================== 3. Late Fusion + Readout ====================
        if use_late_fusion:
            # 🔧 修复：添加Late Fusion模块
            self.late_fusion = LateFusion(
                stab_dim=embed_dim,
                soft_dim=soft_embed_dim,
                output_dim=embed_dim,
                coord_system=coord_system,
                fusion_type=fusion_type,
                dropout=dropout,
            )

            # 🔧 修复：创建Data Readout Head（处理融合后的data qubit特征）
            self.cycle_embed = CycleEmbedding(readout_dim, max_rounds)

            self.data_readout = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, readout_dim),
                nn.GELU(),
            )

            # Deep ResNet layers
            self.resnet_layers = nn.ModuleList([
                ResNetBlock(readout_dim, dropout)
                for _ in range(num_readout_layers)
            ])

            self.final_norm = nn.LayerNorm(readout_dim)
            self.output_layer = nn.Linear(readout_dim, 1)

            # 使用小值初始化输出层
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.output_layer.bias)

        else:
            # 旧版本：不使用Late Fusion（不推荐）
            self.readout = FullReadoutNetwork(
                stab_dim=embed_dim,
                coord_system=coord_system,
                readout_dim=readout_dim,
                num_resnet_layers=num_readout_layers,
                max_rounds=max_rounds,
                pool_direction=pool_direction,
                dropout=dropout,
            )

        # 保存参数用于forward
        self._current_rounds = None

    def forward(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Tensor,
        event_leakage: Tensor,
        final_soft: Optional[Tensor] = None,
        n_rounds: Optional[int] = None,
    ) -> Tensor:
        """前向传播 - 论文完整实现（带Late Fusion）

        Args:
            measurement: 原始测量值 [B, T, n_stab]，soft readout概率
            event: 检测事件 [B, T, n_stab]，软XOR概率
            leakage: 泄漏概率 [B, T, n_stab]，模拟环境通常为0
            event_leakage: 泄漏事件 [B, T, n_stab]，模拟环境通常为0
            final_soft: 最终数据比特软读出 [B, n_data] (必需！)
            n_rounds: 当前轮数（用于Cycle Embedding）

        Returns:
            logit: [B, 1]
        """
        B, T, n_stab = measurement.shape
        device = measurement.device

        # 如果未指定轮数，使用时间步数
        if n_rounds is None:
            n_rounds = T

        # 准备position索引（稳定子在网格中的真实位置）
        position_idx = self.coord_system.scatter_idx.to(device)
        stab_positions = self.coord_system.stab_positions_tensor.to(device)

        # ==================== 1. 逐轮嵌入 ====================
        # 对每个时间步t，使用SyndromeEmbedder处理5个输入
        syndrome_embeds = []
        for t in range(T):
            # 提取第t轮的数据 [B, n_stab]
            measurement_t = measurement[:, t, :]
            event_t = event[:, t, :]
            leakage_t = leakage[:, t, :]
            event_leakage_t = event_leakage[:, t, :]

            # 调用SyndromeEmbedder的5输入接口
            # [B, n_stab] → [B, n_stab, D]
            embed_t = self.syndrome_embedder(
                measurement=measurement_t,
                event=event_t,
                leakage=leakage_t,
                event_leakage=event_leakage_t,
                position_idx=position_idx,
            )
            syndrome_embeds.append(embed_t)

        # 堆叠所有时间步: [B, T, n_stab, D]
        embedded = torch.stack(syndrome_embeds, dim=1)

        # ==================== 2. RNN Core ====================
        # [B, T, n_stab, D] → [B, n_stab, D]
        stab_features = self.rnn_core(
            embedded,
            coord_system=self.coord_system,
            stab_positions=stab_positions,
        )

        # ==================== 3. Late Fusion + Readout ====================
        if self.use_late_fusion:
            # 🔧 修复：使用Late Fusion融合stabilizer特征和final_soft
            if final_soft is None:
                raise ValueError("final_soft is required when use_late_fusion=True")

            # Late Fusion: [B, n_stab, D] + [B, n_data] → [B, n_data, D]
            fused_features = self.late_fusion(stab_features, final_soft)

            # Global Mean Pooling: [B, n_data, D] → [B, D]
            pooled = fused_features.mean(dim=1)

            # Project to readout dim: [B, D] → [B, readout_dim]
            x = self.data_readout(pooled)

            # Add Cycle Embedding
            cycle_embed = self.cycle_embed(n_rounds, B, device)
            x = x + cycle_embed

            # Deep ResNet
            for layer in self.resnet_layers:
                x = layer(x)

            # Final output
            x = self.final_norm(x)
            logit = self.output_layer(x)

        else:
            # 旧版本：不使用Late Fusion（不推荐）
            logit = self.readout(stab_features, n_rounds=n_rounds)

        return logit

    def predict(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Tensor,
        event_leakage: Tensor,
        final_soft: Tensor,
        threshold: float = 0.5,
        n_rounds: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """预测逻辑错误

        Args:
            measurement: 原始测量值 [B, T, n_stab]
            event: 检测事件 [B, T, n_stab]
            leakage: 泄漏概率 [B, T, n_stab]
            event_leakage: 泄漏事件 [B, T, n_stab]
            final_soft: 最终数据比特软读出 [B, n_data]
            threshold: 分类阈值
            n_rounds: 当前轮数

        Returns:
            Tuple of:
                - predictions: 预测标签 [B] (0或1)
                - probabilities: 预测概率 [B]
        """
        logit = self.forward(measurement, event, leakage, event_leakage, final_soft, n_rounds)
        probabilities = torch.sigmoid(logit).squeeze(-1)
        predictions = (probabilities > threshold).long()
        return predictions, probabilities

    def get_num_parameters(self) -> Dict[str, int]:
        """获取模型参数数量统计"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        stats = {
            "syndrome_embedder": count_params(self.syndrome_embedder),
            "rnn_core": count_params(self.rnn_core),
        }

        if self.use_late_fusion:
            stats["late_fusion"] = count_params(self.late_fusion)
            stats["data_readout"] = count_params(self.data_readout)
            stats["resnet_layers"] = count_params(self.resnet_layers)
            stats["output"] = count_params(self.output_layer) + count_params(self.final_norm)
        else:
            stats["readout"] = count_params(self.readout)

        stats["total"] = sum(stats.values())
        return stats


class AlphaQubitDecoderConfig:
    """V2解码器配置类

    提供论文中的标准配置：
    - Sycamore: d_model=320, n_heads=4, readout_layers=16
    - Scaling: d_model=256, n_heads=4, readout_layers=4
    """

    @staticmethod
    def sycamore(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """Sycamore实验配置（论文默认）

        参数量约5M
        """
        return AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=320,
            readout_dim=64,
            n_heads=4,
            num_transformer_layers=3,
            expansion_factor=4,
            num_conv_layers=3,
            num_readout_layers=16,
            dropout=0.0,
            use_late_fusion=True,
            fusion_type="concat",
        )

    @staticmethod
    def scaling(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """Scaling实验配置

        参数量约2M
        """
        return AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=256,
            readout_dim=32,
            n_heads=4,
            num_transformer_layers=3,
            expansion_factor=4,
            num_conv_layers=3,
            num_readout_layers=4,
            dropout=0.0,
            use_late_fusion=True,
            fusion_type="concat",
        )

    @staticmethod
    def large(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """Large配置（当前等同于Sycamore）"""
        return AlphaQubitDecoderConfig.sycamore(coord_system)

    @staticmethod
    def base(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """基础配置，用于本地训练

        参数量约500K
        """
        return AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=128,
            readout_dim=32,
            n_heads=4,
            num_transformer_layers=2,
            expansion_factor=4,
            num_conv_layers=2,
            num_readout_layers=4,
            dropout=0.1,
            use_late_fusion=True,
            fusion_type="concat",
        )

    @staticmethod
    def small(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """小配置，用于快速实验

        参数量约100K
        """
        return AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=64,
            readout_dim=32,
            n_heads=4,
            num_transformer_layers=1,
            expansion_factor=2,
            num_conv_layers=1,
            num_readout_layers=2,
            dropout=0.1,
            use_late_fusion=True,
            fusion_type="concat",
        )

    @staticmethod
    def tiny(coord_system: CoordinateSystem) -> AlphaQubitDecoder:
        """极小配置，用于调试

        参数量约20K
        """
        return AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            readout_dim=16,
            n_heads=2,
            num_transformer_layers=1,
            expansion_factor=2,
            num_conv_layers=1,
            num_readout_layers=1,
            dropout=0.1,
            use_late_fusion=True,
            fusion_type="concat",
        )
