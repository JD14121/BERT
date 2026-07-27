"""
预训练解码器模块 - PretrainDecoder + TemporalReconstructionHead

这个模块实现了预训练阶段的完整解码器：
1. PretrainDecoder: 复用 SyndromeEmbedder + RNNCore，添加预训练 Head
2. TemporalReconstructionHead: 从 RNN Core 中间状态重建 syndrome

架构设计：
    SyndromeEmbedder → RNNCore → TemporalReconstructionHead

关键决策：
- 使用 RNNCore.forward_with_all_states 获取每轮中间状态
- 从每轮中间状态预测该轮的 syndrome bit
- 对所有轮次共享同一个 reconstruction head
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.coordinates import CoordinateSystem
from .embeddings import SyndromeEmbedder
from .rnn_core import RNNCore


class TemporalReconstructionHead(nn.Module):
    """时序重建头

    从 RNN Core 的每轮中间状态预测该轮的 syndrome bit。

    输入: RNN Core 所有中间状态 [B, T, n_stab, D]
    输出: 每轮的 syndrome 预测 logits [B, T, n_stab]

    设计：
    1. 对每轮的状态，使用共享的 MLP 映射到预测 logits
    2. MLP: Linear(D → D/2) → GELU → Linear(D/2 → 1)
    3. 所有轮次共享同一个 head（时序不变性）

    为什么这能工作：
    - RNN Core 的 state 已经累积了所有历史轮次的信息
    - 0.7 缩放机制保留了旧轮次的信息（只是衰减）
    - 从第 t 轮的中间状态预测第 t 轮的 syndrome 是自然的
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """初始化时序重建头

        Args:
            embed_dim: 嵌入维度 D
            hidden_dim: 隐藏层维度，默认 D/2
            dropout: Dropout 比率
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim // 2

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 共享的 reconstruction MLP
        # 输入: [B, n_stab, D] → 输出: [B, n_stab, 1]
        # 注意：输出为 logits，损失函数使用 binary_cross_entropy_with_logits
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_states: Tensor) -> Tensor:
        """前向传播

        Args:
            all_states: RNN Core 的所有中间状态 [B, T, n_stab, D]

        Returns:
            预测 logits [B, T, n_stab]
        """
        B, T, n_stab, D = all_states.shape

        # 重塑为 [B*T, n_stab, D]，对所有轮次共享同一个 head
        states_flat = all_states.reshape(B * T, n_stab, D)

        # 应用共享 MLP
        # [B*T, n_stab, D] → [B*T, n_stab, 1]
        pred_flat = self.reconstruction_mlp(states_flat)

        # 重塑回 [B, T, n_stab]
        pred = pred_flat.squeeze(-1).reshape(B, T, n_stab)

        return pred


class PretrainDecoder(nn.Module):
    """预训练解码器

    与 AlphaQubitDecoder 的区别：
    1. 不使用 final_soft（预训练不需要最终 data qubit 测量）
    2. 不使用 Late Fusion 和 Deep ResNet Readout
    3. 替换为 TemporalReconstructionHead
    4. 使用 RNNCore.forward_with_all_states 获取每轮状态

    架构：
        SyndromeEmbedder → RNNCore → TemporalReconstructionHead

    使用示例：
        ```python
        decoder = PretrainDecoder(coord_system, embed_dim=256)

        # 前向传播
        pred = decoder(
            measurement=masked_meas,    # [B, T, n_stab]
            event=masked_event,         # [B, T, n_stab]
            leakage=leakage,            # [B, T, n_stab]
            event_leakage=event_leakage,# [B, T, n_stab]
        )  # 输出: [B, T, n_stab]

        # 计算损失（仅 mask 位置）
        loss = F.binary_cross_entropy_with_logits(pred[mask_indices], target[mask_indices])
        ```
    """

    def __init__(
        self,
        coord_system: CoordinateSystem,
        embed_dim: int = 256,
        n_heads: int = 4,
        num_transformer_layers: int = 3,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        scale_factor: float = 0.7,
        reconstruction_hidden_dim: Optional[int] = None,
    ):
        """初始化预训练解码器

        Args:
            coord_system: 坐标系统对象
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            num_transformer_layers: SyndromeTransformer 层数
            expansion_factor: GeGLU 扩展因子
            num_conv_layers: 每层 Transformer 中的卷积层数
            dropout: Dropout 比率
            scale_factor: RNN Core 缩放因子
            reconstruction_hidden_dim: 重建头隐藏层维度
        """
        super().__init__()

        self.coord_system = coord_system
        self.n_stab = coord_system.n_stab
        self.embed_dim = embed_dim

        # 保存架构配置（用于迁移到 FineTuneDecoder）
        self.arch_config = {
            'embed_dim': embed_dim,
            'n_heads': n_heads,
            'num_transformer_layers': num_transformer_layers,
            'expansion_factor': expansion_factor,
            'num_conv_layers': num_conv_layers,
            'dropout': dropout,
            'scale_factor': scale_factor,
        }

        # ==================== 1. 综合征嵌入器 ====================
        grid_positions = (coord_system.distance + 1) ** 2
        self.syndrome_embedder = SyndromeEmbedder(
            embed_dim=embed_dim,
            n_positions=grid_positions,
            dropout=dropout,
        )

        # ==================== 2. RNN Core ====================
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

        # ==================== 3. 时序重建头 ====================
        self.reconstruction_head = TemporalReconstructionHead(
            embed_dim=embed_dim,
            hidden_dim=reconstruction_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Optional[Tensor] = None,
        event_leakage: Optional[Tensor] = None,
    ) -> Tensor:
        """前向传播 - 预训练版本

        Args:
            measurement: 原始测量值 [B, T, n_stab]
            event: 检测事件 [B, T, n_stab]
            leakage: 泄漏概率 [B, T, n_stab]（可选）
            event_leakage: 泄漏事件 [B, T, n_stab]（可选）

        Returns:
            pred: 预测 logits [B, T, n_stab]
        """
        B, T, n_stab = measurement.shape
        device = measurement.device

        # 准备 position 索引
        position_idx = self.coord_system.scatter_idx.to(device)
        stab_positions = self.coord_system.stab_positions_tensor.to(device)

        # 处理可选输入
        if leakage is None:
            leakage = torch.zeros_like(measurement)
        if event_leakage is None:
            event_leakage = torch.zeros_like(event)

        # ==================== 1. 逐轮嵌入 ====================
        syndrome_embeds = []
        for t in range(T):
            measurement_t = measurement[:, t, :]
            event_t = event[:, t, :]
            leakage_t = leakage[:, t, :]
            event_leakage_t = event_leakage[:, t, :]

            embed_t = self.syndrome_embedder(
                measurement=measurement_t,
                event=event_t,
                leakage=leakage_t,
                event_leakage=event_leakage_t,
                position_idx=position_idx,
            )
            syndrome_embeds.append(embed_t)

        embedded = torch.stack(syndrome_embeds, dim=1)  # [B, T, n_stab, D]

        # ==================== 2. RNN Core - 获取所有中间状态 ====================
        all_states = self.rnn_core.forward_with_all_states(
            embedded,
            coord_system=self.coord_system,
            stab_positions=stab_positions,
        )  # [B, T, n_stab, D]

        # ==================== 3. 时序重建 ====================
        pred = self.reconstruction_head(all_states)  # [B, T, n_stab]

        return pred

    def get_encoder_state_dict(self) -> Dict[str, Tensor]:
        """获取 Encoder 的 state_dict（用于迁移到微调）

        Returns:
            包含 syndrome_embedder 和 rnn_core 权重的字典
        """
        encoder_state = {}
        encoder_state.update({
            f"syndrome_embedder.{k}": v
            for k, v in self.syndrome_embedder.state_dict().items()
        })
        encoder_state.update({
            f"rnn_core.{k}": v
            for k, v in self.rnn_core.state_dict().items()
        })
        return encoder_state

    def get_num_parameters(self) -> Dict[str, int]:
        """获取模型参数数量统计"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "syndrome_embedder": count_params(self.syndrome_embedder),
            "rnn_core": count_params(self.rnn_core),
            "reconstruction_head": count_params(self.reconstruction_head),
            "total": sum(count_params(m) for m in [self.syndrome_embedder, self.rnn_core, self.reconstruction_head]),
        }


class FineTuneDecoder(nn.Module):
    """微调解码器

    加载预训练的 Encoder 权重，添加完整的 AlphaQubit Readout。

    与 PretrainDecoder 的区别：
    1. 加载预训练的 SyndromeEmbedder 和 RNNCore 权重
    2. 支持 freeze_encoder / 分层学习率
    3. 添加完整的 Late Fusion + Deep ResNet Readout
    4. 需要 final_soft 输入

    使用示例：
        ```python
        # 1. 创建预训练模型并加载权重
        pretrain_model = PretrainDecoder(coord_system)
        pretrain_model.load_state_dict(torch.load('pretrain.pt'))

        # 2. 创建微调解码器
        finetune_model = FineTuneDecoder(
            coord_system=coord_system,
            pretrained_encoder=pretrain_model,
        )

        # 3. 微调训练
        logit = finetune_model(
            measurement=meas,
            event=event,
            leakage=leakage,
            event_leakage=event_leakage,
            final_soft=final_soft,
        )  # [B, 1]
        ```
    """

    def __init__(
        self,
        coord_system: CoordinateSystem,
        pretrained_encoder: Optional[PretrainDecoder] = None,
        pretrained_state_dict: Optional[Dict[str, Tensor]] = None,
        embed_dim: int = 256,
        readout_dim: int = 64,
        n_heads: int = 4,
        num_transformer_layers: int = 3,
        expansion_factor: int = 4,
        num_conv_layers: int = 3,
        num_readout_layers: int = 16,
        dropout: float = 0.1,
        scale_factor: float = 0.7,
        max_rounds: int = 50,
    ):
        """初始化微调解码器

        Args:
            coord_system: 坐标系统对象
            pretrained_encoder: 预训练好的 PretrainDecoder（可选）
            pretrained_state_dict: 预训练权重字典（可选）
            embed_dim: 嵌入维度
            readout_dim: 读出维度
            n_heads: 注意力头数
            num_transformer_layers: Transformer 层数
            expansion_factor: GeGLU 扩展因子
            num_conv_layers: 卷积层数
            num_readout_layers: Deep ResNet 层数
            dropout: Dropout 比率
            scale_factor: RNN Core 缩放因子
            max_rounds: 最大轮数
        """
        super().__init__()

        self.coord_system = coord_system
        self.n_stab = coord_system.n_stab
        self.n_data = coord_system.n_data
        self.embed_dim = embed_dim

        # 如果提供了预训练模型，从中提取架构参数
        if pretrained_encoder is not None:
            arch = pretrained_encoder.arch_config
            n_heads = arch['n_heads']
            num_transformer_layers = arch['num_transformer_layers']
            expansion_factor = arch['expansion_factor']
            num_conv_layers = arch['num_conv_layers']
            scale_factor = arch['scale_factor']
            print(f"[FineTuneDecoder] 从预训练模型提取架构: n_heads={n_heads}, num_layers={num_transformer_layers}, num_conv_layers={num_conv_layers}")

        # ==================== 1. Encoder（加载预训练权重）====================
        grid_positions = (coord_system.distance + 1) ** 2
        self.syndrome_embedder = SyndromeEmbedder(
            embed_dim=embed_dim,
            n_positions=grid_positions,
            dropout=dropout,
        )

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

        # 加载预训练权重
        if pretrained_encoder is not None:
            self._load_pretrained_encoder(pretrained_encoder.get_encoder_state_dict())
        elif pretrained_state_dict is not None:
            self._load_pretrained_encoder(pretrained_state_dict)

        # ==================== 2. Late Fusion + Readout ====================
        from .fusion import LateFusion
        from .readout import ResNetBlock, CycleEmbedding

        self.late_fusion = LateFusion(
            stab_dim=embed_dim,
            soft_dim=64,
            output_dim=embed_dim,
            coord_system=coord_system,
            fusion_type="concat",
            dropout=dropout,
        )

        self.cycle_embed = CycleEmbedding(readout_dim, max_rounds)

        self.data_readout = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, readout_dim),
            nn.GELU(),
        )

        self.resnet_layers = nn.ModuleList([
            ResNetBlock(readout_dim, dropout)
            for _ in range(num_readout_layers)
        ])

        self.final_norm = nn.LayerNorm(readout_dim)
        self.output_layer = nn.Linear(readout_dim, 1)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def _load_pretrained_encoder(self, state_dict: Dict[str, Tensor]):
        """加载预训练 Encoder 权重

        Args:
            state_dict: 包含 syndrome_embedder 和 rnn_core 权重的字典
        """
        # 分离 syndrome_embedder 和 rnn_core 的权重
        embedder_state = {}
        rnn_state = {}

        for key, value in state_dict.items():
            if key.startswith("syndrome_embedder."):
                embedder_state[key.replace("syndrome_embedder.", "")] = value
            elif key.startswith("rnn_core."):
                rnn_state[key.replace("rnn_core.", "")] = value

        # 加载权重
        if embedder_state:
            self.syndrome_embedder.load_state_dict(embedder_state)
        if rnn_state:
            self.rnn_core.load_state_dict(rnn_state)

    def forward(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Tensor,
        event_leakage: Tensor,
        final_soft: Tensor,
        n_rounds: Optional[int] = None,
    ) -> Tensor:
        """前向传播 - 微调版本

        Args:
            measurement: 原始测量值 [B, T, n_stab]
            event: 检测事件 [B, T, n_stab]
            leakage: 泄漏概率 [B, T, n_stab]
            event_leakage: 泄漏事件 [B, T, n_stab]
            final_soft: 最终数据比特软读出 [B, n_data]
            n_rounds: 当前轮数（用于 Cycle Embedding）

        Returns:
            logit: [B, 1]
        """
        B, T, n_stab = measurement.shape
        device = measurement.device

        if n_rounds is None:
            n_rounds = T

        position_idx = self.coord_system.scatter_idx.to(device)
        stab_positions = self.coord_system.stab_positions_tensor.to(device)

        # ==================== 1. 逐轮嵌入 ====================
        syndrome_embeds = []
        for t in range(T):
            measurement_t = measurement[:, t, :]
            event_t = event[:, t, :]
            leakage_t = leakage[:, t, :]
            event_leakage_t = event_leakage[:, t, :]

            embed_t = self.syndrome_embedder(
                measurement=measurement_t,
                event=event_t,
                leakage=leakage_t,
                event_leakage=event_leakage_t,
                position_idx=position_idx,
            )
            syndrome_embeds.append(embed_t)

        embedded = torch.stack(syndrome_embeds, dim=1)

        # ==================== 2. RNN Core ====================
        stab_features = self.rnn_core(
            embedded,
            coord_system=self.coord_system,
            stab_positions=stab_positions,
        )

        # ==================== 3. Late Fusion + Readout ====================
        fused_features = self.late_fusion(stab_features, final_soft)
        pooled = fused_features.mean(dim=1)
        x = self.data_readout(pooled)

        cycle_embed = self.cycle_embed(n_rounds, B, device)
        x = x + cycle_embed

        for layer in self.resnet_layers:
            x = layer(x)

        x = self.final_norm(x)
        logit = self.output_layer(x)

        return logit

    def get_encoder_parameters(self):
        """获取 Encoder 的参数（用于分层学习率）"""
        return list(self.syndrome_embedder.parameters()) + list(self.rnn_core.parameters())

    def get_readout_parameters(self):
        """获取 Readout 的参数（用于分层学习率）"""
        readout_params = []
        readout_params.extend(self.late_fusion.parameters())
        readout_params.extend(self.data_readout.parameters())
        readout_params.extend(self.cycle_embed.parameters())
        readout_params.extend(self.resnet_layers.parameters())
        readout_params.extend(self.final_norm.parameters())
        readout_params.extend(self.output_layer.parameters())
        return readout_params

    def freeze_encoder(self):
        """冻结 Encoder 参数"""
        for param in self.syndrome_embedder.parameters():
            param.requires_grad = False
        for param in self.rnn_core.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻 Encoder 参数"""
        for param in self.syndrome_embedder.parameters():
            param.requires_grad = True
        for param in self.rnn_core.parameters():
            param.requires_grad = True
