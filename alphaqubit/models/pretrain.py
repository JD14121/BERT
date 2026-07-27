"""
自监督预训练模块 - Masked Syndrome Modeling

这个模块实现了基于 AlphaQubit 架构的自监督预训练框架：
1. MaskedSyndromeModeling: BERT 式的 syndrome 序列 masking
2. PretrainDecoder: 预训练解码器（复用 SyndromeEmbedder + RNNCore）
3. TemporalReconstructionHead: 从 RNN Core 状态重建 syndrome

核心设计原则：
- 复用现有架构：SyndromeEmbedder、RNNCore、SyndromeTransformer
- 预训练阶段不需要 label 和 final_soft
- 微调阶段加载预训练 Encoder 权重，替换 Readout Head

与 Syndrome-BERT.md 原方案的区别：
- 保留 5 通道输入（measurement/event/leakage/event_leakage/position）
- 保留 RNN Core（0.7 缩放 + 3 层 Transformer）
- 保留 DilatedConvBlock 空间建模
- 预训练 Head 从 RNN Core 中间状态重建，而非从最终状态
"""

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedSyndromeModeling:
    """Syndrome 序列的 BERT 式 Masking

    对 syndrome 序列进行随机 masking，生成自监督预训练任务。
    模型需要预测被 mask 位置的原始 syndrome bit。

    Masking 规则（与 BERT 一致）：
    1. 随机选择 mask_ratio 的 (time, stab) 位置
    2. 被选中的位置：
       - 80% 概率替换为 [MASK] token（可学习嵌入）
       - 10% 概率替换为随机值（0 或 1）
       - 10% 概率保持原值（但模型仍需预测）

    关键设计决策：
    - 同时 mask measurement 和 event：因为 event = XOR(meas[t], meas[t-1])，
      只 mask 一个会破坏一致性
    - [MASK] 是可学习嵌入：不是固定值 0.5
    - 保留位置编码：即使被 mask，位置信息仍通过 position_idx 保留

    为什么对 syndrome 有效：
    - 空间局部性：相邻 stabilizer 共享 data qubit，高度相关
    - 时间持续性：真实错误通常持续多轮
    - 错误配对：defect 通常成对出现（错误链的两端）
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_token_value: float = 0.5,
        random_replace_prob: float = 0.1,
        keep_original_prob: float = 0.1,
    ):
        """初始化 MaskedSyndromeModeling

        Args:
            mask_ratio: masking 比例，默认 15%
            mask_token_value: [MASK] token 的值（当不使用可学习嵌入时）
            random_replace_prob: 替换为随机值的概率（BERT 的 10%）
            keep_original_prob: 保持原值的概率（BERT 的 10%）
        """
        self.mask_ratio = mask_ratio
        self.mask_token_value = mask_token_value
        self.random_replace_prob = random_replace_prob
        self.keep_original_prob = keep_original_prob

        # 验证概率和为 1
        assert abs(random_replace_prob + keep_original_prob - 0.2) < 1e-6, \
            "random_replace_prob + keep_original_prob 应等于 0.2"

    def mask_sequence(
        self,
        measurement: Tensor,
        event: Tensor,
        leakage: Optional[Tensor] = None,
        event_leakage: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """对 syndrome 序列进行 masking

        Args:
            measurement: 原始测量值 [B, T, n_stab]
            event: 检测事件 [B, T, n_stab]
            leakage: 泄漏概率 [B, T, n_stab]（可选）
            event_leakage: 泄漏事件 [B, T, n_stab]（可选）

        Returns:
            Tuple of:
                - masked_inputs: 字典，包含 masking 后的输入
                  {
                    'measurement': [B, T, n_stab],
                    'event': [B, T, n_stab],
                    'leakage': [B, T, n_stab]（如果输入提供）,
                    'event_leakage': [B, T, n_stab]（如果输入提供）,
                  }
                - mask_indices: [B, T, n_stab] bool tensor，True 表示被 mask
        """
        B, T, n_stab = measurement.shape
        device = measurement.device

        # 1. 生成 mask 位置
        # 每个样本独立随机 mask
        mask_indices = self._generate_mask_indices(B, T, n_stab, device)

        # 2. 对 measurement 和 event 进行 masking
        masked_measurement = self._apply_mask(measurement, mask_indices)
        masked_event = self._apply_mask(event, mask_indices)

        # 3. 构建输出字典
        masked_inputs = {
            'measurement': masked_measurement,
            'event': masked_event,
        }

        # 对可选输入也进行 masking
        if leakage is not None:
            masked_inputs['leakage'] = self._apply_mask(leakage, mask_indices)
        if event_leakage is not None:
            masked_inputs['event_leakage'] = self._apply_mask(event_leakage, mask_indices)

        return masked_inputs, mask_indices

    def _generate_mask_indices(
        self,
        B: int,
        T: int,
        n_stab: int,
        device: torch.device,
    ) -> Tensor:
        """生成 mask 位置索引

        对每个样本，随机选择 mask_ratio 比例的 (t, stab) 位置。

        Args:
            B: batch size
            T: 时间步数
            n_stab: stabilizer 数量
            device: 设备

        Returns:
            [B, T, n_stab] bool tensor，True 表示被 mask
        """
        # 生成随机矩阵
        rand = torch.rand(B, T, n_stab, device=device)

        # 选择 mask_ratio 比例的位置
        mask_indices = rand < self.mask_ratio

        return mask_indices

    def _apply_mask(self, tensor: Tensor, mask_indices: Tensor) -> Tensor:
        """对 tensor 应用 masking

        对 mask 位置：
        - 80% → [MASK] token (mask_token_value)
        - 10% → 随机值（0 或 1）
        - 10% → 保持原值

        Args:
            tensor: [B, T, n_stab]
            mask_indices: [B, T, n_stab] bool

        Returns:
            masking 后的 tensor [B, T, n_stab]
        """
        # 复制原 tensor
        masked = tensor.clone()

        # 生成随机数决定每个 mask 位置的处理方式
        rand = torch.rand_like(tensor)

        # 1. 替换为 [MASK] token（80% 的 mask 位置）
        mask_token_mask = mask_indices & (rand < 0.8)
        masked = torch.where(mask_token_mask, self.mask_token_value, masked)

        # 2. 替换为随机值（10% 的 mask 位置）
        # 随机值：0 或 1，各 50%
        random_mask = mask_indices & (rand >= 0.8) & (rand < 0.9)
        random_values = (torch.rand_like(tensor) > 0.5).float()
        masked = torch.where(random_mask, random_values, masked)

        # 3. 保持原值（10% 的 mask 位置）
        # 不需要操作，因为 masked 已经是原值的复制
        # keep_original_mask = mask_indices & (rand >= 0.9)

        return masked

    def get_targets(self, measurement: Tensor, mask_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """获取预训练目标

        预训练的目标是预测被 mask 位置的原始 syndrome bit。
        我们使用 measurement 作为目标（因为 event 可以从 measurement 推导）。

        Args:
            measurement: 原始测量值 [B, T, n_stab]
            mask_indices: [B, T, n_stab] bool

        Returns:
            Tuple of:
                - targets: [N_masked] 被 mask 位置的原始值
                - flat_mask_indices: [N_masked] flat 索引，用于从预测中提取
        """
        # 提取被 mask 位置的值
        targets = measurement[mask_indices]

        return targets, mask_indices


class SpatialClusterMasking(MaskedSyndromeModeling):
    """空间聚类 Masking

    不仅随机 mask 单个 stabilizer，还 mask 相邻的 stabilizer 块。
    这迫使模型学习更强的空间推理能力。

    策略：
    1. 随机选择一个中心 stabilizer
    2. 以一定概率扩展 mask 到相邻 stabilizer（空间传播）
    3. 形成不规则的 "缺陷簇"
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        cluster_prob: float = 0.3,
        max_cluster_size: int = 4,
        **kwargs
    ):
        """初始化

        Args:
            mask_ratio: 总体 masking 比例
            cluster_prob: 中心点扩展为簇的概率
            max_cluster_size: 最大簇大小
        """
        super().__init__(mask_ratio=mask_ratio, **kwargs)
        self.cluster_prob = cluster_prob
        self.max_cluster_size = max_cluster_size

    def _generate_mask_indices(
        self,
        B: int,
        T: int,
        n_stab: int,
        device: torch.device,
    ) -> Tensor:
        """生成空间聚类 mask"""
        # 先随机选择种子位置
        rand = torch.rand(B, T, n_stab, device=device)
        seed_mask = rand < (self.mask_ratio / self.max_cluster_size)

        # TODO: 实现空间扩展逻辑（需要 stabilizer 邻接信息）
        # 当前简化：直接返回种子 mask
        # 完整实现需要 coord_system 来知道哪些 stabilizer 相邻

        return seed_mask


class TemporalSpanMasking(MaskedSyndromeModeling):
    """时间跨度 Masking

    Mask 连续多轮的同一 stabilizer。
    这迫使模型学习时间持续性理解。

    策略：
    1. 随机选择 stabilizer 和起始轮次
    2. Mask 从起始轮开始的一段连续轮次
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        min_span: int = 2,
        max_span: int = 5,
        **kwargs
    ):
        """初始化

        Args:
            mask_ratio: 总体 masking 比例
            min_span: 最小时间跨度
            max_span: 最大时间跨度
        """
        super().__init__(mask_ratio=mask_ratio, **kwargs)
        self.min_span = min_span
        self.max_span = max_span

    def _generate_mask_indices(
        self,
        B: int,
        T: int,
        n_stab: int,
        device: torch.device,
    ) -> Tensor:
        """生成时间跨度 mask"""
        mask_indices = torch.zeros(B, T, n_stab, dtype=torch.bool, device=device)

        # 计算需要 mask 的总位置数
        total_positions = B * T * n_stab
        target_masked = int(total_positions * self.mask_ratio)

        # 生成随机跨度
        for _ in range(target_masked // self.min_span):
            # 随机选择 batch、stabilizer
            b = torch.randint(0, B, (1,)).item()
            stab = torch.randint(0, n_stab, (1,)).item()

            # 随机选择起始轮和跨度
            span = torch.randint(self.min_span, self.max_span + 1, (1,)).item()
            start_t = torch.randint(0, max(1, T - span + 1), (1,)).item()
            end_t = min(start_t + span, T)

            # Mask 这段跨度
            mask_indices[b, start_t:end_t, stab] = True

        return mask_indices


class PretrainLoss(nn.Module):
    """预训练损失函数

    仅计算被 mask 位置的 BCE 损失。

    可选辅助损失：
    1. 空间一致性损失：相邻 mask 位置的预测应该平滑
    2. 时间一致性损失：同一 stab 在不同轮的预测应该连贯
    """

    def __init__(
        self,
        use_spatial_consistency: bool = False,
        use_temporal_consistency: bool = False,
        consistency_weight: float = 0.1,
    ):
        """初始化

        Args:
            use_spatial_consistency: 是否使用空间一致性损失
            use_temporal_consistency: 是否使用时间一致性损失
            consistency_weight: 一致性损失的权重
        """
        super().__init__()
        self.use_spatial_consistency = use_spatial_consistency
        self.use_temporal_consistency = use_temporal_consistency
        self.consistency_weight = consistency_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask_indices: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """计算预训练损失

        Args:
            pred: 预测 logits [B, T, n_stab] 或 [B, T, n_stab, 1]
            target: 目标值 [B, T, n_stab]
            mask_indices: [B, T, n_stab] bool

        Returns:
            Tuple of:
                - loss: 标量损失
                - metrics: 包含各种指标的字典
        """
        # 确保形状一致
        if pred.dim() == 4:
            pred = pred.squeeze(-1)

        # 1. 主损失：仅计算 mask 位置的 BCE
        masked_pred = pred[mask_indices]
        masked_target = target[mask_indices]

        if masked_pred.numel() == 0:
            # 如果没有 mask 位置，返回 0 损失
            return torch.tensor(0.0, device=pred.device), {
                'loss': 0.0,
                'mask_accuracy': 0.0,
                'mask_count': 0,
            }

        # 使用 logits + BCEWithLogits，兼容 AMP 且数值更稳定
        main_loss = F.binary_cross_entropy_with_logits(masked_pred, masked_target)

        # 2. 可选辅助损失（基于概率，便于实现平滑性约束）
        pred_prob = torch.sigmoid(pred)
        total_loss = main_loss
        metrics = {
            'main_loss': main_loss.item(),
            'mask_count': masked_pred.numel(),
        }

        # 空间一致性损失
        if self.use_spatial_consistency:
            spatial_loss = self._spatial_consistency_loss(pred_prob, mask_indices)
            total_loss = total_loss + self.consistency_weight * spatial_loss
            metrics['spatial_loss'] = spatial_loss.item()

        # 时间一致性损失
        if self.use_temporal_consistency:
            temporal_loss = self._temporal_consistency_loss(pred_prob, mask_indices)
            total_loss = total_loss + self.consistency_weight * temporal_loss
            metrics['temporal_loss'] = temporal_loss.item()

        # 计算 mask 预测准确率
        with torch.no_grad():
            masked_pred_prob = torch.sigmoid(masked_pred)
            masked_pred_binary = (masked_pred_prob > 0.5).float()
            masked_target_binary = (masked_target > 0.5).float()
            mask_accuracy = (masked_pred_binary == masked_target_binary).float().mean()
            metrics['mask_accuracy'] = mask_accuracy.item()

        metrics['loss'] = total_loss.item()

        return total_loss, metrics

    def _spatial_consistency_loss(self, pred: Tensor, mask_indices: Tensor) -> Tensor:
        """空间一致性损失

        相邻 stabilizer 的预测应该平滑。
        这里简化实现：计算相邻 stabilizer 的预测差异。
        """
        # TODO: 实现完整的空间一致性损失（需要邻接信息）
        # 简化版本：L2 正则化
        return torch.tensor(0.0, device=pred.device)

    def _temporal_consistency_loss(self, pred: Tensor, mask_indices: Tensor) -> Tensor:
        """时间一致性损失

        同一 stabilizer 在相邻轮的预测应该连贯。
        """
        # TODO: 实现完整的时间一致性损失
        # 简化版本：L2 正则化
        return torch.tensor(0.0, device=pred.device)
