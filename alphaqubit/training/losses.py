"""
损失函数模块

这个模块实现了AlphaQubit训练中使用的损失函数。

主要损失函数：
1. BCEWithLogitsLoss: 主要的分类损失
   - 输入: logit (未经sigmoid)
   - 自动包含sigmoid，数值更稳定

为什么用BCE？
- 这是一个二分类问题（逻辑错误 vs 无错误）
- BCEWithLogitsLoss = sigmoid + BCELoss
- 数值稳定性更好

损失计算：
    L = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]

其中：
    - x: 模型输出的logit
    - y: 真实标签 (0或1)
    - σ: sigmoid函数

期望值：
- 随机初始化时，loss ≈ ln(2) ≈ 0.693
- 训练后loss应该明显下降
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DecoderLoss(nn.Module):
    """解码器损失函数

    包含主损失（BCE）和可选的辅助损失。

    支持的辅助损失：
    - Label Smoothing: 软化标签，防止过拟合
    - Focal Loss: 关注难分类样本

    使用方式：
        loss_fn = DecoderLoss(label_smoothing=0.1)
        loss, metrics = loss_fn(logits, labels)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        pos_weight: Optional[float] = None,
    ):
        """初始化损失函数

        Args:
            label_smoothing: 标签平滑系数 (0-1)
                           0: 不使用标签平滑
                           0.1: 常用值，将0/1软化为0.05/0.95
            focal_gamma: Focal loss的gamma参数
                        0: 不使用focal loss（退化为BCE）
                        2: 常用值，增加难样本权重
            pos_weight: 正样本权重，用于处理类别不平衡
                       None: 不加权
                       >1: 增加正样本权重
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

        # BCE损失
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Dict[str, float]]:
        """计算损失

        Args:
            logits: 模型输出 [B, 1] 或 [B]
            labels: 真实标签 [B, 1] 或 [B]

        Returns:
            Tuple of:
                - loss: 标量损失值
                - metrics: 包含各种指标的字典
        """
        # 确保形状一致
        if logits.dim() == 2:
            logits = logits.squeeze(-1)
        if labels.dim() == 2:
            labels = labels.squeeze(-1)

        # ==================== 标签平滑 ====================
        if self.label_smoothing > 0:
            # 将标签软化：0 → eps, 1 → 1-eps
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # ==================== BCE损失 ====================
        if self.focal_gamma > 0:
            # Focal Loss
            loss = self._focal_loss(logits, labels)
        else:
            # 标准BCE
            loss = F.binary_cross_entropy_with_logits(
                logits, labels,
                pos_weight=self.pos_weight,
            )

        # ==================== 计算指标 ====================
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # 准确率
            accuracy = (preds == labels.round()).float().mean()

            # 正样本比例
            pos_rate = labels.mean()
            pred_pos_rate = preds.mean()

            # Logit统计
            logit_mean = logits.mean()
            logit_std = logits.std()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "pos_rate": pos_rate.item(),
            "pred_pos_rate": pred_pos_rate.item(),
            "logit_mean": logit_mean.item(),
            "logit_std": logit_std.item(),
        }

        return loss, metrics

    def _focal_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Focal Loss实现

        Focal Loss = -α * (1-p)^γ * log(p)

        γ > 0时，对于高置信度的样本降低权重，
        使模型更关注难分类的样本。

        Args:
            logits: [B]
            labels: [B]

        Returns:
            标量损失
        """
        # 计算概率
        probs = torch.sigmoid(logits)

        # 计算CE
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )

        # 计算focal权重
        # 对于正样本：p_t = probs
        # 对于负样本：p_t = 1 - probs
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.focal_gamma

        # 加权损失
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class BalancedBCELoss(nn.Module):
    """平衡BCE损失

    自动计算正负样本权重以处理类别不平衡。

    在量子纠错中，逻辑错误可能是少数类。
    这个损失函数自动平衡正负样本的贡献。
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """计算平衡BCE损失

        Args:
            logits: [B] 或 [B, 1]
            labels: [B] 或 [B, 1]

        Returns:
            标量损失
        """
        if logits.dim() == 2:
            logits = logits.squeeze(-1)
        if labels.dim() == 2:
            labels = labels.squeeze(-1)

        # 计算正负样本数量
        num_pos = labels.sum()
        num_neg = labels.numel() - num_pos

        # 防止除零
        if num_pos == 0 or num_neg == 0:
            return F.binary_cross_entropy_with_logits(logits, labels)

        # 计算权重：少数类获得更高权重
        pos_weight = num_neg / num_pos

        return F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=torch.tensor([pos_weight], device=logits.device),
        )


def compute_calibration_metrics(
    probs: Tensor,
    labels: Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """计算校准指标

    检查模型的预测概率是否与实际正确率一致。
    理想情况下，预测概率为p的样本中应有p比例是正确的。

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        num_bins: 分箱数量

    Returns:
        校准指标字典
    """
    # 确保是1D
    probs = probs.flatten()
    labels = labels.flatten()

    # 分箱
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    bin_indices = torch.bucketize(probs, bin_boundaries[1:-1])

    # 计算每个bin的统计
    total_samples = 0
    calibration_error = 0.0

    for i in range(num_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]

        # bin内平均预测概率
        avg_confidence = bin_probs.mean()
        # bin内实际正样本比例
        avg_accuracy = bin_labels.mean()

        # 累计误差（加权）
        bin_size = mask.sum().item()
        total_samples += bin_size
        calibration_error += bin_size * abs(avg_confidence - avg_accuracy).item()

    # 期望校准误差 (ECE)
    ece = calibration_error / total_samples if total_samples > 0 else 0.0

    return {
        "ece": ece,  # Expected Calibration Error
        "num_samples": total_samples,
    }
