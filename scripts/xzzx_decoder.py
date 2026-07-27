"""XZZX 适配的 decoder：替换 LateFusion 中旋转码专属的 StabToDataConv（2×2 卷积，
假设 data 在 2×2 stab 中心，XZZX 不成立）为 XZZXStabToData（pool+广播，几何无关）。

LateFusion 输出 [B,n_data,D] 之后会被 Global Mean Pool 池化为 [B,D]，故精确的
stab->data 空间映射非关键；stab 的空间信息已由 RNNCore（SpatialAttentionBias+DilatedConv，
二者在 XZZX 网格上正常工作）编码。XZZXStabToData 把 stab 特征 pool 后投影广播到 n_data。
"""
import torch
import torch.nn as nn
from alphaqubit.models.decoder import AlphaQubitDecoder
from alphaqubit.models.pretrain_decoder import FineTuneDecoder


class XZZXStabToData(nn.Module):
    """XZZX stab->data 映射（替换旋转码 2×2 StabToDataConv）。pool stab -> 投影 -> 广播到 n_data。"""
    def __init__(self, in_dim: int, out_dim: int, n_data: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.n_data = n_data

    def forward(self, stab_features):  # [B, n_stab, in_dim]
        pooled = stab_features.mean(dim=1)          # [B, in_dim]
        out = self.proj(pooled)                      # [B, out_dim]
        out = self.act(self.norm(out))               # [B, out_dim]
        return out.unsqueeze(1).expand(-1, self.n_data, -1)  # [B, n_data, out_dim]


def _patch_late_fusion(model, embed_dim, n_data):
    """替换 model.late_fusion.stab_to_data 为 XZZX 版本（若有 late_fusion）。"""
    if getattr(model, "late_fusion", None) is not None:
        # LateFusion: stab_to_data out_channels = output_dim = embed_dim
        model.late_fusion.stab_to_data = XZZXStabToData(embed_dim, embed_dim, n_data)
    return model


class XZZXAlphaQubitDecoder(AlphaQubitDecoder):
    """AlphaQubitDecoder + XZZX LateFusion 适配。其余继承（SyndromeEmbedder/RNNCore 用 XZZX 坐标系）。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _patch_late_fusion(self, self.embed_dim, self.n_data)


class XZZXFineTuneDecoder(FineTuneDecoder):
    """FineTuneDecoder（BERT 微调）+ XZZX LateFusion 适配。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # FineTuneDecoder 的 readout 可能用 AlphaQubitDecoder 的 LateFusion 结构
        _patch_late_fusion(self, self.embed_dim, getattr(self, "n_data", None) or self.coord_system.n_data)
