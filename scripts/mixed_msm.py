"""MixedStructuredMSM: 重设计的掩码规则（用户要求）。
继承 MaskedSyndromeModeling（保留 _apply_mask 80/10/10 + get_targets target=measurement），
替换 _generate_mask_indices 为三种子策略混合：
  - 40% RandomTokenMask：原 BERT 式随机 token 掩码（局部推断）
  - 30% SpatialClusterMask：mask 空间相邻稳定子簇（XZZX 邻接，强制空间推断）
  - 30% TemporalSpanMask：mask 同一稳定子连续 k 轮（SpanBERT 风格，强制时序推断）
每子策略控制 ~mask_ratio 覆盖率。numpy 建掩码后转 device（避免 GPU 逐元素索引慢）。
"""
import numpy as np
import torch
from alphaqubit.models.pretrain import MaskedSyndromeModeling


class MixedStructuredMSM(MaskedSyndromeModeling):
    def __init__(self, mask_ratio: float = 0.25, coord_system=None,
                 p_random: float = 0.4, p_spatial: float = 0.3, p_temporal: float = 0.3,
                 cluster_radius: int = 1, span_len: int = 4,
                 use_full_round: bool = False, p_full_round: float = 0.2, variable_span: bool = False):
        super().__init__(mask_ratio=mask_ratio)
        self.coord_system = coord_system
        self.cluster_radius = cluster_radius
        self.span_len = span_len
        self.use_full_round = use_full_round          # ③-a 开关（默认 off，保 ① 可复现）
        self.variable_span = variable_span            # ③-a 变长 span 开关
        self._build_adjacency()
        # 概率归一化：use_full_round 时四策略（从 random/spatial/temporal 匀出 p_full_round）
        if use_full_round:
            r, s, t, f = p_random, p_spatial, p_temporal, p_full_round
            tot = r + s + t
            scale = (1.0 - f) / tot if tot > 0 else 0.0
            self.p = [r * scale, s * scale, t * scale, f]
            self.strats = ["random", "spatial", "temporal", "full_round"]
        else:
            self.p = [p_random, p_spatial, p_temporal]
            self.strats = ["random", "spatial", "temporal"]

    def _build_adjacency(self):
        """从 coord_system.stab_positions 建稳定子邻接（Manhattan 距离 <= cluster_radius）。"""
        if self.coord_system is None:
            self.neighbors = None
            return
        pos = self.coord_system.stab_positions_tensor.cpu().numpy()  # [n_stab, 2]
        n = len(pos)
        self.neighbors = []
        for i in range(n):
            d = np.abs(pos - pos[i]).sum(axis=1)        # Manhattan 距离
            self.neighbors.append(np.where(d <= self.cluster_radius)[0].tolist())

    def _generate_mask_indices(self, B, T, n_stab, device):
        mask = np.zeros((B, T, n_stab), dtype=bool)
        target_count = max(1, int(self.mask_ratio * T * n_stab))
        for b in range(B):
            strat = np.random.choice(self.strats, p=self.p)
            if strat == "random":
                mask[b] = np.random.rand(T, n_stab) < self.mask_ratio
            elif strat == "spatial":
                self._spatial_mask(mask[b], T, n_stab, target_count)
            elif strat == "temporal":
                self._temporal_mask(mask[b], T, n_stab, target_count)
            else:  # full_round（③-a）
                self._full_round_mask(mask[b], T, n_stab, target_count)
        return torch.from_numpy(mask).to(device)

    def _spatial_mask(self, m, T, n_stab, target_count):
        """mask 空间簇：随机选 (t, center)，mask center+邻居，直到 ~target_count。"""
        cnt = int(m.sum())
        guard = 0
        while cnt < target_count and guard < target_count * 4:
            t = np.random.randint(T)
            center = np.random.randint(n_stab)
            for s in self.neighbors[center]:
                if not m[t, s]:
                    m[t, s] = True; cnt += 1
                    if cnt >= target_count:
                        break
            guard += 1

    def _temporal_mask(self, m, T, n_stab, target_count):
        """mask 时序跨度：随机选 stab，mask 连续 span 轮，直到 ~target_count。
        ③-a: variable_span 时 span_len∈[2,8] 采样（否则固定 span_len）。"""
        cnt = int(m.sum())
        guard = 0
        while cnt < target_count and guard < target_count * 4:
            s = np.random.randint(n_stab)
            span = int(np.random.randint(2, 9)) if self.variable_span else self.span_len   # [2,8] 变长
            t0 = np.random.randint(0, max(1, T - span + 1))
            for t in range(t0, min(T, t0 + span)):
                if not m[t, s]:
                    m[t, s] = True; cnt += 1
                    if cnt >= target_count:
                        break
            guard += 1

    def _full_round_mask(self, m, T, n_stab, target_count):
        """③-a 整轮丢弃：mask 某 k 轮（k∈[1,3]）的全部稳定子，从邻轮推断（强于单 stab span）。"""
        cnt = int(m.sum())
        guard = 0
        while cnt < target_count and guard < target_count * 4:
            k = int(np.random.randint(1, 4))   # k∈[1,3]
            t0 = np.random.randint(0, max(1, T - k + 1))
            for t in range(t0, min(T, t0 + k)):
                for s in range(n_stab):
                    if not m[t, s]:
                        m[t, s] = True; cnt += 1
                        if cnt >= target_count:
                            break
                if cnt >= target_count:
                    break
            guard += 1
