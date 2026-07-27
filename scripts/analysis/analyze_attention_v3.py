#!/usr/bin/env python3
"""analyze_attention.py v3: 忠实 patch + 完整分析 (审查组 v2 APPROVE_WITH_CONDITIONS 落实)

修正相对 v2:
  [bug1] patched_fwd 漏 LayerNorm + 残差 -> 测到的 attention 失真。
         v3 精确复刻真实 MultiHeadSelfAttention.forward (transformer.py L200-253) + 存权重。
  [bug2] last_attn_weights 每轮覆盖 -> 只剩第 9 轮。
         v3 用 _attn_history list 累积全部 10 轮，再对 (round, batch) 求平均。
  [bug3] json.dump 崩溃 numpy.bool_ 不可序列化。
         v3 显式 bool() + numpy-aware default。
  [gap]  未做预训练 vs 随机初始化 (QC#3)。v3 补齐。
  [caveat] QK^T-only 透明化 (残留项#1): zero learned_bias 实为 QK^T+距离先验 (distance_embed 仍在)。
         v3 另测 qk_pure (use_spatial_bias=False, 真·纯 QK^T) 作严格对照。

分析:
  1. attention 提取 [4 layers × 10 rounds × 8 heads × 48×48] (忠实 patch, list 累积)
  2. DEM 经验关联 [480×480] -> 折叠 [48×48] (10 轮同轮块平均, 假设轮间平稳 -- 残留项#2 声明)
  3. 分解: full / qk_pure / qk_no_learned / bias_only
  4. Pearson 相关 (排除对角线, Bonferroni α=0.05/32 -- 残留项#3: 分母=4层×8头=32)
  5. 预训练 vs 随机初始化 (QC#3)
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

os.environ['PYTHONPATH'] = '/root/beat_mwpm:/root/beat_mwpm/google_paems_data/code'
sys.path.insert(0, '/root/beat_mwpm')
sys.path.insert(0, '/root/beat_mwpm/google_paems_data/code')
sys.path.insert(0, '/root/beat_mwpm/google_paems_data/bert_experiment')

from path_config import DATA_DIR, CONFIG_DIR, GOOGLE_SC, GOOGLE_PATCH
from xzzx_coord import XZZXCoordinateSystem
import stim
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.models.transformer import MultiHeadSelfAttention
from alphaqubit.data.pt_dataset import PTBatchDataset

D, R, BASIS = 7, 10, 'Z'
N_STAB = D * D - 1          # 48
NUM_DET = R * N_STAB         # 480
EMBED, N_HEADS, N_LAYERS = 256, 8, 4
CKPT = '/root/beat_mwpm/google_paems_data/bert_experiment/checkpoints/bert_pretrain_d7/best.pt'
OUT = Path('/root/data/attention_analysis'); OUT.mkdir(parents=True, exist_ok=True)
B = 64   # batch (内存安全; 15GB RAM 实例)
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:6.1f}s] {msg}', flush=True)


# ---------- 1. 坐标系 + 模型 ----------
log('=== ⑤ Attention Analysis (d7, r10) — v3 faithful ===')
cs = XZZXCoordinateSystem(D, stim.Circuit.from_file(
    str(GOOGLE_SC / f'd{D}_at_{GOOGLE_PATCH[D]}' / BASIS / f'r{R:02d}' / 'circuit_ideal.stim')))
stab_positions = cs.stab_positions_tensor
log(f'n_stab={N_STAB}, num_det={NUM_DET}, stab_positions{tuple(stab_positions.shape)}')


def build_model(load_ckpt):
    m = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=N_HEADS,
                        num_transformer_layers=N_LAYERS, dropout=0.1)
    if load_ckpt:
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        log(f'pretrained model loaded, global_step={ck.get("global_step", "?")}')
    else:
        log('random-init model (same arch, no checkpoint)')
    m.eval()
    return m


# ---------- 2. 忠实 patch: 精确复刻真实 forward + 累积 10 轮 attention ----------
orig_fwd = MultiHeadSelfAttention.forward


def patched_fwd(self, x, stab_positions=None):
    """FAITHFUL copy of transformer.py MultiHeadSelfAttention.forward (L200-253).
    唯一新增: 把每轮 attn_weights 存入 self._attn_history (list)。"""
    B, n_stab, D_ = x.shape
    x_norm = self.layer_norm(x)                                    # ← v2 漏 (bug1)
    q = self.q_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.k_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.v_proj(x_norm).view(B, n_stab, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # self.scale = sqrt(head_dim)
    if self.use_spatial_bias:
        bias = self.spatial_bias(stab_positions)                   # [n_heads, n_stab, n_stab]
        attn_scores = attn_scores + bias.unsqueeze(0)
    attn_weights = F.softmax(attn_scores, dim=-1)
    self._attn_history.append(attn_weights.detach().clone())       # ← 累积每轮 (bug2 修复)
    attn_weights = self.dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, n_stab, D_)
    output = self.out_proj(attn_output)
    output = self.dropout(output)
    return x + output                                              # ← v2 漏残差 (bug1)


MultiHeadSelfAttention.forward = patched_fwd
log('MultiHeadSelfAttention patched (faithful)')


def get_attn_layers(model):
    """定位 transformer 层 (rnn_core.transformer.layers)."""
    obj = model.rnn_core.transformer.layers
    return [l for l in obj]


def reset_history(model):
    for l in get_attn_layers(model):
        l.self_attention._attn_history = []


def collect_avg(model):
    """每层: 对 (10 rounds, B) 求平均 -> [n_heads, 48, 48]。"""
    out = []
    for l in get_attn_layers(model):
        hist = l.self_attention._attn_history            # list[10] of [B,8,48,48]
        stacked = torch.stack(hist, dim=0)                # [10, B, 8, 48, 48]
        out.append(stacked.mean(dim=(0, 1)).cpu().numpy())  # [8, 48, 48]
    return out


# ---------- 3. 数据 ----------
ds = PTBatchDataset(str(DATA_DIR / 'd7' / f'ler_d{D}_r{R}_n20000_{BASIS}.pt'))
meas = torch.stack([ds[i]['measurement'] for i in range(B)])     # [B, T, 48]
event = torch.stack([ds[i]['event'] for i in range(B)])
lk = torch.zeros_like(meas)
el = torch.zeros_like(meas)
log(f'data loaded: meas{tuple(meas.shape)}, B={B}')

# ---------- 4. DEM 经验关联 [480,480] -> 折叠 [48,48] ----------
log('computing DEM empirical correlation...')
det = ds._data['detection_events'].numpy()
if det.ndim == 3:
    det = det.reshape(det.shape[0], -1)                  # [N, 480]
det = det[:, :NUM_DET].astype(np.float64)
n_zero_det = int((det.std(axis=0) == 0).sum())
dem_full = np.corrcoef(det.T)                            # [480, 480], 可能含 NaN
dem_spatial = np.zeros((N_STAB, N_STAB))
for r in range(R):
    blk = dem_full[r*N_STAB:(r+1)*N_STAB, r*N_STAB:(r+1)*N_STAB]
    dem_spatial += np.nan_to_num(blk, nan=0.0)
dem_spatial /= R
log(f'DEM spatial {dem_spatial.shape} (10-round within-block avg; assumes inter-round stationarity); '
    f'{n_zero_det}/480 never-firing detectors (NaN->0)')


# ---------- 5. 相关分析工具 ----------
mask = ~np.eye(N_STAB, dtype=bool)
dem_flat = dem_spatial[mask].flatten()
N_COMP = N_LAYERS * N_HEADS
ALPHA = 0.05 / N_COMP


def corr_per_head(attn_avg_list):
    """attn_avg_list: list[N_LAYERS] of [n_heads, 48, 48] -> list of dicts."""
    res = []
    for l in range(len(attn_avg_list)):
        for h in range(attn_avg_list[l].shape[0]):
            a = attn_avg_list[l][h][mask].flatten()
            if np.isnan(a).any() or np.isnan(dem_flat).any() or a.std() == 0:
                r_val, p_val = 0.0, 1.0
            else:
                r_val, p_val = pearsonr(a, dem_flat)
                if np.isnan(r_val):
                    r_val, p_val = 0.0, 1.0
            res.append({'layer': int(l), 'head': int(h),
                        'r': float(r_val), 'p': float(p_val), 'sig': bool(p_val < ALPHA)})
    return res


def set_use_bias(model, flag):
    for l in get_attn_layers(model):
        l.self_attention.use_spatial_bias = flag


def zero_learned_bias(model):
    saved = []
    for l in get_attn_layers(model):
        b = l.self_attention.spatial_bias.learned_bias
        saved.append(b.detach().clone())
        with torch.no_grad():
            b.zero_()
    return saved


def restore_learned_bias(model, saved):
    for l, s in zip(get_attn_layers(model), saved):
        with torch.no_grad():
            l.self_attention.spatial_bias.learned_bias.copy_(s)


def bias_only_list(model):
    """softmax(full bias) per layer -> [n_heads, 48, 48]."""
    out = []
    for l in get_attn_layers(model):
        with torch.no_grad():
            bias = l.self_attention.spatial_bias(stab_positions)   # [n_heads, 48, 48]
            out.append(F.softmax(bias, dim=-1).cpu().numpy())
    return out


# ---------- 6. 预训练模型: 四种 attention ----------
model = build_model(load_ckpt=True)
attn_layers = get_attn_layers(model)
log(f'found {len(attn_layers)} transformer layers @ rnn_core.transformer.layers')

# 6a. full
reset_history(model)
with torch.no_grad():
    _ = model(meas, event, lk, el)
attn_full = collect_avg(model)
log(f'full attention captured: {[a.shape for a in attn_full]}')

# 6b. qk_pure (use_spatial_bias=False)
set_use_bias(model, False)
reset_history(model)
with torch.no_grad():
    _ = model(meas, event, lk, el)
attn_qk_pure = collect_avg(model)
set_use_bias(model, True)
log('qk_pure (no bias) captured')

# 6c. qk_no_learned (zero learned_bias; = QK^T + distance prior)
saved_bias = zero_learned_bias(model)
reset_history(model)
with torch.no_grad():
    _ = model(meas, event, lk, el)
attn_qk_no_learned = collect_avg(model)
restore_learned_bias(model, saved_bias)
log('qk_no_learned (learned_bias=0, dist prior kept) captured')

# 6d. bias only
attn_bias = bias_only_list(model)
log('bias_only (softmax full bias) captured')


# ---------- 7. 相关计算 ----------
def run_corr(name, attn_list):
    res = corr_per_head(attn_list)
    rs = [x['r'] for x in res]
    n_sig = sum(x['sig'] for x in res)
    log(f'  {name:16s} mean r={np.mean(rs):+.4f}  max r={max(rs):+.4f}  '
        f'min r={min(rs):+.4f}  sig={n_sig}/{N_COMP} (α={ALPHA:.4f})')
    return res

log('=== Correlation (excl diagonal, Bonferroni α=0.05/32) ===')
res_full = run_corr('full', attn_full)
res_qk_pure = run_corr('qk_pure', attn_qk_pure)
res_qk_no_learned = run_corr('qk_no_learned', attn_qk_no_learned)
res_bias = run_corr('bias_only', attn_bias)


# ---------- 8. 预训练 vs 随机初始化 (QC#3) ----------
log('=== Pretrained vs Random init (QC#3) ===')
model_rand = build_model(load_ckpt=False)
reset_history(model_rand)
with torch.no_grad():
    _ = model_rand(meas, event, lk, el)
attn_full_rand = collect_avg(model_rand)
set_use_bias(model_rand, False)
reset_history(model_rand)
with torch.no_grad():
    _ = model_rand(meas, event, lk, el)
attn_qk_pure_rand = collect_avg(model_rand)
set_use_bias(model_rand, True)
res_full_rand = run_corr('random.full', attn_full_rand)
res_qk_pure_rand = run_corr('random.qk_pure', attn_qk_pure_rand)


# ---------- 9. 保存结果 (numpy-safe) ----------
def to_jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not serializable: {type(o)}')


summary = {
    'config': {'distance': D, 'rounds': R, 'basis': BASIS, 'n_stab': N_STAB,
               'num_det': NUM_DET, 'batch': B, 'embed_dim': EMBED, 'n_heads': N_HEADS,
               'n_layers': N_LAYERS, 'bonferroni_alpha': ALPHA, 'n_comparisons': N_COMP},
    'pretrained': {
        'full': {'mean_r': float(np.mean([x['r'] for x in res_full])),
                 'max_r': float(max(x['r'] for x in res_full)),
                 'n_sig': int(sum(x['sig'] for x in res_full))},
        'qk_pure': {'mean_r': float(np.mean([x['r'] for x in res_qk_pure])),
                    'max_r': float(max(x['r'] for x in res_qk_pure)),
                    'n_sig': int(sum(x['sig'] for x in res_qk_pure))},
        'qk_no_learned': {'mean_r': float(np.mean([x['r'] for x in res_qk_no_learned])),
                          'max_r': float(max(x['r'] for x in res_qk_no_learned))},
        'bias_only': {'mean_r': float(np.mean([x['r'] for x in res_bias])),
                      'max_r': float(max(x['r'] for x in res_bias))},
    },
    'random': {
        'full': {'mean_r': float(np.mean([x['r'] for x in res_full_rand])),
                 'max_r': float(max(x['r'] for x in res_full_rand))},
        'qk_pure': {'mean_r': float(np.mean([x['r'] for x in res_qk_pure_rand])),
                    'max_r': float(max(x['r'] for x in res_qk_pure_rand))},
    },
}
json.dump({'per_head': res_full}, open(OUT / 'correlation_full.json', 'w'),
          indent=2, default=to_jsonable)
json.dump({'per_head': res_qk_pure}, open(OUT / 'correlation_qk_pure.json', 'w'),
          indent=2, default=to_jsonable)
json.dump({'per_head': res_qk_no_learned}, open(OUT / 'correlation_qk_no_learned.json', 'w'),
          indent=2, default=to_jsonable)
json.dump({'per_head': res_bias}, open(OUT / 'correlation_bias_only.json', 'w'),
          indent=2, default=to_jsonable)
json.dump({'full': res_full_rand, 'qk_pure': res_qk_pure_rand},
          open(OUT / 'correlation_random.json', 'w'), indent=2, default=to_jsonable)
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2, default=to_jsonable)
log('JSON saved')


# ---------- 10. 可视化 ----------
best = max(res_full, key=lambda x: x['r'])

# fig1: best attention vs DEM
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(attn_full[best['layer']][best['head']], cmap='hot')
axes[0].set_title(f"Attention L{best['layer']}H{best['head']} (r={best['r']:.3f})"); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(dem_spatial, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('DEM spatial (10-round avg)'); plt.colorbar(im1, ax=axes[1])
axes[2].scatter(attn_full[best['layer']][best['head']][mask], dem_spatial[mask], alpha=0.3, s=3)
axes[2].set_xlabel('attention'); axes[2].set_ylabel('DEM corr'); axes[2].set_title(f'scatter r={best["r"]:.3f}')
plt.suptitle('⑤ Attention vs DEM (d7 XZZX, pretrained, faithful patch)')
plt.tight_layout(); plt.savefig(OUT / 'fig_attention_vs_dem.png', dpi=150); plt.close()

# fig2: per-head bar (full)
fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(res_full))
ax.bar(x, [r['r'] for r in res_full], color=['green' if r['sig'] else 'gray' for r in res_full])
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Layer × Head'); ax.set_ylabel('Pearson r (full attn vs DEM)')
ax.set_title(f'Per-head correlation (Bonferroni α={ALPHA:.4f})')
ax.set_xticks(list(x)); ax.set_xticklabels([f"L{r['layer']}H{r['head']}" for r in res_full], rotation=90, fontsize=6)
plt.tight_layout(); plt.savefig(OUT / 'fig_per_head.png', dpi=150); plt.close()

# fig3: decomposition grouped bar
fig, ax = plt.subplots(figsize=(14, 5))
width = 0.2
xp = np.arange(len(res_full))
ax.bar(xp - 1.5*width, [r['r'] for r in res_full], width, label='full')
ax.bar(xp - 0.5*width, [r['r'] for r in res_qk_pure], width, label='qk_pure (no bias)')
ax.bar(xp + 0.5*width, [r['r'] for r in res_qk_no_learned], width, label='qk+dist_prior (learned_bias=0)')
ax.bar(xp + 1.5*width, [r['r'] for r in res_bias], width, label='bias_only')
ax.axhline(0, color='black', lw=0.5); ax.legend(fontsize=8)
ax.set_xlabel('Layer × Head'); ax.set_ylabel('Pearson r vs DEM')
ax.set_title('Attention decomposition: full / qk_pure / qk+prior / bias_only')
ax.set_xticks(xp); ax.set_xticklabels([f"L{r['layer']}H{r['head']}" for r in res_full], rotation=90, fontsize=6)
plt.tight_layout(); plt.savefig(OUT / 'fig_decomposition.png', dpi=150); plt.close()

# fig4: pretrained vs random (qk_pure)
fig, ax = plt.subplots(figsize=(6, 6))
rp = [r['r'] for r in res_qk_pure]
rr = [r['r'] for r in res_qk_pure_rand]
ax.scatter(rp, rr, c='steelblue', s=40)
lim = [min(min(rp), min(rr)) - 0.02, max(max(rp), max(rr)) + 0.02]
ax.plot(lim, lim, 'k--', lw=0.8)
ax.set_xlabel('pretrained qk_pure r'); ax.set_ylabel('random qk_pure r')
ax.set_title(f'QC#3: pretrained vs random (qk_pure)\n'
             f'mean pre={np.mean(rp):+.4f} vs rand={np.mean(rr):+.4f}')
plt.tight_layout(); plt.savefig(OUT / 'fig_pretrained_vs_random.png', dpi=150); plt.close()
log('figures saved')


# ---------- 11. 结论 ----------
log('\n=== SUMMARY ===')
log(f"full       mean r={summary['pretrained']['full']['mean_r']:+.4f}  "
    f"max r={summary['pretrained']['full']['max_r']:+.4f}  sig={summary['pretrained']['full']['n_sig']}/{N_COMP}")
log(f"qk_pure    mean r={summary['pretrained']['qk_pure']['mean_r']:+.4f}  "
    f"max r={summary['pretrained']['qk_pure']['max_r']:+.4f}")
log(f"bias_only  mean r={summary['pretrained']['bias_only']['mean_r']:+.4f}  "
    f"max r={summary['pretrained']['bias_only']['max_r']:+.4f}")
log(f"random.full    mean r={summary['random']['full']['mean_r']:+.4f}")
log(f"random.qk_pure mean r={summary['random']['qk_pure']['mean_r']:+.4f}")
log(f"QC#1 (qk_pure r > 0): {'PASS' if summary['pretrained']['qk_pure']['mean_r'] > 0 else 'FAIL'}")
log(f"QC#2 (qk_pure > bias_only): "
    f"{'PASS' if summary['pretrained']['qk_pure']['mean_r'] > summary['pretrained']['bias_only']['mean_r'] else 'FAIL'}")
log(f"QC#3 (pretrained qk_pure > random qk_pure): "
    f"{'PASS' if summary['pretrained']['qk_pure']['mean_r'] > summary['random']['qk_pure']['mean_r'] else 'FAIL'}")

# restore
MultiHeadSelfAttention.forward = orig_fwd
log(f'\n=== DONE ({time.time()-T0:.1f}s) ===  output: {OUT}')
print(json.dumps(summary, indent=2))
