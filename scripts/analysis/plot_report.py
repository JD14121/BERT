#!/usr/bin/env python3
"""plot_report.py: 为实验报告生成科研图表（解析训练日志 + 结果 JSON）。
纯可视化--读取已审查验证的数据,不计算任何科学结果。遵循科研绘图原则:
清晰轴标签/单位、图例、合适标度(log)、关键值标注、一致配色、无 chartjunk。
"""
import re, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 120, 'savefig.bbox': 'tight', 'axes.axisbelow': True})
EXP = Path(__file__).resolve().parent
LOG = EXP.parent / "logs"
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
DISTS = [3, 5, 7]
C = {'mwpm': '#1f77b4', 'bert': '#d62728', 'alphaqubit': '#2ca02c'}
LBL = {'mwpm': 'MWPM', 'bert': 'BERT (Ours)', 'alphaqubit': 'AlphaQubit'}
MK = {'mwpm': 's', 'bert': 'o', 'alphaqubit': '^'}


def parse_runexp(path):
    """解析 run_experiment 日志为三阶段 (aq_pre/aq_ft/bert_ft), 含 pred_pos_rate。"""
    txt = Path(path).read_text(encoding='utf-8', errors='ignore')
    markers = [('aq_pre', 'AlphaQubit 合成监督预训练'), ('aq_ft', 'AlphaQubit 真机微调'),
               ('bert_ft', 'BERT (Ours)')]
    cuts = sorted([(txt.find(m), n) for n, m in markers if txt.find(m) >= 0])
    out = {}
    for i, (s, name) in enumerate(cuts):
        seg = txt[s: cuts[i + 1][0]] if i + 1 < len(cuts) else txt[s:]
        tr = re.findall(r'loss: ([\d.]+) \| accuracy: ([\d.]+)% \| pos_rate: [\d.]+% \| pred_pos_rate: ([\d.]+)%.*?step: (\d+)', seg)
        ev = re.findall(r'\[Eval\] step: (\d+) \| val_loss: ([\d.]+) \| val_acc: ([\d.]+)%', seg)
        out[name] = {
            't_step': np.array([int(t[3]) for t in tr], float),
            't_loss': np.array([float(t[0]) for t in tr]),
            't_acc': np.array([float(t[1]) for t in tr]),
            't_ppr': np.array([float(t[2]) for t in tr]),
            'v_step': np.array([int(e[0]) for e in ev], float),
            'v_loss': np.array([float(e[1]) for e in ev]),
            'v_acc': np.array([float(e[2]) for e in ev]),
        }
    return out


def parse_pretrain(path):
    txt = Path(path).read_text(encoding='utf-8', errors='ignore')
    tr = re.findall(r'main_loss: ([\d.]+) \|.*?mask_accuracy: ([\d.]+)%.*?step: (\d+)', txt)
    ev = re.findall(r'\[Eval\] step: (\d+) \| val_loss: ([\d.]+) \| val_mask_acc: ([\d.]+)%', txt)
    return {
        't_step': np.array([int(t[2]) for t in tr], float),
        't_loss': np.array([float(t[0]) for t in tr]),
        't_mask': np.array([float(t[1]) for t in tr]),
        'v_step': np.array([int(e[0]) for e in ev], float),
        'v_loss': np.array([float(e[1]) for e in ev]),
        'v_mask': np.array([float(e[2]) for e in ev]),
    }


# ===================== 数据加载 =====================
acc, ler, valid, per_round = {m: [] for m in C}, {m: [] for m in C}, {m: [] for m in C}, {}
for d in DISTS:
    s = json.load(open(EXP / f"results_summary_d{d}.json", encoding='utf-8'))
    for m in C: acc[m].append(s['results'][m]['accuracy'])
    r = json.load(open(EXP / f"results_ler_d{d}.json", encoding='utf-8'))[str(d)]
    per_round[d] = r
    for m in C:
        v = r[m]
        ler[m].append(v['ler'] if v else float('nan'))
        valid[m].append(v['is_valid'] if v else False)

runexp = {d: parse_runexp(LOG / f"run_experiment_d{d}.log") for d in DISTS}
pretr = {d: parse_pretrain(LOG / f"bert_pretrain_d{d}.log") for d in DISTS}

# ===================== Fig 1: Accuracy vs distance =====================
fig, ax = plt.subplots(figsize=(6, 4.2))
for m in ['mwpm', 'bert', 'alphaqubit']:
    ax.plot(DISTS, acc[m], MK[m] + '-', color=C[m], label=LBL[m], markersize=8, lw=2)
    for d, y in zip(DISTS, acc[m]):
        ax.annotate(f'{y:.3f}', (d, y), textcoords='offset points', xytext=(0, 9),
                    ha='center', fontsize=8, color=C[m])
ax.set_xlabel('code distance $d$'); ax.set_ylabel('test accuracy')
ax.set_title('Test accuracy vs code distance\n(Google real hard-readout, XZZX, r=10, Z basis)')
ax.set_xticks(DISTS); ax.set_ylim(0.65, 1.0); ax.legend(loc='lower right')
fig.savefig(FIG / "fig1_accuracy_vs_distance.png"); plt.close(fig)

# ===================== Fig 2: LER vs distance (log) =====================
fig, ax = plt.subplots(figsize=(6, 4.2))
for m in ['mwpm', 'bert', 'alphaqubit']:
    xs, ys = [], []
    for d, l, va in zip(DISTS, ler[m], valid[m]):
        if va and l > 0: xs.append(d); ys.append(l)
    ax.plot(xs, ys, MK[m] + '-', color=C[m], label=LBL[m], markersize=8, lw=2)
    for d, l, va in zip(DISTS, ler[m], valid[m]):
        if va and l > 0:
            ax.annotate(f'{l:.3f}', (d, l), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=8, color=C[m])
        else:
            ax.annotate('invalid\n(fit<2)', (d, 0.07), ha='center', fontsize=7.5,
                        color=C[m], fontstyle='italic')
ax.set_yscale('log'); ax.set_xlabel('code distance $d$'); ax.set_ylabel(r'LER $\varepsilon$ (logical error per round)')
ax.set_title('LER vs code distance\n(synthetic PAEMS data, rounds {1,10,13,30,50})')
ax.set_xticks(DISTS); ax.legend(loc='lower left')
fig.savefig(FIG / "fig2_ler_vs_distance.png"); plt.close(fig)

# ===================== Fig 3: BERT pretrain dynamics (d7) =====================
p = pretr[7]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(p['t_step'], p['t_mask'], '-', color=C['bert'], alpha=0.4, lw=1, label='train')
a1.plot(p['v_step'], p['v_mask'], 'o-', color=C['bert'], lw=1.8, ms=5, label='val')
a1.set_xlabel('training step'); a1.set_ylabel('mask accuracy (%)'); a1.set_title('(a) BERT pretrain mask accuracy (d7)')
a1.legend(); a1.set_ylim(55, 92)
a2.plot(p['t_step'], p['t_loss'], '-', color=C['bert'], alpha=0.4, lw=1, label='train')
a2.plot(p['v_step'], p['v_loss'], 'o-', color=C['bert'], lw=1.8, ms=5, label='val')
a2.set_xlabel('training step'); a2.set_ylabel('mask modeling loss'); a2.set_title('(b) BERT pretrain loss (d7)')
a2.legend()
fig.suptitle('BERT self-supervised pretraining dynamics (d7, 10000 steps)', y=1.02)
fig.savefig(FIG / "fig3_bert_pretrain_d7.png"); plt.close(fig)

# ===================== Fig 4: AlphaQubit d7 collapse+recovery (pretrain) =====================
aq = runexp[7]['aq_pre']
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(aq['v_step'], aq['v_acc'], 'o-', color=C['alphaqubit'], lw=1.8, ms=5, label='val accuracy')
ax.axhline(59.0, ls='--', color='gray', alpha=0.6, label='majority-class baseline (~59%)')
ax.axvspan(0, 3000, alpha=0.08, color='red', label='collapse phase (pred_pos=0%)')
ax.set_xlabel('pretrain step'); ax.set_ylabel('validation accuracy (%)', color=C['alphaqubit'])
ax.set_title('AlphaQubit d7 pretrain: all-negative collapse then recovery\n(synthetic supervised pretraining)')
ax.set_ylim(55, 70)
ax2 = ax.twinx()
ax2.plot(aq['t_step'], aq['t_ppr'], '.', color='purple', alpha=0.5, ms=4, label='pred_pos_rate')
ax2.set_ylabel('pred_pos_rate (%)', color='purple'); ax2.set_ylim(-2, 55)
lines, labels = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines + l2, labels + lab2, loc='lower right', fontsize=9)
fig.savefig(FIG / "fig4_aq_d7_collapse.png"); plt.close(fig)

# ===================== Fig 5: d7 finetune val_acc: AQ vs BERT =====================
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, m, ls in [('AlphaQubit', 'alphaqubit', 'o-'), ('BERT (Ours)', 'bert', 's-')]:
    ph = runexp[7]['aq_ft' if m == 'alphaqubit' else 'bert_ft']
    ax.plot(ph['v_step'], ph['v_acc'], ls, color=C[m], lw=1.8, ms=6, label=name)
ax.axhline(0.6982, ls=':', color=C['alphaqubit'], alpha=0.6)
ax.axhline(0.7438, ls=':', color=C['bert'], alpha=0.6)
ax.set_xlabel('finetune step'); ax.set_ylabel('validation accuracy (%)')
ax.set_title('d7 real-data finetune: BERT vs AlphaQubit\n(dashed = final test accuracy)')
ax.legend(loc='lower right')
fig.savefig(FIG / "fig5_d7_finetune.png"); plt.close(fig)

# ===================== Fig 6: LER E(n) vs rounds (faceted) =====================
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
ROUNDS = [1, 10, 13, 30, 50]
for ax, d in zip(axes, DISTS):
    for m in ['mwpm', 'bert', 'alphaqubit']:
        v = per_round[d][m]
        if v is None: continue
        es = [v['per_round'][str(n)]['E'] for n in ROUNDS]
        ax.plot(ROUNDS, es, MK[m] + '-', color=C[m], lw=1.8, ms=6, label=LBL[m])
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('rounds $n$'); ax.set_title(f'd={d}'); ax.set_xscale('log')
axes[0].set_ylabel('logical error rate $E(n)$')
axes[0].legend(fontsize=9)
fig.suptitle('Logical error rate accumulation vs rounds (LER protocol, NN trained at r=10, eval OOD)', y=1.02)
fig.savefig(FIG / "fig6_ler_error_vs_rounds.png"); plt.close(fig)

# ===================== Fig 7: OOD r=1 robustness + BERT advantage =====================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
# (a) E(r=1) across distances
for m in ['mwpm', 'bert', 'alphaqubit']:
    e1 = [per_round[d][m]['per_round']['1']['E'] if per_round[d][m] else float('nan') for d in DISTS]
    a1.plot(DISTS, e1, MK[m] + '-', color=C[m], lw=1.8, ms=7, label=LBL[m])
a1.axhline(0.5, ls='--', color='gray', alpha=0.5, label='random (0.5)')
a1.set_xlabel('code distance $d$'); a1.set_ylabel('$E(r{=}1)$ (1-round error)')
a1.set_title('(a) OOD robustness at r=1\n(model trained at r=10, eval at r=1)')
a1.set_xticks(DISTS); a1.legend(fontsize=9)
# (b) BERT advantage vs distance
dacc = [acc['bert'][i] - acc['alphaqubit'][i] for i in range(3)]
a2.plot(DISTS, [v * 100 for v in dacc], 'o-', color='black', lw=2, ms=8, label='$\\Delta$accuracy (BERT$-$AQ)')
a2.axhline(0, ls='--', color='gray', alpha=0.5)
for d, v in zip(DISTS, dacc):
    a2.annotate(f'{v*100:+.2f}pp', (d, v * 100), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9)
a2.set_xlabel('code distance $d$'); a2.set_ylabel('BERT $-$ AlphaQubit accuracy (pp)')
a2.set_title('(b) BERT advantage grows with distance')
a2.set_xticks(DISTS)
fig.savefig(FIG / "fig7_ood_advantage.png"); plt.close(fig)

print("Generated figures:")
for f in sorted(FIG.glob("fig*.png")):
    print(f"  {f.name}")
