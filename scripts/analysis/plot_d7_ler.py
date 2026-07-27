#!/usr/bin/env python3
"""plot_d7_ler.py (P5): 跨码距 accuracy + LER 对比可视化。
读 results_summary_d{3,5,7}.json (accuracy) + results_ler_d{3,5,7}.json (LER),
画两幅图: (1) accuracy vs distance  (2) LER vs distance (log, invalid 标注)。
纯可视化--不计算任何结果(结果已在JSON中,经eval_ler.py审查验证);仅读取绘图。
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent
FIG = EXP / "figures"; FIG.mkdir(exist_ok=True)
DISTS = [3, 5, 7]
MODELS = ['mwpm', 'bert', 'alphaqubit']
COLORS = {'mwpm': 'tab:blue', 'bert': 'tab:red', 'alphaqubit': 'tab:green'}
LABELS = {'mwpm': 'MWPM', 'bert': 'BERT (Ours)', 'alphaqubit': 'AlphaQubit'}

# ---- accuracy ----
acc = {m: [] for m in MODELS}
for d in DISTS:
    s = json.load(open(EXP/f"results_summary_d{d}.json", encoding='utf-8'))
    for m in MODELS:
        acc[m].append(s['results'][m]['accuracy'])

# ---- LER ----
ler, valid = {m: [] for m in MODELS}, {m: [] for m in MODELS}
for d in DISTS:
    r = json.load(open(EXP/f"results_ler_d{d}.json", encoding='utf-8'))[str(d)]
    for m in MODELS:
        v = r[m]
        if v is None:
            ler[m].append(float('nan')); valid[m].append(False)
        else:
            ler[m].append(v['ler']); valid[m].append(v['is_valid'])

# ---- Fig 1: accuracy vs distance ----
plt.figure(figsize=(6, 4))
for m in MODELS:
    plt.plot(DISTS, acc[m], 'o-', color=COLORS[m], label=LABELS[m], markersize=7)
for d in DISTS:
    for m in MODELS:
        plt.annotate(f'{acc[m][DISTS.index(d)]:.3f}', (d, acc[m][DISTS.index(d)]),
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=7, color=COLORS[m])
plt.xlabel('code distance d'); plt.ylabel('test accuracy')
plt.title('Accuracy vs distance (Google real hard-readout, XZZX, r10, Z)')
plt.xticks(DISTS); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(FIG/"accuracy_vs_distance.png", dpi=120); plt.close()

# ---- Fig 2: LER vs distance (log) ----
plt.figure(figsize=(6, 4))
for m in MODELS:
    ys = [l if (va and l > 0) else None for l, va in zip(ler[m], valid[m])]
    xs = [d for d, y in zip(DISTS, ys) if y is not None]
    yy = [y for y in ys if y is not None]
    plt.plot(xs, yy, 'o-', color=COLORS[m], label=LABELS[m], markersize=7)
    for d, l, va in zip(DISTS, ler[m], valid[m]):
        if va and l > 0:
            plt.annotate(f'{l:.3f}', (d, l), textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=7, color=COLORS[m])
        else:
            plt.annotate('invalid\n(fit<2)', (d, 0.06), ha='center', fontsize=7, color=COLORS[m])
plt.yscale('log'); plt.xlabel('code distance d'); plt.ylabel('LER (logical error per round)')
plt.title('LER vs distance (synthetic PAEMS data, rounds {1,10,13,30,50})')
plt.xticks(DISTS); plt.grid(True, alpha=0.3, which='both'); plt.legend()
plt.tight_layout(); plt.savefig(FIG/"ler_vs_distance.png", dpi=120); plt.close()

print(f"saved: {FIG}/accuracy_vs_distance.png, {FIG}/ler_vs_distance.png")
print("\n=== Accuracy (MWPM/BERT/AQ) ===")
for d in DISTS:
    print(f"  d{d}: {acc['mwpm'][DISTS.index(d)]:.4f} / {acc['bert'][DISTS.index(d)]:.4f} / {acc['alphaqubit'][DISTS.index(d)]:.4f}")
print("=== LER (MWPM/BERT/AQ) ===")
for d in DISTS:
    i = DISTS.index(d)
    def f(m): return f"{ler[m][i]:.6f}" if valid[m][i] else "INVALID"
    print(f"  d{d}: {f('mwpm')} / {f('bert')} / {f('alphaqubit')}")
