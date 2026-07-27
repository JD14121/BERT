#!/usr/bin/env python3
"""plot_cloud_d7.py: 绘制云端 d7（34M）实验图 + 跨码距对比（d3/d5 本地 + d7 云）。
- fig1: accuracy vs distance (MWPM/BERT/AQ)
- fig2: LER vs distance (log y)
- fig3: d7 BERT pretrain mask_acc/loss 曲线
- fig4: d7 BERT finetune val_acc 曲线
"""
import json, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/bert_experiment")
CD7 = BASE / "cloud_d7"
FIG = CD7 / "figures"; FIG.mkdir(parents=True, exist_ok=True)

def load_json(p):
    p = Path(p)
    if not p.exists(): return None
    return json.load(open(p, encoding='utf-8'))

# === accuracy / LER 数据 ===
dists = [3, 5, 7]
acc = {'mwpm':[], 'bert':[], 'alphaqubit':[]}
ler = {'mwpm':[], 'bert':[], 'alphaqubit':[]}
# d3/d5 本地 E1
for d in [3,5]:
    s = load_json(BASE/f"results_summary_d{d}_E1.json")
    l = load_json(BASE/f"results_ler_d{d}_E1.json")
    for m in ['mwpm','bert','alphaqubit']:
        acc[m].append(s['results'][m]['accuracy'])
        ler[m].append(l[str(d)][m]['ler'] if l and l[str(d)][m] else None)
# d7 云
s7 = load_json(CD7/"results_summary_d7_E1_cloud.json")
l7 = load_json(CD7/"results_ler_d7_E1_cloud.json")
for m in ['mwpm','bert','alphaqubit']:
    acc[m].append(s7['results'][m]['accuracy'])
    ler[m].append(l7['7'][m]['ler'] if l7['7'][m] else None)

# === fig1: accuracy vs distance ===
plt.figure(figsize=(7,5))
plt.plot(dists, acc['mwpm'], 'o-', label='MWPM', color='tab:green', lw=2)
plt.plot(dists, acc['bert'], 's-', label='BERT (Ours)', color='tab:blue', lw=2)
plt.plot(dists, acc['alphaqubit'], '^-', label='AlphaQubit (from-scratch)', color='tab:red', lw=2)
# 标注 d7 数据规模
plt.annotate('d7: 34M (cloud)', (7, acc['bert'][2]), textcoords="offset points", xytext=(5,8), fontsize=9, color='tab:blue')
plt.annotate('d3/d5: 2×/10× (local)', (3, acc['bert'][0]), textcoords="offset points", xytext=(5,-15), fontsize=9, color='tab:blue')
plt.xlabel('Code distance d'); plt.ylabel('Test accuracy (real hard-readout)')
plt.title('Accuracy vs Code Distance (big model, 50% synth doping)')
plt.ylim(0.55, 1.0); plt.grid(alpha=0.3); plt.legend()
plt.savefig(FIG/"fig1_accuracy_vs_distance.png", dpi=120, bbox_inches='tight'); plt.close()
print("fig1 saved")

# === fig2: LER vs distance (log) ===
plt.figure(figsize=(7,5))
for m,col,mk in [('mwpm','tab:green','o'),('bert','tab:blue','s'),('alphaqubit','tab:red','^')]:
    ys = [ler[m][i] for i in range(3) if ler[m][i] is not None and ler[m][i]>0]
    xs = [dists[i] for i in range(3) if ler[m][i] is not None and ler[m][i]>0]
    plt.plot(xs, ys, mk+'-', label=m, color=col, lw=2)
plt.yscale('log')
plt.xlabel('Code distance d'); plt.ylabel('LER ε (logical error per round)')
plt.title('LER vs Code Distance (synthetic soft-readout)')
plt.grid(alpha=0.3, which='both'); plt.legend()
plt.savefig(FIG/"fig2_ler_vs_distance.png", dpi=120, bbox_inches='tight'); plt.close()
print("fig2 saved")

# === fig3: d7 BERT pretrain mask_acc/loss 曲线 ===
def parse_log(path, keys):
    """从日志提取 (step, val) 序列。keys: list of (regex, name)."""
    out = {k[1]: [] for k in keys}
    steps = []
    step_re = re.compile(r'step: (\d+)')
    val_res = [(re.compile(p), n) for p,n in keys]
    cur_step = 0
    for line in open(path, encoding='utf-8', errors='ignore'):
        ms = step_re.search(line)
        if ms: cur_step = int(ms.group(1))
        for rgx,n in val_res:
            m = rgx.search(line)
            if m:
                try:
                    out[n].append((cur_step, float(m.group(1))))
                except: pass
    return out

bp = CD7/"train_d7_bert_pretrain.log"
if bp.exists():
    d = parse_log(bp, [(r'val_mask_acc: ([\d.]+)', 'val_mask_acc'), (r'val_loss: ([\d.]+)', 'val_loss')])
    plt.figure(figsize=(8,5))
    if d['val_mask_acc']:
        xs,ys = zip(*d['val_mask_acc']); plt.plot(xs, ys, '.-', label='val mask_acc', color='tab:blue')
    if d['val_loss']:
        xs,ys = zip(*d['val_loss']); plt.plot(xs, [1-y for y in ys], '.-', label='1-val_loss (proxy)', color='tab:orange', alpha=0.5)
    plt.xlabel('step'); plt.ylabel('val mask_acc'); plt.title('Cloud d7 BERT pretrain (34M data, big model)')
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig(FIG/"fig3_d7_bert_pretrain.png", dpi=120, bbox_inches='tight'); plt.close()
    print("fig3 saved")

# === fig4: d7 BERT finetune val_acc 曲线 ===
rf = CD7/"train_d7_run_experiment.log"
if rf.exists():
    # BERT finetune 段：找 "BERT" 之后的 val_acc
    lines = open(rf, encoding='utf-8', errors='ignore').read().split('\n')
    bert_start = next((i for i,l in enumerate(lines) if 'BERT' in l and 'finetune' in l.lower()), 0)
    bert_lines = lines[bert_start:]
    d = parse_log_lines = []
    step_re = re.compile(r'step: (\d+)'); acc_re = re.compile(r"val_acc.: ([\d.]+)|'accuracy': ([\d.]+)")
    pts = []
    cur = 0
    for l in bert_lines:
        ms = step_re.search(l)
        if ms: cur = int(ms.group(1))
        ma = re.search(r"'accuracy': ([\d.]+)", l)
        if ma:
            pts.append((cur, float(ma.group(1))))
    if pts:
        xs,ys = zip(*pts)
        plt.figure(figsize=(8,5))
        plt.plot(xs, ys, '.-', color='tab:blue')
        plt.xlabel('step'); plt.ylabel('val accuracy'); plt.title('Cloud d7 BERT finetune (real 40k + 50% synth)')
        plt.grid(alpha=0.3)
        plt.savefig(FIG/"fig4_d7_bert_finetune.png", dpi=120, bbox_inches='tight'); plt.close()
        print("fig4 saved")

print("ALL FIGURES DONE ->", FIG)
