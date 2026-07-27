#!/usr/bin/env python3
"""plot_final.py: 最终绘图 - 含 125M+100k d7 结果 + 跨码距/数据规模对比"""
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
    return json.load(open(p, encoding='utf-8')) if p.exists() else None

# === 数据收集 ===
# d3/d5 本地 E1
d3s = load_json(BASE/"results_summary_d3_E1.json"); d3l = load_json(BASE/"results_ler_d3_E1.json")
d5s = load_json(BASE/"results_summary_d5_E1.json"); d5l = load_json(BASE/"results_ler_d5_E1.json")
# d7 34M 云端 E1
d7_34s = load_json(CD7/"results_summary_d7_E1_cloud.json"); d7_34l = load_json(CD7/"results_ler_d7_E1_cloud.json")
# d7 125M+100k 云端
d7_125s = load_json(CD7/"results_summary_d7_E1_125M_100k.json"); d7_125l = load_json(CD7/"results_ler_d7_E1_125M_100k.json")

# === fig1: accuracy vs distance (d3/d5/d7-34M/d7-125M) ===
plt.figure(figsize=(8,5))
dists_labels = ['d3\n(2×)', 'd5\n(10×)', 'd7\n(34M)', 'd7\n(125M+100k)']
mwpm_acc = [d3s['results']['mwpm']['accuracy'], d5s['results']['mwpm']['accuracy'], d7_34s['results']['mwpm']['accuracy'], d7_125s['results']['mwpm']['accuracy']]
bert_acc = [d3s['results']['bert']['accuracy'], d5s['results']['bert']['accuracy'], d7_34s['results']['bert']['accuracy'], d7_125s['results']['bert']['accuracy']]
aq_acc = [d3s['results']['alphaqubit']['accuracy'], d5s['results']['alphaqubit']['accuracy'], d7_34s['results']['alphaqubit']['accuracy'], d7_125s['results']['alphaqubit']['accuracy']]
x = range(len(dists_labels))
plt.plot(x, mwpm_acc, 'o-', label='MWPM', color='tab:green', lw=2)
plt.plot(x, bert_acc, 's-', label='BERT (Ours)', color='tab:blue', lw=2)
plt.plot(x, aq_acc, '^-', label='AlphaQubit', color='tab:red', lw=2)
plt.xticks(x, dists_labels); plt.ylabel('Test accuracy (real hard-readout)'); plt.ylim(0.55,1.0)
plt.title('Accuracy: BERT vs MWPM vs AlphaQubit (big model, 50% doping)')
plt.grid(alpha=0.3); plt.legend()
plt.savefig(FIG/"fig_final_accuracy.png", dpi=120, bbox_inches='tight'); plt.close(); print("fig1 saved")

# === fig2: LER vs distance (log) ===
plt.figure(figsize=(8,5))
mwpm_ler = [d3l['3']['mwpm']['ler'], d5l['5']['mwpm']['ler'], d7_34l['7']['mwpm']['ler'], d7_125l['7']['mwpm']['ler']]
bert_ler = [d3l['3']['bert']['ler'], d5l['5']['bert']['ler'], d7_34l['7']['bert']['ler'], d7_125l['7']['bert']['ler']]
plt.plot(x, mwpm_ler, 'o-', label='MWPM', color='tab:green', lw=2)
plt.plot(x, bert_ler, 's-', label='BERT (Ours)', color='tab:blue', lw=2)
plt.yscale('log'); plt.xticks(x, dists_labels); plt.ylabel('LER ε (logical error per round)')
plt.title('LER: BERT vs MWPM (log scale)')
plt.grid(alpha=0.3, which='both'); plt.legend()
plt.savefig(FIG/"fig_final_ler.png", dpi=120, bbox_inches='tight'); plt.close(); print("fig2 saved")

# === fig3: d7 数据规模效应 (accuracy + LER) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
scales = ['1×\n(orig\nsmall)', '34M\n(big\n20k)', '125M\n(big\n100k)']
# accuracy: 1× orig 0.7438, 34M 0.8702, 125M 0.8664
acc_d7 = [0.7438, d7_34s['results']['bert']['accuracy'], d7_125s['results']['bert']['accuracy']]
ax1.bar(scales, acc_d7, color='tab:blue'); ax1.set_ylabel('BERT d7 accuracy'); ax1.set_ylim(0.7, 0.9)
ax1.set_title('d7 BERT accuracy vs data scale'); ax1.grid(alpha=0.3)
for i,v in enumerate(acc_d7): ax1.text(i, v+0.005, f'{v:.4f}', ha='center', fontsize=9)
# LER: 1× orig 0.0402, 34M 0.0181, 125M 0.0137
ler_d7 = [0.0402, d7_34l['7']['bert']['ler'], d7_125l['7']['bert']['ler']]
ax2.bar(scales, ler_d7, color='tab:orange'); ax2.set_ylabel('BERT d7 LER ε'); ax2.set_title('d7 BERT LER vs data scale')
for i,v in enumerate(ler_d7): ax2.text(i, v+0.0005, f'{v:.4f}', ha='center', fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIG/"fig_final_d7_scale.png", dpi=120, bbox_inches='tight'); plt.close(); print("fig3 saved")

# === fig4: 100k pretrain mask_acc 曲线 ===
bp = CD7 / "e1_125M_100k_bert_pretrain.log"
if bp.exists():
    val_accs = []; steps = []
    for line in open(bp, encoding='utf-8', errors='ignore'):
        m = re.search(r"val_mask_acc.: ([\d.]+).*step: (\d+)", line)
        if m: val_accs.append(float(m.group(1))); steps.append(int(m.group(2)))
    if val_accs:
        plt.figure(figsize=(8,4))
        plt.plot(steps, val_accs, '.-', color='tab:blue', markersize=3)
        plt.xlabel('step'); plt.ylabel('val mask_acc'); plt.title('d7 BERT 100k pretrain (125M data, big model)')
        plt.grid(alpha=0.3); plt.axhline(y=0.885, color='r', linestyle='--', label='88.5% ceiling')
        plt.legend()
        plt.savefig(FIG/"fig_final_100k_pretrain.png", dpi=120, bbox_inches='tight'); plt.close(); print("fig4 saved")

print("ALL FIGURES DONE ->", FIG)
