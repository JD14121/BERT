#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_latest_results.py
为 EXPERIMENT_REPORT 最新结果（d5 opt/d5 focal/d7 focal）生成图像。
- 不向 LLM 输入图像，仅生成 PNG 存 figures/
- 英文标签（避中文字体问题），报告正文用中文说明
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"D:\Code\LZai\Ai for QEC\Alpha-qubit\code\alphaquibit-main\alphaquibit-main\google_paems_data\bert_experiment\figures"
os.makedirs(OUT, exist_ok=True)

# ============ 数据（全部已云端核实）============
# d5
d5_labels = ['Old small\n(1.64M)', 'd5 opt\nBCE (12M)', 'd5 focal\nγ=2 (12M)', 'MWPM']
d5_acc   = [0.7980, 0.9338, 0.9378, 0.9428]
d5_ler   = [0.0267, 0.006232, 0.005610, 0.003534]
# d5 per-round E(n)
rounds = [1, 10, 13, 30, 50]
d5_opt_E   = [0.05195, 0.1139, 0.13195, 0.19555, 0.2621]
d5_focal_E = [0.0551, 0.11365, 0.12695, 0.1843, 0.24925]
d5_mwpm_E  = [0.00055, 0.0295, 0.04035, 0.0948, 0.14605]
# d7
d7_r5_E    = [0.08595, 0.18625, 0.20485, 0.3133, 0.39495]
d7_focal_E = [0.91405, 0.35645, 0.58615, 0.49155, 0.5003]
d7_mwpm_E  = [0.0004, 0.023, 0.0284, 0.07265, 0.11505]
# focal effect
focal_acc_change = {'d5': 0.9378-0.9338, 'd7': 0.669-0.8664}  # +0.40pp, -19.74pp

# ============ fig8: d5 优化进展（acc + LER）============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#888888', '#2196F3', '#4CAF50', '#F44336']
bars1 = ax1.bar(d5_labels, [a*100 for a in d5_acc], color=colors)
ax1.set_ylabel('Test Accuracy (%)')
ax1.set_title('(a) d5 BERT Accuracy Progression')
ax1.set_ylim(75, 96)
ax1.axhline(94.28, color='#F44336', linestyle='--', alpha=0.5, label='MWPM')
for b, v in zip(bars1, d5_acc):
    ax1.text(b.get_x()+b.get_width()/2, v*100+0.3, f'{v*100:.2f}', ha='center', fontsize=9)
ax1.legend(loc='lower right', fontsize=8)

bars2 = ax2.bar(d5_labels, d5_ler, color=colors)
ax2.set_ylabel('LER ε (log scale)')
ax2.set_yscale('log')
ax2.set_title('(b) d5 BERT LER Progression')
ax2.axhline(0.003534, color='#F44336', linestyle='--', alpha=0.5, label='MWPM 0.00353')
for b, v in zip(bars2, d5_ler):
    ax2.text(b.get_x()+b.get_width()/2, v*1.08, f'{v:.5f}', ha='center', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)

plt.suptitle('Fig 8 | d5 BERT optimization: large model + 160M data (opt) then focal loss (γ=2)\n'
             'Old→opt: +13.6pp acc, -76.6% LER; opt→focal: +0.40pp acc, -9.98% LER (borderline)', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUT, 'fig8_d5_opt_focal_progression.png'), dpi=130, bbox_inches='tight')
plt.close()
print('[OK] fig8_d5_opt_focal_progression.png')

# ============ fig9: focal 效应 d5 vs d7 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
# (a) acc change
ax1.bar(['d5 (easier)', 'd7 (harder)'], [focal_acc_change['d5']*100, focal_acc_change['d7']*100],
        color=['#4CAF50', '#F44336'])
ax1.set_ylabel('Δ Accuracy vs BCE baseline (pp)')
ax1.set_title('(a) focal γ=2 effect on Accuracy')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.text(0, 0.5, '+0.40\n(stable)', ha='center', fontsize=10, color='#2E7D32')
ax1.text(1, -10, '-19.74\n(CRASH)', ha='center', fontsize=10, color='#B71C1C')
# (b) LER change
ax2.bar(['d5', 'd7'], [-9.98, 0], color=['#4CAF50', '#F44336'])
ax2.bar([1], [100], width=0.6, color='#F44336', alpha=0.3, hatch='//')
ax2.text(0, 5, '-9.98%\n(borderline\nPASS)', ha='center', fontsize=10, color='#2E7D32')
ax2.text(1, 50, 'INVALID\n(fit 1/5,\nmodel broken)', ha='center', fontsize=10, color='#B71C1C')
ax2.set_ylabel('Δ LER vs BCE baseline (%)')
ax2.set_title('(b) focal γ=2 effect on LER')
ax2.set_ylim(-30, 110)
ax2.axhline(0, color='black', linewidth=0.8)

plt.suptitle('Fig 9 | focal loss is task-difficulty dependent: helps d5 (weakly) but crashes d7\n'
             'd7 is harder (97 nodes, 40k real data) — focal over-down-weights easy samples, model loses anchor', fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUT, 'fig9_focal_d5_vs_d7.png'), dpi=130, bbox_inches='tight')
plt.close()
print('[OK] fig9_focal_d5_vs_d7.png')

# ============ fig10: d7 逐轮 LER（R5/focal/MWPM）============
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rounds, d7_r5_E, 'o-', color='#2196F3', label='d7 R5 BCE (baseline, LER=0.01366, valid)', linewidth=2, markersize=8)
ax.plot(rounds, d7_focal_E, 's--', color='#F44336', label='d7 focal γ=2 (LER INVALID, fit 1/5)', linewidth=2, markersize=8)
ax.plot(rounds, d7_mwpm_E, '^:', color='black', label='MWPM (LER=0.00268)', linewidth=2, markersize=8)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='random (E=0.5)')
ax.set_xscale('log')
ax.set_xlabel('Rounds n (log scale)')
ax.set_ylabel('Logical Error Rate E(n)')
ax.set_title('Fig 10 | d7 per-round LER: focal γ=2 crashes (erratic, r=1 E=0.914)\n'
             'vs R5 BCE (sane monotonic) vs MWPM (lowest)')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-0.02, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig10_d7_focal_crash_ler.png'), dpi=130, bbox_inches='tight')
plt.close()
print('[OK] fig10_d7_focal_crash_ler.png')

# ============ fig11: d5 逐轮 LER（opt/focal/MWPM）============
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rounds, d5_opt_E, 'o-', color='#2196F3', label='d5 opt BCE (LER=0.00623)', linewidth=2, markersize=8)
ax.plot(rounds, d5_focal_E, 's--', color='#4CAF50', label='d5 focal γ=2 (LER=0.00561, -9.98%)', linewidth=2, markersize=8)
ax.plot(rounds, d5_mwpm_E, '^:', color='black', label='MWPM (LER=0.00353)', linewidth=2, markersize=8)
ax.set_xscale('log')
ax.set_xlabel('Rounds n (log scale)')
ax.set_ylabel('Logical Error Rate E(n)')
ax.set_title('Fig 11 | d5 per-round LER: focal marginally below BCE baseline\n'
             'focal -9.98% LER (borderline), gap to MWPM 1.59× (was 1.76×)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig11_d5_ler_curves.png'), dpi=130, bbox_inches='tight')
plt.close()
print('[OK] fig11_d5_ler_curves.png')

print('\n=== all 4 figures generated in', OUT, '===')
