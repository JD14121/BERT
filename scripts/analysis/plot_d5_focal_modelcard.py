#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_d5_focal_modelcard.py
为最新最优模型 d5 focal (acc 0.9378, LER 0.005610) 绘制专属模型卡片图。
三联：训练动态(val_acc/loss) + LER 拟合曲线 + pred_pos_rate vs pos_rate 偏差。
不向 LLM 输入图像。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

DATA = r'C:\Users\Administrator\d5focal_data.json'
OUT = r"D:\Code\LZai\Ai for QEC\Alpha-qubit\code\alphaquibit-main\alphaquibit-main\google_paems_data\bert_experiment\figures"

with open(DATA) as f:
    d = json.load(f)
val = d['val']
train = d['train']

val_steps = [v['step'] for v in val]
val_acc = [v['val_acc'] for v in val]
val_loss = [v['val_loss'] for v in val]
tr_steps = [t['step'] for t in train]
tr_pos = [t['pos_rate'] for t in train]
tr_pred = [t['pred_pos_rate'] for t in train]

# LER 拟合曲线
rounds_pts = [1, 10, 13, 30, 50]
focal_E = [0.0551, 0.11365, 0.12695, 0.1843, 0.24925]
mwpm_E = [0.00055, 0.0295, 0.04035, 0.0948, 0.14605]
eps_focal = 0.005610
eps_mwpm = 0.003534
n_curve = np.logspace(0, 1.7, 100)
E_focal_fit = 0.5 * (1 - (1 - 2*eps_focal)**n_curve)
E_mwpm_fit = 0.5 * (1 - (1 - 2*eps_mwpm)**n_curve)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# ---- Panel 1: 训练动态 ----
color1 = '#2196F3'
color2 = '#FF9800'
ax1.plot(val_steps, [a*100 for a in val_acc], 'o-', color=color1, label='val accuracy', linewidth=2, markersize=7)
ax1b = ax1.twinx()
ax1b.plot(val_steps, val_loss, 's--', color=color2, label='val loss', linewidth=2, markersize=6, alpha=0.8)
ax1.set_xlabel('Finetune step')
ax1.set_ylabel('Validation Accuracy (%)', color=color1)
ax1b.set_ylabel('Validation Loss', color=color2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1b.tick_params(axis='y', labelcolor=color2)
ax1.set_title('(a) Training dynamics (d5 focal γ=2)\nval_acc 78.7%→93.85%, converges smoothly')
ax1.axhline(94.28, color='#F44336', linestyle=':', alpha=0.5, label='MWPM acc 94.28%')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3)

# ---- Panel 2: LER 拟合曲线 ----
ax2.plot(n_curve, E_focal_fit, '-', color='#4CAF50', linewidth=2, label=f'focal fit (ε={eps_focal:.5f}, R²=0.994)')
ax2.plot(rounds_pts, focal_E, 'o', color='#4CAF50', markersize=10, label='focal E(n) measured')
ax2.plot(n_curve, E_mwpm_fit, '-', color='black', linewidth=2, alpha=0.6, label=f'MWPM fit (ε={eps_mwpm:.5f})')
ax2.plot(rounds_pts, mwpm_E, '^', color='black', markersize=8, alpha=0.6, label='MWPM E(n) measured')
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax2.set_xscale('log')
ax2.set_xlabel('Rounds n (log scale)')
ax2.set_ylabel('Logical Error Rate E(n)')
ax2.set_title('(b) LER fit (d5 focal)\nε=0.00561, 5/5 valid, 1.59× MWPM')
ax2.legend(loc='upper left', fontsize=8)
ax2.set_ylim(-0.02, 0.4)
ax2.grid(True, alpha=0.3)

# ---- Panel 3: 偏差校准 (pred_pos_rate vs pos_rate) ----
ax3.plot(tr_steps, tr_pos, 'o-', color='#2196F3', label='pos_rate (true positive rate)', linewidth=2, markersize=6)
ax3.plot(tr_steps, tr_pred, 's--', color='#F44336', label='pred_pos_rate (model prediction)', linewidth=2, markersize=6)
ax3.set_xlabel('Finetune step')
ax3.set_ylabel('Positive class rate')
ax3.set_title('(c) Calibration: pred vs true positive rate\nfocal under-predicts (bias ~-0.02), self-corrects late')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(True, alpha=0.3)
# 标注 late bias
ax3.annotate('bias shrinks\nnear step 7000', xy=(7000, 0.336), xytext=(4500, 0.42),
             fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

plt.suptitle('Fig 12 | d5 focal γ=2 model card (latest best: acc 0.9378, LER 0.005610)\n'
             'Best d5 BERT — marginally beats BCE baseline (acc +0.40pp, LER -9.98%); gap to MWPM 0.9pp acc / 1.59× LER',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig(os.path.join(OUT, 'fig12_d5_focal_modelcard.png'), dpi=130, bbox_inches='tight')
plt.close()
print('[OK] fig12_d5_focal_modelcard.png')
