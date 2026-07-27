#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_latest.py: 为 EXPERIMENT_REPORT §3.8 最新结果绘图
Fig 8: d5 BERT 跨迭代进展 (accuracy + LER)
Fig 9: focal γ=2 跨码距对比 (d5 弱正向 vs d7 崩塌)
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = r"D:\Code\LZai\Ai for QEC\Alpha-qubit\code\alphaquibit-main\alphaquibit-main\google_paems_data\bert_experiment"
FIG = os.path.join(BASE, "figures")
CLOUD_D7 = os.path.join(BASE, "cloud_d7")

def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

# ---- d5 numbers ----
d5_old_acc, d5_old_ler = 0.7980, 0.0267  # 旧小模型 (EXPERIMENT_REPORT §3.1/3.2)
d5_opt = load_json(os.path.join(BASE, "results_summary_d5_E1_160M_100k_opt.json"))
d5_opt_acc = d5_opt["results"]["bert"]["accuracy"]
d5_opt_ler = load_json(os.path.join(BASE, "results_ler_d5_E1_160M_100k_opt.json"))["5"]["bert"]["ler"]
d5_focal = load_json(os.path.join(BASE, "results_summary_d5_focal.json"))
d5_focal_acc = d5_focal["results"]["bert"]["accuracy"]
d5_focal_ler = load_json(os.path.join(BASE, "results_ler_d5_focal.json"))["5"]["bert"]["ler"]
d5_mwpm_acc, d5_mwpm_ler = 0.9428, 0.003534

# ---- d7 numbers ----
d7_r5_acc = load_json(os.path.join(CLOUD_D7, "results_summary_d7_E1_125M_100k.json"))["results"]["bert"]["accuracy"]
d7_r5_ler = load_json(os.path.join(CLOUD_D7, "results_ler_d7_E1_125M_100k.json"))["7"]["bert"]["ler"]
d7_focal_j = load_json(os.path.join(BASE, "results_summary_d7_focal.json"))
d7_focal_acc = d7_focal_j["results"]["bert"]["accuracy"]
d7_focal_ler_info = load_json(os.path.join(BASE, "results_ler_d7_focal.json"))["7"]["bert"]
d7_focal_ler_valid = d7_focal_ler_info["is_valid"]
d7_focal_ler_fit = d7_focal_ler_info["n_fit_points"]
d7_mwpm_acc, d7_mwpm_ler = 0.9702, 0.002680

print(f"d5: old={d5_old_acc}/{d5_old_ler}, opt={d5_opt_acc}/{d5_opt_ler}, focal={d5_focal_acc}/{d5_focal_ler}, mwpm={d5_mwpm_acc}/{d5_mwpm_ler}")
print(f"d7: R5={d7_r5_acc}/{d7_r5_ler}, focal={d7_focal_acc}/LER_valid={d7_focal_ler_valid}(fit {d7_focal_ler_fit}/5), mwpm={d7_mwpm_acc}/{d7_mwpm_ler}")

# ============ Fig 8: d5 跨迭代进展 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
labels = ['旧小模型\n(1.64M, 1×)', 'd5 opt\n(12M, 160M)', 'd5 focal\n(γ=2)', 'MWPM']
accs = [d5_old_acc, d5_opt_acc, d5_focal_acc, d5_mwpm_acc]
lers = [d5_old_ler, d5_opt_ler, d5_focal_ler, d5_mwpm_ler]
colors = ['#888888', '#4C72B0', '#55A868', '#C44E52']

bars1 = ax1.bar(labels, accs, color=colors)
ax1.set_ylabel('Test Accuracy')
ax1.set_title('(a) d5 BERT Accuracy 跨迭代进展')
ax1.set_ylim(0.75, 0.97)
for b, v in zip(bars1, accs):
    ax1.text(b.get_x()+b.get_width()/2, v+0.003, f'{v:.4f}', ha='center', fontsize=9)
ax1.axhline(d5_mwpm_acc, color='#C44E52', linestyle='--', linewidth=1, alpha=0.5)

bars2 = ax2.bar(labels, lers, color=colors)
ax2.set_ylabel('LER ε (对数轴)')
ax2.set_title('(b) d5 BERT LER 跨迭代进展')
ax2.set_yscale('log')
for b, v in zip(bars2, lers):
    ax2.text(b.get_x()+b.get_width()/2, v*1.08, f'{v:.5f}', ha='center', fontsize=9)
ax2.axhline(d5_mwpm_ler, color='#C44E52', linestyle='--', linewidth=1, alpha=0.5, label='MWPM')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig8_d5_progress.png"), dpi=120, bbox_inches='tight')
plt.close()
print("saved fig8_d5_progress.png")

# ============ Fig 9: focal 跨码距对比 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
d5_dacc = (d5_focal_acc - d5_opt_acc) * 100
d7_dacc = (d7_focal_acc - d7_r5_acc) * 100
d5_dler = (d5_focal_ler - d5_opt_ler) / d5_opt_ler * 100

# (a) Δacc
ax1.bar(['d5', 'd7'], [d5_dacc, d7_dacc], color=['#55A868', '#C44E52'])
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylabel('Δ Accuracy (pp) vs BCE 基线')
ax1.set_title('(a) focal γ=2 对 accuracy 的影响')
ax1.set_ylim(-25, 5)
for i, v in enumerate([d5_dacc, d7_dacc]):
    ax1.text(i, v + (0.8 if v > 0 else -2.2), f'{v:+.2f}pp', ha='center', fontsize=11, fontweight='bold')
ax1.text(0, -23, '弱正向\n(+0.40pp)', ha='center', fontsize=9, color='#55A868')
ax1.text(1, -23, '崩塌\n(-19.74pp)', ha='center', fontsize=9, color='#C44E52')

# (b) ΔLER%
ax2.bar(['d5'], [d5_dler], color='#55A868')
ax2.bar(['d7'], [0], color='#C44E52', hatch='///', edgecolor='black', linewidth=1.5)
ax2.text(0, d5_dler - 1.5, f'{d5_dler:.2f}%', ha='center', fontsize=11, fontweight='bold', color='#55A868')
ax2.text(0, -13, '边界\n(降幅<10%)', ha='center', fontsize=9, color='#55A868')
ax2.text(1, -6, 'INVALID\n(fit 1/5,\nLER 无法拟合)', ha='center', fontsize=10, color='#C44E52', fontweight='bold')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Δ LER (%) vs BCE 基线')
ax2.set_title('(b) focal γ=2 对 LER 的影响')
ax2.set_ylim(-16, 5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig9_focal_cross_distance.png"), dpi=120, bbox_inches='tight')
plt.close()
print("saved fig9_focal_cross_distance.png")
print("=== done ===")
