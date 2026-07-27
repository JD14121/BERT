#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_architecture.py
绘制 AQ vs BERT(预训练/微调) 架构对比科研图。
三栏：数据流从上到下，颜色编码共享/独有组件。
不向 LLM 输入图像。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# CJK 字体（Windows SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = r"D:\Code\LZai\Ai for QEC\Alpha-qubit\code\alphaquibit-main\alphaquibit-main\google_paems_data\bert_experiment\figures"

# 颜色
C_INPUT = '#ECEFF1'   # 输入 灰
C_SHARED = '#BBDEFB'  # 共享编码器 蓝
C_AQ = '#C8E6C9'      # AQ/微调读出 绿
C_BERT = '#FFE0B2'    # BERT 预训练独有 橙
C_OUT = '#FFCDD2'     # 输出/loss 红
C_PRETRAIN = '#FFF9C4' # 加载预训练 黄边

def box(ax, x, y, w, h, text, color, fontsize=8, edgecolor='black'):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.2,
                       edgecolor=edgecolor, facecolor=color)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.3))

fig, axes = plt.subplots(1, 3, figsize=(17, 11))
titles = ['(a) AQ (AlphaQubitDecoder)\n全监督', '(b) BERT 预训练 (PretrainDecoder)\n自监督掩码', '(c) BERT 微调 (FineTuneDecoder)\n监督，加载预训练 encoder']
for ax, t in zip(axes, titles):
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis('off')
    ax.set_title(t, fontsize=11, fontweight='bold', pad=10)

# ===== Panel 1: AQ =====
ax = axes[0]
box(ax, 0.5, 18, 9, 1.2, '输入: measurement / event / leakage\n[B, T, n_stab]  +  final_soft [B, n_data]', C_INPUT, 8)
arrow(ax, 5, 18, 5, 17.3)
box(ax, 1, 16, 8, 1.2, 'SyndromeEmbedder + FinalDataEmbedder\n逐轮嵌入 -> [B,T,n_stab,D]', C_SHARED, 8)
arrow(ax, 5, 16, 5, 15.3)
box(ax, 1, 14, 8, 1.2, 'RNNCore (0.7缩放 + SyndromeTransformer)\n双向自注意力 + spatial_bias', C_SHARED, 8)
arrow(ax, 5, 14, 5, 13.3)
box(ax, 1, 12, 8, 1.2, 'LateFusion\nstab_features + final_soft', C_AQ, 8)
arrow(ax, 5, 12, 5, 11.3)
box(ax, 1, 10, 8, 1.2, 'FullReadoutNetwork\n深层 ResNet + CycleEmbedding', C_AQ, 8)
arrow(ax, 5, 10, 5, 9.3)
box(ax, 1.5, 8, 7, 1.0, 'logit [B,1]', C_OUT, 9)
arrow(ax, 5, 8, 5, 7.3)
box(ax, 1.5, 6, 7, 1.0, 'BCE(logit, label)\n监督', C_OUT, 9)
# 标注
ax.text(5, 4.5, '全程用 final_soft\n+ LateFusion + 重读出', ha='center', fontsize=8, color='#2E7D32', style='italic')

# ===== Panel 2: BERT 预训练 =====
ax = axes[1]
box(ax, 0.5, 18, 9, 1.2, '输入: measurement / event (部分掩码)\n[B, T, n_stab]   无 final_soft', C_INPUT, 8)
arrow(ax, 5, 18, 5, 17.3)
box(ax, 1, 16, 8, 1.2, 'SyndromeEmbedder\n逐轮嵌入 -> [B,T,n_stab,D]', C_SHARED, 8)
arrow(ax, 5, 16, 5, 15.3)
box(ax, 1, 14, 8, 1.2, 'RNNCore.forward_with_all_states\n0.7缩放 + Transformer', C_SHARED, 8)
arrow(ax, 5, 14, 5, 13.3)
box(ax, 1, 12, 8, 1.2, 'TemporalReconstructionHead\nMLP: D->D/2->1 (轻量)', C_BERT, 8)
arrow(ax, 5, 12, 5, 11.3)
box(ax, 1.5, 10, 7, 1.0, 'pred [B,T,n_stab]', C_OUT, 9)
arrow(ax, 5, 10, 5, 9.3)
box(ax, 1.5, 8, 7, 1.2, 'BCE(pred[mask], target[mask])\n自监督（仅 mask 位置）', C_OUT, 9)
ax.text(5, 6.5, '无 final_soft / 无 LateFusion\n掩码头预训练后丢弃\n学 syndrome 内部结构', ha='center', fontsize=8, color='#E65100', style='italic')
# 掩码策略
box(ax, 1, 4.5, 8, 1.2, 'MixedStructuredMSM 掩码\n40%随机 + 30%空间簇 + 30%时序', C_BERT, 7)

# ===== Panel 3: BERT 微调 =====
ax = axes[2]
box(ax, 0.5, 18, 9, 1.2, '输入: measurement / event / leakage\n[B, T, n_stab]  +  final_soft [B, n_data]', C_INPUT, 8)
arrow(ax, 5, 18, 5, 17.3)
box(ax, 1, 16, 8, 1.2, 'SyndromeEmbedder\n✓ 加载预训练权重', C_PRETRAIN, 8, edgecolor='#F57F17')
arrow(ax, 5, 16, 5, 15.3)
box(ax, 1, 14, 8, 1.2, 'RNNCore\n✓ 加载预训练权重', C_PRETRAIN, 8, edgecolor='#F57F17')
arrow(ax, 5, 14, 5, 13.3)
box(ax, 1, 12, 8, 1.2, 'LateFusion (新增)\nstab_features + final_soft', C_AQ, 8)
arrow(ax, 5, 12, 5, 11.3)
box(ax, 1, 10, 8, 1.2, 'data_readout + ResNet + CycleEmbedding\n(新增)', C_AQ, 8)
arrow(ax, 5, 10, 5, 9.3)
box(ax, 1.5, 8, 7, 1.0, 'logit [B,1]', C_OUT, 9)
arrow(ax, 5, 8, 5, 7.3)
box(ax, 1.5, 6, 7, 1.2, 'BCE(logit, label)\n监督', C_OUT, 9)
ax.text(5, 4.5, 'encoder 从掩码预训练来\n+ 新增 final_soft/LateFusion/readout\n架构 ≈ AQ，仅初始化不同', ha='center', fontsize=8, color='#2E7D32', style='italic')

# 图例
legend_elements = [
    mpatches.Patch(facecolor=C_SHARED, edgecolor='black', label='共享编码器 (SyndromeEmbedder + RNNCore + Transformer)'),
    mpatches.Patch(facecolor=C_AQ, edgecolor='black', label='读出路径 (LateFusion + Readout)'),
    mpatches.Patch(facecolor=C_BERT, edgecolor='black', label='BERT 预训练独有 (掩码 + ReconstructionHead)'),
    mpatches.Patch(facecolor=C_PRETRAIN, edgecolor='#F57F17', label='加载预训练权重'),
    mpatches.Patch(facecolor=C_INPUT, edgecolor='black', label='输入'),
    mpatches.Patch(facecolor=C_OUT, edgecolor='black', label='输出 / Loss'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('AQ vs BERT 解码器架构与数据流对比\n共享编码器核心，区别在预训练目标 + 头部 + final_soft 使用', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(OUT, 'fig_architecture_aq_bert.png'), dpi=140, bbox_inches='tight')
plt.close()
print('[OK] fig_architecture_aq_bert.png')
