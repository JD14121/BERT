"""
PAEMS 实验结果可视化脚本

读取训练日志与 results_summary.json，生成：
1. BERT 预训练损失曲线
2. AlphaQubit 基准训练损失/准确率曲线
3. BERT 微调损失/准确率曲线
4. LER 错误率随 rounds 变化曲线
5. 测试准确率横向对比柱状图
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（Windows 下使用 SimHei，如不存在则回退）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_log(log_path: Path):
    """解析 compare_baseline_paems.py 输出的训练日志"""
    steps = []
    losses = []
    accs = []
    val_steps = []
    val_losses = []
    val_accs = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 训练日志行：包含 step、loss、accuracy
            if 'step:' in line and 'loss:' in line and '[Eval]' not in line:
                m_step = re.search(r'step:\s*(\d+)', line)
                m_loss = re.search(r'loss:\s*([0-9.]+)', line)
                m_acc = re.search(r'accuracy:\s*([0-9.]+)%', line)
                if m_step and m_loss:
                    steps.append(int(m_step.group(1)))
                    losses.append(float(m_loss.group(1)))
                    if m_acc:
                        accs.append(float(m_acc.group(1)))
                    else:
                        accs.append(np.nan)
            # 验证日志行
            if '[Eval]' in line and 'val_loss:' in line:
                m_step = re.search(r'step:\s*(\d+)', line)
                m_vloss = re.search(r'val_loss:\s*([0-9.]+)', line)
                m_vacc = re.search(r'val_acc:\s*([0-9.]+)%', line)
                if m_step and m_vloss:
                    val_steps.append(int(m_step.group(1)))
                    val_losses.append(float(m_vloss.group(1)))
                    if m_vacc:
                        val_accs.append(float(m_vacc.group(1)))
                    else:
                        val_accs.append(np.nan)

    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'accs': np.array(accs) if any(not np.isnan(x) for x in accs) else None,
        'val_steps': np.array(val_steps),
        'val_losses': np.array(val_losses),
        'val_accs': np.array(val_accs) if any(not np.isnan(x) for x in val_accs) else None,
    }


def parse_pretrain_log(log_path: Path):
    """解析 BERT 预训练日志（mask accuracy）"""
    steps = []
    losses = []
    mask_accs = []
    val_steps = []
    val_losses = []
    val_mask_accs = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 'step:' in line and 'main_loss:' in line:
                m_step = re.search(r'step:\s*(\d+)', line)
                m_loss = re.search(r'main_loss:\s*([0-9.]+)', line)
                m_mask = re.search(r'mask_accuracy:\s*([0-9.]+)%', line)
                if m_step and m_loss:
                    steps.append(int(m_step.group(1)))
                    losses.append(float(m_loss.group(1)))
                    if m_mask:
                        mask_accs.append(float(m_mask.group(1)))
            if '[Eval]' in line and 'val_loss:' in line:
                m_step = re.search(r'step:\s*(\d+)', line)
                m_vloss = re.search(r'val_loss:\s*([0-9.]+)', line)
                m_vmask = re.search(r'val_mask_acc:\s*([0-9.]+)%', line)
                if m_step and m_vloss:
                    val_steps.append(int(m_step.group(1)))
                    val_losses.append(float(m_vloss.group(1)))
                    if m_vmask:
                        val_mask_accs.append(float(m_vmask.group(1)))

    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'mask_accs': np.array(mask_accs) if mask_accs else None,
        'val_steps': np.array(val_steps),
        'val_losses': np.array(val_losses),
        'val_mask_accs': np.array(val_mask_accs) if val_mask_accs else None,
    }


def plot_pretrain(log_path: Path, output_dir: Path):
    data = parse_pretrain_log(log_path)
    if len(data['steps']) == 0:
        print(f"[plot_pretrain] 无训练数据: {log_path}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(data['steps'], data['losses'], color=color, alpha=0.8, label='train loss')
    if len(data['val_steps']) > 0:
        ax1.plot(data['val_steps'], data['val_losses'], color=color, marker='o', linestyle='--', label='val loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Mask Accuracy (%)', color=color)
    if data['mask_accs'] is not None:
        ax2.plot(data['steps'], data['mask_accs'], color=color, alpha=0.8, label='train mask acc')
    if data['val_mask_accs'] is not None and len(data['val_mask_accs']) > 0:
        ax2.plot(data['val_steps'], data['val_mask_accs'], color=color, marker='s', linestyle='--', label='val mask acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')

    plt.title('BERT Pretrain on PAEMS d=3')
    plt.tight_layout()
    out = output_dir / 'bert_pretrain_curves.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def plot_supervised(log_path: Path, output_dir: Path, title: str, prefix: str):
    data = parse_log(log_path)
    if len(data['steps']) == 0:
        print(f"[plot_supervised] 无训练数据: {log_path}")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(data['steps'], data['losses'], color=color, alpha=0.8, label='train loss')
    if len(data['val_steps']) > 0:
        ax1.plot(data['val_steps'], data['val_losses'], color=color, marker='o', linestyle='--', label='val loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Accuracy (%)', color=color)
    if data['accs'] is not None:
        mask = ~np.isnan(data['accs'])
        ax2.plot(data['steps'][mask], data['accs'][mask], color=color, alpha=0.8, label='train acc')
    if data['val_accs'] is not None and len(data['val_accs']) > 0:
        mask = ~np.isnan(data['val_accs'])
        ax2.plot(data['val_steps'][mask], data['val_accs'][mask], color=color, marker='s', linestyle='--', label='val acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')

    plt.title(title)
    plt.tight_layout()
    out = output_dir / f'{prefix}_curves.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def plot_ler(summary: dict, output_dir: Path):
    ler = summary.get('ler', {})
    rounds = sorted([int(k) for k in ler.get('mwpm', {}).get('error_rates', {}).keys()])

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, label, color, marker in [
        ('mwpm', 'MWPM', 'tab:green', 'o'),
        ('alphaqubit', 'AlphaQubit (from scratch)', 'tab:blue', 's'),
        ('bert_finetune', 'BERT Pretrain + Finetune', 'tab:orange', '^'),
    ]:
        if name not in ler:
            continue
        rates = [ler[name]['error_rates'][str(r)] for r in rounds]
        ax.plot(rounds, rates, marker=marker, label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title('Logical Error per Round (PAEMS d=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'ler_curves.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")

    #  fidelity 曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, label, color, marker in [
        ('mwpm', 'MWPM', 'tab:green', 'o'),
        ('alphaqubit', 'AlphaQubit (from scratch)', 'tab:blue', 's'),
        ('bert_finetune', 'BERT Pretrain + Finetune', 'tab:orange', '^'),
    ]:
        if name not in ler:
            continue
        fids = [ler[name]['fidelities'][str(r)] for r in rounds]
        ax.plot(rounds, fids, marker=marker, label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Logical Fidelity')
    ax.set_title('Logical Fidelity vs Rounds (PAEMS d=3)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'fidelity_curves.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def plot_test_accuracy(summary: dict, output_dir: Path, alphaqubit_acc: float):
    results = summary.get('results', {})
    names = ['AlphaQubit\n(from scratch)', 'BERT\nPretrain+Finetune', 'MWPM']
    accs = [
        alphaqubit_acc,
        results.get('bert_finetune', {}).get('accuracy', 0.0),
        results.get('mwpm', {}).get('accuracy', 0.0),
    ]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, accs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison (PAEMS d=3)')
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)

    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    out = output_dir / 'test_accuracy_comparison.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def main():
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    else:
        checkpoint_dir = Path('checkpoints/paems_experiment_d3')

    output_dir = checkpoint_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = checkpoint_dir / 'results_summary.json'
    if not summary_path.exists():
        print(f"找不到结果文件: {summary_path}")
        sys.exit(1)

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # 训练日志路径
    pretrain_log = checkpoint_dir.parent / (checkpoint_dir.name + '_pretrain.log')
    baseline_log = checkpoint_dir.parent / (checkpoint_dir.name + '_baseline.log')
    eval_log = checkpoint_dir.parent / (checkpoint_dir.name + '_eval.log')

    # 若不存在，尝试在 checkpoint_dir 内查找
    if not pretrain_log.exists():
        pretrain_log = checkpoint_dir / 'pretrain.log'
    if not baseline_log.exists():
        baseline_log = checkpoint_dir / 'baseline.log'
    if not eval_log.exists():
        eval_log = checkpoint_dir / 'eval.log'

    print(f"输出目录: {output_dir}")
    if pretrain_log.exists():
        plot_pretrain(pretrain_log, output_dir)
    if baseline_log.exists():
        plot_supervised(baseline_log, output_dir, 'AlphaQubit Baseline on PAEMS d=3', 'alphaqubit_baseline')
    if eval_log.exists():
        plot_supervised(eval_log, output_dir, 'BERT Finetune on PAEMS d=3', 'bert_finetune')

    plot_ler(summary, output_dir)

    # AlphaQubit test accuracy 未写入 results_summary（在 baseline 阶段产生），
    # 这里从命令行参数或日志中解析。若未提供，尝试从 baseline 日志解析。
    alphaqubit_acc = None
    if baseline_log.exists():
        with open(baseline_log, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.search(r'\[AlphaQubit Baseline\] Test Accuracy:\s*([0-9.]+)', line)
                if m:
                    alphaqubit_acc = float(m.group(1))
                    break
    if alphaqubit_acc is None:
        alphaqubit_acc = float(input("请输入 AlphaQubit test accuracy: "))

    plot_test_accuracy(summary, output_dir, alphaqubit_acc)

    print("\n全部图像生成完成。")


if __name__ == '__main__':
    main()
