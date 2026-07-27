"""
d=3 v2 vs v3 对比可视化

读取 checkpoints/paems_experiment_d3_v2 与 checkpoints/paems_experiment_d3_v3
的 results_summary.json，生成跨版本对比图。
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_summary(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_accuracy_comparison(summaries: dict, output_dir: Path):
    experiments = list(summaries.keys())
    methods = ['AlphaQubit', 'BERT Finetune', 'MWPM']
    data = {m: [] for m in methods}
    for exp in experiments:
        s = summaries[exp]
        res = s.get('results', {})
        data['BERT Finetune'].append(res.get('bert_finetune', {}).get('accuracy', 0.0))
        data['MWPM'].append(res.get('mwpm', {}).get('accuracy', 0.0))
    data['AlphaQubit'] = [s.get('alphaqubit_accuracy', 0.0) for s in summaries.values()]

    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[method], width, label=method, color=colors[i], alpha=0.8, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Test Accuracy')
    ax.set_title('d=3 Test Accuracy: v2 (250k) vs v3 (500k)')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'v2_vs_v3_accuracy.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def plot_ler_comparison(summaries: dict, output_dir: Path):
    experiments = list(summaries.keys())
    methods = ['AlphaQubit', 'BERT Finetune', 'MWPM']
    data = {m: [] for m in methods}
    valid = {m: [] for m in methods}
    for exp in experiments:
        s = summaries[exp]
        ler = s.get('ler', {})
        for method, key in zip(methods, ['alphaqubit', 'bert_finetune', 'mwpm']):
            if key in ler:
                data[method].append(ler[key].get('ler', np.nan))
                valid[method].append(ler[key].get('is_valid', False))
            else:
                data[method].append(np.nan)
                valid[method].append(False)

    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[method], width, label=method, color=colors[i], alpha=0.8, edgecolor='black')
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                vmark = 'V' if valid[method][j] else 'X'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                        f'{height:.4f}\n({vmark})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('LER (Logical Error per Round)')
    ax.set_title('d=3 LER: v2 (250k) vs v3 (500k)')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'v2_vs_v3_ler.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def plot_ler_curves_comparison(summaries: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    experiments = list(summaries.keys())

    for idx, (exp_name, ax) in enumerate(zip(experiments, axes)):
        s = summaries[exp_name]
        ler = s.get('ler', {})
        rounds = sorted([int(k) for k in ler.get('mwpm', {}).get('error_rates', {}).keys()])

        for name, label, color, marker in [
            ('mwpm', 'MWPM', 'tab:green', 'o'),
            ('alphaqubit', 'AlphaQubit', 'tab:blue', 's'),
            ('bert_finetune', 'BERT Pretrain+Finetune', 'tab:orange', '^'),
        ]:
            if name not in ler:
                continue
            rates = [ler[name]['error_rates'][str(r)] for r in rounds]
            ax.plot(rounds, rates, marker=marker, label=label, linewidth=2, markersize=6, color=color)

        ax.set_xlabel('Rounds')
        ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'LER Curves ({exp_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'v2_vs_v3_ler_curves.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[saved] {out}")


def main():
    output_dir = Path('checkpoints/cross_v3_comparison')
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    mapping = {
        'd=3 v2 (train 250k)': 'checkpoints/paems_experiment_d3_v2',
        'd=3 v3 (train 500k)': 'checkpoints/paems_experiment_d3_v3',
    }
    for name, d in mapping.items():
        path = Path(d) / 'results_summary.json'
        if path.exists():
            summaries[name] = load_summary(path)
        else:
            print(f"[warn] missing {path}")

    # AlphaQubit test accuracy 未写入 results_summary，手动注入
    alphaqubit_accs = {
        'd=3 v2 (train 250k)': 0.8389,
        'd=3 v3 (train 500k)': 0.8396,
    }
    for name, acc in alphaqubit_accs.items():
        if name in summaries:
            summaries[name]['alphaqubit_accuracy'] = acc

    plot_accuracy_comparison(summaries, output_dir)
    plot_ler_comparison(summaries, output_dir)
    plot_ler_curves_comparison(summaries, output_dir)
    print("\nd=3 v2 vs v3 对比图生成完成。")


if __name__ == '__main__':
    main()
