#!/usr/bin/env python3
"""
AlphaQubit评估脚本

使用示例:
    # 评估已训练模型
    python scripts/evaluate.py --checkpoint checkpoints/d3_small/best.pt

    # 与MWPM基线对比
    python scripts/evaluate.py --checkpoint checkpoints/d3_small/best.pt --baseline

    # 计算LER
    python scripts/evaluate.py --checkpoint checkpoints/d3_small/best.pt --ler
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from alphaqubit.data import SurfaceCodeDataset
from alphaqubit.models import AlphaQubitDecoder
from alphaqubit.evaluation import compute_ler, compute_lambda, LERResult, print_ler_summary
from alphaqubit.evaluation.bootstrap import bootstrap_ler
from alphaqubit.experiments.baselines import MWPMBaseline, compare_all_baselines


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AlphaQubit decoder")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")

    # 评估参数
    parser.add_argument("--distance", type=int, default=None,
                       help="Code distance (auto-detect from checkpoint)")
    parser.add_argument("--rounds-list", type=int, nargs="+",
                       default=[3, 5, 7, 9, 11, 12],
                       help="Rounds list for LER calculation")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Samples per rounds value")
    parser.add_argument("--p", type=float, default=0.005,
                       help="Physical error rate")

    # 评估模式
    parser.add_argument("--ler", action="store_true",
                       help="Compute Logical Error Rate")
    parser.add_argument("--baseline", action="store_true",
                       help="Compare with MWPM baseline")
    parser.add_argument("--bootstrap", type=int, default=0,
                       help="Number of bootstrap samples (0=disabled)")

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint重建模型配置
    # 注意：需要保存时包含config信息
    model_state = checkpoint['model_state_dict']

    # 尝试获取训练配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Loaded config: {config}")

    return model_state, checkpoint


@torch.no_grad()
def evaluate_model(model, dataset, num_samples: int = 10000):
    """Evaluate model on dataset."""
    model.eval()
    device = next(model.parameters()).device

    # 生成数据
    batch = dataset.get_batch(num_samples)
    measurement = batch['measurement'].to(device)
    event = batch['event'].to(device)
    leakage = batch['leakage'].to(device)
    event_leakage = batch['event_leakage'].to(device)
    final_soft = batch['final_soft'].to(device)
    labels = batch['label'].to(device)

    # 预测
    predictions, probs = model.predict(measurement, event, leakage, event_leakage, final_soft)

    # 计算指标
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy().flatten()

    error_rate = np.mean(predictions != labels)
    accuracy = 1 - error_rate

    return {
        'error_rate': error_rate,
        'accuracy': accuracy,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probs.cpu().numpy(),
    }


def compute_ler_across_rounds(model, distance, p, rounds_list, num_samples, device):
    """Compute LER by evaluating across different rounds."""
    predictions_by_rounds = {}
    labels_by_rounds = {}

    for n_rounds in rounds_list:
        print(f"  Evaluating rounds={n_rounds}...")

        dataset = SurfaceCodeDataset(
            distance=distance,
            rounds=n_rounds,
            p=p,
            use_soft_readout=True,
            num_samples=num_samples,
        )

        results = evaluate_model(model, dataset, num_samples)
        predictions_by_rounds[n_rounds] = results['predictions']
        labels_by_rounds[n_rounds] = results['labels']

        print(f"    Error rate: {results['error_rate']:.4f}")

    # 计算LER
    ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)

    return ler_result


def main():
    args = parse_args()

    print("="*60)
    print("AlphaQubit Evaluation")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # 加载模型
    print("\n[1] Loading model...")
    model_state, checkpoint = load_model(args.checkpoint, device)

    # TODO: 需要从checkpoint重建完整模型
    # 这里假设用户提供了正确的distance
    if args.distance is None:
        print("Warning: Distance not specified. Please provide --distance")
        return

    # 创建数据集以获取坐标系统
    dataset = SurfaceCodeDataset(
        distance=args.distance,
        rounds=12,
        p=args.p,
        use_soft_readout=True,
        num_samples=100,
    )

    # 重建模型
    from alphaqubit.models import AlphaQubitDecoderConfig
    model = AlphaQubitDecoderConfig.small(dataset.coord_system)
    model.load_state_dict(model_state)
    model = model.to(device)

    print(f"  Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 基本评估
    print("\n[2] Basic evaluation...")
    eval_dataset = SurfaceCodeDataset(
        distance=args.distance,
        rounds=12,
        p=args.p,
        use_soft_readout=True,
        num_samples=args.num_samples,
    )

    results = evaluate_model(model, eval_dataset, args.num_samples)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Error rate: {results['error_rate']:.4f}")

    # LER计算
    if args.ler:
        print("\n[3] Computing LER across rounds...")
        ler_result = compute_ler_across_rounds(
            model=model,
            distance=args.distance,
            p=args.p,
            rounds_list=args.rounds_list,
            num_samples=args.num_samples,
            device=device,
        )

        print("\n  LER Results:")
        print(f"    LER: {ler_result.ler:.6f}")
        print(f"    R²: {ler_result.r_squared:.4f}")
        print(f"    log(F₀): {ler_result.log_f0:.4f}")
        print(f"    Valid fit: {ler_result.is_valid}")

        if ler_result.error_rates:
            print("\n  Error rates by rounds:")
            for n, e in sorted(ler_result.error_rates.items()):
                f = ler_result.fidelities[n]
                print(f"    n={n:2d}: E={e:.4f}, F={f:.4f}")

    # Bootstrap误差估计
    if args.bootstrap > 0:
        print(f"\n[4] Bootstrap error estimation ({args.bootstrap} samples)...")
        # TODO: 实现bootstrap_ler
        print("  Bootstrap not yet implemented")

    # 基线对比
    if args.baseline:
        print("\n[5] Comparing with baselines...")
        baseline_results = compare_all_baselines(
            distance=args.distance,
            rounds=12,
            noise_p=args.p,
            num_samples=args.num_samples,
        )

        print("\n  Baseline Results:")
        for method, result in sorted(baseline_results.items(),
                                     key=lambda x: x[1].error_rate):
            print(f"    {method}: {result.error_rate:.4f}")

        # 与模型对比
        model_rate = results['error_rate']
        mwpm_rate = baseline_results.get('MWPM', None)
        if mwpm_rate:
            improvement = (mwpm_rate.error_rate - model_rate) / mwpm_rate.error_rate * 100
            print(f"\n  Model vs MWPM improvement: {improvement:.1f}%")

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
