#!/usr/bin/env python3
"""
AlphaQubit评估脚本 - 与Baseline对比

计算LER (Logical Error Rate)并与MWPM baseline对比，
生成类似论文的评估报告。

使用示例:
    python scripts/evaluate_with_baseline.py --checkpoint checkpoints/d3_20260202_160048/d3_base/best.pt

    # 指定更多参数
    python scripts/evaluate_with_baseline.py \
        --checkpoint checkpoints/d3_base/best.pt \
        --distance 3 \
        --model-size base \
        --num-samples 50000 \
        --rounds 3 5 7 9 11 13
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.amp import autocast

from alphaqubit.data import SurfaceCodeDataset, PrefetchDataLoader
from alphaqubit.models import AlphaQubitDecoderConfig
from alphaqubit.evaluation.metrics import compute_error_rate, compute_fidelity, fit_ler, LERResult
from alphaqubit.experiments.baselines import MWPMBaseline, compare_all_baselines


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AlphaQubit with baseline comparison")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--distance", type=int, default=3,
                       help="Code distance")
    parser.add_argument("--model-size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"])
    parser.add_argument("--rounds", type=int, nargs="+",
                       default=[3, 5, 7, 9, 11, 13, 17, 21, 25],
                       help="Rounds list for LER calculation")
    parser.add_argument("--num-samples", type=int, default=20000,
                       help="Samples per rounds value")
    parser.add_argument("--p", type=float, default=0.005,
                       help="Physical error rate")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Data generation workers")

    return parser.parse_args()


def load_model(checkpoint_path: str, distance: int, model_size: str, device: str = "cuda"):
    """Load model from checkpoint."""

    # 创建临时数据集获取坐标系统
    temp_dataset = SurfaceCodeDataset(
        distance=distance,
        rounds=12,
        p=0.005,
        num_samples=100,
    )
    coord_system = temp_dataset.coord_system

    # 创建模型
    if model_size == "tiny":
        model = AlphaQubitDecoderConfig.tiny(coord_system)
    elif model_size == "small":
        model = AlphaQubitDecoderConfig.small(coord_system)
    elif model_size == "base":
        model = AlphaQubitDecoderConfig.base(coord_system)
    else:
        model = AlphaQubitDecoderConfig.large(coord_system)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, coord_system


@torch.no_grad()
def evaluate_model_at_rounds(
    model,
    distance: int,
    rounds: int,
    p: float,
    num_samples: int,
    num_workers: int,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model at specific rounds."""

    # 创建数据加载器
    loader = PrefetchDataLoader(
        distance=distance,
        rounds=rounds,
        p=p,
        batch_size=min(2048, num_samples),
        num_workers=num_workers,
        num_batches=(num_samples + 2047) // 2048,
        seed=42 + rounds,  # 不同rounds用不同seed
        device=device,
    )

    loader.start()

    all_predictions = []
    all_labels = []
    all_probs = []

    try:
        samples_collected = 0
        for batch in loader:
            if samples_collected >= num_samples:
                break

            measurement = batch['measurement'].to(device, non_blocking=True)
            event = batch['event'].to(device, non_blocking=True)
            leakage = batch['leakage'].to(device, non_blocking=True)
            event_leakage = batch['event_leakage'].to(device, non_blocking=True)
            final_soft = batch['final_soft'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                preds, probs = model.predict(measurement, event, leakage, event_leakage, final_soft)

            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy().flatten())
            all_probs.append(probs.cpu().numpy())

            samples_collected += measurement.size(0)
    finally:
        loader.shutdown()

    predictions = np.concatenate(all_predictions)[:num_samples]
    labels = np.concatenate(all_labels)[:num_samples]
    probs = np.concatenate(all_probs)[:num_samples]

    error_rate = compute_error_rate(predictions, labels)
    fidelity = compute_fidelity(error_rate)

    return error_rate, fidelity, predictions, labels


def compute_model_ler(
    model,
    distance: int,
    p: float,
    rounds_list: List[int],
    num_samples: int,
    num_workers: int,
    device: str,
) -> LERResult:
    """Compute LER for model across multiple rounds."""

    error_rates = {}
    fidelities = {}

    print(f"\n  {'Rounds':<8} {'Error Rate':<12} {'Fidelity':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12}")

    for rounds in rounds_list:
        e, f, _, _ = evaluate_model_at_rounds(
            model, distance, rounds, p, num_samples, num_workers, device
        )
        error_rates[rounds] = e
        fidelities[rounds] = f
        print(f"  {rounds:<8} {e:<12.4f} {f:<12.4f}")

    # 拟合LER
    fid_list = [fidelities[n] for n in rounds_list]
    ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fid_list)

    return LERResult(
        ler=ler,
        r_squared=r_squared,
        log_f0=log_f0,
        slope=slope,
        intercept=log_f0,
        is_valid=is_valid,
        error_rates=error_rates,
        fidelities=fidelities,
    )


def compute_mwpm_ler(
    distance: int,
    p: float,
    rounds_list: List[int],
    num_samples: int,
) -> LERResult:
    """Compute LER for MWPM baseline."""

    mwpm = MWPMBaseline()

    error_rates = {}
    fidelities = {}

    print(f"\n  {'Rounds':<8} {'Error Rate':<12} {'Fidelity':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12}")

    for rounds in rounds_list:
        result = mwpm.evaluate_single(distance, rounds, p, num_samples)
        e = result.error_rate
        f = compute_fidelity(e)
        error_rates[rounds] = e
        fidelities[rounds] = f
        print(f"  {rounds:<8} {e:<12.4f} {f:<12.4f}")

    # 拟合LER
    fid_list = [fidelities[n] for n in rounds_list]
    ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fid_list)

    return LERResult(
        ler=ler,
        r_squared=r_squared,
        log_f0=log_f0,
        slope=slope,
        intercept=log_f0,
        is_valid=is_valid,
        error_rates=error_rates,
        fidelities=fidelities,
    )


def print_comparison_table(
    model_ler: LERResult,
    mwpm_ler: LERResult,
    rounds_list: List[int],
    distance: int,
    p: float,
):
    """Print comparison table like in the paper."""

    print("\n" + "="*70)
    print(f"  EVALUATION RESULTS - Distance d={distance}, p={p}")
    print("="*70)

    # 详细对比表
    print(f"\n  {'Rounds':<8} {'AlphaQubit':<15} {'MWPM':<15} {'Improvement':<12}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12}")

    for n in rounds_list:
        model_e = model_ler.error_rates.get(n, float('nan'))
        mwpm_e = mwpm_ler.error_rates.get(n, float('nan'))

        if mwpm_e > 0:
            improvement = (mwpm_e - model_e) / mwpm_e * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"

        print(f"  {n:<8} {model_e:<15.4f} {mwpm_e:<15.4f} {imp_str:<12}")

    # LER Summary
    print("\n" + "-"*70)
    print("  LER (Logical Error Rate per Round)")
    print("-"*70)

    print(f"\n  {'Method':<15} {'LER':<15} {'R^2':<10} {'log(F0)':<10} {'Valid':<8}")
    print(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*8}")

    model_valid = "Yes" if model_ler.is_valid else "No"
    mwpm_valid = "Yes" if mwpm_ler.is_valid else "No"

    print(f"  {'AlphaQubit':<15} {model_ler.ler:<15.6f} {model_ler.r_squared:<10.4f} {model_ler.log_f0:<10.4f} {model_valid:<8}")
    print(f"  {'MWPM':<15} {mwpm_ler.ler:<15.6f} {mwpm_ler.r_squared:<10.4f} {mwpm_ler.log_f0:<10.4f} {mwpm_valid:<8}")

    # 相对改进
    if mwpm_ler.ler > 0 and model_ler.ler > 0:
        ler_improvement = (mwpm_ler.ler - model_ler.ler) / mwpm_ler.ler * 100
        ler_ratio = mwpm_ler.ler / model_ler.ler

        print("\n" + "-"*70)
        print("  SUMMARY")
        print("-"*70)
        print(f"\n  LER Improvement: {ler_improvement:+.1f}%")
        print(f"  LER Ratio (MWPM/AlphaQubit): {ler_ratio:.2f}x")

        if ler_improvement > 0:
            print(f"\n  AlphaQubit achieves {ler_improvement:.1f}% lower logical error rate than MWPM!")
        else:
            print(f"\n  MWPM performs better by {-ler_improvement:.1f}%")

    print("\n" + "="*70)


def main():
    args = parse_args()

    print("="*70)
    print("  AlphaQubit Evaluation with Baseline Comparison")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Distance: {args.distance}")
    print(f"  Model size: {args.model_size}")
    print(f"  Physical error rate: {args.p}")
    print(f"  Samples per rounds: {args.num_samples}")
    print(f"  Rounds: {args.rounds}")

    # 加载模型
    print("\n[1/3] Loading model...")
    model, coord_system = load_model(
        args.checkpoint, args.distance, args.model_size, device
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params:,} parameters")

    # 评估AlphaQubit
    print("\n[2/3] Evaluating AlphaQubit...")
    start_time = time.time()
    model_ler = compute_model_ler(
        model=model,
        distance=args.distance,
        p=args.p,
        rounds_list=args.rounds,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        device=device,
    )
    model_time = time.time() - start_time
    print(f"  AlphaQubit evaluation completed in {model_time:.1f}s")
    print(f"  LER = {model_ler.ler:.6f}, R^2 = {model_ler.r_squared:.4f}")

    # 评估MWPM baseline
    print("\n[3/3] Evaluating MWPM baseline...")
    start_time = time.time()
    mwpm_ler = compute_mwpm_ler(
        distance=args.distance,
        p=args.p,
        rounds_list=args.rounds,
        num_samples=args.num_samples,
    )
    mwpm_time = time.time() - start_time
    print(f"  MWPM evaluation completed in {mwpm_time:.1f}s")
    print(f"  LER = {mwpm_ler.ler:.6f}, R^2 = {mwpm_ler.r_squared:.4f}")

    # 打印对比表
    print_comparison_table(model_ler, mwpm_ler, args.rounds, args.distance, args.p)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
