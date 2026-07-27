#!/usr/bin/env python3
"""
AlphaQubit完整实验脚本

运行完整的三阶段实验：
1. Phase 1: 验证Pipeline (d=3)
2. Phase 2: 完整实验 (d=3, d=5) + Λ计算
3. Phase 3: 扩展实验 (d=9)

使用示例:
    # 快速验证（debug模式）
    python scripts/run_experiment.py --phase 1 --debug

    # 完整Phase 1
    python scripts/run_experiment.py --phase 1

    # 完整实验
    python scripts/run_experiment.py --phase all
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from alphaqubit.data import SurfaceCodeDataset
from alphaqubit.models import AlphaQubitDecoderConfig
from alphaqubit.training import Trainer, TrainingConfig
from alphaqubit.evaluation import compute_ler, compute_lambda, print_ler_summary
from alphaqubit.experiments.baselines import MWPMBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaQubit experiments")

    parser.add_argument("--phase", type=str, default="1",
                       choices=["1", "2", "3", "all"],
                       help="Experiment phase to run")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (fewer samples, shorter training)")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_phase1(output_dir: Path, debug: bool = False, seed: int = 42):
    """
    Phase 1: 验证Pipeline (d=3, p=0.005)

    目标：
    1. 完成数据生成 + 坐标映射
    2. 验证所有sanity check
    3. 训练小模型，确认loss下降
    4. 跑PyMatching baseline
    5. 计算LER，确认流程正确
    """
    print("\n" + "="*60)
    print("PHASE 1: Pipeline Validation (d=3)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    # 参数设置
    distance = 3
    rounds = 12
    p = 0.005
    num_samples = 1000 if debug else 50000
    train_steps = 500 if debug else 10000

    # === Step 1: 数据生成验证 ===
    print("\n[Step 1] Data generation validation...")

    train_ds = SurfaceCodeDataset(
        distance=distance,
        rounds=rounds,
        p=p,
        use_soft_readout=True,
        snr=10.0,
        num_samples=num_samples,
        seed=seed,
    )

    metrics = train_ds.validate()
    print(f"  Event density: {metrics['event_density']:.4f}")
    print(f"  Scatter/Gather valid: {metrics['scatter_gather_valid']}")

    # 验证检查
    assert metrics['scatter_gather_valid'], "Scatter/Gather round-trip failed!"
    assert 0.001 < metrics['event_density'] < 0.2, f"Event density out of range: {metrics['event_density']}"
    print("  ✓ Data generation validation passed")

    # === Step 2: 模型sanity check ===
    print("\n[Step 2] Model sanity check...")

    model = AlphaQubitDecoderConfig.tiny(train_ds.coord_system)
    model = model.to(device)

    # 检查初始loss（应接近ln(2) ≈ 0.693）
    batch = train_ds.get_batch(64)
    measurement = batch['measurement'].to(device)
    event = batch['event'].to(device)
    leakage = batch['leakage'].to(device)
    event_leakage = batch['event_leakage'].to(device)
    final_soft = batch['final_soft'].to(device)
    labels = batch['label'].to(device)

    logits = model(measurement, event, leakage, event_leakage, final_soft)
    initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels
    ).item()

    print(f"  Initial loss: {initial_loss:.4f} (expected ~0.693)")
    assert 0.5 < initial_loss < 1.0, f"Initial loss unexpected: {initial_loss}"
    print("  ✓ Initial loss check passed")

    # === Step 3: 短期训练验证 ===
    print("\n[Step 3] Short training validation...")

    val_ds = SurfaceCodeDataset(
        distance=distance,
        rounds=rounds,
        p=p,
        use_soft_readout=True,
        num_samples=num_samples // 10,
        seed=seed + 1000,
    )

    config = TrainingConfig(
        total_steps=train_steps,
        batch_size=256,
        learning_rate=2e-4,
        warmup_steps=100,
        eval_interval=train_steps // 5,
        log_interval=train_steps // 10,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_dir=str(output_dir / "phase1_checkpoints"),
    )

    history = trainer.train()

    # 验证loss下降
    if len(history['train_loss']) > 10:
        early_loss = np.mean(history['train_loss'][:10])
        late_loss = np.mean(history['train_loss'][-10:])
        print(f"  Early loss: {early_loss:.4f}")
        print(f"  Late loss: {late_loss:.4f}")
        assert late_loss < early_loss, "Training did not reduce loss!"
        print("  ✓ Training reduces loss")

    # === Step 4: Baseline对比 ===
    print("\n[Step 4] MWPM baseline...")

    try:
        mwpm = MWPMBaseline(seed=seed)
        baseline_result = mwpm.evaluate_single(
            distance=distance,
            rounds=rounds,
            noise_p=p,
            num_samples=min(10000, num_samples),
        )
        print(f"  MWPM error rate: {baseline_result.error_rate:.4f}")
        print("  ✓ MWPM baseline computed")
    except ImportError:
        print("  ⚠ PyMatching not installed, skipping baseline")

    # === Step 5: LER计算 ===
    print("\n[Step 5] LER calculation...")

    rounds_list = [3, 6, 9, 12]
    predictions_by_rounds = {}
    labels_by_rounds = {}

    model.eval()
    with torch.no_grad():
        for n_rounds in rounds_list:
            test_ds = SurfaceCodeDataset(
                distance=distance,
                rounds=n_rounds,
                p=p,
                use_soft_readout=True,
                num_samples=min(5000, num_samples),
                seed=seed + n_rounds,
            )

            batch = test_ds.get_batch(len(test_ds))
            measurement = batch['measurement'].to(device)
            event = batch['event'].to(device)
            leakage = batch['leakage'].to(device)
            event_leakage = batch['event_leakage'].to(device)
            final_soft = batch['final_soft'].to(device)
            labels = batch['label']

            preds, _ = model.predict(measurement, event, leakage, event_leakage, final_soft)
            predictions_by_rounds[n_rounds] = preds.cpu().numpy()
            labels_by_rounds[n_rounds] = labels.numpy().flatten()

    ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)

    print(f"  LER: {ler_result.ler:.6f}")
    print(f"  R²: {ler_result.r_squared:.4f}")
    print(f"  Valid: {ler_result.is_valid}")

    print("\n" + "="*60)
    print("PHASE 1 COMPLETED")
    print("="*60)

    return {"ler_result": ler_result, "history": history}


def run_phase2(output_dir: Path, debug: bool = False, seed: int = 42):
    """
    Phase 2: 完整实验 (d=3, d=5)

    目标：
    1. 完整训练d=3和d=5
    2. 计算Λ_{3→5}，验证 > 1
    3. 绘制LER曲线 vs baseline
    """
    print("\n" + "="*60)
    print("PHASE 2: Full Experiments (d=3, d=5)")
    print("="*60)

    results = {}

    for distance in [3, 5]:
        print(f"\n--- Training d={distance} ---")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)

        # 参数
        rounds = 12
        p = 0.005
        num_samples = 5000 if debug else 100000
        train_steps = 1000 if debug else 30000

        # 数据集
        train_ds = SurfaceCodeDataset(
            distance=distance,
            rounds=rounds,
            p=p,
            use_soft_readout=True,
            num_samples=num_samples,
            seed=seed,
        )

        val_ds = SurfaceCodeDataset(
            distance=distance,
            rounds=rounds,
            p=p,
            use_soft_readout=True,
            num_samples=num_samples // 10,
            seed=seed + 1000,
        )

        # 模型
        model = AlphaQubitDecoderConfig.small(train_ds.coord_system)
        model = model.to(device)

        # 训练
        config = TrainingConfig(
            total_steps=train_steps,
            batch_size=512 if distance <= 5 else 256,
            learning_rate=2e-4,
            warmup_steps=1000,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            save_dir=str(output_dir / f"phase2_d{distance}"),
        )

        history = trainer.train()

        # 计算LER
        rounds_list = [3, 5, 7, 9, 11, 12]
        predictions_by_rounds = {}
        labels_by_rounds = {}

        model.eval()
        with torch.no_grad():
            for n_rounds in rounds_list:
                test_ds = SurfaceCodeDataset(
                    distance=distance,
                    rounds=n_rounds,
                    p=p,
                    use_soft_readout=True,
                    num_samples=min(10000, num_samples),
                    seed=seed + n_rounds,
                )

                batch = test_ds.get_batch(len(test_ds))
                measurement = batch['measurement'].to(device)
                event = batch['event'].to(device)
                leakage = batch['leakage'].to(device)
                event_leakage = batch['event_leakage'].to(device)
                final_soft = batch['final_soft'].to(device)
                labels = batch['label']

                preds, _ = model.predict(measurement, event, leakage, event_leakage, final_soft)
                predictions_by_rounds[n_rounds] = preds.cpu().numpy()
                labels_by_rounds[n_rounds] = labels.numpy().flatten()

        ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)
        results[distance] = ler_result

        print(f"  LER (d={distance}): {ler_result.ler:.6f}, R²={ler_result.r_squared:.4f}")

    # 计算Λ
    if 3 in results and 5 in results:
        if results[3].is_valid and results[5].is_valid:
            lambda_3_5 = compute_lambda(results[3].ler, results[5].ler)
            print(f"\n  Λ_{{3→5}} = {lambda_3_5:.4f}")
            if lambda_3_5 > 1:
                print("  ✓ Error suppression confirmed (Λ > 1)")
            else:
                print("  ⚠ Warning: Λ ≤ 1, check training")

    print_ler_summary(results)

    print("\n" + "="*60)
    print("PHASE 2 COMPLETED")
    print("="*60)

    return results


def run_phase3(output_dir: Path, debug: bool = False, seed: int = 42):
    """
    Phase 3: 扩展实验 (d=9)
    """
    print("\n" + "="*60)
    print("PHASE 3: Extended Experiments (d=9)")
    print("="*60)

    # 类似Phase 2，但针对d=9
    # 需要更多样本和更长训练

    print("Phase 3 implementation pending...")
    print("Key differences for d=9:")
    print("  - Larger batch size or gradient accumulation")
    print("  - Longer training (50k+ steps)")
    print("  - Lower learning rate (1e-4)")

    return {}


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("AlphaQubit Experiment Runner")
    print("="*60)
    print(f"Phase: {args.phase}")
    print(f"Debug mode: {args.debug}")
    print(f"Output directory: {run_dir}")

    results = {}

    if args.phase in ["1", "all"]:
        results["phase1"] = run_phase1(run_dir, args.debug, args.seed)

    if args.phase in ["2", "all"]:
        results["phase2"] = run_phase2(run_dir, args.debug, args.seed)

    if args.phase in ["3", "all"]:
        results["phase3"] = run_phase3(run_dir, args.debug, args.seed)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to: {run_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
