#!/usr/bin/env python3
"""
AlphaQubit验证检查脚本

运行所有验证检查点（基于alphaqubit_plan.md第7节）

使用示例:
    python scripts/validate_all.py
    python scripts/validate_all.py --verbose
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def section_header(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_pass(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"           {details}")
    return passed


def validate_data_generation(verbose: bool = False):
    """7.1 数据生成验证"""
    section_header("7.1 Data Generation Validation")

    from alphaqubit.data import StimDataGenerator, SoftReadoutSimulator

    all_passed = True

    # Event density
    print("\nChecking event density...")
    gen = StimDataGenerator(distance=3, rounds=12, p=0.005, seed=42)
    stats = gen.get_event_statistics(num_shots=10000)

    density = stats['event_density']
    passed = 0.001 < density < 0.2
    all_passed &= check_pass(
        "Event density in range [0.1%, 20%]",
        passed,
        f"Actual: {density:.4f}"
    )

    # Observable flip rate
    flip_rate = stats['observable_flip_rate']
    passed = 0 <= flip_rate <= 1
    all_passed &= check_pass(
        "Observable flip rate in [0, 1]",
        passed,
        f"Actual: {flip_rate:.4f}"
    )

    # Stim detector consistency
    print("\nChecking Stim consistency...")
    validation = gen.validate_against_stim(num_shots=1000)
    match_rate = validation['stim_match_rate']
    passed = match_rate > 0.95
    all_passed &= check_pass(
        "Detection events match Stim (>95%)",
        passed,
        f"Match rate: {match_rate:.4f}"
    )

    # Soft readout distribution
    print("\nChecking soft readout distribution...")
    sim = SoftReadoutSimulator(snr=10.0, t=0.01)
    dist_stats = sim.get_distribution_stats(num_samples=10000)

    bimodal = dist_stats['soft_given_0_mean'] < 0.3 and dist_stats['soft_given_1_mean'] > 0.7
    all_passed &= check_pass(
        "Soft readout is bimodal",
        bimodal,
        f"P(soft|0)={dist_stats['soft_given_0_mean']:.3f}, P(soft|1)={dist_stats['soft_given_1_mean']:.3f}"
    )

    return all_passed


def validate_coordinate_system(verbose: bool = False):
    """7.2 坐标系统验证"""
    section_header("7.2 Coordinate System Validation")

    from alphaqubit.data import CoordinateSystem

    all_passed = True

    for distance in [3, 5, 7]:
        print(f"\nTesting distance={distance}...")
        coord = CoordinateSystem(distance)

        # Scatter/Gather round-trip
        B, D = 4, 8
        original = torch.randn(B, coord.n_stab, D)
        scattered = coord.scatter(original)
        recovered = coord.gather(scattered)
        passed = torch.allclose(original, recovered)
        all_passed &= check_pass(
            f"Scatter→Gather round-trip (d={distance})",
            passed
        )

        # scatter_idx range
        max_idx = coord.grid_size ** 2 - 1
        in_range = coord.scatter_idx.max() <= max_idx and coord.scatter_idx.min() >= 0
        all_passed &= check_pass(
            f"Scatter indices in valid range (d={distance})",
            in_range
        )

        # Grid mask count
        correct_count = coord.grid_mask.sum().item() == coord.n_stab
        all_passed &= check_pass(
            f"Grid mask has correct count (d={distance})",
            correct_count,
            f"Expected {coord.n_stab}, got {coord.grid_mask.sum().item()}"
        )

    # 2D round-trip
    print("\nTesting 2D operations...")
    coord = CoordinateSystem(5)
    B, D = 2, 4
    original = torch.randn(B, coord.grid_size ** 2, D)
    as_2d = coord.to_2d(original)
    recovered = coord.from_2d(as_2d)
    passed = torch.allclose(original, recovered)
    all_passed &= check_pass("to_2d→from_2d round-trip", passed)

    # Padding reset
    print("\nTesting padding reset...")
    lattice = torch.ones(B, coord.grid_size ** 2, D)
    pad_vec = torch.zeros(D)
    result = coord.reset_padding(lattice, pad_vec)

    # Check padding positions are zero
    padding_mask = ~coord.grid_mask
    padding_values = result[:, padding_mask, :]
    stab_values = coord.gather(result)

    passed = (torch.allclose(padding_values, torch.zeros_like(padding_values)) and
              torch.allclose(stab_values, torch.ones_like(stab_values)))
    all_passed &= check_pass("Padding reset correctness", passed)

    return all_passed


def validate_model_sanity(verbose: bool = False):
    """7.3 模型Sanity Check"""
    section_header("7.3 Model Sanity Check")

    from alphaqubit.data import SurfaceCodeDataset
    from alphaqubit.models import AlphaQubitDecoderConfig

    all_passed = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Create dataset and model
    print("\nInitializing...")
    ds = SurfaceCodeDataset(distance=3, rounds=5, p=0.005, num_samples=1000)
    model = AlphaQubitDecoderConfig.tiny(ds.coord_system)
    model = model.to(device)

    # Get batch
    batch = ds.get_batch(64)
    measurement = batch['measurement'].to(device)
    event = batch['event'].to(device)
    leakage = batch['leakage'].to(device)
    event_leakage = batch['event_leakage'].to(device)
    final_soft = batch['final_soft'].to(device)
    labels = batch['label'].to(device)

    # Initial loss check
    print("\nChecking initial loss...")
    model.eval()
    with torch.no_grad():
        logits = model(measurement, event, leakage, event_leakage, final_soft)
        initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        ).item()

    passed = 0.5 < initial_loss < 1.0
    all_passed &= check_pass(
        "Initial loss ~0.693 (ln2)",
        passed,
        f"Actual: {initial_loss:.4f}"
    )

    # Training reduces loss
    print("\nChecking training reduces loss...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        logits = model(measurement, event, leakage, event_leakage, final_soft)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    passed = losses[-1] < losses[0]
    all_passed &= check_pass(
        "Loss decreases after 100 steps",
        passed,
        f"Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}"
    )

    # Gradient norm check
    print("\nChecking gradient norms...")
    grad_norms = []
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(measurement, event, leakage, event_leakage, final_soft)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        grad_norms.append(grad_norm.item())

    all_finite = all(np.isfinite(g) for g in grad_norms)
    all_passed &= check_pass(
        "Gradient norms are finite",
        all_finite,
        f"Range: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]"
    )

    # Overfit small dataset
    print("\nChecking overfitting on small dataset...")
    small_batch = ds.get_batch(16)
    measurement_small = small_batch['measurement'].to(device)
    event_small = small_batch['event'].to(device)
    leakage_small = small_batch['leakage'].to(device)
    event_leakage_small = small_batch['event_leakage'].to(device)
    final_soft_small = small_batch['final_soft'].to(device)
    labels_small = small_batch['label'].to(device)

    model_overfit = AlphaQubitDecoderConfig.tiny(ds.coord_system).to(device)
    optimizer = torch.optim.AdamW(model_overfit.parameters(), lr=1e-2)

    for _ in range(500):
        optimizer.zero_grad()
        logits = model_overfit(measurement_small, event_small, leakage_small, event_leakage_small, final_soft_small)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_small)
        loss.backward()
        optimizer.step()

    model_overfit.eval()
    with torch.no_grad():
        preds, _ = model_overfit.predict(measurement_small, event_small, leakage_small, event_leakage_small, final_soft_small)
        accuracy = (preds == labels_small.squeeze().long()).float().mean().item()

    passed = accuracy > 0.9
    all_passed &= check_pass(
        "Can overfit small dataset (>90% acc)",
        passed,
        f"Accuracy: {accuracy:.4f}"
    )

    return all_passed


def validate_evaluation(verbose: bool = False):
    """7.4 评估验证"""
    section_header("7.4 Evaluation Validation")

    from alphaqubit.evaluation import compute_ler, compute_fidelity, fit_ler

    all_passed = True

    # Test fidelity calculation
    print("\nChecking fidelity calculation...")
    assert compute_fidelity(0.0) == 1.0
    assert compute_fidelity(0.5) == 0.0
    assert compute_fidelity(0.25) == 0.5
    all_passed &= check_pass("Fidelity calculation correct", True)

    # Test LER fitting with synthetic data
    print("\nChecking LER fitting with synthetic data...")
    # Create synthetic data: E(n) ≈ 0.5 * (1 - (1-2ε)^n) where ε = 0.01
    true_ler = 0.01
    rounds_list = [3, 6, 9, 12]
    fidelities = [(1 - 2 * true_ler) ** n for n in rounds_list]

    ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, fidelities)

    passed = abs(ler - true_ler) < 0.001
    all_passed &= check_pass(
        "LER fitting accuracy",
        passed,
        f"True LER: {true_ler:.4f}, Fitted: {ler:.4f}"
    )

    passed = r_squared > 0.99
    all_passed &= check_pass(
        "LER fit R² > 0.99 for perfect data",
        passed,
        f"R² = {r_squared:.4f}"
    )

    # Test with noisy data
    print("\nChecking LER fitting with noisy data...")
    np.random.seed(42)
    noisy_fidelities = [f + np.random.normal(0, 0.02) for f in fidelities]
    noisy_fidelities = [max(0.01, f) for f in noisy_fidelities]  # Ensure positive

    ler, r_squared, log_f0, slope, is_valid = fit_ler(rounds_list, noisy_fidelities)

    passed = r_squared > 0.9
    all_passed &= check_pass(
        "LER fit R² > 0.9 for noisy data",
        passed,
        f"R² = {r_squared:.4f}"
    )

    return all_passed


def validate_baseline(verbose: bool = False):
    """7.5 Baseline验证"""
    section_header("7.5 Baseline Validation")

    all_passed = True

    try:
        from alphaqubit.experiments.baselines import MWPMBaseline

        print("\nChecking MWPM baseline...")
        mwpm = MWPMBaseline(seed=42)
        result = mwpm.evaluate_single(
            distance=3,
            rounds=12,
            noise_p=0.005,
            num_samples=5000,
        )

        # Error rate should be between 0 and 0.5 (better than random)
        passed = 0 < result.error_rate < 0.5
        all_passed &= check_pass(
            "MWPM error rate in reasonable range",
            passed,
            f"Error rate: {result.error_rate:.4f}"
        )

        # Check LER calculation
        print("\nChecking MWPM LER...")
        ler, error_rates = mwpm.compute_ler(
            distance=3,
            noise_p=0.005,
            rounds_list=[3, 6, 9, 12],
            num_samples=2000,
        )

        passed = 0 < ler < 0.1
        all_passed &= check_pass(
            "MWPM LER in reasonable range",
            passed,
            f"LER: {ler:.6f}"
        )

    except ImportError:
        print("  ⚠ PyMatching not installed, skipping baseline validation")
        all_passed &= check_pass("PyMatching available", False, "Please install: pip install pymatching")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run all validation checks")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("  AlphaQubit Validation Checklist")
    print("  Based on alphaqubit_plan.md Section 7")
    print("="*60)

    results = {}

    results["data_generation"] = validate_data_generation(args.verbose)
    results["coordinate_system"] = validate_coordinate_system(args.verbose)
    results["model_sanity"] = validate_model_sanity(args.verbose)
    results["evaluation"] = validate_evaluation(args.verbose)
    results["baseline"] = validate_baseline(args.verbose)

    # Summary
    section_header("VALIDATION SUMMARY")
    all_passed = True
    for name, passed in results.items():
        passed = bool(passed)  # Ensure boolean type
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed

    print("\n" + "="*60)
    if all_passed:
        print("  ALL VALIDATIONS PASSED ✓")
    else:
        print("  SOME VALIDATIONS FAILED ✗")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
