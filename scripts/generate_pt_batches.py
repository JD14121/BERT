#!/usr/bin/env python3
"""
Generate pre-batched training data as individual .pt files.

Each .pt file = one batch of samples, ready for DataLoader.
Much faster than .npz for large datasets since files load incrementally.

Usage:
    python scripts/generate_pt_batches.py --distance 3 --rounds 12 --num-batches 10000 --batch-size 512
"""
import argparse, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from alphaqubit.data.stim_generator import StimDataGenerator
from alphaqubit.data.soft_readout import SoftReadoutSimulator


def generate_one_batch(args_tuple):
    """Generate one batch in a subprocess."""
    batch_id, distance, rounds, p, batch_size, seed, snr, t = args_tuple

    generator = StimDataGenerator(distance=distance, rounds=rounds, p=p, seed=seed + batch_id * 1000)
    soft_sim = SoftReadoutSimulator(snr=snr, t=t)

    # Sample
    raw_measurements, stim_observables = generator.sample(batch_size)
    ancilla_meas = generator.extract_ancilla_measurements(raw_measurements)
    final_data = generator.extract_final_data(raw_measurements)

    # Soft readout (single I/Q sample)
    soft_meas = soft_sim.simulate(ancilla_meas)
    shots, T, n_stab = soft_meas.shape
    soft_events = np.zeros_like(soft_meas)
    soft_events[:, 0, :] = soft_meas[:, 0, :]
    for t_idx in range(1, T):
        soft_events[:, t_idx, :] = SoftReadoutSimulator.compute_soft_event(
            soft_meas[:, t_idx, :], soft_meas[:, t_idx - 1, :])

    soft_final = soft_sim.simulate(final_data.astype(bool))

    return {
        'measurement': torch.from_numpy(soft_meas.astype(np.float32)),
        'event': torch.from_numpy(soft_events.astype(np.float32)),
        'leakage': torch.zeros(batch_size, T, n_stab, dtype=torch.float32),
        'event_leakage': torch.zeros(batch_size, T, n_stab, dtype=torch.float32),
        'final_soft': torch.from_numpy(soft_final.astype(np.float32)),
        'label': torch.from_numpy(stim_observables.astype(np.float32)).unsqueeze(-1),
        'batch_id': batch_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--num-batches", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/pt_batches")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = args.num_batches * args.batch_size
    print(f"Generating {args.num_batches} batches × {args.batch_size} = {total_samples:,} samples")
    print(f"Output: {output_dir}/")
    print(f"Workers: {args.num_workers}")

    tasks = [
        (i, args.distance, args.rounds, args.p, args.batch_size, args.seed, args.snr, args.t)
        for i in range(args.num_batches)
    ]

    start = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(generate_one_batch, t): t[0] for t in tasks}
        for f in as_completed(futures):
            batch = f.result()
            bid = batch.pop('batch_id')
            torch.save(batch, output_dir / f"batch_{bid:06d}.pt")
            done += 1
            if done % 100 == 0 or done == args.num_batches:
                elapsed = time.time() - start
                rate = (done * args.batch_size) / elapsed
                eta = (args.num_batches - done) * elapsed / done if done > 0 else 0
                print(f"\r  {done}/{args.num_batches} batches | {rate:.0f} samples/s | ETA {eta:.0f}s", end="")

    elapsed = time.time() - start
    print(f"\nDone! {args.num_batches} batches in {elapsed:.1f}s ({total_samples/elapsed:.0f} samples/s)")

    # Save metadata
    meta = {'distance': args.distance, 'rounds': args.rounds, 'p': args.p,
            'snr': args.snr, 'batch_size': args.batch_size, 'num_batches': args.num_batches}
    torch.save(meta, output_dir / "metadata.pt")
    print(f"Metadata saved to {output_dir}/metadata.pt")


if __name__ == "__main__":
    main()
