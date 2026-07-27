#!/usr/bin/env python3
"""
预生成训练数据集

使用多进程加速Stim数据生成，保存到磁盘。
训练时直接加载，无需实时生成。

用法:
    python scripts/generate_dataset.py --distance 3 --rounds 25 --num-samples 1000000
    python scripts/generate_dataset.py --distance 5 --rounds 25 --num-samples 1000000
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaqubit.data.stim_generator import StimDataGenerator
from alphaqubit.data.soft_readout import SoftReadoutSimulator


def generate_chunk(args):
    """生成一个数据块（在子进程中运行）"""
    chunk_id, distance, rounds, p, chunk_size, seed, snr, t = args

    # 每个进程有独立的随机种子
    generator = StimDataGenerator(
        distance=distance,
        rounds=rounds,
        p=p,
        seed=seed + chunk_id * 1000,
    )
    soft_sim = SoftReadoutSimulator(snr=snr, t=t)

    # 采样数据：测量、DEM序 detection_events、observable 三者来自同一次底层采样
    raw_measurements, dem_detection_events, stim_observables = generator.sample_with_matched_detectors(chunk_size)

    # 提取各部分
    ancilla_meas = generator.extract_ancilla_measurements(raw_measurements)
    final_data = generator.extract_final_data(raw_measurements)

    # 软读出转换 — 采样一次I/Q噪声，保持measurement和event一致性
    soft_meas = soft_sim.simulate(ancilla_meas)

    # 从soft_meas计算soft events（不重新采样I/Q噪声！）
    shots, T, n_stab = soft_meas.shape
    soft_events = np.zeros_like(soft_meas)
    soft_events[:, 0, :] = soft_meas[:, 0, :]
    for t in range(1, T):
        soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
            soft_meas[:, t, :], soft_meas[:, t - 1, :],
        )

    soft_final = soft_sim.simulate(final_data.astype(bool))

    return {
        'measurement': soft_meas.astype(np.float32),
        'event': soft_events.astype(np.float32),
        'final_soft': soft_final.astype(np.float32),
        'label': stim_observables.astype(np.float32),
        'detection_events': dem_detection_events.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-generate training dataset")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--t", type=float, default=0.01)
    parser.add_argument("--num-samples", type=int, default=1000000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom output filename (default: d{d}_r{rounds}_n{num_samples}.npz)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 计算需要生成的chunk数量
    num_chunks = (args.num_samples + args.chunk_size - 1) // args.chunk_size

    print("=" * 60)
    print("Pre-generating Training Dataset")
    print("=" * 60)
    print(f"Distance: {args.distance}")
    print(f"Rounds: {args.rounds}")
    print(f"Physical error rate: {args.p}")
    print(f"Total samples: {args.num_samples:,}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of workers: {args.num_workers}")
    print("=" * 60)

    # 准备任务参数
    tasks = [
        (i, args.distance, args.rounds, args.p, args.chunk_size,
         args.seed, args.snr, args.t)
        for i in range(num_chunks)
    ]

    # 收集所有数据（不再保存全零的leakage/event_leakage，节省50%空间）
    all_data = {
        'measurement': [],
        'event': [],
        'final_soft': [],
        'label': [],
        'detection_events': [],
    }

    start_time = time.time()
    completed = 0

    # 使用多进程生成
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(generate_chunk, task): i for i, task in enumerate(tasks)}

        for future in as_completed(futures):
            chunk_data = future.result()
            for key in all_data:
                all_data[key].append(chunk_data[key])

            completed += 1
            elapsed = time.time() - start_time
            samples_done = completed * args.chunk_size
            rate = samples_done / elapsed
            eta = (args.num_samples - samples_done) / rate if rate > 0 else 0

            print(f"\rProgress: {completed}/{num_chunks} chunks "
                  f"({samples_done:,}/{args.num_samples:,} samples) "
                  f"| Rate: {rate:.0f} samples/s | ETA: {eta:.0f}s", end="")

    print()

    # 合并数据
    print("Concatenating data...")
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)[:args.num_samples]

    # 保存
    if args.output_name:
        output_file = output_dir / args.output_name
    else:
        output_file = output_dir / f"d{args.distance}_r{args.rounds}_n{args.num_samples}.npz"
    print(f"Saving to {output_file}...")

    np.savez(
        output_file,
        measurement=all_data['measurement'],
        event=all_data['event'],
        final_soft=all_data['final_soft'],
        label=all_data['label'],
        detection_events=all_data['detection_events'],
        distance=np.array(args.distance),
        rounds=np.array(args.rounds),
        p=np.array(args.p),
        snr=np.array(args.snr),
    )

    elapsed = time.time() - start_time
    file_size = output_file.stat().st_size / (1024**2)

    print("=" * 60)
    print(f"Done! Generated {args.num_samples:,} samples in {elapsed:.1f}s")
    print(f"File size: {file_size:.1f} MB")
    print(f"Rate: {args.num_samples / elapsed:.0f} samples/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
