#!/usr/bin/env python3
"""
预生成训练数据集

使用多进程加速Stim数据生成，保存到磁盘。
训练时直接加载，无需实时生成。

用法:
    python scripts/generate_dataset.py --distance 3 --rounds 25 --num-samples 1000000
    python scripts/generate_dataset.py --distance 5 --rounds 25 --num-samples 1000000
"""

# ── 顶部新增导入（与其他 import 放在一起） ───────────────────────────────
    # from stream_decoder import make_xzzx_decoder_fn
    # from xzzx_decoder import XZZXAlphaQubitDecoder
    # 在 main() 开头新增 argparse 参数：
    #   parser.add_argument("--checkpoint", type=str, default=None)
    #   parser.add_argument("--device", type=str, default="cuda")
    #   parser.add_argument("--decoder-only", action="store_true")

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaqubit.data.stim_generator import StimDataGenerator
from alphaqubit.data.soft_readout import SoftReadoutSimulator

from stream_decoder import make_xzzx_decoder_fn
from xzzx_decoder import XZZXAlphaQubitDecoder
from alphaqubit.data.coordinate_system import CoordinateSystem

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
    parser = argparse.ArgumentParser(description="Stream-generate training dataset (no disk)")
    parser.add_argument("--distance",    type=int,   default=3)
    parser.add_argument("--rounds",      type=int,   default=25)
    parser.add_argument("--p",           type=float, default=0.005)
    parser.add_argument("--snr",         type=float, default=10.0)
    parser.add_argument("--t",           type=float, default=0.01)
    parser.add_argument("--num-samples", type=int,   default=1000000)
    parser.add_argument("--chunk-size",  type=int,   default=10000)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--checkpoint",type=str,   default=None,
                        help="XZZXAlphaQubitDecoder 检查点路径（None=随机初始化）")
    parser.add_argument("--device",       type=str,   default="cuda",
                        help="推理设备 (cuda/cpu)")
    parser.add_argument("--decoder-only", action="store_true",
                        help="流式模式：生成→XZZX解码→丢弃，不写任何文件")
    args = parser.parse_args()

    generator= StimDataGenerator(
        distance=args.distance, rounds=args.rounds,
        p=args.p, seed=args.seed)
    soft_sim= SoftReadoutSimulator(snr=args.snr, t=args.t)

    # ── 构建坐标系 + 解码器 ───────────────────────────────────────────────────
    base_circ    = generator.circuit# StimDataGenerator 持有电路
    coord_system = CoordinateSystem(args.distance, base_circ)

    xzzx_model = XZZXAlphaQubitDecoder(
        coord_system=coord_system,
        embed_dim=256,
    )
    decoder_fn = make_xzzx_decoder_fn(
        model=xzzx_model,
        device=args.device,
        log_interval=10,
        checkpoint_path=args.checkpoint,
    )

    print("=" * 60)
    print(f"Streaming — XZZX Decoder  "
          f"({'no disk' if args.decoder_only else 'decode only'})")
    print(f"Distance={args.distance}Rounds={args.rounds}  N={args.num_samples:,}")
    print(f"Checkpoint: {args.checkpoint or '(随机初始化)'}")
    print("=" * 60)

    start_time = time.time()

    for chunk_idx, chunk in chunk_stream():
        # ── 调用 XZZX 解码器推理，不落盘 ─────────────────────────────────────
        decoder_fn(chunk, chunk_idx)

        samples_done = decoder_fn.state["total"]
        print(
            f"\rchunk {chunk_idx:4d}: {samples_done:>9,}/{args.num_samples:,}  "
            f"running_LR={decoder_fn.state['logical_error_rate']:.5f}",
            end=""
        )

    # ── 最终汇总 ──────────────────────────────────
    s = decoder_fn.state
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[完成] {args.num_samples:,} 样本  耗时 {elapsed:.1f}s")
    print(f"  最终 LER      = {s['logical_error_rate']:.6f}")
    print(f"  正确预测      = {s['correct']:,} / {s['total']:,}")
    print(f"  吞吐量        = {s['total'] / elapsed:.0f} samples/s")
    print(f"  写盘字节      = 0MB（流式模式）")
    print("=" * 60)

if __name__ == "__main__":
    main()
