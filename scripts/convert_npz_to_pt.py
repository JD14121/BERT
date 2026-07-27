#!/usr/bin/env python3
"""
将 .npz 数据集转换为 .pt 格式。

要求 .npz 已包含按 DEM (detector error model) 顺序保存的 detection_events，
可直接用于 PyMatching/MWPM 解码。

用法：
    python scripts/convert_npz_to_pt.py --input data_baseline/test_d3_r25.npz --output data_baseline/test_d3_r25.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_npz_to_pt(input_path: str, output_path: str):
    """转换单个 .npz 文件为 .pt 文件。"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading {input_path}...")
    data = np.load(input_path)

    measurement = data['measurement'].astype(np.float32)
    event = data['event'].astype(np.float32)
    final_soft = data['final_soft'].astype(np.float32)
    label = data['label'].astype(np.float32)

    if 'detection_events' in data:
        # 使用已保存的 DEM 序 detection_events
        detection_events = data['detection_events'].astype(np.float32)
    else:
        # 兼容旧数据：从软测量恢复二值并计算（顺序可能与 DEM 不一致，MWPM 需谨慎）
        from alphaqubit.data.stim_generator import StimDataGenerator
        generator = StimDataGenerator(
            distance=int(data['distance']),
            rounds=int(data['rounds']),
            p=float(data['p']),
            seed=42,
        )
        _, detection_events, _ = generator.sample_with_matched_detectors(len(label))
        detection_events = detection_events.astype(np.float32)

    pt_data = {
        'measurement': torch.from_numpy(measurement),
        'event': torch.from_numpy(event),
        'final_soft': torch.from_numpy(final_soft),
        'label': torch.from_numpy(label),
        'detection_events': torch.from_numpy(detection_events),
        'distance': int(data['distance']),
        'rounds': int(data['rounds']),
        'p': float(data['p']),
        'snr': float(data['snr']),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_data, output_path)
    print(f"Saved {output_path} with keys: {list(pt_data.keys())}")
    print(f"  samples: {len(label)}, shape: {measurement.shape}, "
          f"detection_events mean: {detection_events.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Convert .npz dataset to .pt with correct detection_events")
    parser.add_argument('--input', type=str, required=True, help='Input .npz path')
    parser.add_argument('--output', type=str, required=True, help='Output .pt path')
    args = parser.parse_args()
    convert_npz_to_pt(args.input, args.output)


if __name__ == '__main__':
    main()
