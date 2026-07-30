#!/usr/bin/env python3
"""Generate PAEMS synthetic QEC training data compliant with
``synthetic_data_spec.md`` (v2.0).

Builds a heterogene-parameterized PAEMS noise model, injects it into the
standard stim rotated-surface-code circuit, and samples spec-conformant
``.pt`` datasets (measurement / event / final_soft / label / detection_events
+ optional leakage / event_leakage).

Run in the conda base environment, e.g.:
    D:/anaconda/python.exe generate_paems_data.py --distance 3 --rounds 25 \\
        --num-samples 50000 --split train --out-dir ..
or use ``--manifest`` to emit the full deliverable set.

Reproducibility: a deterministic seed is derived from (split, distance, rounds)
so re-runs of a given configuration are identical. The per-distance PAEMS
parameter set is generated once and shared across all splits / LER rounds of
that distance (all splits see the *same* device calibration — realistic and
keeps train/val/test comparable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# import the sibling noise-model module
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paems_noise_model as pnm  # noqa: E402
import paems_iq_readout as pir   # <-- [新增] 导入 IQ 与软信息读出的数据管线

import stim
from stream_decoder import make_xzzx_decoder_fn
from xzzx_decoder import XZZXAlphaQubitDecoder


# Split/rounds sizes for the default full manifest (large + multi-distance).
DEFAULT_MANIFEST = [
    # (split, distance, rounds, num_samples)
    ("train", 3, 25, 50000),
    ("train", 5, 25, 50000),
    ("val", 3, 25, 10000),
    ("val", 5, 25, 10000),
    ("test", 3, 25, 10000),
    ("test", 5, 25, 10000),
]
DEFAULT_LER_ROUNDS = [3, 6, 9, 12, 15, 18, 21, 25]
DEFAULT_LER_N = 2000


def split_seed(split: str, distance: int, rounds: int) -> int:
    """Deterministic per-configuration sampling seed."""
    h = (abs(hash((split, int(distance), int(rounds)))) % (2 ** 31)) + 1
    return int(h)


def param_seed(distance: int) -> int:
    """Deterministic per-distance parameter seed (shared across splits)."""
    return int(distance) * 7919 + 42


def generate_one(split: str, distance: int, rounds: int, num_samples: int,
                 out_dir: Path, *, params_cache: dict, chunk_size: int,
                 include_leakage: bool, snr: float, t: float,
                 dry_run: bool = False) -> Path:
    """Generate a single spec-named ``.pt`` file. Returns its path."""
    if distance not in params_cache:
        # Generate + persist the shared per-distance parameter set.
        params = pnm.generate_paems_params(distance, seed=param_seed(distance))
        params_file = out_dir / "params" / f"paems_params_d{distance}.json"
        params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        params_cache[distance] = params
        print(f"[params] wrote {params_file}  "
              f"(qubits={len(params['qubits'])}, cx_gates={len(params['cx_gates'])}, "
              f"xtalk_pairs={len(params['crosstalk_pairs'])})")
    params = params_cache[distance]

    seed = split_seed(split, distance, rounds)
    fname = f"{split}_d{distance}_r{rounds}_n{num_samples}.pt"
    out_path = out_dir / fname

    if dry_run:
        print(f"[dry-run] would generate {out_path}  (seed={seed})")
        return out_path

    print(f"[gen] {fname}  seed={seed} leakage={include_leakage} (IQ mode) ...")
    t0 = time.time()

    # 1. 调用新的 IQ 版采样函数
    # 底层依然使用 params，确保门噪声、闲置噪声、串扰等原始 PAEMS 物理参数得到贯彻。
    # 唯有测量阶段替换连续变量采样，软/硬测量均从统一的 IQ 数据推演，保证强自洽。
    data = pir.sample_paems_iq_dataset(
        distance=distance, rounds=rounds, num_samples=num_samples,
        params=params, snr=snr, t=t, seed=seed,
        chunk_size=chunk_size, include_leakage=include_leakage,
    )

    # 2. 调用支持复数存储 (complex64) 的 save_pt
    pir.save_pt(data, out_path)

    dt = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[gen] {fname}  done in {dt:.1f}s  ({size_mb:.1f} MB)  "
          f"label_rate={float(data['label'].mean()):.4f}  "
          f"det_dens={float(data['detection_events'].mean()):.4f}  "
          f"leak_frac={float(data['leakage'].mean()):.4f}  "
          f"num_det={data['detection_events'].shape[1]}")
    return out_path

import gc as _gc

#── 流式生成器核心 ────────────────────────────────────────────────────────────
def paems_chunk_generator(split, distance, rounds, num_samples, params, *,
                           chunk_size, include_leakage, snr, t):
    """
    每次yield 一个 chunk 字典；yield 返回后内部主动 del + gc.collect()。
    子seed 由父seed 派生，与一次性生成在统计上等价。
    """
    parent_seed = split_seed(split, distance, rounds)
    rng = np.random.default_rng(parent_seed)
    samples_done = 0

    while samples_done < num_samples:
        batch_n= min(chunk_size, num_samples - samples_done)
        batch_seed = int(rng.integers(0, 2 ** 31))

        chunk = pnm.sample_paems_dataset(
            distance=distance, rounds=rounds, num_samples=batch_n,
            params=params, snr=snr, t=t,
            seed=batch_seed, chunk_size=batch_n,
            include_leakage=include_leakage,
        )

        yield chunk

        del chunk
        _gc.collect()
        samples_done += batch_n

def stream_and_decode(split, distance, rounds, num_samples, decoder_fn, *,
                      params_cache, chunk_size, include_leakage, snr, t):
    """
    流式处理：生成 → decoder_fn(chunk, chunk_idx) → 丢弃，全程不落盘。

    decoder_fn(chunk: dict, chunk_idx: int) -> Nonechunk 键：measurement, event, final_soft, label,
                detection_events[, leakage, event_leakage]
      处理完直接 return，不要把 chunk 存到外部变量。
    """
    params = _ensure_params(distance, params_cache)

    print(f"[stream] {split}_d{distance}_r{rounds}_n{num_samples} "
          f"chunk={chunk_size} leakage={include_leakage}")
    t0 = time.time()
    samples_done = 0

    for chunk_idx, chunk in enumerate(
        paems_chunk_generator(split, distance, rounds, num_samples, params,
                               chunk_size=chunk_size, include_leakage=include_leakage,
                               snr=snr, t=t)
    ):
        batch_n = int(chunk['label'].shape[0])
        decoder_fn(chunk, chunk_idx)
        samples_done += batch_n
        print(f"  chunk {chunk_idx:4d}: {samples_done:>7d}/{num_samples} "
              f"label_rate={float(chunk['label'].mean()):.4f}  "
              f"det_dens={float(chunk['detection_events'].mean()):.4f}")

    print(f"[stream] done in {time.time() - t0:.1f}s(0 MB written)\n")

def _ensure_params(distance, params_cache, params_dir=None):
    if distance not in params_cache:
        params = pnm.generate_paems_params(distance, seed=param_seed(distance))
        if params_dir is not None:
            pf = params_dir / f"paems_params_d{distance}.json"
            pf.parent.mkdir(parents=True, exist_ok=True)
            with open(pf, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)
        params_cache[distance] = params
    return params_cache[distance]

def build_manifest(scale: int = 1) -> list:
    """Build the deliverable manifest, scaling every sample count by ``scale``.

    ``scale=1`` reproduces the v1 baseline exactly. ``scale>1`` keeps everything
    identical (same distances, rounds, LER round-points, same per-distance
    PAEMS parameter seed -> same virtual chip) — only the number of shots grows.
    """
    rows = [(s, d, r, n * scale) for (s, d, r, n) in DEFAULT_MANIFEST]
    ler_n = DEFAULT_LER_N * scale
    for d in (3, 5):
        for r in DEFAULT_LER_ROUNDS:
            rows.append(("ler", d, r, ler_n))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Generate PAEMS synthetic QEC data.")
    ap.add_argument("--manifest",    action="store_true")
    ap.add_argument("--split",       type=str,   default=None)
    ap.add_argument("--distance",    type=int,   default=3)
    ap.add_argument("--rounds",      type=int,   default=25)
    ap.add_argument("--num-samples", type=int,   default=50000)
    ap.add_argument("--chunk-size",  type=int,   default=4000)
    ap.add_argument("--out-dir",     type=str,
                    default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--snr",         type=float, default=SOFT_SNR)
    ap.add_argument("--t",           type=float, default=SOFT_T)
    ap.add_argument("--no-leakage",  action="store_true")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--scale",       type=int,   default=1)
    ap.add_argument('--stream',action='store_true',
                    help='流式模式：生成→XZZX解码→丢弃，不写.pt')
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='XZZXAlphaQubitDecoder 检查点路径')
    ap.add_argument('--device', type=str, default='cuda')ap.add_argument('--chunk-size', type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    include_leakage = not args.no_leakage
    params_cache: dict = {}

# ── 流式模式（不落盘）────────────────────────────────────────────────────
    if args.stream:
        print(f"[stream] d{args.distance} r{args.rounds} {args.basis} "
              f"N={args.num_samples} chunk={args.chunk_size}")

        # ── 构建 XZZX 坐标系（从 Google 模板电路） ───────────────────────────
        circuit_path = (GOOGLE_SC
                        / f"d{args.distance}_at_{GOOGLE_PATCH[args.distance]}"
                        / args.basis
                        / f"r{args.rounds:02d}"
                        / "circuit_ideal.stim")
        ideal_cir    = stim.Circuit.from_file(str(circuit_path))
        coord_system = XZZXCoordinateSystem(args.distance, ideal_cir)
        print(f"[coord] grid={coord_system.grid_size}×{coord_system.grid_size}"
              f"n_stab={coord_system.n_stab}  n_data={coord_system.n_data}")

        # ── 实例化 XZZX 解码器 ────────────────────────────────────────────────
        xzzx_model = XZZXAlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=256,
        )
        decoder_fn = make_xzzx_decoder_fn(
            model=xzzx_model,
            device=getattr(args, "device", "cuda"),
            log_interval=5,
            checkpoint_path=getattr(args, "checkpoint", None),
        )

        # ── 流式生成 + 解码 ───────────────────────────────────────────────────
        samples_done = 0
        for chunk_idx, chunk in enumerate(stream_one(
                args.distance, args.rounds, args.basis, args.num_samples,
                config_path=cfg,
                snr=args.snr, t=args.t,
                seed=args.seed, chunk_size=args.chunk_size,
            )
        ):
            decoder_fn(chunk, chunk_idx)      # 推理 + 累积 LER，不落盘
            samples_done += chunk["detection_events"].shape[0]
            print(
                f"  chunk {chunk_idx:4d}: {samples_done:>7d}/{args.num_samples}  "
                f"det_dens={float(chunk['detection_events'].mean()):.4f}"
            )

        # ── 打印最终结果 ──────────────────────────────────────────────────────
        s = decoder_fn.state
        print("=" * 60)
        print(f"[stream 完成]  总样本={s['total']:,}")
        print(f"  最终 LER  = {s['logical_error_rate']:.6f}")
        print(f"  正确预测  = {s['correct']:,} / {s['total']:,}")
        print("=" * 60)
        return

if __name__ == "__main__":
    main()
