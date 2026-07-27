#!/usr/bin/env python3
"""generate_manifest.py (P3): 按分布 A 量产单码距数据。
每码距: train 800k + val 100k + test 100k @r10 + LER {1,10,13,30,50}×20k
XZZX, Google 校准配置 (configs/calibrated_d{d}.json), 软信息, .pt + .b8
用法: python generate_manifest.py --distance 3   # 然后 5, 7
"""
import sys, time, argparse, hashlib
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from path_config import DATA_DIR, CONFIG_DIR, GOOGLE_ROUNDS
from generate_google_paems_data import generate_one, save_pt, export_b8

# 分布 A：主数据 r10 (train/val/test) + LER 5 点
MANIFEST = [
    ("train", 10, 800000),
    ("val",   10, 100000),
    ("test",  10, 100000),
    ("ler",    1, 20000),
    ("ler",   10, 20000),
    ("ler",   13, 20000),
    ("ler",   30, 20000),
    ("ler",   50, 20000),
]


def split_seed(split, d, rounds):
    """确定性种子（hashlib.sha256，跨进程可复现；替代非确定的 Python hash())."""
    h = hashlib.sha256(f"{split}_{int(d)}_{int(rounds)}".encode("utf-8")).hexdigest()
    return (int(h, 16) % (2 ** 31)) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--t", type=float, default=0.01)
    ap.add_argument("--scale", type=int, default=1, help="放大 train/val/test 样本数（LER 保持 20k 不放大）")
    args = ap.parse_args()
    d = args.distance
    cfg = CONFIG_DIR / f"calibrated_d{d}.json"
    assert cfg.exists(), f"校准配置不存在: {cfg}（先跑 P2 calibrate_to_google.py）"
    out_dir = DATA_DIR / f"d{d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P3 manifest] d{d} {args.basis} cfg={cfg.name} out={out_dir}", flush=True)
    t_all = time.time()
    for i, (split, rounds, n) in enumerate(MANIFEST, 1):
        assert rounds in GOOGLE_ROUNDS, f"rounds={rounds} 不在 Google 可用集 {GOOGLE_ROUNDS}"
        n_eff = n * args.scale if split != "ler" else n   # LER 保持 20k 不放大
        name = f"{split}_d{d}_r{rounds}_n{n_eff}_{args.basis}"
        seed = split_seed(split, d, rounds)
        t0 = time.time()
        data = generate_one(d, rounds, args.basis, n_eff, config_path=cfg,
                            snr=args.snr, t=args.t, seed=seed, chunk_size=args.chunk_size)
        save_pt(data, out_dir / f"{name}.pt")
        export_b8(data, out_dir, name)
        dt = time.time() - t0
        sz = (out_dir / f"{name}.pt").stat().st_size / 1e6
        print(f"[{i}/{len(MANIFEST)}] {name}: {n_eff} samples in {dt:.0f}s  "
              f"label_rate={float(data['label'].mean()):.4f} det_dens={float(data['detection_events'].mean()):.4f} "
              f"size={sz:.1f}MB", flush=True)
        # 释放
        del data
    print(f"[P3 d{d} DONE] total {time.time() - t_all:.0f}s  out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
