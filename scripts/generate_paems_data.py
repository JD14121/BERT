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
    ap.add_argument("--manifest", action="store_true",
                    help="Generate the full default deliverable set (train/val/test + LER sweeps for d3 and d5).")
    ap.add_argument("--split", type=str, default=None,
                    help="Single-file split name (train/val/test/ler). Requires --distance/--rounds/--num-samples.")
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--chunk-size", type=int, default=4000)
    # ======== 在这里补充缺失的 snr 和 t 参数 ========
    ap.add_argument("--snr", type=float, default=5.0, help="Signal-to-noise ratio for IQ readout")
    ap.add_argument("--t", type=float, default=1.0, help="t parameter for IQ readout")
    # ===============================================
    ap.add_argument("--out-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--no-leakage", action="store_true",
                    help="Disable leakage post-processing (core PAEMS channels only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned file list without generating.")
    ap.add_argument("--scale", type=int, default=1,
                    help="Scale factor for sample counts in --manifest mode "
                         "(1 = v1 baseline; e.g. 5/10 for larger datasets). "
                         "Only the number of shots changes — distances, rounds, "
                         "LER round-points and the per-distance PAEMS parameter "
                         "seed (hence the virtual chip) are unchanged.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    include_leakage = not args.no_leakage

    params_cache: dict = {}

    if args.manifest:
        rows = build_manifest(scale=args.scale)
        print("=" * 70)
        print("Generating full PAEMS manifest")  # 6 train/val/test + 16 LER = 22 files
        print(f"  out_dir        = {out_dir}")
        print(f"  scale          = {args.scale}x (LER N={DEFAULT_LER_N * args.scale})")
        print(f"  leakage        = {include_leakage}")
        print(f"  soft readout   = snr {args.snr}, t {args.t}")
        print(f"  files          = {len(rows)}")
        print("=" * 70)
        if args.dry_run:
            for split, d, r, n in rows:
                print(f"  {split}_d{d}_r{r}_n{n}.pt")
            return
        t0 = time.time()
        for i, (split, d, r, n) in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}]")
            generate_one(split, d, r, n, out_dir, params_cache=params_cache,
                          chunk_size=args.chunk_size, include_leakage=include_leakage,
                          snr=args.snr, t=args.t)
        total_mb = sum(
            p.stat().st_size for p in out_dir.glob("*.pt")
        ) / (1024 * 1024)
        print("\n" + "=" * 70)
        print(f"Manifest complete in {time.time() - t0:.0f}s. "
              f"{len(rows)} files, {total_mb:.0f} MB total in {out_dir}")
        print("=" * 70)
        return

    if args.split is None:
        ap.error("Provide --manifest or --split (with --distance/--rounds/--num-samples).")
    generate_one(args.split, args.distance, args.rounds, args.num_samples, out_dir,
                 params_cache=params_cache, chunk_size=args.chunk_size,
                 include_leakage=include_leakage, snr=args.snr, t=args.t,
                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
