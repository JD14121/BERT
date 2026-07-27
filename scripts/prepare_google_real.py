#!/usr/bin/env python3
"""prepare_google_real.py (P1): 真机 Google b8 -> 硬读出 .pt（与合成数据同 schema）。
measurement/event/final_soft 为硬 0/1（用户决议：微调无软读出，用硬读出）。
多 patch 汇总（d3=9 patch），按 80/10/10 切 train/val/test。rounds=r10 主。
每 patch 用各自 circuit_ideal.stim 提取布局（ancilla/data/detector）；坐标归一化后跨 patch 一致。
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import stim

CODE_DIR = Path(__file__).resolve().parent.parent / "code"   # google_paems_data/code
PROJECT_ROOT = CODE_DIR.parent.parent                        # alphaquibit-main/alphaquibit-main
sys.path.insert(0, str(PROJECT_ROOT))                        # for alphaqubit
sys.path.insert(0, str(CODE_DIR))
from path_config import GOOGLE_SC, GOOGLE_PATCH
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xzzx_coord import XZZXCoordinateSystem

# d3 全部 9 patch；d5 4 patch；d7 1 patch
ALL_PATCHES = {
    3: ["q2_7","q4_5","q4_9","q6_3","q6_7","q6_11","q8_5","q8_9","q10_7"],
    5: ["q4_7","q6_5","q6_9","q8_7"],
    7: ["q6_7"],
}


def load_one_patch(distance, patch, basis, rounds):
    base = GOOGLE_SC / f"d{distance}_at_{patch}" / basis / f"r{rounds:02d}"
    cir = stim.Circuit.from_file(str(base / "circuit_ideal.stim"))
    n_stab = distance**2 - 1
    n_data = distance**2
    num_meas = cir.num_measurements
    num_det = cir.num_detectors
    assert num_meas == rounds * n_stab + n_data, f"{patch} meas layout: {num_meas} != {rounds*n_stab+n_data}"
    assert num_det == rounds * n_stab, f"{patch} det layout: {num_det} != {rounds*n_stab}"
    meas = stim.read_shot_data_file(path=str(base/"measurements.b8"), format='b8',
                                    num_measurements=num_meas, bit_packed=False).astype(np.uint8)
    det = stim.read_shot_data_file(path=str(base/"detection_events.b8"), format='b8',
                                   num_detectors=num_det, bit_packed=False).astype(np.uint8)
    obs = stim.read_shot_data_file(path=str(base/"obs_flips_actual.b8"), format='b8',
                                   num_observables=1, bit_packed=False).astype(np.uint8).reshape(-1)
    # 切分 measurement record: [rounds*n_stab ancilla] + [n_data final]
    anc = meas[:, :rounds*n_stab].reshape(-1, rounds, n_stab)         # [N,r,n_stab] 硬
    final = meas[:, rounds*n_stab:]                                    # [N,n_data] 硬
    event = det.reshape(-1, rounds, n_stab)                            # [N,r,n_stab] 硬 XOR = detection events
    return anc, event, final, obs, det                                 # det [N,num_det] for MWPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=3)
    ap.add_argument('--basis', default='Z')
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    d = args.distance
    out_dir = Path(args.out_dir) if args.out_dir else (CODE_DIR.parent / "data" / f"real_d{d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    patches = ALL_PATCHES[d]
    print(f"[prepare_real] d{d} {args.basis} r{args.rounds} patches={patches}")

    anc_l, ev_l, fin_l, obs_l, det_l = [], [], [], [], []
    for patch in patches:
        anc, ev, fin, obs, det = load_one_patch(d, patch, args.basis, args.rounds)
        anc_l.append(anc); ev_l.append(ev); fin_l.append(fin); obs_l.append(obs); det_l.append(det)
        print(f"  {patch}: {anc.shape[0]} shots (label_rate={obs.mean():.3f} det_dens={det.mean():.4f})")
    measurement = np.concatenate(anc_l, 0).astype(np.float32)
    event = np.concatenate(ev_l, 0).astype(np.float32)
    final_soft = np.concatenate(fin_l, 0).astype(np.float32)
    label = np.concatenate(obs_l, 0).astype(np.float32)
    detection_events = np.concatenate(det_l, 0).astype(np.float32)
    N = label.shape[0]
    print(f"  pooled: N={N} meas{measurement.shape} event{event.shape} final_soft{final_soft.shape} det{detection_events.shape} label_rate={label.mean():.4f}")

    # 80/10/10 划分
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_tr = int(0.8*N); n_va = int(0.1*N)
    splits = {'train': idx[:n_tr], 'val': idx[n_tr:n_tr+n_va], 'test': idx[n_tr+n_va:]}
    n_stab = d*d-1; n_data = d*d; num_det = args.rounds*n_stab
    for split, ids in splits.items():
        pt = {
            'measurement': torch.from_numpy(measurement[ids]),
            'event': torch.from_numpy(event[ids]),
            'final_soft': torch.from_numpy(final_soft[ids]),
            'label': torch.from_numpy(label[ids]),
            'detection_events': torch.from_numpy(detection_events[ids]),
            'distance': int(d), 'rounds': int(args.rounds), 'snr': float('inf'),  # 硬读出
            'n_stab': int(n_stab), 'n_data': int(n_data), 'num_detectors': int(num_det),
            'p': 0.0,   # 补 p 字段（PTBatchDataset 要求；真机硬读出无标量 p，记 0.0 对齐 schema）
            '_meta': {'source': 'google_real', 'basis': args.basis, 'readout': 'hard',
                      'patches': patches, 'seed': int(args.seed)},
        }
        path = out_dir / f"{split}_d{d}_r{args.rounds}_n{len(ids)}_{args.basis}.pt"
        torch.save(pt, str(path))
        print(f"  [{split}] {path.name} N={len(ids)} label_rate={label[ids].mean():.4f} ({path.stat().st_size/1e6:.1f}MB)")
    print("[prepare_real] DONE")


if __name__ == '__main__':
    main()
