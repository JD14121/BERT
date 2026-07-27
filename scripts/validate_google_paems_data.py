#!/usr/bin/env python3
"""validate_google_paems_data.py (P4 QC): 验证生成的 Google-PAEMS 数据。
每文件: 形状/范围/n_stab=d²-1/n_data=d²/num_det=rounds×n_stab + 软读出双峰。
每码距: MWPM sanity-test 集解码（accuracy 应显著>0.5，非随机；与 Google 同码距量级一致）。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import stim
import pymatching

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from path_config import PAEMS_SC, DATA_DIR, CONFIG_DIR, GOOGLE_SC, GOOGLE_PATCH
sys.path.insert(0, str(PAEMS_SC))
from inject_basic_noise import inject_surface_code_noise                    # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit    # noqa: E402

EXPECTED = {3: (8, 9), 5: (24, 25), 7: (48, 49)}   # (n_stab, n_data)


def validate_file(pt_path, d):
    pt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
    n_stab_exp, n_data_exp = EXPECTED[d]
    m, e, fs, de, lb = pt['measurement'], pt['event'], pt['final_soft'], pt['detection_events'], pt['label']
    r = int(pt['rounds']); N = m.shape[0]
    issues = []
    if m.shape[1:] != (r, n_stab_exp): issues.append(f"meas{tuple(m.shape)} exp [N,{r},{n_stab_exp}]")
    if e.shape[1:] != (r, n_stab_exp): issues.append(f"event{tuple(e.shape)}")
    if fs.shape[1:] != (n_data_exp,): issues.append(f"final_soft{tuple(fs.shape)} exp [N,{n_data_exp}]")
    if de.shape[1] != r * n_stab_exp: issues.append(f"det{tuple(de.shape)} exp [N,{r*n_stab_exp}]")
    if lb.shape != (N,): issues.append(f"label{tuple(lb.shape)}")
    if float(m.min()) < 0 or float(m.max()) > 1: issues.append(f"meas range [{float(m.min()):.3f},{float(m.max()):.3f}]")
    if float(lb.min()) < 0 or float(lb.max()) > 1: issues.append(f"label range [{float(lb.min())},{float(lb.max())}]")
    frac_lo = float((m < 0.1).float().mean()); frac_hi = float((m > 0.9).float().mean())
    if frac_lo + frac_hi < 0.7: issues.append(f"soft非双峰 <0.1={frac_lo:.2f} >0.9={frac_hi:.2f}")
    return issues, dict(N=N, label_rate=float(lb.mean()), det_dens=float(de.mean()),
                        frac_lo=frac_lo, frac_hi=frac_hi)


def mwpm_sanity(d, basis='Z', rounds=10, max_shots=10000):
    cfg = CONFIG_DIR / f"calibrated_d{d}.json"
    template = GOOGLE_SC / f"d{d}_at_{GOOGLE_PATCH[d]}" / basis / f"r{rounds:02d}" / "circuit_ideal.stim"
    base, dq, xs, zs, cx = generate_surface_code_circuit(d, rounds, basis,
                                                          code_variant='xzzx', xzzx_template=str(template))
    noisy = inject_surface_code_noise(base, dq, xs, zs, cx, str(cfg))
    dem = noisy.detector_error_model()
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    test_pt = DATA_DIR / f"d{d}" / f"test_d{d}_r{rounds}_n100000_{basis}.pt"
    pt = torch.load(str(test_pt), map_location='cpu', weights_only=False)
    n = min(max_shots, pt['label'].shape[0])
    det = pt['detection_events'][:n].numpy().astype(np.uint8)
    label = pt['label'][:n].numpy().astype(int)
    preds = mwpm.decode_batch(det).flatten()
    acc = float((preds == label).mean())
    return acc, n


def main():
    print("=" * 60); print("P4 QC Validation"); print("=" * 60)
    all_pass = True
    for d in [3, 5, 7]:
        print(f"\n--- d{d} ---")
        for pt in sorted((DATA_DIR / f"d{d}").glob("*.pt")):
            issues, s = validate_file(pt, d)
            ok = not issues
            if not ok: all_pass = False
            print(f"  [{'PASS' if ok else 'FAIL'}] {pt.name}: N={s['N']} label_rate={s['label_rate']:.3f} "
                  f"det_dens={s['det_dens']:.4f} soft(<0.1={s['frac_lo']:.2f},>0.9={s['frac_hi']:.2f})")
            if issues: print(f"         {issues}")
        acc, n = mwpm_sanity(d)
        mwpm_ok = 0.55 < acc < 0.999
        if not mwpm_ok: all_pass = False
        print(f"  MWPM test(n={n}): acc={acc:.4f} {'PASS' if mwpm_ok else 'FAIL'} (应>0.5非随机,与Google同码距量级)")
    print("\n" + "=" * 60)
    print(f"P4 QC OVERALL: {'ALL PASS ✅' if all_pass else 'HAS FAILURES ❌'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
