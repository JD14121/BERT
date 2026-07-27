#!/usr/bin/env python3
"""calibrate_to_google.py (P2): 拟合 PAEMS L1+L2+L4 配置到 Google 真实芯片 syndrome。

网格搜索 level × xtalk，每个配置采样 PAEMS detection events，与 Google 同实例
detection_events.b8 对比（defect rate / per-detector p10,p90,std / Spitz pij hop>=2），
选 gap 最小配置，持久化 configs/calibrated_d{d}.json + 报告 logs/calibration_report.md。

度量函数复用官方 calibrate_l1l2l4_vs_google.py（summary_block / detector_to_qubit_order /
cx_graph / compute_pij_spitz）；配置构建复用本工程已测的 build_config。
"""
import sys, json, time, shutil, argparse
from pathlib import Path
import numpy as np
import stim

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from path_config import PAEMS_SC, TIERS_DIR, GOOGLE_SC, CONFIG_DIR, LOG_DIR, GOOGLE_PATCH
sys.path.insert(0, str(PAEMS_SC))
sys.path.insert(0, str(TIERS_DIR))          # 导入官方 calibrate 模块
import calibrate_l1l2l4_vs_google as cal    # noqa: E402（度量函数；其硬编码路径不影响度量）
from inject_basic_noise import inject_surface_code_noise                    # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit    # noqa: E402
from generate_google_paems_data import build_config                         # noqa: E402（已测）


def load_google(distance, patch, basis, rounds, max_shots=5000):
    base = GOOGLE_SC / f"d{distance}_at_{patch}" / basis.upper() / f"r{rounds:02d}"
    cir = stim.Circuit.from_file(str(base / "circuit_noisy_si1000.stim"))
    dets = stim.read_shot_data_file(path=str(base / "detection_events.b8"), format='b8',
                                    num_detectors=cir.num_detectors, bit_packed=False)
    if dets.shape[0] > max_shots:
        dets = dets[:max_shots]
    return dets.astype(np.uint8), cir


def sample_paems_dets(distance, rounds, basis, config_path, shots, seed=42):
    template = GOOGLE_SC / f"d{distance}_at_{GOOGLE_PATCH[distance]}" / basis.upper() / f"r{rounds:02d}" / "circuit_ideal.stim"
    base, dq, xs, zs, cx = generate_surface_code_circuit(distance, rounds, basis,
                                                          code_variant='xzzx', xzzx_template=str(template))
    noisy = inject_surface_code_noise(base, dq, xs, zs, cx, str(config_path))
    dets = noisy.compile_detector_sampler(seed=seed).sample(shots=shots).astype(np.uint8)
    return dets, noisy


def gap_score(res_p, res_r):
    """综合 gap（越小越好）：defect rate + per-det p10/p90/std + Spitz pij mean（×100 缩放）。"""
    p10p, p90p = np.percentile(res_p['per_det'], [10, 90])
    p10r, p90r = np.percentile(res_r['per_det'], [10, 90])
    return (abs(res_p['rate'] - res_r['rate'])
            + abs(p10p - p10r) + abs(p90p - p90r)
            + abs(res_p['per_det'].std() - res_r['per_det'].std())
            + abs(res_p['pij_cross'].mean() - res_r['pij_cross'].mean()) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--basis', default='Z')
    ap.add_argument('--shots', type=int, default=3000)
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--xtalks', nargs='+', default=['none', 'X1', 'X2', 'X3', 'X4'])
    ap.add_argument('--defect-mults', type=float, nargs='+', default=None,
                    help='defect-multiplier 扫描（refinement）；默认 None=用 mult-scale-with-d')
    args = ap.parse_args()
    patch = GOOGLE_PATCH[args.distance]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[P2] d{args.distance} {patch} {args.basis} r{args.rounds} shots={args.shots} "
          f"levels={args.levels} xtalks={args.xtalks}")

    # Google 参考
    gd, gcir = load_google(args.distance, patch, args.basis, args.rounds, args.shots)
    sort_r, qpd_r = cal.detector_to_qubit_order(gcir)
    import networkx as nx
    sp_r = dict(nx.all_pairs_shortest_path_length(cal.cx_graph(gcir)))
    gd_s = gd[:, sort_r]
    res_r = cal.summary_block("Google-ref", gd_s, gcir, qpd_r, sp_r)
    print(f"  Google: rate={res_r['rate']:.3f}% per_det_p10={np.percentile(res_r['per_det'],10):.3f} "
          f"p90={np.percentile(res_r['per_det'],90):.3f} pij_mean={res_r['pij_cross'].mean():.5f}")

    template = GOOGLE_SC / f"d{args.distance}_at_{patch}" / args.basis.upper() / f"r{args.rounds:02d}" / "circuit_ideal.stim"
    dms = args.defect_mults if args.defect_mults else [None]
    rows = []
    for lv in args.levels:
        for xt in args.xtalks:
            for dm in dms:
                tag = f"L{lv}_{xt}" + (f"_dm{dm:g}" if dm is not None else "")
                cfg = CONFIG_DIR / f"cal_d{args.distance}_r{args.rounds}_{tag}.json"
                try:
                    build_config(args.distance, args.rounds, template, level=lv, xtalk=xt,
                                 out_path=cfg, mult_scale_with_d=(dm is None),
                                 defect_multiplier=dm)
                    pd, pcir = sample_paems_dets(args.distance, args.rounds, args.basis, cfg, args.shots)
                    sort_p, qpd_p = cal.detector_to_qubit_order(pcir)
                    n = min(pd.shape[1], gd.shape[1])
                    pd_s = pd[:, sort_p][:, :n]; qpd_p = qpd_p[:n]
                    gd_s2 = gd_s[:, :n]
                    sp_p = dict(nx.all_pairs_shortest_path_length(cal.cx_graph(pcir)))
                    res_p = cal.summary_block(tag, pd_s, pcir, qpd_p, sp_p)
                    g = gap_score(res_p, res_r)
                    rows.append((g, lv, xt, str(cfg), res_p))
                    print(f"  {tag}: gap={g:.4f}  rate={res_p['rate']:.3f}% "
                          f"p10={np.percentile(res_p['per_det'],10):.3f} p90={np.percentile(res_p['per_det'],90):.3f} "
                          f"pij={res_p['pij_cross'].mean():.5f}")
                except Exception as e:
                    print(f"  {tag}: FAIL {type(e).__name__}: {e}")

    if not rows:
        print("[P2] 无成功配置，终止"); return
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    print(f"\n=== BEST: L{best[1]}_{best[2]} gap={best[0]:.4f} cfg={best[3]} ===")

    # 持久化最优配置
    calibrated = CONFIG_DIR / f"calibrated_d{args.distance}.json"
    shutil.copyfile(best[3], calibrated)
    # 报告
    rep = LOG_DIR / f"calibration_report_d{args.distance}.md"
    with open(rep, 'w', encoding='utf-8') as f:
        f.write(f"# P2 校准报告 d{args.distance} {patch} {args.basis} r{args.rounds}\n\n")
        f.write(f"## Google 参考\n- defect rate: {res_r['rate']:.3f}%\n"
                f"- per-det p10/p90: {np.percentile(res_r['per_det'],10):.3f}/{np.percentile(res_r['per_det'],90):.3f}\n"
                f"- Spitz pij hop>=2 mean: {res_r['pij_cross'].mean():.5f}\n\n")
        f.write("## 配置搜索（按 gap 升序）\n\n| rank | level | xtalk | gap | rate% | p10 | p90 | pij_mean |\n|---|---|---|---|---|---|---|---|\n")
        for i, (g, lv, xt, cfg, rp) in enumerate(rows):
            f.write(f"| {i+1} | L{lv} | {xt} | {g:.4f} | {rp['rate']:.3f} | "
                    f"{np.percentile(rp['per_det'],10):.3f} | {np.percentile(rp['per_det'],90):.3f} | "
                    f"{rp['pij_cross'].mean():.5f} |\n")
        f.write(f"\n## 选定\n- **L{best[1]}_{best[2]}** gap={best[0]:.4f}\n- 配置: {calibrated}\n")
    print(f"[report] {rep}")
    print(f"[calibrated config] {calibrated}")
    print("[P2] DONE")


if __name__ == '__main__':
    main()
