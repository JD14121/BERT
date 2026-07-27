#!/usr/bin/env python3
"""generate_google_paems_data.py
Google 噪声校准 PAEMS 软信息数据生成器（Option B，XZZX，停用 leakage）。

管线（P1 烟测 + P3 量产共用）：
  1. 官方 generate_surface_code_circuit(code_variant='xzzx', xzzx_template=Google .stim)
     -> 基础 XZZX 电路（Google 真实拓扑，保留 DETECTOR/OBSERVABLE_INCLUDE）
  2. 官方 inject_surface_code_noise(base, dq, xs, zs, cx, 校准配置 JSON) -> 带噪电路
  3. 同种子 measurement sampler + detector sampler -> measurement record / detection_events / label 同一 shot
  4. SoftReadoutSimulator(移植自 alphaqubit/data/soft_readout.py) -> measurement/event/final_soft 软信息
  5. 存 .pt(spec 格式，含软信息) + 导出 .b8(detection_events/obs_flips，Google 格式兼容)

R2: leakage 停用（官方校准 L3 故意排除；软信息由软读出满足）。
R5: 路径经 path_config 集中管理（修 D1 硬编码）。
R6: record 布局断言 rounds*n_stab + n_data == num_measurements。
注意：xzzx 模板路径下，电路轮数 = 模板轮数（Google 仅 {1,10,13,30,...}），故 rounds 必须在 GOOGLE_ROUNDS 中。
"""
from __future__ import annotations
import sys, os, json, math, argparse, subprocess, time
from pathlib import Path
import numpy as np
import stim

# [新增] 引入 IQ 直接读出模块
import paems_iq_readout as pir
# ----------------- 注意：删除以下原有代码 -----------------
# 删除 class SoftReadoutSimulator: ... 完整类
# 删除 def soft_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray: ... 完整方法
# ----------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from path_config import (PAEMS_SC, TIERS_DIR, DATA_DIR, CONFIG_DIR, LOG_DIR,
                         GOOGLE_PATCH, GOOGLE_ROUNDS, google_template_path)

sys.path.insert(0, str(PAEMS_SC))
from inject_basic_noise import inject_surface_code_noise          # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit  # noqa: E402

def _n_data_from_circuit(circuit: stim.Circuit) -> int:
    """末尾 M 指令的 target 数 = data qubit 数。
    XZZX 电路用 M（非 MR）测稳定子：每轮 M(n_stab) + 末轮 M(n_data)；末 M 即 data 测量。
    不依赖模板 dq-d3 r1 的 dq=17（8 stab+9 data 全归 data）错误，末 M=9 正确。"""
    last_m = None
    for inst in circuit.flattened():
        if inst.name in ("M", "MX", "MY", "MZ"):
            last_m = inst
    return len(last_m.targets_copy()) if last_m is not None else 0


# ---------------- 校准配置构建（官方 gen_level_params + gen_pair_overrides） ----------------
def build_config(distance, rounds, xzzx_template, *, level=2, xtalk='X2',
                 defect_seed=7, mult_scale_with_d=True, defect_multiplier=None, out_path=None):
    """构建 PAEMS L<level> + crosstalk 配置 JSON（xzzx 拓扑）。返回配置路径。
    defect_multiplier 若给定，用 --defect-multiplier（单一固定值，覆盖 min/max）；否则用 mult-scale-with-d。"""
    tiers = TIERS_DIR
    tmp = tiers / 'tmp_test'
    tmp.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path else tmp / f'_gpaems_cfg_d{distance}_r{rounds}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p_class = tmp / f'_gpaems_class_d{distance}_r{rounds}.json'

    cargs = [sys.executable, str(tiers / 'gen_level_params.py'),
             '--level', str(level), '--distance', str(distance), '--rounds', str(rounds),
             '--seed', '42', '--defect-seed', str(defect_seed),
             '--out', str(p_class), '--code-variant', 'xzzx',
             '--xzzx-template', str(xzzx_template)]
    if defect_multiplier is not None:
        cargs += ['--defect-multiplier', str(defect_multiplier)]
    elif mult_scale_with_d:
        # 镜像 calibrate：mult_max(d)=max(5, 8+(d-5)*2)；d5->8, d7->12（gen_level_params
        # 无 --mult-scale-with-d，需自行算 --defect-mult-max）
        mult_max = max(5.0, 8.0 + (distance - 5) * 2.0)
        cargs += ['--defect-mult-max', str(mult_max)]
    r = subprocess.run(cargs, capture_output=True, cwd=str(tiers))
    if r.returncode != 0:
        raise RuntimeError(f"gen_level_params 失败:\n{r.stderr.decode(errors='replace')}")

    if xtalk == 'none':
        with open(p_class, encoding='utf-8') as f:
            d = json.load(f)
        d['crosstalk_pairs'] = {}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
    else:
        xt_json = tiers / 'crosstalk_presets' / f'crosstalk_{xtalk}.json'
        r = subprocess.run([sys.executable, str(tiers / 'gen_pair_overrides.py'),
                            '--in', str(p_class), '--out', str(out_path), '--merge',
                            '--crosstalk-config', str(xt_json)],
                           capture_output=True, cwd=str(tiers))
        if r.returncode != 0:
            raise RuntimeError(f"gen_pair_overrides 失败:\n{r.stderr.decode(errors='replace')}")

    # 将离散的读取翻转清零，把读出的噪声代理权全部移交给 IQ 模型
    with open(out_path, 'r', encoding='utf-8') as f:
        final_cfg = json.load(f)
    if 'qubits' in final_cfg:
        for q_id, q_params in final_cfg['qubits'].items():
            if 'measurement_spam_rate' in q_params:
                q_params['measurement_spam_rate'] = 0.0
            if 'data_measurement_error' in q_params:
                q_params['data_measurement_error'] = 0.0
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_cfg, f, indent=2)

    return out_path


# ---------------- b8 导出（手工小端打包，与 stim b8 兼容） ----------------
def write_b8(bits_2d: np.ndarray, path: Path):
    """bits_2d: [shots, nbits] 0/1 -> stim b8（小端，每 shot 8 位对齐）。"""
    bits = np.asarray(bits_2d, dtype=np.uint8)
    shots, nbits = bits.shape
    bytes_per_shot = (nbits + 7) // 8
    out = np.zeros((shots, bytes_per_shot), dtype=np.uint8)
    for b in range(nbits):
        out[:, b >> 3] |= bits[:, b] << (b & 7)
    with open(path, 'wb') as f:
        f.write(out.tobytes())


# ---------------- 核心生成 ----------------
def generate_one(distance, rounds, basis, num_samples, *, config_path, snr=10.0, t=0.01,
                 seed=42, chunk_size=2000, rng_seed=None):
    """生成 num_samples 条 spec-conformant 样本（XZZX, Google 校准噪声, 软读出, 无 leakage）。
    rng_seed: 可选。非 None 时软读出用注入式 np.random.default_rng(rng_seed)（跨噪声编排层
    传 seed+1，与 stim 采样种子解耦）；None 时沿用 np.random.seed(seed) 全局流，行为零变化。"""
    template = google_template_path(distance, basis, rounds)
    base, dq, xs, zs, cx = generate_surface_code_circuit(
        distance, rounds, basis, code_variant='xzzx', xzzx_template=str(template))
    noisy = inject_surface_code_noise(base, dq, xs, zs, cx, str(config_path))

    n_data = _n_data_from_circuit(noisy)          # 末 M target 数（不依赖模板 dq-d3 r1 dq 错误）
    total_records = noisy.num_measurements
    # R6: 从测量 record 反推 n_stab（不依赖 xs/zs 列表-XZZX 模板下列表含未测量边界比特，
    # 实测 d5 xs+zs=27 但每轮只测 24=d^2-1 个 ancilla）。record 布局 = rounds*n_stab(ancilla MR) + n_data(final M)
    assert (total_records - n_data) % rounds == 0, (
        f"R6 record 布局异常: (num_meas {total_records} - n_data {n_data}) % rounds {rounds} != 0")
    n_stab = (total_records - n_data) // rounds
    assert total_records == rounds * n_stab + n_data
    if n_stab != distance * distance - 1:
        print(f"[warn] n_stab={n_stab} != d^2-1={distance**2-1}（XZZX 模板边界差异，以 record 反推为准）")

    # ... 在测量器定义部分 ...
    meas_sampler = noisy.compile_sampler(seed=seed)
    m2d = noisy.compile_m2d_converter()

    # [修改为使用 IQ 模拟器]
    iq_sim = pir.ThreeCenterIQReadoutSimulator(snr=snr, t=t)
    iq_rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng(seed)

    num_det = noisy.num_detectors

    # [新增预分配 IQ (complex64) 数组]
    measurement_iq = np.empty((num_samples, rounds, n_stab), dtype=np.complex64)
    final_iq = np.empty((num_samples, n_data), dtype=np.complex64)
    # 保留原浮点数组预分配
    measurement = np.empty((num_samples, rounds, n_stab), dtype=np.float32)
    event = np.empty((num_samples, rounds, n_stab), dtype=np.float32)
    final_soft = np.empty((num_samples, n_data), dtype=np.float32)
    label = np.empty((num_samples,), dtype=np.float32)
    detection_events = np.empty((num_samples, num_det), dtype=np.float32)

    start = 0
    while start < num_samples:
        chunk = min(chunk_size, num_samples - start)

        # 此时采出的 raw_meas 是纯净态(SPAM已被剥离)
        true_bits = meas_sampler.sample(shots=chunk)

        # 1. 直接对基态做复空间高斯采样 (此处无 leakage, 所以传 None)
        iq_all = iq_sim.sample_iq(true_bits, leaked=None, rng=iq_rng)

        # 2. 从 IQ 返回硬决策，再过探测器转化
        hard_bits = iq_sim.hard_decision(iq_all)
        detobs = m2d.convert(measurements=hard_bits, separate_observables=True)
        dets = np.asarray(detobs[0], dtype=bool)
        obs = np.asarray(detobs[1], dtype=bool).reshape(-1)

        # 3. 数据分隔切片 (分拆出 IQ 层级)
        iq_anc = iq_all[:, :rounds * n_stab].reshape(chunk, rounds, n_stab)
        iq_fin = iq_all[:, rounds * n_stab:]

        # 4. 基于 IQ 得出 P(|1> | z) 软似然
        soft_meas = iq_sim.to_soft_prob(iq_anc)
        soft_event = np.empty_like(soft_meas)
        soft_event[:, 0, :] = soft_meas[:, 0, :]
        for tt in range(1, rounds):
            soft_event[:, tt, :] = pir.soft_xor(soft_meas[:, tt, :], soft_meas[:, tt - 1, :])
        soft_final = iq_sim.to_soft_prob(iq_fin)

        # 5. 组装落盘
        measurement_iq[start:start + chunk] = iq_anc
        final_iq[start:start + chunk] = iq_fin
        measurement[start:start + chunk] = soft_meas
        event[start:start + chunk] = soft_event
        final_soft[start:start + chunk] = soft_final
        label[start:start + chunk] = obs.astype(np.float32)
        detection_events[start:start + chunk] = dets.astype(np.float32)
        start += chunk

    return {
        "measurement_iq": measurement_iq, "final_iq": final_iq,  # [增加 IQ 返回值]
        "measurement": measurement, "event": event, "final_soft": final_soft,
        "label": label, "detection_events": detection_events,
        "distance": int(distance), "rounds": int(rounds),
        "snr": float(snr), "n_stab": int(n_stab), "n_data": int(n_data),
        "num_detectors": int(detection_events.shape[1]),
        "_meta": {"seed": int(seed), "t": float(t), "basis": basis.upper(),
                  "code": "xzzx", "leakage": False, "config": str(config_path),
                  "xzzx_template": str(template), "patch": GOOGLE_PATCH[distance],
                  "readout_model": "three_center_IQ_no_spam"},
    }

    num_det = noisy.num_detectors
    # 预分配输出数组（避免 list+concat 的 2× 内存峰值，d7 1M 必需）
    measurement = np.empty((num_samples, rounds, n_stab), dtype=np.float32)
    event = np.empty((num_samples, rounds, n_stab), dtype=np.float32)
    final_soft = np.empty((num_samples, n_data), dtype=np.float32)
    label = np.empty((num_samples,), dtype=np.float32)
    detection_events = np.empty((num_samples, num_det), dtype=np.float32)
    start = 0
    while start < num_samples:
        chunk = min(chunk_size, num_samples - start)
        raw_meas = meas_sampler.sample(shots=chunk)                 # [chunk, total_records] bool
        detobs = m2d.convert(measurements=raw_meas, separate_observables=True)
        dets = np.asarray(detobs[0], dtype=bool)                    # [chunk, num_det]
        obs = np.asarray(detobs[1], dtype=bool).reshape(-1)         # [chunk]

        anc = raw_meas[:, :rounds * n_stab].reshape(chunk, rounds, n_stab).astype(bool)
        final = raw_meas[:, rounds * n_stab:].astype(bool)          # [chunk, n_data]

        soft_meas = soft.simulate(anc)                              # [chunk, rounds, n_stab]
        soft_event = np.empty_like(soft_meas)
        soft_event[:, 0, :] = soft_meas[:, 0, :]
        for tt in range(1, rounds):
            soft_event[:, tt, :] = soft_xor(soft_meas[:, tt, :], soft_meas[:, tt - 1, :])
        soft_final = soft.simulate(final)                           # [chunk, n_data]

        measurement[start:start + chunk] = soft_meas
        event[start:start + chunk] = soft_event
        final_soft[start:start + chunk] = soft_final
        label[start:start + chunk] = obs.astype(np.float32)
        detection_events[start:start + chunk] = dets.astype(np.float32)
        start += chunk

    return {
        "measurement": measurement, "event": event, "final_soft": final_soft,
        "label": label, "detection_events": detection_events,
        "distance": int(distance), "rounds": int(rounds),
        "snr": float(snr), "n_stab": int(n_stab), "n_data": int(n_data),
        "num_detectors": int(detection_events.shape[1]),
        "_meta": {"seed": int(seed), "t": float(t), "basis": basis.upper(),
                  "code": "xzzx", "leakage": False, "config": str(config_path),
                  "xzzx_template": str(template), "patch": GOOGLE_PATCH[distance]},
    }


def save_pt(data, out_path):
    import torch
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    pt = {}
    # [支持复数 IQ 张量]
    for k in ("measurement_iq", "final_iq"):
        if k in data:
            pt[k] = torch.from_numpy(np.asarray(data[k], dtype=np.complex64))
    # [维持原来的 float32 张量]
    for k in ("measurement", "event", "final_soft", "label", "detection_events"):
        pt[k] = torch.from_numpy(np.asarray(data[k], dtype=np.float32))
    for k in ("distance", "rounds", "snr", "n_stab", "n_data", "num_detectors"):
        pt[k] = data[k]
    pt["p"] = 0.0
    pt["_meta"] = data["_meta"]
    torch.save(pt, str(out_path))


def export_b8(data, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    det = data["detection_events"].astype(np.uint8)
    obs = data["label"].astype(np.uint8).reshape(-1, 1)
    write_b8(det, out_dir / f"{name}_detection_events.b8")
    write_b8(obs, out_dir / f"{name}_obs_flips.b8")
    return out_dir / f"{name}_detection_events.b8", out_dir / f"{name}_obs_flips.b8"


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='P1 烟测：1k 样本 d5 r10 Z')
    ap.add_argument('--distance', type=int, default=5)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--basis', type=str, default='Z')
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--xtalk', type=str, default='X2')
    ap.add_argument('--snr', type=float, default=10.0)
    ap.add_argument('--t', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', type=str, default=str(DATA_DIR / 'smoke'))
    args = ap.parse_args()

    if args.smoke:
        args.distance, args.rounds, args.basis, args.num_samples = 5, 10, 'Z', 1000

    print(f"[smoke/P1] d{args.distance} r{args.rounds} {args.basis} N={args.num_samples} "
          f"level={args.level} xtalk={args.xtalk} snr={args.snr}")

    template = google_template_path(args.distance, args.basis, args.rounds)
    print(f"[template] {template} (exists={template.exists()})")

    t0 = time.time()
    cfg = build_config(args.distance, args.rounds, template, level=args.level,
                       xtalk=args.xtalk, out_path=CONFIG_DIR / f"smoke_d{args.distance}_r{args.rounds}.json")
    print(f"[config] built in {time.time()-t0:.1f}s -> {cfg}")

    t0 = time.time()
    data = generate_one(args.distance, args.rounds, args.basis, args.num_samples,
                        config_path=cfg, snr=args.snr, t=args.t, seed=args.seed)
    print(f"[gen] {args.num_samples} samples in {time.time()-t0:.1f}s")

    # ---- 烟测诊断 ----
    print("\n=== P1 smoke diagnostics ===")
    print(f"  shapes: meas{data['measurement'].shape} event{data['event'].shape} "
          f"final_soft{data['final_soft'].shape} det{data['detection_events'].shape} label{data['label'].shape}")
    print(f"  n_stab={data['n_stab']} n_data={data['n_data']} num_det={data['num_detectors']} "
          f"(expect n_stab=d^2-1={args.distance**2-1}, n_data=d^2={args.distance**2})")
    print(f"  label rate={float(data['label'].mean()):.4f} (应非0/1、非精确0.5)")
    print(f"  det density={float(data['detection_events'].mean()):.4f}")
    # 软读出分布（S3）
    soft = data['measurement']
    print(f"  soft meas: min={soft.min():.3f} max={soft.max():.3f} mean={soft.mean():.3f} "
          f"frac<0.1={float((soft<0.1).mean()):.3f} frac>0.9={float((soft>0.9).mean()):.3f}")
    print(f"  final_soft: min={data['final_soft'].min():.3f} max={data['final_soft'].max():.3f}")
    # 软读出双峰（按真值分组近似：用硬测量 anc 做真值）
    print(f"  event[0]==meas[0] check: {np.allclose(data['event'][:,0,:], data['measurement'][:,0,:])}")

    # ---- 存盘 ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = f"smoke_d{args.distance}_r{args.rounds}_{args.basis}_n{args.num_samples}"
    pt_path = out_dir / f"{name}.pt"
    save_pt(data, pt_path)
    print(f"\n[save .pt] {pt_path} ({pt_path.stat().st_size/1024:.0f} KB)")

    det_b8, obs_b8 = export_b8(data, out_dir, name)
    print(f"[export .b8] {det_b8.name} ({det_b8.stat().st_size} B), {obs_b8.name} ({obs_b8.stat().st_size} B)")

    # ---- b8 round-trip 验证（R8 结构）----
    det_rt = stim.read_shot_data_file(path=str(det_b8), format='b8',
                                      num_detectors=data['num_detectors'])
    obs_rt = stim.read_shot_data_file(path=str(obs_b8), format='b8', num_observables=1)
    match_det = np.array_equal(det_rt, data['detection_events'].astype(bool))
    match_obs = np.array_equal(obs_rt.reshape(-1), data['label'].astype(bool))
    print(f"[b8 round-trip] det match={match_det} obs match={match_obs}")
    print("\n=== P1 smoke DONE ===")


if __name__ == '__main__':
    main()
