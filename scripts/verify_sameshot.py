"""审查组正确性验证：同种子两 sampler 是否产生同一 shot？
方法：用 m2d 转换器从 measurement record 派生 detection_events/observable，
与 detector_sampler 直接采样结果逐位比对。完全一致 -> 两 sampler 同 shot（spec §3.2 成立）。
"""
import sys
import numpy as np
import stim
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from path_config import PAEMS_SC, CONFIG_DIR, google_template_path
sys.path.insert(0, str(PAEMS_SC))
from inject_basic_noise import inject_surface_code_noise
from surface_code_generate_circuits import generate_surface_code_circuit

d, r, basis, S, seed = 5, 10, 'Z', 50, 42
template = google_template_path(d, basis, r)
base, dq, xs, zs, cx = generate_surface_code_circuit(d, r, basis, code_variant='xzzx', xzzx_template=str(template))
cfg = CONFIG_DIR / 'smoke_d5_r10.json'
noisy = inject_surface_code_noise(base, dq, xs, zs, cx, str(cfg))

# 两 sampler 同种子
meas = noisy.compile_sampler(seed=seed).sample(shots=S)                 # [S, num_meas] bool
dets_ds, obs_ds = noisy.compile_detector_sampler(seed=seed).sample(shots=S, separate_observables=True)
dets_ds = np.asarray(dets_ds, dtype=bool)
obs_ds = np.asarray(obs_ds, dtype=bool).reshape(-1)

# m2d 从 measurement 派生（provably 同 shot）
m2d = noisy.compile_m2d_converter()
out = m2d.convert(measurements=meas, separate_observables=True)
print("m2d.convert return type:", type(out).__name__,
      "| len:", len(out) if isinstance(out, (tuple, list)) else "n/a")
dets_m2d = np.asarray(out[0], dtype=bool)
obs_m2d = np.asarray(out[1], dtype=bool).reshape(-1)

print(f"\nshapes: meas{meas.shape} dets_ds{dets_ds.shape} dets_m2d{dets_m2d.shape} "
      f"obs_ds{obs_ds.shape} obs_m2d{obs_m2d.shape}")
print(f"detection_events 逐位一致: {np.array_equal(dets_ds, dets_m2d)}")
print(f"observable 逐位一致      : {np.array_equal(obs_ds, obs_m2d)}")
if not np.array_equal(dets_ds, dets_m2d):
    diff = (dets_ds != dets_m2d).mean()
    print(f"  >>> 不一致比例: {diff:.4f}（两 sampler 非 same-shot，须改用 m2d 派生！）")
else:
    print("  >>> 两 sampler 同种子 = 同一 shot，spec §3.2 成立。建议量产仍用 m2d 以 provably 一致。")
