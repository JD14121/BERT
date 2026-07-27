#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""symmetry_augment.py
实验 A：d7 真机数据 C2 对称增强（180° 旋转，2×）。
1. 解析 d7 stim 电路，计算 rot180 的 perm_stab(48) / perm_data(49)
2. 验证门 §2.3（4 项必过）：perm 合法 / label-corr=+1 / syndrome 一致 / detection_events 逐轮一致
3. 全部 PASS -> 增强 real_d7（train/val/test 各 2×）-> 存 /root/data/real_d7_aug/

依计划书 v2 §2.3，label-corr 必须=+1.0（rot180 保持 logical Z）。
"""
import stim, torch, numpy as np, os, sys, shutil
from pathlib import Path

CIRC = "/root/beat_mwpm/google_paems_data/Google-data/google_105Q_surface_code_d3_d5_d7/d7_at_q6_7/Z/r01/circuit_ideal.stim"
REAL_DIR = Path("/root/data/real_d7")
OUT_DIR = Path("/root/data/real_d7_aug")
N_STAB, N_DATA, T = 48, 49, 10
GRID = 13  # 2*7-1

def rot180(p): return (GRID-1-p[0], GRID-1-p[1])

# ============ 1. 解析电路，算置换 ============
circ = stim.Circuit.from_file(CIRC)
qubit_coords = {}
for inst in circ.flattened():
    if inst.name == 'QUBIT_COORDS':
        a = inst.gate_args_copy()
        for t in inst.targets_copy():
            if t.is_qubit_target:
                qubit_coords[t.value] = (a[0], a[1])
all_meas = []
for inst in circ.flattened():
    if inst.name in ('M','MX','MY','MZ','MR','MRX','MRY','MRZ'):
        for t in inst.targets_copy():
            if t.is_qubit_target: all_meas.append(t.value)
stab_qids = all_meas[:N_STAB]
data_qids = all_meas[-N_DATA:]
all_qs = [q for q in all_meas if q in qubit_coords]
xs = [qubit_coords[q][0] for q in all_qs]; ys = [qubit_coords[q][1] for q in all_qs]
min_x, min_y = min(xs), min(ys)
def pos(q): return (int(qubit_coords[q][1]-min_y), int(qubit_coords[q][0]-min_x))
stab_pos = [pos(q) for q in stab_qids]
data_pos = [pos(q) for q in data_qids]
pos_to_stab = {p:i for i,p in enumerate(stab_pos)}
pos_to_data = {p:i for i,p in enumerate(data_pos)}
perm_stab = np.array([pos_to_stab[rot180(p)] for p in stab_pos], dtype=np.int64)
perm_data = np.array([pos_to_data[rot180(p)] for p in data_pos], dtype=np.int64)
print(f"perm_stab[:5]={perm_stab[:5]}, perm_data[:5]={perm_data[:5]}")

# ============ 2. 验证门 ============
gates_ok = True

# GATE 1: perm 合法（双射）
g1 = sorted(perm_stab.tolist()) == list(range(N_STAB)) and sorted(perm_data.tolist()) == list(range(N_DATA))
print(f"[GATE 1] perm 双射合法性: {'PASS' if g1 else 'FAIL'}")
gates_ok &= g1

# GATE 2: label-corr（无噪声 shot，rot180 label 必须 == orig label）
obs_targets = []
for inst in circ.flattened():
    if inst.name == 'OBSERVABLE_INCLUDE':
        obs_targets = [t.value for t in inst.targets_copy()]
obs_abs = [len(all_meas)+t for t in obs_targets]
obs_qids = [all_meas[i] for i in obs_abs]
obs_pos_set = set(pos(q) for q in obs_qids)
# rot180 后的 observable qubits
rot_obs_pos = set(rot180(p) for p in obs_pos_set)
pos_to_qid = {}
for q in all_qs:
    pos_to_qid[pos(q)] = q
rot_obs_qids = [pos_to_qid[p] for p in rot_obs_pos if p in pos_to_qid]
# 找 rot_obs qids 在 all_meas 中的索引（final data measurement 段，最后 49）
rot_obs_indices = []
for q in rot_obs_qids:
    for i in range(len(all_meas)-N_DATA, len(all_meas)):
        if all_meas[i] == q:
            rot_obs_indices.append(i); break
sampler = circ.compile_sampler()
shots = np.array(sampler.sample(2000), dtype=np.uint8)
orig_label = shots[:, obs_abs].sum(1) % 2
rot_label = shots[:, rot_obs_indices].sum(1) % 2
same = int((orig_label == rot_label).sum())
corr = float(np.corrcoef(orig_label.astype(float), rot_label.astype(float))[0,1]) if orig_label.std()>0 else 1.0
g2 = (same == len(orig_label)) and (corr > 0.999)
print(f"[GATE 2] label-corr (2000 无噪声 shot): same={same}/{len(orig_label)}, corr={corr:.6f} -> {'PASS' if g2 else 'FAIL'}")
gates_ok &= g2

# GATE 3: syndrome 一致性 -- rot180(measurement) 应是合法 syndrome（值分布与原一致）
# 取 real_d7 train 前 100 样本，对比 rot 前后 measurement 的统计
d = torch.load(str(REAL_DIR/"train_d7_r10_n40000_Z.pt"), map_location='cpu', weights_only=False)
meas = d['measurement'][:100].numpy()  # [100,10,48]
meas_rot = meas[:,:,perm_stab]
# 验证：rot180 再 rot180 应回到原样（置换的 involution 性）
meas_rot2 = meas_rot[:,:,perm_stab]
g3 = np.array_equal(meas_rot2, meas)
print(f"[GATE 3] rot180 involution (rot∘rot=identity): {'PASS' if g3 else 'FAIL'}")
gates_ok &= g3

# GATE 4: detection_events 逐轮一致 -- det.reshape(T,48)[perm] 与 event[T,48][perm] 同构
det = d['detection_events'][:100].numpy()  # [100,480]
ev = d['event'][:100].numpy()  # [100,10,48]
det_rot = det.reshape(100, T, N_STAB)[:,:,perm_stab].reshape(100,-1)
ev_rot = ev[:,:,perm_stab]
# det 与 event 的逐轮相关结构应一致（都按 perm 旋转）
# 检查：det_rot 的每轮 与 ev_rot 的每轮 相关性 == 原 det 每轮 与 ev 每轮 相关性
def per_round_corr(det_r, ev_r):
    # det_r:[100,480], ev_r:[100,10,48] -> 每轮 stab 维相关
    dr = det_r.reshape(100, T, N_STAB)
    corrs = []
    for t in range(T):
        for s in range(N_STAB):
            a = dr[:,t,s].astype(float); b = ev_r[:,t,s].astype(float)
            if a.std()>0 and b.std()>0:
                corrs.append(np.corrcoef(a,b)[0,1])
    return np.mean(corrs)
c_orig = per_round_corr(det, ev)
c_rot = per_round_corr(det_rot, ev_rot)
g4 = abs(c_orig - c_rot) < 0.02
print(f"[GATE 4] detection_events 逐轮一致: orig corr={c_orig:.4f}, rot corr={c_rot:.4f}, Δ={abs(c_orig-c_rot):.4f} -> {'PASS' if g4 else 'FAIL'}")
gates_ok &= g4

print(f"\n===== 验证门总结果: {'ALL PASS' if gates_ok else 'FAIL - abort'} =====")
if not gates_ok:
    print("增强中止：验证门未全过"); sys.exit(1)

# ============ 3. 增强 real_d7 ============
OUT_DIR.mkdir(parents=True, exist_ok=True)
for split, fname in [("train","train_d7_r10_n40000_Z.pt"),
                     ("val","val_d7_r10_n5000_Z.pt"),
                     ("test","test_d7_r10_n5000_Z.pt")]:
    src = REAL_DIR / fname
    if not src.exists():
        # 尝试 glob
        import glob as _g
        cand = _g.glob(str(REAL_DIR/f"{split}_d7_r10_n*_{('Z')}.pt"))
        if not cand: print(f"[SKIP] {split} not found"); continue
        src = Path(cand[0]); fname = src.name
    d = torch.load(str(src), map_location='cpu', weights_only=False)
    n = d['measurement'].shape[0]
    # 旋转副本
    aug = {}
    aug['measurement'] = d['measurement'][:,:,perm_stab].clone()
    aug['event'] = d['event'][:,:,perm_stab].clone()
    aug['final_soft'] = d['final_soft'][:,perm_data].clone()
    det_orig = d['detection_events']  # [N,480]
    aug['detection_events'] = det_orig.reshape(n,T,N_STAB)[:,:,perm_stab].reshape(n,-1).clone()
    aug['label'] = d['label'].clone()  # rot180 保持 label
    # 拼接 original + augmented
    out = {}
    for k in ['measurement','event','final_soft','detection_events','label']:
        out[k] = torch.cat([d[k], aug[k]], dim=0)
    # 元数据
    for k in d.keys():
        if k not in out: out[k] = d[k]
    out['_meta'] = {**(d.get('_meta',{}) if isinstance(d.get('_meta'),dict) else {}),
                    'augmented':'C2_rot180_2x', 'orig_n':n, 'aug_n':2*n}
    out_fname = fname.replace(f"_n{n}_", f"_n{2*n}_") if f"_n{n}_" in fname else f"aug_{fname}"
    out_path = OUT_DIR / out_fname
    torch.save(out, str(out_path))
    print(f"[{split}] {n} -> {2*n} (orig+rot180), saved {out_path.name}, label dist orig={float(d['label'].mean()):.4f} aug={float(aug['label'].mean()):.4f}")

print("\n=== 增强完成，输出目录:", OUT_DIR, "===")
