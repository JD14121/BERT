"""paems_iq_readout.py — Direct-IQ readout extension for the PAEMS noise model.

Instead of letting stim decide the binary readout outcome (via X_ERROR SPAM)
and then *post-hoc* adding soft noise, this module samples the raw IQ signal
directly from the true post-circuit quantum state. Readout uncertainty is thus
modeled exactly once, so the soft posterior and the hard decision are two
readings of the SAME IQ sample and stay mutually consistent.

Three-center model:
    |0> -> center c0     (ground)
    |1> -> center c1     (excited)
    |2> -> center c2     (leaked; a physically distinct IQ cluster)

Leakage (from paems_noise_model.simulate_leakage) decides which measurement
positions are in |2> at readout time; those are sampled from c2 rather than
being flipped 50/50. Detectors/observables are rebuilt from the IQ hard
decision through stim's measurements->detectors converter, guaranteeing the
detection_events match the same IQ shot that produced the soft information.

Run directly:
    python paems_iq_readout.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from paems_noise_model import (
    _base_surface_code_circuit,
    _extract_layout,
    _copy_targets,
    build_crosstalk_lookup,
    _add_spectator_crosstalk,
    single_qubit_noise,
    two_qubit_noise,
    readout_noise,
    simulate_leakage,
    soft_xor,
    generate_paems_params,
)

# ---------------------------------------------------------------------------
# 1. Three-center IQ readout simulator
# ---------------------------------------------------------------------------

class ThreeCenterIQReadoutSimulator:
    """Gaussian IQ readout over three state centers (|0>, |1>, |2>=leaked).

    Each center k has a complex mean c_k and (isotropic) std sigma_k. A single
    IQ sample z ~ CN(c_state, sigma_state) is drawn per measurement. From z we
    derive both the soft posteriors and the hard binary decision.

    Parameters
    ----------
    c0, c1, c2 : complex
        IQ-plane centers for |0>, |1>, |2>. Defaults place them on a triangle:
        c0 = 0, c1 = 1, c2 = 0.5 + 0.75j (a distinct off-axis leaked cluster).
    snr : float
        Signal-to-noise ratio; sets the shared base sigma = 1/sqrt(2*snr),
        scaled to the |0>-|1> separation so SNR keeps its usual meaning.
    t : float
        |1> signal shrink factor (mu_1 = |c1| * (1 - t/2)), matching the
        original SoftReadoutSimulator convention.
    priors : (p0, p1, p2)
        Prior state occupation used in the posterior. Defaults to a mild
        leakage prior; overridden per-shot when a leaked mask is supplied.
    """

    def __init__(self,
                 c0: complex = 0.0 + 0.0j,
                 c1: complex = 1.0 + 0.0j,
                 c2: complex = 0.5 + 0.75j,
                 snr: float = 10.0,
                 t: float = 0.01,
                 sigma_scale: Tuple[float, float, float] = (1.0, 1.0, 1.2),
                 priors: Tuple[float, float, float] = (0.5, 0.49, 0.01)):
        assert snr > 0 and 0.0 < t < 1.0
        self.snr = float(snr)
        self.t = float(t)

        sep = abs(c1 - c0)
        assert sep > 0, "c0 and c1 must differ"
        # Shrink |1> toward |0> by t/2 along the c0->c1 axis (readout decay).
        self.c0 = complex(c0)
        self.c1 = complex(c0 + (c1 - c0) * (1.0 - t / 2.0))
        self.c2 = complex(c2)

        base_sigma = sep / math.sqrt(2.0 * snr)
        self.sigma = np.array([base_sigma * s for s in sigma_scale], dtype=np.float64)
        self.centers = np.array([self.c0, self.c1, self.c2], dtype=np.complex128)
        self.priors = np.array(priors, dtype=np.float64)
        self.priors /= self.priors.sum()

    # --- sampling -----------------------------------------------------------

    def sample_iq(self, true_bits: np.ndarray, leaked: Optional[np.ndarray],
                  rng: np.random.Generator) -> np.ndarray:
        """Draw one complex IQ value per measurement from the true state.

        true_bits : bool/int ndarray  (0 -> |0>, 1 -> |1>)   any shape
        leaked    : bool ndarray same shape (True -> |2>), or None
        """
        shape = true_bits.shape
        state = np.where(true_bits.astype(bool), 1, 0).astype(np.int64)
        if leaked is not None:
            state = np.where(leaked.astype(bool), 2, state)

        mean = self.centers[state]                     # complex means
        sig = self.sigma[state]                        # per-sample sigma
        noise = (rng.normal(0.0, 1.0, size=shape) +
                 1j * rng.normal(0.0, 1.0, size=shape))
        z = mean + noise * sig
        return z.astype(np.complex64)

    # --- soft posteriors ----------------------------------------------------

    def _log_likelihoods(self, iq: np.ndarray) -> np.ndarray:
        """Return log N(z | c_k, sigma_k) for k in {0,1,2}. Shape (..., 3)."""
        z = iq.astype(np.complex128)[..., None]        # (..., 1)
        c = self.centers[(None,) * iq.ndim]            # (..., 3) via broadcast
        s2 = self.sigma ** 2
        # 2D isotropic complex Gaussian: -|z-c|^2/(2 s^2) - log(2 pi s^2)
        d2 = np.abs(z - c) ** 2
        return (-d2 / (2.0 * s2) - np.log(2.0 * np.pi * s2)).astype(np.float64)

    def posteriors(self, iq: np.ndarray,
                   priors: Optional[np.ndarray] = None) -> np.ndarray:
        """P(state | z) for each of the three centers. Shape (..., 3)."""
        log_lr = self._log_likelihoods(iq)
        pri = self.priors if priors is None else priors
        log_post = log_lr + np.log(np.clip(pri, 1e-12, None))
        log_post -= log_post.max(axis=-1, keepdims=True)
        p = np.exp(log_post)
        p /= p.sum(axis=-1, keepdims=True)
        return p.astype(np.float32)

    def to_soft_prob(self, iq: np.ndarray) -> np.ndarray:
        """P(qubit == 1 | z), marginalizing out the leaked state (|2> folded
        into the excited manifold for the syndrome bit)."""
        p = self.posteriors(iq)
        # Treat |2> as "not a clean 0": fold half of its mass into 1 is a
        # modeling choice; here we report P(1) over the {0,1} manifold only.
        p0, p1 = p[..., 0], p[..., 1]
        denom = np.clip(p0 + p1, 1e-12, None)
        return (p1 / denom).astype(np.float32)

    def leak_prob(self, iq: np.ndarray) -> np.ndarray:
        """P(qubit == |2> | z)  — a separate leakage soft channel."""
        return self.posteriors(iq)[..., 2].astype(np.float32)

    # --- hard decision ------------------------------------------------------

    def hard_decision(self, iq: np.ndarray) -> np.ndarray:
        """Binary readout for detector reconstruction. A leaked sample is
        projected onto the nearer of {|0>, |1>} (leakage removed for the
        syndrome bit; the leak channel carries the |2> information)."""
        z = iq.astype(np.complex128)[..., None]
        d = np.abs(z - self.centers[(None,) * iq.ndim])
        # ignore the |2> center for the *binary* bit
        d01 = d[..., :2]
        return (np.argmin(d01, axis=-1) == 1)

    def hard_state(self, iq: np.ndarray) -> np.ndarray:
        """3-way nearest-center classification: 0/1/2."""
        z = iq.astype(np.complex128)[..., None]
        d = np.abs(z - self.centers[(None,) * iq.ndim])
        return np.argmin(d, axis=-1).astype(np.uint8)

# ---------------------------------------------------------------------------
# 2. Circuit builder WITHOUT readout SPAM (readout handled by IQ model)
# ---------------------------------------------------------------------------

def build_paems_circuit_no_readout_spam(base_circuit: stim.Circuit,
                                        params: Dict, rounds: int) -> stim.Circuit:
    """Same PAEMS gate/idle noise as build_paems_noisy_circuit, but WITHOUT the
    readout-decision X_ERRORs (measurement_spam_rate on stabs, and
    data_measurement_error on the final M). Those are replaced by the IQ model.
    Initialization noise (data_init_error after R) is kept."""
    data_qubits, stab_qubits, cx_pairs = _extract_layout(base_circuit)
    data_set = set(data_qubits)
    cx_lookup = {(c, t): str(gid) for gid, (c, t) in enumerate(cx_pairs, start=1)}
    cx_lookup = {k: v for k, v in cx_lookup.items() if v in params["cx_gates"]}
    pair_lookup = build_crosstalk_lookup(params.get("crosstalk_pairs", {}))
    xtalk_on = bool(pair_lookup)
    qparams = params["qubits"]

    flat = base_circuit.flattened()
    last_mr_idx = max((i for i, inst in enumerate(flat) if inst.name == "MR"),
                      default=-1)

    noisy = stim.Circuit()
    for i, inst in enumerate(flat):
        name = inst.name

        if name in ("R", "RX"):
            targets = [t.value for t in inst.targets_copy()]
            noisy.append("R", targets)
            for q in targets:
                if q in data_set and qparams[str(q)]["data_init_error"] > 0:
                    noisy.append("X_ERROR", q, qparams[str(q)]["data_init_error"])

        elif name == "H":
            targets = [t.value for t in inst.targets_copy()]
            noisy.append("H", targets)
            for q in targets:
                nrm = single_qubit_noise(qparams[str(q)])
                noisy.append("PAULI_CHANNEL_1", q, list(nrm["px_py_pz"]))
                if nrm["p1"] > 0:
                    noisy.append("DEPOLARIZE1", q, nrm["p1"])
                if xtalk_on:
                    _add_spectator_crosstalk(noisy, q, pair_lookup)

        elif name == "CX":
            ts = inst.targets_copy()
            for j in range(0, len(ts), 2):
                t_ctrl, t_tgt = ts[j], ts[j + 1]
                if not (t_ctrl.is_qubit_target and t_tgt.is_qubit_target):
                    noisy.append("CX", [t_ctrl, t_tgt])
                    continue
                c, t = t_ctrl.value, t_tgt.value
                noisy.append("CX", [c, t])
                if (c, t) in cx_lookup:
                    n2 = two_qubit_noise(qparams[str(c)], qparams[str(t)],
                                         params["cx_gates"][cx_lookup[(c, t)]])
                    noisy.append("PAULI_CHANNEL_1", c, list(n2["control_pauli"]))
                    noisy.append("PAULI_CHANNEL_1", t, list(n2["target_pauli"]))
                    if n2["p2"] > 0:
                        noisy.append("DEPOLARIZE2", [c, t], n2["p2"])
                if xtalk_on:
                    _add_spectator_crosstalk(noisy, c, pair_lookup)
                    _add_spectator_crosstalk(noisy, t, pair_lookup)

        elif name == "MR":
            targets = [t.value for t in inst.targets_copy()]
            is_last_round = (i == last_mr_idx)
            if not is_last_round:                 # data idle/decoherence noise kept
                for q in targets:
                    if q in data_set:
                        noisy.append("PAULI_CHANNEL_1", q, list(readout_noise(qparams[str(q)])))
            # NO X_ERROR(measurement_spam_rate) — readout is the IQ model's job
            noisy.append("MR", targets)

        elif name == "M":
            targets = [t.value for t in inst.targets_copy()]
            # NO X_ERROR(data_measurement_error) — readout is the IQ model's job
            noisy.append("M", targets)

        else:
            noisy.append(name, _copy_targets(inst), inst.gate_args_copy())

    return noisy

# ---------------------------------------------------------------------------
# 3. Detector reconstruction from IQ hard decision
# ---------------------------------------------------------------------------

def rebuild_detectors(circuit: stim.Circuit,
                      hard_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map IQ hard-decision measurements -> (detection_events, observables)
    using the circuit's own DETECTOR/OBSERVABLE definitions. Guarantees the
    detectors are consistent with the same IQ shot that gave the soft info."""
    converter = circuit.compile_m2d_converter()
    dets, obs = converter.convert(measurements=hard_bits.astype(np.bool_),
                                  separate_observables=True)
    return np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool).reshape(-1)

def _derive_seeds(seed: Optional[int] = None):
    """把一个可选主种子拆成互相独立的随机流。

    seed=None  -> 每次运行从 OS 熵源取新种子（不可复现，模拟真实随机噪声）
    seed=int   -> 完全可复现
    """
    ss = np.random.SeedSequence(seed)          # seed=None 时自动取 OS 熵
    c_stim, c_iq, c_leak = ss.spawn(3)         # 三条互不相关的子序列
    # stim 采样器需要一个整数种子
    stim_seed = int(np.random.default_rng(c_stim).integers(1, 2**63 - 1))
    iq_rng = np.random.default_rng(c_iq)       # IQ 高斯噪声流
    leak_master = np.random.default_rng(c_leak)  # 派生每个 chunk 的泄漏种子
    return stim_seed, iq_rng, leak_master

# ---------------------------------------------------------------------------
# 4. Full IQ sampling pipeline
# ---------------------------------------------------------------------------

def sample_paems_iq_dataset(distance: int, rounds: int, num_samples: int,
                            params: Dict, *,
                            iq: Optional[ThreeCenterIQReadoutSimulator] = None,
                            snr: float = 10.0, t: float = 0.01,
                            seed: Optional[int] = None,          # ← 默认 None = 随机
                            chunk_size: int = 4000,
                            include_leakage: bool = True) -> Dict[str, np.ndarray]:
    base = _base_surface_code_circuit(distance, rounds)
    data_qubits, stab_qubits, cx_pairs = _extract_layout(base)
    n_stab = distance ** 2 - 1
    n_data = distance ** 2

    noisy = build_paems_circuit_no_readout_spam(base, params, rounds)

    # ---- 派生独立随机流（seed=None 时每次运行都不同）----
    stim_seed, rng, leak_master = _derive_seeds(seed)
    meas_sampler = noisy.compile_sampler(seed=stim_seed)

    if iq is None:
        iq = ThreeCenterIQReadoutSimulator(snr=snr, t=t)

    # (删除原来的 rng = np.random.default_rng(seed + 7))

    iq_meas_l, iq_final_l = [], []
    soft_meas_l, soft_final_l = [], []
    leak_soft_l, final_leak_soft_l = [], []
    det_l, obs_l, leak_l = [], [], []

    start = 0
    while start < num_samples:
        chunk = min(chunk_size, num_samples - start)

        true_bits = meas_sampler.sample(shots=chunk)

        if include_leakage:
            # 每个 chunk 从泄漏主流抽一个独立整数种子
            chunk_leak_seed = int(leak_master.integers(1, 2**63 - 1))
            affected = simulate_leakage(noisy, data_qubits, stab_qubits, cx_pairs,
                                        params, chunk,
                                        seed=chunk_leak_seed,     # ← 改这里
                                        batch_size=max(400, chunk))
            leaked_mask = affected.astype(bool)
            stab_leak = affected[:, :rounds * n_stab].reshape(chunk, rounds, n_stab)
        else:
            leaked_mask = None
            stab_leak = np.zeros((chunk, rounds, n_stab), dtype=np.uint8)

        iq_all = iq.sample_iq(true_bits, leaked_mask, rng)   # rng 已是独立 IQ 流

        # detectors/observables from the SAME IQ shot (hard decision)
        hard = iq.hard_decision(iq_all)
        dets, obs = rebuild_detectors(noisy, hard)

        # split records
        iq_anc = iq_all[:, :rounds * n_stab].reshape(chunk, rounds, n_stab)
        iq_fin = iq_all[:, rounds * n_stab:]

        soft_anc = iq.to_soft_prob(iq_anc)                   # P(1|IQ)
        soft_fin = iq.to_soft_prob(iq_fin)
        leak_anc = iq.leak_prob(iq_anc)                      # P(|2>|IQ)
        leak_fin = iq.leak_prob(iq_fin)

        iq_meas_l.append(iq_anc.astype(np.complex64))
        iq_final_l.append(iq_fin.astype(np.complex64))
        soft_meas_l.append(soft_anc)
        soft_final_l.append(soft_fin)
        leak_soft_l.append(leak_anc)
        final_leak_soft_l.append(leak_fin)
        det_l.append(dets)
        obs_l.append(obs)
        leak_l.append(stab_leak.astype(np.float32))

        start += chunk

    measurement = np.concatenate(soft_meas_l, axis=0)
    final_soft = np.concatenate(soft_final_l, axis=0)

    event = np.zeros_like(measurement)
    event[:, 0, :] = measurement[:, 0, :]
    for tt in range(1, rounds):
        event[:, tt, :] = soft_xor(measurement[:, tt, :], measurement[:, tt - 1, :])

    return {
        "measurement_iq": np.concatenate(iq_meas_l, axis=0),
        "final_iq": np.concatenate(iq_final_l, axis=0),
        "measurement": measurement,
        "event": event,
        "final_soft": final_soft,
        "leak_soft": np.concatenate(leak_soft_l, axis=0),
        "final_leak_soft": np.concatenate(final_leak_soft_l, axis=0),
        "detection_events": np.concatenate(det_l, axis=0).astype(np.float32),
        "label": np.concatenate(obs_l, axis=0).astype(np.float32),
        "leakage": np.concatenate(leak_l, axis=0),
        "distance": int(distance),
        "rounds": int(rounds),
        "snr": float(iq.snr),
        "_meta": {
            "master_seed": seed,                 # None 表示随机运行；整数表示可复现
            "stim_seed": int(stim_seed),         # 实际使用的 stim 种子（回填即可复现）
            "seed": (int(seed) if seed is not None else None),  # 兼容旧字段
            "t": float(iq.t),
            "readout_model": "three_center_IQ",
            "centers": [complex(iq.c0), complex(iq.c1), complex(iq.c2)],
            "sigma": iq.sigma.tolist(),
            "include_leakage": bool(include_leakage),
        },
    }

# ---------------------------------------------------------------------------
# 4b. 流式 IQ 采样生成器（不积累全量数组，处理完即释放）
# ---------------------------------------------------------------------------
import gc as _gc

def stream_paems_iq_dataset(distance: int, rounds: int, num_samples: int,params: Dict, *,
                             iq: Optional[ThreeCenterIQReadoutSimulator] = None,
                             snr: float = 10.0, t: float = 0.01,
                             seed: Optional[int] = None,
                             chunk_size: int = 4000,
                             include_leakage: bool = True):
    """
    生成器版本：每次 yield 一个 chunk 大小的数据字典，调用方处理后立即丢弃，
    全程内存只占用一个 chunk，不落盘。

    chunk字典键与sample_paems_iq_dataset 返回值一致：measurement_iq, final_iq, measurement, event, final_soft,
      leak_soft, final_leak_soft, detection_events, label, leakage

    用法示例：
        for chunk in stream_paems_iq_dataset(3, 25, 50000, params, seed=42):
            logits = decoder(chunk['detection_events'])   # 直接推理
            # chunk离开作用域后被GC 回收，不保存
    """
    base = _base_surface_code_circuit(distance, rounds)
    data_qubits, stab_qubits, cx_pairs = _extract_layout(base)
    n_stab = distance ** 2 - 1

    noisy = build_paems_circuit_no_readout_spam(base, params, rounds)
    stim_seed, rng, leak_master = _derive_seeds(seed)
    meas_sampler = noisy.compile_sampler(seed=stim_seed)

    if iq is None:
        iq = ThreeCenterIQReadoutSimulator(snr=snr, t=t)

    start = 0
    while start < num_samples:
        chunk_n = min(chunk_size, num_samples - start)
        true_bits = meas_sampler.sample(shots=chunk_n)

        #──泄漏模拟 ──────────────────────────────────────────────────────
        if include_leakage:
            chunk_leak_seed = int(leak_master.integers(1, 2 ** 63 - 1))
            affected = simulate_leakage(
                noisy, data_qubits, stab_qubits, cx_pairs,
                params, chunk_n, seed=chunk_leak_seed,
                batch_size=max(400, chunk_n))
            leaked_mask = affected.astype(bool)
            stab_leak = affected[:, :rounds * n_stab].reshape(chunk_n, rounds, n_stab)
        else:
            leaked_mask = None
            stab_leak = np.zeros((chunk_n, rounds, n_stab), dtype=np.uint8)

        # ── IQ 采样 →硬决策 → 探测器 ────────────────────────────────────
        iq_all = iq.sample_iq(true_bits, leaked_mask, rng)
        hard = iq.hard_decision(iq_all)
        dets, obs = rebuild_detectors(noisy, hard)

        # ── 拆分 ancilla / final──────────────────────────────────────────iq_anc = iq_all[:, :rounds * n_stab].reshape(chunk_n, rounds, n_stab)
        iq_fin = iq_all[:, rounds * n_stab:]

        soft_anc = iq.to_soft_prob(iq_anc)
        soft_fin = iq.to_soft_prob(iq_fin)
        leak_anc = iq.leak_prob(iq_anc)
        leak_fin = iq.leak_prob(iq_fin)

        # event = soft XOR 相邻轮差分
        event = np.zeros_like(soft_anc)
        event[:, 0, :] = soft_anc[:, 0, :]
        for tt in range(1, rounds):
            event[:, tt, :] = soft_xor(soft_anc[:, tt, :], soft_anc[:, tt - 1, :])

        chunk_data = {
            "measurement_iq":iq_anc.astype(np.complex64),
            "final_iq":          iq_fin.astype(np.complex64),
            "measurement":       soft_anc,
            "event":             event,
            "final_soft":        soft_fin,
            "leak_soft":         leak_anc,
            "final_leak_soft":   leak_fin,
            "detection_events":  dets.astype(np.float32),
            "label":             obs.astype(np.float32),
            "leakage":           stab_leak.astype(np.float32),
        }

        yield chunk_data   # ← 交给调用方，调用方 return 后 GC 可回收

        # 主动清理（大chunk 场景显式触发）
        del chunk_data, iq_all, true_bits, hard, dets, obs
        _gc.collect()

        start += chunk_n

# ---------------------------------------------------------------------------
# 5. Saving (complex-aware)
# ---------------------------------------------------------------------------

def save_pt(data: Dict, out_path: str | Path) -> None:
    """Save the IQ dataset as a torch .pt file. Complex fields are stored as
    complex64 tensors (torch supports them natively)."""
    import torch
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def T(x):
        arr = np.asarray(x)
        if np.iscomplexobj(arr):
            return torch.from_numpy(arr.astype(np.complex64))
        return torch.from_numpy(arr.astype(np.float32))

    pt = {k: T(v) for k, v in data.items()
          if isinstance(v, np.ndarray)}
    pt.update({"distance": int(data["distance"]),
               "rounds": int(data["rounds"]),
               "snr": float(data["snr"]),
               "_meta": {kk: (str(vv) if isinstance(vv, list) else vv)
                         for kk, vv in data.get("_meta", {}).items()}})
    torch.save(pt, str(out_path))

# ---------------------------------------------------------------------------
# 6. Self-consistency verification
# ---------------------------------------------------------------------------

def verify(distance: int = 3, rounds: int = 3, n: int = 2000,
           seed: Optional[int] = None,      # ← 采样种子：None = 每次随机
           device_seed: int = 7) -> None:   # ← 设备参数种子：固定 = 同一块芯片
    params = generate_paems_params(distance, seed=device_seed)
    iqsim = ThreeCenterIQReadoutSimulator(snr=8.0, t=0.02)
    data = sample_paems_iq_dataset(distance, rounds, n, params,
                                   iq=iqsim, seed=seed, include_leakage=True)

    print(f"    run stim_seed = {data['_meta']['stim_seed']}  "
          f"(master_seed={data['_meta']['master_seed']})")

    # (a)+(b) reconstruct hard bits from stored IQ and compare to soft argmax
    iq_anc = data["measurement_iq"]
    soft = data["measurement"]
    hard = iqsim.hard_decision(iq_anc)
    agree = np.mean((soft > 0.5) == hard)
    print(f"[b] P(1)>0.5 vs hard_decision agreement: {agree:.4f}  (expect ~1.0)")

    # (c) leaked ground truth vs leak_soft
    leak_gt = data["leakage"].astype(bool)
    if leak_gt.any():
        mean_leak_when_leaked = data["leak_soft"][leak_gt].mean()
        mean_leak_when_clean = data["leak_soft"][~leak_gt].mean()
        print(f"[c] mean P(|2>) leaked={mean_leak_when_leaked:.3f}  "
              f"clean={mean_leak_when_clean:.3f}  (leaked should be higher)")
    else:
        print("[c] no leakage in this batch (increase n or lp)")

    # (d) detector count
    base = _base_surface_code_circuit(distance, rounds)
    noisy = build_paems_circuit_no_readout_spam(base, params, rounds)
    n_det = noisy.num_detectors
    print(f"[d] detection_events shape={data['detection_events'].shape}  "
          f"circuit detectors={n_det}  match={data['detection_events'].shape[1] == n_det}")

    # value ranges
    print(f"    measurement range: [{soft.min():.3f}, {soft.max():.3f}]")
    print(f"    label mean (logical error rate): {data['label'].mean():.4f}")
    print(f"    IQ example (first 3): {iq_anc.reshape(-1)[:3]}")

if __name__ == "__main__":
    verify()               # seed=None -> 每次运行数值都不同
    # 需要复现某次运行时，回填该次打印的 stim_seed：
    # verify(seed=123456789)

    # --- 可选：生成并保存一个完整数据集 ---
    # params = generate_paems_params(distance=3, seed=42)   # 芯片参数固定
    # iqsim = ThreeCenterIQReadoutSimulator(
    #     c0=0+0j, c1=1+0j, c2=0.5+1.0j,
    #     snr=10.0, t=0.01,
    #     sigma_scale=(1.0, 1.0, 1.3),
    #     priors=(0.5, 0.48, 0.02),
    # )
    # ds = sample_paems_iq_dataset(3, 5, 10_000, params, iq=iqsim, seed=None)
    # save_pt(ds, "data/d3_r5_iq.pt")
    # print(f"saved dataset, stim_seed={ds['_meta']['stim_seed']}")
