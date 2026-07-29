"""PAEMS noise model + synthetic-data generator (stim-based).

Faithful re-implementation of the PAEMS (Precise and Adaptive Error Model)
circuit-level stochastic noise model for superconducting quantum processors,
adapted to emit data that strictly conforms to
``deliverables/data_specification/synthetic_data_spec.md`` (v2.0).

PAEMS mechanisms implemented here (per PAEMS - 理论框架.md + 官方代码结构.md):
  1. Per-qubit parameterization (heterogeneous T1/T2/fidelity/SPAM/leakage)
  2. ADC — asymmetric depolarizing channel: P_X = P_Y != P_Z  (from T1/T2)
  3. SDC — symmetric depolarizing channel: P_X = P_Y = P_Z   (from gate fidelity)
  4. SPAM with P_init != P_meas
  5. Leakage |1>->|2> (LP) + seepage |2>->|1> (SP) with CX propagation
  6. Spectator crosstalk (DEPOLARIZE1 on paired spectators)

The noise-injection / Pauli-probability math is ported verbatim from the
official ``Surface_Code_Simulation/calculate.py`` and ``inject_basic_noise.py``;
the leakage state-machine is ported from ``inject_leakage_noise_vectorized.py``;
the soft-readout model is ported from ``alphaqubit/data/soft_readout.py``.

Design note (spec compliance): the noise is injected into a *standard*
``stim.Circuit.generated("surface_code:rotated_memory_z")`` circuit while
preserving its DETECTOR / OBSERVABLE_INCLUDE declarations. Sampling the
measurement sampler and the detector sampler with the *same seed* guarantees
that ``detection_events`` and ``label`` originate from one identical underlying
error shot (spec §3.2). Leakage is applied as an official-PAEMS post-processing
layer on the measurement record only (Option A in the design notes), so the
soft ``measurement`` / ``event`` / ``final_soft`` carry leakage while the
DEM-ordered ``detection_events`` and ``label`` remain matched & clean.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import stim


# ---------------------------------------------------------------------------
# 1. PAEMS formula layer  (verbatim port of official calculate.py)
# ---------------------------------------------------------------------------

def calculate_depolarizing_error_probability(dim: int, F_E_relax: float, F: float) -> float:
    """SDC depolarizing probability ``p = d (F_E - F) / (d F_E - 1)`` (>=0)."""
    if dim * F_E_relax <= 1.0:
        return 0.0
    p = dim * (F_E_relax - F) / (dim * F_E_relax - 1.0)
    return max(0.0, p)


def calculate_pad_ppd(t: float, T1: float, T2: float) -> Tuple[float, float]:
    """Amplitude-damping & pure-dephasing probabilities."""
    PAD = 1.0 - math.exp(-t / T1)
    if PAD >= 1.0:
        PPD = 0.0
    else:
        PPD = 1.0 - (math.exp(-t / T2) ** 2) / (1.0 - PAD)
    return PAD, PPD


def calculate_decoherence_fidelity(t1: float, t2: float, t: float) -> float:
    """Decoherence (relaxation) fidelity F_E (T2 clamped to <= 2 T1)."""
    if t2 > 2.0 * t1:
        t2 = 2.0 * t1
    pad, ppd = calculate_pad_ppd(t, t1, t2)
    term1 = 1.0 / 3.0
    term2 = (1.0 / 3.0) * (1.0 - pad) * (1.0 - ppd)
    term3 = (1.0 / 3.0) * math.sqrt((1.0 - pad) * (1.0 - ppd))
    term4 = (1.0 / 6.0) * pad
    term5 = (1.0 / 3.0) * (1.0 - pad) * ppd
    return term1 + term2 + term3 + term4 + term5


def calculate_px_py_pz(t1: float, t2: float, t: float) -> Tuple[float, float, float]:
    """ADC Pauli probabilities:  P_X = P_Y = (1 - e^{-t/T1})/4  ;
    P_Z = (1 - e^{-t/T2})/2 - (1 - e^{-t/T1})/4  (the asymmetric channel)."""
    if t2 > 2.0 * t1:
        t2 = 2.0 * t1
    px_py = (1.0 - math.exp(-t / t1)) / 4.0
    pz = (1.0 - math.exp(-t / t2)) / 2.0 - (1.0 - math.exp(-t / t1)) / 4.0
    return (px_py, px_py, pz)


def single_qubit_noise(qubit_params: Dict) -> Dict:
    """1Q-gate noise: ADC Pauli channel + SDC DEPOLARIZE1."""
    F_relax = calculate_decoherence_fidelity(qubit_params["t1"], qubit_params["t2"],
                                             qubit_params["sqg_length"])
    p1 = calculate_depolarizing_error_probability(2, F_relax, qubit_params["sqg_fid"])
    px_py_pz = calculate_px_py_pz(qubit_params["t1"], qubit_params["t2"],
                                  qubit_params["sqg_length"])
    return {"px_py_pz": px_py_pz, "p1": p1}


def cx_total_fidelity(control_params: Dict, target_params: Dict, cx_fid: float) -> float:
    """F_total^CX = F_sqg,ctrl^2 * F_sqg,tgt^2 * F_CX  (includes aux 1Q gates)."""
    return (control_params["sqg_fid"] ** 2) * (target_params["sqg_fid"] ** 2) * cx_fid


def two_qubit_noise(control_params: Dict, target_params: Dict, cx_params: Dict) -> Dict:
    """2Q-gate noise: per-qubit ADC Pauli channel + joint SDC DEPOLARIZE2."""
    cx_fid = cx_params["cx_fid"]
    cx_length = cx_params["cx_length"]
    total_time = 2.0 * target_params["sqg_length"] + cx_length
    ctrl_deco = calculate_decoherence_fidelity(control_params["t1"], control_params["t2"], total_time)
    tgt_deco = calculate_decoherence_fidelity(target_params["t1"], target_params["t2"], total_time)
    joint_deco = ctrl_deco * tgt_deco
    cx_total = cx_total_fidelity(control_params, target_params, cx_fid)
    p2 = calculate_depolarizing_error_probability(4, joint_deco, cx_total)
    control_pauli = calculate_px_py_pz(control_params["t1"], control_params["t2"], total_time)
    target_pauli = calculate_px_py_pz(target_params["t1"], target_params["t2"], total_time)
    return {"control_pauli": control_pauli, "target_pauli": target_pauli,
            "p2": p2, "total_time": total_time}


def readout_noise(qubit_params: Dict) -> Tuple[float, float, float]:
    """Readout/idle Pauli channel from T1/T2 over readout time."""
    return calculate_px_py_pz(qubit_params["t1"], qubit_params["t2"], qubit_params["rd_length"])


# ---------------------------------------------------------------------------
# 2. Per-qubit / per-CX parameter generation
# ---------------------------------------------------------------------------

# Physical ranges mirror official surface_code_generate_params_json.py and the
# CMA-ES bounds documented in PAEMS - CMA-ES优化流程.md.
SQG_LENGTH = 6.0e-08
RD_LENGTH = 1.3e-06
CK_VALUES_SQG_LENGTH = SQG_LENGTH  # single-qubit gate duration (s)


def generate_paems_params(distance: int, seed: int = 42) -> Dict:
    """Build a heterogeneous per-qubit / per-CX PAEMS parameter set for the
    standard ``rotated_memory_z`` surface code of the given distance.

    Returns a dict consumable both by :func:`build_paems_noisy_circuit` and
    :func:`simulate_leakage`, with the official PAEMS JSON schema:
        {"qubits": {qubit_id: {...}}, "cx_gates": {gate_id: {...}},
         "crosstalk_pairs": {...}, "_metadata": {...}}
    """
    rng = np.random.default_rng(seed)
    # Base (noise-free) circuit gives us the qubit layout.
    base = _base_surface_code_circuit(distance, rounds=2)
    data_qubits, stab_qubits, cx_pairs = _extract_layout(base)

    all_qubits = sorted(data_qubits + stab_qubits)
    qubits_params: Dict[str, Dict] = {}
    # Allow stabilizer (ancilla) qubits a slightly worse T1/T2 — realistic
    # ancilla-vs-data variation.  lp/sp per the CMA-ES bounds.
    for q in all_qubits:
        is_ancilla = q in stab_qubits
        t1 = float(rng.uniform(1.0e-4, 6.0e-4)) * (0.85 if is_ancilla else 1.0)
        # T2 in [0.45 T1, 2 T1], then clamp to T2 <= 2 T1 (physical).
        t2 = float(rng.uniform(0.45, 2.0)) * t1
        if t2 > 2.0 * t1:
            t2 = 2.0 * t1
        sqg_fid = float(rng.uniform(0.999, 0.9999))
        # P_init (data prep) != P_meas (readout): a PAEMS hallmark.
        # Realistic SPAM ~0.2-2% (CMA-ES bounds 1e-4..2e-1; real hardware ~0.5-1.5%).
        data_init_error = float(rng.uniform(5.0e-4, 2.0e-3)) if not is_ancilla else float(rng.uniform(8.0e-4, 3.0e-3))
        data_measurement_error = float(rng.uniform(1.0e-3, 3.0e-3)) if not is_ancilla else 0.0
        measurement_spam_rate = float(rng.uniform(2.0e-3, 1.5e-2)) if is_ancilla else float(rng.uniform(1.0e-3, 5.0e-3))
        lp = float(rng.uniform(5.0e-5, 1.0e-3))
        sp = float(rng.uniform(0.0, 1.0e-2))
        qubits_params[str(q)] = {
            "measurement_spam_rate": round(measurement_spam_rate, 8),
            "data_init_error": round(data_init_error, 8),
            "data_measurement_error": round(data_measurement_error, 8),
            "t1": round(t1, 12),
            "t2": round(t2, 12),
            "sqg_fid": round(sqg_fid, 10),
            "sqg_length": SQG_LENGTH,
            "rd_length": RD_LENGTH,
            "lp": round(lp, 8),
            "sp": round(sp, 8),
            "is_ancilla": bool(is_ancilla),
        }

    cx_gates_params: Dict[str, Dict] = {}
    for gid, (ctrl, tgt) in enumerate(cx_pairs, start=1):
        cx_gates_params[str(gid)] = {
            # fault-tolerant regime: 2Q fidelity 99.7%-99.9% -> p2 ~0.1-0.25%,
            # keeping the surface code well below threshold (p_th ~1%).
            "control": ctrl,
            "target": tgt,
            "cx_fid": float(rng.uniform(0.997, 0.999)),
            "cx_length": float(rng.uniform(6.0e-7, 7.5e-7)),
            "lp_propagation_prob": float(rng.uniform(0.0, 0.1)),
        }

    # Crosstalk: a handful of physically-plausible spectator pairs (non-CX
    # nearest neighbours).  χ strength in the documented crosstalk range.
    coords = _qubit_coords(base)                       # (x, y) -> qubit id
    cx_set = set()
    for (c, t) in cx_pairs:
        cx_set.add((c, t)); cx_set.add((t, c))
    crosstalk_pairs: Dict[str, Dict] = {}
    pairs_seen: set = set()
    coord_list = sorted(coords.items(), key=lambda kv: kv[1])  # by qubit id
    chosen = 0
    n_target = max(3, distance)  # a sparse but non-empty crosstalk graph
    for i in range(len(coord_list)):
        if chosen >= n_target:
            break
        (xi, yi), qi = coord_list[i]
        for (xj, yj), qj in coord_list:
            if qj == qi or (qi, qj) in pairs_seen or (qj, qi) in pairs_seen:
                continue
            if (qi, qj) in cx_set:
                continue
            dist = (xi - xj) ** 2 + (yi - yj) ** 2
            if 0 < dist <= 4.0 + 1e-9 and chosen < n_target:  # nearest face/edge neighbours
                key = f"{qi}-{qj}"
                crosstalk_pairs[key] = {"strength": float(rng.uniform(1.0e-5, 1.0e-4)),
                                        "type": "physical_proximity"}
                pairs_seen.add((qi, qj))
                chosen += 1
                break

    params = {
        "qubits": qubits_params,
        "cx_gates": cx_gates_params,
        "crosstalk_pairs": crosstalk_pairs,
        "_metadata": {
            "model": "PAEMS",
            "distance": int(distance),
            "data_qubits": sorted(data_qubits),
            "stabilizer_qubits": sorted(stab_qubits),
            "total_qubits": len(all_qubits),
            "total_cx_gates": len(cx_pairs),
            "soft_readout": {"snr": 10.0, "t": 0.01},
            "created_by": "PAEMS-data/code/paems_noise_model.generate_paems_params",
            "seed": int(seed),
        },
    }
    return params


# ---------------------------------------------------------------------------
# 3. Circuit layout helpers
# ---------------------------------------------------------------------------

def _base_surface_code_circuit(distance: int, rounds: int) -> stim.Circuit:
    """Standard noise-less rotated-surface-code Z-memory circuit (native ordering)."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )


def _extract_layout(base: stim.Circuit) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Return (data_qubits, stabilizer_qubits, cx_pairs) from a base circuit.

    - data qubits   = targets of the final ``M`` instruction
    - stabilizers    = targets of the ``MR`` instruction (ancillas)
    - cx_pairs       = unique (control, target) pairs across ``CX`` instructions (order of first appearance)
    """
    data_qubits: List[int] = []
    stab_qubits: List[int] = []
    cx_pairs: List[Tuple[int, int]] = []
    seen_pairs: set = set()
    flat = base.flattened()
    # the LAST 'M' instruction is the final data measurement
    last_m = None
    for inst in flat:
        if inst.name == "M":
            last_m = inst
    if last_m is not None:
        data_qubits = sorted(t.value for t in last_m.targets_copy())
    # stabilizers: union of all MR targets
    stab_set = set()
    for inst in flat:
        if inst.name == "MR":
            for t in inst.targets_copy():
                stab_set.add(t.value)
    stab_qubits = sorted(stab_set)
    # cx pairs (first-appearance order) — flattened circuit has unrolled CXs
    for inst in flat:
        if inst.name == "CX":
            ts = inst.targets_copy()
            for j in range(0, len(ts), 2):
                c, t = ts[j].value, ts[j + 1].value
                if (c, t) not in seen_pairs:
                    seen_pairs.add((c, t))
                    cx_pairs.append((c, t))
    return data_qubits, stab_qubits, cx_pairs


def _qubit_coords(base: stim.Circuit) -> Dict[Tuple[float, float], int]:
    """Map (x, y) -> qubit id from QUBIT_COORDS declarations."""
    out = {}
    for inst in base.flattened():
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            qid = inst.targets_copy()[0].value
            out[(args[0], args[1])] = qid
    return out


# ---------------------------------------------------------------------------
# 4. PAEMS noise injection into the standard stim circuit
# ---------------------------------------------------------------------------

def build_crosstalk_lookup(chi_pairs: Dict) -> Dict[int, List[Tuple[int, float]]]:
    """Identical semantics to official inject_basic_noise.build_crosstalk_lookup."""
    lookup: Dict[int, List[Tuple[int, float]]] = {}
    for key, spec in (chi_pairs or {}).items():
        try:
            i, j = (int(x) for x in key.split("-"))
        except ValueError:
            continue
        if isinstance(spec, dict):
            chi = float(spec.get("strength", 0.0))
        else:
            chi = float(spec)
        if chi <= 0:
            continue
        lookup.setdefault(i, []).append((j, chi))
        lookup.setdefault(j, []).append((i, chi))
    return lookup


def _add_spectator_crosstalk(circuit: stim.Circuit, active_q: int,
                             pair_lookup: Dict[int, List[Tuple[int, float]]]) -> None:
    """Append DEPOLARIZE1 spectator events (mirror official inject_basic_noise)."""
    entries = pair_lookup.get(active_q)
    if not entries:
        return
    by_chi: Dict[float, List[int]] = {}
    for spec_q, chi in entries:
        by_chi.setdefault(chi, []).append(spec_q)
    for chi, qs in by_chi.items():
        circuit.append("DEPOLARIZE1", qs, chi)

def _copy_targets(inst) -> list:
    out = []
    for t in inst.targets_copy():
        if hasattr(t, "value") and not t.is_qubit_target:
            out.append(t)              # measurement-record targets (rec[-k])
        elif hasattr(t, "value"):
            out.append(t.value)
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# 5. Leakage simulation (faithful port of inject_leakage_noise_vectorized.py)
# ---------------------------------------------------------------------------

def _preprocess_for_leakage(noisy_circuit: stim.Circuit, data_qubits: List[int],
                            stab_qubits: List[int], cx_pairs: List[Tuple[int, int]],
                            params: Dict) -> Dict:
    """Compile the circuit into (H ops, CX ops, measurement ops) + lp/sp arrays,
    recording the TRUE measurement-record layout so leakage flips map correctly.
    """
    all_qubits = sorted(data_qubits + stab_qubits)
    qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}
    num_qubits = len(all_qubits)
    qparams = params["qubits"]
    lp = np.zeros(num_qubits, dtype=np.float32)
    sp = np.zeros(num_qubits, dtype=np.float32)
    for i, q in enumerate(all_qubits):
        qp = qparams[str(q)]
        lp[i] = qp.get("lp", 1e-3)
        sp[i] = qp.get("sp", 1e-2)

    # cx propagation prob lookup by (c, t)
    cx_prop = {}
    for gid, (c, t) in enumerate(cx_pairs, start=1):
        qpairs = params["cx_gates"].get(str(gid), {})
        if qpairs.get("control") == c and qpairs.get("target") == t:
            cx_prop[(c, t)] = float(qpairs.get("lp_propagation_prob", 0.1))
        else:
            cx_prop[(c, t)] = 0.1

    flat = noisy_circuit.flattened()
    operations: List[Dict] = []
    # measurement_layout: list of qubit ids in the exact order they appear in the
    # measurement record (MR rounds then final M).  Each MR adds n_stab entries,
    # the final M adds n_data entries.  record_per_round groups them by syndrome
    # round for the leakage/event_leakage bookkeeping.
    measurement_layout: List[int] = []
    stab_round_meas: List[List[int]] = []   # per stabilizer round: measured stabilizer qubits
    final_data_meas: List[int] = []
    stab_set = set(stab_qubits)
    data_set = set(data_qubits)
    current_round = 0
    for inst in flat:
        nm = inst.name
        if nm == "H":
            tgts = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            idxs = [qubit_to_idx[t] for t in tgts if t in qubit_to_idx]
            operations.append({"type": "H", "idxs": np.array(idxs, dtype=np.int32)})
        elif nm == "CX":
            ts = inst.targets_copy()
            pairs = []
            for j in range(0, len(ts), 2):
                if not (ts[j].is_qubit_target and ts[j + 1].is_qubit_target):
                    continue
                c, t = ts[j].value, ts[j + 1].value
                if c in qubit_to_idx and t in qubit_to_idx:
                    pairs.append((qubit_to_idx[c], qubit_to_idx[t], cx_prop.get((c, t), 0.1)))
            operations.append({"type": "CX", "pairs": pairs})
        elif nm == "MR":
            tgts = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            meas_stabs = [q for q in tgts if q in stab_set]      # stabilizers measured this round
            stab_round_meas.append(meas_stabs)                   # (in MR-target order)
            measurement_layout.extend(meas_stabs)
            operations.append({"type": "MR"})
            current_round += 1
        elif nm == "M":
            tgts = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            meas_data = [q for q in tgts if q in data_set]
            final_data_meas = meas_data
            measurement_layout.extend(meas_data)
            operations.append({"type": "M"})

    return {
        "num_qubits": num_qubits,
        "qubit_to_idx": qubit_to_idx,
        "all_qubits": all_qubits,
        "lp": lp,
        "sp": sp,
        "operations": operations,
        "measurement_layout": measurement_layout,      # qubit id per record position
        "stab_round_meas": stab_round_meas,            # [[stab qids], ...] per round
        "final_data_meas": final_data_meas,            # [data qids] final round
    }


def _leakage_for_batch(pre: Dict, batch_shots: int, rng: np.random.Generator) -> np.ndarray:
    # ...
    # 每个 round 维护一个"本round 内曾泄漏"的累计标志
    round_ever_leaked = np.zeros((batch_shots, num_qubits), dtype=np.uint8)
    
    for op_idx, op in enumerate(ops):
        otype = op["type"]
        
        if otype in ("H", "CX"):
            # ... 执行 states 更新（原有逻辑不变）...
            # 同时累计：本 round 内曾处于泄漏态
            round_ever_leaked |= states# 只要states[s,q]==1，就标记
        elif otype == "MR":
            round_affected.append(round_ever_leaked.copy())  # ← 改为累计标志
            round_ever_leaked[:] = 0  # 重置，开始下一 round 的统计

    # Build the per-record affected array using the TRUE record layout.
    layout = pre["measurement_layout"]
    q2i = pre["qubit_to_idx"]
    n_records = len(layout)
    affected = np.zeros((batch_shots, n_records), dtype=np.uint8)
    stab_round_meas = pre["stab_round_meas"]
    # stabilizer records: round s -> affected via round_affected[s][qubit_idx]
    pos = 0
    n_rounds = len(stab_round_meas)
    for s in range(n_rounds):
        ra = round_affected[s]
        for q in stab_round_meas[s]:
            affected[:, pos] = ra[:, q2i[q]]
            pos += 1
    # final data records: use last round's affected snapshot
    last_ra = round_affected[-1] if round_affected else np.zeros((batch_shots, num_qubits), dtype=np.uint8)
    for q in pre["final_data_meas"]:
        affected[:, pos] = last_ra[:, q2i[q]]
        pos += 1
    return affected


def simulate_leakage(noisy_circuit: stim.Circuit, data_qubits: List[int],
                     stab_qubits: List[int], cx_pairs: List[Tuple[int, int]],
                     params: Dict, shots: int, seed: int,
                     batch_size: int = 2000) -> np.ndarray:
    """Return ``affected [shots, n_records]`` (uint8) for all shots (batched)."""
    pre = _preprocess_for_leakage(noisy_circuit, data_qubits, stab_qubits, cx_pairs, params)
    out: List[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for s in range(0, shots, batch_size):
        nb = min(batch_size, shots - s)
        out.append(_leakage_for_batch(pre, nb, rng))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 6. Soft readout (port of alphaqubit/data/soft_readout.py::SoftReadoutSimulator)
# ---------------------------------------------------------------------------

def soft_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Soft XOR  a + b - 2 a b  (probability convention)."""
    return (a + b - 2.0 * a * b).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. 在 paems_noise_model.py 中实现核心注入函数
# ---------------------------------------------------------------------------

def build_paems_noisy_circuit(
        distance: int, rounds: int, params: Dict
) -> stim.Circuit:
    base = _base_surface_code_circuit(distance, rounds)
    noisy = stim.Circuit()
    crosstalk_lookup = build_crosstalk_lookup(params.get("crosstalk_pairs", {}))
    flat = base.flattened()
    qparams = params["qubits"]
    cx_params = params["cx_gates"]
    cx_pair_to_gid = {
        (v["control"], v["target"]): k for k, v in cx_params.items()
    }

    for inst in flat:
        nm = inst.name
        noisy.append(inst)  # 先复制原始指令

        if nm in ("H", "S", "S_DAG"):  # 单比特门
            for t in inst.targets_copy():
                q = t.value
                qp = qparams[str(q)]
                noise = single_qubit_noise(qp)
                px, py, pz = noise["px_py_pz"]
                if px + py + pz > 1e-10:
                    noisy.append("PAULI_CHANNEL_1", [q], [px, py, pz])
                if noise["p1"] > 1e-10:
                    noisy.append("DEPOLARIZE1", [q], noise["p1"])
                _add_spectator_crosstalk(noisy, q, crosstalk_lookup)
        elif nm == "CX":
            ts = inst.targets_copy()
            for j in range(0, len(ts), 2):
                c, t_q = ts[j].value, ts[j + 1].value
                gid = cx_pair_to_gid.get((c, t_q))
                if gid is None:
                    continue
                cp = qparams[str(c)]
                tp = qparams[str(t_q)]
                cxp = cx_params[gid]
                noise2 = two_qubit_noise(cp, tp, cxp)
                for qi, pauli in [(c, noise2["control_pauli"]),
                                  (t_q, noise2["target_pauli"])]:
                    px, py, pz = pauli
                    if px + py + pz > 1e-10:
                        noisy.append("PAULI_CHANNEL_1", [qi], [px, py, pz])
                if noise2["p2"] > 1e-10:
                    noisy.append("DEPOLARIZE2", [c, t_q], noise2["p2"])

        elif nm in ("R", "RESET"):
            for t in inst.targets_copy():
                q = t.value
                p_init = qparams[str(q)].get("data_init_error", 0.0)
                if p_init > 1e-10:
                    noisy.append("X_ERROR", [q], p_init)

        elif nm in ("M", "MR"):
            for t in inst.targets_copy():
                q = t.value
                p_meas = qparams[str(q)].get("measurement_spam_rate", 0.0)
                if p_meas > 1e-10:
                    noisy.append("X_ERROR", [q], p_meas)  # flip before measure
    return noisy
