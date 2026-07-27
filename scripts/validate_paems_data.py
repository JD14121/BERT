#!/usr/bin/env python3
"""QA/QC validator for PAEMS synthetic data — enforces
``synthetic_data_spec.md`` (v2.0) sections 3 (semantic relations),
4.1/4.3 (format & naming), 5 (consumer assumptions), 7 (consistency checks),
plus PAEMS-specific noise-characteristic checks.

Exit code 0 = all files pass; non-zero = one or more checks failed. A Markdown
report is written next to the data (default: ``../QA_REPORT.md``).

Run in the conda base env:
    D:/anaconda/python.exe validate_paems_data.py [--data-dir ..]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    sys.exit(f"torch import failed (need conda base env): {exc}")

try:
    import pymatching
except Exception as exc:  # pragma: no cover
    sys.exit(f"pymatching import failed (need conda base env): {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paems_noise_model as pnm  # noqa: E402

NAME_RE = re.compile(r"^(train|val|test|ler)_d(\d+)_r(\d+)_n(\d+)\.pt$")


def soft_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b - 2.0 * a * b).astype(np.float32)


def validate_file(pt_path: Path, params_cache: dict) -> Tuple[bool, Dict, List[str]]:
    """Run all spec checks on one ``.pt`` file. Returns (all_pass, metrics, failures)."""
    failures: List[str] = []
    m: Dict = {}

    # ---- naming (spec §4.3) ----
    mm = NAME_RE.match(pt_path.name)
    if not mm:
        failures.append(f"name: '{pt_path.name}' does not match "
                        "{{split}}_d{distance}_r{rounds}_n{N}.pt")
        return False, m, failures
    split, dist_name, rounds_name, n_name = mm.groups()
    dist_name, rounds_name, n_name = int(dist_name), int(rounds_name), int(n_name)

    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    required = ["measurement", "event", "final_soft", "label",
                "detection_events", "distance", "rounds", "p", "snr"]
    for k in required:
        if k not in data:
            failures.append(f"missing required field '{k}'")

    distance = int(data["distance"])
    rounds = int(data["rounds"])
    n_stab_expected = distance ** 2 - 1
    n_data_expected = distance ** 2

    meas = data["measurement"].numpy().astype(np.float32)
    event = data["event"].numpy().astype(np.float32)
    final_soft = data["final_soft"].numpy().astype(np.float32)
    label = data["label"].numpy().astype(np.float32)
    det = data["detection_events"].numpy().astype(np.float32)
    leak = data.get("leakage")
    leak = leak.numpy().astype(np.float32) if leak is not None else None

    N = meas.shape[0]
    T, n_stab = meas.shape[1], meas.shape[2]

    # ---- shape consistency (spec §5/§7.3/§11) ----
    m["N"], m["T"], m["n_stab"], m["n_data"] = N, T, n_stab, final_soft.shape[1]
    m["num_detectors"] = det.shape[1]
    if meas.shape != (N, T, n_stab):
        failures.append(f"measurement shape {meas.shape} != ({N},{T},{n_stab})")
    if event.shape != (N, T, n_stab):
        failures.append(f"event shape {event.shape} != ({N},{T},{n_stab})")
    if final_soft.shape[0] != N or final_soft.shape[1] != n_data_expected:
        failures.append(f"final_soft shape {final_soft.shape} != ({N},{n_data_expected})")
    if label.shape != (N,) and label.shape != (N, 1):
        failures.append(f"label shape {label.shape} != ({N},) [or ({N},1)]")
    if det.shape[0] != N:
        failures.append(f"detection_events shape[0] {det.shape[0]} != {N}")
    if distance != dist_name:
        failures.append(f"distance field {distance} != filename d{dist_name}")
    if rounds != rounds_name:
        failures.append(f"rounds field {rounds} != filename r{rounds_name}")
    if rounds != T:
        failures.append(f"rounds {rounds} != measurement T {T}")
    if n_stab != n_stab_expected:
        failures.append(f"n_stab {n_stab} != d^2-1 {n_stab_expected}")
    if n_data_expected != final_soft.shape[1]:
        failures.append(f"n_data {final_soft.shape[1]} != d^2 {n_data_expected}")
    # filename N must match sample count
    if N != n_name:
        failures.append(f"sample count {N} != filename n{n_name}")

    # ---- dtypes (spec §9) ----
    for k in ["measurement", "event", "final_soft", "label", "detection_events"]:
        if str(data[k].dtype) != "torch.float32":
            failures.append(f"{k} dtype {data[k].dtype} != float32")

    # ---- measurement <-> event consistency (spec §3.1/§7.1) ----
    re_event = np.zeros_like(meas)
    re_event[:, 0, :] = meas[:, 0, :]
    for t in range(1, T):
        re_event[:, t, :] = soft_xor(meas[:, t, :], meas[:, t - 1, :])
    diff = float(np.abs(re_event - event).max())
    m["meas_event_max_diff"] = diff
    if diff >= 1e-5:
        failures.append(f"measurement/event soft-XOR max diff {diff:.2e} >= 1e-5")

    # ---- measurement range ----
    m["meas_min"], m["meas_max"] = float(meas.min()), float(meas.max())
    if m["meas_min"] < -1e-6 or m["meas_max"] > 1 + 1e-6:
        failures.append(f"measurement out of [0,1]: [{m['meas_min']},{m['meas_max']}]")

    # ---- detection_events & label same-shot (spec §3.2/§7.2): MWPM sanity ----
    if distance not in params_cache:
        params_cache[distance] = pnm.generate_paems_params(distance, seed=distance * 7919 + 42)
    params = params_cache[distance]
    base = pnm._base_surface_code_circuit(distance, rounds)
    noisy = pnm.build_paems_noisy_circuit(base, params, rounds)
    dem = noisy.detector_error_model()
    mwpm = pymatching.Matching.from_detector_error_model(dem)
    if det.shape[1] != dem.num_detectors:
        failures.append(f"detection_events width {det.shape[1]} != DEM num_detectors {dem.num_detectors}")
    else:
        preds = mwpm.decode_batch((det > 0.5).astype(np.uint8) if det.dtype.kind == "f"
                                  else det.astype(np.uint8))
        preds = preds.flatten().astype(int)
        lab = label.flatten().astype(int) if label.ndim > 1 else label.astype(int)
        # PyMatching decode_batch returns predictions; the actual logical =
        # (predicted observable flip). Compare against label.
        same = preds == lab
        acc = float(same.mean()) if len(same) else 0.0
        m["mwpm_accuracy"] = acc
        # same-shot real data -> MWPM non-trivial & well above chance 0.5
        if acc <= 0.55:
            failures.append(f"MWPM accuracy {acc:.3f} <= 0.55 — detection_events "
                            "& label likely NOT from the same shot (spec §3.2)")

    # ---- noise characteristics ----
    det_bin = (det > 0.5)
    m["det_density"] = float(det_bin.mean())
    m["label_rate"] = float(label.mean())
    m["leak_fraction"] = float(leak.mean()) if leak is not None else 0.0
    # PAEMS hallmark: ADC asymmetry P_X = P_Y != P_Z verified on params
    px_list, pz_list = [], []
    for q, qp in params["qubits"].items():
        px, py, pz = pnm.calculate_px_py_pz(qp["t1"], qp["t2"], qp["sqg_length"])
        px_list.append(px); pz_list.append(pz)
    m["paems_px_mean"] = float(np.mean(px_list))
    m["paems_pz_mean"] = float(np.mean(pz_list))
    m["paems_asymmetric"] = bool(abs(float(np.mean(px_list)) - float(np.mean(pz_list))) > 0)
    # P_init != P_meas hallmark
    p_init = np.mean([qp["data_init_error"] for qp in params["qubits"].values()])
    p_meas = np.mean([qp["measurement_spam_rate"] for qp in params["qubits"].values()
                      if qp["is_ancilla"]])
    m["paems_p_init_mean"] = float(p_init)
    m["paems_p_meas_mean"] = float(p_meas)
    m["paems_init_ne_meas"] = bool(abs(p_init - p_meas) > 0)

    # detection density plausibility (QEC data, not all-zero / all-one)
    if m["det_density"] < 1e-5 or m["det_density"] > 0.5:
        failures.append(f"detection density {m['det_density']:.4f} outside plausible QEC range")

    # leakage consistency: event_leakage is soft-XOR of leakage (if present)
    ev_leak = data.get("event_leakage")
    if leak is not None and ev_leak is not None:
        ev_leak = ev_leak.numpy().astype(np.float32)
        re_el = np.zeros_like(leak)
        re_el[:, 0, :] = leak[:, 0, :]
        for t in range(1, T):
            re_el[:, t, :] = soft_xor(leak[:, t, :], leak[:, t - 1, :])
        el_diff = float(np.abs(re_el - ev_leak).max())
        m["leak_event_max_diff"] = el_diff
        if el_diff >= 1e-5:
            failures.append(f"leakage/event_leakage soft-XOR diff {el_diff:.2e} >= 1e-5")

    return (len(failures) == 0), m, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--report", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "QA_REPORT.md"))
    ap.add_argument("--glob", type=str, default="*.pt",
                    help="glob pattern for files to validate (default *.pt)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    files = sorted(p for p in data_dir.glob(args.glob) if NAME_RE.match(p.name))
    if not files:
        print(f"No spec-named .pt files found in {data_dir}")
        return 1

    params_cache: dict = {}
    results: List[Tuple[Path, bool, Dict, List[str]]] = []
    print(f"Validating {len(files)} file(s) in {data_dir}\n")
    for f in files:
        ok, m, fails = validate_file(f, params_cache)
        results.append((f, ok, m, fails))
        status = "PASS" if ok else "FAIL"
        # reconstruct distance from filename (already validated by NAME_RE)
        fdist = int(f.name.split('_d')[1].split('_')[0])
        print(f"[{status}] {f.name}  "
              f"N={m.get('N')} d={fdist} r={m.get('T')} "
              f"mwpm={m.get('mwpm_accuracy','?')} det={m.get('det_density','?')} "
              f"lab={m.get('label_rate','?')} leak={m.get('leak_fraction','?')}")
        for fail in fails:
            print(f"        - {fail}")

    # ---- write markdown report ----
    report = Path(args.report).resolve()
    lines = ["# PAEMS Data QA Report\n",
             f"Data dir: `{data_dir}`\n",
             f"Files validated: {len(files)} | "
             f"PASS: {sum(1 for _, ok, _, _ in results if ok)} | "
             f"FAIL: {sum(1 for _, ok, _, _ in results if not ok)}\n",
             "\n## Per-file summary\n",
             "| File | Pass | N | T | n_stab | n_data | num_det | MWPM acc | det density | label rate | leak frac | meas-event diff |",
             "|------|------|---|---|--------|--------|---------|----------|-------------|------------|-----------|------------------|"]
    for f, ok, mm, fails in results:
        lines.append(
            f"| {f.name} | {'✅' if ok else '❌'} | {mm.get('N')} | {mm.get('T')} | "
            f"{mm.get('n_stab')} | {mm.get('n_data')} | {mm.get('num_detectors')} | "
            f"{mm.get('mwpm_accuracy','—')} | {mm.get('det_density','—')} | "
            f"{mm.get('label_rate','—')} | {mm.get('leak_fraction','—')} | "
            f"{mm.get('meas_event_max_diff','—')} |"
        )
    lines.append("\n## PAEMS model fidelity (shared per-distance params)\n")
    lines.append("| distance | P_X (mean) | P_Z (mean) | asymmetric P_X!=P_Z | P_init | P_meas | init!=meas |")
    lines.append("|----------|------------|------------|---------------------|--------|--------|------------|")
    seen = set()
    for f, ok, mm, fails in results:
        d = int(f.name.split('_d')[1].split('_')[0])
        if d in seen:
            continue
        seen.add(d)
        lines.append(f"| d={d} | {mm.get('paems_px_mean','—')} | {mm.get('paems_pz_mean','—')} | "
                     f"{'yes' if mm.get('paems_asymmetric') else 'no'} | "
                     f"{mm.get('paems_p_init_mean','—')} | {mm.get('paems_p_meas_mean','—')} | "
                     f"{'yes' if mm.get('paems_init_ne_meas') else 'no'} |")
    if any(not ok for _, ok, _, _ in results):
        lines.append("\n## Failures\n")
        for f, ok, mm, fails in results:
            for fail in fails:
                lines.append(f"- **{f.name}**: {fail}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report}")
    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
