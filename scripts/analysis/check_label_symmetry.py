"""Empirically verify label preservation under C4 rotations.

Sample noiseless shots from the d7 XZZX circuit, then for each rotation,
check whether the rotated observable gives the same label as the original.

If the labels differ, that rotation CANNOT be used for Z-memory augmentation.
"""
import stim
from pathlib import Path
import numpy as np

circ_path = Path(r'D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/Google-data/google_105Q_surface_code_d3_d5_d7/d7_at_q6_7/Z/r01/circuit_ideal.stim')
circ = stim.Circuit.from_file(str(circ_path))

# Get qubit coords and measurement targets
qubit_coords = {}
for inst in circ.flattened():
    if inst.name == 'QUBIT_COORDS':
        a = inst.gate_args_copy()
        for t in inst.targets_copy():
            if t.is_qubit_target:
                qubit_coords[t.value] = (a[0], a[1])

all_meas_targets = []
for inst in circ.flattened():
    if inst.name in ('M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'):
        for t in inst.targets_copy():
            if t.is_qubit_target:
                all_meas_targets.append(t.value)

all_qs = [q for q in all_meas_targets if q in qubit_coords]
xs = [qubit_coords[q][0] for q in all_qs]
ys = [qubit_coords[q][1] for q in all_qs]
min_x, min_y = min(xs), min(ys)

pos_to_qid = {}
for q in all_qs:
    x, y = qubit_coords[q]
    pos = (int(y - min_y), int(x - min_x))
    pos_to_qid[pos] = q

# Get original observable
obs_qubits = []
for inst in circ.flattened():
    if inst.name == 'OBSERVABLE_INCLUDE':
        targets = [t.value for t in inst.targets_copy()]
        abs_indices = [len(all_meas_targets) + t for t in targets]
        obs_qubits = [all_meas_targets[i] for i in abs_indices]

obs_pos = set()
for q in obs_qubits:
    x, y = qubit_coords[q]
    obs_pos.add((int(y - min_y), int(x - min_x)))

print(f'Original logical Z (NW diagonal) positions: {sorted(obs_pos)}')

# Define rotations
def rot90(p, n=13): return (p[1], n - 1 - p[0])
def rot180(p, n=13): return (n - 1 - p[0], n - 1 - p[1])
def rot270(p, n=13): return (n - 1 - p[1], p[0])

transforms = {
    'identity': lambda p: p,
    'rot90': rot90,
    'rot180': rot180,
    'rot270': rot270,
}

# For each rotation, find the measurement indices of the rotated observable
print('\n=== Rotated observable measurement indices ===')
rotated_indices = {}
for name, tf in transforms.items():
    rotated_pos = set(tf(p) for p in obs_pos)
    rotated_qids = [pos_to_qid[p] for p in rotated_pos if p in pos_to_qid]
    indices = []
    for q in rotated_qids:
        for i in range(len(all_meas_targets)):
            if all_meas_targets[i] == q:
                indices.append(i)
                break
    rotated_indices[name] = sorted(indices)
    print(f'  {name}: qubits at {sorted(rotated_pos)}')
    print(f'    qubit IDs: {rotated_qids}')
    print(f'    meas indices: {rotated_indices[name]}')

# Sample noiseless shots
sampler = circ.compile_sampler()
n_shots = 10000
print(f'\nSampling {n_shots} noiseless shots...')
shots = sampler.sample(n_shots)
# stim returns numpy array of bool (True=1)
shots = np.array(shots, dtype=np.uint8)
print(f'Shots shape: {shots.shape}')

# Compute original label and rotated labels for each shot
# Label = XOR of measurement records at observable indices
# In stim, shots are bitstrings where True=1 (measurement outcome 1)
labels = {}
for name, indices in rotated_indices.items():
    # XOR of bits at these indices
    label_vals = shots[:, indices].sum(axis=1) % 2
    labels[name] = label_vals
    print(f'  {name} label distribution: 0={np.sum(label_vals == 0)}, 1={np.sum(label_vals == 1)}')

# Check correlation between original and rotated labels
print('\n=== Label correlation (original vs rotated) ===')
orig_labels = labels['identity']
for name in ['rot90', 'rot180', 'rot270']:
    rot_labels = labels[name]
    same = np.sum(orig_labels == rot_labels)
    diff = np.sum(orig_labels != rot_labels)
    total = len(orig_labels)
    print(f'  {name}: same={same} ({same/total*100:.1f}%), diff={diff} ({diff/total*100:.1f}%)')
    # Check if it's a deterministic flip (always opposite)
    if diff == total:
        print(f'    -> DETERMINISTIC FLIP (label always inverts)')
    elif same == total:
        print(f'    -> LABEL PRESERVED (always same)')
    else:
        print(f'    -> NO DETERMINISTIC RELATIONSHIP (label is INDEPENDENT)')

# Additional: check XOR of original and rotated
print('\n=== XOR(original, rotated) distribution ===')
for name in ['rot90', 'rot180', 'rot270']:
    xor_vals = (orig_labels + labels[name]) % 2
    n_zero = np.sum(xor_vals == 0)
    n_one = np.sum(xor_vals == 1)
    print(f'  {name}: XOR=0 (same)={n_zero}, XOR=1 (diff)={n_one}')
    if n_one == 0:
        print(f'    -> Labels are IDENTICAL (rotated observable = original logical Z)')
    elif n_zero == 0:
        print(f'    -> Labels are ALWAYS OPPOSITE (rotated observable = logical Z complement)')
    else:
        print(f'    -> Labels are INDEPENDENT (rotated observable = different logical, e.g. logical X)')

print('\n=== CONCLUSION ===')
print('If labels are INDEPENDENT for a rotation, that rotation CANNOT be used')
print('for Z-memory data augmentation (the rotated syndrome would have a')
print('random label relative to the original, so training on it would be')
print('training on mislabeled data).')
