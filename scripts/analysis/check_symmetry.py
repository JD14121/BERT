"""Check whether 90-degree rotation of the d7 XZZX code preserves the logical Z label.

Key question: does the logical Z observable (NW diagonal) map to itself (or its
complement) under 90-degree rotation, or does it map to a DIFFERENT logical
operator (logical X)?

If it maps to logical X, then 90-degree rotation CANNOT be used for data
augmentation of Z-memory data (we don't have the logical X label).
"""
import stim
from pathlib import Path
import numpy as np

circ_path = Path(r'D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/Google-data/google_105Q_surface_code_d3_d5_d7/d7_at_q6_7/Z/r01/circuit_ideal.stim')
circ = stim.Circuit.from_file(str(circ_path))

# Get qubit coords
qubit_coords = {}
for inst in circ.flattened():
    if inst.name == 'QUBIT_COORDS':
        a = inst.gate_args_copy()
        for t in inst.targets_copy():
            if t.is_qubit_target:
                qubit_coords[t.value] = (a[0], a[1])

# Get all measurement targets in order
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

# Map positions to qubit IDs
pos_to_qid = {}
for q in all_qs:
    x, y = qubit_coords[q]
    pos = (int(y - min_y), int(x - min_x))
    pos_to_qid[pos] = q

# Get original logical Z observable qubits
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

# Define C4 rotations
def rot90(p, n=13): return (p[1], n - 1 - p[0])
def rot180(p, n=13): return (n - 1 - p[0], n - 1 - p[1])
def rot270(p, n=13): return (n - 1 - p[1], p[0])

# Compute rotated observable positions
transforms = {
    'identity': lambda p: p,
    'rot90': rot90,
    'rot180': rot180,
    'rot270': rot270,
}

# Get all detectors
detectors = []
for inst in circ.flattened():
    if inst.name == 'DETECTOR':
        targets = [t.value for t in inst.targets_copy() if t.is_measurement_record_target]
        abs_indices = [len(all_meas_targets) + t for t in targets]
        detectors.append(abs_indices)

print(f'Total detectors: {len(detectors)}')

# For each rotation, compute the rotated observable and check commutation with detectors
print('\n=== Commutation test: does rotated observable commute with all detectors? ===')
print('(If commutes with all -> it is a valid logical operator or stabilizer)')
print('(If anticommutes with some -> it is NOT a valid Z-basis logical)')
print()

for name, tf in transforms.items():
    rotated_pos = set(tf(p) for p in obs_pos)
    rotated_qids = [pos_to_qid[p] for p in rotated_pos if p in pos_to_qid]

    # Find measurement record indices for these qubits (in the final data measurement round)
    # For r=1: 48 stab meas (round 1) + 49 data meas (final) = 97 total
    # Data measurements are the last 49
    rotated_indices = []
    for q in rotated_qids:
        # Find q in all_meas_targets (should be in the data section, i.e., last 49)
        for i in range(len(all_meas_targets) - 49, len(all_meas_targets)):
            if all_meas_targets[i] == q:
                rotated_indices.append(i)
                break

    rotated_set = set(rotated_indices)
    commute = 0
    anticommute = 0
    for det in detectors:
        det_set = set(det)
        overlap = len(rotated_set & det_set)
        if overlap % 2 == 0:
            commute += 1
        else:
            anticommute += 1

    status = 'COMMUTES (valid)' if anticommute == 0 else f'ANTICOMMUTES with {anticommute} (INVALID as Z-logical)'
    print(f'  {name}: {len(rotated_indices)} qubits, {status}')

# Additional: check if the rotated observable is equivalent to the original
# (i.e., differs by a set of stabilizers = detectors)
print('\n=== Equivalence test: is rotated observable equivalent to original? ===')
print('(Two observables are equivalent iff their XOR is in the detector row space)')

# Build detector matrix over GF(2)
n_meas = len(all_meas_targets)
det_matrix = np.zeros((len(detectors), n_meas), dtype=np.uint8)
for i, det in enumerate(detectors):
    for idx in det:
        det_matrix[i, idx] = 1

# Get original observable vector
orig_vec = np.zeros(n_meas, dtype=np.uint8)
for idx in orig_obs_indices if 'orig_obs_indices' in dir() else []:
    orig_vec[idx] = 1
# Recompute orig indices
orig_obs_indices = []
for inst in circ.flattened():
    if inst.name == 'OBSERVABLE_INCLUDE':
        targets = [t.value for t in inst.targets_copy()]
        orig_obs_indices = [len(all_meas_targets) + t for t in targets]
orig_vec = np.zeros(n_meas, dtype=np.uint8)
for idx in orig_obs_indices:
    orig_vec[idx] = 1

# Gaussian elimination over GF(2) to check if diff is in row space
def gf2_rank(matrix):
    m = matrix.copy()
    rows, cols = m.shape
    rank = 0
    for c in range(cols):
        # Find pivot
        pivot = None
        for r in range(rank, rows):
            if m[r, c] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        m[[rank, pivot]] = m[[pivot, rank]]
        # Eliminate
        for r in range(rows):
            if r != rank and m[r, c] == 1:
                m[r] = (m[r] + m[rank]) % 2
        rank += 1
    return rank, m

def in_row_space(matrix, vec):
    """Check if vec is in the row space of matrix over GF(2)."""
    augmented = np.vstack([matrix, vec[np.newaxis, :]])
    rank_orig, _ = gf2_rank(matrix)
    rank_aug, _ = gf2_rank(augmented)
    return rank_aug == rank_orig

for name, tf in transforms.items():
    rotated_pos = set(tf(p) for p in obs_pos)
    rotated_qids = [pos_to_qid[p] for p in rotated_pos if p in pos_to_qid]
    rotated_indices = []
    for q in rotated_qids:
        for i in range(len(all_meas_targets) - 49, len(all_meas_targets)):
            if all_meas_targets[i] == q:
                rotated_indices.append(i)
                break
    rot_vec = np.zeros(n_meas, dtype=np.uint8)
    for idx in rotated_indices:
        rot_vec[idx] = 1

    diff = (orig_vec + rot_vec) % 2
    is_equivalent = in_row_space(det_matrix, diff)
    print(f'  {name}: equivalent to original = {is_equivalent}')

# Also check if any of the rotated observables is equivalent to logical X
# Logical X is not defined in the circuit (only logical Z is).
# But we can check: is the rotated observable a NEW logical operator (not equivalent to Z)?
# If it commutes with all detectors AND is not equivalent to Z, then it's logical X.
print('\n=== Conclusion ===')
print('If rot90/rot270 commute with all detectors but are NOT equivalent to original,')
print('then they represent logical X (a different logical operator).')
print('This means 90/270-degree rotation CANNOT be used for Z-memory augmentation.')
