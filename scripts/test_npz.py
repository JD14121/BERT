"""Verify NPZDataset works correctly."""
import sys; sys.path.insert(0, '.')
from alphaqubit.data import NPZDataset

ds = NPZDataset('data/test_d3_r12_n10000.npz')
print(f'Samples: {len(ds)}')
print(f'Distance: {ds.distance}, Rounds: {ds.rounds}, p: {ds.p}')
s = ds[0]
print(f'measurement: {s["measurement"].shape}')
print(f'event: {s["event"].shape}')
print(f'leakage: {s["leakage"].shape} (all zeros: {(s["leakage"]==0).all().item()})')
print(f'stab_pos_idx: {s["stab_pos_idx"].shape}')
metrics = ds.validate()
print(f'Event density: {metrics["event_density"]:.4f}')
print(f'Label flip rate: {metrics["label_flip_rate"]:.4f}')
print(f'Scatter/Gather: {metrics["scatter_gather_valid"]}')
print('=== NPZDataset OK ===')
