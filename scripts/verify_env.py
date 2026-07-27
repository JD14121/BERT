"""Verify quantum_env setup for AlphaQubit training."""
import sys
sys.path.insert(0, '.')
import torch
from alphaqubit.data import SurfaceCodeDataset
from alphaqubit.models import AlphaQubitDecoderConfig

print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
print()

# Data pipeline
print('[1] Data pipeline (d=3, rounds=25)...')
ds = SurfaceCodeDataset(distance=3, rounds=25, p=0.005, num_samples=100, seed=42)
metrics = ds.validate()
print(f'  Event density: {metrics["event_density"]:.4f}')
print(f'  Scatter/Gather: {metrics["scatter_gather_valid"]}')

# Model
print('[2] Model (base, ~500K params)...')
model = AlphaQubitDecoderConfig.base(ds.coord_system)
n = sum(p.numel() for p in model.parameters())
print(f'  Params: {n:,}')

# Forward pass
print('[3] Forward pass (batch=32)...')
batch = ds.get_batch(32)
model = model.cuda()
with torch.no_grad():
    logits = model(
        batch['measurement'].cuda(),
        batch['event'].cuda(),
        batch['leakage'].cuda(),
        batch['event_leakage'].cuda(),
        batch['final_soft'].cuda(),
    )
    print(f'  Logits: shape={logits.shape}, mean={logits.mean().item():.4f}')

vram_mb = torch.cuda.memory_allocated() / 1024**2
print(f'  VRAM used: {vram_mb:.0f} MB')
print()
print('=== quantum_env pipeline OK ===')
