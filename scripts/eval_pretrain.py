"""Evaluate trained AlphaQubit model vs MWPM baseline."""
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, '.')
import torch, numpy as np
from alphaqubit.data import SurfaceCodeDataset
from alphaqubit.models import AlphaQubitDecoder, AlphaQubitDecoderConfig
from alphaqubit.evaluation.metrics import compute_error_rate, compute_fidelity, fit_ler, compute_lambda
from alphaqubit.experiments.baselines import MWPMBaseline

device = torch.device('cuda')
DISTANCE = 3
P = 0.005
N_SAMPLES = 20000
EVAL_ROUNDS = [3, 5, 7, 9, 11, 12, 15]

# Load model
print(f'Loading model: checkpoints/pretrain/d{DISTANCE}_base/best.pt')
temp_ds = SurfaceCodeDataset(distance=DISTANCE, rounds=12, p=P, num_samples=100)
model = AlphaQubitDecoderConfig.base(temp_ds.coord_system).to(device)
ckpt = torch.load(f'checkpoints/pretrain/d{DISTANCE}_base/best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Trained for {ckpt.get("step", "?")} steps, val_loss={ckpt.get("val_loss", "?"):.4f}')

# Evaluate AlphaQubit
print(f'\n{"="*60}')
print(f'AlphaQubit Evaluation (d={DISTANCE}, p={P})')
print(f'{"="*60}')

aq_errors, aq_fids = {}, {}
for r in EVAL_ROUNDS:
    ds = SurfaceCodeDataset(distance=DISTANCE, rounds=r, p=P, num_samples=N_SAMPLES, seed=r+42)
    batch = ds.get_batch(N_SAMPLES)
    m, e, l, el, fs = (batch[k].to(device) for k in ['measurement','event','leakage','event_leakage','final_soft'])
    lbl = batch['label'].cpu().numpy().flatten()
    with torch.no_grad():
        preds, _ = model.predict(m, e, l, el, fs)
    preds = preds.cpu().numpy()
    err = compute_error_rate(preds, lbl)
    fid = compute_fidelity(err)
    aq_errors[r] = err
    aq_fids[r] = fid

# Evaluate MWPM baseline
print(f'\nMWPM Baseline...')
mwpm = MWPMBaseline()
mwpm_errors, mwpm_fids = {}, {}
for r in EVAL_ROUNDS:
    result = mwpm.evaluate_single(DISTANCE, r, P, N_SAMPLES)
    mwpm_errors[r] = result.error_rate
    mwpm_fids[r] = compute_fidelity(result.error_rate)

# Print comparison table
print(f'\n{"Rounds":<8} {"AlphaQubit E":<15} {"AlphaQubit F":<15} {"MWPM E":<15} {"MWPM F":<15} {"Improvement":<12}')
print(f'{"-"*8} {"-"*15} {"-"*15} {"-"*15} {"-"*15} {"-"*12}')
for r in EVAL_ROUNDS:
    aq_e = aq_errors[r]
    aq_f = aq_fids[r]
    mw_e = mwpm_errors[r]
    mw_f = mwpm_fids[r]
    imp = (mw_e - aq_e) / mw_e * 100 if mw_e > 0 else 0
    print(f'{r:<8} {aq_e:<15.6f} {aq_f:<15.4f} {mw_e:<15.6f} {mw_f:<15.4f} {imp:+.1f}%')

# LER
print(f'\n{"="*60}')
print(f'LER (Logical Error Rate per Round)')
print(f'{"="*60}')

aq_ler, aq_r2, aq_logf0, _, aq_valid = fit_ler(EVAL_ROUNDS, [aq_fids[r] for r in EVAL_ROUNDS])
mw_ler, mw_r2, mw_logf0, _, mw_valid = fit_ler(EVAL_ROUNDS, [mwpm_fids[r] for r in EVAL_ROUNDS])

print(f'\n{"Method":<15} {"LER":<15} {"R^2":<10} {"Valid":<8}')
print(f'{"-"*15} {"-"*15} {"-"*10} {"-"*8}')
print(f'{"AlphaQubit":<15} {aq_ler:<15.8f} {aq_r2:<10.4f} {str(aq_valid):<8}')
print(f'{"MWPM":<15} {mw_ler:<15.8f} {mw_r2:<10.4f} {str(mw_valid):<8}')

if aq_ler > 0 and mw_ler > 0:
    lam = compute_lambda(mw_ler, aq_ler)
    print(f'\nLambda (MWPM LE / AlphaQubit LE): {lam:.4f}')
    print(f'LER ratio: {mw_ler/aq_ler:.2f}x (AlphaQubit {"better" if mw_ler/aq_ler > 1 else "worse"})')

print(f'\n{"="*60}')
print(f'Summary: AlphaQubit LER={aq_ler:.6f}, MWPM LER={mw_ler:.6f}')
print(f'{"="*60}')
