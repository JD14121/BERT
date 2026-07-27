#!/usr/bin/env python3
"""fix_patch2.py - 修复未生效的 patch"""
from pathlib import Path

# === 1. pretrain_trainer.py: 梯度累积 ===
f = "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"
s = open(f).read()

old_amp = """            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()"""

new_amp = """            accum = getattr(self.config, 'gradient_accumulation_steps', 1)
            self.scaler.scale(loss / accum).backward()

            if (self.global_step + 1) % accum == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics['grad_norm'] = grad_norm.item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                metrics['grad_norm'] = 0.0"""

if old_amp in s:
    s = s.replace(old_amp, new_amp, 1)
    print("[OK] pretrain_trainer AMP: 梯度累积")
else:
    print("[SKIP] pretrain_trainer AMP 锚点未找到")

old_noamp = """            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            self.optimizer.step()"""

new_noamp = """            accum = getattr(self.config, 'gradient_accumulation_steps', 1)
            (loss / accum).backward()

            if (self.global_step + 1) % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics['grad_norm'] = grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                metrics['grad_norm'] = 0.0"""

if old_noamp in s:
    s = s.replace(old_noamp, new_noamp, 1)
    print("[OK] pretrain_trainer 非AMP: 梯度累积")
else:
    print("[SKIP] pretrain_trainer 非AMP 锚点未找到")

open(f, 'w').write(s)

# === 2. bert_pretrain.py: TF32 + grad-accum ===
f2 = "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"
s2 = open(f2).read()

if 'set_float32_matmul_precision' not in s2:
    lines = s2.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == 'import torch':
            lines.insert(i + 1, 'torch.backends.cudnn.benchmark = True')
            lines.insert(i + 2, 'torch.set_float32_matmul_precision("high")')
            s2 = '\n'.join(lines)
            print("[OK] bert_pretrain.py: +TF32 +cudnn")
            break
    else:
        print("[SKIP] bert_pretrain.py: import torch 未找到")
else:
    print("[SKIP] bert_pretrain.py: TF32 已有")

if '--grad-accum' not in s2:
    s2 = s2.replace(
        "    ap.add_argument('--device', default='cuda')",
        "    ap.add_argument('--device', default='cuda')\n    ap.add_argument('--grad-accum', type=int, default=1, help='梯度累积步数')",
        1
    )
    s2 = s2.replace(
        'save_interval=500)',
        'save_interval=500, gradient_accumulation_steps=args.grad_accum)',
        1
    )
    print("[OK] bert_pretrain.py: +--grad-accum")
else:
    print("[SKIP] bert_pretrain.py: --grad-accum 已有")

open(f2, 'w').write(s2)

# === 3. run_experiment.py: finetune() + gradient_accumulation_steps ===
f3 = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s3 = open(f3).read()

old_sig = "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0, min_steps=0, patience=10000):"
new_sig = "def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0, min_steps=0, patience=10000, gradient_accumulation_steps=1):"
if old_sig in s3:
    s3 = s3.replace(old_sig, new_sig, 1)
    print("[OK] run_experiment finetune(): +gradient_accumulation_steps")
else:
    print("[SKIP] finetune() 签名未找到")

# TrainingConfig 创建处
for old_cfg in [
    'early_stopping_patience=patience, focal_gamma=focal_gamma, min_steps=min_steps)',
    'early_stopping_patience=patience, min_steps=min_steps)',
]:
    new_cfg = old_cfg[:-1] + ', gradient_accumulation_steps=gradient_accumulation_steps)'
    if old_cfg in s3:
        s3 = s3.replace(old_cfg, new_cfg, 1)
        print("[OK] run_experiment TrainingConfig: +gradient_accumulation_steps")
        break
else:
    print("[SKIP] TrainingConfig 创建处未找到")

open(f3, 'w').write(s3)

# === 4. 验证 ===
print("\n=== 最终验证 ===")
for f, name in [
    ("/root/beat_mwpm/alphaqubit/training/trainer.py", "trainer.py"),
    ("/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py", "pretrain_trainer.py"),
    ("/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py", "bert_pretrain.py"),
    ("/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py", "run_experiment.py"),
]:
    s = open(f).read()
    tf32 = "set_float32_matmul_precision" in s
    cudnn = "cudnn.benchmark" in s
    ga = "gradient_accumulation_steps" in s
    compile_ok = "use_compile: bool = True" in s or "use_compile=True" in s
    accum_loop = "accum = getattr" in s
    print(f"  {name}: TF32={'Y' if tf32 else 'N'} cudnn={'Y' if cudnn else 'N'} grad_accum={'Y' if ga else 'N'} compile={'Y' if compile_ok else 'N'} accum_loop={'Y' if accum_loop else 'N'}")

import re
m = re.search(r'def finetune\(.*?\):', s3)
print(f"\n  finetune(): {m.group(0)[:120] if m else 'NOT FOUND'}")
