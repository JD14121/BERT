#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_acceleration.py
为云端代码添加 55M 训练加速优化：
1. TF32: torch.set_float32_matmul_precision('high') -- V100 Tensor Core, matmul ~3× 加速
2. cudnn.benchmark: 固定形状自动调优
3. torch.compile: use_compile=True (PyTorch 2.0+)
4. 梯度累积: gradient_accumulation_steps 配置 + 训练循环 patch (bs64 -> 有效 bs256)

在云端执行: python3 patch_acceleration.py
"""
import shutil
from pathlib import Path

TRAINER = Path("/root/beat_mwpm/alphaqubit/training/trainer.py")
PRETRAINER = Path("/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py")
BERT_PRETRAIN = Path("/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py")
RUN_EXP = Path("/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py")

def patch_file(path, old, new, desc=""):
    s = open(path).read()
    if old not in s:
        print(f"  [SKIP] {desc}: 锚点未找到")
        return False
    if new in s:
        print(f"  [EXISTS] {desc}: 已有")
        return True
    s = s.replace(old, new, 1)
    open(path, 'w').write(s)
    print(f"  [OK] {desc}")
    return True

# ========== 1. trainer.py ==========
print("=== Patching trainer.py ===")
shutil.copy(str(TRAINER), str(TRAINER) + '.bak_accel')

# 1a. 添加 gradient_accumulation_steps 到 TrainingConfig
patch_file(TRAINER,
    "    use_amp: bool = True               # 使用混合精度训练（大幅加速）\n    use_compile: bool = False          # 使用torch.compile（PyTorch 2.0+）",
    "    use_amp: bool = True               # 使用混合精度训练（大幅加速）\n    use_compile: bool = True           # 使用torch.compile（PyTorch 2.0+）\n    gradient_accumulation_steps: int = 1  # 梯度累积步数（bs64×4=有效bs256）",
    "TrainingConfig: +gradient_accumulation_steps +use_compile=True")

# 1b. patch _train_step AMP 路径（梯度累积）
OLD_AMP = """            # 反向传播（使用scaler）
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()

            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()"""

NEW_AMP = """            # 反向传播（使用scaler + 梯度累积）
            accum = getattr(self.config, 'gradient_accumulation_steps', 1)
            self.scaler.scale(loss / accum).backward()

            if (self.global_step + 1) % accum == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics['grad_norm'] = grad_norm.item()
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                metrics['grad_norm'] = 0.0"""

patch_file(TRAINER, OLD_AMP, NEW_AMP, "trainer _train_step AMP: 梯度累积")

# 1c. patch _train_step 非 AMP 路径
OLD_NOAMP = """            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            metrics['grad_norm'] = grad_norm.item()"""

NEW_NOAMP = """            # 反向传播（梯度累积）
            accum = getattr(self.config, 'gradient_accumulation_steps', 1)
            (loss / accum).backward()

            if (self.global_step + 1) % accum == 0:
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics['grad_norm'] = grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                metrics['grad_norm'] = 0.0"""

patch_file(TRAINER, OLD_NOAMP, NEW_NOAMP, "trainer _train_step 非AMP: 梯度累积")

# 1d. 移除非 AMP 路径末尾的 optimizer.step()（已移入累积条件）
patch_file(TRAINER,
    """            metrics['grad_norm'] = grad_norm.item()
            self.optimizer.step()
        else:
            metrics['grad_norm'] = 0.0""",
    """            metrics['grad_norm'] = grad_norm.item()
        else:
            metrics['grad_norm'] = 0.0""",
    "trainer: 移除冗余 optimizer.step()")

# ========== 2. pretrain_trainer.py ==========
print("\n=== Patching pretrain_trainer.py ===")
shutil.copy(str(PRETRAINER), str(PRETRAINER) + '.bak_accel')

# 2a. 添加 gradient_accumulation_steps
patch_file(PRETRAINER,
    "    use_amp: bool = True\n    use_compile: bool = False",
    "    use_amp: bool = True\n    use_compile: bool = True\n    gradient_accumulation_steps: int = 1",
    "PretrainConfig: +gradient_accumulation_steps +use_compile=True")

# 2b. patch _train_step AMP 路径
patch_file(PRETRAINER, OLD_AMP, NEW_AMP, "pretrainer _train_step AMP: 梯度累积")

# 2c. patch _train_step 非 AMP 路径
patch_file(PRETRAINER, OLD_NOAMP, NEW_NOAMP, "pretrainer _train_step 非AMP: 梯度累积")

# 2d. 移除非 AMP 路径末尾的 optimizer.step()
patch_file(PRETRAINER,
    """            metrics['grad_norm'] = grad_norm.item()
            self.optimizer.step()
        else:
            metrics['grad_norm'] = 0.0""",
    """            metrics['grad_norm'] = grad_norm.item()
        else:
            metrics['grad_norm'] = 0.0""",
    "pretrainer: 移除冗余 optimizer.step()")

# ========== 3. bert_pretrain.py + run_experiment.py: TF32 + cudnn ==========
print("\n=== Patching bert_pretrain.py + run_experiment.py: TF32 + cudnn ===")

TF32_BLOCK = """
# ===== 加速优化 =====
torch.backends.cudnn.benchmark = True       # 固定形状自动调优
torch.set_float32_matmul_precision('high')  # TF32 (V100 Tensor Core, matmul ~3×)
"""

for f, name in [(BERT_PRETRAIN, "bert_pretrain.py"), (RUN_EXP, "run_experiment.py")]:
    s = open(f).read()
    if 'set_float32_matmul_precision' in s:
        print(f"  [EXISTS] {name}: TF32 已有")
    else:
        # 在第一个 import torch 后插入
        s = s.replace("import torch", "import torch" + TF32_BLOCK, 1)
        open(f, 'w').write(s)
        print(f"  [OK] {name}: +TF32 +cudnn.benchmark")

# ========== 4. 验证 ==========
print("\n=== 验证 ===")
for f, name in [(TRAINER, "trainer.py"), (PRETRAINER, "pretrain_trainer.py")]:
    s = open(f).read()
    checks = [
        ("gradient_accumulation_steps", "gradient_accumulation_steps" in s),
        ("use_compile=True", "use_compile: bool = True" in s),
        ("accum 梯度累积", "accum = getattr" in s),
        ("zero_grad 移入条件", "self.optimizer.zero_grad()" in s and s.count("self.optimizer.zero_grad()") <= s.count("if (self.global_step + 1) % accum == 0:")),
    ]
    print(f"  {name}:")
    for label, ok in checks:
        print(f"    [{'✓' if ok else '✗'}] {label}")

for f, name in [(BERT_PRETRAIN, "bert_pretrain.py"), (RUN_EXP, "run_experiment.py")]:
    s = open(f).read()
    print(f"  {name}: [{'✓' if 'set_float32_matmul_precision' in s else '✗'}] TF32, [{'✓' if 'cudnn.benchmark' in s else '✗'}] cudnn")

print("\n=== Patch 完成 ===")
print("备份: trainer.py.bak_accel, pretrain_trainer.py.bak_accel")
print("\n使用方式: 在 PretrainConfig/TrainingConfig 中设 gradient_accumulation_steps=4")
print("  (bs64 × 4 累积 = 有效 bs256)")
