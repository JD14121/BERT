#!/usr/bin/env python3
"""fix_pt.py - 修复 pretrain_trainer.py"""
import shutil, py_compile

f = "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"

# 从备份恢复
try:
    shutil.copy(f + ".bak_accel", f)
    print("[OK] 从 .bak_accel 恢复")
except:
    print("[SKIP] 无备份")

s = open(f).read()

# 移除已有的 TF32 行
lines = s.split("\n")
new_lines = [l for l in lines if "cudnn.benchmark" not in l and "set_float32_matmul_precision" not in l]

# 找第一个顶层 import torch
insert_after = -1
for i, line in enumerate(new_lines):
    if line == "import torch" or line.startswith("import torch "):
        insert_after = i
        break

if insert_after >= 0:
    new_lines.insert(insert_after + 1, "torch.backends.cudnn.benchmark = True")
    new_lines.insert(insert_after + 2, "torch.set_float32_matmul_precision('high')")
    print(f"[OK] TF32 插入到 line {insert_after + 1}")
else:
    for i, line in enumerate(new_lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_after = i
            break
    if insert_after >= 0:
        new_lines.insert(insert_after + 1, "import torch")
        new_lines.insert(insert_after + 2, "torch.backends.cudnn.benchmark = True")
        new_lines.insert(insert_after + 3, "torch.set_float32_matmul_precision('high')")
        print(f"[OK] TF32 + import torch 插入到 line {insert_after + 1}")

open(f, "w").write("\n".join(new_lines))

# 重新应用梯度累积 patch
s = open(f).read()
if "accum = getattr" not in s:
    # config
    if "gradient_accumulation_steps" not in s:
        s = s.replace(
            "use_compile: bool = False",
            "use_compile: bool = True\n    gradient_accumulation_steps: int = 1",
            1
        )
        print("[OK] +gradient_accumulation_steps config")

    # AMP path - use single-line markers to avoid triple-quote issues
    s = s.replace(
        "            self.optimizer.zero_grad()\n            self.scaler.scale(loss).backward()",
        "            accum = getattr(self.config, 'gradient_accumulation_steps', 1)\n            self.scaler.scale(loss / accum).backward()",
        1
    )

    # Wrap the unscale/clip/step/update in if block
    s = s.replace(
        "            self.scaler.unscale_(self.optimizer)\n            grad_norm = torch.nn.utils.clip_grad_norm_(",
        "            if (self.global_step + 1) % accum == 0:\n                self.scaler.unscale_(self.optimizer)\n                grad_norm = torch.nn.utils.clip_grad_norm_(",
        1
    )
    # Indent the remaining lines and add zero_grad + else
    s = s.replace(
        "            metrics['grad_norm'] = grad_norm.item()\n\n            self.scaler.step(self.optimizer)\n            self.scaler.update()",
        "                metrics['grad_norm'] = grad_norm.item()\n                self.scaler.step(self.optimizer)\n                self.scaler.update()\n                self.optimizer.zero_grad()\n            else:\n                metrics['grad_norm'] = 0.0",
        1
    )
    print("[OK] AMP 梯度累积")

    open(f, "w").write(s)

# 语法检查
try:
    py_compile.compile(f, doraise=True)
    print("[OK] 语法正确")
except py_compile.PyCompileError as e:
    print(f"[ERR] 语法错误: {str(e)[:200]}")

# 验证
s = open(f).read()
for k, v in [("TF32", "set_float32_matmul_precision" in s),
             ("cudnn", "cudnn.benchmark" in s),
             ("grad_accum", "gradient_accumulation_steps" in s),
             ("accum_loop", "accum = getattr" in s),
             ("compile", "use_compile: bool = True" in s)]:
    print(f"  {k}={'Y' if v else 'N'}")
