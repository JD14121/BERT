#!/usr/bin/env python3
"""fix_bp.py - 修复 bert_pretrain.py 的 TF32 插入位置"""
import re

f = "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py"

# 从备份恢复
import shutil
bak = f + ".bak_accel"
try:
    shutil.copy(bak, f)
    print("[OK] 从 .bak_accel 恢复")
except:
    print("[SKIP] 无备份，直接修复当前文件")

s = open(f).read()

# 移除已有的 TF32 行（如果有）
s = s.replace("torch.backends.cudnn.benchmark = True\n", "")
s = s.replace("torch.set_float32_matmul_precision('high')\n", "")

# 找第一个非缩进的 import torch（顶层）
lines = s.split("\n")
insert_after = -1
for i, line in enumerate(lines):
    if line == "import torch" or line.startswith("import torch "):
        insert_after = i
        break

if insert_after >= 0:
    lines.insert(insert_after + 1, "torch.backends.cudnn.benchmark = True")
    lines.insert(insert_after + 2, "torch.set_float32_matmul_precision('high')")
    open(f, "w").write("\n".join(lines))
    print(f"[OK] TF32 插入到 line {insert_after + 1} (import torch 之后)")
else:
    # 没有 import torch，在文件开头加
    s = "import torch\ntorch.backends.cudnn.benchmark = True\ntorch.set_float32_matmul_precision('high')\n" + s
    open(f, "w").write(s)
    print("[OK] TF32 前置（无 import torch 找到）")

# 同样修复 trainer.py 和 pretrain_trainer.py（如果有同样问题）
for f2 in ["/root/beat_mwpm/alphaqubit/training/trainer.py",
           "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    s2 = open(f2).read()
    # 检查是否有缩进问题
    lines2 = s2.split("\n")
    for i, line in enumerate(lines2):
        if "torch.backends.cudnn.benchmark" in line and line.startswith(" "):
            # 缩进了，移到正确位置
            print(f"[WARN] {f2.split('/')[-1]} line {i+1}: TF32 在缩进内，可能有问题")
    # 如果 TF32 不在文件里，添加
    if "set_float32_matmul_precision" not in s2:
        # 找 import torch
        for i, line in enumerate(lines2):
            if line == "import torch" or line.startswith("import torch "):
                lines2.insert(i + 1, "torch.backends.cudnn.benchmark = True")
                lines2.insert(i + 2, "torch.set_float32_matmul_precision('high')")
                open(f2, "w").write("\n".join(lines2))
                print(f"[OK] {f2.split('/')[-1]}: +TF32")
                break

# 语法检查
import py_compile
for f3 in [f, "/root/beat_mwpm/alphaqubit/training/trainer.py", "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py"]:
    try:
        py_compile.compile(f3, doraise=True)
        print(f"[OK] {f3.split('/')[-1]}: 语法正确")
    except py_compile.PyCompileError as e:
        print(f"[ERR] {f3.split('/')[-1]}: {str(e)[:100]}")
