#!/usr/bin/env python3
"""final_tf32_fix.py - 添加 TF32 + cudnn 到缺失文件"""
import os

TF32_CODE = "\ntorch.backends.cudnn.benchmark = True  # 固定形状自动调优\ntorch.set_float32_matmul_precision('high')  # TF32 V100 Tensor Core ~3x\n"

files = [
    "/root/beat_mwpm/alphaqubit/training/trainer.py",
    "/root/beat_mwpm/alphaqubit/training/pretrain_trainer.py",
    "/root/beat_mwpm/google_paems_data/bert_experiment/bert_pretrain.py",
]

for fpath in files:
    s = open(fpath).read()
    if "set_float32_matmul_precision" in s:
        print(f"[SKIP] {os.path.basename(fpath)}: TF32 already exists")
        continue
    lines = s.split("\n")
    last_import = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = i
    if last_import >= 0:
        lines.insert(last_import + 1, "torch.backends.cudnn.benchmark = True")
        lines.insert(last_import + 2, "torch.set_float32_matmul_precision('high')")
        open(fpath, "w").write("\n".join(lines))
        print(f"[OK] {os.path.basename(fpath)}: +TF32 +cudnn (after import line {last_import+1})")
    else:
        s = "torch.backends.cudnn.benchmark = True\ntorch.set_float32_matmul_precision('high')\n" + s
        open(fpath, "w").write(s)
        print(f"[OK] {os.path.basename(fpath)}: +TF32 +cudnn (prepended)")

# 验证
print("\n=== 验证 ===")
for fpath in files + ["/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"]:
    s = open(fpath).read()
    name = os.path.basename(fpath)
    tf = "Y" if "set_float32_matmul_precision" in s else "N"
    cd = "Y" if "cudnn.benchmark" in s else "N"
    ga = "Y" if "gradient_accumulation_steps" in s else "N"
    al = "Y" if "accum = getattr" in s else "N"
    uc = "Y" if ("use_compile: bool = True" in s or "use_compile=True" in s) else "N"
    print(f"  {name}: TF32={tf} cudnn={cd} grad_accum={ga} accum_loop={al} compile={uc}")
