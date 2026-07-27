#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_real_suffix.py: 给 run_experiment.py 加 --real-suffix，支持对称增强数据 real_d{d}{suffix}/
- 加 --real-suffix (default '')
- real_pt 查 real_d{d}{args.real_suffix}/
向后兼容（suffix='' 时用原 real_d{d}/）。
"""
import sys
F = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s = open(F).read()

# 1. 加 --real-suffix 参数（在 --ft-suffix 后）
OLD_ARGS = "    ap.add_argument('--ft-suffix',default='',help='bert_finetune_d{d}{suffix} 目录与 results 文件名后缀')\n    args=ap.parse_args()"
NEW_ARGS = ("    ap.add_argument('--ft-suffix',default='',help='bert_finetune_d{d}{suffix} 目录与 results 文件名后缀')\n"
            "    ap.add_argument('--real-suffix',default='',help='real_d{d}{suffix} 真机数据目录后缀 (symaug 用 _aug)')\n"
            "    args=ap.parse_args()")
assert OLD_ARGS in s, "FAIL: argparse 锚点未找到"
s = s.replace(OLD_ARGS, NEW_ARGS)

# 2. real_pt 用 real_d{d}{args.real_suffix}
OLD_REAL = 'def real_pt(split): return glob.glob(str(DATA_DIR/f"real_d{d}"/f"{split}_d{d}_r{r}_*_{basis}.pt"))[0]'
NEW_REAL = 'def real_pt(split): return glob.glob(str(DATA_DIR/f"real_d{d}{args.real_suffix}"/f"{split}_d{d}_r{r}_*_{basis}.pt"))[0]'
assert OLD_REAL in s, "FAIL: real_pt 锚点未找到"
s = s.replace(OLD_REAL, NEW_REAL)

# 校验
assert "--real-suffix" in s and "real_d{d}{args.real_suffix}" in s
open(F, "w").write(s)
print("[OK] patched run_experiment.py: + --real-suffix, real_pt 用 real_d{d}{args.real_suffix}")
