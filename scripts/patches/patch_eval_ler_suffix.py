#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_eval_ler_suffix.py
给云端 eval_ler.py 加 --ft-suffix 支持 focal 消融 checkpoint。
- 加 --ft-suffix 参数 (default '')
- main 设全局 _FT_SUFFIX (沿用 _MODEL_KW 全局模式)
- load_bert 用 _FT_SUFFIX 找 bert_finetune_d{d}{suffix}/best.pt

向后兼容：不传 --ft-suffix 时行为与原版一致（suffix=''）。
应用：在云端 python3 patch_eval_ler_suffix.py
"""
import sys

F = "/root/beat_mwpm/google_paems_data/bert_experiment/eval_ler.py"
s = open(F).read()

# ---- 1. argparse 加 --ft-suffix + main 设全局 _FT_SUFFIX ----
OLD = ("    ap.add_argument('--num-readout-layers', type=int, default=6)\n"
       "    args=ap.parse_args(); dev=args.device")
NEW = ("    ap.add_argument('--num-readout-layers', type=int, default=6)\n"
       "    ap.add_argument('--ft-suffix', default='', help='bert_finetune_d{d}{suffix} 后缀 (focal 消融用)')\n"
       "    args=ap.parse_args(); dev=args.device\n"
       "    global _FT_SUFFIX\n"
       "    _FT_SUFFIX = args.ft_suffix")
assert OLD in s, "FAIL: eval_ler argparse 锚点未找到"
s = s.replace(OLD, NEW)

# ---- 2. load_bert 用 _FT_SUFFIX ----
OLD_CKPT = 'pre_ckpt=EXP/"checkpoints"/f"bert_pretrain_d{d}"/"best.pt"; ft_ckpt=EXP/"checkpoints"/f"bert_finetune_d{d}"/"best.pt"'
NEW_CKPT = 'pre_ckpt=EXP/"checkpoints"/f"bert_pretrain_d{d}"/"best.pt"; ft_ckpt=EXP/"checkpoints"/f"bert_finetune_d{d}{_FT_SUFFIX}"/"best.pt"'
assert OLD_CKPT in s, "FAIL: load_bert ft_ckpt 锚点未找到"
s = s.replace(OLD_CKPT, NEW_CKPT)

open(F, "w").write(s)
print("[OK] patched eval_ler.py:")
print("  + --ft-suffix (default '')")
print("  + main 设全局 _FT_SUFFIX")
print("  + load_bert 用 _FT_SUFFIX 找 bert_finetune_d{d}{suffix}/best.pt")
print("\n消融 LER 命令:")
print("  python eval_ler.py --distances 5 --ft-suffix _focal")
