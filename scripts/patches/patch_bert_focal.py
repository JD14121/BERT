#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_bert_focal.py
给云端 run_experiment.py 加 focal loss BERT 微调消融支持（不破坏基线行为）。
- 加 --bert-focal-gamma (default 0.0 = BCE)
- 加 --start-from (aq_pretrain | bert_finetune)
- 加 --ft-suffix
- --start-from bert_finetune 时跳过 MWPM + AQ（省 4.6h）
- BERT 微调调用传 focal_gamma + save_dir 加 suffix
- results 文件名加 suffix

向后兼容：不传新参数时行为与原版完全一致。
应用：在云端 python3 patch_bert_focal.py
"""
import sys

F = "/root/beat_mwpm/google_paems_data/bert_experiment/run_experiment.py"
s = open(F).read()

# ---- 1. 加 3 个 CLI 参数（--batch-size 行后）----
OLD_ARGS = ("    ap.add_argument('--batch-size',type=int,default=256,help='训练 batch size（d5/d7 大模型用 128 防 OOM）')\n"
            "    args=ap.parse_args()")
NEW_ARGS = ("    ap.add_argument('--batch-size',type=int,default=256,help='训练 batch size（d5/d7 大模型用 128 防 OOM）')\n"
            "    ap.add_argument('--bert-focal-gamma',type=float,default=0.0,help='BERT finetune focal_gamma (0=BCE, 2=focal)')\n"
            "    ap.add_argument('--start-from',default='aq_pretrain',choices=['aq_pretrain','bert_finetune'],help='bert_finetune=跳过 AQ+MWPM 直接做 BERT 微调')\n"
            "    ap.add_argument('--ft-suffix',default='',help='bert_finetune_d{d}{suffix} 目录与 results 文件名后缀')\n"
            "    args=ap.parse_args()")
assert OLD_ARGS in s, "FAIL: argparse 锚点未找到"
s = s.replace(OLD_ARGS, NEW_ARGS)

# ---- 2. MWPM 块包 if（start-from=bert_finetune 时跳过）----
OLD_MWPM = ("    # === MWPM (PAEMS 校准 DEM, R5=b) ===\n"
            "    print(\"\\n=== MWPM (PAEMS calibrated DEM) ===\")\n"
            "    t0=time.time(); results['mwpm']={'accuracy':mwpm_eval(d, real_pt('test'), basis, r)}\n"
            "    print(f\"MWPM test acc={results['mwpm']['accuracy']:.4f} ({time.time()-t0:.0f}s)\")")
NEW_MWPM = ("    # === MWPM (PAEMS 校准 DEM, R5=b) (start-from=bert_finetune 时跳过) ===\n"
            "    if args.start_from == 'aq_pretrain':\n"
            "        print(\"\\n=== MWPM (PAEMS calibrated DEM) ===\")\n"
            "        t0=time.time(); results['mwpm']={'accuracy':mwpm_eval(d, real_pt('test'), basis, r)}\n"
            "        print(f\"MWPM test acc={results['mwpm']['accuracy']:.4f} ({time.time()-t0:.0f}s)\")\n"
            "    else:\n"
            "        print(f\"[SKIP] MWPM (start-from={args.start_from})\")")
assert OLD_MWPM in s, "FAIL: MWPM 块锚点未找到"
s = s.replace(OLD_MWPM, NEW_MWPM)

# ---- 3. AQ 块包 if（start-from=bert_finetune 时跳过）----
OLD_AQ = ("    # === AlphaQubit 基准: 合成监督预训练 -> 真机微调 ===\n"
          "    print(\"\\n=== AlphaQubit: 合成监督预训练 -> 真机微调 ===\")\n"
          "    aq=XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=args.embed_dim, n_heads=args.n_heads, num_transformer_layers=args.num_transformer_layers, num_readout_layers=args.num_readout_layers, dropout=0.1, use_late_fusion=True).to(dev)\n"
          "    print(f\"AlphaQubit 合成监督预训练 {args.aq_pretrain_steps} 步...\")\n"
          "    finetune(aq, syn_train, syn_val_eval, dev, args.aq_pretrain_steps, lr=2e-4, bs=args.batch_size, save_dir=str(EXP/\"checkpoints\"/f\"aq_pretrain_d{d}\"), focal_gamma=2.0, min_steps=10000, patience=10)\n"
          "    print(\"AlphaQubit 真机微调...\")\n"
          "    finetune(aq, real_train, real_val, dev, args.finetune_steps, lr=1e-4, bs=args.batch_size, save_dir=str(EXP/\"checkpoints\"/f\"aq_finetune_d{d}\"), focal_gamma=2.0, min_steps=2000, patience=8)\n"
          "    aq=aq.to(dev); results['alphaqubit']=evaluate_model(aq, real_test, dev)\n"
          "    print(f\"AlphaQubit test acc={results['alphaqubit']['accuracy']:.4f} loss={results['alphaqubit']['loss']:.4f}\")")
NEW_AQ = ("    # === AlphaQubit 基准 (start-from=bert_finetune 时跳过) ===\n"
          "    if args.start_from == 'aq_pretrain':\n"
          "        print(\"\\n=== AlphaQubit: 合成监督预训练 -> 真机微调 ===\")\n"
          "        aq=XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=args.embed_dim, n_heads=args.n_heads, num_transformer_layers=args.num_transformer_layers, num_readout_layers=args.num_readout_layers, dropout=0.1, use_late_fusion=True).to(dev)\n"
          "        print(f\"AlphaQubit 合成监督预训练 {args.aq_pretrain_steps} 步...\")\n"
          "        finetune(aq, syn_train, syn_val_eval, dev, args.aq_pretrain_steps, lr=2e-4, bs=args.batch_size, save_dir=str(EXP/\"checkpoints\"/f\"aq_pretrain_d{d}\"), focal_gamma=2.0, min_steps=10000, patience=10)\n"
          "        print(\"AlphaQubit 真机微调...\")\n"
          "        finetune(aq, real_train, real_val, dev, args.finetune_steps, lr=1e-4, bs=args.batch_size, save_dir=str(EXP/\"checkpoints\"/f\"aq_finetune_d{d}\"), focal_gamma=2.0, min_steps=2000, patience=8)\n"
          "        aq=aq.to(dev); results['alphaqubit']=evaluate_model(aq, real_test, dev)\n"
          "        print(f\"AlphaQubit test acc={results['alphaqubit']['accuracy']:.4f} loss={results['alphaqubit']['loss']:.4f}\")\n"
          "    else:\n"
          "        print(f\"[SKIP] AlphaQubit stages (start-from={args.start_from})\")")
assert OLD_AQ in s, "FAIL: AQ 块锚点未找到"
s = s.replace(OLD_AQ, NEW_AQ)

# ---- 4. BERT 微调调用加 focal_gamma + suffix ----
OLD_BERT_CALL = 'save_dir=str(EXP/"checkpoints"/f"bert_finetune_d{d}"), min_steps=2000, patience=10)'
NEW_BERT_CALL = 'save_dir=str(EXP/"checkpoints"/f"bert_finetune_d{d}{args.ft_suffix}"), focal_gamma=args.bert_focal_gamma, min_steps=2000, patience=10)'
assert OLD_BERT_CALL in s, "FAIL: BERT 微调调用锚点未找到"
s = s.replace(OLD_BERT_CALL, NEW_BERT_CALL)

# ---- 5. results 文件名加 suffix ----
OLD_OUT = 'out=EXP/f"results_summary_d{d}.json"'
NEW_OUT = 'out=EXP/f"results_summary_d{d}{args.ft_suffix}.json"'
assert OLD_OUT in s, "FAIL: results 文件名锚点未找到"
s = s.replace(OLD_OUT, NEW_OUT)

# ---- 写回 + 校验 ----
open(F, "w").write(s)
print("[OK] patched run_experiment.py:")
print("  + --bert-focal-gamma / --start-from / --ft-suffix (向后兼容默认)")
print("  + MWPM + AQ 跳过逻辑 (start-from=bert_finetune)")
print("  + BERT finetune focal_gamma + save_dir suffix")
print("  + results 文件名 suffix")
print("\n消融命令:")
print("  python run_experiment.py --distance 5 --start-from bert_finetune \\")
print("    --bert-focal-gamma 2.0 --ft-suffix _focal \\")
print("    --embed-dim 256 --n-heads 8 --num-transformer-layers 4 --num-readout-layers 6 \\")
print("    --batch-size 512 --finetune-steps 8000 --mix-synth-ratio 0.5")
