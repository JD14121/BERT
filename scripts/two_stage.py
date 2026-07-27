#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""two_stage.py
实验 C：d7 两阶段模态微调。
- Stage1: bert_pretrain_d7 -> 合成软读出 80k 子集 finetune 8k 步 (BCE, lr1e-4) -> bert_finetune_d7_stage1
- Stage2: 加载 stage1 -> 真机硬读出 40k+20k mix finetune 5k 步 (BCE, lr7e-5) -> bert_finetune_d7_twostage
- 评估 real_test + 存 results_summary_d7_twostage.json
依计划书 v2 §3.2。
"""
import sys, os, torch, json, glob, re
import numpy as np
from pathlib import Path

EXP = Path('/root/beat_mwpm/google_paems_data/bert_experiment')
sys.path.insert(0, str(EXP))
sys.path.insert(0, '/root/beat_mwpm/google_paems_data/code')
sys.path.insert(0, '/root/beat_mwpm')  # alphaqubit 包根
os.chdir(str(EXP))

import stim
from xzzx_coord import XZZXCoordinateSystem
from compressed_npy_dataset import load_compressed_npy
from alphaqubit.data.pt_dataset import PTBatchDataset
from xzzx_decoder import XZZXFineTuneDecoder
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from torch.utils.data import Subset, ConcatDataset
from run_experiment import finetune, evaluate_model
from path_config import DATA_DIR as _DD, google_template_path

DATA_DIR = Path(_DD)
d, r, basis, dev = 7, 10, 'Z', 'cuda'
EMBED, HEADS, LAYERS, READOUT = 256, 8, 4, 6

# coord system
circ = stim.Circuit.from_file(str(google_template_path(d, basis, 1)))
cs = XZZXCoordinateSystem(d, circ)

# data
def real_pt(split):
    return glob.glob(str(DATA_DIR/f"real_d{d}"/f"{split}_d{d}_r{r}_*_{basis}.pt"))[0]
def syn_pt(split):
    files = list(DATA_DIR.glob(f"d{d}/{split}_d{d}_r{r}_n*_{basis}.pt"))
    return sorted(files, key=lambda p:int(re.search(r'n(\d+)_',p.name).group(1)))[-1]
syn_train = load_compressed_npy(d, r, basis, DATA_DIR)
real_train = PTBatchDataset(real_pt('train'))
real_val = PTBatchDataset(real_pt('val'))
real_test = PTBatchDataset(real_pt('test'))
syn_val_full = PTBatchDataset(str(syn_pt('val')))
syn_val = Subset(syn_val_full, range(min(20000, len(syn_val_full))))  # 20k 子集加速 eval
print(f"data: syn_train={len(syn_train)} real_train={len(real_train)} real_test={len(real_test)} syn_val(subset)={len(syn_val)}")

def build_bert_pretrain(ckpt):
    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS, num_transformer_layers=LAYERS, dropout=0.1)
    pre.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])
    pre = pre.to(dev)
    return XZZXFineTuneDecoder(coord_system=cs, pretrained_encoder=pre, embed_dim=EMBED, readout_dim=64,
                               n_heads=HEADS, num_transformer_layers=LAYERS, num_readout_layers=READOUT, dropout=0.1).to(dev)

def build_bert_full(ckpt):
    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS, num_transformer_layers=LAYERS, dropout=0.1)
    bert = XZZXFineTuneDecoder(coord_system=cs, pretrained_encoder=pre, embed_dim=EMBED, readout_dim=64,
                               n_heads=HEADS, num_transformer_layers=LAYERS, num_readout_layers=READOUT, dropout=0.1)
    bert.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])
    return bert.to(dev)

results = {}

# ===== Stage 1 =====
print("\n=== Stage 1: bert_pretrain_d7 -> 合成软读出 80k, 8k 步, BCE, lr1e-4 ===")
bert1 = build_bert_pretrain(EXP/'checkpoints'/'bert_pretrain_d7'/'best.pt')
synth_sub = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), 80000, replace=False))
finetune(bert1, synth_sub, syn_val, dev, 8000, lr=1e-4, bs=256,
         save_dir=str(EXP/'checkpoints'/'bert_finetune_d7_stage1'), min_steps=2000, patience=10)
bert1 = bert1.to(dev)
s1 = evaluate_model(bert1, real_test, dev)
print(f"[Stage1] real_test acc={s1['accuracy']:.4f} loss={s1['loss']:.4f}")
results['stage1_real_test'] = s1

# ===== Stage 2 =====
print("\n=== Stage 2: 加载 stage1 -> 真机硬读出 40k+20k mix, 5k 步, BCE, lr7e-5 ===")
bert2 = build_bert_full(EXP/'checkpoints'/'bert_finetune_d7_stage1'/'best.pt')
n_mix = int(len(real_train) * 0.5)  # 20k synth
synth_sub2 = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), n_mix, replace=False))
train_ds = ConcatDataset([real_train, synth_sub2])
print(f"Stage2 train: real {len(real_train)} + synth {n_mix} = {len(train_ds)}")
finetune(bert2, train_ds, real_val, dev, 5000, lr=7e-5, bs=256,
         save_dir=str(EXP/'checkpoints'/'bert_finetune_d7_twostage'), min_steps=1000, patience=10)
bert2 = bert2.to(dev)
s2 = evaluate_model(bert2, real_test, dev)
print(f"[Stage2 twostage] real_test acc={s2['accuracy']:.4f} loss={s2['loss']:.4f}")
results['bert_twostage'] = s2

json.dump({'config':{'distance':d,'stage1':'synth_80k_8ksteps_BCE_lr1e-4','stage2':'real_40k+20kmix_5ksteps_BCE_lr7e-5'},
           'results':results}, open(str(EXP/'results_summary_d7_twostage.json'),'w'), indent=2, ensure_ascii=False)
print(f"\n=== DONE. saved results_summary_d7_twostage.json ===")
print(json.dumps(results, indent=2))
