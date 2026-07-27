#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""local_train_5m.py
本地 5.5M 模型训练（d3/d5, RTX 4070 SUPER 12GB）。
复用 run_experiment.py 的 make_coord/evaluate_model/finetune + bert_pretrain.py 的 PretrainTrainer/MixedStructuredMSM。

模型：embed=192 / 6 heads / 3 Transformer 层 / 4 readout = 5.22M finetune / 4.98M pretrain
预训练：PAEMS-data/v3（r=25, 500k, PAEMS 噪声）-> MSM 自监督
微调：Google 真机硬读出（.b8 -> .pt via prepare_google_real.py, r=10, XZZX）-> 监督

用法：
  # 0. 先准备真机数据（仅需一次）
  python prepare_google_real.py --distance 3 --rounds 10
  python prepare_google_real.py --distance 5 --rounds 10

  # 1. 预训练
  python local_train_5m.py --distance 3 --stage pretrain --steps 10000
  # 2. 微调
  python local_train_5m.py --distance 3 --stage finetune --steps 3000
  # 3. 评估
  python local_train_5m.py --distance 3 --stage eval
"""
import sys, os, argparse, json, time, glob, re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Subset, ConcatDataset, DataLoader

# ========== 路径 ==========
PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PROJECT_ROOT / "google_paems_data" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(EXP))

import stim
from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR
from xzzx_coord import XZZXCoordinateSystem
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from xzzx_decoder import XZZXFineTuneDecoder as FineTuneDecoder  # XZZX 版有 _patch_late_fusion
from alphaqubit.training.trainer import Trainer, TrainingConfig
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig
from mixed_msm import MixedStructuredMSM
from run_experiment import make_coord, evaluate_model, finetune

GOOGLE_SYNTH_DIR = Path(r"E:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/data")

# ========== 5.5M 模型配置 ==========
EMBED, HEADS, TLAYERS, RLAYERS = 192, 6, 3, 4
DROPOUT = 0.1

# ========== 数据加载 ==========
def load_paems(d):
    """Google XZZX 合成数据（E 盘, r=10, 800k, 软读出 snr=10）"""
    train = PTBatchDataset(str(GOOGLE_SYNTH_DIR / f"d{d}" / f"train_d{d}_r10_n800000_Z.pt"))
    val = PTBatchDataset(str(GOOGLE_SYNTH_DIR / f"d{d}" / f"val_d{d}_r10_n100000_Z.pt"))
    print(f"Google XZZX 合成 d{d}: train={len(train)} val={len(val)} (r=10, snr=10)")
    return train, val

def load_real(d, r=10, basis='Z'):
    """Google 真机（E 盘, r=10, 硬读出）"""
    tp = glob.glob(str(GOOGLE_SYNTH_DIR / f"real_d{d}" / f"train_d{d}_r{r}_*_{basis}.pt"))
    vp = glob.glob(str(GOOGLE_SYNTH_DIR / f"real_d{d}" / f"val_d{d}_r{r}_*_{basis}.pt"))
    xp = glob.glob(str(GOOGLE_SYNTH_DIR / f"real_d{d}" / f"test_d{d}_r{r}_*_{basis}.pt"))
    assert tp, f"真机数据不存在: {GOOGLE_SYNTH_DIR}/real_d{d}/"
    train = PTBatchDataset(tp[0]); val = PTBatchDataset(vp[0] if vp else tp[0]); test = PTBatchDataset(xp[0] if xp else tp[0])
    print(f"Google real d{d}: train={len(train)} val={len(val)} test={len(test)} (r={r})")
    return train, val, test

# ========== 预训练 ==========
def do_pretrain(d, steps=10000, bs=256, lr=2e-4, mask_ratio=0.25):
    cs = make_coord(d)  # XZZXCoordinateSystem（复用 run_experiment）
    train_ds, val_ds = load_paems(d)
    val_sub = Subset(val_ds, range(min(20000, len(val_ds))))  # 子采样加速 eval

    model = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                            num_transformer_layers=TLAYERS, dropout=DROPOUT)
    n = sum(p.numel() for p in model.parameters())
    print(f"PretrainDecoder: {n/1e6:.2f}M params, embed={EMBED} heads={HEADS} layers={TLAYERS}")

    masking = MixedStructuredMSM(mask_ratio=mask_ratio, coord_system=cs,
                                 p_random=0.4, p_spatial=0.3, p_temporal=0.3)
    cfg = PretrainConfig(total_steps=steps, batch_size=bs, eval_interval=500,
                         learning_rate=lr, device='cuda', use_amp=True,
                         mask_ratio=mask_ratio, early_stopping_patience=10000,
                         save_interval=500)
    save_dir = str(EXP / "checkpoints" / f"bert_pretrain_d{d}_5m")
    trainer = PretrainTrainer(model=model, train_dataset=train_ds, val_dataset=val_sub,
                              config=cfg, save_dir=save_dir)
    trainer.masking = masking
    # 从 checkpoint 续训
    ckpt = Path(save_dir) / 'best.pt'
    if ckpt.exists():
        trainer.load_checkpoint(str(ckpt))
        print(f"[RESUME] 已加载 {ckpt}，从 step {trainer.global_step} 续训")
    print(f"预训练启动: {steps} 步, bs={bs}, lr={lr}, mask_ratio={mask_ratio}")
    trainer.train()
    print(f"预训练完成 -> {save_dir}/best.pt")
    print(f"预训练完成 -> {save_dir}/best.pt")

# ========== 微调 ==========
def do_finetune(d, steps=3000, bs=256, lr=1e-4, mix_ratio=0.2):
    cs = make_coord(d)
    real_train, real_val, real_test = load_real(d)

    # 加载预训练 encoder
    ckpt = EXP / "checkpoints" / f"bert_pretrain_d{d}_5m" / "best.pt"
    assert ckpt.exists(), f"预训练 ckpt 不存在: {ckpt}，先跑 --stage pretrain"
    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                          num_transformer_layers=TLAYERS, dropout=DROPOUT)
    pre.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])

    bert = FineTuneDecoder(coord_system=cs, pretrained_encoder=pre,
                           embed_dim=EMBED, readout_dim=64, n_heads=HEADS,
                           num_transformer_layers=TLAYERS, num_readout_layers=RLAYERS, dropout=DROPOUT)
    n = sum(p.numel() for p in bert.parameters())
    print(f"FineTuneDecoder: {n/1e6:.2f}M params")

    # 合成掺杂
    if mix_ratio > 0:
        syn_train, _ = load_paems(d)
        n_mix = int(len(real_train) * mix_ratio)
        syn_sub = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), n_mix, replace=False))
        train_ds = ConcatDataset([real_train, syn_sub])
        print(f"微调数据: real {len(real_train)} + synth {n_mix} = {len(train_ds)}")
    else:
        train_ds = real_train

    save_dir = str(EXP / "checkpoints" / f"bert_finetune_d{d}_5m")
    finetune(bert, train_ds, real_val, 'cuda', steps, lr=lr, bs=bs, save_dir=save_dir)

    # 评估 test accuracy
    bert = bert.to('cuda')
    results = evaluate_model(bert, real_test, 'cuda')
    print(f"\n=== d{d} test accuracy: {results['accuracy']:.4f} loss: {results['loss']:.4f} ===")

    # 存结果
    out = {'distance': d, 'model': '5.5M', 'results': results,
           'config': {'embed': EMBED, 'heads': HEADS, 'tlayers': TLAYERS, 'rlayers': RLAYERS,
                      'finetune_steps': steps, 'lr': lr, 'mix_ratio': mix_ratio}}
    json.dump(out, open(str(EXP / f"results_summary_d{d}_5m.json"), 'w'), indent=2)
    print(f"saved results_summary_d{d}_5m.json")

# ========== 评估 ==========
def do_eval(d):
    cs = make_coord(d)
    _, _, real_test = load_real(d)
    ckpt = EXP / "checkpoints" / f"bert_finetune_d{d}_5m" / "best.pt"
    assert ckpt.exists(), f"微调 ckpt 不存在: {ckpt}"

    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                          num_transformer_layers=TLAYERS, dropout=DROPOUT)
    bert = FineTuneDecoder(coord_system=cs, pretrained_encoder=pre,
                           embed_dim=EMBED, readout_dim=64, n_heads=HEADS,
                           num_transformer_layers=TLAYERS, num_readout_layers=RLAYERS, dropout=DROPOUT)
    bert.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])
    bert = bert.to('cuda')

    results = evaluate_model(bert, real_test, 'cuda')
    print(f"\n=== d{d} 5.5M test accuracy: {results['accuracy']:.4f} ===")
    print("(MWPM/LER 评估请用 eval_ler.py --distances", d, ")")

# ========== 主入口 ==========
def main():
    ap = argparse.ArgumentParser(description="本地 5.5M 模型训练 (d3/d5)")
    ap.add_argument('--distance', type=int, default=3, choices=[3, 5])
    ap.add_argument('--stage', default='pretrain', choices=['pretrain', 'finetune', 'eval'])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--mask-ratio', type=float, default=0.25)
    ap.add_argument('--mix-ratio', type=float, default=0.2)
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"本地 5.5M 训练 | d{args.distance} | {args.stage}")
    print(f"模型: embed={EMBED} heads={HEADS} layers={TLAYERS} readout={RLAYERS}")
    print(f"{'='*60}\n")

    if args.stage == 'pretrain':
        do_pretrain(args.distance, steps=args.steps or 10000, bs=args.batch_size,
                    lr=args.lr or 2e-4, mask_ratio=args.mask_ratio)
    elif args.stage == 'finetune':
        do_finetune(args.distance, steps=args.steps or 3000, bs=args.batch_size,
                    lr=args.lr or 1e-4, mix_ratio=args.mix_ratio)
    elif args.stage == 'eval':
        do_eval(args.distance)

if __name__ == '__main__':
    main()
