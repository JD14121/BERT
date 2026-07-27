#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""two_stage_55m.py
55M 模型两阶段微调（预训练后运行）。
策略优化（基于全部实验教训）：
1. 两阶段模态微调（12M 实验证最优 d7 方案 +1.46pp acc, -41.6% LER）
2. Stage 1 用 80k（深度对齐 25epoch > 1M 2epoch，实证最优）
3. Stage 2 dropout 0.2（防 40k 真机过拟合，55M 参数/样本比恶化）
4. Stage 2 lr 5e-5（55M 大模型需更低 lr 稳定）
5. bs64（V100-32GB 显存约束，55M d7 ~17GB）
6. 早停 patience 10（防过拟合 + 省时间）
7. 50% 合成掺杂（实证最优 mix ratio）

用法（云端）：
  # 先跑预训练
  python bert_pretrain.py --distance 7 --embed-dim 512 --n-heads 8 \
    --num-transformer-layers 6 --steps 30000 --batch-size 64 --lr 1e-4 \
    --save-dir checkpoints/bert_pretrain_d7_55m

  # 再跑两阶段
  python two_stage_55m.py --distance 7 --embed-dim 512 --n-heads 8 \
    --num-transformer-layers 6 --num-readout-layers 8 \
    --pretrain-dir checkpoints/bert_pretrain_d7_55m
"""
import sys, os, json, glob, re, argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Subset, ConcatDataset

EXP = Path('/root/beat_mwpm/google_paems_data/bert_experiment')
sys.path.insert(0, str(EXP))
sys.path.insert(0, '/root/beat_mwpm/google_paems_data/code')
sys.path.insert(0, '/root/beat_mwpm')
os.chdir(str(EXP))

import stim
from path_config import DATA_DIR, GOOGLE_SC, GOOGLE_PATCH
from xzzx_coord import XZZXCoordinateSystem
from compressed_npy_dataset import load_compressed_npy
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from xzzx_decoder import XZZXFineTuneDecoder
from run_experiment import finetune, evaluate_model


def main():
    ap = argparse.ArgumentParser(description="55M 两阶段微调")
    ap.add_argument('--distance', type=int, default=7, choices=[3, 5, 7])
    ap.add_argument('--embed-dim', type=int, default=448)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--num-transformer-layers', type=int, default=6)
    ap.add_argument('--num-readout-layers', type=int, default=8)
    ap.add_argument('--pretrain-dir', type=str, default=None,
                    help='预训练 ckpt 目录（默认 checkpoints/bert_pretrain_d{d}_55m）')
    ap.add_argument('--batch-size', type=int, default=64, help='bs64 for d7, bs128 for d5')
    ap.add_argument('--grad-accum', type=int, default=4, help='梯度累积步数 (bs64×4=bs256, bs128×2=bs256)')
    # Stage 1 参数
    ap.add_argument('--stage1-steps', type=int, default=8000)
    ap.add_argument('--stage1-lr', type=float, default=1e-4)
    ap.add_argument('--stage1-synth-n', type=int, default=80000, help='80k 深度对齐')
    ap.add_argument('--stage1-dropout', type=float, default=0.1)
    # Stage 2 参数
    ap.add_argument('--stage2-steps', type=int, default=5000)
    ap.add_argument('--stage2-lr', type=float, default=5e-5, help='55M 需更低 lr')
    ap.add_argument('--stage2-dropout', type=float, default=0.2, help='防 40k 过拟合')
    ap.add_argument('--stage2-mix-ratio', type=float, default=0.5)
    ap.add_argument('--ft-suffix', type=str, default='_55m')
    args = ap.parse_args()

    d, r, basis, dev = args.distance, 10, 'Z', 'cuda'
    EMBED, HEADS, TLAY = args.embed_dim, args.n_heads, args.num_transformer_layers
    RLAY = args.num_readout_layers
    BS = args.batch_size

    # 坐标系
    circ = stim.Circuit.from_file(str(GOOGLE_SC / f"d{d}_at_{GOOGLE_PATCH[d]}" / basis / "r01" / "circuit_ideal.stim"))
    cs = XZZXCoordinateSystem(d, circ)

    # 数据
    def real_pt(split):
        return glob.glob(str(DATA_DIR / f"real_d{d}" / f"{split}_d{d}_r{r}_*_{basis}.pt"))[0]
    def syn_pt(split):
        files = list(DATA_DIR.glob(f"d{d}/{split}_d{d}_r{r}_n*_{basis}.pt"))
        return sorted(files, key=lambda p: int(re.search(r'n(\d+)_', p.name).group(1)))[-1]

    # 合成数据（npy_compressed for d7, .pt for d5/d3）
    if (DATA_DIR / f"d{d}" / "npy_compressed" / "meta.json").exists():
        syn_train = load_compressed_npy(d, r, basis, DATA_DIR)
        print(f"合成数据: npy_compressed ({len(syn_train)} 样本)")
    else:
        syn_train = PTBatchDataset(str(syn_pt('train')))
        print(f"合成数据: .pt ({len(syn_train)} 样本)")

    real_train = PTBatchDataset(real_pt('train'))
    real_val = PTBatchDataset(real_pt('val'))
    real_test = PTBatchDataset(real_pt('test'))
    syn_val = PTBatchDataset(str(syn_pt('val')))
    syn_val_sub = Subset(syn_val, range(min(20000, len(syn_val))))
    print(f"真机: train={len(real_train)} val={len(real_val)} test={len(real_test)}")

    # 预训练 ckpt
    pretrain_dir = args.pretrain_dir or str(EXP / "checkpoints" / f"bert_pretrain_d{d}_55m")
    pretrain_ckpt = Path(pretrain_dir) / "best.pt"
    assert pretrain_ckpt.exists(), f"预训练 ckpt 不存在: {pretrain_ckpt}"

    # 构建模型
    def build_model(ckpt_path, dropout):
        pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                              num_transformer_layers=TLAY, dropout=dropout)
        pre.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=False)['model_state_dict'])
        pre = pre.to(dev)
        bert = XZZXFineTuneDecoder(coord_system=cs, pretrained_encoder=pre,
                                   embed_dim=EMBED, readout_dim=64, n_heads=HEADS,
                                   num_transformer_layers=TLAY, num_readout_layers=RLAY,
                                   dropout=dropout).to(dev)
        n = sum(p.numel() for p in bert.parameters())
        print(f"  模型: {n/1e6:.1f}M params (embed={EMBED} heads={HEADS} T={TLAY} R={RLAY} dropout={dropout})")
        return bert

    results = {}

    # ===== Stage 1: 合成软读出 =====
    print(f"\n{'='*60}")
    print(f"Stage 1: 合成软读出 {args.stage1_synth_n/1000:.0f}k, {args.stage1_steps} 步, lr={args.stage1_lr}, dropout={args.stage1_dropout}")
    print(f"{'='*60}")
    bert1 = build_model(pretrain_ckpt, args.stage1_dropout)
    synth_sub = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), args.stage1_synth_n, replace=False))
    stage1_dir = str(EXP / "checkpoints" / f"bert_finetune_d{d}_stage1_55m")
    finetune(bert1, synth_sub, syn_val_sub, dev, args.stage1_steps,
             lr=args.stage1_lr, bs=BS, save_dir=stage1_dir, min_steps=2000, patience=10,
             gradient_accumulation_steps=args.grad_accum)
    bert1 = bert1.to(dev)
    s1 = evaluate_model(bert1, real_test, dev)
    print(f"[Stage1] real_test acc={s1['accuracy']:.4f}")
    results['stage1_real_test'] = s1

    # ===== Stage 2: 真机硬读出 =====
    print(f"\n{'='*60}")
    print(f"Stage 2: 真机 {len(real_train)} + mix {int(len(real_train)*args.stage2_mix_ratio)} synth, {args.stage2_steps} 步, lr={args.stage2_lr}, dropout={args.stage2_dropout}")
    print(f"{'='*60}")
    stage1_ckpt = Path(stage1_dir) / "best.pt"
    bert2 = build_model(stage1_ckpt, args.stage2_dropout)
    n_mix = int(len(real_train) * args.stage2_mix_ratio)
    synth_sub2 = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), n_mix, replace=False))
    train_ds = ConcatDataset([real_train, synth_sub2])
    print(f"  训练数据: real {len(real_train)} + synth {n_mix} = {len(train_ds)}")
    stage2_dir = str(EXP / "checkpoints" / f"bert_finetune_d{d}{args.ft_suffix}")
    finetune(bert2, train_ds, real_val, dev, args.stage2_steps,
             lr=args.stage2_lr, bs=BS, save_dir=stage2_dir, min_steps=1500, patience=10,
             gradient_accumulation_steps=args.grad_accum)
    bert2 = bert2.to(dev)
    s2 = evaluate_model(bert2, real_test, dev)
    print(f"\n[Stage2 {args.ft_suffix}] real_test acc={s2['accuracy']:.4f} loss={s2['loss']:.4f}")
    results['bert_55m'] = s2

    # 保存结果
    out = {
        'config': {
            'distance': d, 'model': '55M',
            'embed': EMBED, 'heads': HEADS, 'tlayers': TLAY, 'rlayers': RLAY,
            'stage1': f'synth_{args.stage1_synth_n}_{args.stage1_steps}steps_lr{args.stage1_lr}_drop{args.stage1_dropout}',
            'stage2': f'real_{len(real_train)}+mix{int(args.stage2_mix_ratio*100)}%_{args.stage2_steps}steps_lr{args.stage2_lr}_drop{args.stage2_dropout}',
            'batch_size': BS,
        },
        'results': results
    }
    out_path = str(EXP / f"results_summary_d{d}{args.ft_suffix}.json")
    json.dump(out, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f"\n=== DONE. saved {out_path} ===")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
