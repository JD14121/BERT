#!/usr/bin/env python3
"""run_experiment.py (P3): 三模型对比 d3（AlphaQubit 基准 / MWPM / BERT=Ours）。
- AlphaQubit: XZZXAlphaQubitDecoder 合成软数据监督预训练 -> 真机硬数据微调
- BERT: 加载 pretrain_encoder.pt -> XZZXFineTuneDecoder 真机硬数据微调（+合成软掺杂）
- MWPM: PAEMS 校准 DEM (R5=b) 解码真机 test detection_events
- 评估: test accuracy (+ LER)
"""
import sys, os, json, time, argparse
from pathlib import Path
os.environ['TQDM_DISABLE'] = '1'
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
PROJECT_ROOT = CODE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR, CONFIG_DIR, PAEMS_SC
from xzzx_coord import XZZXCoordinateSystem
from xzzx_decoder import XZZXAlphaQubitDecoder, XZZXFineTuneDecoder
import stim, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(PAEMS_SC))
from inject_basic_noise import inject_surface_code_noise                # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit  # noqa: E402
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.decoder import AlphaQubitDecoder
from alphaqubit.models.pretrain_decoder import PretrainDecoder, FineTuneDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig


def make_coord(d, basis='Z', r=10):
    cir = stim.Circuit.from_file(str(GOOGLE_SC/f"d{d}_at_{GOOGLE_PATCH[d]}"/basis/f"r{r:02d}"/"circuit_ideal.stim"))
    return XZZXCoordinateSystem(d, cir)


def evaluate_model(model, ds, device, bs=1024):
    model.eval(); loader = DataLoader(ds, batch_size=bs, shuffle=False)
    tc=ts=0.0; tl=0.0
    with torch.no_grad():
        for b in loader:
            m=b['measurement'].to(device); e=b['event'].to(device); fs=b['final_soft'].to(device)
            lb=b['label'].to(device)
            lk=torch.zeros_like(m); el=torch.zeros_like(m)
            logit=model(m,e,lk,el,fs,n_rounds=m.shape[1])
            pred=(torch.sigmoid(logit)>0.5).float()
            tc+=(pred==lb).float().sum().item(); ts+=lb.shape[0]; tl+=F.binary_cross_entropy_with_logits(logit,lb).item()*lb.shape[0]
    return {'accuracy':tc/ts, 'loss':tl/ts}


def finetune(model, train_ds, val_ds, device, steps, lr=1e-4, bs=256, save_dir=None, focal_gamma=0.0, min_steps=0, patience=10000):
    cfg=TrainingConfig(total_steps=steps, batch_size=bs, eval_interval=500, log_interval=500,
                       learning_rate=lr, device=device, use_amp=True, early_stopping_patience=patience, focal_gamma=focal_gamma, min_steps=min_steps)
    tr=Trainer(model=model, train_dataset=train_ds, val_dataset=val_ds, config=cfg, save_dir=save_dir)
    tr.train()
    return model


def mwpm_eval(d, real_test_pt, basis='Z', r=10):
    import pymatching
    cfg=CONFIG_DIR/f"calibrated_d{d}.json"
    tmpl=GOOGLE_SC/f"d{d}_at_{GOOGLE_PATCH[d]}"/basis/f"r{r:02d}"/"circuit_ideal.stim"
    base,dq,xs,zs,cx=generate_surface_code_circuit(d,r,basis,code_variant='xzzx',xzzx_template=str(tmpl))
    noisy=inject_surface_code_noise(base,dq,xs,zs,cx,str(cfg))
    dem=noisy.detector_error_model()
    mwpm=pymatching.Matching.from_detector_error_model(dem)
    pt=torch.load(str(real_test_pt),map_location='cpu',weights_only=False)
    det=pt['detection_events'].numpy().astype(np.uint8); lb=pt['label'].numpy().astype(int)
    preds=mwpm.decode_batch(det).flatten()
    return float((preds==lb).mean())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--distance',type=int,default=3); ap.add_argument('--rounds',type=int,default=10)
    ap.add_argument('--basis',default='Z'); ap.add_argument('--device',default='cuda')
    ap.add_argument('--aq-pretrain-steps',type=int,default=10000)
    ap.add_argument('--finetune-steps',type=int,default=3000)
    ap.add_argument('--mix-synth-ratio',type=float,default=0.5,help='BERT微调掺杂合成软数据比例')
    ap.add_argument('--embed-dim',type=int,default=256)
    ap.add_argument('--n-heads',type=int,default=8)
    ap.add_argument('--num-transformer-layers',type=int,default=4)
    ap.add_argument('--num-readout-layers',type=int,default=6)
    ap.add_argument('--train-n',type=int,default=0,help='合成 train N（0=glob 取最大=10×/2×；E0 回归传 800000）')
    ap.add_argument('--val-n',type=int,default=0,help='合成 val N（0=glob 最大）')
    ap.add_argument('--batch-size',type=int,default=256,help='训练 batch size（d5/d7 大模型用 128 防 OOM）')
    ap.add_argument('--bert-focal-gamma',type=float,default=0.0,help='BERT finetune focal_gamma (0=BCE, 2=focal)')
    ap.add_argument('--start-from',default='aq_pretrain',choices=['aq_pretrain','bert_finetune'],help='bert_finetune=跳过 AQ+MWPM 直接做 BERT 微调')
    ap.add_argument('--ft-suffix',default='',help='bert_finetune_d{d}{suffix} 目录与 results 文件名后缀')
    ap.add_argument('--real-suffix',default='',help='real_d{d}{suffix} 真机数据目录后缀 (symaug 用 _aug)')
    args=ap.parse_args(); d,r,basis,dev=args.distance,args.rounds,args.basis,args.device
    cs=make_coord(d,basis,r)
    EXP=Path(__file__).resolve().parent
    # 数据（合成 N 随 scale 变，glob 取最大或 --train-n 指定；真机用 glob）
    import glob, re
    def real_pt(split): return glob.glob(str(DATA_DIR/f"real_d{d}{args.real_suffix}"/f"{split}_d{d}_r{r}_*_{basis}.pt"))[0]
    def syn_pt(split, n_override):
        files=list(DATA_DIR.glob(f"d{d}/{split}_d{d}_r{r}_n*_{basis}.pt"))
        if n_override and n_override>0:
            sel=[f for f in files if f"n{n_override}_" in f.name]
            assert sel, f"未找到 d{d} {split} n={n_override}"
            return sel[0]
        return sorted(files, key=lambda p:int(re.search(r'n(\d+)_',p.name).group(1)))[-1]
    from compressed_npy_dataset import load_compressed_npy; syn_train=load_compressed_npy(d, r, basis, DATA_DIR)
    syn_val=PTBatchDataset(str(syn_pt('val', args.val_n)))
    # 10× 数据 syn_val 达 1M，全量 eval 慢；子采样 200k 供 AQ 预训练 eval（真机 real_val 小，无需子采样）
    from torch.utils.data import Subset as _Subset
    _VE = 200000
    syn_val_eval = _Subset(syn_val, range(min(_VE, len(syn_val)))) if len(syn_val) > _VE else syn_val
    real_train=PTBatchDataset(real_pt('train'))
    real_val=PTBatchDataset(real_pt('val'))
    real_test=PTBatchDataset(real_pt('test'))
    print(f"[P3] d{d} r{r} {basis} | syn_train={len(syn_train)} real_train={len(real_train)} real_test={len(real_test)}")

    results={}

    # === MWPM (PAEMS 校准 DEM, R5=b) (start-from=bert_finetune 时跳过) ===
    if args.start_from == 'aq_pretrain':
        print("\n=== MWPM (PAEMS calibrated DEM) ===")
        t0=time.time(); results['mwpm']={'accuracy':mwpm_eval(d, real_pt('test'), basis, r)}
        print(f"MWPM test acc={results['mwpm']['accuracy']:.4f} ({time.time()-t0:.0f}s)")
    else:
        print(f"[SKIP] MWPM (start-from={args.start_from})")

    # === AlphaQubit 基准 (start-from=bert_finetune 时跳过) ===
    if args.start_from == 'aq_pretrain':
        print("\n=== AlphaQubit: 合成监督预训练 -> 真机微调 ===")
        aq=XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=args.embed_dim, n_heads=args.n_heads, num_transformer_layers=args.num_transformer_layers, num_readout_layers=args.num_readout_layers, dropout=0.1, use_late_fusion=True).to(dev)
        print(f"AlphaQubit 合成监督预训练 {args.aq_pretrain_steps} 步...")
        finetune(aq, syn_train, syn_val_eval, dev, args.aq_pretrain_steps, lr=2e-4, bs=args.batch_size, save_dir=str(EXP/"checkpoints"/f"aq_pretrain_d{d}"), focal_gamma=2.0, min_steps=10000, patience=10)
        print("AlphaQubit 真机微调...")
        finetune(aq, real_train, real_val, dev, args.finetune_steps, lr=1e-4, bs=args.batch_size, save_dir=str(EXP/"checkpoints"/f"aq_finetune_d{d}"), focal_gamma=2.0, min_steps=2000, patience=8)
        aq=aq.to(dev); results['alphaqubit']=evaluate_model(aq, real_test, dev)
        print(f"AlphaQubit test acc={results['alphaqubit']['accuracy']:.4f} loss={results['alphaqubit']['loss']:.4f}")
    else:
        print(f"[SKIP] AlphaQubit stages (start-from={args.start_from})")

    # === BERT (Ours): 加载预训练 encoder -> 真机微调 (+合成软掺杂) ===
    print("\n=== BERT (Ours): 加载 encoder -> 真机微调 ===")
    best_path=EXP/"checkpoints"/f"bert_pretrain_d{d}"/"best.pt"
    pre=PretrainDecoder(coord_system=cs, embed_dim=args.embed_dim, n_heads=args.n_heads, num_transformer_layers=args.num_transformer_layers, dropout=0.1)
    pre.load_state_dict(torch.load(str(best_path),map_location='cpu',weights_only=False)['model_state_dict'])
    pre=pre.to(dev)
    bert=XZZXFineTuneDecoder(coord_system=cs, pretrained_encoder=pre, embed_dim=args.embed_dim, readout_dim=64,
                             n_heads=args.n_heads, num_transformer_layers=args.num_transformer_layers, num_readout_layers=args.num_readout_layers, dropout=0.1).to(dev)
    # 合成软掺杂：mix_ratio 真机 + (1-mix_ratio) 合成（按子集）
    if args.mix_synth_ratio>0:
        n_mix=int(len(real_train)*args.mix_synth_ratio)
        synth_sub=Subset(syn_train, np.random.default_rng(42).choice(len(syn_train),n_mix,replace=False))
        # 简化：用真机为主，合成子集并入（需合并 dataset-用 ConcatDataset）
        from torch.utils.data import ConcatDataset
        train_ds=ConcatDataset([real_train, synth_sub])
        print(f"BERT 微调掺杂: real {len(real_train)} + synth {n_mix} = {len(train_ds)}")
    else:
        train_ds=real_train
    finetune(bert, train_ds, real_val, dev, args.finetune_steps, lr=1e-4, bs=args.batch_size, save_dir=str(EXP/"checkpoints"/f"bert_finetune_d{d}{args.ft_suffix}"), focal_gamma=args.bert_focal_gamma, min_steps=2000, patience=10)
    bert=bert.to(dev); results['bert']=evaluate_model(bert, real_test, dev)
    print(f"BERT test acc={results['bert']['accuracy']:.4f} loss={results['bert']['loss']:.4f}")

    # === 汇总 ===
    print("\n=== P3 results ===")
    for k,v in results.items(): print(f"  {k}: {v}")
    out=EXP/f"results_summary_d{d}{args.ft_suffix}.json"
    json.dump({'config':vars(args),'results':results}, open(out,'w',encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"saved {out}")


if __name__=='__main__':
    main()
