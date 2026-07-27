#!/usr/bin/env python3
"""eval_ler.py (P4): LER 评估 d3/d5/d7 三模型（AlphaQubit / BERT=Ours / MWPM）。
加载已训练 checkpoint（aq_finetune_d{d}, bert_finetune_d{d}），在 LER 数据 ler_d{d}_r{n}.pt
上评估，复用 alphaqubit.evaluation.metrics.compute_ler（不自造拟合）。

协议与声明（审查组要求）：
- LER 轮次 {1,10,13,30,50}，AlphaQubit Nature 2024 协议：NN 在 r=10 训练、多轮 OOD 评估。
- ⚠️ LER 在【合成 PAEMS 校准数据】上评估（软读出 snr=10.0），非真机；真机仅 r=10。
- ⚠️ 读出模态失配：NN 在真机硬读出(snr=inf)微调，LER 在合成软读出(snr=10)评估；预训练见过软读出(缓解)。
- MWPM 每轮独立建 DEM（DEMs 轮次相关，num_det=n·n_stab），不复用 r=10 DEM。
- 三模型协议一致：同 LER 数据集、同 0.5 阈值、同 compute_ler。
- 透明度：记录每模型每轮 E(n)/F(n)、min_fidelity 过滤后实际拟合点数、R²/log F₀/slope/is_valid。
"""
import sys, os, json, argparse
from pathlib import Path
os.environ['TQDM_DISABLE'] = '1'
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
PROJECT_ROOT = CODE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR, CONFIG_DIR, PAEMS_SC
from xzzx_coord import XZZXCoordinateSystem
from xzzx_decoder import XZZXAlphaQubitDecoder, XZZXFineTuneDecoder
import stim, torch, numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, str(PAEMS_SC))
from inject_basic_noise import inject_surface_code_noise                # noqa: E402
from surface_code_generate_circuits import generate_surface_code_circuit  # noqa: E402
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.evaluation.metrics import compute_ler

# v2 模型尺寸（与 run_experiment.py 默认一致；main() 可用 CLI 覆盖；E0 回归传 128/4/2/4）
_MODEL_KW = dict(embed_dim=256, n_heads=8, num_transformer_layers=4, num_readout_layers=6)

LER_ROUNDS = [1, 10, 13, 30, 50]


def make_coord(d, basis='Z', r=10):
    cir = stim.Circuit.from_file(str(GOOGLE_SC/f"d{d}_at_{GOOGLE_PATCH[d]}"/basis/f"r{r:02d}"/"circuit_ideal.stim"))
    return XZZXCoordinateSystem(d, cir)


def nn_preds_by_rounds(model, d, dev, bs=256):
    """对每轮 LER 数据分 batch 前向，返回 {n: preds[int]}, {n: labels[int]}。"""
    preds_by_r, labels_by_r = {}, {}
    model.eval()
    for n in LER_ROUNDS:
        if n >= 50:
            print(f"    (n={n}: v2 CycleEmbedding 已改正弦外推，无未训练 MLP 回退；r=50 仍为 OOD 轮次，E({n}) 解读需谨慎)")
        ds = PTBatchDataset(str(DATA_DIR/f"d{d}"/f"ler_d{d}_r{n}_n20000_Z.pt"))
        loader = DataLoader(ds, batch_size=bs, shuffle=False)
        preds_l, labs_l = [], []
        with torch.no_grad():
            for b in loader:
                m=b['measurement'].to(dev); e=b['event'].to(dev); fs=b['final_soft'].to(dev)
                lb=b['label']
                lk=torch.zeros_like(m); el=torch.zeros_like(m)   # 与 run_experiment.py:41-42 一致
                logit=model(m,e,lk,el,fs,n_rounds=m.shape[1])
                pred=(torch.sigmoid(logit)>0.5).float().cpu()
                preds_l.append(pred); labs_l.append(lb)
        preds_by_r[n]=torch.cat(preds_l).numpy().astype(int).flatten()
        labels_by_r[n]=torch.cat(labs_l).numpy().astype(int).flatten()
        e_rate = float(np.mean(preds_by_r[n]!=labels_by_r[n]))
        print(f"    n={n:3d}: E={e_rate:.4f} F={1-2*e_rate:.4f}")
    return preds_by_r, labels_by_r


def mwpm_preds_by_rounds(d, basis='Z'):
    """MWPM 每轮独立建 DEM 解码，返回 {n: preds}, {n: labels}。"""
    import pymatching
    preds_by_r, labels_by_r = {}, {}
    for n in LER_ROUNDS:
        cfg=CONFIG_DIR/f"calibrated_d{d}.json"
        tmpl=GOOGLE_SC/f"d{d}_at_{GOOGLE_PATCH[d]}"/basis/f"r{n:02d}"/"circuit_ideal.stim"
        base,dq,xs,zs,cx=generate_surface_code_circuit(d,n,basis,code_variant='xzzx',xzzx_template=str(tmpl))
        noisy=inject_surface_code_noise(base,dq,xs,zs,cx,str(cfg))
        dem=noisy.detector_error_model()
        mwpm=pymatching.Matching.from_detector_error_model(dem)
        pt=torch.load(str(DATA_DIR/f"d{d}"/f"ler_d{d}_r{n}_n20000_Z.pt"),map_location='cpu',weights_only=False)
        det=pt['detection_events'].numpy().astype(np.uint8); lb=pt['label'].numpy().astype(int).flatten()
        assert det.shape[1]==dem.num_detectors, f"d{d} n={n}: det_width={det.shape[1]} != dem num_det={dem.num_detectors}"
        preds=mwpm.decode_batch(det).flatten().astype(int)
        preds_by_r[n]=preds; labels_by_r[n]=lb
        e_rate=float(np.mean(preds!=lb))
        print(f"    n={n:3d}: E={e_rate:.4f} F={1-2*e_rate:.4f} (dem num_det={dem.num_detectors})")
    return preds_by_r, labels_by_r


def ler_summary(preds_by_r, labels_by_r):
    """compute_ler + 透明度信息（每轮 E/F、拟合点数）。"""
    res = compute_ler(preds_by_r, labels_by_r)
    per_round = {int(n): {'E': float(res.error_rates[n]), 'F': float(res.fidelities[n])} for n in LER_ROUNDS}
    n_fit = int(sum(1 for n in LER_ROUNDS if res.fidelities[n] > 0.1))   # fit_ler min_fidelity=0.1 过滤后
    return {
        'ler': float(res.ler), 'r_squared': float(res.r_squared),
        'log_f0': float(res.log_f0), 'slope': float(res.slope),
        'is_valid': bool(res.is_valid),
        'n_fit_points': n_fit, 'n_total_points': len(LER_ROUNDS),
        'per_round': per_round,
    }


def load_aq(d, cs, dev):
    EXP=Path(__file__).resolve().parent
    ckpt=EXP/"checkpoints"/f"aq_finetune_d{d}"/"best.pt"
    if not ckpt.exists():
        print(f"  [skip] AlphaQubit d{d}: {ckpt.name} 不存在（P3 未完成）"); return None
    aq=XZZXAlphaQubitDecoder(coord_system=cs, dropout=0.1, use_late_fusion=True, **_MODEL_KW).to(dev)
    aq.load_state_dict(torch.load(str(ckpt),map_location='cpu',weights_only=False)['model_state_dict'])
    return aq.to(dev)


def load_bert(d, cs, dev):
    EXP=Path(__file__).resolve().parent
    pre_ckpt=EXP/"checkpoints"/f"bert_pretrain_d{d}"/"best.pt"; ft_ckpt=EXP/"checkpoints"/f"bert_finetune_d{d}{_FT_SUFFIX}"/"best.pt"
    if not (pre_ckpt.exists() and ft_ckpt.exists()):
        print(f"  [skip] BERT d{d}: pretrain/finetune checkpoint 不存在（P2/P3 未完成）"); return None
    _pre_kw={k:_MODEL_KW[k] for k in ('embed_dim','n_heads','num_transformer_layers')}  # PretrainDecoder 无 readout 层
    pre=PretrainDecoder(coord_system=cs, dropout=0.1, **_pre_kw)
    pre.load_state_dict(torch.load(str(pre_ckpt),map_location='cpu',weights_only=False)['model_state_dict'])
    pre=pre.to(dev)
    bert=XZZXFineTuneDecoder(coord_system=cs, pretrained_encoder=pre, readout_dim=64, dropout=0.1, **_MODEL_KW).to(dev)
    bert.load_state_dict(torch.load(str(ft_ckpt),map_location='cpu',weights_only=False)['model_state_dict'])
    return bert.to(dev)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--distances', type=int, nargs='+', default=[3,5,7])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--embed-dim', type=int, default=256)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--num-transformer-layers', type=int, default=4)
    ap.add_argument('--num-readout-layers', type=int, default=6)
    ap.add_argument('--ft-suffix', default='', help='bert_finetune_d{d}{suffix} 后缀 (focal 消融用)')
    args=ap.parse_args(); dev=args.device
    global _FT_SUFFIX
    _FT_SUFFIX = args.ft_suffix
    global _MODEL_KW
    _MODEL_KW = dict(embed_dim=args.embed_dim, n_heads=args.n_heads,
                     num_transformer_layers=args.num_transformer_layers, num_readout_layers=args.num_readout_layers)
    EXP=Path(__file__).resolve().parent
    for d in args.distances:
        print(f"\n=== LER d{d} ===")
        cs=make_coord(d,'Z',10)
        r={}
        print("  [AlphaQubit]")
        aq=load_aq(d,cs,dev)
        if aq is not None:
            pr,lr=nn_preds_by_rounds(aq,d,dev,args.bs); r['alphaqubit']=ler_summary(pr,lr)
            del aq; torch.cuda.empty_cache()
        else:
            r['alphaqubit']=None
        print("  [BERT]")
        bert=load_bert(d,cs,dev)
        if bert is not None:
            pr,lr=nn_preds_by_rounds(bert,d,dev,args.bs); r['bert']=ler_summary(pr,lr)
            del bert; torch.cuda.empty_cache()
        else:
            r['bert']=None
        print("  [MWPM]")
        pr,lr=mwpm_preds_by_rounds(d); r['mwpm']=ler_summary(pr,lr)
        def _fmt(s): return f"LER={s['ler']:.6f}(valid={s['is_valid']},fit={s['n_fit_points']}/5)" if s else "SKIP"
        print(f"  -> AQ {_fmt(r['alphaqubit'])} | BERT {_fmt(r['bert'])} | MWPM {_fmt(r['mwpm'])}")
        json.dump({str(d):r}, open(EXP/f"results_ler_d{d}.json",'w',encoding='utf-8'), indent=2, ensure_ascii=False)
    print("\n=== LER Summary ===")
    for d in args.distances:
        rd=json.load(open(EXP/f"results_ler_d{d}.json",encoding='utf-8'))[str(d)]
        for m in ['alphaqubit','bert','mwpm']:
            v=rd[m]
            if v is None:
                print(f"  d{d} {m:10s}: SKIP (checkpoint missing)"); continue
            print(f"  d{d} {m:10s}: LER={v['ler']:.6f} R²={v['r_squared']:.4f} fit={v['n_fit_points']}/{v['n_total_points']} valid={v['is_valid']}")
    print("saved results_ler_d{d}.json")


if __name__=='__main__':
    main()
