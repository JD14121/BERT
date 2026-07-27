#!/usr/bin/env python3
"""bert_pretrain.py (P2): BERT 自监督预训练（Mixed Structured MSM）on 合成 d3 软读出数据。
PretrainDecoder + XZZXCoordinateSystem + MixedStructuredMSM（重设计掩码）+ PretrainTrainer。
无标签（仅 measurement/event）。监控 mask accuracy。
"""
import sys, os, argparse, time
from pathlib import Path

os.environ['TQDM_DISABLE'] = '1'
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
PROJECT_ROOT = CODE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR
from xzzx_coord import XZZXCoordinateSystem
from mixed_msm import MixedStructuredMSM
import stim, torch
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig


class IQFilterDataset(PTBatchDataset):
    """IQ 适配数据集：在 DataLoader 加载时剥离复数张量。
    BERT 掩码解码器实际需要重构的是由 IQ 推算而来的、处于 0~1 的软概率（event/measurement）。
    过滤掉 base_iq 原始簇系可以避免 GPU 显存 OOM 以及 PyTorch AMP 的 complex64 类型崩溃。
    """
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        # 将后缀为 _iq 的所有复数字段从当前 Batch 中剔除
        return {k: v for k, v in batch.items() if not k.endswith('_iq')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--basis', default='Z')
    ap.add_argument('--embed-dim', type=int, default=128)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--num-transformer-layers', type=int, default=2)
    ap.add_argument('--mask-ratio', type=float, default=0.25)
    ap.add_argument('--use-round-mask', action='store_true', help='③-a: 启用整轮丢弃+变长span掩码（E2）')
    ap.add_argument('--train-n', type=int, default=0, help='合成 train N（0=glob 最大=10×/2×；E0 传 800000）')
    ap.add_argument('--val-n', type=int, default=0, help='合成 val N（0=glob 最大）')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--save-dir', type=str, default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    d, r, basis = args.distance, args.rounds, args.basis

    save_dir = Path(args.save_dir) if args.save_dir else (CODE_DIR.parent / "bert_experiment" / "checkpoints" / f"bert_pretrain_d{d}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 数据（合成软读出；N 随 scale 变，glob 取最大或 --train-n 指定）
    import re
    def _syn(split, n_override):
        # 支持更灵活地寻找当前 d 目录下甚至外层目录下的对应文件
        pattern_strict = f"d{d}/{split}_d{d}_r{r}_n*_{basis}.pt"
        pattern_loose = f"**/{split}_d{d}_r{r}*.pt"

        files = list(DATA_DIR.glob(pattern_strict))
        if not files:
            files = list(DATA_DIR.glob(pattern_loose))  # 针对路径可能没建文件夹时的回退寻找

        if not files:
            raise FileNotFoundError(
                f"\n[严重错误] 找不到必要的预训练数据！\n"
                f"模型试图在 '{DATA_DIR}' 中寻找匹配 '{split}_d{d}_r{r}_n*_{basis}.pt' 的文件，但一无所获。\n"
                f"请先运行数据生成脚本生成 {split} 数据，并确保其存放在正确的目录。"
            )

        if n_override and n_override > 0:
            sel = [f for f in files if f"n{n_override}" in f.name]
            if not sel:
                raise FileNotFoundError(
                    f"找到了部分文件: {[f.name for f in files]}，\n"
                    f"但没有找到精确匹配样本数 n={n_override} 的文件！"
                )
            return sel[0]

        # 安全地提取文件名中由 'n' 开头的数字进行大小排序，防止正则匹配不到而闪退
        def get_n(p):
            match = re.search(r'n(\d+)', p.name)
            return int(match.group(1)) if match else 0

        return sorted(files, key=get_n)[-1]

    train_pt = _syn('train', args.train_n)
    val_pt = _syn('val', args.val_n)

    # 在 trainer 构建之前插入
    train_ds = IQFilterDataset(train_pt, batch_size=args.batch_size)
    val_ds_eval = IQFilterDataset(val_pt, batch_size=args.batch_size)

    # XZZX 坐标系
    cir = stim.Circuit.from_file(str(GOOGLE_SC / f"d{d}_at_{GOOGLE_PATCH[d]}" / basis / f"r{r:02d}" / "circuit_ideal.stim"))
    cs = XZZXCoordinateSystem(d, cir)
    print(f"[XZZX coord] grid={cs.grid_size}x{cs.grid_size} n_stab={cs.n_stab} n_data={cs.n_data}")

    # 模型 + 重设计掩码
    model = PretrainDecoder(coord_system=cs, embed_dim=args.embed_dim, n_heads=args.n_heads,
                            num_transformer_layers=args.num_transformer_layers, dropout=0.1)
    masking = MixedStructuredMSM(mask_ratio=args.mask_ratio, coord_system=cs,
                                 p_random=0.4, p_spatial=0.3, p_temporal=0.3,
                                 cluster_radius=1, span_len=4,
                                 use_full_round=args.use_round_mask, variable_span=args.use_round_mask,
                                 p_full_round=0.2)
    print(f"[model] PretrainDecoder params={sum(p.numel() for p in model.parameters())} | MixedStructuredMSM(0.4/0.3/0.3)")

    config = PretrainConfig(total_steps=args.steps, batch_size=args.batch_size,
                            eval_interval=500, log_interval=200,
                            learning_rate=args.lr, mask_ratio=args.mask_ratio,
                            device=args.device, use_amp=True, early_stopping_patience=10000)
    trainer = PretrainTrainer(model=model, train_dataset=train_ds, val_dataset=val_ds_eval,
                              config=config, save_dir=str(save_dir))
    trainer.masking = masking   # 替换为重设计掩码（审查组：如实替换，非默认随机）

    t0 = time.time()
    trainer.train()
    val_metrics = trainer.evaluate()
    print(f"\n[P2 DONE] {args.steps} steps in {time.time()-t0:.0f}s | val mask_acc={val_metrics.get('mask_accuracy',0):.4f} val_loss={val_metrics.get('loss',0):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'global_step': step,
        'hparams': {
            'embed_dim': args.embed_dim,
            'n_heads': args.n_heads,
            'num_transformer_layers': args.num_transformer_layers,
            'distance': args.distance,
            'rounds': args.rounds,
        }
    }, ckpt_path)

if __name__ == '__main__':
    main()
