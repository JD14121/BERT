#!/usr/bin/env python3
"""bert_pretrain.py (P2): BERT 自监督预训练（Mixed Structured MSM）on合成 d3 软读出数据。
PretrainDecoder + XZZXCoordinateSystem + MixedStructuredMSM + PretrainTrainer。
预训练完成后可选地调用 XZZXFineTuneDecoder 做流式解码评估（--eval-stream）。
"""
import sys, os, argparse, time, json, gc
from pathlib import Path

os.environ['TQDM_DISABLE'] = '1'
CODE_DIR     = Path(__file__).resolve().parent.parent / "code"
PROJECT_ROOT = CODE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR
from xzzx_coord import XZZXCoordinateSystem
from mixed_msm import MixedStructuredMSM
from xzzx_decoder import XZZXAlphaQubitDecoder, XZZXFineTuneDecoder
from stream_decoder import make_xzzx_decoder_fn

import stim, torch, numpy as np
from torch.utils.data import IterableDataset, DataLoader
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig
import paems_noise_model as pnm
import paems_iq_readout as pir

# ── 流式数据集（不从磁盘加载 .pt，每epoch 按需生成） ────────────────────────────
class StreamingPAEMSDataset(IterableDataset):
    """
    流式 PAEMS 软读出数据集。
    - 不预加载任何 .pt 文件，不积累全量数组
    - 每次__iter__ 按 chunk_size 生成，chunk处理完即GC 回收
    - 只保留float32 软概率字段（剔除 _iq 复数字段，避免 AMP崩溃）
    """
    def __init__(self, distance: int, rounds: int, basis: str,
                 num_samples: int, params: dict, *,
                 snr: float = 10.0, t: float = 0.01,
                 seed: int = 42, chunk_size: int = 4096,
                 include_leakage: bool = False):
        self.distance        = distance
        self.rounds          = rounds
        self.basis           = basis
        self.num_samples     = num_samples
        self.params          = params
        self.snr             = snr
        self.t               = t
        self.seed            = seed
        self.chunk_size      = chunk_size
        self.include_leakage = include_leakage

    def __iter__(self):
        for chunk in pir.stream_paems_iq_dataset(
            self.distance, self.rounds, self.num_samples, self.params,
            snr=self.snr, t=self.t, seed=self.seed,
            chunk_size=self.chunk_size,
            include_leakage=self.include_leakage,
        ):
            batch_n = int(chunk['label'].shape[0])
            for i in range(batch_n):
                # 只yield BERT 预训练需要的 float32 字段
                yield {
                    'measurement':chunk['measurement'][i],
                    'event':            chunk['event'][i],
                    'final_soft':       chunk['final_soft'][i],
                    'detection_events': chunk['detection_events'][i],
                    'label':            chunk['label'][i],
                }
            del chunk
            gc.collect()

    def __len__(self):
        return self.num_samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance',type=int,   default=3)
    ap.add_argument('--rounds',                type=int,   default=10)
    ap.add_argument('--basis',                 default='Z')
    ap.add_argument('--embed-dim',             type=int,   default=128)
    ap.add_argument('--n-heads',               type=int,   default=4)
    ap.add_argument('--num-transformer-layers',type=int,   default=2)
    ap.add_argument('--mask-ratio',            type=float, default=0.25)
    ap.add_argument('--use-round-mask',        action='store_true')
    ap.add_argument('--train-n',type=int,   default=800000,
                    help='流式 train 样本数（旧版用 0=从文件 glob，现在直接指定）')
    ap.add_argument('--val-n',                 type=int,   default=100000,
                    help='流式 val 样本数')
    ap.add_argument('--chunk-size',            type=int,   default=4096,
                    help='每次生成的 chunk 大小')
    ap.add_argument('--batch-size',            type=int,   default=256)
    ap.add_argument('--steps',                 type=int,   default=10000)
    ap.add_argument('--lr',                    type=float, default=2e-4)
    ap.add_argument('--save-dir',              type=str,   default=None)
    ap.add_argument('--device',                default='cuda')
    #── 新增：预训练结束后流式解码评估 ─────────────────────────────────────────
    ap.add_argument('--eval-stream',           action='store_true',
                    help='预训练结束后用XZZXFineTuneDecoder 做流式解码评估')
    ap.add_argument('--eval-n',                type=int,   default=10000,
                    help='流式评估样本数')
    ap.add_argument('--finetune-checkpoint',   type=str,   default=None,
                    help='XZZXFineTuneDecoder 检查点路径（None=使用刚预训练的权重）')
    args = ap.parse_args()

    d, r, basis = args.distance, args.rounds, args.basis

    save_dir = (Path(args.save_dir) if args.save_dir
                else CODE_DIR.parent / "bert_experiment" / "checkpoints" / f"bert_pretrain_d{d}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载或生成 per-distance 设备参数 ──────────────────────────────────────
    params_path = CODE_DIR.parent / "params" / f"paems_params_d{d}.json"
    if params_path.exists():
        with open(params_path, encoding='utf-8') as f:
            params = json.load(f)
        print(f"[params] loaded {params_path}")
    else:
        params = pnm.generate_paems_params(d, seed=d * 7919+ 42)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
        print(f"[params] generated + saved {params_path}")

    # ── XZZX 坐标系（从 Google 模板电路） ─────────────────────────────────────
    circuit_path = (GOOGLE_SC
                    / f"d{d}_at_{GOOGLE_PATCH[d]}"
                    / basis
                    / f"r{r:02d}"
                    / "circuit_ideal.stim")
    cir = stim.Circuit.from_file(str(circuit_path))
    cs= XZZXCoordinateSystem(d, cir)
    print(f"[XZZX coord] grid={cs.grid_size}×{cs.grid_size}"
          f"n_stab={cs.n_stab}  n_data={cs.n_data}")

    # ── 流式数据集（不落盘、不预加载） ───────────────────────────────────────
    train_ds = StreamingPAEMSDataset(
        d, r, basis, args.train_n, params,
        snr=10.0, t=0.01, seed=42,
        chunk_size=args.chunk_size, include_leakage=False,
    )
    val_ds_eval = StreamingPAEMSDataset(
        d, r, basis, args.val_n, params,
        snr=10.0, t=0.01, seed=9999,
        chunk_size=args.chunk_size, include_leakage=False,
    )
    print(f"[dataset] StreamingPAEMSDataset"
          f"train_n={args.train_n:,}  val_n={args.val_n:,}  "
          f"(no .pt files loaded — pure streaming)")

    # ── PretrainDecoder 模型 + MixedStructuredMSM 掩码 ────────────────────────
    model = PretrainDecoder(
        coord_system=cs,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        dropout=0.1,
    )
    masking = MixedStructuredMSM(
        mask_ratio=args.mask_ratio,
        coord_system=cs,
        p_random=0.4, p_spatial=0.3, p_temporal=0.3,
        cluster_radius=1, span_len=4,
        use_full_round=args.use_round_mask,variable_span=args.use_round_mask,
        p_full_round=0.2,
    )
    print(f"[model] PretrainDecoder  "
          f"params={sum(p.numel() for p in model.parameters()):,}  "
          f"| MixedStructuredMSM(p_random=0.4 / p_spatial=0.3 / p_temporal=0.3)")

    # ── 预训练 ─────────────────────────────────────────────────────────────────
    config = PretrainConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        eval_interval=500,
        log_interval=200,
        learning_rate=args.lr,
        mask_ratio=args.mask_ratio,
        device=args.device,
        use_amp=True,
        early_stopping_patience=10000,
    )
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds_eval,
        config=config,
        save_dir=str(save_dir),
    )
    trainer.masking = masking  # 替换为重设计掩码

    t0 = time.time()
    trainer.train()
    val_metrics = trainer.evaluate()
    print(f"\n[P2 预训练完成] {args.steps} steps耗时 {time.time()-t0:.0f}s  "
          f"val_mask_acc={val_metrics.get('mask_accuracy', 0):.4f}  "
          f"val_loss={val_metrics.get('loss', 0):.4f}")

    # ── 可选：流式解码评估（--eval-stream） ───────────────────────────────────
    if args.eval_stream:
        print("\n" + "=" * 60)
        print("[eval-stream] 使用 XZZXFineTuneDecoder 流式解码评估")
        print("=" * 60)

        #构建微调解码器，注入刚完成预训练的 encoder 权重
        finetune_model = XZZXFineTuneDecoder(
            coord_system=cs,
            pretrained_encoder=model,# 直接传入已预训练的 PretrainDecoder
            embed_dim=args.embed_dim,
            readout_dim=64,
            num_readout_layers=16,
        )

        # 构建解码器回调（可选加载 finetune checkpoint）
        decoder_fn = make_xzzx_decoder_fn(
            model=finetune_model,
            device=args.device,
            log_interval=5,
            checkpoint_path=args.finetune_checkpoint,
        )

        # 构建评估用流式数据集（独立种子，避免与训练数据重叠）
        eval_ds = StreamingPAEMSDataset(
            d, r, basis, args.eval_n, params,
            snr=10.0, t=0.01, seed=77777,
            chunk_size=args.chunk_size, include_leakage=False,
        )

        # 流式推理：chunk 生成 → XZZXFineTuneDecoder 解码 → 丢弃
        samples_done = 0
        t_eval = time.time()
        chunk_buffer = []    # 临时缓存同一 chunk 的样本（IterableDataset 逐样本 yield）

        # 直接用 stream_paems_iq_dataset 而不是 IterableDataset，保持 chunk粒度
        for chunk_idx, chunk in enumerate(
            pir.stream_paems_iq_dataset(
                d, r, args.eval_n, params,
                snr=10.0, t=0.01, seed=77777,
                chunk_size=args.chunk_size,
                include_leakage=False,
            )
        ):
            decoder_fn(chunk, chunk_idx)   # 推理 + 累积 LER，不落盘
            samples_done += chunk['detection_events'].shape[0]
            print(
                f"eval chunk {chunk_idx:4d}: "
                f"{samples_done:>7d}/{args.eval_n}  "
                f"running_LER={decoder_fn.state['logical_error_rate']:.5f}"
            )

        # 最终评估汇总
        s = decoder_fn.state
        eval_time = time.time() - t_eval
        print("=" * 60)
        print(f"[eval-stream 完成]  总样本={s['total']:,}耗时={eval_time:.1f}s")
        print(f"  最终 LER= {s['logical_error_rate']:.6f}")
        print(f"  正确预测= {s['correct']:,} / {s['total']:,}")
        print(f"  吞吐量       = {s['total'] / eval_time:.0f} samples/s")
        print(f"  写盘字节     = 0 MB（流式模式）")
        print("=" * 60)

        # 释放评估用模型
        del finetune_model, decoder_fn, eval_ds
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
