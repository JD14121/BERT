#!/usr/bin/env python3
"""
微调脚本 - scripts/finetune.py（流式 XZZX 版）

改动：
1. SurfaceCodeDataset → StreamingPAEMSDataset（不落盘）
2. FineTuneDecoder → XZZXFineTuneDecoder（XZZX LateFusion 适配）
3. PretrainDecoder 权重加载 → 先加载进 XZZXAlphaQubitDecoder，再传 encoder
4. 新增 XZZXCoordinateSystem 构建
5. Trainer 改用 IterableDataset 兼容模式

使用示例：
    python scripts/finetune.py \
        --pretrain_checkpoint checkpoints/pretrain_d3/best.pt \
        --distance 3 --rounds 25 \
        --batch_size 512 --total_steps 30000 \
        --save_dir checkpoints/finetune_d3
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import stim
import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

from xzzx_coord import XZZXCoordinateSystem
from xzzx_decoder import XZZXAlphaQubitDecoder, XZZXFineTuneDecoder
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig

try:
    from path_config import GOOGLE_SC, GOOGLE_PATCHHAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

import paems_noise_model as pnm
import paems_iq_readout as pir

#── 流式数据集（与其他脚本统一） ───────────────────────────────────────────────
class StreamingPAEMSDataset(IterableDataset):
    """不预加载 .pt，按需逐 chunk 生成软读出数据，chunk 处理完即GC 回收。"""

    def __init__(self, distance, rounds, num_samples, params, *,
                 snr=10.0, t=0.01, seed=42, chunk_size=4096,
                 include_leakage=False):
        self.distance        = distance
        self.rounds          = rounds
        self.num_samples     = num_samples
        self.params          = params
        self.snr             = snr
        self.t               = t
        self.seed            = seed
        self.chunk_size      = chunk_size
        self.include_leakage = include_leakage

    def __iter__(self):
        n_stab = self.distance ** 2 - 1
        for chunk in pir.stream_paems_iq_dataset(
            self.distance, self.rounds, self.num_samples, self.params,
            snr=self.snr, t=self.t, seed=self.seed,
            chunk_size=self.chunk_size,
            include_leakage=self.include_leakage,):
            batch_n = int(chunk['label'].shape[0])
            for i in range(batch_n):
                leakage = (chunk['leakage'][i] if'leakage' in chunk
                           else np.zeros((self.rounds, n_stab), dtype=np.float32))
                event_leakage = (chunk.get('event_leakage', chunk.get('leakage'))[i]
                                 if 'leakage' in chunk
                                 else np.zeros_like(leakage))
                yield {
                    'measurement':chunk['measurement'][i],
                    'event':          chunk['event'][i],
                    'final_soft':     chunk['final_soft'][i],
                    'detection_events': chunk['detection_events'][i],
                    'label':          chunk['label'][i],
                    'leakage':        leakage,
                    'event_leakage':  event_leakage,
                }
            del chunk
            gc.collect()

    def __len__(self):
        return self.num_samples

# ── 工具函数 ───────────────────────────────────────────────────────────────────
def _load_or_gen_params(distance: int, params_dir: Path) -> dict:
    params_path = params_dir / f"paems_params_d{distance}.json"
    if params_path.exists():
        with open(params_path, encoding='utf-8') as f:
            return json.load(f)
    params = pnm.generate_paems_params(distance, seed=distance * 7919+ 42)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    print(f"[params] generated {params_path}")
    return params

def _build_coord_system(distance: int, rounds: int) -> XZZXCoordinateSystem:
    """Google模板优先，回退stim 标准电路。"""
    if HAS_GOOGLE and GOOGLE_PATCH.get(distance):
        circuit_path = (GOOGLE_SC
                        / f"d{distance}_at_{GOOGLE_PATCH[distance]}"
                        / "Z"
                        / f"r{rounds:02d}"
                        / "circuit_ideal.stim")
        if circuit_path.exists():
            cir = stim.Circuit.from_file(str(circuit_path))
            cs = XZZXCoordinateSystem(distance, cir)
            print(f"[coord] XZZX Google 模板  "
                  f"n_stab={cs.n_stab}  n_data={cs.n_data}")
            return cs

    base = pnm._base_surface_code_circuit(distance, rounds)
    cs = XZZXCoordinateSystem(distance, base)
    print(f"[coord] XZZX stim 回退  n_stab={cs.n_stab}  n_data={cs.n_data}")
    return cs

def _load_pretrained_encoder(checkpoint_path: str, coord_system, embed_dim: int,device: str = "cpu") -> XZZXAlphaQubitDecoder:
    """
    将预训练检查点加载进 XZZXAlphaQubitDecoder。
    兼容两种检查点格式：
      - 旧格式：PretrainDecoder.state_dict（不含 late_fusion）
      - 新格式：XZZXAlphaQubitDecoder.state_dict
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    encoder = XZZXAlphaQubitDecoder(coord_system=coord_system, embed_dim=embed_dim)

    #尝试严格加载，失败则宽松加载（旧 PretrainDecoder 权重只覆盖公共层）
    try:
        encoder.load_state_dict(state, strict=True)
        print(f"[pretrain] strict load OKstep={ckpt.get('global_step', 'unknown')}")
    except RuntimeError:
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[pretrain] partial load  "
              f"missing={len(missing)}  unexpected={len(unexpected)}  "
              f"step={ckpt.get('global_step', 'unknown')}")if missing:
            print(f"  missing keys (first 5): {missing[:5]}")

    return encoder.to(device)

# ── 参数解析 ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='AlphaQubit XZZX 微调（流式版）')

    # 预训练权重
    parser.add_argument('--pretrain_checkpoint', type=str, required=True,
                        help='预训练检查点路径')
    parser.add_argument('--freeze_encoder',   action='store_true')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                        help='Encoder 学习率比例')

    # 数据参数
    parser.add_argument('--distance', type=int,default=3)
    parser.add_argument('--rounds',   type=int,   default=25)
    parser.add_argument('--snr',      type=float, default=10.0)
    parser.add_argument('--t',        type=float, default=0.01)
    parser.add_argument('--no_leakage', action='store_true')
    parser.add_argument('--train_n', type=int, default=100000,
                        help='流式训练样本数')
    parser.add_argument('--val_n',   type=int, default=10000,
                        help='流式验证样本数')
    parser.add_argument('--chunk_size', type=int, default=4096)
    parser.add_argument('--params_dir', type=str, default=None,
                        help='PAEMS params JSON 目录（默认与save_dir 同级的 params/）')

    # 模型参数
    parser.add_argument('--embed_dim',type=int, default=256)
    parser.add_argument('--readout_dim',         type=int, default=64)
    parser.add_argument('--num_readout_layers',  type=int, default=16)

    # 训练参数
    parser.add_argument('--batch_size',type=int,   default=512)
    parser.add_argument('--total_steps',   type=int,   default=30000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay',  type=float, default=0.08)

    # 其他
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--device',   type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints/finetune')

    return parser.parse_args()

# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    params_dir = (Path(args.params_dir) if args.params_dir
                  else save_dir.parent / "params")

    print("=" * 60)
    print("AlphaQubit XZZX 微调（流式版）")
    print("=" * 60)
    print(f"预训练检查点 : {args.pretrain_checkpoint}")
    print(f"码距离       : {args.distance}轮数: {args.rounds}")
    print(f"训练样本     : {args.train_n:,}  验证样本: {args.val_n:,}")
    print(f"冻结 Encoder : {args.freeze_encoder}")
    print(f"Encoder LR比: {args.encoder_lr_ratio}")
    print("=" * 60)

    # [1/5] 加载 PAEMS params
    print("\n[1/5] 加载设备参数...")
    params = _load_or_gen_params(args.distance, params_dir)

    # [2/5] 构建 XZZX 坐标系
    print("\n[2/5] 构建 XZZX 坐标系...")
    cs = _build_coord_system(args.distance, args.rounds)

    # [3/5] 加载预训练 encoder（→ XZZXAlphaQubitDecoder）
    print("\n[3/5] 加载预训练权重...")
    pretrained_encoder = _load_pretrained_encoder(
        args.pretrain_checkpoint, cs, args.embed_dim, device="cpu"
    )

    # [4/5] 构建流式数据集
    print("\n[4/5] 构建流式数据集（不加载 .pt 文件）...")
    include_leakage = not args.no_leakage

    train_dataset = StreamingPAEMSDataset(
        distance=args.distance, rounds=args.rounds,
        num_samples=args.train_n, params=params,
        snr=args.snr, t=args.t,
        seed=args.seed, chunk_size=args.chunk_size,
        include_leakage=include_leakage,
    )
    val_dataset = StreamingPAEMSDataset(
        distance=args.distance, rounds=args.rounds,
        num_samples=args.val_n, params=params,
        snr=args.snr, t=args.t,
        seed=args.seed + 9999, chunk_size=args.chunk_size,
        include_leakage=include_leakage,
    )
    print(f"  train_n={args.train_n:,}  val_n={args.val_n:,}"
          f"  leakage={include_leakage}  chunk={args.chunk_size}")

    # [5/5] 构建 XZZXFineTuneDecoder + Trainer
    print("\n[5/5] 构建 XZZXFineTuneDecoder...")
    model = XZZXFineTuneDecoder(
        coord_system=cs,
        pretrained_encoder=pretrained_encoder,  # XZZXAlphaQubitDecoder
        embed_dim=args.embed_dim,
        readout_dim=args.readout_dim,
        num_readout_layers=args.num_readout_layers,
    )

    if args.freeze_encoder:
        model.freeze_encoder()
        print("  Encoder 已冻结")

    total_params= sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量 : {trainable_params:,}")

    config = TrainingConfig(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        #IterableDataset 必须关闭 shuffle
        shuffle_train=False,
        dataloader_num_workers=0,)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=str(save_dir),
    )

    #分层学习率
    if not args.freeze_encoder and args.encoder_lr_ratio != 1.0:
        from torch.optim import AdamW
        trainer.optimizer = AdamW([
            {
                'params': model.get_encoder_parameters(),
                'lr': args.learning_rate * args.encoder_lr_ratio,
                'name': 'encoder',
            },
            {
                'params': model.get_readout_parameters(),
                'lr': args.learning_rate,
                'name': 'readout',
            },
        ], weight_decay=args.weight_decay)print(f"  分层LR: encoder={args.learning_rate * args.encoder_lr_ratio:.2e}"
              f"  readout={args.learning_rate:.2e}")

    print("\n" + "=" * 60)
    print("开始微调！")
    print("=" * 60 + "\n")

    history = trainer.train()

    print("\n" + "=" * 60)
    print("微调完成！")
    print(f"最佳验证损失 : {trainer.best_val_loss:.4f}")
    print(f"检查点路径   : {save_dir}/best.pt")
    print("=" * 60)
    return history

if __name__ == '__main__':
    main()
