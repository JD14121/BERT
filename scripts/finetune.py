"""
微调脚本 - scripts/finetune.py

用于加载预训练权重并微调到下游解码任务。

使用示例：
    python scripts/finetune.py \
        --pretrain_checkpoint checkpoints/pretrain_d3/best.pt \
        --distance 3 \
        --rounds 25 \
        --batch_size 512 \
        --total_steps 30000 \
        --save_dir checkpoints/finetune_d3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from alphaqubit.data.dataset import SurfaceCodeDataset
from alphaqubit.models.pretrain_decoder import FineTuneDecoder, PretrainDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaQubit 微调')

    # 预训练权重
    parser.add_argument('--pretrain_checkpoint', type=str, required=True,
                        help='预训练检查点路径')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='是否冻结 Encoder')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                        help='Encoder 学习率比例')

    # 数据参数
    parser.add_argument('--distance', type=int, default=3, help='码距离')
    parser.add_argument('--rounds', type=int, default=25, help='纠错轮数')
    parser.add_argument('--p', type=float, default=0.005, help='物理错误率')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--readout_dim', type=int, default=64, help='读出维度')
    parser.add_argument('--num_readout_layers', type=int, default=16, help='ResNet 层数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    parser.add_argument('--total_steps', type=int, default=30000, help='总训练步数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.08, help='权重衰减')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--save_dir', type=str, default='checkpoints/finetune', help='保存目录')

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 从checkpoint 读取保存时的超参数，而不是用命令行默认值
    ckpt = torch.load(args.pretrain_checkpoint, map_location='cpu', weights_only=False)
    saved_hparams = ckpt.get('hparams', {})
    embed_dim = saved_hparams.get('embed_dim', args.embed_dim)

    # 2. 必须先建coord_system，再建模型
    import stim
    from path_config import GOOGLE_SC, GOOGLE_PATCH
    from xzzx_coord import XZZXCoordinateSystem
    circuit_path = GOOGLE_SC / f"d{args.distance}_at_{GOOGLE_PATCH[args.distance]}" \
                   / "Z" / f"r{args.rounds:02d}" / "circuit_ideal.stim"
    cs = XZZXCoordinateSystem(
        args.distance,
        stim.Circuit.from_file(str(circuit_path))
    )

    # 3. 用与预训练一致的维度重建PretrainDecoder
    pretrain_model = PretrainDecoder(
        coord_system=cs,  # ← 必须传真实 coord_system
        embed_dim=embed_dim,  # ← 从 checkpoint 读取
        n_heads=saved_hparams.get('n_heads', 4),
        num_transformer_layers=saved_hparams.get('num_transformer_layers', 2),
    )
    pretrain_model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"  预训练权重加载成功（embed_dim={embed_dim}）")

    # finetune.py：必须用与预训练相同来源的 .pt 文件
    from alphaqubit.data.pt_dataset import PTBatchDataset
    from bert_pretrain import IQFilterDataset  # 复用已有的 IQ 过滤数据集

    # 使用与bert_pretrain.py 同源的 .pt 数据
    train_dataset = IQFilterDataset(
        DATA_DIR / f"d{args.distance}" / f"train_d{args.distance}_r{args.rounds}_n100000_Z.pt",
        batch_size=args.batch_size,
    )
    val_dataset = IQFilterDataset(
        DATA_DIR / f"d{args.distance}" / f"val_d{args.distance}_r{args.rounds}_n10000_Z.pt",
        batch_size=args.batch_size,
    )

    # XZZX微调解码器
    from xzzx_decoder import XZZXFineTuneDecoder

    model = XZZXFineTuneDecoder(
        coord_system=cs,  # XZZXCoordinateSystem
        pretrained_encoder=pretrain_model,
        embed_dim=embed_dim,
        readout_dim=args.readout_dim,
        num_readout_layers=args.num_readout_layers,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 创建配置和 Trainer
    print("\n[4/4] 创建 Trainer...")
    config = TrainingConfig(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=args.save_dir,
    )

    # 如果分层学习率，修改优化器
    if not args.freeze_encoder and args.encoder_lr_ratio != 1.0:
        from torch.optim import AdamW
        trainer.optimizer = AdamW([
            {
                'params': model.get_encoder_parameters(),
                'lr': args.learning_rate * args.encoder_lr_ratio,
                'name': 'encoder'
            },
            {
                'params': model.get_readout_parameters(),
                'lr': args.learning_rate,
                'name': 'readout'
            },
        ], weight_decay=args.weight_decay)
        print(f"  使用分层学习率: Encoder={args.learning_rate * args.encoder_lr_ratio:.2e}, Readout={args.learning_rate:.2e}")

    # 开始训练
    print("\n" + "=" * 60)
    print("开始微调！")
    print("=" * 60 + "\n")

    history = trainer.train()

    print("\n" + "=" * 60)
    print("微调完成！")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最终检查点: {args.save_dir}/best.pt")
    print("=" * 60)

    return history


if __name__ == '__main__':
    main()
