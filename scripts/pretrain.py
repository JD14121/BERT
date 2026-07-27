"""
预训练脚本 - scripts/pretrain.py

用于运行自监督预训练的命令行脚本。

使用示例：
    python scripts/pretrain.py \
        --distance 3 \
        --rounds 25 \
        --p 0.005 \
        --batch_size 512 \
        --total_steps 100000 \
        --save_dir checkpoints/pretrain_d3
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from alphaqubit.data.pretrain_dataset import PretrainDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaQubit 自监督预训练')

    # 数据参数
    parser.add_argument('--distance', type=int, default=3, help='码距离')
    parser.add_argument('--rounds', type=int, default=25, help='纠错轮数')
    parser.add_argument('--p', type=float, default=0.005, help='物理错误率')
    parser.add_argument('--snr', type=float, default=10.0, help='信噪比')
    parser.add_argument('--t', type=float, default=0.01, help='归一化测量时间')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_transformer_layers', type=int, default=3, help='Transformer 层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512, help='批大小')
    parser.add_argument('--total_steps', type=int, default=100000, help='总训练步数')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warmup 步数')

    # Masking 参数
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking 比例')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--save_dir', type=str, default='checkpoints/pretrain', help='保存目录')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 工作进程数')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("AlphaQubit 自监督预训练")
    print("=" * 60)
    print(f"码距离: {args.distance}")
    print(f"纠错轮数: {args.rounds}")
    print(f"物理错误率: {args.p}")
    print(f"嵌入维度: {args.embed_dim}")
    print(f"总训练步数: {args.total_steps}")
    print(f"批大小: {args.batch_size}")
    print(f"Masking 比例: {args.mask_ratio}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 60)

    # 创建数据集
    print("\n[1/4] 创建数据集...")
    train_dataset = PretrainDataset(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        use_soft_readout=True,
        snr=args.snr,
        t=args.t,
        num_samples=1000000,  # 大虚拟大小，实际在线生成
        seed=args.seed,
    )

    val_dataset = PretrainDataset(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        use_soft_readout=True,
        snr=args.snr,
        t=args.t,
        num_samples=10000,
        seed=args.seed + 1,  # 不同种子
    )

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  Stabilizer 数量: {train_dataset.n_stab}")
    print(f"  Data qubit 数量: {train_dataset.n_data}")

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = PretrainDecoder(
        coord_system=train_dataset.coord_system,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        dropout=args.dropout,
    )

    param_stats = model.get_num_parameters()
    print(f"  总参数量: {param_stats['total']:,}")
    print(f"  - SyndromeEmbedder: {param_stats['syndrome_embedder']:,}")
    print(f"  - RNNCore: {param_stats['rnn_core']:,}")
    print(f"  - ReconstructionHead: {param_stats['reconstruction_head']:,}")

    # 创建配置
    print("\n[3/4] 创建训练配置...")
    config = PretrainConfig(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )

    # 创建 Trainer
    print("\n[4/4] 创建 Trainer...")
    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        save_dir=args.save_dir,
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始预训练！")
    print("=" * 60 + "\n")

    history = trainer.train()

    print("\n" + "=" * 60)
    print("预训练完成！")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最终检查点: {args.save_dir}/final.pt")
    print("=" * 60)

    return history


if __name__ == '__main__':
    main()
