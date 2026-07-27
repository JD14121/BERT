#!/usr/bin/env python3
"""
AlphaQubit训练脚本

使用示例:
    # 训练 d=3 模型（多进程在线数据生成，推荐）
    python scripts/train.py --distance 3 --rounds 12 --steps 30000 --num-workers 8

    # 训练 d=5 模型（小批量调试）
    python scripts/train.py --distance 5 --rounds 6 --steps 1000 --batch-size 128 --debug

    # 不使用多进程（原始方式，慢）
    python scripts/train.py --distance 3 --rounds 12 --num-workers 0
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from alphaqubit.data import SurfaceCodeDataset, CoordinateSystem, PrefetchDataLoader, NPZDataset, PTBatchDataset
from alphaqubit.models import AlphaQubitDecoder, AlphaQubitDecoderConfig
from alphaqubit.training import Trainer, TrainingConfig, train_decoder
from alphaqubit.evaluation.metrics import compute_error_rate, compute_fidelity, fit_ler, LERResult


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaQubit decoder")

    # 数据参数
    parser.add_argument("--distance", type=int, default=3, help="Code distance")
    parser.add_argument("--rounds", type=int, default=12, help="Number of error correction rounds")
    parser.add_argument("--p", type=float, default=0.005, help="Physical error rate")
    parser.add_argument("--snr", type=float, default=10.0, help="Soft readout SNR")

    # 训练参数
    parser.add_argument("--steps", type=int, default=30000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")

    # 模型参数
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["tiny", "small", "base", "scaling", "sycamore", "large"])

    # 其他
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer samples)")
    parser.add_argument("--use-soft", action="store_true", default=True,
                       help="Use soft readout")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples per epoch (default: 100000, debug: 10000)")
    parser.add_argument("--log-interval", type=int, default=None,
                       help="Log every N steps (default: 100, debug: 10)")
    parser.add_argument("--eval-interval", type=int, default=None,
                       help="Evaluate every N steps (default: 500, debug: 100)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of data generation workers (0=single thread, 8+ recommended for 9950X3D)")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Path to pre-generated .npz file or .pt batch directory (skips online generation)")

    return parser.parse_args()


def train_with_prefetch(args, model, coord_system, save_dir, num_samples):
    """使用多进程预取的训练循环"""
    import time
    from torch.amp import autocast, GradScaler
    from torch.optim import AdamW
    from alphaqubit.training.scheduler import WarmupCosineScheduler
    from alphaqubit.training.losses import DecoderLoss

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.steps, min_lr_ratio=0.1)
    loss_fn = DecoderLoss()
    scaler = GradScaler()

    # 创建多进程数据加载器
    print(f"  Using {args.num_workers} worker processes for data generation")
    train_loader = PrefetchDataLoader(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        use_soft_readout=args.use_soft,
        snr=args.snr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.steps,  # 每个batch是一个step
        seed=args.seed,
        device=str(device),
    )

    # 验证数据加载器（使用更多workers加速）
    val_loader = PrefetchDataLoader(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        use_soft_readout=args.use_soft,
        snr=args.snr,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,  # 使用同样多的workers
        num_batches=20,  # 减少验证batch数（20个足够评估了）
        seed=args.seed + 1000,
        device=str(device),
    )

    log_interval = args.log_interval if args.log_interval else 10
    eval_interval = args.eval_interval if args.eval_interval else 100

    # 启动数据生成
    train_loader.start()

    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'learning_rate': []}
    best_val_loss = float('inf')
    start_time = time.time()

    # 创建进度条
    if HAS_TQDM:
        pbar = tqdm(total=args.steps, desc="Training", unit="step", dynamic_ncols=True)
    else:
        pbar = None

    try:
        for step, batch in enumerate(train_loader):
            model.train()

            # 移动数据到GPU
            measurement = batch['measurement'].to(device, non_blocking=True)
            event = batch['event'].to(device, non_blocking=True)
            leakage = batch['leakage'].to(device, non_blocking=True)
            event_leakage = batch['event_leakage'].to(device, non_blocking=True)
            final_soft = batch['final_soft'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # 前向传播（混合精度）
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(measurement, event, leakage, event_leakage, final_soft)
                loss, metrics = loss_fn(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 记录
            history['train_loss'].append(loss.item())
            history['train_acc'].append(metrics['accuracy'])
            history['learning_rate'].append(scheduler.get_last_lr()[0])

            # 更新进度条
            if pbar:
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{metrics['accuracy']:.2%}"})

            # 日志
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                log_msg = (f"loss: {loss.item():.4f} | accuracy: {metrics['accuracy']:.2%} | "
                          f"pos_rate: {metrics['pos_rate']:.2%} | pred_pos_rate: {metrics['pred_pos_rate']:.2%} | "
                          f"logit_mean: {metrics['logit_mean']:.4f} | logit_std: {metrics['logit_std']:.4f} | "
                          f"grad_norm: {grad_norm.item():.4f} | step: {step} | "
                          f"lr: {scheduler.get_last_lr()[0]:.4f} | elapsed: {elapsed:.2f}")
                if pbar:
                    tqdm.write(log_msg)
                else:
                    print(log_msg)

            # 验证
            if step > 0 and step % eval_interval == 0:
                model.eval()
                val_loader.start()
                val_losses = []
                val_accs = []

                with torch.no_grad():
                    for val_batch in val_loader:
                        m = val_batch['measurement'].to(device, non_blocking=True)
                        e = val_batch['event'].to(device, non_blocking=True)
                        l = val_batch['leakage'].to(device, non_blocking=True)
                        el = val_batch['event_leakage'].to(device, non_blocking=True)
                        fs = val_batch['final_soft'].to(device, non_blocking=True)
                        lb = val_batch['label'].to(device, non_blocking=True)

                        with autocast(device_type='cuda', dtype=torch.float16):
                            logits = model(m, e, l, el, fs)
                            vloss, vmetrics = loss_fn(logits, lb)
                        val_losses.append(vloss.item())
                        val_accs.append(vmetrics['accuracy'])

                val_loader.shutdown()

                val_loss = sum(val_losses) / len(val_losses)
                val_acc = sum(val_accs) / len(val_accs)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                val_msg = f"[Eval] step: {step} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2%}"
                if pbar:
                    tqdm.write(val_msg)
                else:
                    print(val_msg)

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = save_dir / 'best.pt'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'step': step,
                        'val_loss': val_loss,
                    }, save_path)
                    if pbar:
                        tqdm.write(f"[Save] Best model saved, val_loss: {val_loss:.4f}")

    finally:
        if pbar:
            pbar.close()
        train_loader.shutdown()

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_dir / 'final.pt')

    return history


def main():
    args = parse_args()

    # 创建数据集
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = 10000 if args.debug else 100000

    print("="*60)
    print("AlphaQubit Training")
    print("="*60)
    print(f"Distance: {args.distance}")
    print(f"Rounds: {args.rounds}")
    print(f"Physical error rate: {args.p}")
    print(f"Model size: {args.model_size}")
    print(f"Total steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples per epoch: {num_samples:,}")
    print(f"Data workers: {args.num_workers}")
    print("="*60)

    # 设置随机种子
    torch.manual_seed(args.seed)

    # ==================== Disk Dataset Mode ====================
    if args.data_file:
        data_path = Path(args.data_file)
        if data_path.is_dir():
            print(f"\n[1/3] Loading pre-generated batch dataset from {data_path}...")
            dataset = PTBatchDataset(str(data_path))
        else:
            print(f"\n[1/3] Loading pre-generated .npz dataset from {data_path}...")
            dataset = NPZDataset(str(data_path))
        coord_system = dataset.coord_system

        print(f"  Samples: {len(dataset):,}")
        print(f"  Distance: {dataset.distance}")
        print(f"  Rounds: {dataset.rounds}")
        print(f"  p: {dataset.p}")

        # 验证数据
        print("\n[2/3] Validating dataset...")
        metrics = dataset.validate()
        print(f"  Event density: {metrics['event_density']:.4f}")
        print(f"  Label flip rate: {metrics['label_flip_rate']:.4f}")
        print(f"  Scatter/Gather valid: {metrics['scatter_gather_valid']}")

        # 创建模型
        print("\n[3/3] Creating model...")
        if args.model_size == "tiny":
            model = AlphaQubitDecoderConfig.tiny(coord_system)
        elif args.model_size == "small":
            model = AlphaQubitDecoderConfig.small(coord_system)
        elif args.model_size == "base":
            model = AlphaQubitDecoderConfig.base(coord_system)
        elif args.model_size == "scaling":
            model = AlphaQubitDecoderConfig.scaling(coord_system)
        else:
            model = AlphaQubitDecoderConfig.sycamore(coord_system)

        param_stats = model.get_num_parameters()
        print(f"  Total parameters: {param_stats['total']:,}")

        # 创建DataLoader
        if isinstance(dataset, PTBatchDataset):
            # Each item is already a batch — use batch_size=None
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            total_batches = len(dataset)
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )
            total_batches = len(dataset) // args.batch_size

        # 计算总steps对应的epochs
        epochs_needed = (args.steps + total_batches - 1) // total_batches
        print(f"\n  Total batches per epoch: {total_batches:,}")
        print(f"  Epochs needed: {epochs_needed} (for {args.steps} steps)")

        # 保存目录
        save_dir = Path(args.save_dir) / f"d{args.distance}_{args.model_size}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_interval = args.log_interval if args.log_interval else 250
        eval_interval = args.eval_interval if args.eval_interval else 1000
        print(f"\n  Log interval: {log_interval} steps")
        print(f"  Eval interval: {eval_interval} steps")

        # 训练（使用循环DataLoader）
        print("\n[Training] Starting from disk dataset...\n")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        from torch.optim import AdamW
        from alphaqubit.training.scheduler import WarmupCosineScheduler
        from alphaqubit.training.losses import DecoderLoss
        from torch.amp import autocast, GradScaler
        import time
        try:
            from tqdm import tqdm; HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.steps, min_lr_ratio=0.1)
        loss_fn = DecoderLoss()
        scaler = GradScaler()

        # 创建验证集（从在线数据生成一定量）
        val_dataset = SurfaceCodeDataset(
            distance=args.distance, rounds=args.rounds, p=args.p,
            use_soft_readout=args.use_soft, snr=args.snr,
            num_samples=min(100000, len(dataset) // 10), seed=args.seed + 1000,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size * 2,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        best_val_loss = float('inf')
        step = 0
        start_time = time.time()
        pbar = tqdm(total=args.steps, desc="Training", unit="step", dynamic_ncols=True) if HAS_TQDM else None

        try:
            while step < args.steps:
                for batch in train_loader:
                    if step >= args.steps:
                        break
                    model.train()

                    measurement = batch['measurement'].to(device, non_blocking=True)
                    event = batch['event'].to(device, non_blocking=True)
                    leakage = batch['leakage'].to(device, non_blocking=True)
                    event_leakage = batch['event_leakage'].to(device, non_blocking=True)
                    final_soft = batch['final_soft'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = model(measurement, event, leakage, event_leakage, final_soft)
                        loss, metrics = loss_fn(logits, labels)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{metrics['accuracy']:.2%}"})

                    if step % log_interval == 0:
                        elapsed = time.time() - start_time
                        tqdm.write(
                            f"loss: {loss.item():.4f} | accuracy: {metrics['accuracy']:.2%} | "
                            f"pos_rate: {metrics['pos_rate']:.2%} | pred_pos_rate: {metrics['pred_pos_rate']:.2%} | "
                            f"logit_mean: {metrics['logit_mean']:.4f} | logit_std: {metrics['logit_std']:.4f} | "
                            f"grad_norm: {grad_norm.item():.4f} | step: {step} | "
                            f"lr: {scheduler.get_last_lr()[0]:.6f} | elapsed: {elapsed:.1f}s"
                        )

                    if step > 0 and step % eval_interval == 0:
                        model.eval()
                        val_losses, val_accs = [], []
                        with torch.no_grad():
                            for val_batch in val_loader:
                                m = val_batch['measurement'].to(device, non_blocking=True)
                                e = val_batch['event'].to(device, non_blocking=True)
                                l = val_batch['leakage'].to(device, non_blocking=True)
                                el = val_batch['event_leakage'].to(device, non_blocking=True)
                                fs = val_batch['final_soft'].to(device, non_blocking=True)
                                lb = val_batch['label'].to(device, non_blocking=True)
                                with autocast(device_type='cuda', dtype=torch.float16):
                                    vlogits = model(m, e, l, el, fs)
                                    vloss, vmetrics = loss_fn(vlogits, lb)
                                val_losses.append(vloss.item())
                                val_accs.append(vmetrics['accuracy'])
                        val_loss = sum(val_losses) / len(val_losses)
                        val_acc = sum(val_accs) / len(val_accs)
                        tqdm.write(f"[Eval] step: {step} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2%}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'val_loss': val_loss},
                                      save_dir / 'best.pt')
                            tqdm.write(f"[Save] Best model saved, val_loss: {val_loss:.4f}")
                        model.train()

                    step += 1
        finally:
            if pbar: pbar.close()
            torch.save({'model_state_dict': model.state_dict(), 'step': step}, save_dir / 'final.pt')

        print(f"\n{'='*60}")
        print(f"Training completed! Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print(f"{'='*60}")

    # 使用多进程数据生成
    elif args.num_workers > 0:
        print("\n[1/4] Setting up multi-process data generation...")

        # 创建临时数据集获取坐标系统
        temp_dataset = SurfaceCodeDataset(
            distance=args.distance,
            rounds=args.rounds,
            p=args.p,
            use_soft_readout=args.use_soft,
            snr=args.snr,
            num_samples=1000,
            seed=args.seed,
        )

        # 验证数据
        print("\n[2/4] Validating data generation...")
        metrics = temp_dataset.validate()
        print(f"  Event density: {metrics['event_density']:.4f}")
        print(f"  Scatter/Gather valid: {metrics['scatter_gather_valid']}")

        coord_system = temp_dataset.coord_system

        # 创建模型
        print("\n[3/4] Creating model...")
        if args.model_size == "tiny":
            model = AlphaQubitDecoderConfig.tiny(coord_system)
        elif args.model_size == "small":
            model = AlphaQubitDecoderConfig.small(coord_system)
        elif args.model_size == "base":
            model = AlphaQubitDecoderConfig.base(coord_system)
        elif args.model_size == "scaling":
            model = AlphaQubitDecoderConfig.scaling(coord_system)
        else:
            # sycamore / large
            model = AlphaQubitDecoderConfig.sycamore(coord_system)

        param_stats = model.get_num_parameters()
        print(f"  Total parameters: {param_stats['total']:,}")

        # 保存目录
        save_dir = Path(args.save_dir) / f"d{args.distance}_{args.model_size}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n[4/4] Starting training with multi-process data generation...")
        log_interval = args.log_interval if args.log_interval else 10
        eval_interval = args.eval_interval if args.eval_interval else 100
        print(f"  Log interval: {log_interval} steps")
        print(f"  Eval interval: {eval_interval} steps")

        history = train_with_prefetch(args, model, coord_system, save_dir, num_samples)

    else:
        # 原始单线程模式
        print("\n[1/4] Creating datasets (single-thread mode)...")
        train_dataset = SurfaceCodeDataset(
            distance=args.distance,
            rounds=args.rounds,
            p=args.p,
            use_soft_readout=args.use_soft,
            snr=args.snr,
            num_samples=num_samples,
            seed=args.seed,
        )

        val_dataset = SurfaceCodeDataset(
            distance=args.distance,
            rounds=args.rounds,
            p=args.p,
            use_soft_readout=args.use_soft,
            snr=args.snr,
            num_samples=num_samples // 10,
            seed=args.seed + 1000,
        )

        # 验证数据
        print("\n[2/4] Validating data generation...")
        metrics = train_dataset.validate()
        print(f"  Event density: {metrics['event_density']:.4f}")
        print(f"  Scatter/Gather valid: {metrics['scatter_gather_valid']}")

        # 创建模型
        print("\n[3/4] Creating model...")
        coord_system = train_dataset.coord_system

        if args.model_size == "tiny":
            model = AlphaQubitDecoderConfig.tiny(coord_system)
        elif args.model_size == "small":
            model = AlphaQubitDecoderConfig.small(coord_system)
        elif args.model_size == "base":
            model = AlphaQubitDecoderConfig.base(coord_system)
        elif args.model_size == "scaling":
            model = AlphaQubitDecoderConfig.scaling(coord_system)
        else:
            # sycamore / large
            model = AlphaQubitDecoderConfig.sycamore(coord_system)

        param_stats = model.get_num_parameters()
        print(f"  Total parameters: {param_stats['total']:,}")

        # 配置训练
        log_interval = args.log_interval if args.log_interval else 10
        eval_interval = args.eval_interval if args.eval_interval else 500

        print("\n[4/4] Starting training...")
        print(f"  Log interval: {log_interval} steps")
        print(f"  Eval interval: {eval_interval} steps")

        config = TrainingConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=args.warmup,
            eval_interval=eval_interval,
            log_interval=log_interval,
            seed=args.seed,
        )

        # 保存目录
        save_dir = Path(args.save_dir) / f"d{args.distance}_{args.model_size}"

        # 训练
        history = train_decoder(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            save_dir=str(save_dir),
        )

    print("\n" + "="*60)
    print("Training completed!")
    if history['val_loss']:
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
    elif history['train_loss']:
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("="*60)

    # 训练后自动评估
    print("\n" + "="*60)
    print("Post-Training Evaluation")
    print("="*60)

    best_checkpoint = save_dir / 'best.pt'
    if best_checkpoint.exists():
        post_training_evaluation(
            checkpoint_path=str(best_checkpoint),
            distance=args.distance,
            model_size=args.model_size,
            p=args.p,
            train_rounds=args.rounds,  # 使用训练时的rounds
            num_workers=args.num_workers if args.num_workers > 0 else 4,
        )
    else:
        print("No best checkpoint found, skipping evaluation.")


def post_training_evaluation(
    checkpoint_path: str,
    distance: int,
    model_size: str,
    p: float = 0.005,
    train_rounds: int = 25,
    num_workers: int = 8,
    num_samples: int = 10000,
    eval_rounds: list = None,
):
    """训练后自动评估，输出与MWPM对比的表格

    评估rounds会根据训练rounds自动生成，覆盖从小到训练rounds的范围。
    """
    from torch.amp import autocast
    import numpy as np

    # 根据训练rounds自动生成评估rounds列表
    if eval_rounds is None:
        # 生成合理的评估点：从小rounds到训练rounds
        if train_rounds <= 15:
            eval_rounds = [3, 5, 7, 9, 11, train_rounds] if train_rounds > 11 else [3, 5, 7, 9, train_rounds]
        elif train_rounds <= 30:
            eval_rounds = [3, 7, 11, 17, train_rounds]
        elif train_rounds <= 60:
            eval_rounds = [5, 11, 21, 31, train_rounds]
        else:
            # 对于很长的rounds，按比例取点
            step = train_rounds // 5
            eval_rounds = [step, step*2, step*3, step*4, train_rounds]

        # 确保不重复且排序
        eval_rounds = sorted(list(set(eval_rounds)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("\n[1/3] Loading trained model...")
    temp_dataset = SurfaceCodeDataset(distance=distance, rounds=12, p=p, num_samples=100)
    coord_system = temp_dataset.coord_system

    if model_size == "tiny":
        model = AlphaQubitDecoderConfig.tiny(coord_system)
    elif model_size == "small":
        model = AlphaQubitDecoderConfig.small(coord_system)
    elif model_size == "base":
        model = AlphaQubitDecoderConfig.base(coord_system)
    else:
        model = AlphaQubitDecoderConfig.large(coord_system)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 评估AlphaQubit
    print("\n[2/3] Evaluating AlphaQubit...")
    model_results = {}

    for rounds in eval_rounds:
        loader = PrefetchDataLoader(
            distance=distance, rounds=rounds, p=p,
            batch_size=min(2048, num_samples),
            num_workers=num_workers,
            num_batches=(num_samples + 2047) // 2048,
            seed=42 + rounds,
            device=str(device),
        )
        loader.start()

        all_preds, all_labels = [], []
        try:
            for batch in loader:
                m = batch['measurement'].to(device, non_blocking=True)
                e = batch['event'].to(device, non_blocking=True)
                l = batch['leakage'].to(device, non_blocking=True)
                el = batch['event_leakage'].to(device, non_blocking=True)
                fs = batch['final_soft'].to(device, non_blocking=True)
                lb = batch['label'].to(device, non_blocking=True)

                with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                    preds, _ = model.predict(m, e, l, el, fs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(lb.cpu().numpy().flatten())
        finally:
            loader.shutdown()

        preds = np.concatenate(all_preds)[:num_samples]
        labels = np.concatenate(all_labels)[:num_samples]
        error_rate = compute_error_rate(preds, labels)
        model_results[rounds] = error_rate

    # 评估MWPM
    print("\n[3/3] Evaluating MWPM baseline...")
    try:
        from alphaqubit.experiments.baselines import MWPMBaseline
        mwpm = MWPMBaseline()
        mwpm_results = {}
        for rounds in eval_rounds:
            result = mwpm.evaluate_single(distance, rounds, p, num_samples)
            mwpm_results[rounds] = result.error_rate
        has_mwpm = True
    except Exception as e:
        print(f"  MWPM evaluation failed: {e}")
        mwpm_results = {r: float('nan') for r in eval_rounds}
        has_mwpm = False

    # 计算LER
    model_fids = [compute_fidelity(model_results[r]) for r in eval_rounds]
    model_ler, model_r2, _, _, model_valid = fit_ler(eval_rounds, model_fids)

    if has_mwpm:
        mwpm_fids = [compute_fidelity(mwpm_results[r]) for r in eval_rounds]
        mwpm_ler, mwpm_r2, _, _, mwpm_valid = fit_ler(eval_rounds, mwpm_fids)
    else:
        mwpm_ler, mwpm_r2, mwpm_valid = float('nan'), float('nan'), False

    # 打印结果表格
    print("\n" + "="*70)
    print(f"  EVALUATION RESULTS - Distance d={distance}, p={p}")
    print("="*70)

    print(f"\n  {'Rounds':<8} {'AlphaQubit':<15} {'MWPM':<15} {'Improvement':<12}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*12}")

    for r in eval_rounds:
        model_e = model_results[r]
        mwpm_e = mwpm_results[r]
        if has_mwpm and mwpm_e > 0:
            imp = (mwpm_e - model_e) / mwpm_e * 100
            imp_str = f"{imp:+.1f}%"
        else:
            imp_str = "N/A"
        print(f"  {r:<8} {model_e:<15.4f} {mwpm_e:<15.4f} {imp_str:<12}")

    print("\n" + "-"*70)
    print("  LER (Logical Error Rate per Round)")
    print("-"*70)

    print(f"\n  {'Method':<15} {'LER':<15} {'R^2':<10} {'Valid':<8}")
    print(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*8}")

    model_v = "Yes" if model_valid else "No"
    mwpm_v = "Yes" if mwpm_valid else "No"
    print(f"  {'AlphaQubit':<15} {model_ler:<15.6f} {model_r2:<10.4f} {model_v:<8}")
    if has_mwpm:
        print(f"  {'MWPM':<15} {mwpm_ler:<15.6f} {mwpm_r2:<10.4f} {mwpm_v:<8}")

    # Summary
    if has_mwpm and mwpm_ler > 0 and model_ler > 0:
        ler_imp = (mwpm_ler - model_ler) / mwpm_ler * 100
        ler_ratio = mwpm_ler / model_ler

        print("\n" + "-"*70)
        print("  SUMMARY")
        print("-"*70)
        print(f"\n  LER Improvement: {ler_imp:+.1f}%")
        print(f"  LER Ratio (MWPM/AlphaQubit): {ler_ratio:.2f}x")

        if ler_imp > 0:
            print(f"\n  >>> AlphaQubit achieves {ler_imp:.1f}% LOWER error rate than MWPM!")
        else:
            print(f"\n  >>> MWPM performs better by {-ler_imp:.1f}% (need more training)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
