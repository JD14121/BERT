"""
对比实验脚本 - scripts/compare_baseline.py

这个脚本运行完整的对比实验：
1. AlphaQubit 基准（from scratch，全监督）
2. MWPM 基准（使用 PyMatching）
3. BERT 预训练 + 部分标签微调（Pretrain → Finetune）

使用预生成的 .pt 数据集，并添加 LER（Logical Error per Round）指标。
"""

import argparse
import os
import sys
from pathlib import Path

# 禁用 tqdm 进度条，避免在 Windows 重定向日志时产生大量 I/O 开销
os.environ['TQDM_DISABLE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from alphaqubit.data.npz_dataset import NPZDataset
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.decoder import AlphaQubitDecoder
from alphaqubit.models.pretrain_decoder import PretrainDecoder, FineTuneDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig
from alphaqubit.evaluation.metrics import compute_ler, LERResult


def load_dataset(path: str):
    """根据扩展名加载数据集"""
    if path.endswith('.pt'):
        return PTBatchDataset(path)
    else:
        return NPZDataset(path, mmap_mode=None)


def parse_args():
    parser = argparse.ArgumentParser(description='对比实验: AlphaQubit vs MWPM vs BERT Pretrain+Finetune')

    parser.add_argument('--train_npz', type=str, default='data_baseline/train_d3_r25.npz')
    parser.add_argument('--val_npz', type=str, default='data_baseline/val_d3_r25.npz')
    parser.add_argument('--test_npz', type=str, default='data_baseline/test_d3_r25.npz')

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--num_readout_layers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alphaqubit_steps', type=int, default=10000)
    parser.add_argument('--pretrain_steps', type=int, default=10000)
    parser.add_argument('--finetune_steps', type=int, default=3000)
    parser.add_argument('--finetune_subset_ratio', type=float, default=0.1,
                        help='用于 BERT 微调的有标签数据比例（相对训练集）')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints/comparison')

    # LER 配置
    parser.add_argument('--ler_rounds', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 25])
    parser.add_argument('--ler_samples', type=int, default=2000,
                        help='每个 rounds 用于 LER 评估的样本数')
    parser.add_argument('--skip_ler', action='store_true', help='跳过 LER 评估')

    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'pretrain', 'baseline', 'eval'],
                        help='实验阶段：pretrain 先跑 BERT 预训练；baseline 跑 AlphaQubit；eval 跑 BERT 微调+MWPM+LER')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='eval 阶段使用的 BERT 预训练检查点路径（默认 save_dir/bert_pretrain/best.pt）')

    return parser.parse_args()


def evaluate_model(model, dataset, device, batch_size=1024):
    """评估模型在数据集上的准确率"""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            measurement = batch['measurement'].to(device)
            event = batch['event'].to(device)
            leakage = batch['leakage'].to(device)
            event_leakage = batch['event_leakage'].to(device)
            final_soft = batch['final_soft'].to(device)
            labels = batch['label'].to(device)

            logits = model(measurement, event, leakage, event_leakage, final_soft)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            batch_size_actual = measurement.size(0)
            total_correct += (preds == labels).float().sum().item()
            total_samples += batch_size_actual
            total_loss += loss.item() * batch_size_actual

    return {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples,
    }


def evaluate_model_at_rounds(model, eval_datasets, device, batch_size=1024):
    """在多个 rounds 上评估模型，返回每个 rounds 的预测和标签"""
    from torch.utils.data import DataLoader

    predictions_by_rounds = {}
    labels_by_rounds = {}

    model.eval()
    for rounds, dataset in eval_datasets.items():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                measurement = batch['measurement'].to(device)
                event = batch['event'].to(device)
                leakage = batch['leakage'].to(device)
                event_leakage = batch['event_leakage'].to(device)
                final_soft = batch['final_soft'].to(device)
                labels = batch['label'].to(device)

                # 截断到指定 rounds
                measurement = measurement[:, :rounds, :]
                event = event[:, :rounds, :]
                leakage = leakage[:, :rounds, :]
                event_leakage = event_leakage[:, :rounds, :]

                logits = model(measurement, event, leakage, event_leakage, final_soft, n_rounds=rounds)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        predictions_by_rounds[rounds] = np.concatenate(all_preds).flatten()
        labels_by_rounds[rounds] = np.concatenate(all_labels).flatten()

    return predictions_by_rounds, labels_by_rounds


def create_subset_dataset(dataset, ratio: float, seed: int = 42):
    """创建数据集的有标签子集"""
    from torch.utils.data import Subset
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.choice(n, size=int(n * ratio), replace=False)
    return Subset(dataset, indices)


def train_alphaqubit_baseline(train_ds, val_ds, test_ds, args):
    """训练 AlphaQubit 基准模型"""
    print("\n" + "=" * 60)
    print("[1/4] AlphaQubit 基准模型（from scratch）")
    print("=" * 60)

    model = AlphaQubitDecoder(
        coord_system=train_ds.coord_system,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        num_readout_layers=args.num_readout_layers,
        dropout=0.1,
        use_late_fusion=True,
    )

    config = TrainingConfig(
        total_steps=args.alphaqubit_steps,
        batch_size=args.batch_size,
        eval_interval=1000,
        log_interval=500,
        learning_rate=2e-4,
        device=args.device,
        use_amp=True,
        early_stopping_patience=10000,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_dir=f"{args.save_dir}/alphaqubit_baseline",
    )

    trainer.train()

    test_metrics = evaluate_model(model, test_ds, args.device)
    print(f"\n[AlphaQubit Baseline] Test Accuracy: {test_metrics['accuracy']:.4f}, Test Loss: {test_metrics['loss']:.4f}")

    return model, test_metrics


def train_bert_pretrain(train_ds, val_ds, args):
    """BERT 预训练（无标签，mask 预测）"""
    print("\n" + "=" * 60)
    print("[2/4] BERT 预训练模型")
    print("=" * 60)

    from alphaqubit.data.pt_dataset import PTBatchDataset

    model = PretrainDecoder(
        coord_system=train_ds.coord_system,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        dropout=0.1,
    )

    def to_pt_path(npz_path: str) -> str:
        if npz_path.endswith('.pt'):
            return npz_path
        pt_path = npz_path.replace('.npz', '.pt')
        if not Path(pt_path).exists():
            raise FileNotFoundError(
                f"预训练需要 .pt 格式数据，找不到: {pt_path}\n"
                f"请先用 scripts/convert_npz_to_pt.py 将 {npz_path} 转换为 .pt"
            )
        return pt_path

    pretrain_train_ds = PTBatchDataset(to_pt_path(args.train_npz))
    pretrain_val_ds = PTBatchDataset(to_pt_path(args.val_npz))

    config = PretrainConfig(
        total_steps=args.pretrain_steps,
        batch_size=args.batch_size,
        eval_interval=1000,
        log_interval=500,
        learning_rate=2e-4,
        mask_ratio=0.15,
        device=args.device,
        use_amp=True,
        early_stopping_patience=10000,
    )

    trainer = PretrainTrainer(
        model=model,
        train_dataset=pretrain_train_ds,
        val_dataset=pretrain_val_ds,
        config=config,
        save_dir=f"{args.save_dir}/bert_pretrain",
    )

    trainer.train()

    val_metrics = trainer.evaluate()
    print(f"\n[BERT Pretrain] Mask Val Accuracy: {val_metrics['mask_accuracy']:.4f}, Val Loss: {val_metrics['loss']:.4f}")

    # 额外保存一份纯 encoder 权重，便于 eval 阶段加载
    encoder_path = Path(f"{args.save_dir}/bert_pretrain/pretrain_encoder.pt")
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.get_encoder_state_dict(), encoder_path)
    print(f"[BERT Pretrain] Encoder weights saved to {encoder_path}")

    return model, val_metrics


def train_bert_finetune(pretrain_model, train_ds, val_ds, test_ds, args):
    """BERT 部分标签微调"""
    print("\n" + "=" * 60)
    print("[3/4] BERT 预训练 + 部分标签微调")
    print("=" * 60)

    finetune_train_ds = create_subset_dataset(train_ds, args.finetune_subset_ratio, seed=args.seed)
    print(f"微调使用 {len(finetune_train_ds)} / {len(train_ds)} 有标签样本 ({args.finetune_subset_ratio*100:.0f}%)")

    model = FineTuneDecoder(
        coord_system=train_ds.coord_system,
        pretrained_encoder=pretrain_model,
        embed_dim=args.embed_dim,
        readout_dim=64,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        num_readout_layers=args.num_readout_layers,
        dropout=0.1,
    )

    config = TrainingConfig(
        total_steps=args.finetune_steps,
        batch_size=args.batch_size,
        eval_interval=500,
        log_interval=200,
        learning_rate=1e-4,  # 微调通常使用稍小学习率
        device=args.device,
        use_amp=True,
        early_stopping_patience=10000,
    )

    trainer = Trainer(
        model=model,
        train_dataset=finetune_train_ds,
        val_dataset=val_ds,
        config=config,
        save_dir=f"{args.save_dir}/bert_finetune",
    )

    trainer.train()

    test_metrics = evaluate_model(model, test_ds, args.device)
    print(f"\n[BERT Finetune] Test Accuracy: {test_metrics['accuracy']:.4f}, Test Loss: {test_metrics['loss']:.4f}")

    return model, test_metrics


def evaluate_mwpm(test_ds, args):
    """评估 MWPM 基准"""
    print("\n" + "=" * 60)
    print("[4/4] MWPM 基准（PyMatching）")
    print("=" * 60)

    try:
        import pymatching
    except ImportError:
        print("PyMatching 未安装，跳过 MWPM 评估")
        print("安装命令: pip install pymatching")
        return None

    from alphaqubit.data.stim_generator import StimDataGenerator
    gen = StimDataGenerator(distance=test_ds.distance, rounds=test_ds.rounds, p=test_ds.p, seed=args.seed)
    dem = gen.circuit.detector_error_model(decompose_errors=True)
    mwpm = pymatching.Matching.from_detector_error_model(dem)

    num_test = len(test_ds)
    det_events = test_ds._data['detection_events'][:num_test].reshape(num_test, -1)
    labels = test_ds._data['label'][:num_test].int().numpy()

    predictions = mwpm.decode_batch(det_events.numpy())
    predictions = predictions.flatten()

    accuracy = np.mean(predictions == labels)
    print(f"\n[MWPM] Test Accuracy: {accuracy:.4f}")

    return {'accuracy': accuracy, 'predictions': predictions, 'labels': labels}


def evaluate_mwpm_at_rounds(eval_datasets, args):
    """在多个 rounds 上评估 MWPM"""
    try:
        import pymatching
    except ImportError:
        return None, None

    from alphaqubit.data.stim_generator import StimDataGenerator

    predictions_by_rounds = {}
    labels_by_rounds = {}

    for rounds, dataset in eval_datasets.items():
        gen = StimDataGenerator(distance=dataset.distance, rounds=rounds, p=dataset.p, seed=args.seed)
        dem = gen.circuit.detector_error_model(decompose_errors=True)
        mwpm = pymatching.Matching.from_detector_error_model(dem)

        n = len(dataset)
        det_events = dataset._data['detection_events'][:n].reshape(n, -1)
        labels = dataset._data['label'][:n].int().numpy()

        preds = mwpm.decode_batch(det_events.numpy()).flatten()
        predictions_by_rounds[rounds] = preds
        labels_by_rounds[rounds] = labels

    return predictions_by_rounds, labels_by_rounds


def generate_ler_eval_datasets(distance, rounds_list, p, snr, num_samples, seed, output_dir):
    """生成用于 LER 评估的多 rounds 小数据集（.pt 格式）"""
    from alphaqubit.data.stim_generator import StimDataGenerator
    from alphaqubit.data.soft_readout import SoftReadoutSimulator
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    print(f"\n生成 LER 评估数据: rounds={rounds_list}, 每轮 {num_samples} 样本...")

    for rounds in rounds_list:
        generator = StimDataGenerator(distance=distance, rounds=rounds, p=p, seed=seed + rounds)
        soft_sim = SoftReadoutSimulator(snr=snr, t=0.01)

        raw_measurements, dem_detection_events, stim_observables = generator.sample_with_matched_detectors(num_samples)
        ancilla_meas = generator.extract_ancilla_measurements(raw_measurements)
        final_data = generator.extract_final_data(raw_measurements)

        soft_meas = soft_sim.simulate(ancilla_meas)
        shots, T, n_stab = soft_meas.shape
        soft_events = np.zeros_like(soft_meas)
        soft_events[:, 0, :] = soft_meas[:, 0, :]
        for t in range(1, T):
            soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                soft_meas[:, t, :], soft_meas[:, t - 1, :]
            )
        soft_final = soft_sim.simulate(final_data.astype(bool))

        pt_path = output_dir / f"ler_d{distance}_r{rounds}_n{num_samples}.pt"
        torch.save({
            'measurement': torch.from_numpy(soft_meas.astype(np.float32)),
            'event': torch.from_numpy(soft_events.astype(np.float32)),
            'leakage': torch.zeros(shots, T, n_stab, dtype=torch.float32),
            'event_leakage': torch.zeros(shots, T, n_stab, dtype=torch.float32),
            'final_soft': torch.from_numpy(soft_final.astype(np.float32)),
            'label': torch.from_numpy(stim_observables.astype(np.float32)),
            'detection_events': torch.from_numpy(dem_detection_events.astype(np.float32)),
            'distance': distance,
            'rounds': rounds,
            'p': p,
            'snr': snr,
        }, pt_path)

        datasets[rounds] = PTBatchDataset(str(pt_path))
        print(f"  rounds={rounds}: saved {pt_path}")

    return datasets


def print_ler_result(name: str, result: LERResult):
    """打印 LER 结果"""
    print(f"\n[{name}] LER Results:")
    print(f"  LER: {result.ler:.6f}")
    print(f"  R²:  {result.r_squared:.4f}")
    print(f"  log(F₀): {result.log_f0:.4f}")
    print(f"  Valid: {result.is_valid}")
    if result.error_rates:
        print("  Error rates by rounds:")
        for n in sorted(result.error_rates.keys()):
            print(f"    n={n:2d}: E={result.error_rates[n]:.4f}, F={result.fidelities[n]:.4f}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("对比实验: AlphaQubit vs MWPM vs BERT Pretrain+Finetune")
    print("=" * 60)
    print(f"配置: embed_dim={args.embed_dim}, n_heads={args.n_heads}, "
          f"layers={args.num_transformer_layers}, readout_layers={args.num_readout_layers}")
    print(f"训练步数: AlphaQubit={args.alphaqubit_steps}, "
          f"BERT pretrain={args.pretrain_steps}, BERT finetune={args.finetune_steps}")

    print("\n加载数据集...")
    train_ds = load_dataset(args.train_npz)
    val_ds = load_dataset(args.val_npz)
    test_ds = load_dataset(args.test_npz)

    print(f"训练集: {len(train_ds)} 样本")
    print(f"验证集: {len(val_ds)} 样本")
    print(f"测试集: {len(test_ds)} 样本")

    results = {}

    # 阶段 1：BERT 预训练（无标签，mask 预测）
    if args.stage in ('all', 'pretrain'):
        bert_pretrained, results['bert_pretrain'] = train_bert_pretrain(train_ds, val_ds, args)

    # 阶段 2：AlphaQubit 基准
    if args.stage in ('all', 'baseline'):
        alphaqubit_model, results['alphaqubit'] = train_alphaqubit_baseline(train_ds, val_ds, test_ds, args)

    # 阶段 3：BERT 微调 + MWPM + LER 评估
    if args.stage in ('all', 'eval'):
        if args.stage == 'eval':
            # eval 阶段需要加载预训练模型
            pretrain_ckpt = args.pretrain_checkpoint or f"{args.save_dir}/bert_pretrain/final.pt"
            print(f"\n[Eval] 加载 BERT 预训练检查点: {pretrain_ckpt}")
            bert_pretrained = PretrainDecoder(
                coord_system=train_ds.coord_system,
                embed_dim=args.embed_dim,
                n_heads=args.n_heads,
                num_transformer_layers=args.num_transformer_layers,
                dropout=0.1,
            )
            checkpoint = torch.load(pretrain_ckpt, map_location='cpu', weights_only=False)
            bert_pretrained.load_state_dict(checkpoint['model_state_dict'])
            bert_pretrained = bert_pretrained.to(args.device)
            print("[Eval] 预训练模型加载完成")

            # eval 阶段也需要 AlphaQubit 模型用于 LER 对比
            alphaqubit_model = AlphaQubitDecoder(
                coord_system=train_ds.coord_system,
                embed_dim=args.embed_dim,
                n_heads=args.n_heads,
                num_transformer_layers=args.num_transformer_layers,
                num_readout_layers=args.num_readout_layers,
                dropout=0.1,
                use_late_fusion=True,
            )
            alphaqubit_ckpt = f"{args.save_dir}/alphaqubit_baseline/final.pt"
            if Path(alphaqubit_ckpt).exists():
                checkpoint = torch.load(alphaqubit_ckpt, map_location='cpu', weights_only=False)
                alphaqubit_model.load_state_dict(checkpoint['model_state_dict'])
                alphaqubit_model = alphaqubit_model.to(args.device)
                print("[Eval] AlphaQubit 基准模型加载完成")
            else:
                print(f"[Eval] 警告：找不到 AlphaQubit 检查点 {alphaqubit_ckpt}，将跳过 AlphaQubit LER")
                alphaqubit_model = None

        bert_finetuned, results['bert_finetune'] = train_bert_finetune(bert_pretrained, train_ds, val_ds, test_ds, args)
        results['mwpm'] = evaluate_mwpm(test_ds, args)

        # 总结
        print("\n" + "=" * 60)
        print("对比结果总结")
        print("=" * 60)
        for name, metrics in results.items():
            if metrics is not None:
                print(f"{name:20s}: {metrics}")

        # LER 评估
        if not args.skip_ler:
            print("\n" + "=" * 60)
            print("LER (Logical Error per Round) 评估")
            print("=" * 60)

            ler_dir = f"{args.save_dir}/ler_eval_data"
            ler_datasets = generate_ler_eval_datasets(
                distance=test_ds.distance,
                rounds_list=args.ler_rounds,
                p=test_ds.p,
                snr=test_ds.snr,
                num_samples=args.ler_samples,
                seed=args.seed,
                output_dir=ler_dir,
            )

            if alphaqubit_model is not None:
                aq_preds, aq_labels = evaluate_model_at_rounds(alphaqubit_model, ler_datasets, args.device)
                aq_ler = compute_ler(aq_preds, aq_labels)
                print_ler_result("AlphaQubit", aq_ler)
            else:
                aq_ler = None

            bert_preds, bert_labels = evaluate_model_at_rounds(bert_finetuned, ler_datasets, args.device)
            bert_ler = compute_ler(bert_preds, bert_labels)
            print_ler_result("BERT Finetune", bert_ler)

            mwpm_preds, mwpm_labels = evaluate_mwpm_at_rounds(ler_datasets, args)
            if mwpm_preds is not None:
                mwpm_ler = compute_ler(mwpm_preds, mwpm_labels)
                print_ler_result("MWPM", mwpm_ler)

            print("\n" + "=" * 60)
            print("LER 横向对比")
            print("=" * 60)
            if aq_ler is not None:
                print(f"AlphaQubit    LER: {aq_ler.ler:.6f} (R²={aq_ler.r_squared:.3f}, valid={aq_ler.is_valid})")
            print(f"BERT Finetune LER: {bert_ler.ler:.6f} (R²={bert_ler.r_squared:.3f}, valid={bert_ler.is_valid})")
            if mwpm_preds is not None:
                print(f"MWPM          LER: {mwpm_ler.ler:.6f} (R²={mwpm_ler.r_squared:.3f}, valid={mwpm_ler.is_valid})")


if __name__ == '__main__':
    main()
