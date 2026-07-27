"""
PAEMS 合成数据对比实验脚本 - paems_experiment/compare_baseline_paems.py

基于 scripts/compare_baseline.py 改造，专门用于 PAEMS-data 合成噪声数据集：
1. AlphaQubit 基准（from scratch，全监督）
2. MWPM 基准（使用 PAEMS 噪声模型构建 DEM，而非均匀 Stim 噪声）
3. BERT 预训练 + 部分标签微调（Pretrain → Finetune）
4. LER（Logical Error per Round）评估，直接使用 PAEMS-data 中预生成的 LER 扫描文件

所有修改隔离在 paems_experiment/ 目录下，原 compare_baseline.py 不受影响。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 禁用 tqdm 进度条，避免在 Windows 重定向日志时产生大量 I/O 开销
os.environ['TQDM_DISABLE'] = '1'

# 将项目根目录加入路径，以便导入 alphaqubit 包
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.decoder import AlphaQubitDecoder
from alphaqubit.models.pretrain_decoder import PretrainDecoder, FineTuneDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig
from alphaqubit.evaluation.metrics import compute_ler, LERResult


def load_dataset(path: str):
    """加载 .pt 格式数据集"""
    if not path.endswith('.pt'):
        raise ValueError(f"PAEMS 实验仅支持 .pt 格式数据，收到: {path}")
    return PTBatchDataset(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PAEMS 合成数据对比实验: AlphaQubit vs MWPM vs BERT Pretrain+Finetune'
    )

    parser.add_argument('--train_pt', type=str, default='PAEMS-data/v1/train_d3_r25_n50000.pt')
    parser.add_argument('--val_pt', type=str, default='PAEMS-data/v1/val_d3_r25_n10000.pt')
    parser.add_argument('--test_pt', type=str, default='PAEMS-data/v1/test_d3_r25_n10000.pt')
    parser.add_argument('--paems_data_dir', type=str, default='PAEMS-data/v1',
                        help='PAEMS-data 根目录，用于定位 LER 扫描文件与噪声模型参数')

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--num_readout_layers', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alphaqubit_steps', type=int, default=20000)
    parser.add_argument('--pretrain_steps', type=int, default=20000)
    parser.add_argument('--finetune_steps', type=int, default=5000)
    parser.add_argument('--finetune_subset_ratio', type=float, default=0.1,
                        help='用于 BERT 微调的有标签数据比例（相对训练集）')
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='BERT 预训练时的掩码比例')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints/paems_experiment')

    # LER 配置：直接使用 PAEMS-data 中预生成的 LER 扫描文件
    parser.add_argument('--ler_rounds', type=int, nargs='+',
                        default=[3, 6, 9, 12, 15, 18, 21, 25],
                        help='用于 LER 评估的 rounds 序列')
    parser.add_argument('--ler_samples', type=int, default=2000,
                        help='每个 rounds 用于 LER 评估的样本数，需与 PAEMS-data 文件名一致')
    parser.add_argument('--skip_ler', action='store_true', help='跳过 LER 评估')

    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'pretrain', 'baseline', 'eval'],
                        help='实验阶段：pretrain 先跑 BERT 预训练；baseline 跑 AlphaQubit；eval 跑 BERT 微调+MWPM+LER')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='eval 阶段使用的 BERT 预训练检查点路径（默认 save_dir/bert_pretrain/best.pt）')
    parser.add_argument('--finetune_checkpoint', type=str, default=None,
                        help='eval 阶段使用的 BERT 微调检查点路径（默认 save_dir/bert_finetune/best.pt）')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='eval 阶段跳过 BERT 微调，直接加载 --finetune_checkpoint')

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

                # PAEMS LER 文件本身已按对应 rounds 生成，直接取全部时间步
                # 保留截断逻辑以兼容未来可能传入的长 rounds 文件
                t_max = min(rounds, measurement.size(1))
                measurement = measurement[:, :t_max, :]
                event = event[:, :t_max, :]
                leakage = leakage[:, :t_max, :]
                event_leakage = event_leakage[:, :t_max, :]

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

    model = PretrainDecoder(
        coord_system=train_ds.coord_system,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        num_transformer_layers=args.num_transformer_layers,
        dropout=0.1,
    )

    config = PretrainConfig(
        total_steps=args.pretrain_steps,
        batch_size=args.batch_size,
        eval_interval=1000,
        log_interval=500,
        learning_rate=2e-4,
        mask_ratio=args.mask_ratio,
        device=args.device,
        use_amp=True,
        early_stopping_patience=10000,
    )

    trainer = PretrainTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
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
        learning_rate=1e-4,
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


def _build_paems_dem(distance: int, rounds: int, paems_data_dir: Path):
    """使用 PAEMS 噪声模型构建 detector error model"""
    code_dir = str(paems_data_dir / 'code')
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import paems_noise_model as pnm

    params = pnm.generate_paems_params(distance, seed=distance * 7919 + 42)
    base = pnm._base_surface_code_circuit(distance, rounds)
    noisy = pnm.build_paems_noisy_circuit(base, params, rounds)
    return noisy.detector_error_model()


def evaluate_mwpm(test_ds, args):
    """评估 MWPM 基准（使用 PAEMS 噪声模型）"""
    print("\n" + "=" * 60)
    print("[4/4] MWPM 基准（PyMatching + PAEMS noise model）")
    print("=" * 60)

    try:
        import pymatching
    except ImportError:
        print("PyMatching 未安装，跳过 MWPM 评估")
        print("安装命令: pip install pymatching")
        return None

    paems_data_dir = Path(args.paems_data_dir)
    dem = _build_paems_dem(test_ds.distance, test_ds.rounds, paems_data_dir)
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
    """在多个 rounds 上评估 MWPM（使用 PAEMS 噪声模型）"""
    try:
        import pymatching
    except ImportError:
        return None, None

    paems_data_dir = Path(args.paems_data_dir)

    predictions_by_rounds = {}
    labels_by_rounds = {}

    for rounds, dataset in eval_datasets.items():
        dem = _build_paems_dem(dataset.distance, rounds, paems_data_dir)
        mwpm = pymatching.Matching.from_detector_error_model(dem)

        n = len(dataset)
        det_events = dataset._data['detection_events'][:n].reshape(n, -1)
        labels = dataset._data['label'][:n].int().numpy()

        preds = mwpm.decode_batch(det_events.numpy()).flatten()
        predictions_by_rounds[rounds] = preds
        labels_by_rounds[rounds] = labels

    return predictions_by_rounds, labels_by_rounds


def load_ler_datasets(distance: int, rounds_list, num_samples: int, paems_data_dir: Path):
    """加载 PAEMS-data 中预生成的 LER 扫描文件"""
    datasets = {}
    print(f"\n加载 PAEMS LER 评估数据: rounds={rounds_list}, 每轮 {num_samples} 样本...")

    for rounds in rounds_list:
        pt_path = paems_data_dir / f"ler_d{distance}_r{rounds}_n{num_samples}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"找不到 PAEMS LER 文件: {pt_path}\n"
                f"请确认 --ler_rounds 与 --ler_samples 与 PAEMS-data 目录中的文件一致。"
            )
        datasets[rounds] = PTBatchDataset(str(pt_path))
        print(f"  rounds={rounds}: loaded {pt_path}")

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


def save_results(results: dict, ler_results: dict, args):
    """将实验结果保存为 JSON"""
    save_path = Path(args.save_dir) / 'results_summary.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        'config': {
            'train_pt': args.train_pt,
            'val_pt': args.val_pt,
            'test_pt': args.test_pt,
            'embed_dim': args.embed_dim,
            'n_heads': args.n_heads,
            'num_transformer_layers': args.num_transformer_layers,
            'num_readout_layers': args.num_readout_layers,
            'batch_size': args.batch_size,
            'alphaqubit_steps': args.alphaqubit_steps,
            'pretrain_steps': args.pretrain_steps,
            'finetune_steps': args.finetune_steps,
            'finetune_subset_ratio': args.finetune_subset_ratio,
            'seed': args.seed,
            'device': args.device,
        },
        'results': {},
        'ler': {},
    }

    for name, metrics in results.items():
        if metrics is None:
            continue
        if isinstance(metrics, dict):
            m = {k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else str(type(v))
                 for k, v in metrics.items()}
            out['results'][name] = m

    for name, result in ler_results.items():
        if result is None:
            continue
        out['ler'][name] = {
            'ler': float(result.ler),
            'r_squared': float(result.r_squared),
            'log_f0': float(result.log_f0),
            'is_valid': bool(result.is_valid),
            'error_rates': {str(k): float(v) for k, v in result.error_rates.items()},
            'fidelities': {str(k): float(v) for k, v in result.fidelities.items()},
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n实验结果已保存: {save_path}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("PAEMS 合成数据对比实验")
    print("=" * 60)
    print(f"配置: embed_dim={args.embed_dim}, n_heads={args.n_heads}, "
          f"layers={args.num_transformer_layers}, readout_layers={args.num_readout_layers}")
    print(f"训练步数: AlphaQubit={args.alphaqubit_steps}, "
          f"BERT pretrain={args.pretrain_steps}, BERT finetune={args.finetune_steps}")
    print(f"BERT mask_ratio={args.mask_ratio}, 微调标签比例={args.finetune_subset_ratio*100:.0f}%")
    print(f"PAEMS 数据目录: {args.paems_data_dir}")

    print("\n加载数据集...")
    train_ds = load_dataset(args.train_pt)
    val_ds = load_dataset(args.val_pt)
    test_ds = load_dataset(args.test_pt)

    print(f"训练集: {len(train_ds)} 样本")
    print(f"验证集: {len(val_ds)} 样本")
    print(f"测试集: {len(test_ds)} 样本")
    print(f"码距 d={test_ds.distance}, 轮数 rounds={test_ds.rounds}, p={test_ds.p:.6f}, snr={test_ds.snr}")

    results = {}
    ler_results = {}

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
            pretrain_ckpt = args.pretrain_checkpoint or f"{args.save_dir}/bert_pretrain/best.pt"
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
            alphaqubit_ckpt = f"{args.save_dir}/alphaqubit_baseline/best.pt"
            if Path(alphaqubit_ckpt).exists():
                checkpoint = torch.load(alphaqubit_ckpt, map_location='cpu', weights_only=False)
                alphaqubit_model.load_state_dict(checkpoint['model_state_dict'])
                alphaqubit_model = alphaqubit_model.to(args.device)
                print("[Eval] AlphaQubit 基准模型加载完成")
            else:
                print(f"[Eval] 警告：找不到 AlphaQubit 检查点 {alphaqubit_ckpt}，将跳过 AlphaQubit LER")
                alphaqubit_model = None

        finetune_ckpt = args.finetune_checkpoint or f"{args.save_dir}/bert_finetune/best.pt"
        if args.skip_finetune and Path(finetune_ckpt).exists():
            print(f"\n[Eval] 跳过 BERT 微调，直接加载检查点: {finetune_ckpt}")
            bert_finetuned = FineTuneDecoder(
                coord_system=train_ds.coord_system,
                pretrained_encoder=bert_pretrained,
                embed_dim=args.embed_dim,
                readout_dim=64,
                n_heads=args.n_heads,
                num_transformer_layers=args.num_transformer_layers,
                num_readout_layers=args.num_readout_layers,
                dropout=0.1,
            ).to(args.device)
            checkpoint = torch.load(finetune_ckpt, map_location='cpu', weights_only=False)
            bert_finetuned.load_state_dict(checkpoint['model_state_dict'])
            print("[Eval] BERT 微调模型加载完成")
            results['bert_finetune'] = evaluate_model(bert_finetuned, test_ds, args.device)
            print(f"\n[BERT Finetune] Test Accuracy: {results['bert_finetune']['accuracy']:.4f}, Test Loss: {results['bert_finetune']['loss']:.4f}")
        else:
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

            paems_data_dir = Path(args.paems_data_dir)
            ler_datasets = load_ler_datasets(
                distance=test_ds.distance,
                rounds_list=args.ler_rounds,
                num_samples=args.ler_samples,
                paems_data_dir=paems_data_dir,
            )

            if alphaqubit_model is not None:
                aq_preds, aq_labels = evaluate_model_at_rounds(alphaqubit_model, ler_datasets, args.device)
                aq_ler = compute_ler(aq_preds, aq_labels)
                print_ler_result("AlphaQubit", aq_ler)
                ler_results['alphaqubit'] = aq_ler
            else:
                aq_ler = None

            bert_preds, bert_labels = evaluate_model_at_rounds(bert_finetuned, ler_datasets, args.device)
            bert_ler = compute_ler(bert_preds, bert_labels)
            print_ler_result("BERT Finetune", bert_ler)
            ler_results['bert_finetune'] = bert_ler

            mwpm_preds, mwpm_labels = evaluate_mwpm_at_rounds(ler_datasets, args)
            if mwpm_preds is not None:
                mwpm_ler = compute_ler(mwpm_preds, mwpm_labels)
                print_ler_result("MWPM", mwpm_ler)
                ler_results['mwpm'] = mwpm_ler

            print("\n" + "=" * 60)
            print("LER 横向对比")
            print("=" * 60)
            if aq_ler is not None:
                print(f"AlphaQubit    LER: {aq_ler.ler:.6f} (R²={aq_ler.r_squared:.3f}, valid={aq_ler.is_valid})")
            print(f"BERT Finetune LER: {bert_ler.ler:.6f} (R²={bert_ler.r_squared:.3f}, valid={bert_ler.is_valid})")
            if mwpm_preds is not None:
                print(f"MWPM          LER: {mwpm_ler.ler:.6f} (R²={mwpm_ler.r_squared:.3f}, valid={mwpm_ler.is_valid})")

        save_results(results, ler_results, args)


if __name__ == '__main__':
    main()
