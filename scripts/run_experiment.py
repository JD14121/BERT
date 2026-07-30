#!/usr/bin/env python3
"""
AlphaQubit 完整实验脚本（流式重构版）

数据层全部替换为 StreamingPAEMSDataset（不落盘、不预分配全量数组），
推理层使用 make_xzzx_decoder_fn（XZZXAlphaQubitDecoder，分批 GPU 推理）。
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

#坐标系 / 解码器
from xzzx_coord import XZZXCoordinateSystem
from xzzx_decoder import XZZXAlphaQubitDecoder
from stream_decoder import make_xzzx_decoder_fn

# 训练基础设施（保持原有）
from alphaqubit.training import Trainer, TrainingConfig
from alphaqubit.evaluation import compute_ler, compute_lambda, print_ler_summary
from alphaqubit.experiments.baselines import MWPMBaseline

# PAEMS 数据生成
import paems_noise_model as pnm
import paems_iq_readout as pir

# Google路径配置
try:
    from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR
    import stimHAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

#── 流式数据集（复用 bert_pretrain.py 中已定义的版本） ─────────────────────────
class StreamingPAEMSDataset(IterableDataset):
    """
    按需流式生成软读出数据，不预加载任何文件，不积累全量数组。
    chunk处理完即GC 回收，只持有一个 chunk_size 大小的内存窗口。
    """
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
        for chunk in pir.stream_paems_iq_dataset(
            self.distance, self.rounds, self.num_samples, self.params,
            snr=self.snr, t=self.t, seed=self.seed,
            chunk_size=self.chunk_size,
            include_leakage=self.include_leakage,
        ):
            batch_n = int(chunk['label'].shape[0])
            for i in range(batch_n):
                # leakage 字段：若没有则填零（兼容 include_leakage=False）
                leakage = (chunk['leakage'][i]if 'leakage' in chunk
                           else np.zeros(
                               (self.rounds, self.distance**2 - 1), dtype=np.float32))
                event_leakage = (chunk.get('event_leakage', chunk['leakage'])[i]
                                 if 'leakage' in chunk
                                 else np.zeros_like(leakage))
                yield {
                    'measurement':chunk['measurement'][i],
                    'event':            chunk['event'][i],
                    'final_soft':       chunk['final_soft'][i],
                    'detection_events': chunk['detection_events'][i],
                    'label':            chunk['label'][i],
                    'leakage':          leakage,
                    'event_leakage':    event_leakage,
                }
            del chunk
            gc.collect()

    def __len__(self):
        return self.num_samples

# ── 通用工具函数 ───────────────────────────────────────────────────────────────
def _load_or_gen_params(distance: int, params_dir: Path) -> dict:
    """加载或生成 per-distance 设备参数（只写一个小JSON，不写数据）。"""
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

def _build_coord_system(distance: int, rounds: int):
    """
    构建坐标系：优先用 Google XZZX 模板，回退用标准 stim 生成电路。
    """
    if HAS_GOOGLE and GOOGLE_PATCH.get(distance):
        import stim
        circuit_path = (GOOGLE_SC
                        / f"d{distance}_at_{GOOGLE_PATCH[distance]}"
                        / "Z"
                        / f"r{rounds:02d}"
                        / "circuit_ideal.stim")
        if circuit_path.exists():
            cir = stim.Circuit.from_file(str(circuit_path))
            cs= XZZXCoordinateSystem(distance, cir)
            print(f"[coord] XZZX Google模板  "
                  f"grid={cs.grid_size}×{cs.grid_size}  "
                  f"n_stab={cs.n_stab}  n_data={cs.n_data}")
            return cs

    # 回退：用stim 标准电路生成坐标系
    import stim as _stim
    base = pnm._base_surface_code_circuit(distance, rounds)
    cs= XZZXCoordinateSystem(distance, base)
    print(f"[coord] XZZX stim 回退  "
          f"grid={cs.grid_size}×{cs.grid_size}  "
          f"n_stab={cs.n_stab}  n_data={cs.n_data}")
    return cs

@torch.no_grad()
def _stream_ler_eval(model, distance, rounds_list, num_samples, params,
                     device, seed=0, chunk_size=4096, batch_size=512):
    """
    流式 LER 评估：对 rounds_list 中每个 n_rounds 独立生成测试集，
    分批推理，不把整个测试集加载到 GPU。

    返回 predictions_by_rounds, labels_by_rounds（均为 np.ndarray）。
    """
    model.eval()
    predictions_by_rounds = {}
    labels_by_rounds      = {}

    for n_rounds in rounds_list:
        test_ds = StreamingPAEMSDataset(
            distance=distance,
            rounds=n_rounds,
            num_samples=num_samples,
            params=params,
            snr=10.0, t=0.01,
            seed=seed + n_rounds,
            chunk_size=chunk_size,
            include_leakage=False,
        )
        loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        all_preds  = []
        all_labels = []

        for batch in loader:
            preds, _ = model.predict(
                batch['measurement'].to(device),
                batch['event'].to(device),
                batch['leakage'].to(device),
                batch['event_leakage'].to(device),batch['final_soft'].to(device),
            )
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].numpy().flatten())

            del batch, preds
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        predictions_by_rounds[n_rounds] = np.concatenate(all_preds)
        labels_by_rounds[n_rounds]      = np.concatenate(all_labels)lr = 1.0 - np.mean(
            predictions_by_rounds[n_rounds] == labels_by_rounds[n_rounds]
        )
        print(f"    rounds={n_rounds:2d}  LER={lr:.5f}  "
              f"n={len(all_labels[0]) * len(all_labels):,}")

    return predictions_by_rounds, labels_by_rounds

# ── Phase 1 ────────────────────────────────────────────────────────────────────
def run_phase1(output_dir: Path, debug: bool = False, seed: int = 42,
               checkpoint: str = None):
    print("\n" + "=" * 60)
    print("PHASE 1: Pipeline Validation (d=3) —流式版")
    print("=" * 60)

    device= "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    distance= 3
    rounds      = 12
    num_samples = 500if debug else 5000
    train_steps = 200 if debug else 3000

    params_dir = output_dir / "params"
    params = _load_or_gen_params(distance, params_dir)
    cs= _build_coord_system(distance, rounds)

    # === Step 1: 流式数据集健全性检查 ===
    print("\n[Step 1] 流式数据集健全性检查...")
    probe_ds = StreamingPAEMSDataset(
        distance, rounds, min(256, num_samples), params,
        snr=10.0, t=0.01, seed=seed, chunk_size=256,
    )
    probe_loader = DataLoader(probe_ds, batch_size=64)
    sample_batch = next(iter(probe_loader))

    event_density = float(sample_batch['detection_events'].mean())
    print(f"  event density (probe): {event_density:.4f}")
    assert0.001 < event_density < 0.4, f"event density 异常: {event_density}"
    assert sample_batch['measurement'].shape[1] == rounds, "rounds 维度不匹配"
    print("  ✓ 流式数据集健全性通过")

    # === Step 2: 模型 sanity check ===
    print("\n[Step 2] 模型 sanity check...")
    model = XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=128)
    model = model.to(device)

    sb = {k: v.to(device) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
    logits = model(
        sb['measurement'], sb['event'],
        sb['leakage'], sb['event_leakage'], sb['final_soft'],
    )
    initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, sb['label']
    ).item()
    print(f"  初始 loss: {initial_loss:.4f}（期望 ~0.693）")
    assert 0.5 < initial_loss < 1.0, f"初始 loss 异常: {initial_loss}"
    print("  ✓ 初始 loss 检查通过")

    # === Step 3: 短期训练 ===
    print("\n[Step 3] 短期训练验证...")
    train_ds = StreamingPAEMSDataset(
        distance, rounds, num_samples, params,
        seed=seed, chunk_size=512,
    )
    val_ds = StreamingPAEMSDataset(
        distance, rounds, num_samples // 5, params,
        seed=seed + 1000, chunk_size=512,
    )
    config = TrainingConfig(
        total_steps=train_steps, batch_size=128,
        learning_rate=2e-4, warmup_steps=50,
        eval_interval=train_steps // 5,
        log_interval=train_steps // 10,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_dir=str(output_dir / "phase1_checkpoints"),
    )
    history = trainer.train()

    if len(history.get('train_loss', [])) > 10:
        early = np.mean(history['train_loss'][:10])
        late  = np.mean(history['train_loss'][-10:])
        print(f"  early loss={early:.4f}  late loss={late:.4f}")
        assert late < early, "训练未降低 loss！"
        print("  ✓ Loss 下降确认")

    # === Step 4: MWPM Baseline（可选） ===
    print("\n[Step 4] MWPM baseline（可选）...")
    try:
        mwpm   = MWPMBaseline(seed=seed)
        result = mwpm.evaluate_single(
            distance=distance, rounds=rounds,
            noise_p=0.005, num_samples=min(2000, num_samples),
        )
        print(f"  MWPM error rate: {result.error_rate:.4f}")
        print("  ✓ MWPM 完成")
    except (ImportError, Exception) as e:
        print(f"  ⚠ MWPM 跳过: {e}")

    # === Step 5: 流式 LER 评估 ===
    print("\n[Step 5] 流式 LER 评估...")
    rounds_list = [3, 6, 9, 12]

    predictions_by_rounds, labels_by_rounds = _stream_ler_eval(
        model=model,
        distance=distance,
        rounds_list=rounds_list,
        num_samples=min(2000, num_samples),
        params=params,
        device=device,
        seed=seed + 9000,
        chunk_size=512,
        batch_size=256,
    )

    ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)
    print(f"  LER={ler_result.ler:.6f}  R²={ler_result.r_squared:.4f}"
          f"  valid={ler_result.is_valid}")

    print("\n" + "=" * 60)
    print("PHASE 1 完成（流式版）")
    print("=" * 60)
    return {"ler_result": ler_result, "history": history}

# ── Phase 2 ────────────────────────────────────────────────────────────────────
def run_phase2(output_dir: Path, debug: bool = False, seed: int = 42,
               checkpoint: str = None):
    print("\n" + "=" * 60)
    print("PHASE 2: Full Experiments (d=3, d=5) — 流式版")
    print("=" * 60)

    results = {}

    for distance in [3, 5]:
        print(f"\n--- Training d={distance} ---")

        device= "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        rounds      = 12
        num_samples = 2000 if debug else 50000
        train_steps = 500if debug else 15000

        params_dir = output_dir / "params"
        params = _load_or_gen_params(distance, params_dir)
        cs     = _build_coord_system(distance, rounds)

        # 流式训练集/ 验证集
        train_ds = StreamingPAEMSDataset(
            distance, rounds, num_samples, params,
            seed=seed, chunk_size=2048,
        )
        val_ds = StreamingPAEMSDataset(
            distance, rounds, num_samples // 10, params,
            seed=seed + 1000, chunk_size=2048,
        )

        model = XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=256)
        model = model.to(device)

        config = TrainingConfig(
            total_steps=train_steps,
            batch_size=512if distance <= 5 else 256,
            learning_rate=2e-4,
            warmup_steps=500,
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            save_dir=str(output_dir / f"phase2_d{distance}"),
        )
        trainer.train()

        # 流式 LER 评估
        rounds_list = [3, 5, 7, 9, 11, 12]
        predictions_by_rounds, labels_by_rounds = _stream_ler_eval(
            model=model,
            distance=distance,
            rounds_list=rounds_list,
            num_samples=min(5000, num_samples),
            params=params,
            device=device,
            seed=seed + 9000,
            chunk_size=1024,
            batch_size=512,
        )

        ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)
        results[distance] = ler_result
        print(f"  LER (d={distance}): {ler_result.ler:.6f}  "
              f"R²={ler_result.r_squared:.4f}")

    # Λ计算
    if3 in results and 5 in results:
        if results[3].is_valid and results[5].is_valid:
            lambda_val = compute_lambda(results[3].ler, results[5].ler)
            print(f"\n  Λ_{{3→5}} = {lambda_val:.4f}")
            print("  ✓ 错误抑制确认" if lambda_val > 1else "  ⚠ Λ≤1，检查训练")

    print_ler_summary(results)
    print("\n" + "=" * 60)
    print("PHASE 2 完成（流式版）")
    print("=" * 60)
    return results

# ── Phase 3 ────────────────────────────────────────────────────────────────────
def run_phase3(output_dir: Path, debug: bool = False, seed: int = 42,
               checkpoint: str = None):
    print("\n" + "=" * 60)
    print("PHASE 3: Extended Experiments (d=9) — 流式版")
    print("=" * 60)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    distance    = 9
    rounds      = 12
    num_samples = 1000  if debug else 100000
    train_steps = 1000  if debug else 50000

    params_dir = output_dir / "params"
    params = _load_or_gen_params(distance, params_dir)
    cs     = _build_coord_system(distance, rounds)

    train_ds = StreamingPAEMSDataset(
        distance, rounds, num_samples, params,
        seed=seed, chunk_size=1024,
    )
    val_ds = StreamingPAEMSDataset(
        distance, rounds, num_samples // 10, params,
        seed=seed + 1000, chunk_size=1024,
    )

    # d=9 用更大embed_dim，降低 lr
    model = XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=512)
    model = model.to(device)

    config = TrainingConfig(
        total_steps=train_steps,
        batch_size=128,
        learning_rate=1e-4,
        warmup_steps=2000,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_dir=str(output_dir / f"phase3_d{distance}"),
    )
    trainer.train()

    rounds_list = [3, 6, 9, 12]
    predictions_by_rounds, labels_by_rounds = _stream_ler_eval(
        model=model,
        distance=distance,
        rounds_list=rounds_list,
        num_samples=min(5000, num_samples),
        params=params,
        device=device,
        seed=seed + 9000,
        chunk_size=512,
        batch_size=128,         # d=9 每样本更大，batch缩小
    )

    ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)
    print(f"  LER (d={distance}): {ler_result.ler:.6f}  "
          f"R²={ler_result.r_squared:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 3 完成（流式版）")
    print("=" * 60)
    return {distance: ler_result}

# ── main ───────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaQubit experiments（流式版）")
    parser.add_argument("--phase",       type=str, default="1",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--debug",       action="store_true",
                        help="快速冒烟：小样本 + 少步数")
    parser.add_argument("--output-dir",  type=str, default="experiments")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="可选：加载已有检查点继续训练或直接评估")
    parser.add_argument("--eval-only",   action="store_true",
                        help="跳过训练，直接做流式 LER 评估（需配合 --checkpoint）")
    parser.add_argument("--distance",    type=int, default=None,
                        help="覆盖默认 distance（仅单Phase 模式有效）")
    parser.add_argument("--rounds",      type=int, default=12,
                        help="训练轮数（覆盖默认值）")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="覆盖默认 num_samples")
    parser.add_argument("--batch-size",  type=int, default=None,
                        help="覆盖默认 batch_size")
    parser.add_argument("--embed-dim",   type=int, default=None,
                        help="覆盖默认 embed_dim")
    parser.add_argument("--chunk-size",  type=int, default=2048,
                        help="流式生成的chunk 大小（调大可提升吞吐，但占用更多内存）")
    parser.add_argument("--device",      type=str, default=None,
                        help="cuda / cpu（默认自动检测）")
    return parser.parse_args()

def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写本次运行的配置快照，方便复现
    run_cfg = {
        "phase":       args.phase,
        "debug":       args.debug,
        "seed":        args.seed,
        "checkpoint":  args.checkpoint,
        "eval_only":   args.eval_only,
        "timestamp":   datetime.now().isoformat(),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # ── eval-only 快速入口（跳过训练，直接流式评估） ────────────────────────────
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only 需要指定 --checkpoint")

        distance = args.distance or 3
        rounds   = args.rounds
        device   = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        params_dir = output_dir / "params"
        params = _load_or_gen_params(distance, params_dir)
        cs     = _build_coord_system(distance, rounds)

        model = XZZXAlphaQubitDecoder(
            coord_system=cs,
            embed_dim=args.embed_dim or 256,
        )
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state =ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model = model.to(device)
        print(f"[eval-only] loaded checkpoint: {args.checkpoint}"
              f"step={ckpt.get('global_step', 'unknown')}")

        rounds_list = [3, 6, 9, 12]
        num_samples = args.num_samples or 5000

        predictions_by_rounds, labels_by_rounds = _stream_ler_eval(
            model=model,
            distance=distance,
            rounds_list=rounds_list,
            num_samples=num_samples,
            params=params,
            device=device,
            seed=args.seed + 9000,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size or 512,
        )

        ler_result = compute_ler(predictions_by_rounds, labels_by_rounds)
        print_ler_summary({distance: ler_result})

        #保存评估结果
        eval_out = output_dir / f"eval_only_d{distance}.json"
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump({
                "distance":  distance,
                "rounds":    rounds_list,
                "ler":       float(ler_result.ler),
                "r_squared": float(ler_result.r_squared),
                "is_valid":  bool(ler_result.is_valid),
                "checkpoint":args.checkpoint,
            }, f, indent=2)
        print(f"[eval-only] 结果已保存 -> {eval_out}")
        return

    # ── 正常训练 + 评估流程 ─────────────────────────────────────────────────────
    all_results = {}
    t_total= time.time() if'time' in dir() else __import__('time').time()

    #兼容 time未在顶层 import 的情况
    import time as _time

    if args.phase in ("1", "all"):
        t0= _time.time()
        res = run_phase1(
            output_dir,
            debug=args.debug,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        all_results["phase1"] = res
        print(f"[Phase 1 耗时] {_time.time() - t0:.0f}s\n")

    if args.phase in ("2", "all"):
        t0  = _time.time()
        res = run_phase2(
            output_dir,
            debug=args.debug,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        all_results["phase2"] = res
        print(f"[Phase 2 耗时] {_time.time() - t0:.0f}s\n")

    if args.phase in ("3", "all"):
        t0  = _time.time()
        res = run_phase3(
            output_dir,
            debug=args.debug,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        all_results["phase3"] = res
        print(f"[Phase 3 耗时] {_time.time() - t0:.0f}s\n")

    # ── 汇总打印 & 写入 JSON ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("全部 Phase 完成 — 最终汇总")
    print("=" * 60)

    summary = {}
    for phase_key, phase_res in all_results.items():
        if isinstance(phase_res, dict):
            for k, v in phase_res.items():
                if hasattr(v, "ler"):
                    tag = f"{phase_key}_d{k}"
                    summary[tag] = {
                        "ler":       float(v.ler),
                        "r_squared": float(v.r_squared),
                        "is_valid":  bool(v.is_valid),
                    }
                print(f"  {tag}: LER={v.ler:.6f}  R²={v.r_squared:.4f}")summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[汇总] 已保存 -> {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
