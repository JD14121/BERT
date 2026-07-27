# Focal Loss BERT 微调消融实验计划书

> 版本：2026-07-20 ｜ 状态：**审查组 APPROVED（3 门禁已云端核实 PASS）** ｜ 执行时机：d5 opt 跑完出基线 LER 后

## 1. 背景与目的

d5 BERT 微调日志（69 步实测）显示**系统性欠预测正类（逻辑错误类）偏差**：
- `pos_rate`（真实正类比例）= 0.351，`pred_pos_rate`（模型预测）= 0.315，bias = **-0.036**
- 75% 的步 `pred_pos_rate < pos_rate`，模型偏向多数类（无错误）
- 对 LER 致命：欠预测逻辑错误 = false negative = 漏报 -> LER 上升（QEC 里漏报比误报严重得多）

当前 d5 opt：AQ 预训练/微调用 focal（gamma=2），**BERT 微调用 BCE（gamma=0）**。multi_code_bert 已验证 focal+plain 采样修复塌缩并提 lift。

**目的**：将 BERT 微调的 focal_gamma 从 0 改为 2（**其余完全不变**），测试能否纠正欠预测偏差、降低 LER，而不损 accuracy。

## 2. 实验设计（单变量）

**唯一变量**：BERT 微调 `focal_gamma` 0.0 -> 2.0

不变项（与基线 d5 opt 完全一致）：
| 项 | 值 |
|---|---|
| 起点 checkpoint | `bert_pretrain_d5/best.pt`（已训，12:46，143MB）|
| 数据 | real_d5（train 160k / val 20k / test 20k）+ 合成 d5 子集（mix_ratio 0.5, seed 42）|
| 超参 | finetune_steps 8000, lr 1e-4, bs 512, embed 256 / heads 8 / layers 4 / readout 6 |
| 模型 | XZZXFineTuneDecoder（同基线）|
| 采样 | plain 自然采样（train 分布 = test，无失配，不崩）|

保存到独立目录 `bert_finetune_d5_focal/`（**不覆盖基线**），结果存 `results_summary_d5_focal.json` + `results_ler_d5_focal.json`。

## 3. 代码改动（d5 opt 跑完后应用，不触碰正在运行的代码）

### 3.1 run_experiment.py
- 加 `--bert-focal-gamma`（default 0.0，向后兼容）
- 加 `--start-from`（default `aq_pretrain`；`bert_finetune` 跳过 AQ + BERT pretrain，直接从 bert_pretrain_d5 起点做 BERT 微调 -- 复用全部数据加载代码，保证与基线一致）
- 加 `--ft-suffix`（default `""`；`_focal` -> 存 `bert_finetune_d5_focal/`，结果 `results_summary_d5_focal.json`）
- BERT 微调调用（line 143）传 `focal_gamma=args.bert_focal_gamma`，**保留硬编码 `min_steps=2000, patience=10` 不变**（M2），save_dir 加 suffix

**`--start-from bert_finetune` 跳过/保留块（S3，实现后须审查组确认）**：
- 保留：argparse（L71-84）、coord setup（L85）、数据加载 syn/real（L87-105）
- 跳过：MWPM eval（L110-113，省 1-2min，不影响 BERT）、AQ pretrain+finetune（L115-123）
- 保留：BERT checkpoint 加载（L127-129）、BERT finetune（L134-145，改 focal_gamma+suffix）、results 保存（L147-152，文件名加 suffix）

### 3.2 eval_ler.py
- 加 `--ft-suffix`，评估 `bert_finetune_d5{suffix}/best.pt`（其余 LER 逻辑不动，已审查批准过）

### 3.3 门禁核实结果（审查组 M1/M2/M3，已云端 PASS）

- **M1 pos_weight = None ✅**：云端 `trainer.py:206-208` 构造 `DecoderLoss(label_smoothing=..., focal_gamma=...)`，不传 pos_weight -> None。patch_focal.py 只改 run_experiment.py 的 finetune 签名，未碰 DecoderLoss 构造。切 focal 仅改 gamma 一项，**单变量成立**。
- **M2 min_steps/patience 对齐 ✅**：`run_experiment.py:143` BERT 微调调用硬编码 `min_steps=2000, patience=10`，argparse 无对应 CLI 参数。focal 消融复用同一调用行（只加 focal_gamma + 改 save_dir），自动继承。
- **M3 batch_size = 512 ✅**：d5 opt 实际命令 `--batch-size 512`（ps 确认）。本地旧 E1 的 bs=128 是历史结果，非 d5 opt 基线。focal 用 512 = 基线。

## 4. 数据与资源

| 资源 | 状态 |
|---|---|
| bert_pretrain_d5/best.pt | ✅ 已有 |
| real_d5/ (570MB) | ✅ 已有 |
| 合成 d5 npy_compressed | ✅ 已有 |
| GPU | V100 32GB（d5 opt 跑完后释放）|
| 时间 | BERT 微调 8000 步 ~80min + eval_ler ~15min = ~1.5h |
| 磁盘 | bert_finetune_d5_focal/ ~1.5GB |

## 5. 执行流程

1. 等 d5 opt 完成（定时 19:27 通知），记录**基线** accuracy + LER（results_summary_d5_E1_160M_100k_opt.json / results_ler_d5_E1_160M_100k_opt.json）
2. **核查 pos_weight**（§3.3），若非默认则在结果里披露
3. 应用代码 patch（3.1 + 3.2）
4. 跑 focal 微调：
```bash
python run_experiment.py --distance 5 --start-from bert_finetune \
  --bert-focal-gamma 2.0 --ft-suffix _focal \
  --embed-dim 256 --n-heads 8 --num-transformer-layers 4 --num-readout-layers 6 \
  --batch-size 512 --finetune-steps 8000 --mix-synth-ratio 0.5
```
5. 跑 LER：`python eval_ler.py --distances 5 --ft-suffix _focal`
6. 对比基线 vs focal

## 6. 验收标准 (QC)

| QC | 标准 |
|---|---|
| 微调完成 | 8000 步（或早停，min_steps 2000）|
| val_acc 合理 | ~0.92–0.93，与基线 ±2pp 内（大幅低 = focal 损 accuracy）|
| **偏差缩小** | `|pred_pos_rate - pos_rate|` 应 < 基线 0.036（验证机制成立）|
| LER 拟合 | R² >= 0.9（valid）|
| **训练步数对齐**（M2）| 记录基线与 focal 实际训练步数，若不一致须披露 |
| ECE 校准参考（S2）| 记录基线与 focal 的 val ECE（`compute_calibration_metrics` 已有），focal 已知扭曲校准，作参考非门禁 |

**胜利判定（S1 量化）**：
- **PASS**：focal LER 相对基线降幅 **≥ 10%**（单 seed 下 <10% 无法区分效应与噪声）
- **不确定**：降幅 < 10% -> 标注"需多 seed 复核"
- **FAIL**：focal 升 LER；或 focal 降 accuracy 但不降 LER；或 focal 不降 accuracy 也不降 LER（neutral 无效）
- 理想：focal LER 接近 MWPM 0.0034

## 7. 透明声明

1. focal 实现丢弃 pos_weight（losses.py `_focal_loss` 不用 pos_weight）-- 与 AQ 阶段 focal 设置一致，非新引入简化；§3.3 预核查会确认影响
2. focal gamma=2 是常用默认（multi_code_bert 已验证）
3. 训练随机性（dropout 等）无法完全消除，但 seed 42 数据子集固定，差异应主要来自 focal
4. AQ 阶段不重跑（与 BERT 微调无关），节省 4.6h GPU -- 不影响单变量完整性

## 8. 风险

- **focal 可能降 accuracy**：欠采样多数类 -> accuracy 小跌可接受（LER 是主指标）
- **GPU 时序**：必须等 d5 opt 完全释放 GPU 后再跑，避免 OOM
- **pos_weight 双变量**：§3.3 核查，若非默认需披露
- **--start-from 新增逻辑**：跳过 AQ 的代码路径需审查组确认无副作用（不误跳 BERT pretrain 加载）
