# 击败 MWPM · 实验设计书（BEAT_MWPM_DESIGN）

> **版本**：2026-07-16 v2  |  **状态**：P0 审查 APPROVE_WITH_CONDITIONS（8 必改已落实，P1 可启动）  |  **环境**：conda `quantum_env` + RTX 4070 SUPER 12GB；数据生成 conda `base`
>
> **目标**：在 Google 105Q 真机校准噪声（XZZX, Z 基, r=10）下，通过**自监督预训练的进阶目标（③）+ 更大模型 + 10× 数据 + 混合模态掺杂**，让 BERT 解码器在 **d3 LER 与 accuracy 双指标上击败 MWPM**，d5 为 stretch，d7 诚实缩小差距。
>
> **路线**：① 纯规模基线 + ③ 进阶 MSM 主攻，② MWPM 蒸馏/级联作条件兜底。

---

## 0. 背景与核心创新

本项目（`google_paems_data/`）已完成的实验（`EXPERIMENT_REPORT.md`，2026-07-15）验证了 BERT（自监督掩码 syndrome 建模 MSM 预训练 + 少量真机微调）相对 AlphaQubit（from-scratch）的三条优势，共享深层机制"**MSM 学到 physics-grounded、distribution-robust 的表示，在标签稀缺/分布偏移时补偿监督不足**"：

- **C1 预训练优势随码距放大**：accuracy d3 −0.45pp -> d5 +1.8pp -> d7 +4.6pp。
- **C2 OOD 轮次泛化**：r=1 BERT 错误率 4–8%，AlphaQubit d5/d7 反转至 92–95%。
- **C3 跨分布迁移**：合成软读出预训练 -> 真机硬读出微调有效。

但 MWPM 仍是精度最高基线（LER 最低 R²≈1.0，accuracy 全程最高）。本设计目标即**越过 MWPM 这条线**，且胜利尽量来自核心创新本身（③）而非复制 MWPM（②）。

### 当前基线结果（E0 v2 强制重跑校准）

| 指标 | d3 | d5 | d7 |
|---|---|---|---|
| LER（合成软读出）MWPM / BERT | 0.0109 / 0.0221 | 0.0034 / 0.0267 | 0.0027 / 0.0402 |
| Accuracy（真机硬读出）MWPM / BERT | 0.9125 / 0.9027 | 0.9428 / 0.7980 | 0.9702 / 0.7438 |

### 已锁定决策

1. **主轴 A（击败 MWPM）先做，B/C/D 后续**。
2. **A2（合成 LER，聚焦 d3/d5）+ A4（扩 d7 真机数据再冲 accuracy）**。
3. **模型可放大**：embed 128->256、Transformer 2->4 层、readout 4->6 层（~6-8M 参数）。
4. **数据规模**：d3 适量 2×，d5/d7 10×（train 8M）；**允许删除原 1× 数据**释放空间。
5. **微调可掺杂合成数据**（从既有 20% 提到 ~50%），且须确保掺杂有效。
6. **③ 进阶 MSM**：③-a 轮级掩码+变长跨度（主）、③-b 轮次不变对比（核心新组件）；**砍掉 ③-c 标签辅助**（保自监督纯粹）；**前置必修轮次嵌入外推**。
7. **② MWPM 蒸馏/级联**作 P6 条件兜底，透明披露混合解码。

---

## 1. 数据与掺杂有效性

### 1.1 10× 合成数据生成与存储

| 码距 | 当前 train | 新 train | .pt 体积 |
|---|---|---|---|
| d3 | 800k | ~1.6M（2×） | ~1.6GB |
| d5 | 800k | 8M（10×） | ~24GB |
| d7 | 800k | 8M（10×） | ~47GB |

- 复用 `code/generate_manifest.py`（新增 `--scale 10`，乘 train/val/test，**LER 保持 20k 不放大**）；噪声/拓扑/校准配置不变。（`generate_google_paems_data.py` 仅作 smoke/单文件，无 `--scale`）
- 生成器极快（stim：800k d5 约 15s），8M 估 d5~150s / d7~5min。
- **存储**：三码距 10× 合成 ~94GB（train/val/test 10× + LER 20k ~3GB），D: 余 121GB，**即使保留旧 1×（~20GB）也放得下**。删旧 1× 合成数据（用户已批准）仅为留余量，**非必须**。**真机数据 `data/real_d{3,5,7}/` 必须保留**（accuracy 评估与微调来源，不可删）。
- 10× 文件名含新 N（如 `train_d5_r10_n8000000_Z.pt`），与旧 `n800000` 共存，删旧按 N 区分。
- **必修生成器遗留**：`generate_google_paems_data.py::save_pt` 与 `prepare_google_real.py` 均漏存 `'p'` 字段 -> 补 `'p': 0.0` 对齐 schema，避免重跑后 `PTBatchDataset` KeyError（已知遗留，见 SELF_MEMORY 2026-07-15）。

### 1.2 掺杂有效性机制

> **v2（P0 审查后）**：原 modality flag 经审查有 train/eval 一致性漏洞（微调若只用 hard(flag0)，LER 评估用 soft(flag1) 时该模式未训练）且须改 trainer/dataset 等 6 文件。**已砍 flag**，回归已验证的无 flag 混合模态掺杂。

真机=硬读出(0/1, snr=∞)、合成=软读出(连续, snr=10)。直接把 10× 合成软读出倒进真机硬读出微调会淹没真机。两层机制：

1. **无 flag 混合模态掺杂**：合成保持软读出（连续 [0,1]）、真机硬读出（0/1）直接混合微调。hard 0/1 ⊂ soft [0,1]，模型自然兼容两种模态（原 20% 掺杂已验证可行）。**无 flag -> accuracy 评估（真机硬）与 LER 评估（合成软）均与训练混合分布一致，无 train/eval 模态失配漏洞**（审查漏洞#8 消除）。
2. **真机加权保护**：`WeightedRandomSampler` 使有效 real:synth≈1:1，防 10× 合成（8M）淹没真机（d7 仅 40k）。

### 1.3 掺杂有效性验证（核心实验，非护栏）

| 消融 | 变量 | 假说 |
|---|---|---|
| **A** 掺杂比例 × 码距 | {0%,20%,50%,80%} × {d3,d5,d7} | 增益随码距放大（d3 数据富余可能不增/略降，d7 数据稀缺应显著提升）= **C1 因果复现** |

> v2：砍 flag 后无"模态匹配/flag 开关"子臂（原 B/C），消融聚焦掺杂比例×码距。

> "掺杂比例 × 码距"曲线本质上是对 C1 的因果复现：数据补充价值随标签稀缺度（码距）增长。这让"掺杂是否有效"从防御性检查变成支撑主论点的实验。

### 1.4 数据流（v2 砍 flag）

```
10× 合成软读出(连续[0,1]) ─┐
                           ├─ ConcatDataset + WeightedRandomSampler(real:synth≈1:1) -> 混合微调
真机硬读出(0/1) ───────────┘
评估：accuracy 在真机硬测试集；LER 在合成软测试集（均与训练混合分布一致，无模态失配漏洞）
```

---

## 2. 更大模型 + ③ 进阶 MSM 目标

### 2.1 更大模型（① 容量杠杆）

- `PretrainDecoder`（`alphaqubit.models.pretrain_decoder`）：embed 128->**256**，Transformer 2->**4 层**（n_heads 4->**8** 保持 head_dim=32），readout 4->**6 层**。~1.5M -> **~6-8M 参数**。
- d7（97 节点）：bs 256->**128**，AMP，必要时梯度累积。预案已验证。
- XZZX 适配（`xzzx_decoder.py::XZZXStabToData`、`xzzx_coord.py::XZZXCoordinateSystem`）**不变**。
- **v2：modality flag 已砍**（P0 审查），`SyndromeEmbedder`/decoder/trainer 均不动，模型 forward 签名不变。

### 2.2 ③ 进阶 MSM（扩展 `mixed_msm.py::MixedStructuredMSM`，不替换）

现有 `MixedStructuredMSM`：40% RandomTokenMask / 30% SpatialClusterMask / 30% TemporalSpanMask（span_len=4，单稳定子连续轮）。

**③-a 轮级掩码 + 变长时序跨度**（强化 30% temporal 分支）
- 新增**整轮丢弃**：mask 某 k 轮的**全部**稳定子，从邻轮推断（强于单稳定子 span）。
- span_len 固定 4 -> **采样 2-8 变长**。
- 对症：LER 在 OOD 轮次 {1,13,30,50} 拟合 ε = 跨轮推断。

**③-b 轮次不变对比学习**（新增损失，叠加在 MSM 重建损失）
- 正样本对：同一 shot 的两种轮次视图（如 stride-2 抽样 5 轮 vs 全 10 轮）；负样本：batch 内其他 shot。
- InfoNCE 损失，权重 λ≈0.1，加到 MSM loss。
- 对症：强制轮次不变表示 -> 修掉 r=1 反转（当前 AQ 92-95% 错误）。**核心新组件**。

> ③-c 逻辑标签辅助任务**已砍**（保自监督纯粹）。

### 2.3 前置必修：轮次嵌入外推

- 现状：`CycleEmbedding`（`readout.py`）为 `max_rounds=50` 的**学习型索引查找**。**审查补全诊断**：训练仅在 r=10 -> **仅 embedding index 10 有梯度**；r=1/13/30/50 **全部使用未训练的随机初始化索引**（非仅 n≥50 走 MLP 回退），污染所有 OOD 轮次预测。
- **必修**：改为**连续正弦位置编码** `PE(pos,2i)=sin(pos/10000^(2i/d))`，对任意 n_rounds 外推；删越界 MLP 回退。是 ③-b 改善 OOD 轮次的前提。

### 2.4 因果链：③ -> LER 击败 MWPM

```
LER ε 在 {1,10,13,30,50} 拟合
  <- 当前 NN 在 r=30/50 塌缩到随机、r=1 反转
③-a 轮级掩码 -> 强化跨轮推断 -> 改善 r=13/30 预测
③-b 轮次不变对比 -> 修掉 r=1 反转 + 稳定 r=50
前置修复轮次嵌入外推 -> r=30/50 不再走未训练回退
  => LER 拟合点数 3/5 -> 5/5、ε 下降 -> 越过 MWPM ε
```

### 2.5 训练配置与时间

- 预训练：10× 数据，步数 10k->**20-30k**，lr 2e-4，warmup 500，cosine decay，AMP，bs 128-256。
- 微调：真机+合成掺杂，3k->**5-10k 步**，lr 1e-4。
- 时间：模型 ~2-3×/步 × 步数 2-3× ≈ 6-9× 预训练。d7 47min/10k -> ~5h/30k；三码距 pretrain+finetune ≈ **1-2 天 GPU**（后台串行）。

---

## 3. 评估、胜利线、消融矩阵、分阶段、风险

### 3.1 评估指标与胜利线

| 轨 | 评估集 | 指标 | 胜利线 |
|---|---|---|---|
| **A2 · LER** | 合成软读出 r{1,10,13,30,50}×20k | ε=(1-exp(slope))/2；R²≥0.9,\|logF₀\|≤0.2,ε>0,n_fit≥2 | **d3 BERT ε < MWPM 0.0109（主胜）**；d5 stretch |
| **A4 · Accuracy** | 真机硬读出 test（d3 45k/d5 20k/d7 5k） | 预测准确率 | **d3 BERT > MWPM 0.9125（主胜）**；d5 stretch；**d7 仅缩小差距** |

- LER 复用 `alphaqubit.evaluation.metrics.compute_ler`；MWPM 每轮独立 DEM。
- 主胜锁定 **d3 双指标**；d5 stretch；d7 受真机 40k 封顶，诚实不谎报。

### 3.2 消融矩阵

| 编号 | 模型 | 数据 | 掺杂 | ③ | 目的 |
|---|---|---|---|---|---|
| **E0** 回归基线（v2 强制） | 现 1.5M | 1× | 20% | off | CycleEmbedding 改动影响所有 NN，**必须用新代码+旧超参(128/4/2/4)重跑 d3**，对比旧 0.9027 建"无回退"参照（审查必改#3） |
| **E1** 纯规模① | 大 6-8M | 10× | 50% | off | ① 容量+数据+掺杂纯效果 |
| **E2** ①+③ 完整 | 大 | 10× | 50% | ③-a+③-b | **主推** |
| **E3** 掺杂有效性 | 大 | 10× | {0,20,50,80}%×{d3,d5,d7} | off | 掺杂有效性 + C1 复现（v2 砍 flag，聚焦掺杂比例×码距） |
| **E4** ③ 分解 | 大 | 10× | 50% | ③-a only / ③-b only / 两者 | 隔离 ③-a 与 ③-b 贡献 |

### 3.3 分阶段（审查门逐阶段）

| 阶段 | 内容 | 审查门 |
|---|---|---|
| **P1** 数据 | 10× 生成 d3/d5/d7（`generate_manifest.py --scale`）+ 删旧 1× + 修 `p`/seed(hashlib) + LER 保持 20k | 生成器改动须审 |
| **P2** ① 代码+基线 | 大模型 CLI + 掺杂 + **轮次嵌入正弦外推** + 路径 glob；E0 回归 + E1 训练 d3->d5->d7 | 代码审查 + smoke + E0 无回退 |
| **P3** ③ 目标（E2） | 加 ③-a + ③-b，d3 -> d5 -> d7 | mask_acc lift>0 |
| **P4** 消融（E3/E4） | 掺杂比例×码距、③ 分解 | - |
| **P5** 评估 | LER(d3/d5) + accuracy(d3/d5/d7) + 跨码距汇总 | eval_ler 代码须审 |
| **P6** 兜底（条件） | 若 ①+③ 未越线，启用 ② 蒸馏/级联，透明披露混合解码 | - |
| **P7** QC+报告 | figures + EXPERIMENT_REPORT 更新 | 最终 sign-off |

- 串行不并行（GPU 单卡）；d3 先行验证 pipeline。每阶段独立审查 subagent（审查组与代码组分离）。

### 3.4 风险与对策

| 风险 | 对策 |
|---|---|
| d7 大模型 OOM | bs128 + 梯度累积（预案已验证） |
| ① 纯规模仍饱和在 MWPM 下 | ③ 抬天花板；② 兜底 |
| ③-b 对比不 work/反伤 | E4 消融，可退回 ③-a only |
| 掺杂反伤（合成淹没真机） | E3 掺杂比例×码距 + WeightedRandomSampler real 加权（v2 砍 flag，无模态失配漏洞） |
| 轮次嵌入修复改变行为 | 先在现模型验证修复无回退 |
| LER 拟合点数不足 | 透明披露 n_fit_points；必要时加轮次点 |
| d7 accuracy 不可越 | 诚实定位为"缩小差距" |
| 时间 1-2 天 GPU | 后台串行任务 |

### 3.5 最终成功判据

- **主**：BERT(①+③) **d3 LER ε < MWPM** 且 **d3 accuracy > MWPM**。
- **stretch**：d5 LER 越线。
- **诚实**：d7 双指标显著缩小差距；d5 accuracy 缩小差距。
- **机制**：E3 证明掺杂增益随码距放大（C1 复现）；E4 证明 ③ 改善 OOD 轮次。

---

## 4. 范围与交付

### 4.1 本 spec 范围（in-scope）
- ① 纯规模基线（E1）+ ③ 进阶 MSM（E2）+ 掺杂有效性/③ 消融（E3/E4）。
- 数据 10× 生成、混合模态掺杂（无 flag）、轮次嵌入修复。
- d3/d5/d7 LER + accuracy 评估。

### 4.2 兜底（contingency，P6 条件触发）
- ② MWPM 蒸馏（软标签）或 NN+MWPM 置信度级联。仅在 ①+③ 未越线时启用，透明披露。

### 4.3 不在本 spec（out-of-scope，留 B/C/D）
- B 跨码距联合预训练 / 跨码族迁移 / 跨噪声模型。
- C 算法执行期间解码 / 实时流式解码。
- D 消融打磨发表（除 E3/E4 已含的核心消融）。

### 4.4 交付物
- `BEAT_MWPM_DESIGN.md`（本文件）
- 10× 数据（`data/d{3,5,7}/`，删旧）
- 代码改动：`generate_manifest.py`(--scale+seed hashlib)、生成器/真机制备补 `p`、`MixedStructuredMSM` 扩展 ③-a/③-b、`readout.py` 轮次嵌入正弦外推、`run_experiment.py`/`bert_pretrain.py`/`eval_ler.py` 适配大模型+路径 glob+掺杂（v2 砍 flag，embeddings/decoder/trainer 不动）
- `results_summary_d{3,5,7}.json` + `results_ler_d{3,5,7}.json` + 消融结果
- `figures/` 更新 + `EXPERIMENT_REPORT.md` 更新

---

## 5. 关键文件索引

| 文件 | 作用 |
|---|---|
| `code/generate_manifest.py` | 10× 量产（`--scale`）+ seed hashlib |
| `code/generate_google_paems_data.py` | `save_pt` 补 `p`（无 `--scale`，仅 smoke） |
| `bert_experiment/prepare_google_real.py` | 真机硬读出制备，补 `p` |
| `bert_experiment/mixed_msm.py` | `MixedStructuredMSM`，扩展 ③-a/③-b |
| `bert_experiment/xzzx_decoder.py` | `XZZXFineTuneDecoder`/`XZZXStabToData`（不变） |
| `bert_experiment/xzzx_coord.py` | `XZZXCoordinateSystem`（不变） |
| `bert_experiment/bert_pretrain.py` | 预训练入口，适配大模型 + ③ |
| `bert_experiment/run_experiment.py` | 三模型对比主脚本，适配大模型 + 掺杂 |
| `bert_experiment/eval_ler.py` | LER 评估（已审查批准），适配大模型 |
| `alphaqubit/models/pretrain_decoder.py` | `PretrainDecoder`/`FineTuneDecoder` |
| `alphaqubit/models/embeddings.py` | `SyndromeEmbedder`（v2 不动，flag 已砍） |
| `alphaqubit/evaluation/metrics.py` | `compute_ler` |
| `bert_experiment/EXPERIMENT_REPORT.md` | 既有报告，本实验完成后更新 |
