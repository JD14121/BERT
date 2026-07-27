# 云端 d7（34M 数据）实验报告

> **版本**：2026-07-17 ｜ **环境**：云端 A800-SXM4-40GB（40GB）+ 34M 合成数据 + 大模型 + bs256
> **目标**：验证"尽可能多数据 + 大模型"对 d7（最难码距）的提升，对比 1× 基线与 MWPM。

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 服务器 | NVIDIA A800-SXM4-40GB（40GB ECC），8 核 CPU，15GB RAM，246GB LVM 数据盘 |
| 数据 | d7 合成 train **34M**（34×1M 分片，XZZX，Google 校准噪声，软读出 snr=10）；val/test 100k；LER r{1,10,13,30,50}×20k；真机 d7 40k(1 patch) |
| 模型 | PretrainDecoder 大模型：embed 256 / 4 层 Transformer(n_heads=8) / readout 6 层，~11.8M 参数 |
| 训练 | bert_pretrain 20k 步 + run_experiment（AQ 20k pretrain + AQ/BERT 8k finetune + MWPM），bs256，AMP，50% 合成掺杂 |
| 框架 | torch 2.6.0+cu124（驱动 550=CUDA12.4），stim 1.16，pymatching 2.4 |
| 关键工程 | 分片生成（generate_one 预分配>15GB RAM -> 34×1M 分片）、PTBatchDataset mmap=True（15GB RAM 加载 34M）、LVM 合并数据盘 |

---

## 2. 结果

### 2.1 Test Accuracy（真机硬读出 d7）

![fig1](figures/fig1_accuracy_vs_distance.png)

| 模型 | 云 d7（34M+大模型） | 本地 d7（8M+bs64，原 1× 小模型基线 0.7438） | vs MWPM |
|---|---|---|---|
| MWPM | 0.9702 | 0.9702 | - |
| **BERT** | **0.8702** | 0.7438（1×小模型） | **−10pp**（差距 22.6pp->10pp 大幅缩小） |
| AlphaQubit | 0.5996（塌缩） | 0.6982 | from-scratch 大模型塌缩 |

### 2.2 LER（合成软读出 d7）

![fig2](figures/fig2_ler_vs_distance.png)

| 模型 | 云 d7 LER (fit) | 本地 d7（1× 小模型 0.0402） | MWPM |
|---|---|---|---|
| **BERT** | **0.0181** (5/5, R²=0.992) | 0.0402 (2/5) | 0.00268 (5/5) |
| MWPM | 0.00268 (5/5) | 0.0027 | - |
| AQ | 0.0782 (3/5) | 0.0655 | - |

### 2.3 训练曲线

![fig3](figures/fig3_d7_bert_pretrain.png) ｜ ![fig4](figures/fig4_d7_bert_finetune.png)

- **BERT 预训练**（34M）：val mask_acc 39.7%(step0) -> 88.1%(step20k) plateau，val_loss 0.3045 Best 持续保存。
- **BERT 微调**（真机 40k + 50% 合成）：val_acc 69%(step500) -> 87%(step8k)，**健康无塌缩**（pred_pos_rate 0.38-0.44 稳定）。

---

## 3. 核心发现

### 3.1 数据规模效应（d7 BERT）
| 数据/模型 | BERT accuracy | BERT LER |
|---|---|---|
| 1× + 小模型（原基线） | 0.7438 | 0.0402 |
| **34M + 大模型（云）** | **0.8702** | **0.0181** |
| **提升** | **+12.6pp** | **−55%** |

34M 数据 + 大模型把 d7 BERT accuracy 从 0.7438 推到 0.8702（+12.6pp），LER 从 0.0402 砍到 0.0181（−55%）。**数据规模是高码距提升的关键杠杆**。

### 3.2 BERT vs AlphaQubit：预训练防塌缩
- **AQ（from-scratch，大模型）在 d5/d7 塌缩**：d5 AQ 0.6465、d7 AQ 0.5996（pred_pos_rate=0%，全负类，val_acc=多数类基线）。大模型 from-scratch 在高码距不稳定。
- **BERT（34M 预训练）免疫塌缩**：d5 BERT 0.9058、d7 BERT 0.8702，微调 pred_pos_rate 健康。
- **BERT > AQ 优势随码距放大**：d3 持平（0.933≈0.934）-> d5 +26pp -> d7 +27pp。**自监督预训练的价值在高码距（数据稀缺+决策复杂）凸显，强力验证核心论点 C1**。

### 3.3 跨码距总览（大模型 ① E1）
| d | 数据 | BERT acc | MWPM acc | BERT LER | MWPM LER |
|---|---|---|---|---|---|
| 3 | 2× | **0.9330** ✓主胜 | 0.9125 | 0.0111 | 0.0107 |
| 5 | 10× | 0.9058 | 0.9428 | 0.0106 | 0.0035 |
| 7 | 34M(云) | 0.8702 | 0.9702 | 0.0181 | 0.00268 |

- **d3 accuracy 主胜**：BERT 0.9330 > MWPM 0.9125（① 纯规模即越线）。
- d5/d7：BERT < MWPM（高码距 MWPM 近最优），但 BERT 随数据规模持续逼近（d7 差距 22.6pp->10pp）。
- **mask_acc 天花板 ~87%**（跨 d3/d5/d7 一致，大模型+34M 未打破）-> **③ 进阶 MSM 是打破天花板、进一步逼近 MWPM 的关键**。

### 3.4 d3 LER 近主胜
d3 BERT LER 0.0111 vs MWPM 0.0107（差 0.0004，5/5 拟合）。③-a（轮级掩码）预期越过此线，完成 d3 双指标主胜。

---

## 4. 工程亮点（云端 15GB RAM + 40GB GPU 突破）

| 挑战 | 方案 |
|---|---|
| generate_one 预分配 34M 数组（61GB）> 15GB RAM | 分片生成：34×1M shards（每片 6GB，RAM 安全）+ shard_dataset.ConcatDataset 加载 |
| PTBatchDataset torch.load 全量 RAM | patch `mmap=True`（34 shards 加载仅 <1GB RAM） |
| 单盘 149GB < 34M(294GB) | LVM 合并 vdb+vdc = 246GB 数据盘 |
| 驱动 550=CUDA12.4（不支持 cu128） | torch 2.6.0+cu124（项目代码兼容） |
| 本地 12GB GPU d7 bs128 OOM | 云端 A800 40GB，bs256（17GB 用，余 23GB） |

---

## 5. 结论

1. **34M 数据 + 大模型显著提升 d7**：BERT accuracy +12.6pp（0.7438->0.8702）、LER −55%（0.0402->0.0181），与 MWPM 差距从 22.6pp 缩至 10pp。**数据规模是高码距的核心杠杆**。
2. **d3 accuracy 主胜达成**：BERT 0.9330 > MWPM 0.9125。
3. **预训练防塌缩论点强力验证**：BERT（预训练）在高码距稳定，AQ（from-scratch 大模型）塌缩；BERT>AQ 优势随码距放大。
4. **MWPM 在高码距仍最强**（近最优），但 BERT 随数据/模型规模持续逼近。
5. **下一步**：③-a（轮级掩码）打破 mask_acc 天花板、越过 d3 LER 线（0.0107）；③-b（轮次不变对比）修 d5 r=1 反转；更大模型/更多数据继续逼近 d5/d7 的 MWPM。

---

## 6. 交付物（cloud_d7/）
- `results_summary_d7_E1_cloud.json`（accuracy）
- `results_ler_d7_E1_cloud.json`（LER + per-round E/F）
- `bert_pretrain_d7_best.pt`（143MB，34M 预训练 encoder）
- `bert_finetune_d7_best.pt`（145MB，d7 BERT 微调）
- `train_d7_bert_pretrain.log` / `train_d7_run_experiment.log`（训练曲线）
- `figures/fig{1-4}_*.png`（accuracy/LER vs distance + 训练曲线）
- 本报告
