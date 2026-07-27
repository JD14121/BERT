# 击败 MWPM · 工程实施计划书 (P0) · v2

> 版本：2026-07-16 v2 ｜ 状态：**P0 终裁 APPROVE（8 必改 + 3 残留文档级条件全落实），可进 P1** ｜ 编制：代码组 ｜ 审查：审查组（独立 subagent）
> 遵循：前期规划 -> 审查批准 -> 执行（quantum_env）-> 逐阶段审查门（未批不执行）。
> 设计依据：`BEAT_MWPM_DESIGN.md` v2（已同步：砍 modality flag、E0 强制重跑）。
> v2 变更（应 P0 审查 8 必改）：①P1 命令改用 `generate_manifest.py`+新增 `--scale`；②补 `generate_manifest.py` seed 改 hashlib（可复现）；③**砍 modality flag**（消漏洞#8，免改 trainer/dataset/embeddings/decoder）；④E0 强制重跑；⑤LER 保持 20k 不放大（存储 ~94GB）；⑥CycleEmbedding 诊断补全（所有 OOD 轮次索引未训练）；⑦LER 拟合不对称披露；⑧阶段编号映射表 + 轮次嵌入修复归属 P2。

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development（推荐）或 executing-plans 逐阶段实施。本仓库**非 git**，无 commit；以"审查门 + 日志 + 检查点"作版本节点。

**Goal:** 在 Google 105Q 真机校准 XZZX 噪声下，用①纯规模基线 + ③进阶 MSM，使 BERT 在 d3 的 LER 与 accuracy 双指标击败 MWPM（d5 stretch，d7 缩小差距）。

**Architecture:** 复用 `bert_experiment/` 三模型框架。①放大模型（embed256/4层/6层）+ 10× 合成 + 混合模态掺杂（真机硬+合成软，无 flag，real 加权）；③在 `MixedStructuredMSM` 加轮级掩码（③-a）与轮次不变对比（③-b），并修 `CycleEmbedding` 为正弦外推。②蒸馏/级联条件兜底。

**Tech Stack:** PyTorch（quantum_env, RTX4070S 12GB）、stim 1.16、pymatching 2.4、`alphaqubit/`（PretrainDecoder/Trainer/compute_ler）。

---

## Global Constraints（工程铁律，用户 2026-07-16 重申）

- **环境**：代码开发/调试/训练在 conda **quantum_env**（`/d/condapy/quantum_env/python`，GPU）；数据生成在 conda **base**（`D:/anaconda/python.exe`）。命令前缀 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`。
- **审查机制**：审查组与代码组分离（独立 subagent）。每阶段产出须审查组 **APPROVE 后方可执行下一阶段**；新增/改动代码须代码审查后才运行。
- **捷径/造假零容忍**：禁数据造假、不合理假设、工程简化走捷径；任何简化须透明披露。真机 test 不入训练；LER 数据独立；eval()+no_grad；MWPM 同校准 DEM。
- **串行 GPU**：不并行训练（单卡 12GB）。
- **非 git**：以审查门 + `logs/` + `checkpoints/` 作版本节点。
- **HP 选择策略**（审查风险#4）：超参（mix_ratio/λ/步数）用**固定值**或在 **val 集**上选，**禁止据 test 表现调参**；报告须声明选择策略。

---

## 阶段编号映射（审查必改#7）

| 计划书阶段 | spec §3.3 阶段 | 内容 |
|---|---|---|
| P0 | - | 本计划书审查 |
| P1 | spec P1（数据部分） | 10× 数据生成 + schema 修复（p/scale/seed） |
| P2 | spec P1（代码部分）+ P2 前置 | ① 代码：模型 CLI + 掺杂 + CycleEmbedding 正弦外推 + 路径 glob |
| P3 | spec P2 | ① E1 训练（含 E0 回归基线） |
| P4 | spec P3 前置 | ③ 代码：③-a + ③-b |
| P5 | spec P3 | ③ E2 训练 |
| P6 | spec P4 | 消融 E3/E4 |
| P7 | spec P5 | LER+accuracy 评估 + 汇总 |
| P8 | spec P6 | ② 兜底（条件） |
| P9 | spec P7 | QC + 报告 |

> 轮次嵌入修复统一归 **P2**（模型代码改动），spec §3.3 原列 P1 已更正。

---

## 1. 数据资产与生成流程（P1）

### 1.1 现状（已核）
合成 train/val/test r10 各 800k/100k/100k @ `data/d{d}/`；LER r{1,10,13,30,50}×20k；真机 `data/real_d{d}/`（d3 450k/d5 200k/d7 50k=1 patch，**保留不删**）；校准 `configs/calibrated_d{d}.json`。

### 1.2 P1 生成流程（conda base）

**P1 代码改动（须审查）**：
1. **`generate_manifest.py` 增 `--scale`**：`argparse` 增 `--scale N`（默认 1）。循环内：`n = n * args.scale if split != "ler" else n`（**LER 保持 20k 不放大**，因 20k 足够拟合 E(n)，且省存储）。输出文件名含新 N。
2. **`generate_manifest.py::split_seed` 改 hashlib**（审查风险#5 可复现）：把 `abs(hash((split,d,rounds))) % 2**31 + 1` 改为 `int(hashlib.sha256(f"{split}_{int(d)}_{int(rounds)}".encode()).hexdigest(),16) % (2**31) + 1`（确定性，跨进程可复现）。`import hashlib`。
3. **`generate_google_paems_data.py::save_pt`（~L195-204）补 `'p': 0.0`**；**`prepare_google_real.py`（L85-94 pt dict）补 `'p': 0.0`**（`PTBatchDataset:50` 要求 `'p'` 字段，审查已确认现有数据含 p=0.0，用当前代码重生成会 KeyError）。

**生成命令**（d5 示例，d7 同 `--scale 10`，d3 用 `--scale 2`）：
```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 D:/anaconda/python.exe -u \
  google_paems_data/code/generate_manifest.py --distance 5 --scale 10 --chunk-size 5000 \
  2>&1 | tee google_paems_data/logs/gen_d5_10x.log
```
（量产脚本即 `generate_manifest.py`，非 `generate_google_paems_data.py`--后者无 `--scale/--manifest`，仅作 smoke/单文件）

**删旧 1× 合成**（用户已批准，留余量；非必须）：删 `data/d{3,5,7}/train|val|test_d{d}_r10_n800000|n100000_*.pt`。**真机 `data/real_d{d}/` 与 LER 20k 严禁删**。

**P1 验收**：形状正确（d5 meas[N,10,24] final_soft[N,25]；d7 meas[N,10,48] final_soft[N,49]）；label_rate/det_dens 与 1× 一致；`'p'` 字段存在；`PTBatchDataset` 加载无 KeyError；MWPM sanity d3≈0.91/d5≈0.97/d7≈0.97。**可复现性**：同命令重跑生成字节一致（hashlib seed）。
**P1 审查门**：`--scale`/seed/p 三处改动 + 删除清单 + 生成校验 -> 审查组批准。

---

## 2. 代码变更清单（file structure，v2 砍 flag 后）

| 文件 | 改动 | 阶段 |
|---|---|---|
| `code/generate_manifest.py` | 增 `--scale`；`split_seed` 改 hashlib | P1 |
| `code/generate_google_paems_data.py` | `save_pt` 补 `'p':0.0` | P1 |
| `bert_experiment/prepare_google_real.py` | pt dict 补 `'p':0.0` | P1 |
| `bert_experiment/run_experiment.py` | 模型超参 CLI 化(256/8/4/6)；合成 .pt 路径 n800000/n100000 改 glob（L83-84）；`--mix-synth-ratio` 默认 0.5；`WeightedRandomSampler` real 加权 | P2 |
| `bert_experiment/bert_pretrain.py` | 合成 .pt 路径改 glob（L46）；模型超参 CLI 一致；③-b 对比损失头 + `--use-contrastive`/`--contrastive-lambda`；接入 ③-a | P2/P4 |
| `bert_experiment/eval_ler.py` | 模型超参 CLI；合成 .pt 路径改 glob（L47,76）；适配大模型 | P7 |
| `alphaqubit/models/readout.py` | `CycleEmbedding`（L107-162）`nn.Embedding(50,dim)` -> 正弦位置编码，删越界 MLP 回退 | P2 |
| `bert_experiment/mixed_msm.py` | ③-a：`_full_round_mask` + 变长 span[2,8] | P4 |

> **砍 flag 影响**：`embeddings.py`/`decoder.py`/`pretrain_decoder.py`/`trainer.py`/`pretrain_trainer.py`/`xzzx_decoder.py` **均不动**（模型 forward 签名不变，调用仍 `model(m,e,lk,el,fs,n_rounds=...)`）。审查漏洞#8 消除。

---

## 3. P2：① 代码实现（模型 CLI + 掺杂 + CycleEmbedding 正弦 + 路径 glob）

> **P2 审查门**：全部代码改动须审查组代码审查 APPROVE 后才进 P3。

### 3.1 模型超参 CLI 化
`run_experiment.py`（L99,110,113-114）、`eval_ler.py`（L105-106,116,119-120）硬编码 128/4/2/4 -> CLI `--embed-dim 256 --n-heads 8 --num-transformer-layers 4 --num-readout-layers 6`。`bert_pretrain.py`（L30-32 已 CLI embed/heads/transformer_layers；PretrainDecoder 无 readout 层，仅保证与微调 embed/heads/layers 一致）。验收：参数 ~6-8M；d7 bs128 前向无 OOM。

### 3.2 合成 .pt 路径 glob 化（审查必改#2）
`run_experiment.py:83-84` 硬编码 `train_d{d}_r{r}_n800000_Z.pt`/`n100000` -> 10× 后 N 变为 8000000/1000000 会找不到。改为 glob `train_d{d}_r{r}_n*_Z.pt`（取最大 N 或 CLI `--train-n`）。`bert_pretrain.py:46`、`eval_ler.py:47,76` 同改。真机路径已是 glob（L82）。

### 3.3 混合模态掺杂（砍 flag，审查漏洞#8 已消）
现状（L116-122）：`ConcatDataset([real_train, synth_sub])`，ratio 0.2，合成为软读出。
改：
- `--mix-synth-ratio` 默认 0.2 -> **0.5**。
- **不阈值化**：合成保持软读出（连续 [0,1]），真机硬读出（0/1）。两者直接混合，模型自然兼容（hard 0/1 ⊂ soft [0,1]，原 20% 掺杂已验证可行）。**无 modality flag** -> accuracy 评估（真机硬）与 LER 评估（合成软）均与训练混合分布一致，无 train/eval 模态失配漏洞。
- **real 加权**：`WeightedRandomSampler`（real 权重 = `len_synth/len_real` × synth 权重），保证有效 real:synth≈1:1，防 10× 合成（8M）淹没真机（d7 仅 40k）。
验收：单测一个 epoch 内 real 被采次数 ≈ synth；混合 batch 含 0/1 与连续值。

### 3.4 CycleEmbedding 正弦外推（审查必改#5 诊断补全）
现状：`CycleEmbedding`（`readout.py:107-162`）= `nn.Embedding(50, dim)` 查找（L130）+ n≥50 越界 MLP 回退（L150-156）。**审查补全诊断**：训练仅在 r=10 -> **仅 embedding index 10 有梯度**；r=1/13/30/50 **全部使用未训练的随机初始化索引**（非仅 n≥50 走 MLP）。被 `decoder.py:176`/`readout.py:285`/`pretrain_decoder.py:415` 复用。
改：`nn.Embedding` 查找 -> **正弦位置编码** `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(...)`，对任意 n_rounds 外推；删 L150-156 越界回退。三处复用自动受益。
验收：单测 n_rounds=50/70 前向不越界；**E0 回归校验**（见 P3）对比修复前后 LER E(n) 在 r=1/13/30/50 不再异常。

---

## 4. P3：① 基线训练（E0 回归 + E1）d3->d5->d7

### 4.1 E0 回归基线（审查必改#3 强制）
CycleEmbedding 改动影响所有 NN。E0 须用**新代码 + 旧超参（128/4/2/4）+ 1× 数据**重跑 d3，建立"无回退"参照（对比 `EXPERIMENT_REPORT` 旧 d3 数字 0.9027）。若 E0 与旧数字显著偏离，须排查 CycleEmbedding 副作用后再进 E1。
```bash
... bert_pretrain.py --distance 3 --steps 10000 --embed-dim 128 --n-heads 4 --num-transformer-layers 2
... run_experiment.py --distance 3 --embed-dim 128 --n-heads 4 --num-transformer-layers 2 --num-readout-layers 4 --mix-synth-ratio 0.2
```
验收：E0 d3 accuracy 与旧 0.9027 偏差 < 1pp（允许正弦嵌入带来的小幅变化）；LER E(n) 在 OOD 轮次不再异常跳变。

### 4.2 E1 = 大模型 + 10× + 50% 掺杂 + ③ off，d3->d5->d7
```bash
... bert_pretrain.py --distance 3 --steps 25000 --embed-dim 256 --n-heads 8 --num-transformer-layers 4 --mask-ratio 0.25
... run_experiment.py --distance 3 --embed-dim 256 --n-heads 8 --num-transformer-layers 4 --num-readout-layers 6 \
    --mix-synth-ratio 0.5 --aq-pretrain-steps 25000 --finetune-steps 8000
```
（d5/d7 同理；d7 bs 降 128 + 梯度累积预案；d7 预训练步数视塌缩恢复可延至 30k）
验收：BERT mask_acc lift>0；MWPM sanity 同 P1；无 OOM；d3 BERT accuracy 是否 > 0.9125（主胜观察点）。
**P3 审查门**：E0 无回退 + E1 smoke + MWPM sanity + 无造假 -> 批准。

---

## 5. P4：③ 代码实现（③-a + ③-b）

> **P4 审查门**：③ 代码须审查组代码审查 APPROVE 后才进 P5。

### 5.1 ③-a 轮级掩码 + 变长 span（mixed_msm.py）
`_temporal_mask`（L64-76）span_len 固定 4 -> 每次采样 `span_len ∈ [2,8]`。新增 `_full_round_mask(m,T,n_stab,target_count)`：随机 t0 + k∈[1,3]，mask `[t0:t0+k, :]` 全部稳定子（整轮丢弃）。`_generate_mask_indices` 四策略混合（如 30/25/25/20）。验收：mask 覆盖率 ≈ mask_ratio；整轮 mask 选中轮 stab 全 True；span ∈ [2,8]。

### 5.2 ③-b 轮次不变对比（bert_pretrain.py + 新损失）
InfoNCE，权重 λ（`--contrastive-lambda 0.1`）。正样本对：同 shot 两轮次视图（全 T 轮 vs stride-2 抽 5 轮插值回 T）；负样本：batch 内其他 shot。`L = L_MSM + λ·L_NCE`，τ=0.5。需 `PretrainDecoder` 暴露池化特征（若无加 `encode(...)->pooled`）。`--use-round-mask`/`--use-contrastive` 开关，默认 off（保 E1 可复现）。
验收：正样本相似度 > 负样本；λ=0 退化原 MSM；mask_acc 不被拖垮。
> **审查科学性提示**：③-b 对 BERT 的边际贡献须 E4 验证，不预设（BERT 已无 r=1 反转，③-b 主要强化 OOD 轮次泛化，非"修反转"）。

---

## 6. P5：③ 训练 E2（d3->d5->d7）
E2 = E1 配置 + ③-a + ③-b on。
```bash
... bert_pretrain.py --distance 3 --steps 25000 --embed-dim 256 --n-heads 8 --num-transformer-layers 4 \
    --mask-ratio 0.25 --use-round-mask --use-contrastive --contrastive-lambda 0.1
... run_experiment.py --distance 3 ... --mix-synth-ratio 0.5 ...
```
验收：mask_acc lift>0；对比损失收敛；d3 LER 拟合点数 ≥ E1 且 ε 下降；d3 accuracy 观察是否 > 0.9125。
**P5 审查门**：③ 学习信号正常 + LER 改善 + 无造假 -> 批准。

---

## 7. P6：消融 E3（掺杂×码距）+ E4（③ 分解）

### 7.1 E3 掺杂有效性 × 码距（C1 复现）
- `--mix-synth-ratio {0,0.2,0.5,0.8}` × `--distance {3,5,7}`，大模型、③ off。12 格。输出 Δacc=acc(ratio)−acc(0) 随码距曲线。假说：Δacc 随 d 放大（C1 复现）。
- **审查风险#2 披露**：各格用 5k finetune 小步数看趋势，须声明"消融趋势基于小步数，可能不外推到全训练（8-25k）"。
- （砍 flag 后无 modality-matched/flag 子臂；消融聚焦掺杂比例×码距。）

### 7.2 E4 ③ 分解
③ {off, ③-a only, ③-b only, ③-a+③-b} × d3/d5/d7，掺杂 0.5。输出各组件对 LER OOD 轮次与 ε 贡献。
验收：趋势符合假说或诚实反例；各格独立训练不共享 test。
**P6 审查门**：消融设计合理 + 一致性 + 小步数限制已披露 -> 批准。

---

## 8. P7：评估（LER + accuracy + 跨码距汇总）

> **P7 审查门**：`eval_ler.py` 适配大模型 + 路径 glob + **更新 L45-46 旧诊断警告（CycleEmbedding 正弦修复后"n≥50 走 MLP 回退"已过时）** 的增量改动须代码审查。

- accuracy：`run_experiment.py` 输出 `results_summary_d{d}.json`。E0/E1/E2/消融各一份。
- LER：`eval_ler.py --distances 3 5 7`，复用 `compute_ler`；MWPM 每轮独立 DEM（`eval_ler.py:65-83` 已正确，num_det=n·n_stab）。
- **LER 有效性**：`is_valid=(R²≥0.9) 且 (|log F₀|≤0.2) 且 (ler>0)`；`results_ler_d{d}.json` 记录每模型每轮 E(n)/F(n)、实际拟合点数、R²/log F₀/slope/is_valid。
- **审查必改#6 LER 拟合不对称披露**：MWPM 5/5 点 F>0.1 全拟合；NN 3-4/5 点（OOD 退化点 F<0.1 被 min_fidelity 预过滤）。**ε 比较基于不同点集**，系统有利于 NN 的 ε 数值。胜利判定与报告须显式标注此不对称（非造假，n_fit_points 已透明记录）。
- 跨码距图：`plot_report.py` 扩展 d3/d5/d7 accuracy+LER。

---

## 9. P8：② 兜底（条件触发）
仅当 ①+③ 在 d3 未越线时启用，**透明披露混合解码**：蒸馏（MWPM 软标签）或级联（NN 低置信转 MWPM，级联 ≥ MWPM）。报告标注"② 为混合解码，非纯 NN 越线"。
**P8 审查门**：② 设计审查 + 透明披露 -> 批准。

---

## 10. P9：QC + 报告 + SELF_MEMORY
QC：结果与日志逐位核对；趋势物理合理；OOD 现象合法披露。更新 `EXPERIMENT_REPORT.md`（E0/E1/E2/消融 + 胜负 + LER 不对称标注）。`SELF_MEMORY.md` Evolution Log 追加。最终审查组 sign-off。

---

## 11. 资源需求与时间节点（串行，quantum_env + RTX4070S 12GB）

| 阶段 | 内容 | 预估 |
|---|---|---|
| P1 | 10× 生成（train/val/test only，LER 保持 20k）+ p/scale/seed 修复 + 删旧 | ~30 min |
| P2 | ① 代码实现 + 代码审查 | ~2 h |
| P3 | E0 回归(d3) + E1 训练 d3->d5->d7 | ~9-13 h |
| P4 | ③ 代码实现 + 代码审查 | ~2 h |
| P5 | ③ E2 训练 d3->d5->d7 | ~8-12 h |
| P6 | 消融 E3(12)+E4(12) 小步数 | ~6-10 h |
| P7 | 评估 + 图 | ~1 h |
| P8 | ② 兜底（条件） | ~3-5 h |
| P9 | QC + 报告 | ~1-2 h |
| **总计** | | **~2-3 天 GPU**（串行后台） |

**存储（审查必改#4 更正）**：10× 仅 train/val/test（LER 保持 20k）。d5 train 24+val 3+test 3=30GB；d7 train 47+val 6+test 6=59GB；d3 ~2GB；LER 20k 已有 ~3GB。合计 **~94GB**，D: 余 121GB，删旧 1× 后余量 ~27GB 充足。

---

## 12. 质量控制标准

- **MWPM sanity**：d3≈0.91/d5≈0.97/d7≈0.97（与 P1 一致）。
- **E0 无回退**：新代码+旧超参 d3 acc 与旧 0.9027 偏差 <1pp；LER OOD 轮次无异常跳变。
- **LER 有效性**：`is_valid=(R²≥0.9) 且 (|log F₀|≤0.2) 且 (ler>0)`；记录实际拟合点数；**披露 MWPM 5/5 vs NN 3-4/5 不对称**。
- **胜利判定**：d3 BERT LER ε<0.0109 且 acc>0.9125（主胜）；d5 LER 越线（stretch）；d7 缩小差距（诚实）。
- **d3 LER 胜利可达性披露（审查风险#3）**：当前 BERT 在 d3 **每一轮** E(n) 均高于 MWPM（含 r=10 训练轮）。③ 主要改善 OOD 轮次泛化，对 r=10 改善有限。d3 LER 越线非常激进，若未达成须如实报告而非调参凑数。
- **掺杂有效性（C1 复现）**：Δacc 随码距放大趋势（或诚实反例）；小步数限制已披露。
- **HP 策略**：固定值或 val 调优，禁 test 调优；报告声明。
- **可复现性**：hashlib seed，同命令字节级一致。
- **无造假**：真机 test 不入训练；LER 独立；eval()+no_grad；MWPM 同校准 DEM；消融各格独立；简化（d7 bs128、E3 小步数、LER 保持 20k）均透明披露。

---

## 13. 风险与缓解

| 风险 | 缓解 |
|---|---|
| d7 大模型 OOM | bs128 + 梯度累积（预案） |
| ① 纯规模仍饱和在 MWPM 下 | ③ 抬天花板；P8 ② 兜底 |
| ③-b 对比不 work/反伤 | E4 消融，退回 ③-a only |
| CycleEmbedding 正弦改动有副作用 | P3 E0 回归校验无回退 |
| LER 拟合点数不足/不对称 | 透明披露 n_fit_points 与不对称；必要时加 r=70/90 |
| d3 LER 越线过激进 | 诚实报告，不调参凑数 |
| 消融小步数不外推 | 披露限制；关键格全步数复核 |
| HP 过拟合 test | 固定值/val 调优，禁 test 调优 |
| 时间 2-3 天 | 后台串行；d3 先行 |
| seed 可复现 | hashlib（已修） |

---

## 14. 交付物
- 10× 数据 `data/d{3,5,7}/`（train/val/test 10×，LER 保持 20k；删旧 1× 合成；保留真机）
- 代码：`generate_manifest.py`(--scale+seed)、生成器/真机制备补 p、`run_experiment.py`(CLI+glob+掺杂+sampler)、`bert_pretrain.py`(glob+③-b)、`eval_ler.py`(CLI+glob)、`readout.py`(CycleEmbedding 正弦)、`mixed_msm.py`(③-a)
- `results_summary_d{3,5,7}_{E0,E1,E2}.json` + 消融 + `results_ler_d{3,5,7}_{E1,E2}.json`
- `figures/` 更新；`EXPERIMENT_REPORT.md` + `SELF_MEMORY.md` 更新

---

## 审查组请就以下逐项终裁（APPROVE / APPROVE_WITH_CONDITIONS / REJECT）

1. **科学性**：③-a/③-b 对症 LER OOD；CycleEmbedding 正弦外推；胜利线（含 d3 LER 激进性披露）。
2. **可行性**：`generate_manifest.py --scale` 方案、存储 ~94GB、d7 显存、2-3 天、砍 flag 后改动面是否准确。
3. **实验设计**：E0 回归 + E1/E2 + E3(C1) + E4 隔离贡献；三模型协议一致；MWPM 每轮独立 DEM。
4. **捷径/造假核查**：合成冒充真机/d7 封顶掩盖/LER 截断/消融 test 泄漏/② 伪装/真机 test 入训练/MWPM DEM 不同源。
5. **逻辑一致性**：8 项必改是否全部落实；阶段映射表；与 spec v2 一致；审查漏洞#8（砍 flag）是否真正消除。
