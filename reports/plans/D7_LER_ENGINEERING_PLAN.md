# D7 扩展 + LER 评估 · 工程计划书 (P0)

> 版本：2026-07-15 ｜ 状态：**审查组 APPROVE_WITH_CONDITIONS→条件已落实·批准执行** ｜ 编制：代码组 ｜ 审查：审查组（独立 subagent）
> 遵循：前期规划→审查批准→执行（quantum_env）→逐阶段审查门。

---

## 0. 背景与目标

Google-BERT 三模型对比实验（AlphaQubit 基准 / BERT=Ours / MWPM）已完成 **d3、d5** 的 test accuracy 对比（真机硬读出微调）。本工程两项目标：

1. **d7 全流程**：把同一实验协议扩展到 d=7（数据→预训练→全量对比）。
2. **LER 评估**：为 d3/d5/d7 三模型补 LER（Logical Error per Round）指标，当前 `run_experiment.py` 只有 accuracy、无 LER。

**为何需要**：d3/d5 accuracy 已显示 BERT>AlphaQubit（高码距预训练增益更大）但远低于 MWPM；LER 是 AlphaQubit Nature 2024 的核心指标，衡量"每轮累积逻辑错误率"，比单点 accuracy 更能反映解码器随轮次的稳定性，是论文级对比的必备项。

> ⚠️ **重要声明（审查组要求披露）**：
> - **LER 评估在合成 PAEMS 校准数据上进行（软读出 snr=10.0），非真机硬件数据**。真机 Google 数据仅有 r=10 单轮次，无法支持多轮 LER。故 accuracy 对比在真机硬读出上、LER 对比在合成软读出上--双轨设计，结果报告/论文须同步标注。
> - **读出模态失配**：NN 在真机硬读出（snr=inf）上微调，LER 在合成软读出（snr=10.0）上评估，存在分布偏移；预训练阶段见过软读出（缓解）。MWPM 非学习型无此问题。

## 1. 数据资产盘点（已逐项验证 ✅）

| 资产 | d3 | d5 | d7 | 备注 |
|---|---|---|---|---|
| 合成 train/val/test (r10) | ✅ | ✅ | ✅ 800k/100k/100k | `data/d{d}/` |
| LER 数据 r{1,10,13,30,50}×20k | ✅ | ✅ | ✅ | `data/d{d}/ler_d{d}_r{n}_n20000_Z.pt` |
| PAEMS 校准配置 | ✅ | ✅ | ✅ | `configs/calibrated_d{d}.json` |
| Google 源 stim+b8 (r10) | ✅ | ✅ | ✅ q6_7/50k shots | `Google-data/.../d7_at_q6_7/Z/r10/` |
| 真机硬读出 .pt | ✅ 450k | ✅ 200k | ❌ **待生成** | `data/real_d{d}/` |
| BERT 预训练 ckpt | ✅ | ✅ | ❌ **待训练** | `checkpoints/bert_pretrain_d{d}/` |
| LER 评估代码 | ❌ | ❌ | ❌ | `run_experiment.py` 无 LER 逻辑 |

**d7 几何**：n_stab=d²−1=48，n_data=d²=49，num_det=r×48（r10→480，实测一致）。Google 源 r10 shots=50000（单 patch q6_7）。

## 2. 数据生成流程

**P1：d7 真机硬读出数据**（复用既有 `prepare_google_real.py`，已审查过的代码，仅运行）：
```
prepare_google_real.py --distance 7 --basis Z --rounds 10
```
- 源：`d7_at_q6_7/Z/r10/{measurements,detection_events,obs_flips_actual}.b8`（50k shots）
- 输出：`data/real_d7/{train,val,test}_d7_r10_n{40000,5000,5000}_Z.pt`（80/10/10，seed=42）
- schema 与 d3/d5 一致：measurement[N,r,48]/event[N,r,48]/final_soft[N,49]/label[N]/detection_events[N,480]，硬读出 snr=inf
- **风险**：d7 仅 1 patch → 40k train（vs d5 160k、d3 360k）。**缓解**：合成预训练 800k 主导特征学习，真机仅做 3k 步微调；BERT 已含 20% 合成掺杂。

## 3. 代码变更

### 3.1 d7 全流程（无新代码，复用 `run_experiment.py`）
`run_experiment.py` 已参数化 `--distance`，d7 直接运行。需先有 `bert_pretrain_d7/best.pt`：
```
bert_pretrain.py --distance 7 --steps 10000   # P2
run_experiment.py --distance 7                 # P3
```
**显存风险**：d7 节点数 97（vs d5 49，~2x）。bs=256 可能 OOM → 预案降 bs=128（不变则 256）。

### 3.2 LER 评估（**新增** `eval_ler.py`，需审查组代码审查）
独立脚本，加载已训练 checkpoint，在 LER 数据上评估，复用 `alphaqubit.evaluation.metrics.compute_ler`（**不自造拟合**）。

**设计**：
```
for d in [3,5,7]:
    cs = make_coord(d,'Z',10)
    aq   = XZZXAlphaQubitDecoder(...); load aq_finetune_d{d}/best.pt
    pre  = PretrainDecoder(...); load bert_pretrain_d{d}/best.pt
    bert = XZZXFineTuneDecoder(pretrained_encoder=pre,...); load bert_finetune_d{d}/best.pt
    for model_name, model in [('alphaqubit',aq),('bert',bert)]:
        preds_by_r, labels_by_r = {}, {}
        for n in [1,10,13,30,50]:
            ds = PTBatchDataset(ler_d{d}_r{n}.pt)
            preds = (sigmoid(model(m,e,0,0,fs,n_rounds=n))>0.5)   # 逐 batch
            preds_by_r[n]=preds; labels_by_r[n]=ds.label
        ler = compute_ler(preds_by_r, labels_by_r)   # LERResult
    # MWPM: 每轮次独立建 DEM（num_det=n*48）
    mwpm_preds_by_r, mwpm_labels_by_r = {}, {}
    for n in [1,10,13,30,50]:
        dem = inject_surface_code_noise(generate_surface_code_circuit(d,n,...),cfg).detector_error_model()
        mwpm = pymatching.Matching.from_detector_error_model(dem)
        ds = PTBatchDataset(ler_d{d}_r{n}.pt)
        mwpm_preds_by_r[n] = mwpm.decode_batch(ds.detection_events)
        mwpm_labels_by_r[n] = ds.label
    mwpm_ler = compute_ler(mwpm_preds_by_r, mwpm_labels_by_r)
    save results_ler_d{d}.json
```

**关键正确性点**（审查重点）：
- `compute_ler` 输入 `{rounds: preds}`+`{rounds: labels}`，每轮 labels 来自该轮 LER 数据自身（非共享）。
- NN 在 r=10 训练，r={1,13,30,50} 为 OOD 评估——这是 AlphaQubit Nature 2024 标准 LER 协议（单轮训练→多轮评估），接受。
- MWPM 每轮次建独立 DEM（DEMs 轮次相关，num_det=n·n_stab），不复用 r=10 DEM。
- 三模型评估协议一致：同 LER 数据集、同阈值（0.5）、同 compute_ler。

## 4. 资源需求与时间节点（串行，quantum_env + RTX4070S 12GB）

| 阶段 | 内容 | 预估耗时 |
|---|---|---|
| P1 | d7 真机数据生成（stim 极快） | ~1 min |
| P2 | d7 BERT 预训练 10k steps | ~25 min |
| P3 | d7 全量实验（AQ 10k pretrain+3k finetune / BERT 3k finetune / MWPM） | ~2–3 h（d7 节点 97 vs d5 49，每步 2–4× 慢，审查组修正） |
| P4 | LER 评估 d3/d5/d7（加载 ckpt，5轮×3模型×3码距） | ~15 min |
| P5 | QC + 汇总 + 可视化 | ~10 min |
| **总计** | | **~3–4 h**（审查组上调） |

GPU 串行（不并行，避免 OOM）。d7 bs 预案 128。

## 5. 阶段划分与审查门

| 阶段 | 内容 | 审查门 |
|---|---|---|
| **P0** | 本计划书 | **科学性/可行性/捷径审查 → 待批准** |
| P1 | d7 真机数据生成 + smoke 形状校验 | 复用既有代码（已审查）；运行结果校验 |
| P2 | d7 BERT 预训练 | mask_acc lift>0（学习信号）、pretrain>scratch |
| P3 | d7 全量实验 | MWPM d7 sanity、loss/acc 合理性、无 OOM |
| P4 | LER 评估代码 + 运行 | **新增代码审查** + LER R²≥0.9、无造假 |
| P5 | QC + 汇总 | 跨码距 Λ 合理性、结果一致 |

**准则**：每阶段通过审查后方可进下一阶段；新增代码（eval_ler.py）须审查组代码审查后才运行。

## 6. 质量控制标准

- **MWPM d7 accuracy** ≈ 0.976（与 Google 数据生成 sanity d7=0.9763 同量级；真机 test 集允许偏差）。
- **label_rate** d7 合理区间（~0.4–0.55），与合成 d7 一致性可接受。
- **LER 有效性判据（已更正）**：`is_valid = (R²≥0.9) 且 (|log F₀|≤0.2) 且 (ler>0)`（含 ler>0，审查组指出原稿漏写）；`F(n)>0.1` 是 fit_ler 的 min_fidelity **预过滤**，非 is_valid 判据。
- **拟合透明度（审查组必改）**：`results_ler_d{d}.json` 须记录每模型每轮 E(n)/F(n)、min_fidelity 过滤后**实际参与拟合点数**、R²/log F₀/slope/is_valid。弱解码器（NN 高码距）r=30/r=50 的 F 可能跌破 0.1 被过滤仅剩 3 点（3 点 2 参数 R²≥0.9 几乎必然通过、统计力弱）-> 必须披露各模型实际拟合点数，不得只报 LER 数值。
- **同 shot 一致性**：detection_events 与 label matched（spec §3.2，Google-PAEMS 工程已验证 m2d 逐位一致）。
- **无造假**：真机 test 不参与训练；LER 数据独立于 train/val/test；评估用 eval()、no_grad；MWPM 用同校准 DEM。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| d7 仅 1 patch → 40k 真机 train | 合成预训练主导 + 20% 掺杂 |
| d7 OOM | 降 bs 至 128 |
| r=1/50 OOD 评估不准 | 标准 LER 协议，接受；R² 不达标则标记 invalid |
| d7 BERT 预训练无学习信号 | P2 验证 mask_acc lift>0，否则排查 |
| LER .pt schema 跨轮次不一致 | P4 先校验 r=1/r=50 的 schema 与字段 |

## 8. 交付物

- `data/real_d7/{train,val,test}_d7_r10_*.pt`
- `checkpoints/{bert_pretrain_d7,aq_pretrain_d7,aq_finetune_d7,bert_finetune_d7}/`
- `results_summary_d7.json`（accuracy）
- `eval_ler.py` + `results_ler_d{3,5,7}.json`（LER）
- 跨码距 d3/d5/d7 accuracy+LER 对比图
- SELF_MEMORY Evolution Log 更新

---

## 审查组请就以下逐项裁决（APPROVE / APPROVE_WITH_CONDITIONS / REJECT）

1. **科学性**：LER 方法（复用 compute_ler，log F vs n 线性拟合，ε=(1−exp(slope))/2）是否正确；OOD 多轮评估协议是否合理。
2. **可行性**：d7 数据/显存/时间估算是否可靠；1 patch 40k train 是否足以支撑结论。
3. **实验设计**：三模型评估协议是否一致；MWPM 每轮独立 DEM 是否必要且正确。
4. **捷径/造假核查**：是否存在合成冒充真机、LER 拟合不当截断、MWPM DEM 与 test 不同源等问题。
5. **逻辑一致性**：阶段划分、审查门、QC 标准是否自洽。
