# 工程计划书：Google 真机数据 BERT 解码实验（AlphaQubit / MWPM / Ours=BERT）

> 版本：v1.0  日期：2026-07-14
> 任务：基准 AlphaQubit + MWPM，Ours=BERT；合成数据预训练 + 真机数据微调
> 环境：conda quantum_env（GPU RTX4070S）
> 流程准则：计划->审查组审查->批准->执行->代码审查->QC，未通过审查不得进入下一阶段

---

## 1. 任务定义

三模型对比解码实验：
- **AlphaQubit 基准**：同架构从头训练（全监督）
- **MWPM 基准**：PyMatching + 校准配置 DEM
- **Ours = BERT**：BERT 风格自监督预训练（MSM）+ 微调

数据策略：
- **预训练**（AlphaQubit 从头 + BERT MSM）：用**合成数据**（`google_paems_data/data/`，XZZX，Google 校准噪声，**含软读出**，已生成 3.3M 样本）。BERT 预训练**无标签**。
- **微调**（BERT）+ **AlphaQubit 全监督训练**：用**真机 Google 数据**（`google_paems_data/Google-data/`，XZZX，**硬读出**-用户决议），允许掺杂部分合成数据若提高精度。
- **参考**：BERT 设计参考 `deliverables/report/syndrome_bert_technical_report.md` + `deliverables/syndrome_bert_proposal.md`，并**重新设计掩码规则**（用户要求，最好有改进）。

### 1.1 成功标准
- [S1] 三模型在真机 Google 测试集上输出 test accuracy + LER
- [S2] BERT 预训练 mask accuracy > 85%（学习到 syndrome 表示）
- [S3] BERT 微调后 test accuracy 可与 AlphaQubit/MWPM 横向对比；LER 拟合 valid
- [S4] 真机硬读出微调与合成软读出预训练的域差距可控（不崩溃到随机）

---

## 2. 数据

### 2.1 合成数据（预训练用，已生成）
`google_paems_data/data/d{3,5,7}/`：XZZX，Google 校准噪声，**软读出**（measurement/event/final_soft ∈[0,1]），每码距 train 800k/val 100k/test 100k @r10 + LER。BERT 预训练用 train（无标签，仅 measurement/event）。

### 2.2 真机 Google 数据（微调用，须准备）
`google_paems_data/Google-data/google_105Q_surface_code_d3_d5_d7/`：XZZX，b8 格式，**硬读出**（measurements.b8 = 0/1），含 detection_events.b8 / obs_flips_actual.b8 / circuit_ideal.stim / metadata.json。
**须写转换脚本** `prepare_google_real.py`：把 b8 转为 .pt schema（与合成数据同字段），但 measurement/event/final_soft 为**硬 0/1**：
- `measurement [N,T,n_stab]` = measurements.b8 的 ancilla 部分（硬 0/1）
- `event [N,T,n_stab]` = detection_events.b8 reshape（硬 XOR）
- `final_soft [N,n_data]` = measurements.b8 末轮 data 部分（硬 0/1）
- `label [N]` = obs_flips_actual.b8
- `detection_events [N,num_det]` = detection_events.b8
- 选 patch：d3=q10_7, d5=q8_7, d7=q6_7（与合成数据一致）；rounds 用 r10 主 + LER {1,10,13,30,50}
- train/val/test 划分：Google 每 (patch,basis,rounds) 50k shots，按 80/10/10 切（或按 rounds 留出）

### 2.3 域差距说明
预训练软读出 [0,1] vs 微调硬读出 {0,1}。硬读出是软读出的极限情形（置信度无穷大）。decoder 的 SyndromeEmbedder(Linear(measurement)) 对 0/1 与 [0,1] 都可处理，域差距可控；可选微调时掺少量合成软数据缓解。

---

## 3. 关键技术挑战与方案

### 3.1 XZZX 坐标系适配（核心）
- 现有 `CoordinateSystem(distance)` 默认旋转表面码布局；`PTBatchDataset` 用默认。
- **方案**：`CoordinateSystem` 已支持 `circuit` 参数（`_parse_circuit_coordinates` 从 QUBIT_COORDS+测量顺序解析稳定子坐标）。对 XZZX，构造 `CoordinateSystem(d, circuit=google_circuit_ideal.stim)` 提取 XZZX 真实稳定子网格位置。
- **须验证（P1）**：`_stim_to_grid(x//2,y//2)` 对 Google XZZX 坐标是否正确（Google 坐标间距可能非 2）；若不符需适配 grid 映射。新建 `XZZXPTBatchDataset`（或 PTBatchDataset 加 circuit 参数）用 XZZX 坐标系。
- decoder（AlphaQubitDecoder/PretrainDecoder）接口不变，仅 coord_system 换为 XZZX 版。

### 3.2 真机硬读出微调
- 真机数据 measurement/event/final_soft = 硬 0/1（§2.2）。decoder 直接消费（Linear 输入）。
- 不合成软读出（用户决议）。微调时可掺合成软数据（混合 batch）若提高精度。

### 3.3 BERT 掩码规则重设计（用户核心要求）
现有 MSM（deliverables）：随机 mask 25% token，80%->0.5/10%随机/10%保留，同时 mask measurement+event，target=measurement，BCE。
**重设计：Mixed Structured MSM**（强制学习空间+时序+局部三类推断）：
- 每 batch 混合 3 种掩码子策略（按样本分配）：
  - **40% RandomTokenMask**：原 BERT 式随机 token 掩码（局部推断）
  - **30% SpatialClusterMask**：mask 空间相邻稳定子簇（用 XZZX 坐标系邻接，半径 r 内整簇 mask）-强制空间推断
  - **30% TemporalSpanMask**：mask 同一稳定子连续 k 轮（SpanBERT 风格）-强制时序推断
- 总 mask_ratio ≈ 0.25-0.30；对 measurement 与 event 应用同一 mask（保持 event=soft_xor(meas_t,meas_{t-1}) 物理一致性）
- target = 原始 measurement；loss = soft-BCE（适配软读出预训练数据）仅算 mask 位
- 保留 80/10/10 替换规则（0.5/随机/保留）
- **改进点**：结构化掩码（簇+跨度）比纯随机更难，迫使模型学空间拓扑与时序传播，预期下游迁移更好（deliverables §9.2 也建议此方向）

### 3.4 rounds 不一致
- 合成与真机主数据均 r10；BERT 预训练 r10，微调 r10。CycleEmbedding 帮助泛化不同 rounds。LER 评估用 {1,10,13,30,50}。

---

## 4. 实验流程

| 阶段 | 模型 | 数据 | 任务 | 输出 |
|---|---|---|---|---|
| P1 | 数据准备 | 真机 b8->.pt（硬读出）+ XZZX 坐标系验证 | 烟测 | 真机 .pt + XZZX coord 跑通 |
| P2 | BERT 预训练 | 合成 train（软读出，无标签） | Mixed Structured MSM | 预训练 encoder |
| P3 | AlphaQubit 基准 | 真机 train（硬读出，有标签） | 全监督分类 | from-scratch 模型 |
| P4 | BERT 微调 | 真机 train（硬读出，可掺合成软） | 分类微调 | 微调模型 |
| P5 | MWPM 基准 | 真机 detection_events | DEM 解码 | 预测 |
| P6 | 评估 | 真机 test + LER | acc + LER | results_summary |

先做哪码距：**d3 起步**（最小，快验证全流程），跑通后 d5/d7。

---

## 5. 资源/时间/QC
- GPU RTX4070S 12GB；quantum_env
- 预训练 ~1.5-3h（d3 20k 步）；微调/AlphaQubit ~1h；MWPM 分钟级
- 磁盘：真机 .pt 小（硬 0/1 紧凑），合成已 12G
- QC：[S1] 三模型 test acc+LER；[S2] mask acc>85%；[S3] LER valid；[S4] 域差距不崩溃（BERT 微调 acc 显著>0.5）

---

## 6. 审查机制（审查组 vs 代码组）
- 代码组：quantum_env 实现，隔离在 `google_paems_data/bert_experiment/`
- 审查组：计划审查（本阶段）+ 代码审查（逻辑/语义/缺陷）+ 实验设计评估 + 反走捷径 + 一致性验证
- 反走捷径：① 真机微调必须用真 Google 数据（不得用合成冒充）② BERT 预训练确实无标签 MSM ③ 掩码规则如实实现（结构化掩码不得退化为纯随机）④ MWPM 用校准配置 DEM ⑤ 域差距如实报告

---

## 7. 待用户决策（批准前）
1. **起始码距**：d3 起步（推荐，快验证）还是直接 d3/d5/d7 全做？
2. **微调掺杂**：是否允许微调 batch 掺合成软数据（默认允许，若提精度）？
3. **掩码重设计**：Mixed Structured MSM（40%随机/30%空间簇/30%时序跨度）是否采纳？或调整比例？
4. **AlphaQubit 基准训练数据**：用真机硬读出全监督（与 BERT 微调同数据，公平对比）？
5. **批准 P1**（数据准备 + XZZX 坐标系烟测）？
