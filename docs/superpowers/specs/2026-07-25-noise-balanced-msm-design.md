# 噪声位平衡 MSM（Noise-Balanced MSM, NB-MSM）实验设计 v2

> 日期：2026-07-25（v2 修订：2026-07-26）
> 主题：测试"掩码中被掩位的 defect(event=1)/非defect(event=0) 位 1:1 平衡"能否让 BERT-1 5.5M 在 d5/d7 突破
> 工程铁律：审查组/代码组分离、逐阶段审查门、quantum_env 训练、禁数据造假/走捷径、透明披露

> **v2 修订原因**：v1 基于"event 密度 ~6%/3%"的错误估算设计 per-shot 采样，审查组 P0 抓出 B1/B2 Blocker。2026-07-26 实测 event 密度（probe_event_density.py）推翻该估算，v2 改用 **batch 级池采样** + 加入 **d3 作对照锚点** + 用物理论证重定 1:1 分界为 event（D3）。

---

## 1. 背景与假设

### 1.1 实测的三层标签分布（d3/d5 train 800k，2026-07-26 探测）

数据中有**三个层级**的"不均"物理含义不同，必须区分：

| 层级 | 统计 | d3 | d5 | d7（仅 event）| 物理含义 | MSM 是否预测 |
|---|---|---|---|---|---|---|
| **shot 级 label** | label=1 占比 | 28.78% | 34.66% | — | 逻辑 observable 翻转（量子信息被破坏） | 否（微调才预测） |
| **位级 event** | event=1 占比 | 47.26% | 35.73% | 29.90% | 稳定子在相邻轮**检出 defect**（物理错误直接证据） | 是（event = soft_XOR(meas[t],meas[t-1])） |
| **位级软值不确定度** | u=1-\|2p-1\|>0.2 占比 | 0.62% | 0.62% | — | 读出器**看不清**该位（I/Q 落两高斯峰重叠区） | 是（MSM 重建软值） |

measurement 软值双峰：d5 58.5%<0.1 + 37.6%>0.9，仅 0.62% 在 [0.4,0.6]。这正是用户最初引用的"96% 平凡位"——它描述的是**读出层确定度**，不是"有噪声位"占比。

### 1.2 对 MSM 的影响（物理论证）

MSM 掩掉一个 measurement 位让模型从空间邻居 + 时序历史重建。模型学的是 syndrome 时空条件分布，编码两件事：**错误链推断**（从 defect 模式推错误走向）+ **抗读出噪声**（从含噪软值推真测量）。下游微调预测 label=逻辑错误，由 syndrome 错误链决定，故**预训练应为错误链推断服务**。

- **event=1（defect 位）重建难也最有价值**：defect 意味错误发生，软值受"错误 + 读出噪声"双重影响，须靠邻居 defect 模式推断错误链走向——高码距解码核心难点。
- **event=0（非 defect 位）重建最平凡**：无错误，软值主要由读出噪声决定，邻居大概率也无错，几乎"照抄邻居"。
- **软值不确定度 0.62% 是读出层信息缺失**，不是物理错误位；1:1 拉到 50% = 偏读出层补全、不针对错误链，且 0.62% 过极端会扭曲预训练表示。

### 1.3 核心假设
把掩码中被掩位的 "defect(event=1) : 非defect(event=0)" 从自然占比（d5 36:64 / d7 30:70）拉到 **1:1**，BERT 预训练在 defect 位上等量练习错误链推断，预训练表示质量提升，进而让 d5/d7 微调上限突破历史 5.5M 基线（d5 0.8721 / d7 0.7782）。

### 1.4 物理旁证（d3 作锚点）
d3 event=1 天然 47.26%（**近 1:1**）且 BERT 已超 MWPM；d5/d7 event=1 偏离 1:1（36%/30%）且未突破。这反向支持"event 1:1 是关键"。故 **d3 入实验作对照锚点**：验证 1:1 在已近平衡码距上不退步（NB-MSM 在 d3 上应 ≈ 原 MSM，作 sanity）。

### 1.5 噪声位定义（D3，物理干净）
- **defect 位**：被掩位所在稳定子在当轮 `event > 0.5`。
- **非 defect 位**：`event ≤ 0.5`。
- `event` 张量由数集直接提供（shape 与 measurement 同 `[N, T, n_stab]`），无需重算。

---

## 2. 干预点（唯一变量）

**只改 MSM 掩码选择**。新增 `NoiseBalancedMSM`（继承 `MaskedSyndromeModeling`，与 `MixedStructuredMSM` 同级）：

- 输入：batch `measurement [B, T, n_stab]` + `event [B, T, n_stab]`。
- 噪声掩码：`noise_mask = event > 0.5`（逐位布尔）。
- 目标掩码覆盖率 = `mask_ratio`（默认 0.25，与原一致）→ 每样本 `target = mask_ratio * T * n_stab`，每 batch `B * target` 个被掩位。
- **batch 级池采样（v2 关键修正）**：把全 batch 的 defect 位索引拼成池 `P_defect`，非 defect 位拼成池 `P_clean`，各无放回采样 `B * target / 2` 个，再按 shot 分配回 `[B, T, n_stab]`。**不 per-shot 采样**（per-shot 在 d5/d7 缺口 2×/4× 不足以 1:1；batch 级池在三码距都 2.4-3.8× 充足，见 §4.2）。
- **不足补足**：若 `P_defect` 总数 < `B * target / 2`（实测不会发生，但留 guard），用非 defect 位补足并记录补足比例（透明披露）。
- 结构先验（空间/时序）默认 **off**，保证与原 MSM 对照变量单一。本轮 `NoiseBalancedMSM` 构造不引入 `coord_system`（留接口，避免误导；后续若加需扩签名）。

### 2.1 不动的部分
- 数据：E 盘现有 d3/d5/d7 合成 800k + 真机硬读出 + LER，**不重新生成**（实测 batch 级池充足，无需 d7 过采样）。
- 模型：5.5M（embed=192/6h/3层/4R），`local_train_5m.py` 配置不动。
- Loss：不改（不加不确定度加权）。
- 微调协议：不改。

---

## 3. 对照设计（配对）

| 臂 | 掩码策略 | 数据 | 模型 | 种子 | 预训练步数 | 微调步数 |
|---|---|---|---|---|---|---|
| Arm-0 对照 | 原 `MixedStructuredMSM`（40/30/30） | E 盘 d{d} | 5.5M | 42 | 20000 | 8000 |
| Arm-1 干预 | `NoiseBalancedMSM`（1:1 defect） | 同上 | 5.5M | 42 | 20000 | 8000 |

- 唯一变量 = 掩码策略。
- **d3/d5/d7 各两臂** = 6 次预训练 + 6 次微调。d3 作对照锚点（§1.4 物理旁证：d3 event 已近 1:1，NB-MSM 应 ≈ 原 MSM，作 sanity，不期望主胜）。
- 主胜若达成，补 seed=123 复核（对齐跨噪声工程 seed-2 训，单 seed 显著性不可信）。
- 种子说明：两臂同初始种子 42，但干预路径不同导致 np.random 流分叉（Arm-0 用 choice(strats)/rand，Arm-1 用 argwhere/choice(idx)）——这是干预本身的差异，非 bug；报告披露"同初始种子，干预路径分叉"。

---

## 4. 数据

### 4.1 现有资产（E 盘，已核查）
- `data/d3/train_d3_r10_n800000_Z.pt`、`val_d3_r10_n100000_Z.pt`
- `data/d5/train_d5_r10_n800000_Z.pt`（2.4GB）、`val_d5_r10_n100000_Z.pt`
- `data/d7/train_d7_r10_n800000_Z.pt`（4.8GB）、`val_d7_r10_n100000_Z.pt`
- `data/real_d3/{train, val, test}`、`data/real_d5/{train 160k, val 20k, test 20k}`、`data/real_d7/{train 40k, val 5k, test 5k}`
- `data/d{3,5,7}/ler_d{d}_r{1,10,13,30,50}_n20000_Z.pt`
- 均旧 `SoftReadoutSimulator`（snr=10）生成。

### 4.2 噪声位池充足性核算（v2 实测，2026-07-26 probe_event_density.py）

**实测 event 密度**（推翻 v1 的 ~6%/3% 估算）：
| 码距 | T×n_stab | event=1 密度 | 每 shot defect 位数（均值）|
|---|---|---|---|
| d3 | 10×8=80 | 47.26% | ~38 |
| d5 | 10×24=240 | 35.73% | ~86 |
| d7 | 10×48=480 | 29.90% | ~144 |

**batch 级池 1:1 可达性**（bs=256, mask_ratio=0.25, 每样本 target=0.25×T×n_stab，每 batch 需 `B×target/2` defect 位）：
| 码距 | 每 batch 需 defect 位 (B×target/2) | 256-shot 池期望 defect 位数 (256×密度×T×n_stab) | 比值 | 结论 |
|---|---|---|---|---|
| d3 | 256×20/2=2560 | 256×0.4726×80=9692 | 3.79× | 充足（d3 event 已近 1:1，干预 uplift 小）|
| d5 | 256×60/2=7680 | 256×0.3573×240=21952 | 2.86× | 充足 |
| d7 | 256×120/2=15360 | 256×0.2990×480=36740 | 2.39× | 充足 |

**结论**：d3/d5/d7 全量 800k 做 batch 级池采样，1:1 在三码距上都 2.4-3.8× 充足，**无需 d7 过采样高密度子集**（v1 的过采样分支弃用）。补足比例应≈0。

---

## 5. 模型与训练协议

- 5.5M：embed=192 / 6 heads / 3 Transformer 层 / 4 readout = 5.22M finetune / 4.98M pretrain。
- 预训练：20k 步，bs=256，lr=2e-4，mask_ratio=0.25，AMP，eval_interval=500。
- 微调：8k 步，bs=256，lr=1e-4，mix_ratio=0.2（真机 + 20% 合成掺杂，对齐历史 0.9330 协议）。
- 评估：test accuracy + LER（r∈{1,10,13,30,50}，复用 `eval_ler.py`）。
- 环境：`/d/condapy/quantum_env/python`（quantum_env，RTX 4070 SUPER），`PYTHONIOENCODING=utf-8 PYTHONUTF8=1`。

---

## 6. 验收标准

### 6.1 主胜
Arm-1 在 d5 或 d7 上 test acc **同时**满足：
- 超 Arm-0（配对 bootstrap p<0.05，per-sample 预测 npz）；
- 超历史 5.5M 基线（d5 0.8721 / d7 0.7782）。
- **d3 不期望主胜**（event 已近 1:1，uplift 小），作 sanity：Arm-1 应 ≈ Arm-0，不退步即过。

### 6.2 次胜
LER：Arm-1 < Arm-0（跨轮次趋势 + 拟合 ε）。

### 6.3 归因证据（必交）
- 记录两臂实际被掩位的 event 分布：**Arm-0 应 ≈ 自然 event 密度**（d3 47% / d5 36% / d7 30%，因原 MSM 不区分位类别），**Arm-1 应 ≈ 50%**（证 1:1 干预落实）。v2 修正：v1 写的"Arm-0≈4%"基于错误密度估算，实测证伪。
- 记录 Arm-1 补足比例（应≈0，证 batch 级池充足，§4.2 比值 2.4-3.8×）。

### 6.4 负面结果透明
若 Arm-1 不胜，如实报告。审查 1:1 是否反而引入分布偏移：预训练人为 1:1（defect 50%）vs 微调真机分布（defect 自然密度 d5 36% / d7 30%），可能拉大域差距——这是实验要回答的问题，非缺陷。

---

## 7. 工程铁律落实（逐阶段审查门）

| 阶段 | 内容 | 审查门 |
|---|---|---|
| P0 | 本设计文档 | 审查组（独立 subagent）审查计划书科学性/可行性 |
| P1 | 实现 `NoiseBalancedMSM` + driver 脚本 + 单测 | 审查组代码审查（逻辑/语义/缺陷/造假/捷径） |
| P2 | 6 次预训练 + 6 次微调（d3/d5/d7 × 两臂） | 训练日志监控 |
| P3 | eval_ler + 配对 bootstrap + 归因统计 | 审查组评估审查 |
| P4 | QC 汇总 + 报告 | 审查组最终 sign-off |

- 训练在 quantum_env；数据生成不涉及（用现有）。
- 任何"工程简化"透明披露（如噪声位补足）。
- 数据造假/不合理假设零容忍。

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 1:1 人为偏移拉大预训练-微调域差距 | 实验本身回答；若 Arm-1 < Arm-0，证伪假设，如实报告 |
| 噪声位成簇导致邻居也被掩→推断锚点丢失 | 1:1 是位级混合（非整簇难），平凡位仍占一半作锚点；若严重可加"邻居不全掩"约束（本轮先不加，待 P3 观察） |
| 单 seed 显著性 | 主胜后补 seed=123，两 seed 一致才裁决 |
| d7 5.5M 容量不足（历史 0.7782） | 本实验只测掩码策略变量，容量限制是基线属性，两臂同等承受 |

---

## 9. 交付物

- `NoiseBalancedMSM` 实现 + 单测
- driver 脚本（基于 `local_train_5m.py` 扩展，支持 --mask-strategy {original, noise_balanced}）
- 4 组 ckpt + 训练历史 JSON
- 配对 bootstrap 结果 + 归因统计（被掩位 event 分布）
- `results_summary_nb_msm.json` + 实验报告
- SELF_MEMORY.md Evolution Log 追加