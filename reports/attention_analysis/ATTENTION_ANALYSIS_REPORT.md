# ⑤ Attention 分析报告：Attention 是否学到了 DEM 错误关联结构？

> 版本：2026-07-20 ｜ 分析对象：d7 XZZX, r10, Z 基, **BERT 预训练 100k 步** checkpoint（`bert_pretrain_d7/best.pt`, global_step=96000）
> 模型：PretrainDecoder 大模型（embed 256 / 4 层 SyndromeTransformer × 8 heads / readout 6 层, ~11.8M 参数）
> 环境：云端 V100-SXM2-32GB（CPU-only 运行, 21.4s）｜ 审查组 v2 APPROVE_WITH_CONDITIONS 全部落实
> 脚本：`analyze_attention_v3.py`（忠实 patch 版，随结果存档于本目录）

---

## 0. 背景与方法

**目标**：检验预训练 BERT 的 attention 是否从数据中学到了量子纠错的错误关联结构（以 DEM 经验关联矩阵为 ground truth）。

**模型架构**（核实服务器实际代码，与本地 `transformer.py` 一致）：
```
For round t in [0..9]:
    RNN state = (state + syndrome_embed[t]) * 0.7        ← 时序递推（0.7 缩放）
    per round: SyndromeTransformer × 4 层
        attention = softmax(QK^T/√d + spatial_bias)      ← QK^T 与偏置混杂
all_states: [B, 10, 48, 256]                              ← 10 轮时序表示
```
- `spatial_bias = distance_embed(成对距离) + learned_bias`（[n_heads, 48, 48]）
- `MultiHeadSelfAttention.forward` 含 Pre-LN + 残差（`x + output`）

**DEM 经验关联**：用 20000 样本的 `detection_events` [N,480] 计算 Pearson 相关 [480,480]，折叠为 [48,48]（10 轮同轮块平均，**假设轮间平稳**）。0/480 detector 从不触发（无 NaN）。

**相关分析**：排除对角线，32 个 (layer×head) 组合全部报告，Bonferroni α=0.05/32=0.00156。

---

## 1. v2 脚本的 3 个 bug（已修复，结果不可信）

审查组 v2 批准后，执行中发现 v2 脚本 `patched_fwd`（monkey-patch 捕获 attention）有 3 个正确性 bug，导致**之前打印的 r 值（0.03/0.06）失真**：

| Bug | v2 错误 | v3 修复 |
|---|---|---|
| **[1] patch 不忠实** | `q_proj(x)` 直接用原 x，漏 `layer_norm`；`return out_proj(out)` 漏残差 | 精确复刻真实 forward（`x_norm=layer_norm(x)` → Q/K/V；`return x+output`） |
| **[2] 只捕获第 9 轮** | `last_attn_weights` 每轮覆盖，10 轮 forward 后只剩末轮 | `_attn_history` list 累积全部 10 轮，对 (round,batch) 求平均 |
| **[3] JSON 崩溃** | `numpy.bool_` 不可序列化 | 显式 `bool()` + numpy-aware default |

另补齐 v2 缺失的 **QC#3 预训练 vs 随机初始化对比**。

---

## 2. 核心结果

### 2.1 总览（mean / max Pearson r，排除对角线，Bonferroni α=0.00156）

| 条件 | mean r | max r | sig (n/32) | 说明 |
|---|---|---|---|---|
| **full**（QK^T+bias，预训练） | **+0.157** | +0.384 | **20/32** | 模型实际 attention |
| **qk_pure**（关 bias，纯 QK^T，预训练） | +0.028 | +0.092 | 6/32 | 真·QK^T 通路（反事实） |
| **qk_no_learned**（learned_bias=0，QK^T+距离先验，预训练） | +0.156 | +0.392 | 20/32 | free-form 偏置置零 |
| **bias_only**（softmax 全 bias，预训练） | +0.186 | +0.394 | 30/32 | 偏置单独 |
| random.full（随机初始化） | -0.009 | +0.154 | 16/32 | 对照 |
| random.qk_pure（随机初始化） | -0.006 | +0.034 | 0/32 | 对照 |

### 2.2 逐 head 深度模式（full attention）

| Layer | 8 个 head 的 r | 特征 |
|---|---|---|
| 0（输入层） | 0.00, -0.09, 0.04, 0.03, 0.05, 0.03, 0.02, 0.03 | 几乎全弱，仅 H1 负相关 sig |
| 1 | 0.01, 0.00, **0.34**, -0.06, -0.06, **0.23**, **0.33**, **0.12** | H2/H5/H6/H7 强 |
| 2 | **0.34**, **0.22**, 0.06, **0.08**, **0.36**, **0.16**, -0.13, **0.37** | 6/8 强（H6 负） |
| 3（最深） | **0.33, 0.28, 0.23, 0.37, 0.38, 0.31, 0.32, 0.31** | **全部 8 head 强正相关（0.23–0.38）** |

**最优 head**：L3H4，r=0.383（p=6e-80）。**深度效应**：越深的层 attention 与 DEM 对齐越强，Layer 3 全员显著。

### 2.3 qk_pure 通过 Bonferroni 的 6 个 head

L3H0(0.092)、L2H7(0.081)、L1H6(0.081)、L3H3(0.076)、L2H0(0.076)、L3H6(0.067)——集中在深层特定 head，QK^T 通路确实学到少量 DEM 结构。

---

## 3. QC 裁决

| QC | 标准 | 结果 | 数值 |
|---|---|---|---|
| QC#1 | qk_pure r > 0（从数据学到物理结构） | **PASS** | +0.028 > 0（弱但正） |
| QC#2 | qk_pure > bias_only（是"学到"非"记忆"） | **FAIL** | +0.028 < +0.186 |
| QC#3 | 预训练 qk_pure > 随机 qk_pure（预训练有效） | **PASS** | +0.028 > -0.006 |
| QC#4 | 全 32 head 报告 + Bonferroni | **PASS** | 32/32 报告，α=0.00156 |
| QC#5 | 排除对角线 | **PASS** | `~np.eye(48)` |
| QC#6 | eval + no_grad | **PASS** | `model.eval()` + `torch.no_grad()` |

---

## 4. 诚实解读

### 4.1 预训练确实让 attention 对齐 DEM（成立）
- 预训练 full attention mean r=+0.157（20/32 sig，深层 L3 全员 0.23–0.38），随机初始化 mean r=-0.009（≈0）。
- **预训练 vs 随机：明确信号**。预训练使 attention 获得了与错误关联结构统计一致的空间模式。

### 4.2 但对齐主要由 spatial bias 承载，非 QK^T 内容通路（不成立"强主张"）
- `bias_only`（mean 0.186, 30/32 sig）**显著强于** `qk_pure`（mean 0.028, 6/32 sig）→ QC#2 FAIL。
- `full`(0.157) ≈ `qk_no_learned`(0.156)：**free-form `learned_bias` 贡献几乎为零**（置零后 r 几乎不变）。
- 真正承载 DEM 对齐的是 `distance_embed`（基于成对距离的函数式空间先验）+ QK^T 的联合，其中**距离先验是主导项**（qk_pure 0.028 → qk_no_learned 0.156，加入距离先验后 r 跃升 5.6×）。
- `bias_only`(0.186) > `full`(0.157)：加入 QK^T 反而略微"稀释"了纯偏置的 DEM 相关（QK^T 相对 DEM 引入噪声）。

### 4.3 QK^T 通路学到少量但真实的 DEM 信号
- `qk_pure` mean r=+0.028 > 随机 -0.006，6/32 head 通过 Bonferroni（L3H0/L2H7/L1H6/L3H3/L2H0/L3H6）。
- 这是**预训练在 Q/K 投影中注入的、超越空间先验的可学习信号**，但量级小（r~0.03–0.09）。

### 4.4 综合结论
> **预训练 BERT 的 attention 与 DEM 错误关联结构存在统计显著的对齐（深层尤甚，L3 全员 r=0.23–0.38），但该对齐约 80–90% 由空间偏置的函数式距离先验承载，QK^T 内容通路仅贡献少量（r~0.03–0.09）但显著的学习信号。**
>
> 因此，**"attention 学到了错误关联图"的强主张仅部分成立**：模型确实通过预训练获得了 DEM 一致的空间注意模式（vs 随机明显），但这更多是"空间归纳偏置 + 少量 QK^T 学习"的产物，而非 QK^T 通路主导的、从数据中自由学得的错误关联表示。

---

## 5. 透明声明（残留项）

1. **QK^T-only 是反事实分析**：`qk_pure` 通过 `use_spatial_bias=False` 重跑 forward 获得，改变了深层输入分布，非"同一次 forward 的分解"。`bias_only` 是静态 softmax(bias)，非 forward。解读已区分"学得"（qk_pure，弱）vs"注入"（bias，强）。
2. **DEM 折叠假设轮间平稳**：[480,480] 折叠为 [48,48] 时对 10 轮同轮块取平均，假设轮间关联平稳。d7 r10 下合理但非严格。
3. **Bonferroni 分母=32**（4 层 × 8 head），非 320（未对 4 种条件分别校正）；条件间独立报告。
4. **DEM 是经验共激活相关**（detection_events 的 Pearson），非 stim DEM 的字面边权重；作为"哪些 detector 统计耦合"的代理。
5. B=64 样本平均 attention（DEM 用全 20000 样本）；attention 模式跨样本稳定，B=64 足代表。

---

## 6. 交付物

| 文件 | 内容 |
|---|---|
| `summary.json` | 总览 mean/max r + sig + QC 裁决 |
| `correlation_full.json` | 预训练 full attention 逐 head r/p/sig |
| `correlation_qk_pure.json` | 预训练纯 QK^T 逐 head |
| `correlation_qk_no_learned.json` | 预训练 learned_bias=0 逐 head |
| `correlation_bias_only.json` | 预训练 bias 单独逐 head |
| `correlation_random.json` | 随机初始化 full + qk_pure 逐 head |
| `fig_attention_vs_dem.png` | 最优 head attention 热图 + DEM + 散点 |
| `fig_per_head.png` | 32 head 的 full r 柱状图（sig 着色） |
| `fig_decomposition.png` | full/qk_pure/qk_no_learned/bias_only 分解柱状图 |
| `fig_pretrained_vs_random.png` | QC#3 预训练 vs 随机 qk_pure 散点 |
| `analyze_attention_v3.py` | 忠实 patch 分析脚本（存档） |

---

## 7. 下一步候选

1. **更高 mask_acc 模型**：当前 100k 步 mask_acc 仅 88.5%。若 ③-b 对比学习或更长训练打破天花板，QK^T 通路 r 可能提升——重测可验证"学到"是否增强。
2. **分层 readout 分析**：L3 全员强对齐，是否对应 readout 对 L3 表示的更高依赖？
3. **轮次维度分析**（计划 §5）：RNN all_states [B,10,48,256] 的表示几何随轮次变化、逻辑错误可分性。
4. **d3/d5 对比**：低码距下 QK^T 通路占比是否不同（d3 主胜已达成，attention 结构或更"学得"）。
