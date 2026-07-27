# d7 微调优化工程计划书 v2（修订版，二审）

> 版本：2026-07-20 v2（依审查组 REJECT 修订）｜ 状态：待二审 ｜ 环境：云端 V100-32GB
> 基线：d7 R5（BERT acc 0.8664, LER 0.013661, MWPM 0.9702/0.002680）
> v1 审查裁决：REJECT（C1/C2 对称群高估 D4->C2、C5 mix_ratio 破坏单变量、M1 验证门无量化判据、M2 C 多变量、M3 stage2 训练量过低；C4 已澄清为审查组误搜本地，d7 focal 崩盘真实）

## 1. 背景与目标（同 v1）
d7 R5 accuracy 触顶 ~0.87，LER 缓降。两个微调优化：
- **A（对称增强）**：用**C2 对称（180° 旋转，2×）**增强真机数据，攻 accuracy + 40k 封顶
- **C（两阶段模态）**：stage1 软读出对齐 LER -> stage2 硬读出保 accuracy，攻 LER 模态失配

起点同 R5：bert_pretrain_d7（100k 步）、大模型 ~12M、bs256、**BCE**（d7 focal 已云端核实崩盘 acc 0.669，不用）。

## 2. 实验 A：QEC 对称性数据增强（v2 修订）

### 2.1 原理（修订：D4 8× -> C2 2×）
审查组实证验证（解析 d7_at_q6_7 stim 电路，10000 shot）：
- **布局对称群 = C4**（4 旋转：identity/rot90/rot180/rot270 布局不变；4 反射全破布局）
- **label 关系**（Z-memory）：
  - identity：label 保持（corr=+1）✓
  - rot90：label **独立随机**（same=49.8%，corr≈0）✗ **禁用**
  - rot180：label 完全保持（same=100%，corr=+1）✓
  - rot270：label **独立随机**（same=48.9%，corr≈0）✗ **禁用**
- **90°/270° 把 logical Z 映射到 logical X**（Z-memory 无 X label -> 随机）
- **唯一安全增强 = identity + rot180 = C2（2×）**

### 2.2 实现（symmetry_augment.py）
- 仅实现 180° 旋转的索引置换：perm_stab(48)、perm_data(49)
- 变换字段：measurement/event 按 perm_stab 置换 stab 维；final_soft 按 perm_data；detection_events 按 (轮次不变, perm_stab)；**label 不变**（rot180 保持）
- 输出 augmented real_d7（40k original + 40k rot180 = 80k）

### 2.3 验证门（修订：加实证 label-corr 量化判据）
**必过才全量训练**：
1. **布局对称性**：验证 rot180 下 stab/data 位置完全不变（perm 是合法置换）
2. **syndrome 一致性**：取已知错误 E，验证 rot180(syndrome(E)) == syndrome(rot180(E))
3. **label-corr 实证检验**（M1 修订）：采样 N≥2000 无噪声 shot，计算 corr(label_original, label_rot180)。**必须 corr=+1.0**（允许数值误差 <0.001）。若 corr≠+1，abort
4. **detection_events 逐轮一致**（S3）：验证 detection_events.reshape(T, n_stab) 按 perm_stab 置换后与 event 置换逐元素相等
- 四项全过才进 2.4

### 2.4 训练（修订：预算重算 + synth 锁量 + 步数调）
- 数据：80k augmented real + **20k synth（绝对量锁定同 R5，不随 real 增强涨）** = 100k
  - 注：v1 的 mix 0.5×320k=160k synth 破坏单变量（C5），v2 锁 synth=20k，仅增强 real，单变量为"real 增强 2×"
- 超参：bs256，**12000 步**（修订：100k/256=390 步/epoch，12k 步=30.7 epoch，对齐 R5 的 34 epoch；v1 的 8k 步=20.5 epoch 偏低），BCE，lr 1e-4 cosine，min_steps 3000，patience 10
- 存 `bert_finetune_d7_symaug/`，结果 `results_summary_d7_symaug.json`

### 2.5 QC（修订：胜利线"且"逻辑）
| 项 | 标准 |
|---|---|
| 验证门 §2.3 | 四项全过（含 label-corr=+1）|
| 增强数据 | 80k 样本，label 分布与原 40k 一致（corr=+1）|
| accuracy | vs 0.8664 |
| LER | vs 0.013661，valid（5/5）|
| **胜利线（S1 修订）** | accuracy 升 ≥1pp **且** LER 不恶化（变化 <5%）；**或** LER 降 ≥10% **且** accuracy 不掉超 1pp |
| null result 路径 | 若 2× 增强无显著增益（accuracy 持平、LER 持平），如实报告"2× 对称增强未破墙"（R3->R5 已示数据饱和，2× modest 增量可能 null），非失败而是有效结论 |

## 3. 实验 C：两阶段模态微调（v2 修订）

### 3.1 原理（同 v1）
攻模态失配：stage1 软读出对齐 LER eval，stage2 硬读出保 accuracy。

### 3.2 实现（修订：承认组合优化 + stage2 提步）
- **Stage 1（软读出对齐）**：bert_pretrain_d7 -> finetune on 合成软读出子集（80k 随机采样自 125M，seed 42），bs256，8000 步，BCE，lr 1e-4，存 `bert_finetune_d7_stage1/`
- **Stage 2（硬读出保 accuracy）**：加载 stage1 ckpt -> finetune on real硬读出（40k + 20k synth mix 0.5），bs256，**5000 步**（M3 修订：v1 的 3k×5e-5 有效训练量仅 R5 的 1.9% 过低；v2 提到 5k 步），lr 7e-5（折中：低于 R5 的 1e-4 防 forgetting，高于 v1 的 5e-5 保训练量），存 `bert_finetune_d7_twostage/`

### 3.3 透明声明（M2 修订：承认组合优化）
- **C 是组合优化，非单变量消融**：同时改 schedule（单->两阶段）+ stage2 lr/步数 + 新增 stage1 数据。若 C 胜出，**不能归因于单一因素**，仅证明"两阶段+调参"组合有效
- 若需隔离 stage1 效应，后续可补 C1'（仅加 stage1，stage2 同 R5 8k/1e-4）；本次不拆，先验证组合方向

### 3.4 风险与对策
- **Stage2 erase stage1 软读出对齐**：lr 7e-5（折中）+ 5k 步（不多于 R5）+ 监控 stage2 后 LER 是否保持改善
- **Stage1 过拟合合成**：限 80k 子集 + 8k 步
- **C 内在张力**（残留风险）：LER eval 时模型处 stage2（real-aligned）状态，stage1 synth 对齐可能被部分冲刷--若 LER 未降，说明 stage2 冲刷了 stage1，需调 stage2 lr/步

### 3.5 QC
| 项 | 标准 |
|---|---|
| Stage1 | 8k 步完成，合成软读出 val_acc ~0.85+ |
| Stage2 | 5k 步完成，real test accuracy ≥ 0.85（不掉超基线 1.6pp）|
| **LER** | vs 0.013661，期望降 ≥10% |
| 胜利线 | LER 降 ≥10% **且** accuracy 不掉超 2pp |

## 4. 资源（修订）
| 资源 | 状态 |
|---|---|
| bert_pretrain_d7/best.pt (100k) | ✅ |
| real_d7 (40k/5k/5k) | ✅ |
| synth d7 npy_compressed (125M) | ✅ |
| GPU | V100-32GB 空闲 |
| 时间 | A: 增强+验证门 ~20min + 微调 12k步~100min + eval 15min = ~2.2h；C: stage1 80min + stage2 50min + eval 15min = ~2.4h |
| 总 | ~4.6h（串行）|

## 5. 执行顺序（铁律：串行）
1. **二审本计划书 v2**
2. 实验 A：写 symmetry_augment.py -> 验证门 §2.3（必过）-> 全量增强 -> 微调 12k 步 -> eval_ler
3. 实验 C：stage1 微调 -> stage2 微调 -> eval_ler
4. 汇总 A/C vs R5 对比

## 6. 透明声明（修订）
1. **对称群经实证为 C2（2×）**，非 D4（8×）--审查组实证验证，本计划据实修订。2× 是 modest 增量，可能 null result（已预设报告路径）
2. **label-corr 实证门**（§2.3.3）是安全网：即使 180° 理论保持 label，仍实证验证 corr=+1 才训练，防任何 label 处理 bug
3. **synth 绝对量锁 20k**（同 R5），保证 A 单变量=real 增强 2×
4. **C 承认组合优化**（§3.3），不单变量归因
5. **180° 旋转增强样本与原样本同 patch 同噪声实现**（仅空间翻转），非独立新信息--可能过拟合 patch 特定噪声；2× modest，风险可控
6. accuracy ~0.87 墙可能微调破不了；A 是微调层最可能破墙的（更多真实数据），C 主攻 LER；两者非保证

## 7. 风险
- A 验证门失败（label-corr≠1）-> abort A，仅跑 C
- A 2× 增强无增益 -> null result，如实报告
- C stage2 冲刷 stage1 -> 调 lr/步或报告"两阶段在该配置下无效"
- GPU OOM 低风险（d7 bs256 已验证）
