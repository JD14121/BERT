# Noise-Balanced MSM (NB-MSM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 测试"掩码中被掩位的有噪声/无噪声位 1:1 平衡"能否让 BERT-1 5.5M 模型在 d5/d7 突破历史基线（d5 0.8721 / d7 0.7782）。

**Architecture:** 新增 `NoiseBalancedMSM`（继承 `MaskedSyndromeModeling`），按 `event>0.5` 用 **batch 级池采样**（v2 修正，per-shot 在真实密度下不达 1:1）把被掩位分 defect/非defect 两类各采 50%，其余训练/微调/评估协议与原 `MixedStructuredMSM` 完全一致。配对对照：同数据、同 5.5M 模型、同种子、同步数，唯一变量=掩码策略。跑 **d3/d5/d7 各两臂** = 6 次预训练 + 6 次微调 + LER 评估 + 配对 bootstrap。d3 作 sanity 锚点（event 已近 1:1，不期望主胜）。

**Tech Stack:** PyTorch 2.7+cu128（quantum_env `/d/condapy/quantum_env/python`）、stim 1.16、numpy 1.26、pymatching、scipy、matplotlib。Windows 命令前缀 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`。

## Global Constraints

- **训练环境**：conda `quantum_env`，解释器 `/d/condapy/quantum_env/python`（GPU RTX 4070 SUPER 12GB）。数据生成不涉及（用现有 E 盘数据）。
- **编码**：所有训练命令前缀 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`，否则中文日志 GBK 乱码。
- **工程铁律**：审查组（独立 subagent）与代码组分离；逐阶段审查门未批不执行；禁数据造假/不合理假设/走捷径；任何简化透明披露。
- **5.5M 模型配置**：embed=192 / n_heads=6 / num_transformer_layers=3 / num_readout_layers=4（`local_train_5m.py` 既有 EMBED/HEADS/TLAYERS/RLAYERS，不改）。
- **协议对齐历史 0.9330**：预训练 20k 步 / bs=256 / lr=2e-4 / mask_ratio=0.25；微调 8k 步 / bs=256 / lr=1e-4 / mix_ratio=0.2（真机 + 20% 合成掺杂）；种子 42。
- **数据路径**：E 盘 `E:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/data/`（`local_train_5m.py::GOOGLE_SYNTH_DIR` 已硬编码指向）。
- **代码落点**：本实验代码写在 BERT 交付包 `D:/Code/LZai/Ai for QEC/BERT/`，复用其 `alphaqubit/` 核心库与 `scripts/local_train_5m.py`。`NoiseBalancedMSM` 落在 `D:/Code/LZai/Ai for QEC/BERT/scripts/noise_balanced_msm.py`（与 `mixed_msm.py` 同级）。
- **LER 数据轮次**：{1,10,13,30,50}×20k（E 盘已备）。
- **MWPM 锚点**：d5 0.9428 / d7 0.9702（不重跑，作对照参照线）。

---

## File Structure

| 文件 | 责任 | 创建/修改 |
|---|---|---|
| `D:/Code/LZai/Ai for QEC/BERT/scripts/noise_balanced_msm.py` | `NoiseBalancedMSM` 类：1:1 噪声平衡掩码选择 | 新建 |
| `D:/Code/LZai/Ai for QEC/BERT/scripts/tests/test_noise_balanced_msm.py` | NB-MSM 单测：1:1 比例、补足、形状、event>0.5 分界 | 新建 |
| `D:/Code/LZai/Ai for QEC/BERT/scripts/run_nb_msm.py` | 实验驱动：基于 `local_train_5m.py`，加 `--mask-strategy {original,noise_balanced}`、`--seed`、`--tag`，预训练/微调/评估三阶段 | 新建 |
| `D:/Code/LZai/Ai for QEC/BERT/scripts/paired_bootstrap_nb.py` | 配对 bootstrap 裁决：两臂 per-sample 预测 npz → H1 裁决 | 新建 |
| `D:/Code/LZai/Ai for QEC/BERT/scripts/analyze_mask_distribution.py` | 归因统计：两臂实际被掩位 event 分布（Arm-0≈4% / Arm-1≈50%）+ 补足比例 | 新建 |
| `D:/Code/LZai/Ai for QEC/BERT/scripts/local_train_5m.py` | 仅被 `run_nb_msm.py` 复用其 `load_paems/load_real/EMBED/HEADS/...`，不修改 | 不改 |

**单元边界**：`NoiseBalancedMSM` 只负责"选哪些位被掩"（输出 `mask_indices [B,T,n_stab] bool`），继承父类 `_apply_mask/get_targets`（80/10/10 替换 + measurement 目标），不碰模型/loss/数据。`run_nb_msm.py` 只负责编排，复用既有训练函数。`paired_bootstrap_nb.py` / `analyze_mask_distribution.py` 是只读后处理，不互相依赖。

---

## Task 1: NoiseBalancedMSM 实现 + 单测（TDD）

**Files:**
- Create: `D:/Code/LZai/Ai for QEC/BERT/scripts/noise_balanced_msm.py`
- Test: `D:/Code/LZai/Ai for QEC/BERT/scripts/tests/test_noise_balanced_msm.py`

**Interfaces:**
- Consumes: `alphaqubit.models.pretrain.MaskedSyndromeModeling`（父类，提供 `_apply_mask`/`get_targets`/`mask_sequence`）。
- Produces: `class NoiseBalancedMSM(MaskedSyndromeModeling)`，构造签名 `NoiseBalancedMSM(mask_ratio=0.25, mask_token_value=0.5, random_replace_prob=0.1, keep_original_prob=0.1)`。`mask_sequence(measurement, event, leakage=None, event_leakage=None)` 返回 `(masked_inputs, mask_indices)`，与父类签名/返回完全一致（父类 `mask_sequence` 调 `_generate_mask_indices(B,T,n_stab,device)`，本子类**重写 `_generate_mask_indices`**，但需要 `event`——见 Step 3 说明：通过实例属性 `self._current_event` 传递）。

- [ ] **Step 1: 写失败的单测**

Create `D:/Code/LZai/Ai for QEC/BERT/scripts/tests/test_noise_balanced_msm.py`:

```python
"""NoiseBalancedMSM 单测：1:1 噪声位/平凡位平衡。"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # BERT root（alphaqubit 可导入）

import torch
import numpy as np
from noise_balanced_msm import NoiseBalancedMSM


def _make_batch(B=4, T=10, n_stab=24, noise_frac=0.06, seed=0):
    """造 measurement + event：event 密度 ~noise_frac。measurement 软值与 event 弱相关。"""
    rng = np.random.default_rng(seed)
    event = (rng.random((B, T, n_stab)) < noise_frac).astype(np.float32)
    # measurement 软值：event=1 位偏中间(0.3-0.7)，event=0 位偏极端(<0.1 或 >0.9)
    meas = np.where(event > 0.5,
                    rng.uniform(0.3, 0.7, (B, T, n_stab)),
                    np.where(rng.random((B, T, n_stab)) < 0.5,
                             rng.uniform(0.0, 0.1, (B, T, n_stab)),
                             rng.uniform(0.9, 1.0, (B, T, n_stab)))).astype(np.float32)
    return torch.from_numpy(meas), torch.from_numpy(event)


def test_mask_shape_and_dtype():
    m, e = _make_batch()
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    masked_inputs, mask = msm.mask_sequence(m, e)
    assert mask.shape == m.shape
    assert mask.dtype == torch.bool
    assert 'measurement' in masked_inputs and 'event' in masked_inputs


def test_mask_ratio_approx():
    """被掩位总数 ≈ mask_ratio * T * n_stab（per sample）。"""
    m, e = _make_batch(B=8, T=10, n_stab=24, noise_frac=0.5)  # 50% 噪声，池子充足
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    frac = mask.float().mean().item()
    assert 0.20 < frac < 0.30, f"mask 覆盖率 {frac} 偏离 0.25"


def test_one_to_one_balance_when_pool_sufficient():
    """噪声位池充足时，被掩位中噪声位占比 ≈ 0.5。"""
    m, e = _make_batch(B=16, T=10, n_stab=24, noise_frac=0.5)  # 50% 噪声
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    noise_mask = (e > 0.5)
    masked_noise = (mask & noise_mask).float().sum().item()
    masked_total = mask.float().sum().item()
    ratio = masked_noise / max(masked_total, 1)
    assert 0.40 < ratio < 0.60, f"噪声位占比 {ratio} 偏离 0.5（池子充足时应≈0.5）"


def test_fallback_when_noise_pool_insufficient():
    """噪声位不足时用平凡位补足，不报错，总覆盖率仍≈mask_ratio。"""
    m, e = _make_batch(B=4, T=10, n_stab=24, noise_frac=0.01)  # 几乎无噪声位
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    frac = mask.float().mean().item()
    assert 0.20 < frac < 0.30, f"补足后覆盖率 {frac} 偏离 0.25"
    # 噪声位占比应 < 0.5（不足才补足），记录但不 fail
    noise_mask = (e > 0.5)
    masked_noise = (mask & noise_mask).float().sum().item()
    masked_total = mask.float().sum().item()
    ratio = masked_noise / max(masked_total, 1)
    assert ratio <= 0.5 + 1e-6, f"噪声位不足时占比 {ratio} 不应超过 0.5"


def test_event_threshold_boundary():
    """event=0.5 恰好分界：>0.5 为噪声位。"""
    m = torch.zeros(1, 2, 4)
    e = torch.tensor([[[0.5, 0.51, 0.49, 0.6]],
                      [[0.0, 1.0, 0.5, 0.7]]], dtype=torch.float32)
    msm = NoiseBalancedMSM(mask_ratio=0.5)
    # 主要验证不崩 + 分界正确性通过 mask 行为间接体现
    _, mask = msm.mask_sequence(m, e)
    assert mask.shape == m.shape


def test_realistic_d5_regime_batch_pool():
    """v2 关键测试：d5 真实密度 36%（非少数类），batch 级池采样下被掩 defect 位占比应≈0.5。
    per-shot 采样在此密度下只会得 25%（审查组 B1 抓的 bug），batch 级池应达 50%。"""
    m, e = _make_batch(B=256, T=10, n_stab=24, noise_frac=0.36, seed=1)  # d5 真实密度
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    noise_mask = (e > 0.5)
    masked_noise = (mask & noise_mask).float().sum().item()
    masked_total = mask.float().sum().item()
    ratio = masked_noise / max(masked_total, 1)
    assert 0.42 < ratio < 0.58, f"d5 真实密度 batch 池采样 defect 占比 {ratio} 应≈0.5（per-shot 会得 0.25）"
    frac = mask.float().mean().item()
    assert 0.20 < frac < 0.30, f"覆盖率 {frac} 偏离 0.25"


def test_realistic_d7_regime_batch_pool():
    """v2 关键测试：d7 真实密度 30%，batch 级池采样下被掩 defect 位占比应≈0.5。
    per-shot 采样在此密度下只会得 12%（审查组 B1 抓的 bug），batch 级池应达 50%。"""
    m, e = _make_batch(B=256, T=10, n_stab=48, noise_frac=0.30, seed=2)  # d7 真实密度
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    noise_mask = (e > 0.5)
    masked_noise = (mask & noise_mask).float().sum().item()
    masked_total = mask.float().sum().item()
    ratio = masked_noise / max(masked_total, 1)
    assert 0.42 < ratio < 0.58, f"d7 真实密度 batch 池采样 defect 占比 {ratio} 应≈0.5（per-shot 会得 0.12）"


def test_get_targets_inherited():
    """get_targets 沿用父类：返回 measurement[mask]。"""
    m, e = _make_batch(B=2, T=5, n_stab=8, noise_frac=0.5)
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    _, mask = msm.mask_sequence(m, e)
    targets, flat = msm.get_targets(m, mask)
    assert targets.shape[0] == int(mask.sum())
    assert torch.allclose(targets, m[mask])
```

- [ ] **Step 2: 运行单测确认失败**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -m pytest scripts/tests/test_noise_balanced_msm.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'noise_balanced_msm'`（或导入错误）。

- [ ] **Step 3: 实现 NoiseBalancedMSM**

Create `D:/Code/LZai/Ai for QEC/BERT/scripts/noise_balanced_msm.py`:

```python
"""NoiseBalancedMSM: defect/非defect 位 1:1 平衡掩码（NB-MSM，v2 batch 级池采样）。

继承 MaskedSyndromeModeling（保留 _apply_mask 80/10/10 + get_targets target=measurement），
重写 _generate_mask_indices：按 event>0.5 把全 batch 候选位分 defect 池/非defect 池，
各无放回采 B*target/2 个，再按 shot 分配回 [B,T,n_stab]，使被掩位中两类各占 50%。

v2 修正（审查组 B1）：v1 per-shot 采样在 d5/d7 真实密度下 defect 位不足 half，
1:1 无法实现（d5 落 25%、d7 落 12%）。batch 级池采样在三码距都 2.4-3.8× 充足（实测）。
defect=1 是物理错误直接证据，掩 defect 位练错误链推断，服务下游逻辑错误分类。

event 传递机制：父类 mask_sequence(measurement, event, ...) 调
_generate_mask_indices(B,T,n_stab,device) 不传 event，故在 mask_sequence 入口
把 event 存到 self._current_event，_generate_mask_indices 读它。
"""
import numpy as np
import torch
from alphaqubit.models.pretrain import MaskedSyndromeModeling


class NoiseBalancedMSM(MaskedSyndromeModeling):
    def __init__(self, mask_ratio: float = 0.25, mask_token_value: float = 0.5,
                 random_replace_prob: float = 0.1, keep_original_prob: float = 0.1):
        super().__init__(mask_ratio=mask_ratio,
                         mask_token_value=mask_token_value,
                         random_replace_prob=random_replace_prob,
                         keep_original_prob=keep_original_prob)
        self._current_event = None      # [B, T, n_stab] tensor，由 mask_sequence 注入
        self._last_fill_ratio = 0.0    # 上次 batch defect 位不足时的补足比例（透明披露用）

    def mask_sequence(self, measurement, event, leakage=None, event_leakage=None):
        # 注入 event 供 _generate_mask_indices 使用（device 对齐）
        self._current_event = event
        try:
            return super().mask_sequence(measurement, event, leakage, event_leakage)
        finally:
            self._current_event = None

    def _generate_mask_indices(self, B, T, n_stab, device):
        """batch 级池 1:1 采样：全 batch defect 位拼池采 B*target/2，非 defect 拼池采 B*target/2。"""
        mask = np.zeros((B, T, n_stab), dtype=bool)
        target_per_sample = max(1, int(self.mask_ratio * T * n_stab))
        total_target = B * target_per_sample
        half = total_target // 2

        # 取 event 布尔（device 对齐 → cpu numpy）
        if self._current_event is not None:
            ev = self._current_event
            if ev.device != device:
                ev = ev.to(device)
            event_np = (ev.cpu().numpy() > 0.5)        # [B, T, n_stab] bool
        else:
            event_np = np.zeros((B, T, n_stab), dtype=bool)  # 无 event 退化为全非defect（保兼容，不应在正常径）

        # 构造全 batch 候选位池：扁平索引 (b*T*n_stab + t*n_stab + s)
        flat_noise = np.argwhere(event_np.reshape(-1)).reshape(-1)        # defect 位的全局扁平索引
        flat_clean = np.argwhere(~event_np.reshape(-1)).reshape(-1)       # 非 defect 位
        n_noise_pick = min(half, len(flat_noise))
        n_clean_pick = total_target - n_noise_pick                          # defect 不足则非defect补足
        if n_clean_pick > len(flat_clean):
            n_clean_pick = len(flat_clean)                                  # 极端兜底

        total_filled = half - n_noise_pick  # 补足数（应为 0，实测 batch 级池充足）
        self._last_fill_ratio = (total_filled / max(1, n_noise_pick + total_filled))

        # 无放回采样
        rng = np.random.default_rng()  # 用全局流（set_seed 已 np.random.seed 控制；这里 default_rng() 取 OS 熵
        # —— 为可复现性改用 np.random 全局流 choice：
        if n_noise_pick > 0:
            pick_n = np.random.choice(flat_noise, n_noise_pick, replace=False)
        else:
            pick_n = np.array([], dtype=int)
        if n_clean_pick > 0:
            pick_c = np.random.choice(flat_clean, n_clean_pick, replace=False)
        else:
            pick_c = np.array([], dtype=int)

        # 扁平索引 → (b, t, s) 写入 mask
        stride = T * n_stab
        for idx in pick_n:
            b = idx // stride; rem = idx % stride; t = rem // n_stab; s = rem % n_stab
            mask[b, t, s] = True
        for idx in pick_c:
            b = idx // stride; rem = idx % stride; t = rem // n_stab; s = rem % n_stab
            mask[b, t, s] = True
        return torch.from_numpy(mask).to(device)
```

- [ ] **Step 4: 运行单测确认通过**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -m pytest scripts/tests/test_noise_balanced_msm.py -v
```
Expected: 8 passed。

- [ ] **Step 5: 提交（本仓库非 git，跳过 commit，记录到 SELF_MEMORY）**

> 注：工作区 `is a git repository: false`，commit 步骤改为在 SELF_MEMORY.md Evolution Log 记录。后续任务同。

---

## Task 2: 实验驱动脚本 run_nb_msm.py

**Files:**
- Create: `D:/Code/LZai/Ai for QEC/BERT/scripts/run_nb_msm.py`

**Interfaces:**
- Consumes: `local_train_5m.py` 的 `load_paems(d)`、`load_real(d)`、`EMBED/HEADS/TLAYERS/RLAYERS/DROPOUT`、`GOOGLE_SYNTH_DIR`、`EXP`；`run_experiment.py`（主工作区）的 `make_coord/evaluate_model/finetune`；`alphaqubit.training.pretrain_trainer.PretrainTrainer/PretrainConfig`；`alphaqubit.models.pretrain_decoder.PretrainDecoder`；`xzzx_decoder.XZZXFineTuneDecoder`；`mixed_msm.MixedStructuredMSM`；`noise_balanced_msm.NoiseBalancedMSM`。
- Produces: CLI `run_nb_msm.py --distance {5,7} --mask-strategy {original,noise_balanced} --stage {pretrain,finetune,eval} --seed 42 --tag <tag> --steps <n>`。检查点存 `EXP/checkpoints/{pre,ft}_d{d}_{tag}_{seed}/best.pt`，结果存 `EXP/results_summary_d{d}_{tag}_{seed}.json`。`eval` 阶段额外写 per-sample 预测 npz `EXP/preds_d{d}_{tag}_{seed}.npz`（供配对 bootstrap）。

- [ ] **Step 1: 写 run_nb_msm.py**

Create `D:/Code/LZai/Ai for QEC/BERT/scripts/run_nb_msm.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_nb_msm.py: NB-MSM 配对对照实验驱动。

唯一变量 = MSM 掩码策略（original=MixedStructuredMSM 40/30/30 vs noise_balanced=NoiseBalancedMSM 1:1）。
其余对齐历史 0.9330 协议：5.5M 模型、20k 预训练、8k 微调、20% 合成掺杂、种子 42。
检查点带 tag+seed 后缀，防两臂/两种子互相覆盖。

用法：
  /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy original    --stage pretrain  --seed 42 --tag arm0
  /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy original    --stage finetune --seed 42 --tag arm0
  /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy original    --stage eval     --seed 42 --tag arm0
  /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy noise_balanced --stage pretrain  --seed 42 --tag arm1
  ... (finetune / eval)
"""
import sys, os, argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader

PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PROJECT_ROOT / "google_paems_data" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
# BERT 交付包的 scripts/（noise_balanced_msm / mixed_msm / local_train_5m 在此）
sys.path.insert(0, r"D:/Code/LZai/Ai for QEC/BERT/scripts")
os.chdir(str(EXP))

import stim
from path_config import GOOGLE_SC, GOOGLE_PATCH, DATA_DIR
from xzzx_coord import XZZXCoordinateSystem
from alphaqubit.data.pt_dataset import PTBatchDataset
from alphaqubit.models.pretrain_decoder import PretrainDecoder
from xzzx_decoder import XZZXFineTuneDecoder as FineTuneDecoder
from alphaqubit.training.trainer import Trainer, TrainingConfig
from alphaqubit.training.pretrain_trainer import PretrainTrainer, PretrainConfig
from mixed_msm import MixedStructuredMSM
from noise_balanced_msm import NoiseBalancedMSM
from run_experiment import make_coord, evaluate_model, finetune
# 复用 local_train_5m 的数据加载与模型常量
from local_train_5m import load_paems, load_real, EMBED, HEADS, TLAYERS, RLAYERS, DROPOUT, GOOGLE_SYNTH_DIR

MASK_RATIO = 0.25


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_masking(strategy, coord_system):
    if strategy == 'original':
        return MixedStructuredMSM(mask_ratio=MASK_RATIO, coord_system=coord_system,
                                   p_random=0.4, p_spatial=0.3, p_temporal=0.3)
    elif strategy == 'noise_balanced':
        return NoiseBalancedMSM(mask_ratio=MASK_RATIO)
    else:
        raise ValueError(f"unknown mask-strategy {strategy}")


def ckpt_dirs(d, tag, seed):
    base = EXP / "checkpoints"
    pre = base / f"bert_pretrain_d{d}_{tag}_s{seed}"
    ft = base / f"bert_finetune_d{d}_{tag}_s{seed}"
    return pre, ft


def do_pretrain(d, strategy, tag, seed, steps=20000, bs=256, lr=2e-4):
    set_seed(seed)
    cs = make_coord(d)
    train_ds, val_ds = load_paems(d)
    val_sub = Subset(val_ds, range(min(20000, len(val_ds))))
    model = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                            num_transformer_layers=TLAYERS, dropout=DROPOUT)
    masking = build_masking(strategy, cs)
    cfg = PretrainConfig(total_steps=steps, batch_size=bs, eval_interval=500,
                         learning_rate=lr, device='cuda', use_amp=True,
                         mask_ratio=MASK_RATIO, early_stopping_patience=100000,
                         save_interval=1000, seed=seed)
    pre_dir, _ = ckpt_dirs(d, tag, seed)
    trainer = PretrainTrainer(model=model, train_dataset=train_ds, val_dataset=val_sub,
                              config=cfg, save_dir=str(pre_dir))
    trainer.masking = masking
    print(f"[NB-MSM] pretrain d{d} {strategy} seed={seed} tag={tag}: {steps} steps bs={bs}")
    trainer.train()
    # 保存训练历史
    json.dump(trainer.history,
              open(str(EXP / f"pretrain_history_d{d}_{tag}_s{seed}.json"), 'w'),
              indent=2)
    print(f"[NB-MSM] pretrain done -> {pre_dir}/best.pt")


def do_finetune(d, strategy, tag, seed, steps=8000, bs=256, lr=1e-4, mix_ratio=0.2):
    set_seed(seed)
    cs = make_coord(d)
    real_train, real_val, real_test = load_real(d)
    pre_dir, ft_dir = ckpt_dirs(d, tag, seed)
    ckpt = pre_dir / 'best.pt'
    assert ckpt.exists(), f"预训练 ckpt 不存在: {ckpt}，跑 --stage pretrain"
    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                          num_transformer_layers=TLAYERS, dropout=DROPOUT)
    pre.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])
    bert = FineTuneDecoder(coord_system=cs, pretrained_encoder=pre,
                           embed_dim=EMBED, readout_dim=64, n_heads=HEADS,
                           num_transformer_layers=TLAYERS, num_readout_layers=RLAYERS, dropout=DROPOUT)
    if mix_ratio > 0:
        syn_train, _ = load_paems(d)
        n_mix = int(len(real_train) * mix_ratio)
        syn_sub = Subset(syn_train, np.random.default_rng(42).choice(len(syn_train), n_mix, replace=False))
        train_ds = ConcatDataset([real_train, syn_sub])
        print(f"[NB-MSM] finetune data: real {len(real_train)} + synth {n_mix} = {len(train_ds)}")
    else:
        train_ds = real_train
    finetune(bert, train_ds, real_val, 'cuda', steps, lr=lr, bs=bs, save_dir=str(ft_dir))
    # 评估 test acc
    bert = bert.to('cuda')
    results = evaluate_model(bert, real_test, 'cuda')
    print(f"\n=== d{d} {strategy} seed={seed} tag={tag} test acc: {results['accuracy']:.4f} ===")
    out = {'distance': d, 'mask_strategy': strategy, 'tag': tag, 'seed': seed,
           'model': '5.5M', 'results': results,
           'config': {'embed': EMBED, 'heads': HEADS, 'tlayers': TLAYERS, 'rlayers': RLAYERS,
                      'pretrain_steps': 20000, 'finetune_steps': steps, 'lr': lr,
                      'mix_ratio': mix_ratio, 'mask_ratio': MASK_RATIO}}
    json.dump(out, open(str(EXP / f"results_summary_d{d}_{tag}_s{seed}.json"), 'w'), indent=2)


def do_eval(d, strategy, tag, seed):
    """test acc + per-sample 预测 npz（供配对 bootstrap）。"""
    set_seed(seed)
    cs = make_coord(d)
    _, _, real_test = load_real(d)
    _, ft_dir = ckpt_dirs(d, tag, seed)
    ckpt = ft_dir / 'best.pt'
    assert ckpt.exists(), f"微调 ckpt 不存在: {ckpt}"
    pre = PretrainDecoder(coord_system=cs, embed_dim=EMBED, n_heads=HEADS,
                          num_transformer_layers=TLAYERS, dropout=DROPOUT)
    bert = FineTuneDecoder(coord_system=cs, pretrained_encoder=pre,
                           embed_dim=EMBED, readout_dim=64, n_heads=HEADS,
                           num_transformer_layers=TLAYERS, num_readout_layers=RLAYERS, dropout=DROPOUT)
    bert.load_state_dict(torch.load(str(ckpt), map_location='cpu', weights_only=False)['model_state_dict'])
    bert = bert.to('cuda').eval()
    loader = DataLoader(real_test, batch_size=1024, shuffle=False)
    preds, labels = [], []
    import torch.nn.functional as F
    with torch.no_grad():
        for b in loader:
            m = b['measurement'].to('cuda'); e = b['event'].to('cuda')
            fs = b['final_soft'].to('cuda'); lb = b['label']
            lk = torch.zeros_like(m); el = torch.zeros_like(m)
            logit = bert(m, e, lk, el, fs, n_rounds=m.shape[1])
            pred = (torch.sigmoid(logit) > 0.5).float().cpu()
            preds.append(pred); labels.append(lb)
    preds = torch.cat(preds).numpy().astype(int).flatten()
    labels = torch.cat(labels).numpy().astype(int).flatten()
    acc = float((preds == labels).mean())
    np.savez(str(EXP / f"preds_d{d}_{tag}_s{seed}.npz"), preds=preds, labels=labels)
    print(f"=== d{d} {strategy} seed={seed} tag={tag} test acc: {acc:.4f} (npz saved) ===")
    json.dump({'distance': d, 'mask_strategy': strategy, 'tag': tag, 'seed': seed,
               'accuracy': acc},
              open(str(EXP / f"eval_acc_d{d}_{tag}_s{seed}.json"), 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser(description="NB-MSM 配对对照实验")
    ap.add_argument('--distance', type=int, required=True, choices=[5, 7])
    ap.add_argument('--mask-strategy', required=True, choices=['original', 'noise_balanced'])
    ap.add_argument('--stage', required=True, choices=['pretrain', 'finetune', 'eval'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', required=True, help='臂标识，如 arm0/arm1')
    ap.add_argument('--steps', type=int, default=None, help='覆盖步数（默认 pretrain=20000/finetune=8000）')
    args = ap.parse_args()
    print(f"\n{'='*60}\nNB-MSM | d{args.distance} | {args.mask_strategy} | {args.stage} | seed={args.seed} tag={args.tag}\n{'='*60}\n")
    if args.stage == 'pretrain':
        do_pretrain(args.distance, args.mask_strategy, args.tag, args.seed,
                    steps=args.steps or 20000)
    elif args.stage == 'finetune':
        do_finetune(args.distance, args.mask_strategy, args.tag, args.seed,
                    steps=args.steps or 8000)
    elif args.stage == 'eval':
        do_eval(args.distance, args.mask_strategy, args.tag, args.seed)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 烟测——import 与 argparse 不崩**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --help
```
Expected: 打印 argparse 帮助，无 ImportError。

- [ ] **Step 3: 烟测——eval 路径在无 ckpt 时报错信息正确（证路径拼接）**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy original --stage eval --seed 42 --tag arm0 2>&1 | tail -5
```
Expected: `AssertionError: 微调 ckpt 不存在: .../bert_finetune_d5_arm0_s42/best.pt`（证路径拼接正确，未跑训练）。

- [ ] **Step 4: 记录到 SELF_MEMORY**

---

## Task 3: 掩码分布归因脚本 analyze_mask_distribution.py

**Files:**
- Create: `D:/Code/LZai/Ai for QEC/BERT/scripts/analyze_mask_distribution.py`

**Interfaces:**
- Consumes: `noise_balanced_msm.NoiseBalancedMSM`、`mixed_msm.MixedStructuredMSM`、`local_train_5m.load_paems`、`run_experiment.make_coord`。
- Produces: CLI `analyze_mask_distribution.py --distance {5,7} --num-batches 50`，输出 `EXP/mask_distribution_d{d}.json`，含两臂被掩位 event 分布（噪声位占比）+ NB-MSM 补足比例。

- [ ] **Step 1: 写脚本**

Create `D:/Code/LZai/Ai for QEC/BERT/scripts/analyze_mask_distribution.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归因统计：验证 NB-MSM 干预落实——两臂被掩位的 event 分布。

预期（v2 实测密度）：Arm-0 (original) 被掩位 defect 占比 ≈ 数据自然 event 密度（d3~47%, d5~36%, d7~30%）；
      Arm-1 (noise_balanced) ≈ 50%（batch 级池采样充足时）。记录 NB-MSM 补足比例（应≈0）。
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"
sys.path.insert(0, str(EXP)); sys.path.insert(0, str(PROJECT_ROOT / "google_paems_data" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, r"D:/Code/LZai/Ai for QEC/BERT/scripts")
os = __import__('os'); os.chdir(str(EXP))

from mixed_msm import MixedStructuredMSM
from noise_balanced_msm import NoiseBalancedMSM
from local_train_5m import load_paems
from run_experiment import make_coord


def measure(strategy, d, num_batches, bs=256):
    cs = make_coord(d)
    if strategy == 'original':
        msm = MixedStructuredMSM(mask_ratio=0.25, coord_system=cs, p_random=0.4, p_spatial=0.3, p_temporal=0.3)
    else:
        msm = NoiseBalancedMSM(mask_ratio=0.25)
    train_ds, _ = load_paems(d)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    noise_ratios = []
    fill_ratios = []
    coverage = []
    n = 0
    for b in loader:
        if n >= num_batches: break
        m = b['measurement']; e = b['event']
        _, mask = msm.mask_sequence(m, e)
        noise_mask = (e > 0.5)
        masked_noise = (mask & noise_mask).float().sum().item()
        masked_total = mask.float().sum().item()
        noise_ratios.append(masked_noise / max(masked_total, 1))
        coverage.append(masked_total / m.numel())
        if strategy == 'noise_balanced':
            fill_ratios.append(getattr(msm, '_last_fill_ratio', 0.0))
        n += 1
    return {
        'strategy': strategy, 'distance': d, 'num_batches': n,
        'masked_noise_ratio_mean': float(np.mean(noise_ratios)),
        'masked_noise_ratio_std': float(np.std(noise_ratios)),
        'coverage_mean': float(np.mean(coverage)),
        'fill_ratio_mean': float(np.mean(fill_ratios)) if fill_ratios else None,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, required=True, choices=[5, 7])
    ap.add_argument('--num-batches', type=int, default=50)
    args = ap.parse_args()
    out = {}
    for strat in ['original', 'noise_balanced']:
        r = measure(strat, args.distance, args.num_batches)
        print(f"[{strat}] d{args.distance}: masked_noise_ratio={r['masked_noise_ratio_mean']:.4f} "
              f"coverage={r['coverage_mean']:.4f} fill={r['fill_ratio_mean']}")
        out[strat] = r
    out['data_event_density'] = float((load_paems(args.distance)[0][:1000]['event'] > 0.5).float().mean())
    json.dump(out, open(str(EXP / f"mask_distribution_d{args.distance}.json"), 'w'), indent=2)
    print(f"saved -> {EXP}/mask_distribution_d{args.distance}.json")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行 d5 归因（CPU 即可，~1min）**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python analyze_mask_distribution.py --distance 5 --num-batches 50 2>&1 | tail -10
```
Expected: `original` masked_noise_ratio ≈ 自然密度（d3 0.47 / d5 0.36 / d7 0.30），`noise_balanced` ≈ 0.50，fill_ratio ≈ 0.0（batch 级池充足）。保存 JSON。

- [ ] **Step 3: 记录到 SELF_MEMORY**

---

## Task 4: 配对 bootstrap 裁决脚本 paired_bootstrap_nb.py

**Files:**
- Create: `D:/Code/LZai/Ai for QEC/BERT/scripts/paired_bootstrap_nb.py`

**Interfaces:**
- Consumes: `EXP/preds_d{d}_{arm0/arm1}_s{seed}.npz`（Task 2 eval 产出）。
- Produces: `EXP/bootstrap_H1_d{d}_s{seed}.json`，含两臂 acc、差值、CI、p 值、H1 裁决。

- [ ] **Step 1: 写脚本**

Create `D:/Code/LZai/Ai for QEC/BERT/scripts/paired_bootstrap_nb.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对 bootstrap 裁决：Arm-1 (noise_balanced) vs Arm-0 (original) 的 test acc 差。

per-sample 配对（同 test 集、同种子下两臂预测对同一批样本），10k 次重采样。
裁决规则（预注册，写死）：
  H1_supported:  diff > 0 且 CI 下界 > 0 且 p < 0.05
  H1_refuted:    diff < 0 且 CI 上界 < 0 且 p < 0.05
  inconclusive:   其他
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"


def load_preds(d, tag, seed):
    f = EXP / f"preds_d{d}_{tag}_s{seed}.npz"
    assert f.exists(), f"缺少 {f}（先跑 run_nb_msm.py --stage eval）"
    z = np.load(str(f))
    return z['preds'], z['labels']


def paired_bootstrap(preds1, preds0, labels, B=10000, seed=0):
    """配对：per-sample correct 指示，重采样取差值均值。返回 diff, ci_lo, ci_hi, p。"""
    correct1 = (preds1 == labels).astype(np.float64)
    correct0 = (preds0 == labels).astype(np.float64)
    paired_diff = correct1 - correct0
    n = len(paired_diff)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = paired_diff[idx].mean()
    diff = paired_diff.mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # 单侧 p：H1: diff>0，p = P(diff_boot <= 0)
    p = float((diffs <= 0).mean())
    return float(diff), float(ci_lo), float(ci_hi), p


def verdict(diff, ci_lo, ci_hi, p, alpha=0.05):
    if diff > 0 and ci_lo > 0 and p < alpha:
        return 'H1_supported'
    if diff < 0 and ci_hi < 0 and p < alpha:
        return 'H1_refuted'
    return 'inconclusive'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, required=True, choices=[5, 7])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag1', default='arm1', help='干预臂')
    ap.add_argument('--tag0', default='arm0', help='对照臂')
    ap.add_argument('--B', type=int, default=10000)
    args = ap.parse_args()
    p1, lb = load_preds(args.distance, args.tag1, args.seed)
    p0, _ = load_preds(args.distance, args.tag0, args.seed)
    assert len(p1) == len(p0) == len(lb), "两臂预测数不一致（test 集应同）"
    acc1 = float((p1 == lb).mean()); acc0 = float((p0 == lb).mean())
    diff, ci_lo, ci_hi, p = paired_bootstrap(p1, p0, lb, B=args.B, seed=args.seed)
    v = verdict(diff, ci_lo, ci_hi, p)
    out = {'distance': args.distance, 'seed': args.seed,
           'acc_arm1_noise_balanced': acc1, 'acc_arm0_original': acc0,
           'diff': diff, 'ci_95': [ci_lo, ci_hi], 'p_one_sided': p,
           'B': args.B, 'verdict': v}
    json.dump(out, open(str(EXP / f"bootstrap_H1_d{args.distance}_s{args.seed}.json"), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 烟测——缺 npz 时报错正确**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python paired_bootstrap_nb.py --distance 5 --seed 42 2>&1 | tail -3
```
Expected: `AssertionError: 缺少 .../preds_d5_arm1_s42.npz`（证依赖检查正确）。

- [ ] **Step 3: 记录到 SELF_MEMORY**

---

## Task 5: P1 代码审查门（审查组独立 subagent）

**Files:**
- Review: `noise_balanced_msm.py`、`run_nb_msm.py`、`analyze_mask_distribution.py`、`paired_bootstrap_nb.py`、`tests/test_noise_balanced_msm.py`

**Interfaces:**
- Consumes: Task 1-4 全部产物 + 设计文档 `D:/Code/LZai/Ai for QEC/BERT/docs/superpowers/specs/2026-07-25-noise-balanced-msm-design.md`。
- Produces: 审查组代码审查报告（APPROVE / APPROVE_WITH_CONDITIONS / REJECT + 必改项）。

- [ ] **Step 1: 派独立审查 subagent 审 P1 代码**

用 Agent 工具派一个**独立 subagent**（审查组，与写代码的 main 分离），prompt 要点：
> 审查 NB-MSM 工程的 P1 代码：`D:/Code/LZai/Ai for QEC/BERT/scripts/{noise_balanced_msm.py, run_nb_msm.py, analyze_mask_distribution.py, paired_bootstrap_nb.py, tests/test_noise_balanced_msm.py}`。
> 对照设计文档 `.../2026-07-25-noise-balanced-msm-design.md`。审查：①逻辑错误（1:1 采样是否真 1:1、补足逻辑、event>0.5 分界）；②语义错误（是否破坏父类 80/10/10 + measurement 目标）；③潜在缺陷（event 传递机制 self._current_event 在多线程 DataLoader 下是否安全、device 对齐、np.random 全局流 vs 种子可控性）；④造假/走捷径（是否真跑了还是假数据、归因脚本是否真测分布）；⑤公平性（两臂是否真同种子同协议、ckpt 是否防覆盖、微调掺杂子集是否两臂同一 rng(42)）；⑥配对 bootstrap 裁决规则是否预注册写死。输出 APPROVE / APPROVE_WITH_CONDITIONS / REJECT + 必改项清单。

- [ ] **Step 2: 落实审查组必改项（若有）**

按审查报告逐条修复，复审通过后才进 P2。

- [ ] **Step 3: 记录审查结论到 SELF_MEMORY**

---

## Task 6: P2 训练——d3 两臂（sanity 锚点）

**Files:**
- Run: `run_nb_msm.py`（d3, arm0=original, arm1=noise_balanced, seed=42）

**Interfaces:**
- Consumes: Task 2 驱动脚本 + E 盘 d3 数据。
- Produces: `checkpoints/bert_{pre,ft}_d3_{arm0,arm1}_s42/best.pt`、`preds_d3_{arm0,arm1}_s42.npz`、`results_summary_d3_{arm0,arm1}_s42.json`。

> **d3 作 sanity 锚点**（设计 §1.4）：d3 event=1 天然 47% 近 1:1，NB-MSM uplift 极小，Arm-1 应 ≈ Arm-0（不退步即过，不期望主胜）。d3 节点少（8 stab），训练最快，先跑验证 pipeline + 干预不破坏已近平衡码距。

- [ ] **Step 1: Arm-0 (original) 预训练 d3**

Run（后台，预训练 ~1.5h）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 3 --mask-strategy original --stage pretrain --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d3_arm0_pretrain.log"
```
Expected: 20k 步完成，`bert_pretrain_d3_arm0_s42/best.pt` 存在。

- [ ] **Step 2: Arm-0 微调 d3**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 3 --mask-strategy original --stage finetune --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d3_arm0_finetune.log"
```
Expected: test acc 落在历史 5.5M d3 基线 0.9190 附近（±1pp）。

- [ ] **Step 3: Arm-0 eval d3**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 3 --mask-strategy original --stage eval --seed 42 --tag arm0
```
Expected: `preds_d3_arm0_s42.npz` 保存。

- [ ] **Step 4: Arm-1 (noise_balanced) 预训练 d3**

Run（后台）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 3 --mask-strategy noise_balanced --stage pretrain --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d3_arm1_pretrain.log"
```
Expected: 20k 步完成，`bert_pretrain_d3_arm1_s42/best.pt` 存在。

- [ ] **Step 5: Arm-1 微调 d3**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 3 --mask-strategy noise_balanced --stage finetune --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d3_arm1_finetune.log"
```
Expected: test acc（sanity：应 ≈ Arm-0，不退步即过）。

- [ ] **Step 6: Arm-1 eval d3**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 3 --mask-strategy noise_balanced --stage eval --seed 42 --tag arm1
```
Expected: `preds_d3_arm1_s42.npz` 保存。

- [ ] **Step 7: 记录到 SELF_MEMORY**

---

## Task 7: P2 训练——d5 两臂（预训练+微调+eval）

**Files:**
- Run: `run_nb_msm.py`（d5, arm0=original, arm1=noise_balanced, seed=42）

**Interfaces:**
- Consumes: Task 2 驱动脚本 + E 盘 d5 数据。
- Produces: `checkpoints/bert_{pre,ft}_d5_{arm0,arm1}_s42/best.pt`、`preds_d5_{arm0,arm1}_s42.npz`、`results_summary_d5_{arm0,arm1}_s42.json`。

- [ ] **Step 1: Arm-0 (original) 预训练 d5**

Run（后台，预训练 ~1.7h）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 5 --mask-strategy original --stage pretrain --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d5_arm0_pretrain.log"
```
Expected: 20k 步完成，`bert_pretrain_d5_arm0_s42/best.pt` 存在，val mask_acc 记录。

- [ ] **Step 2: Arm-0 微调 d5**

Run（~34min）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 5 --mask-strategy original --stage finetune --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d5_arm0_finetune.log"
```
Expected: test acc 落在历史 5.5M d5 基线 0.8721 附近（±1pp，证对照锚点同源）。

- [ ] **Step 3: Arm-0 eval（出 per-sample npz）**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy original --stage eval --seed 42 --tag arm0
```
Expected: `preds_d5_arm0_s42.npz` 保存，acc 与 finetune 阶段一。

- [ ] **Step 4: Arm-1 (noise_balanced) 预训练 d5**

Run（后台）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 5 --mask-strategy noise_balanced --stage pretrain --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d5_arm1_pretrain.log"
```
Expected: 20k 步完成，`bert_pretrain_d5_arm1_s42/best.pt` 存在。

- [ ] **Step 5: Arm-1 微调 d5**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 5 --mask-strategy noise_balanced --stage finetune --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d5_arm1_finetune.log"
```
Expected: test acc（与 Arm-0 比，看是否突破 0.8721）。

- [ ] **Step 6: Arm-1 eval d5**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 5 --mask-strategy noise_balanced --stage eval --seed 42 --tag arm1
```
Expected: `preds_d5_arm1_s42.npz` 保存。

- [ ] **Step 7: 记录到 SELF_MEMORY**

---

## Task 8: P2 训练——d7 两臂（预训练+微调+eval）

**Files:**
- Run: `run_nb_msm.py`（d7, arm0, arm1, seed=42）

**Interfaces:**
- Consumes: Task 2 驱动脚本 + E 盘 d7 数据。
- Produces: `checkpoints/bert_{pre,ft}_d7_{arm0,arm1}_s42/best.pt`、`preds_d7_{arm0,arm1}_s42.npz`、`results_summary_d7_{arm0,arm1}_s42.json`。

> ⚠️ d7 节点 97，5.5M 容量历史 0.7782。bs=256 训练需监控 OOM；若 OOM 降 bs=128 + 梯度累积（或加 `--batch-size 128`，但本驱动未实现 grad_accum，OOM 则先记录待审查组批再改）。预训练每步较 d5 慢，~3-4h。

- [ ] **Step 1: Arm-0 预训练 d7**

Run（后台）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 7 --mask-strategy original --stage pretrain --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d7_arm0_pretrain.log"
```
Expected: 20k 步完成无 OOM，best.pt 存在。若 OOM，记录现象，暂停待审查组批 bs 降级方案。

- [ ] **Step 2: Arm-0 微调 d7**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 7 --mask-strategy original --stage finetune --seed 42 --tag arm0 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d7_arm0_finetune.log"
```
Expected: test acc 落在历史 5.5M d7 基线 0.7782 附近（±2pp）。

- [ ] **Step 3: Arm-0 eval d7**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 7 --mask-strategy original --stage eval --seed 42 --tag arm0
```
Expected: `preds_d7_arm0_s42.npz` 保存。

- [ ] **Step 4: Arm-1 预训练 d7**

Run（后台）:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 7 --mask-strategy noise_balanced --stage pretrain --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d7_arm1_pretrain.log"
```
Expected: 20k 步完成无 OOM。

- [ ] **Step 5: Arm-1 微调 d7**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u run_nb_msm.py --distance 7 --mask-strategy noise_balanced --stage finetune --seed 42 --tag arm1 2>&1 | tee -a "D:/Code/LZai/Ai for QEC/BERT/logs/nb_d7_arm1_finetune.log"
```
Expected: test acc（看是否突破 0.7782）。

- [ ] **Step 6: Arm-1 eval d7**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python run_nb_msm.py --distance 7 --mask-strategy noise_balanced --stage eval --seed 42 --tag arm1
```
Expected: `preds_d7_arm1_s42.npz` 保存。

- [ ] **Step 7: 记录到 SELF_MEMORY**

---

## Task 9: P3 评估——配对 bootstrap + LER + 归因

**Files:**
- Run: `paired_bootstrap_nb.py`、`analyze_mask_distribution.py`、`eval_ler.py`（主工作区，已支持 `--pretrain-ckpt/--finetune-ckpt/--tag` + 5.5M 尺寸 CLI）

**Interfaces:**
- Consumes: Task 6/7/8 产出的 ckpt + npz。
- Produces: `bootstrap_H1_d{5,7}_s42.json`、`mask_distribution_d{5,7}.json`、`results_ler_d{5,7}_{arm0,arm1}.json`。

- [ ] **Step 1: 归因统计 d5/d7（证干预落实）**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python analyze_mask_distribution.py --distance 3 --num-batches 50
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python analyze_mask_distribution.py --distance 5 --num-batches 50
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python analyze_mask_distribution.py --distance 7 --num-batches 50
```
Expected: Arm-0 defect 占比 ≈ 数据自然密度（d3 47% / d5 36% / d7 30%），Arm-1 ≈ 0.50，fill_ratio ≈ 0。若 Arm-1 严重偏离 0.5 或 fill 高，审查干预是否真落实。

- [ ] **Step 2: 配对 bootstrap d3/d5/d7**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python paired_bootstrap_nb.py --distance 5 --seed 42
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python paired_bootstrap_nb.py --distance 7 --seed 42
```
Expected: 输出两臂 acc、diff、CI、p、verdict（H1_supported / refuted / inconclusive）。

- [ ] **Step 3: LER 评估 d5 两臂**

Run（eval_ler 在主工作区，注入 5.5M 尺寸 + 本实验 ckpt 路径 + tag）:
```bash
cd "D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/bert_experiment" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python eval_ler.py --distances 5 --embed-dim 192 --n-heads 6 --num-transformer-layers 3 --num-readout-layers 4 --pretrain-ckpt "D:/Code/LZai/Ai for QEC/BERT/scripts/../../../../../../google_paems_data/bert_experiment/checkpoints/bert_pretrain_d5_arm0_s42/best.pt" --finetune-ckpt ".../bert_finetune_d5_arm0_s42/best.pt" --tag arm0_s42
```
> ⚠️ ckpt 实际路径：`run_nb_msm.py` 把 ckpt 存在**主工作区** `EXP/checkpoints/`（因 `os.chdir(EXP)` 且 `EXP = PROJECT_ROOT/google_paems_data/bert_experiment`）。所以 ckpt 路径是 `D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main/google_paems_data/bert_experiment/checkpoints/bert_pretrain_d5_arm0_s42/best.pt`。用此绝对路径替换上面 `--pretrain-ckpt/--finetune-ckpt`。对 arm1 同理，tag=arm1_s42。

Expected: `results_ler_d5_arm0_s42.json` / `results_ler_d5_arm1_s42.json`，含 ler/r_squared/per_round。

- [ ] **Step 4: LER 评估 d7 两臂**

同 Step 3，`--distances 7`，注入 d7 的 arm0/arm1 ckpt，tag=arm{0,1}_s42。

- [ ] **Step 5: 记录到 SELF_MEMORY**

---

## Task 10: seed=123 复核（仅当 seed=42 主胜时）

**Files:**
- Run: `run_nb_msm.py`（d5/d7, 两臂, seed=123）+ `paired_bootstrap_nb.py --seed 123`

**Interfaces:**
- Consumes: Task 8 裁决为 H1_supported 的码距。
- Produces: `preds_d{d}_{arm0,arm1}_s123.npz`、`bootstrap_H1_d{d}_s123.json`。

> 触发条件：seed=42 下某码距 verdict=H1_supported。按跨噪声工程 seed-2 教训，单 seed 显著性不可信，必须 seed=123 复核一致才最终裁决。若 seed=42 无主胜，本任务跳过。

- [ ] **Step 1-6: 复跑主胜码距的两臂（pretrain+finetune+eval, seed=123）**

对主胜码距 d，依次跑 arm0/arm1 的 pretrain/finetune/eval，`--seed 123`。命令同 Task 6/7/8 但 seed=123、tag=arm{0,1}（npz 自动带 s123 后缀）。

- [ ] **Step 7: 配对 bootstrap seed=123**

Run:
```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python paired_bootstrap_nb.py --distance <主胜d> --seed 123
```
Expected: 两 seed 方向一致（均 H1_supported）才最终裁决 H1 成立；不一致则降级 inconclusive（对齐跨噪声工程结局）。

- [ ] **Step 8: 记录到 SELF_MEMORY**

---

## Task 11: P4 QC 汇总 + 报告 + 审查组最终 sign-off

**Files:**
- Create: `D:/Code/LZai/Ai for QEC/BERT/reports/nb_msm/NB_MSM_EXPERIMENT_REPORT.md`
- Update: `SELF_MEMORY.md`（工作区根，Evolution Log 追加）

**Interfaces:**
- Consumes: Task 1-10 全部产物。
- Produces: 实验报告 + SELF_MEMORY Evolution Log + 审查组最终 sign-off。

- [ ] **Step 1: 汇总 results_summary + bootstrap + LER + 归因到报告**

写 `NB_MSM_EXPERIMENT_REPORT.md`，含：
- 设计假设 + 干预点（NB-MSM 1:1 event 平衡）
- 配对对照表（d5/d7 × 两臂 × acc/LER）
- 归因证据（被掩位 event 分布：Arm-0≈自然密度 vs Arm-1≈0.50）
- 配对 bootstrap 裁决（diff/CI/p/verdict，seed=42 + 若复核 seed=123）
- 透明披露：补足比例、d7 OOM 情况（若有）、LER 模态失配声明（NN 真机硬读出微调、LER 合成软读出评估，沿用 eval_ler.py 既有声明）
- 结论：H1 成立/证伪/inconclusive + 解读

- [ ] **Step 2: SELF_MEMORY.md Evolution Log 追加一行**

格式对齐历史 Evolution Log：
```
- **[NB-MSM 实验 2026-07-25]** 用户怀疑 96% 平凡位稀释 MSM 信号。新增 NoiseBalancedMSM（event>0.5 分界，被掩位噪声/平凡 1:1）。配对对照 5.5M d5/d7 两臂同种子。结果：<填>。归因：Arm-0 噪声位占比<X>% vs Arm-1<Y>%（干预落实）。bootstrap verdict：<>。seed=123 复核：<>。结论<>。
```

- [ ] **Step 3: 派审查组最终 sign-off**

用 Agent 工具派独立审查 subagent，审查：①产物完整性（4 臂 ckpt+npz+results+bootstrap+ler+归因全在）；②数值合理性（Arm-0≈历史基线、归因分布符合预期）；③反作弊（per-sample npz 真从模型前向产出、未读旧 ckpt）；④公平性（两臂真同协议）；⑤透明披露完整。输出 APPROVE / APPROVE_WITH_CONDITIONS / REJECT。

- [ ] **Step 4: 落实最终 sign-off 条件（若有），更新 SELF_MEMORY**

---

## Self-Review（plan 自审）

**1. Spec 覆盖：**
- §1 假设 → Task 1 实现体现（1:1 event）✅
- §2 干预点（只改掩码）→ Task 1 只重写 `_generate_mask_indices`，不改模型/loss/数据 ✅
- §3 配对对照 → Task 2 `--mask-strategy` + 同种子/步数/协议，Task 6/7/8 跑两臂 ✅
- §4 数据（现有 E 盘）→ Task 2 复用 `load_paems/load_real`，不生成 ✅
- §5 协议（20k/8k/0.2 掺杂/seed42）→ Task 2 默认值写死 ✅
- §6 验收（主胜/次胜/归因/负面透明）→ Task 8 bootstrap+归因，Task 10 报告透明披露 ✅
- §7 铁律（逐阶段审查门）→ Task 5 P1 代码审查、Task 10 P4 sign-off ✅
- §9 交付物 → Task 1-10 全覆盖 ✅

**2. 占位符扫描：** 无 TBD/TODO。Task 10 Step 1-6 "命令同 Task 6/7/8 但 seed=123" 是条件触发复用（非占位），已说明触发条件。Task 9 Step 3 ckpt 路径已明确为绝对路径并加 ⚠️ 说明。✅

**3. 类型一致性：** `NoiseBalancedMSM` 构造签名 Task 1 定义 = Task 2/3 调用一致（`mask_ratio=0.25`）；`mask_sequence` 返回 `(masked_inputs, mask_indices)` 与父类一致；`_last_fill_ratio` 属性 Task 1 定义 = Task 3 读取一致；`run_nb_msm.py` ckpt 命名 `{pre,ft}_d{d}_{tag}_s{seed}` 在 Task 2/6/7/8/9 一致。✅

**4. 已知风险点（透明）**
- `self._current_event` 在 num_workers>0 时有进程边界风险——`local_train_5m.py` 用默认 `num_workers=0`（PretrainConfig 默认），单线程安全。Task 5 审查组会验。
- `np.random` 全局流：`_generate_mask_indices` 用 `np.random.choice`，种子由 `run_nb_msm.py::set_seed` 通过 `np.random.seed` 控制。可复现性依赖此——Task 5 审查组会验。
- d7 bs=256 OOM 风险：Task 7 已预警 + 暂停待审查组批降级方案。

---

## Execution Handoff

Plan complete and saved to `D:/Code/LZai/Ai for QEC/BERT/docs/superpowers/plans/2026-07-25-noise-balanced-msm.md`. Two execution options:

**1. Subagent-Driven (recommended)** - 每个 Task 派独立 subagent 实现，Task 间审查，快速迭代。

**2. Inline Execution** - 在本会话按 executing-plans 批量执行，带检查点。

**但按工程铁律**：Task 5（P1 代码审查门）和 Task 10（P4 sign-off）必须派独立审查 subagent，不能由写代码的 main 兼任。建议 Subagent-Driven 以便天然隔离写代码与审查。

Which approach?