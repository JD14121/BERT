
---

## 从代码确认的 7 个事实

**事实 1：`_train_step` 和 `evaluate` 都通过同一个路径调用掩码**

`pretrain_trainer.py` 两处调用完全对称：

```python
# _train_step 和 evaluate 里代码形式完全一致
masked_inputs, mask_indices = self.masking.mask_sequence(
    measurement, event, leakage, event_leakage
)
```

`self.masking` 是个普通实例属性，`run_nb_msm.py` 里的 `trainer.masking = masking` 这一行猴子补丁是安全且充分的——之后所有调用都走新对象，包括训练循环和验证循环，没有"绕过"的旁路。

**事实 2：`num_workers=0` 是默认值，单进程**

`PretrainConfig.num_workers: int = 0`，DataLoader 在同一个进程里顺序执行。这意味着 `self._current_event` 的存取是完全安全的——设完就读，读完就清，Python GIL 保证中间不可能被抢占。之前对多线程的担忧可以排除。

**事实 3：masking 同时作用于 measurement 和 event，位置相同**

```python
masked_measurement = self._apply_mask(measurement, mask_indices)
masked_event       = self._apply_mask(event, mask_indices)
```

位置 `(t, s)` 一旦被选中，`measurement[t,s]` 和 `event[t,s]` 会被**同一个掩码**同时替换（80% 变0.5、10% 变随机、10% 保原值）。这堵死了"event[t,s]可见+ measurement[t,s] 被掩"这条直接泄漏路径。Q2 由代码确认，不需要运行时验证。

**事实 4：前向路径仍有 67% 的解析恢复概率——这是 Q3 的关键**

虽然 `event[t,s]` 被掩了，但 **`event[t+1,s]` 和 `measurement[t+1,s]` 没有受到影响**（除非 `(t+1,s)` 独立被掩，概率 `mask_ratio=0.25`）。若 convention 为 A（`event[t+1] = XOR(meas[t+1], meas[t])`），则：

```
meas[t] = (event[t+1,s] - meas[t+1,s]) / (1 - 2·meas[t+1,s])
```

分母需远离 0。文档自报99.4% 的软值落在 `[0, 0.1]∪ [0.9, 1]`，`|1 - 2·meas[t+1]| ≥ 0.8`，数值极为稳定。粗估可恢复比例：`(T-1)/T × (1- 0.25) × 0.994 ≈ 0.67`。约 2/3 的被掩位可以从"下一轮的邻居"绕一步推回来——模型不需要真正理解错误链，只需做一次软XOR 反算。这是预训练任务信息量是否充分的核心问题，必须靠探针脚本实测。

**事实 5：leakage 被掩会变成一个免费的"掩码位置指示器"**

`PTBatchDataset.__getitem__` 里：

```python
leakage       = torch.zeros(T, self.n_stab, dtype=torch.float32)
event_leakage = torch.zeros(T, self.n_stab, dtype=torch.float32)
```

这两个张量永远是全零。但 `mask_sequence` 会把它们按同一个 `mask_indices` 过一遍 `_apply_mask`：80% 的被掩位上，原来是 0 的 leakage 变成 0.5（`mask_token_value`）。效果是：**预训练阶段的 leakage 通道会在被掩位上打上 "=0.5" 的标记**，模型可以直接从 leakage 通道读出哪些位置需要被预测，而不必通过 syndrome 的时空上下文推理。预训练目标因此存在信息捷径。微调阶段 leakage 全是 0，这个捷径消失——留下的是一个分布不匹配。两臂共享这个问题，所以不影响 A vs B 的比较，但会压低双臂的预训练质量上限。

**事实 6：`MixedStructuredMSM` 的 per-shot 掩码数并不固定**

40% 概率走 random策略：`mask[b] = np.random.rand(T, n_stab) < self.mask_ratio`，这是 Bernoulli i.i.d.，per-shot 掩码数服从 Binomial(T·n_stab, 0.25)。d5 标准差约 6.7，d7 约 9.5。另外 60% 走 spatial/temporal，用 while 循环逼近 `target_count`，接近固定。所以 Arm-0 本身就有 per-shot 掩码数波动，NB-MSM 改成固定配额反而更均匀，不引入新混淆。

**事实 7：`PTBatchDataset.__getitem__` 只接受 int，切片会报错**

```python
def __getitem__(self, idx: int) -> Dict[str, Tensor]:
    measurement = self._data['measurement'][idx]  # idx必须是整数
```

`dataset[:1000]` 会把slice 对象传进来，PyTorch Dataset 不会自动处理，要么抛 TypeError，要么拿到单个元素的 dict（因为 Python 可能把 slice 当整数转换失败）。但好消息是 `PTBatchDataset` 提供了 `get_batch(batch_size)` 方法，可以直接用：

```python
batch = ds.get_batch(1000)          # 返回 dict[str, Tensor]，形状正确
density = float((batch['event'] > 0.5).float().mean())
```

另外注意 `get_batch` 用了 `np.random.choice`，会消费全局numpy 随机流——归因脚本调用它时要在`set_seed` 之后。

还有一个从代码推出的**新问题**，之前没提到：`leakage` 和 `event_leakage` 在数据集里永远是全零，但 `mask_sequence` 会对它们也过`_apply_mask`。80% 的被掩位上，全零的 leakage 会被替换成 `mask_token_value=0.5`，于是 leakage 通道在预训练阶段变成了**掩码位置的免费标记牌**——模型只需要看 leakage 通道是不是 0.5，就能直接知道哪里需要预测，完全不用通过时空上下文推理。两臂共享这个问题，不影响 A vs B 的对比，但会压低双臂预训练质量的上限，应当记录在透明披露里。

---

## 综合确认后的事实总结对照表

| 原方案假设 | 代码实际情况 | 影响 |
|---|---|---|
| `_current_event` 多线程不安全 | `num_workers=0`，单进程，无风险 | 阶段 0 只需保留 raise，不需要其他并发保护 |
| trainer 有旁路绕过 `mask_sequence` | 两处调用完全对称，猴子补丁充分 | run_nb_msm.py 的写法是安全的 |
| `_apply_mask` 可能只掩 measurement | measurement 和 event 同 mask同时掩 | `event[t,s]` 被掩，直接读它的泄漏路径关闭 |
| 前向路径 `event[t+1,s]` 的泄漏 | `event[t+1,s]` 未被掩，softXOR 反算稳定 | **约 67% 的被掩位可绕路恢复，需Q3 实测** |
| leakage 通道中性| 全零 leakage 经apply_mask 后变 0.5，暴露掩码位置 | 预训练有捷径，双臂共有 |
| MixedStructuredMSM per-shot 数固定 | 40% 路径是Bernoulli i.i.d.，有Binomial 方差 | NB-MSM 固定配额不引入新的不公平 |
| `dataset[:1000]` 能用 | `__getitem__` 只接受 int，会报错 | 归因脚本最后一行必须换成 `get_batch` |

还有一个**新增的关键推论**：`MixedStructuredMSM` 的 temporal span 策略（30% 概率）会把同一个 stabilizer 的连续多轮一起掩，比如掩 `t=3,4,5` 的位置 `s`。此时 `event[t+1=4,s]` 本身也在被掩之列，前向恢复路径被切断，该位置成为**真正难以恢复的目标**。而 NB-MSM 明确关掉了结构先验（设计 §2），所有被掩位都是逐元素独立选的，前向邻居几乎总是可见。**这意味着 NB-MSM 的预训练任务比 Arm-0 更容易，而不是更难**——方向与设计意图相反，这是阶段 1 探针的核心预测之一，若实测证实，整个假设需要重新审视。

---

## 修订后各阶段实现步骤

### 阶段 0：P0 修复（30分钟，零GPU）

共5 处，全部是防崩和防静默错误，优先级最高。

**0.1 argparse加d3**

`run_nb_msm.py` 和 `analyze_mask_distribution.py`，两处都要改：

```python
ap.add_argument('--distance', type=int, required=True, choices=[3, 5, 7])
```

**0.2 fallback 改raise**

`noise_balanced_msm.py::_generate_mask_indices` 开头：

```python
if self._current_event is None:
    raise RuntimeError(
        "[NoiseBalancedMSM] _generate_mask_indices 未拿到 event tensor。\n"
        "正常路径必须通过 mask_sequence() 调用，它会先把 event 存入 self._current_event。\n"
        "直接调用 _generate_mask_indices 或绕过 mask_sequence 的任何路径都会触发此错误。\n"
        "（num_workers=0 已确认，此错误不是并发竞态，而是调用顺序错误。）"
    )
```

理由已确认：`num_workers=0` 单进程下`_current_event` 的设置-读取-清除 是原子的，抛错是逻辑保证，不是并发保护。

**0.3 归因脚本最后一行**

原代码：
```python
out['data_event_density'] = float((load_paems(args.distance)[0][:1000]['event'] > 0.5).float().mean())
```

存在两个问题：`[:1000]` 切片会报错；重复全量加载 2-5 GB 数据。`measure()` 函数里已经调用了 `load_paems`，把 loader 传出来复用即可：

```python
# measure() 改成同时返回 loader 或 dataset
def measure(strategy, d, num_batches, bs=256):
    cs = make_coord(d)
    ...
    train_ds, _ = load_paems(d)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    ...
    # 统计自然event 密度（复用已有loader，不重复加载）
    nat_density = compute_natural_density(loader, max_batches=20)
    return {..., 'natural_event_density': nat_density}

def compute_natural_density(loader, max_batches=20):
    tot = cnt = 0.0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        e = b['event']
        tot += float((e > 0.5).sum())
        cnt += e.numel()
    return tot / max(cnt, 1.0)
```

**0.4 清理死代码**

`noise_balanced_msm.py` 里删除 `rng = np.random.default_rng()` 这一行（建了不用，被注释掉的自陈改用全局流），并删掉相关注释里的矛盾说明——这不是小事，一旦有人读代码，注释和实现打架会引发信任问题。

**0.5 `torch.where` 标量类型明确化（预防 AMP 下的隐式类型转换）**

`_apply_mask` 里：

```python
# 原代码
masked = torch.where(mask_token_mask, self.mask_token_value, masked)

# 改为显式 tensor，避免 AMP 下 float16 与 Python float 混合
fill_val = torch.full_like(masked, self.mask_token_value)
masked = torch.where(mask_token_mask, fill_val, masked)
```

这条在 float32 训练下不会报错，但 `use_amp=True` 时模型前向是 float16，若 `masked` 在 autocast 区域内被强制转换，`mask_token_value` 作为 Python float 会触发广播警告甚至静默精度损失。

---

### 阶段 1：泄漏探针（半天，零 GPU，一个 batch前向）

这是整条链里性价比最高的一步。它能在启动任何训练之前，用一个 batch 的时间，验证或推翻 NB-MSM 假设的物理前提。

探针需要回答 4 个问题：

- **Q1**：`event[t]` 的时间配对方向（convention A或 B）——决定 Q3恢复公式的方向
- **Q3**：被掩位通过前向邻居的可解析恢复比例，**按 defect / non-defect 分层**——NB-MSM 假设"defect 位更难"，这里直接验证
- **Q4**：平凡基线（抄前一轮、全局均值）在 defect / non-defect 位上的重建误差——若两类误差相近，假设前提消失
- **Q5**：Arm-0 (`MixedStructuredMSM`) 的 per-shot 掩码数分布——作为两臂公平性的基准

完整探针脚本 `probe_msm_leakage.py`，放在 `BERT/scripts/` 下：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSM 泄漏探针 + 分层难度基线
用途：在正式训练前验证 NB-MSM 假设前提。只读一个 batch，不训练，不写ckpt。

运行：
  cd BERT/scripts
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python probe_msm_leakage.py --distance 5
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

#——————————————————————————————————————————
# sys.path 配置（与 run_nb_msm.py 一致）
# ——————————————————————————————————————————
PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PROJECT_ROOT / "google_paems_data" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, r"D:/Code/LZai/Ai for QEC/BERT/scripts")
import os; os.chdir(str(EXP))

from local_train_5m import load_paems
from run_experiment import make_coord
from alphaqubit.models.pretrain import MaskedSyndromeModeling
from mixed_msm import MixedStructuredMSM


#——————————————————————————————————————————
# Q1：event 时间约定方向
# ——————————————————————————————————————————
def check_event_convention(m: np.ndarray, e: np.ndarray) -> dict:
    """
    判断 event[t] 对应哪对测量值。
    Convention A：event[t] ≈ XOR(meas[t], meas[t-1])
    Convention B：event[t] ≈ XOR(meas[t], meas[t+1])
    soft-XOR：f(p,q) = p + q - 2pq
    """
    def soft_xor(p, q):
        return p + q - 2.0 * p * q

    # Convention A：用 meas[t] 和 meas[t-1] 预测 event[t]，t=1..T-1
    pred_A = soft_xor(m[:, 1:, :], m[:, :-1, :])
    mae_A  = float(np.abs(e[:, 1:, :] - pred_A).mean())

    # Convention B：用 meas[t] 和 meas[t+1] 预测 event[t]，t=0..T-2
    pred_B = soft_xor(m[:, :-1, :], m[:, 1:, :])
    mae_B  = float(np.abs(e[:, :-1, :] - pred_B).mean())

    # round0 特殊检查：可能是零初始化或 meas[0] 本身
    e0_mean= float(e[:, 0, :].mean())
    m0_mean  = float(m[:, 0, :].mean())
    # round0 约定 A 需要 meas[-1]（不存在），常见处理是 event[0] = meas[0]
    mae_round0_as_meas = float(np.abs(e[:, 0, :] - m[:, 0, :]).mean())
    mae_round0_as_zero = float(np.abs(e[:, 0, :]).mean())

    winner = 'A' if mae_A < mae_B else 'B'
    return {
        'convention': winner,
        'mae_convention_A': round(mae_A, 6),
        'mae_convention_B': round(mae_B, 6),
        'event_round0_mean': round(e0_mean, 4),
        'meas_round0_mean':  round(m0_mean, 4),
        'mae_round0_vs_meas[0]': round(mae_round0_as_meas, 6),
        'mae_round0_vs_zero':round(mae_round0_as_zero, 6),
        'interpretation': (
            "event[t] = XOR(meas[t], meas[t-1])，round0 特殊处理"
            if winner == 'A' else
            "event[t] = XOR(meas[t], meas[t+1])，泄漏路径需反向"
        ),
    }


# ——————————————————————————————————————————
# Q3：前向路径可解析恢复比例（convention A 下）
# ——————————————————————————————————————————
def check_forward_leakage(m: np.ndarray, e: np.ndarray,
                          mask: np.ndarray, convention: str) -> dict:
    """
    假设 convention A：event[t+1] = XOR(meas[t+1], meas[t])
    则：meas[t] = (event[t+1] - meas[t+1]) / (1 - 2·meas[t+1])

    一个被掩的meas[t,s] 可解析恢复当且仅当：
      1. t+1 < T（不是最后一轮）
      2. event[t+1, s] 未被掩（可见）
      3. meas[t+1, s] 未被掩（可见）
      4. |1 - 2·meas[t+1,s]| > threshold（分母稳定，即 meas[t+1,s] 不在 [0.4,0.6]）

    Convention B时路径反向（向t-1 看），当前只实现 A。
    """
    if convention != 'A':
        return {'note': 'Convention B 路径未实现，请手动对称处理'}

    B, T, S = m.shape
    DENOM_THRESH = 0.2# |1 - 2p| > 0.2 即 p ∉ [0.4, 0.6]

    results = {
        'total_masked': int(mask.sum()),
        'mask_ratio_actual': float(mask.mean()),
    }

    # 前向邻居可见性矩阵（t< T-1 且 (t+1,s) 未被掩）
    fwd_visible = np.zeros((B, T, S), dtype=bool)
    fwd_visible[:, :-1, :] = ~mask[:, 1:, :]

    # 分母稳定性矩阵（meas[t+1,s] 远离 0.5）
    denom =1.0 - 2.0 * m[:, 1:, :]         # [B, T-1, S]
    denom_stable = np.zeros((B, T, S), dtype=bool)
    denom_stable[:, :-1, :] = np.abs(denom) > DENOM_THRESH

    # 合并：可恢复 = 被掩 & 有前向邻居 & 分母稳定
    recoverable = mask & fwd_visible & denom_stable

    # 实际恢复误差计算
    ib, it, is_ = np.where(recoverable[:, :-1, :])   # 只有 t < T-1 的才可恢复
    rec_count = 0
    mae_list = []
    if len(ib) > 0:
        e_fwd = e[ib, it + 1, is_]
        m_fwd = m[ib, it + 1, is_]
        denom_val = 1.0 - 2.0 * m_fwd
        m_hat = (e_fwd - m_fwd) / denom_val
        # clamp 到 [0,1]：理论上不需要，但极端软值时可能越界
        m_hat = np.clip(m_hat, 0.0, 1.0)
        mae = np.abs(m_hat - m[ib, it, is_])
        mae_list = mae
        rec_count = len(ib)

    # 按defect / non-defect 分层
    defect_mask = e > 0.5
    for label, sel in (('defect', mask & defect_mask),
                       ('non_defect', mask & ~defect_mask)):
        n = int(sel.sum())
        rec_this = recoverable & (sel[:, :-1, :] if False else sel)
        # 精确计算：sel中 t < T-1 的可恢复子集
        ib2, it2, is2 = np.where(sel[:, :-1, :] & recoverable[:, :-1, :])
        n_rec = len(ib2)
        results[label] = {
            'n_masked': n,
            'n_recoverable': n_rec,
            'recoverable_frac': round(n_rec / max(n, 1), 4),
        }
        if n_rec > 0:
            e_f = e[ib2, it2 + 1, is2]
            m_f = m[ib2, it2 + 1, is2]
            d_v = 1.0 - 2.0 * m_f
            m_h = np.clip((e_f - m_f) / d_v, 0.0, 1.0)
            results[label]['recovery_mae'] = round(
                float(np.abs(m_h - m[ib2, it2, is2]).mean()), 6)

    results['overall_recoverable_frac'] = round(
        recoverable.sum() / max(mask.sum(), 1), 4)
    results['soft_value_outside_04_06'] = round(
        float((np.abs(m - 0.5) > 0.1).mean()), 4)

    # 关键结论
    if results['overall_recoverable_frac'] > 0.5:
        results['verdict'] = (
            "⚠️  超过 50% 的被掩位可由前向邻居解析恢复。"
            "预训练任务存在大面积捷径，'defect 位重建更有价值'的假设需结合 Q4 验证。"
        )
    else:
        results['verdict'] = "恢复比例 < 50%，泄漏程度可接受，继续 Q4 验证分层难度。"

    return results


# ——————————————————————————————————————————
# Q4：平凡基线分层难度（假设前提直接验证）
# ——————————————————————————————————————————
def check_trivial_baselines(m: np.ndarray, e: np.ndarray,
                            mask: np.ndarray) -> dict:
    """
    在被掩位上计算三种平凡预测的MAE，按 defect / non-defect 分层。

    若 defect 位的基线 MAE 与 non-defect 位相近（差距< 2×），
    则"defect 位更难以重建"的假设前提不成立。
    """
    B, T, S = m.shape

    # 三种平凡预测
    preds = {
        'copy_prev_round': np.concatenate(
            [m[:, :1, :], m[:, :-1, :]], axis=1),       # meas[t-1,s]
        'copy_next_round': np.concatenate(
            [m[:, 1:, :], m[:, -1:, :]], axis=1),       # meas[t+1,s]
        'global_prior': np.full_like(m, m.mean()),       # 全局均值
    }

    defect_mask = e > 0.5
    results = {}

    for pname, pred in preds.items():
        err = np.abs(pred - m)
        row = {}
        for label, sel in (('all',mask),
                            ('defect',     mask & defect_mask),
                            ('non_defect', mask & ~defect_mask)):
            n = int(sel.sum())
            row[label] = {
                'n': n,
                'mae': round(float(err[sel].mean()), 6) if n > 0 else None,
            }
        # 关键比值：defect vs non-defect 的 MAE 之比
        d_mae = row['defect']['mae']
        c_mae = row['non_defect']['mae']
        if d_mae and c_mae and c_mae > 1e-9:
            ratio = d_mae / c_mae
            row['defect_vs_nondefe ct_ratio'] = round(ratio, 3)
            if ratio < 1.5:
                row['verdict'] = (
                    f"⚠️  defect 位 MAE ({d_mae:.4f}) 与 non-defect 位"
                    f" ({c_mae:.4f}) 差距 < 1.5×。"
                    "'defect 位更难重建'的假设前提不成立，NB-MSM 动机需重新评估。"
                )
            else:
                row['verdict'] = (
                    f"defect 位 MAE {ratio:.2f}× non-defect，"
                    "假设前提成立，继续进行阶段 2。"
                )
        results[pname] = row

    return results


# ——————————————————————————————————————————
# Q5：Arm-0 per-shot 掩码数分布
# ——————————————————————————————————————————
def check_pershot_count(cs, d: int, bs: int = 256) -> dict:
    """
    统计 MixedStructuredMSM 的 per-shot 掩码数分布。
    Arm-0 的基准：40% 路径是 Bernoulli i.i.d.（有方差），
    60% 路径是 while 循环逼近 target_count（近似固定）。
    NB-MSM 改用固定配额后方差更小，此处记录以证明不引入不公平。
    """
    msm = MixedStructuredMSM(
        mask_ratio=0.25, coord_system=cs,
        p_random=0.4, p_spatial=0.3, p_temporal=0.3
    )
    # 构造假数据（只需形状，值不影响 MixedStructuredMSM 的掩码生成）
    T = 10
    n_stab = (d**2 - 1)
    m = torch.zeros(bs, T, n_stab)
    e = torch.zeros(bs, T, n_stab)
    _, mask = msm.mask_sequence(m, e)
    counts = mask.reshape(bs, -1).sum(dim=1).float()
    target = int(0.25 * T * n_stab)
    return {
        'target_per_shot': target,
        'mean': round(float(counts.mean()), 2),
        'std':  round(float(counts.std()), 2),
        'min':  int(counts.min()),
        'max':  int(counts.max()),
        'binomial_theoretical_std': round(float((0.25 * 0.75 * T * n_stab) ** 0.5), 2),
        'note': (
            "40% 路径走 Bernoulli i.i.d.，理论 std 如上；"
            "NB-MSM 固定配额 std≈0，方差更小，不引入新混淆。"
        ),
    }


# ——————————————————————————————————————————
# 主函数
# ——————————————————————————————————————————
def main():
    ap = argparse.ArgumentParser(description="MSM 泄漏探针")
    ap.add_argument('--distance', type=int, required=True, choices=[3, 5, 7])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"MSM 泄漏探针  d={args.distance}  bs={args.batch_size}")
    print(f"{'='*60}\n")

    # 加载一个 batch
    ds, _ = load_paems(args.distance)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    m = batch['measurement'].numpy().astype(np.float32)# [B,T,S]
    e = batch['event'].numpy().astype(np.float32)

    # Arm-1掩码（NB-MSM 需要先有实现，否则只跑Arm-0）
    # 这里用 base class 的纯随机掩码作为泄漏上界估算
    mask_obj = MaskedSyndromeModeling(mask_ratio=0.25)
    m_t = torch.from_numpy(m)
    e_t = torch.from_numpy(e)
    _, mask_t = mask_obj.mask_sequence(m_t, e_t)
    mask = mask_t.numpy()

    report = {
        'distance': args.distance,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'shape': {'B': m.shape[0], 'T': m.shape[1], 'n_stab': m.shape[2]},
        'natural_event_density': round(float((e > 0.5).mean()), 4),
    }

    print("── Q1: event 时间约定 ──")
    q1 = check_event_convention(m, e)
    report['Q1_convention'] = q1
    print(json.dumps(q1, indent=2, ensure_ascii=False))

    convention = q1['convention']

    print("\n── Q3: 前向路径恢复比例 ──")
    q3 = check_forward_leakage(m, e, mask, convention)
    report['Q3_leakage'] = q3
    print(json.dumps(q3, indent=2, ensure_ascii=False))

    print("\n── Q4: 平凡基线分层难度 ──")
    q4 = check_trivial_baselines(m, e, mask)
    report['Q4_baselines'] = q4
    print(json.dumps(q4, indent=2, ensure_ascii=False))

    print("\n── Q5: Arm-0 per-shot 掩码数分布 ──")
    cs = make_coord(args.distance)
    q5 = check_pershot_count(cs, args.distance, args.batch_size)
    report['Q5_pershot'] = q5
    print(json.dumps(q5, indent=2, ensure_ascii=False))

    #── leakage 通道捷径提示 ──
    report['leakage_channel_note'] = (
        "PTBatchDataset 的 leakage/event_leakage 全为零，"
        "经_apply_mask 后被掩位变0.5（mask_token_value）。"
        "预训练阶段模型可从 leakage 通道直接读出掩码位置，"
        "存在免费捷径。双臂共有此问题，不影响 A vs B 对比，"
        "但压低预训练质量上限，已记录于透明披露。"
    )

    # ── 综合 gate 判据 ──
    print("\n── 综合 Gate 判据 ──")
    q3_frac = q3.get('overall_recoverable_frac', 1.0)
    q4_ratio = None
    for pname in ('copy_prev_round', 'global_prior'):
        r = q4.get(pname, {}).get('defect_vs_non_defect_ratio')
        if r:
            q4_ratio = r
            break

    gate = {}
    if q3_frac > 0.5:
        gate['status'] = 'BLOCKED'
        gate['reason'] = (
            f"Q3 可恢复比例 {q3_frac:.0%} > 50%。"
            "必须先修复泄漏（补掩 event[t+1] 或改用时序块掩码），"
            "再进行阶段 2。"
        )
    elif q4_ratio is not None and q4_ratio < 1.5:
        gate['status'] = 'ASSUMPTION_FAILED'
        gate['reason'] = (
            f"Q4 defect/non-defect MAE 比值 {q4_ratio:.2f} < 1.5。"
            "'defect 位更难重建' 假设前提不成立。"
            "建议直接写负面结论，省去 30GPU 小时。"
        )
    else:
        gate['status'] = 'PASS'
        gate['reason'] = (
            f"Q3 可恢复比例 {q3_frac:.0%} ≤ 50%，"
            f"Q4 defect/non-defect 比值 {q4_ratio}。"
            "可进入阶段 2。"
        )
    report['gate'] = gate
    print(json.dumps(gate, indent=2, ensure_ascii=False))

    out_path = Path(f"probe_leakage_d{args.distance}.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n报告已保存 → {out_path}")


if __name__ == '__main__':
    main()
```

**运行方式**（无GPU，约 10秒）：

```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts"
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python probe_msm_leakage.py --distance 5
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python probe_msm_leakage.py --distance 7
```

**Gate 判据与三路分支处置**：

| Q3 可恢复比例 | Q4 defect/non-defect 比值 | 状态 | 处置 |
|---|---|---|---|
| > 50% | 任意 | BLOCKED | 必须先修泄漏再进阶段 2，见附录 |
| ≤ 50% | < 1.5 | ASSUMPTION_FAILED | 假设前提不成立，写负面结论收工 |
| ≤ 50% | ≥ 1.5 | PASS | 进入阶段 2 |

**若状态为 BLOCKED 的修复方案**：最小修改是在 `mask_sequence` 重写后，额外对被掩位的前向事件补掩：

```python
# NoiseBalancedMSM.mask_sequence 里，super() 调用之后追加：
def mask_sequence(self, measurement, event, leakage=None, event_leakage=None):
    self._current_event = event
    try:
        masked_inputs, mask_indices = super().mask_sequence(
            measurement, event, leakage, event_leakage)
        # 补掩 event[t+1,s]：若 (t,s) 被掩，则 (t+1,s) 的 event 也应被掩
        # 避免"从下一轮event反解当前 measurement"的捷径
        fwd_mask = torch.zeros_like(mask_indices)
        fwd_mask[:, :-1, :] = mask_indices[:, 1:, :]   # (t,s) 被掩→(t-1,s)的 fwd 需掩
        # 更直接：被掩位(t,s)的"上游" event[t+1,s] 补掩
        extra = torch.zeros_like(mask_indices)
        extra[:, 1:, :] = mask_indices[:, :-1, :]       # mask[t,s]=True → extra[t+1,s]=True
        combined = mask_indices | extra
        # 对 event 重新 apply（measurement 不变，泄漏路径在 event 侧）
        masked_inputs['event'] = self._apply_mask(event, combined)
        return masked_inputs, mask_indices   # mask_indices 不变，目标不变
    finally:
        self._current_event = None
```

注意这只补掩 `event` 通道，`measurement[t+1]` 仍然可见——反解公式同时需要两者，只要 `event[t+1]` 不可见，反解路径就被切断。

---

###阶段 2：`NoiseBalancedMSM` 重写（半天，需先过阶段 1 gate）

**核心改动**：放弃原来的 batch 级池（引入 per-shot 掩码强度与 label 相关），改为 per-shot 固定配额双边采样，从事实6 已知 Arm-0 有方差，NB-MSM 改固定反而更均匀，不引入新混淆。

完整实现 `noise_balanced_msm.py`：

```python
"""
NoiseBalancedMSM v3：per-shot 固定配额，defect / non-defect 各占一半。

【v1 问题】batch 级池：defect 位入选概率 0.35vs clean 位 0.194（d5），导致高defect shot 的掩码更强（与 label 正相关）。
【v2 问题】batch 级池在 shot 级别仍有相同的混淆，单测只验证 batch聚合。
【v3 修正】per-shot 固定配额：每shot 精确选 n//2 个 defect 位
          + (n - n//2) 个 non-defect 位，唯一变量是 defect 比例。

Convention已由 probe_msm_leakage.py 确认（Q1）。
泄漏状态已由 probe_msm_leakage.py 确认（Q3），若BLOCKED 需先补掩 event[t+1]。
"""
import numpy as np
import torch
from alphaqubit.models.pretrain import MaskedSyndromeModeling


class NoiseBalancedMSM(MaskedSyndromeModeling):

    def __init__(self, mask_ratio: float = 0.25,
                 mask_token_value: float = 0.5,
                 random_replace_prob: float = 0.1,
                 keep_original_prob: float = 0.1,
                 rng_seed: int = 0,
                 fix_forward_leak: bool = True):
        """
        Args:
            fix_forward_leak: 若 Q3 状态为 BLOCKED，设为 True 补掩 event[t+1]。
                Q3 状态为 PASS 时可设False（单一变量保证）。
        """
        super().__init__(
            mask_ratio=mask_ratio,
            mask_token_value=mask_token_value,
            random_replace_prob=random_replace_prob,
            keep_original_prob=keep_original_prob,
        )
        self._current_event = None
        # 专属随机生成器，与全局 np.random 流隔离，保证可复现性
        self._rng = np.random.default_rng(rng_seed)
        self.fix_forward_leak = fix_forward_leak
        # 透明披露统计
        self._last_stats: dict = {}

    #──────────────────────────────────────────────
    # 公开接口（覆写父类）
    # ──────────────────────────────────────────────
    def mask_sequence(self, measurement, event,
                      leakage=None, event_leakage=None):
        """将 event 注入实例，调用父类流程，可选补掩 event[t+1]。"""
        self._current_event = event
        try:
            masked_inputs, mask_indices = super().mask_sequence(
                measurement, event, leakage, event_leakage)

            # 可选：补掩 event[t+1] 切断前向恢复路径
            if self.fix_forward_leak:
                extra = torch.zeros_like(mask_indices)
                extra[:, 1:, :] = mask_indices[:, :-1, :]   # mask[t,s]→extra[t+1,s]
                masked_inputs['event'] = self._apply_mask(event, mask_indices | extra)

            return masked_inputs, mask_indices
        finally:
            self._current_event = None   # 清除，防止 eval 路径拿到脏数据

    # ──────────────────────────────────────────────
    # 核心：per-shot 固定配额双边采样
    # ──────────────────────────────────────────────
    def _generate_mask_indices(self, B, T, n_stab, device):
        if self._current_event is None:
            raise RuntimeError(
                "[NoiseBalancedMSM] _generate_mask_indices 未拿到 event。\n"
                "必须通过 mask_sequence() 调用，不可直接调用此方法。"
            )

        L = T * n_stab
        n = max(2, int(self.mask_ratio * L))   # 每 shot 总掩码数（固定）
        n_def = n // 2                          # defect 配额
        n_cln = n - n_def                       # non-defect 配额

        # [B, L] bool，cpu numpy
        ev_flat = (self._current_event.detach().cpu().numpy() > 0.5
                   ).reshape(B, L)

        mask_flat = np.zeros((B, L), dtype=bool)

        # 统计变量
        deficit_shots = 0
        deficit_positions = 0

        for i in range(B):
            def_idx = np.where(ev_flat[i])[0]     # 本shot defect 位全集
            cln_idx = np.where(~ev_flat[i])[0]    # 本 shot non-defect 位全集

            k_def = min(n_def, len(def_idx))       # 实际能选多少 defect
            k_cln = min(n - k_def, len(cln_idx))  # non-defect 补足到 n

            # 记录缺口
            if k_def < n_def:
                deficit_shots += 1
                deficit_positions += (n_def - k_def)

            # 无放回采样（使用专属 rng，隔离全局流）
            if k_def > 0:
                chosen_def = self._rng.choice(def_idx, k_def, replace=False)
                mask_flat[i, chosen_def] = Trueif k_cln > 0:
                chosen_cln = self._rng.choice(cln_idx, k_cln, replace=False)
                mask_flat[i, chosen_cln] = True

        # 透明披露
        self._last_stats = {
            'n_per_shot': n,
            'n_def_quota': n_def,
            'n_cln_quota': n_cln,
            'deficit_shots': deficit_shots,
            'deficit_positions': deficit_positions,
            'batch_defect_share': float(
                (mask_flat & ev_flat).sum() / max(mask_flat.sum(), 1)),
            'pershot_count_min': int(mask_flat.sum(axis=1).min()),
            'pershot_count_max': int(mask_flat.sum(axis=1).max()),
        }

        return torch.from_numpy(
            mask_flat.reshape(B, T, n_stab)
        ).to(device)
```

**单测补充**（追加进`test_noise_balanced_msm.py`）：

```python
def test_pershot_count_exactly_fixed():
    """per-shot 掩码数必须恒定，不依赖 defect 密度。"""
    for frac, S, tag in ((0.36, 24, 'd5'), (0.30, 48, 'd7'), (0.47, 8, 'd3')):
        m, e = _make_batch(B=64, T=10, n_stab=S, noise_frac=frac, seed=10)
        msm = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=0)
        _, mask = msm.mask_sequence(m, e)
        counts = mask.reshape(64, -1).sum(dim=1)
        n_expected = int(0.25 * 10 * S)
        assert counts.min() == n_expected and counts.max() == n_expected, (
            f"{tag}: per-shot 掩码数应恒为 {n_expected}，"
            f"实得 [{counts.min()},{counts.max()}]"
        )


def test_pershot_defect_share_at_real_density():
    """真实密度下逐shot 的 defect 占比应恒为 0.5（无缺口 shot）。"""
    for frac, S, tag in ((0.36, 24, 'd5'), (0.30, 48, 'd7')):
        m, e = _make_batch(B=128, T=10, n_stab=S, noise_frac=frac, seed=11)
        msm = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=0)
        _, mask = msm.mask_sequence(m, e)
        dm = (e > 0.5)
        def_in_mask = (mask & dm).reshape(128, -1).sum(dim=1).float()
        total = mask.reshape(128, -1).sum(dim=1).float()
        share = def_in_mask / total
        assert share.min().item() > 0.49 and share.max().item() < 0.51, (
            f"{tag}: per-shot defect 占比应≈0.5，"
            f"实得 [{share.min():.3f},{share.max():.3f}]"
        )assert msm._last_stats['deficit_shots'] == 0, \
            f"{tag}: 真实密度下不应出现缺口 shot"


def test_deficit_path_still_fixes_total_count():
    """极低defect 密度下，缺口由 non-defect 补足，总掩码数仍固定。"""
    m, e = _make_batch(B=16, T=10, n_stab=24, noise_frac=0.01, seed=12)
    msm = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=0)
    _, mask = msm.mask_sequence(m, e)
    n_expected = int(0.25 * 10 * 24)
    counts = mask.reshape(16, -1).sum(dim=1)
    assert counts.min() == n_expected and counts.max() == n_expectedassert msm._last_stats['deficit_shots'] > 0, "缺口情况应被记录"


def test_raises_without_event():
    """直接调 _generate_mask_indices 而不经mask_sequence 必须抛错。"""
    import pytest
    msm = NoiseBalancedMSM(mask_ratio=0.25)
    with pytest.raises(RuntimeError, match="未拿到 event"):
        msm._generate_mask_indices(2, 10, 24, torch.device('cpu'))


def test_reproducible_with_same_seed():
    """相同 rng_seed 两次调用结果必须完全一致。"""
    m, e = _make_batch(B=32, T=10, n_stab=24, noise_frac=0.36, seed=13)
    r1 = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=999).mask_sequence(m, e)[1]
    r2 = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=999).mask_sequence(m, e)[1]
    assert torch.equal(r1, r2), "相同 rng_seed 的两次调用结果必须相同"


def test_different_seed_differ():
    """不同 rng_seed 的结果应当不同（防止 seed 被忽略）。"""
    m, e = _make_batch(B=32, T=10, n_stab=24, noise_frac=0.36, seed=14)
    r1 = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=0).mask_sequence(m, e)[1]
    r2 = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=1).mask_sequence(m, e)[1]
    assert not torch.equal(r1, r2), "不同 rng_seed 的结果应当不同"


def test_fix_forward_leak_masks_event_fwd():
    """fix_forward_leak=True 时，被掩位(t,s)对应的 event[t+1,s] 应被掩。"""
    m, e = _make_batch(B=4, T=10, n_stab=8, noise_frac=0.36, seed=15)
    msm = NoiseBalancedMSM(mask_ratio=0.25, rng_seed=0, fix_forward_leak=True)
    masked_inputs, mask_indices = msm.mask_sequence(m, e)
    # 验证：任何 mask[t,s]=True 的位置，event_out[t+1,s] 不应等于原始值
    # （因为被_apply_mask 替换了，80% 变0.5，其余变随机或保原值 → 期望值不等）
    # 注意 10% 保原值仍可能相等，只检查均值偏移
    masked_ev = masked_inputs['event']
    # 找mask 为 True 且 t< T-1 的位置，看t+1 处event 是否被修改
    changed = (masked_ev != e)# [B, T, S]
    # t+1 处被额外掩的位：extra[t+1,s] = mask[t,s]
    extra = torch.zeros_like(mask_indices)
    extra[:, 1:, :] = mask_indices[:, :-1, :]
    # 在 extra=True 且原event不满足 10% 保原值条件的位上，应有改变
    # 放宽检查：80% 的 extra 位上 event 应被替换为 0.5
    frac_changed = (changed & extra).float().sum() / extra.float().sum().clamp(min=1)
    assert frac_changed.item() > 0.7, (
        f"fix_forward_leak=True 时，约 80% 的前向 event 位应被替换，"
        f"实际{frac_changed:.2%}"
    )
```

**运行全部单测**：

```bash
cd "D:/Code/LZai/Ai for QEC/BERT"
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \-m pytest scripts/tests/test_noise_balanced_msm.py -v --tb=short
```

预期：全部通过，含原有 8 条+ 新增 7 条，共 15 条。

**注意**：`run_nb_msm.py` 里的 `build_masking` 函数需要同步更新，传入 `rng_seed` 与 `--seed` 对齐，且 `fix_forward_leak` 的开关取决于阶段 1 的 gate 结果：

```python
def build_masking(strategy, coord_system, seed=42, fix_leak=True):
    if strategy == 'original':
        return MixedStructuredMSM(
            mask_ratio=MASK_RATIO, coord_system=coord_system,
            p_random=0.4, p_spatial=0.3, p_temporal=0.3)
    elif strategy == 'noise_balanced':
        return NoiseBalancedMSM(
            mask_ratio=MASK_RATIO,
            rng_seed=seed,            # 与训练种子绑定，两臂共享基础 seed
            fix_forward_leak=fix_leak # 阶段1 gate=BLOCKED 时设True
        )
    else:
        raise ValueError(f"未知策略: {strategy}")
```

---

### 阶段 3：统计口径（2小时，含预注册，训练前完成）

**3.1 换掉伪 p值**

`paired_bootstrap_nb.py` 当前的 `p = P(diff_boot≤ 0)` 是 bootstrap 分布围绕观测值的左尾概率，不是"H0 为真时的尾概率"，不是合法 p 值。对二值正确性的配对检验，标准做法是 **McNemar 精确检验**：

```python
# paired_bootstrap_nb.py 替换 paired_bootstrap 函数，保留 CI 计算
from scipy.stats import binomtest
import numpy as np


def mcnemar_exact(correct1: np.ndarray, correct0: np.ndarray,
                  alternative: str = 'greater') -> dict:
    """
    配对 McNemar 精确检验。
    b = arm1 对、arm0 错的不一致对数
    c = arm1 错、arm0 对的不一致对数
    H1: b > c（arm1 更准确），单侧。

    Args:
        correct1: [N] bool，arm1 每样本是否正确
        correct0: [N] bool，arm0 每样本是否正确
        alternative: 'greater'（H1 arm1 > arm0）或 'two-sided'

    Returns:
        包含 b、c、p_exact、discordant_rate、diff的字典
    """
    b = int(np.sum(correct1 & ~correct0))   # arm1 对、arm0 错
    c = int(np.sum(~correct1 & correct0))   # arm1 错、arm0 对
    n = len(correct1)
    if b + c == 0:
        return {
            'b': 0, 'c': 0, 'discordant_rate': 0.0,
            'diff': 0.0, 'p_exact': 1.0,
            'note': '无不一致对，两臂在所有样本上预测完全相同'
        }
    result = binomtest(b, b + c, 0.5, alternative=alternative)
    return {
        'b': b,
        'c': c,
        'discordant_rate': round((b + c) / n, 4),
        'diff': round((b - c) / n, 4),    # 效应量（pp）
        'p_exact': round(float(result.pvalue), 6),
    }


def bootstrap_ci(correct1: np.ndarray, correct0: np.ndarray,
                 B: int = 10000, seed: int = 0) -> dict:
    """
    配对 bootstrap 95% CI（仅用于描述效应量区间，不用于 p 值判断）。
    重采样单位为样本，保持配对关系。
    """
    paired_diff = correct1.astype(np.float64) - correct0.astype(np.float64)
    n = len(paired_diff)
    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot_diffs[i] = paired_diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        'observed_diff': round(float(paired_diff.mean()), 4),
        'ci_95_lo': round(float(ci_lo), 4),
        'ci_95_hi': round(float(ci_hi), 4),
        'B': B,
    }
```

`main()` 里把两个函数都跑，分工明确：McNemar 定显著性，bootstrap CI 描述效应量规模：

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, required=True, choices=[3, 5, 7])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag1', default='arm1')
    ap.add_argument('--tag0', default='arm0')
    ap.add_argument('--B', type=int, default=10000)
    args = ap.parse_args()

    p1, lb = load_preds(args.distance, args.tag1, args.seed)
    p0, _= load_preds(args.distance, args.tag0, args.seed)
    assert len(p1) == len(p0) == len(lb), \
        f"两臂预测数不一致：arm1={len(p1)}, arm0={len(p0)}, label={len(lb)}"

    correct1 = (p1 == lb)
    correct0 = (p0 == lb)
    acc1 = float(correct1.mean())
    acc0 = float(correct0.mean())

    mc = mcnemar_exact(correct1, correct0, alternative='greater')
    ci = bootstrap_ci(correct1, correct0, B=args.B, seed=args.seed)

    v = verdict_preregistered(
        diff=mc['diff'],
        ci_lo=ci['ci_95_lo'],
        ci_hi=ci['ci_95_hi'],
        p_exact=mc['p_exact'],
        mde=None,           # 实测MDE 由 3.2 填入
    )

    out = {
        'distance': args.distance, 'seed': args.seed,
        'acc_arm1': round(acc1, 6), 'acc_arm0': round(acc0, 6),
        'mcnemar': mc,
        'bootstrap_ci': ci,
        'verdict': v,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    path = EXP / f"paired_test_d{args.distance}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n结果已保存 → {path}")
```

**3.2 先量MDE，再决定跑不跑**

McNemar 的最小可检测效应（单侧 α=0.05、power=0.80）：

```
MDE ≈ (z_α + z_β) / √(N × discordant_rate)= 2.49 / √(N × d_rate)
```

`discordant_rate` 不用猜，可以用任意两个已有的同臂 checkpoint（不同 seed 但同策略）先跑一次eval，拿到两份 npz 后计算：

```python
# estimate_mde.py —— 训练前运行，只需已有的任意两个 ckpt（同臂不同初始化）
import numpy as np, json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-a', required=True)
    ap.add_argument('--npz-b', required=True)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--power', type=float, default=0.80)
    args = ap.parse_args()

    za =1.645# 单侧 α=0.05
    zb = 0.842  # power=0.80
    z = za + zb  # 2.49 近似（精确值见 scipy.stats.norm.ppf）

    a = np.load(args.npz_a)
    b = np.load(args.npz_b)
    lb = a['labels']
    c_a = (a['preds'] == lb)
    c_b = (b['preds'] == lb)
    discordant = int((c_a & ~c_b).sum() + (~c_a & c_b).sum())
    n = len(lb)
    d_rate = discordant / n
    if d_rate < 1e-9:
        print("两份预测完全一致，无法估算 MDE")
        return
    mde = z / (n * d_rate) ** 0.5
    print(json.dumps({
        'N': n,
        'discordant': discordant,
        'discordant_rate': round(d_rate, 4),
        'MDE_pp': round(mde * 100, 3),
        'note': f'要检测 > {mde*100:.2f}pp的差异，power={args.power}, alpha={args.alpha}'
    }, indent=2))

if __name__ == '__main__':
    main()
```

运行：

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python estimate_mde.py \--npz-a preds_d5_arm0_s42.npz \
  --npz-b preds_d5_arm0_s123.npz   # 用同臂不同 seed 的两个ckpt
```

把输出的 `MDE_pp` 填进预注册文件，再决定是否启动 d7 的训练。d7 只有 5000 个测试样本，预期 MDE 约 1.1pp，与预期效应量同阶，很可能只能得到 inconclusive——这个结论在开训前就应该写进预注册。

**3.3 种子设计**

把"赢了才复现seed=123"改成**无条件预注册**。两个原因：条件性停止引入方向性偏倚；2 个 seed 无法估计 seed 方差。

改为：**d5 两臂各 3 个 seed（42/123/2024），共 6 次预训练 + 6 次微调，全部无条件跑完**。成本与原计划的 d3+d5+d7 各两臂（6+6）持平，换来了 seed方差估计和一个可解释的码距。d3/d7 等d5 出结论后再做。

**3.4 预注册裁决规则**（训练启动前写进文件锁死）

新建 `preregistration_d5.json`，内容：

```json
{
  "date": "2026-07-29",
  "hypothesis": "H1: NB-MSM (per-shot1:1 defect) 在 d5 真机test acc 上优于 MixedStructuredMSM",
  "primary_outcome": "d5 真机 test accuracy（二值正确性，McNemar 精确检验）",
  "seeds": [42, 123, 2024],
  "verdict_rule": {
    "H1_supported": "3/3 seed 的 McNemar 单侧 p< 0.05，且 3/3 seed 的 diff > 0，且 3-seed diff 均值 > MDE（见estimate_mde.py 实测值）",
    "H1_refuted": "3/3 seed 的 McNemar 单侧 p < 0.05（对立方向）且 3/3 diff< 0",
    "inconclusive": "其余所有情况，含方向一致但幅度 < MDE"
  },
  "secondary_outcome": "LER（仅描述性，不参与裁决，因存在模态失配：预训练合成数据、微调真机硬读出、LER 评估合成软读出）",
  "no_post_hoc": "不做事后子集分析，不拆分 defect/non-defect 子集单独报告，不以 LER 替代主指标",
  "mde_pp": "待estimate_mde.py 实测后填入",
  "locked": true,
  "note": "本文件训练启动后不可修改。任何偏离需在报告中明确声明为协议外分析。"
}
```

**3.5 npz 同时存logit（校准检查用）**

`do_eval` 里改一行：

```python
# 原
preds.append((torch.sigmoid(logit) > 0.5).float().cpu())

# 改：同时保存 logit 供AUC 和校准分析
logits_list.append(logit.cpu())
preds.append((torch.sigmoid(logit) > 0.5).float().cpu())

# 保存时
np.savez(str(EXP / f"preds_d{d}_{tag}_s{seed}.npz"),
         preds=preds_arr, labels=labels_arr, logits=logits_arr)
```

报告里补一条AUC 对照：若两臂 AUC 差异与 accuracy 差异方向一致，说明是判别力提升；若 accuracy 差异大而 AUC 差异小，说明是阈值偏移，不是真实提升。

---

### 阶段 4：数据/物理层排查（半天，只读，与阶段 1 并行）

这一步的期望价值最高——它可能在训练启动前，就解释掉 NN 落后MWPM 7-19pp 的现象，进而改变整个实验的优先级。

**4.1 一个可在 5分钟内验证的假设**

设计文档报告的 event 密度（d3 47%、d5 36%、d7 30%）随码距**递减**，与物理直觉（更大的码距理论上物理错误率不变）相反。拿旋转表面码的边界稳定子占比来拟合：

```
边界 weight-2稳定子比例 f_boundary = 2(d-1) / (d²-1) = 2/(d+1)
f_boundary(d=3)=0.500, f_boundary(d=5)=0.333, f_boundary(d=7)=0.250
```

假设边界稳定子因Pauli frame / 极性预处理错误导致恒定firing（detection fraction≈90%），其余稳定子真实检测密度约p_det：

```
density = 0.9 × f_boundary + p_det × (1 - f_boundary)
```

用三个码距联立：p_det(d3)≈0.05，p_det(d5)≈0.08，p_det(d7)≈0.10，落在 AQ2 Fig.2a 的 Willow 真机量级（5-10%）。**这一个假设同时解释了三个码距的绝对值和递减趋势**，且把"其余部分"对齐到正常量级。

**4.2 诊断脚本 `diagnose_event_channel.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event通道物理合理性诊断。
主要问题：event 密度 30-47% 远高于 Willow/Sycamore 的 5-10%，
且随码距递减——疑似边界稳定子预处理错误（Pauli frame / 极性）。

运行：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \diagnose_event_channel.py --distance 5
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# sys.path 配置（同run_nb_msm.py，略）
PROJECT_ROOT = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main/alphaquibit-main")
EXP = PROJECT_ROOT / "google_paems_data" / "bert_experiment"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PROJECT_ROOT / "google_paems_data" / "code"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, r"D:/Code/LZai/Ai for QEC/BERT/scripts")
import os; os.chdir(str(EXP))

from local_train_5m import load_paems
from run_experiment import make_coord


def diagnose(d: int, max_batches: int = 20, bs: int = 512) -> dict:
    cs = make_coord(d)
    ds, _ = load_paems(d)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    meas_list, event_list = [], []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        meas_list.append(b['measurement'].numpy())
        event_list.append(b['event'].numpy())

    m = np.concatenate(meas_list, axis=0).astype(np.float32)# [N, T, S]
    e = np.concatenate(event_list, axis=0).astype(np.float32)
    N, T, S = m.shape

    report = {'distance': d, 'N_shots': N, 'T': T, 'n_stab': S}

    # ── 1. 全局统计 ──────────────────────────────
    report['global_event_density'] = round(float((e > 0.5).mean()), 4)
    report['global_meas_density_below_01'] = round(float((m < 0.1).mean()), 4)
    report['global_meas_density_above_09'] = round(float((m > 0.9).mean()), 4)
    report['global_meas_uncertain_04_06'] = round(float(
        ((m > 0.4) & (m < 0.6)).mean()), 4)

    # ── 2. 逐稳定子 event 率（核心诊断）──────────
    per_stab = (e > 0.5).mean(axis=(0, 1))   # [S]，按稳定子平均
    sorted_rates = np.sort(per_stab)
    report['per_stab_event_rate'] = {
        'min':round(float(sorted_rates.min()), 4),
        'max':    round(float(sorted_rates.max()), 4),
        'mean':   round(float(sorted_rates.mean()), 4),
        'p10':    round(float(np.percentile(sorted_rates, 10)), 4),
        'p50':    round(float(np.percentile(sorted_rates, 50)), 4),
        'p90':    round(float(np.percentile(sorted_rates, 90)), 4),
        'bimodal_suspected': bool(
            sorted_rates.max() - sorted_rates.min() > 0.3),'high_rate_stabs_frac': round(
            float((per_stab > 0.7).sum() / S), 4),
        'low_rate_stabs_frac':  round(
            float((per_stab < 0.15).sum() / S), 4),
    }
    #若双峰（一簇 ~0.9 + 一簇 ~0.05），与边界稳定子极性错误假设吻合
    report['per_stab_event_rate']['interpretation'] = (
        "检测到双峰分布：高频稳定子可能为边界 weight-2，极性处理有误"if report['per_stab_event_rate']['bimodal_suspected']
        else "分布较均匀，无明显双峰，边界极性假设待进一步验证"
    )

    # ── 3. 按稳定子 weight 分组（需coord_system 提供 weight信息）──
    try:
        stab_pos = cs.stab_positions_tensor.cpu().numpy()  # [S, 2]
        # 估算 weight：边界稳定子坐标在格点边缘（x 或 y 为0或d-1）
        # 注意：不同 coord_system 的坐标系可能不同，此处为近似
        is_boundary = (
            (stab_pos[:, 0] <=0.5) | (stab_pos[:, 0] >= d - 1.5) |
            (stab_pos[:, 1] <= 0.5) | (stab_pos[:, 1] >= d - 1.5)
        )
        boundary_rate = float(per_stab[is_boundary].mean()) if is_boundary.any() else None
        interior_rate = float(per_stab[~is_boundary].mean()) if (~is_boundary).any() else None
        n_boundary = int(is_boundary.sum())
        # 理论边界比例
        theory_boundary_frac = 2 * (d - 1) / (d**2 - 1)
        report['boundary_vs_interior'] = {
            'n_boundary_stabs': n_boundary,
            'theory_boundary_frac': round(theory_boundary_frac, 4),
            'actual_boundary_frac': round(n_boundary / S, 4),
            'boundary_event_rate': round(boundary_rate, 4) if boundary_rate else None,
            'interior_event_rate': round(interior_rate, 4) if interior_rate else None,'rate_ratio': round(boundary_rate / max(interior_rate, 1e-9), 2)if (boundary_rate and interior_rate) else None,
        }
        if boundary_rate and interior_rate and boundary_rate > 3* interior_rate:
            report['boundary_vs_interior']['verdict'] = (
                f"⚠️ 边界稳定子 event 率（{boundary_rate:.2%}）是内部的 "
                f"{boundary_rate/interior_rate:.1f}×。强烈支持边界极性预处理错误假设。"
                "建议对比 stim 生成的原始 detection_events 和本数据集的 event 字段。"
            )
        else:
            report['boundary_vs_interior']['verdict'] = (
                "边界/内部 event 率差异不显著，极性假设证据不足。"
            )
    except Exception as ex:
        report['boundary_vs_interior'] = {'error': str(ex)}

    # ── 4. 逐轮event 率（时间稳定性）───────────
    per_round = (e > 0.5).mean(axis=(0, 2))   # [T]
    report['per_round_event_rate'] = {
        'round_0': round(float(per_round[0]), 4),
        'round_mid': round(float(per_round[T // 2]), 4),
        'round_last': round(float(per_round[-1]), 4),
        'std_across_rounds': round(float(per_round.std()), 4),
        'note': (
            "round_0 通常特殊（初始化约定），round_last 也特殊（data qubit 读出）。"
            "中间各轮应基本平稳。"
        ),
    }

    # ── 5. 硬比特检测密度（绕过软值定义）─────────
    hard_m = (m > 0.5).astype(np.float32)
    # 硬检测事件：相邻轮测量值不同
    hard_det = (hard_m[:, 1:, :] != hard_m[:, :-1, :]).astype(np.float32)
    report['hard_detection_density'] = round(float(hard_det.mean()), 4)
    # 若软event 密度远高于硬检测密度，说明软值里有额外信息或定义不一致
    soft_hard_ratio = report['global_event_density'] / max(
        report['hard_detection_density'], 1e-9)
    report['soft_vs_hard_event_ratio'] = round(soft_hard_ratio, 2)
    if abs(soft_hard_ratio - 1.0) > 0.3:
        report['soft_vs_hard_note'] = (
            f"软值 event 密度 / 硬检测密度 = {soft_hard_ratio:.2f}，偏离 1.0 超30%。"
            "可能原因：①软值阈值 0.5 不是硬检测的正确分界；"
            "②soft-XOR 定义与detection event 定义不同；"
            "③预处理时存在额外极性翻转。"
        )

    # ── 6. shot label 与 event 密度的相关性 ──────
    labels = np.array([b['label'].numpy().flatten() for b in
                       [next(iter(DataLoader(ds, batch_size=N, shuffle=False)))]][0])[:N]
    event_density_per_shot = (e > 0.5).mean(axis=(1, 2))   # [N]
    corr = float(np.corrcoef(labels, event_density_per_shot)[0, 1])
    report['label_event_density_correlation'] = round(corr, 4)
    if abs(corr) > 0.3:
        report['label_event_correlation_note'] = (
            f"label 与 per-shot event 密度的Pearson r = {corr:.3f}，"
            "两者相关性较强。NB-MSM 选更多 defect 位的 shot 理论上掩码数固定（v3 per-shot 配额），"
            "但 defect 位的空间分布与 label仍然相关，应在报告中披露。"
        )

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distance', type=int, required=True, choices=[3, 5, 7])
    ap.add_argument('--max-batches', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    report = diagnose(args.distance, args.max_batches, args.batch_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(f"event_channel_diagnosis_d{args.distance}.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n报告已保存 → {out}")


if __name__ == '__main__':
    main()
```

**运行**（与阶段 1 并行，只读数据，不占GPU）：

```bash
cd "D:/Code/LZai/Ai for QEC/BERT/scripts"
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  diagnose_event_channel.py --distance 3
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  diagnose_event_channel.py --distance 5
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  diagnose_event_channel.py --distance 7
```

**4.3 三类结果及处置**

| `per_stab_event_rate.bimodal_suspected` | `boundary_vs_interior.rate_ratio` | 含义 | 处置 |
|---|---|---|---|
| True | > 3 | 边界极性预处理错误 | 修复预处理或排除边界稳定子后重跑阶段 1；掩码比例问题退居次要 |
| False | < 2 | 无双峰，密度普遍偏高 | 检查stim 生成脚本的 SI1000 版本（AQ1 vs AQ2 解释不同，事件率有差异） |
| 任意 | 任意 | 软/硬检测密度比值 > 1.3 | 检查 soft-XOR 定义与 detection event 定义是否一致 |

**4.4 "准确率随码距下降"的另一个解释**

报告里应补充一条分析：自报真机训练量real_d3≈多位置共约 400k，real_d5=160k，real_d7=40k，差了 4-10倍。准确率 d3 0.9190→ d5 0.8721 → d7 0.7782的排序与真机训练量完全一致，与码距（物理上更大码距有更低LER）相反。MWPM 不需要训练，随码距改善是正常物理行为。

这意味着 d7 的性能瓶颈大概率是**真机数据量**而不是掩码策略，NB-MSM 在d7 上的效果即使存在也会被数据量噪声淹没，d7 的 GPU 资源应优先分配给扩大真机微调数据量（数据增强/更多轮次），而不是掩码策略实验。这个结论应记录在预注册文件里，作为 d7 结果权重降级的依据。

---

###阶段 5：P1 代码审查门（修订后清单）

审查组subagent 的审查 prompt需要更新，覆盖 v3 的改动和新增的诊断脚本：

```
审查 NB-MSM v3 工程的 P1 代码：
scripts/noise_balanced_msm.py
scripts/tests/test_noise_balanced_msm.py
scripts/run_nb_msm.py
scripts/probe_msm_leakage.py
scripts/diagnose_event_channel.py
scripts/analyze_mask_distribution.py
scripts/paired_bootstrap_nb.py
scripts/estimate_mde.py

对照设计文档和本次修订确认的 7 个事实，审查以下要点：

【A逻辑正确性】
A1. _generate_mask_indices 的per-shot 固定配额：每shot 掩码总数是否恒为 int(mask_ratio × T × n_stab)，不依赖 defect 密度？
A2. 专属 rng（np.random.default_rng(seed)）是否与全局 np.random 流完全隔离？
    build_masking 里 rng_seed 是否与训练 seed绑定？两臂是否传入相同 seed？
A3. fix_forward_leak=True 时，extra矩阵的方向是否正确？
    被掩 mask[t,s]=True → 需补掩event[t+1,s]（前向），不是 event[t-1,s]（后向）。
A4. _current_event的 finally 清除：confirm它在 mask_sequence 的任何异常路径下都会被清除？
A5. do_eval里 logit 的保存：确认是 raw logit（在sigmoid之前），不是概率。

【B 语义正确性】
B1. 父类 _apply_mask 的 80/10/10 规则是否完整保留？
    NoiseBalancedMSM 只重写 _generate_mask_indices 和 mask_sequence，
    未触碰 _apply_mask 和 get_targets，确认？
B2. get_targets 仍返回 measurement[mask_indices]（目标是 measurement，不是 event），确认？
B3. McNemar 检验的 b、c 计数方向：
    b = arm1对且arm0错（支持H1）
    c = arm1错且arm0对（反对H1）
    单侧 'greater' 即 b > c，确认与 binomtest 参数对齐？

【C 公平性与可复现性】
C1. 两臂训练种子：set_seed(seed) 在 do_pretrain/do_finetune 开头调用，Arm-0 和 Arm-1 传入相同 seed=42（及后续123/2024），确认？
C2. 微调掺杂子集的 rng：np.random.default_rng(42)，两臂相同，确认？
C3. ckpt 目录含 tag+seed 后缀（bert_pretrain_d5_arm1_s42），两臂不会互相覆盖，确认？
C4. do_eval 读取的是 ft_dir/best.pt 而不是 pre_dir，确认？
C5. 预注册文件 preregistration_d5.json 的 locked=true，
    训练启动后任何修改都需在报告中明确声明，已写入，确认？

【D 反作弊】
D1. probe_msm_leakage.py 是否真用了一个真实数据 batch（load_paems），
    而不是全零张量或伪造数据？
D2. diagnose_event_channel.py 的 boundary判定坐标系：
    coord_system.stab_positions_tensor 的坐标单位是否与 d的取值匹配？
    （若坐标已归一化到[-0.5,0.5]，则边界判定阈值需对应调整）
D3. per-sample npz 里的 preds 来自模型前向（batch循环），
    未读取已有结果文件或硬编码，确认？

【E 透明披露完整性】
E1. leakage 通道捷径（全零→被掩后变0.5）已记录在 probe报告和最终报告中？
E2. fix_forward_leak 的设定（True/False）及Q3 gate 结论已记录？
E3. NB-MSM 与 MixedStructuredMSM 的 per-shot 掩码数方差差异已记录？
E4. label与 per-shot event 密度的相关性（Q6 结果）已记录？
E5. d7 真机数据量是d5 的 1/4 这一混淆因素已在报告中披露？

输出格式：APPROVE / APPROVE_WITH_CONDITIONS / REJECT + 必改项清单（按 A/B/C/D/E 分节）
```

**审查必过才能进阶段6**，不能以"边跑边改"替代。

---

### 阶段 6：D5 三seed 训练

**d5 六次预训练按顺序排队，避免同时占满 12GB显存**：

```bash
# ── Arm-0 seed=42 ──────────────────────────────────────────────
cd "D:/Code/LZai/Ai for QEC/BERT/scripts"
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u \
  run_nb_msm.py --distance 5 --mask-strategy original \
  --stage pretrain --seed 42 --tag arm0 \
  2>&1 | tee -a logs/nb_d5_arm0_s42_pretrain.log

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u \
  run_nb_msm.py --distance 5 --mask-strategy original \
  --stage finetune --seed 42 --tag arm0 \
  2>&1 | tee -a logs/nb_d5_arm0_s42_finetune.log

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  run_nb_msm.py --distance 5 --mask-strategy original \
  --stage eval --seed 42 --tag arm0

# ── Arm-1 seed=42 ──────────────────────────────────────────────
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u \
  run_nb_msm.py --distance 5 --mask-strategy noise_balanced \
  --stage pretrain --seed 42 --tag arm1 \
  2>&1 | tee -a logs/nb_d5_arm1_s42_pretrain.log

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python -u \
  run_nb_msm.py --distance 5 --mask-strategy noise_balanced \
  --stage finetune --seed 42 --tag arm1 \
  2>&1 | tee -a logs/nb_d5_arm1_s42_finetune.log

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  run_nb_msm.py --distance 5 --mask-strategy noise_balanced \
  --stage eval --seed 42 --tag arm1

# ── Arm-0 seed=123 ─────────────────────────────────────────────
# （arm0/arm1 的 seed=123/2024 依此类推，tag 不变，seed参数变）
```

**每次训练结束后立即验证 ckpt 存在**：

```bash
ls -lh checkpoints/bert_pretrain_d5_arm0_s42/best.pt
ls -lh checkpoints/bert_pretrain_d5_arm1_s42/best.pt
```

**训练中途监控指标**（每 500 步打一次 val_loss 和 mask_acc）：

若某臂的 val_mask_acc 在 5000 步内没有超过 0.70，说明预训练没有收敛，需要检查：
- Arm-1 的 `fix_forward_leak` 是否设对了（Q3 gate 结论）
- `build_masking` 里 `rng_seed` 是否正确传入
- leakage 通道捷径是否被模型直接利用（val_mask_acc 异常快速达到 0.8反而是坏信号）

---

### 阶段 7：D5 统计裁决

**7.1 MDE 实测**（用 seed=42 的两个 Arm-0 npz，在 seed=123 的 Arm-0跑完后）：

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
  estimate_mde.py \
  --npz-a preds_d5_arm0_s42.npz \
  --npz-b preds_d5_arm0_s123.npz
```

把输出的 `MDE_pp` 填入 `preregistration_d5.json`。

**7.2 三seed 配对检验**：

```bash
for SEED in 42 123 2024; do
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /d/condapy/quantum_env/python \
    paired_bootstrap_nb.py --distance 5 --seed $SEED
done
```

**7.3 三 seed 汇总裁决**：

```python
# summarize_d5.py —— 读三份 paired_test_d5_s*.json，输出最终裁决
import json
from pathlib import Path

EXP = Path(r"D:/Code/LZai/Ai for QEC/Alpha-qubit/code/alphaquibit-main"r"/alphaquibit-main/google_paems_data/bert_experiment")

results = []
for seed in [42, 123, 2024]:
    p = EXP / f"paired_test_d5_s{seed}.json"
    if p.exists():
        results.append(json.loads(p.read_text()))

diffs= [r['mcnemar']['diff'] for r in results]
ps= [r['mcnemar']['p_exact'] for r in results]
cis_lo  = [r['bootstrap_ci']['ci_95_lo'] for r in results]

# 读取预注册的 MDE
prereg  = json.loads((EXP / 'preregistration_d5.json').read_text())
mde_pp  = prereg.get('mde_pp', None)

# 裁决逻辑（严格按预注册规则）
all_positive = all(d > 0 for d in diffs)
all_sig= all(p< 0.05 for p in ps)
above_mde    = (mde_pp and all(d > mde_pp / 100 for d in diffs))

if all_positive and all_sig and above_mde:
    final ='H1_SUPPORTED'
elif all(d < 0 for d in diffs) and all_sig:
    final = 'H1_REFUTED'
else:
    final = 'INCONCLUSIVE'

summary = {
    'n_seeds': len(results),
    'diffs': [round(d, 4) for d in diffs],
    'p_exacts': [round(p, 6) for p in ps],
    'ci_95_los': [round(c, 4) for c in cis_lo],
    'mde_pp': mde_pp,
    'final_verdict': final,
    'note': '按preregistration_d5.json 规则裁决，协议外分析需单独声明',
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
(EXP / 'final_verdict_d5.json').write_text(
    json.dumps(summary, indent=2, ensure_ascii=False))
```

---

### 阶段 8：报告结构（P4 sign-off 前）

报告 `NB_MSM_EXPERIMENT_REPORT.md` 按以下结构组织，强制要求各节按顺序写，不允许跳过或合并：

```
## 1. 实验动机与假设
  - 背景：BERT-1 5.5M 在 d5/d7 落后 MWPM 7-19pp
  - 假设链：event密度不均匀 → defect 位重建更难 → 1:1 平衡改善预训练
  - 假设前提验证结果（Q3/Q4 probe 结论）
  - d7 数据量混淆因素声明

## 2. 干预设计
  - 唯一变量：掩码策略（MixedStructuredMSM vs NoiseBalancedMSM v3）
  - v3 vs v1/v2 的修订原因（per-shot 配额vs batch 级池）
  - fix_forward_leak 设定及依据

## 3. 透明披露
  - leakage 通道捷径（全零→0.5 标记）
  - NB-MSM per-shot 掩码数恒定，Arm-0 有Binomial 方差
  - label 与 per-shot event 密度的相关性（r值）
  - 预训练合成软读出、微调真机硬读出、LER 评估合成软读出的模态失配
  - d7 真机数据量（40k）是 d5（160k）的 1/4

## 4. 归因证据
  - Arm-0 被掩位defect 占比≈ 自然密度（d5 36% / d7 30%）
  - Arm-1 被掩位 defect 占比 ≈ 50%（干预落实）
  - NB-MSM _last_stats 中 deficit_shots（应为 0）

## 5. 主结果
  - 三seed（42/123/2024）× 两臂 的 test accuracy表格
  - McNemar p值、效应量diff（pp）、95% CI 表格
  - 最终裁决（H1_SUPPORTED / H1_REFUTED / INCONCLUSIVE）

## 6. 次要结果（描述性）
  - LER 对照（附模态失配声明）
  - 预训练 val_mask_acc 曲线对比（两臂收敛速度）

## 7. 结论与解读
  - 若 H1_SUPPORTED：效应量与 MDE 的比值；可能机制；建议后续实验
  - 若 H1_REFUTED：1:1 平衡的负面效果解释；预训练-微调域差距分析
  - 若 INCONCLUSIVE：样本量估算；建议的后续样本量或替代方向

## 8. 协议外分析（若有）
  - 明确标注为非预注册，解读权重降级
```

---

### 阶段执行顺序总结

```
并行：阶段 0（30分钟）+ 阶段 4（1小时，无GPU）↓
       阶段 1（10秒，Q3/Q4 gate）↓
  gate=PASS → 阶段 2（实现 + 单测，半天）
  gate=BLOCKED → 修复 fix_forward_leak，重跑 gate
  gate=ASSUMPTION_FAILED → 写负面结论，不进阶段 6↓
       阶段 3（预注册，2小时，不占GPU）
           ↓阶段 5（P1 审查门，必过）
           ↓
       阶段 6（d5 六次训练，顺序排队，约 4× 2h = 8h）
           ↓
       阶段 7（统计裁决，1小时）
           ↓
       阶段 8（报告 + P4 sign-off）
```

**d3/d7 的位置**：d3 在 d5 结论出来后，若INCONCLUSIVE 才跑（作为诊断，不作主胜证据）；d7 在 d5 H1_SUPPORTED 且 MDE 估算通过后才决定是否跑，考虑到 40k 测试集的样本量限制，大概率直接跳过，在报告里用 d7 数据量不足作依据声明。