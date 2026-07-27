# 量子纠错解码器训练数据形式规范

> 版本：v3.0  
> 适用范围：任意量子纠错码、任意噪声模型生成的合成数据  
> 目标：定义后续新模型生成数据时必须满足的形式规范，确保与当前训练、评估、MWPM 基准代码兼容  
> 重要声明：本规范是**纯形式契约**，不强制使用 Stim、不强制任何特定噪声模型、不强制 `p`/`snr` 与 Stim 电路参数等价。新模型可采用自定义量子模拟器、神经网络生成模型或其他噪声通道，只要输出满足本规范即可。

---

## 1. 概述

本规范是一份**数据形式契约**，不强制规定数据必须如何物理生成（如使用 Stim、自定义模拟器、神经网络生成模型、或其他噪声模型），也不强制使用任何特定的噪声模型参数。数据生成方可采用任意物理模型、错误通道或采样策略。本规范只约束最终输出形式：

1. **字段名与数据类型**
2. **张量形状**
3. **字段之间的语义关系**
4. **文件格式与命名规范**
5. **一致性检查的通用标准**

任何数据生成方法（Stim、自定义量子电路模拟器、神经网络生成模型等）只要输出的 `.pt` 或 `.npz` 文件满足本规范，即可被当前训练脚本、`PTBatchDataset`、MWPM 评估和 LER 评估直接使用。

---

## 2. 数据提供方契约（Data Provider Contract）

数据生成方负责生成并保存以下信息。本规范不约束生成方式，只约束最终输出形式。

### 2.1 必需字段

| 字段名 | 形状 | 类型 | 取值范围/说明 |
|--------|------|------|---------------|
| `measurement` | `[N, T, n_stab]` | float32 | 每轮每个 stabilizer 的（软）测量结果。范围建议 [0, 1]，但可扩展。0 表示确定测得 0，1 表示确定测得 1，中间值表示不确定性。 |
| `event` | `[N, T, n_stab]` | float32 | 每轮每个 stabilizer 的（软）detection event。应与 `measurement` 满足 `event[t] = XOR-like(measurement[t], measurement[t-1])`，第一轮 `event[0] = measurement[0]`。 |
| `final_soft` | `[N, n_data]` | float32 | 最终轮次对 data qubit 的（软）测量结果。范围建议 [0, 1]。 |
| `label` | `[N]` 或 `[N, n_logical]` | float32 | 每个样本的逻辑 observable 标签。对于单逻辑 qubit 为 `[N]`；多逻辑 qubit 为 `[N, n_logical]`，每个元素为 0 或 1。 |
| `detection_events` | `[N, T, n_stab]` 或 `[N, num_detectors]` | float32 | 供 MWPM 使用的 detection events。必须与 `label` 来自同一次底层错误采样，且顺序与对应码的 DEM（detector error model）一致。 |
| `distance` | 标量 | int | 码距离。对于非传统距离概念的码族，可用其等价参数替代，但需保证 `n_stab` 和 `n_data` 的计算与本规范一致。 |
| `rounds` | 标量 | int | 纠错轮数 T。 |
| `p` | 标量 | float | 物理错误率或等效噪声强度参数。仅作为元数据记录，不用于约束噪声模型形式。 |
| `snr` | 标量 | float | 软读出信噪比或等效读出质量参数。仅作为元数据记录。 |

### 2.2 派生量（由数据使用方计算）

数据使用方将根据 `distance` 推导：

```python
n_stab = distance ** 2 - 1      # 稳定子数量
n_data = distance ** 2          # 数据比特数量
T = rounds
N = len(label)
```

**重要**：如果生成的码不满足 `n_stab = d² - 1` 和 `n_data = d²`，则 `distance` 字段仅作为标识符使用，此时 `measurement.shape[-1]`、`event.shape[-1]`、`final_soft.shape[-1]` 将直接决定 `n_stab` 和 `n_data`，不再依赖上述公式。

### 2.3 可选字段

| 字段名 | 形状 | 类型 | 说明 |
|--------|------|------|------|
| `leakage` | `[N, T, n_stab]` | float32 | 泄漏概率。如不提供，`PTBatchDataset` 将自动填充为 0。 |
| `event_leakage` | `[N, T, n_stab]` | float32 | 泄漏事件变化。如不提供，`PTBatchDataset` 将自动填充为 0。 |
| `raw_measurements` | `[N, total_measurements]` | bool/int8/float32 | 原始测量记录。仅用于调试或特殊评估。 |
| `family` 或 `code_family` | 标量或 `[N]` | int/str | 码族标识符。多码族训练时使用。 |
| `seed` | 标量 | int | 生成该批次数据使用的随机种子，便于复现。 |

---

## 3. 字段语义关系（必须满足）

### 3.1 measurement 与 event 的关系

`event` 必须从 `measurement` 计算得到，二者必须一致：

- 二值情况：
  ```python
  event[:, 0, :] = measurement[:, 0, :]
  event[:, t, :] = measurement[:, t, :] XOR measurement[:, t-1, :]
  ```

- 软值情况：
  ```python
  event[:, 0, :] = measurement[:, 0, :]
  event[:, t, :] = soft_xor(measurement[:, t, :], measurement[:, t-1, :])
  ```

其中 `soft_xor(a, b) = a + b - 2*a*b` 是 XOR 的软概率版本，确保当 `a, b ∈ [0, 1]` 时输出也在 [0, 1] 且对应二值 XOR 的期望。

**要求**：`np.abs(recomputed_event - event).max()` 应小于 `1e-5`（浮点误差级别）。

### 3.2 detection_events 与 label 的关系

`detection_events` 和 `label` 必须来自**同一次底层错误采样**。

**为什么重要**：MWPM 使用 `detection_events` 解码得到预测，再与 `label` 对比计算 accuracy。如果二者不来自同一次采样，MWPM accuracy 将无意义。

**等价表述**：对于每个样本 `i`，存在一个底层错误配置 `E_i`，使得：
- `detection_events[i]` 是 `E_i` 在 syndrome 测量上的投影。
- `label[i]` 是 `E_i` 在 logical observable 上的投影。

### 3.3 final_soft 与 label 的关系

`final_soft` 是最终轮对 data qubit 的测量，用于 readout 阶段的 late fusion。它不应向模型泄露 `label` 本身，但自然与 `label` 相关。

**要求**：`final_soft` 必须与 `measurement`/`event` 来自同一次实验 shot。

---

## 4. 文件格式

### 4.1 推荐格式：`.pt`（PyTorch 张量字典）

```python
pt_data = {
    'measurement': torch.from_numpy(measurement.astype(np.float32)),
    'event': torch.from_numpy(event.astype(np.float32)),
    'final_soft': torch.from_numpy(final_soft.astype(np.float32)),
    'label': torch.from_numpy(label.astype(np.float32)),
    'detection_events': torch.from_numpy(detection_events.astype(np.float32)),
    'distance': int(distance),
    'rounds': int(rounds),
    'p': float(p),
    'snr': float(snr),
}
# 可选字段
pt_data['leakage'] = torch.zeros(N, T, n_stab, dtype=torch.float32)
pt_data['event_leakage'] = torch.zeros(N, T, n_stab, dtype=torch.float32)

torch.save(pt_data, 'train_d3_r25_n50000.pt')
```

### 4.2 兼容性格式：`.npz`（NumPy）

```python
np.savez(
    'train_d3_r25_n50000.npz',
    measurement=measurement.astype(np.float32),
    event=event.astype(np.float32),
    final_soft=final_soft.astype(np.float32),
    label=label.astype(np.float32),
    detection_events=detection_events.astype(np.float32),
    distance=np.array(distance),
    rounds=np.array(rounds),
    p=np.array(p),
    snr=np.array(snr),
)
```

需通过 `convert_npz_to_pt.py` 转换为 `.pt` 后使用。

### 4.3 文件名规范

```
{split}_d{distance}_r{rounds}_n{num_samples}.{ext}
```

- `{split}`：`train` / `val` / `test` / `ler`
- `{distance}`：码距离或等价标识符
- `{rounds}`：纠错轮数
- `{num_samples}`：样本总数
- `{ext}`：`pt` 或 `npz`

**示例**：
- `train_d3_r25_n50000.pt`
- `val_d5_r25_n10000.pt`
- `ler_d3_r15_n2000.pt`

---

## 5. 数据使用方假设（Data Consumer Assumptions）

当前训练/评估代码对输入数据做以下假设。数据提供方必须满足：

1. `measurement` 和 `event` 的第一维是样本数 `N`，第二维是时间 `T`，第三维是 stabilizer 数 `n_stab`。
2. `final_soft` 的第一维是 `N`，第二维是 data qubit 数 `n_data`。
3. `label` 的第一维是 `N`。对于二分类逻辑错误检测，形状为 `[N]`，每个元素为 0 或 1。
4. `distance` 和 `rounds` 是标量，且与 `measurement.shape[1:]` 一致。
5. `detection_events` 的每个样本可以 reshape 为 `[T, n_stab]` 或已经是 `[num_detectors]`。MWPM 评估时会被 flatten。
6. 缺失 `leakage` 和 `event_leakage` 时，代码会自动补零。

---

## 6. 数据加载接口

### 6.1 PTBatchDataset

```python
from alphaqubit.data.pt_dataset import PTBatchDataset

ds = PTBatchDataset('train_d3_r25_n50000.pt')
print(ds.distance, ds.rounds, ds.p, ds.snr)
print(len(ds))

sample = ds[0]
# sample['measurement']: [T, n_stab]
# sample['event']: [T, n_stab]
# sample['final_soft']: [n_data]
# sample['label']: [1]
# sample['leakage']: [T, n_stab] (zeros if not provided)
# sample['event_leakage']: [T, n_stab] (zeros if not provided)
```

### 6.2 字段访问约定

- `measurement`、`event`、`final_soft`、`label` 必须存在。
- `leakage` 和 `event_leakage` 可省略，由 `PTBatchDataset` 动态生成全零张量。
- `detection_events` 必须存在，用于 MWPM 评估。

---

## 7. 一致性检查（通用版）

数据生成完成后，建议执行以下检查。这些检查不依赖具体噪声模型或生成工具。

### 7.1 measurement-event 一致性检查

```python
recomputed_event = np.zeros_like(measurement)
recomputed_event[:, 0, :] = measurement[:, 0, :]
for t in range(1, T):
    recomputed_event[:, t, :] = measurement[:, t, :] + measurement[:, t-1, :] - 2 * measurement[:, t, :] * measurement[:, t-1, :]

diff = np.abs(recomputed_event - event).max()
assert diff < 1e-5, f"measurement 和 event 不一致，最大差异 {diff}"
```

### 7.2 MWPM 一致性检查（如果提供 detection_events）

```python
# dem 为对应码的 detector error model
mwpm = pymatching.Matching.from_detector_error_model(dem)

det_events = data['detection_events'].reshape(N, -1)
labels = data['label']
preds = mwpm.decode_batch(det_events).flatten()
acc = np.mean(preds == labels)
print(f'MWPM accuracy: {acc:.4f}')
```

**说明**：MWPM accuracy 本身不是数据形式要求，而是验证 `detection_events` 与 `label` 是否来自同一次采样的有效手段。如果 accuracy 显著偏离该码在该噪声下的预期值，应检查数据一致性。

### 7.3 形状一致性检查

```python
N, T, n_stab = measurement.shape
assert event.shape == (N, T, n_stab)
assert final_soft.shape[0] == N
assert label.shape[0] == N
assert detection_events.shape[0] == N
assert rounds == T
```

---

## 8. 多 rounds LER 评估数据

用于 LER 评估的数据集应满足：

1. 与训练/测试数据相同的 `distance` 和噪声参数（或目标评估参数）。
2. 每个 rounds 数 `n` 单独保存为一个文件。
3. 文件名：`ler_d{distance}_r{n}_n{num_samples}.pt`
4. 字段形状与训练数据一致，仅 `rounds=n`。
5. 同样包含 `detection_events`。

---

## 9. 后续模型数据迁移 Checklist

新数据生成模型/脚本只需确保以下形式要求：

- [ ] 文件格式为 `.pt`（推荐）或 `.npz`（需转换）。
- [ ] 字段名：`measurement`, `event`, `final_soft`, `label`, `detection_events`, `distance`, `rounds`, `p`, `snr`。
- [ ] 数据类型：张量为 `float32`；`distance`、`rounds` 为标量；`p`、`snr` 为标量浮点数。
- [ ] 形状：
  - `measurement`: `[N, T, n_stab]`
  - `event`: `[N, T, n_stab]`
  - `final_soft`: `[N, n_data]`
  - `label`: `[N]` 或 `[N, n_logical]`
  - `detection_events`: `[N, T, n_stab]` 或 `[N, num_detectors]`
- [ ] `event` 可由 `measurement` 通过 soft XOR 恢复。
- [ ] `detection_events` 与 `label` 来自同一次底层错误采样。
- [ ] `measurement`、`event`、`final_soft`、`label` 来自同一次实验 shot。
- [ ] 通过 MWPM 一致性检查（如使用该基准）。
- [ ] 文件名符合 `{split}_d{distance}_r{rounds}_n{num_samples}.{ext}` 规范。

**本规范不强制也不假设**：
- 使用 Stim 生成电路或构造 DEM。
- 使用任何特定噪声模型（SI1000、去极化、PAEMS、泄漏模型等）。
- `p` 必须等于电路中的某个具体错误概率；`p` 仅为元数据，可记录等效物理错误率、平均错误率或任意噪声强度指标。
- `snr` 必须来自特定软读出模拟；`snr` 仅为元数据，可记录任意读出质量指标。
- 任何特定的软读出模拟方法、门集合或电路结构。

---

## 10. 参考实现（示例）

以下实现仅作为示例，展示如何生成满足本规范的数据。新模型可采用完全不同的生成方式。

- `scripts/generate_dataset.py`：基于 Stim 批量生成 `.npz` 数据。
- `scripts/convert_npz_to_pt.py`：转换为 `.pt`。
- `alphaqubit/data/pt_dataset.py`：`PTBatchDataset` 加载接口。

---

## 11. 附录：字段详细形状表

### 11.1 d=3 时

```python
n_stab = 3**2 - 1 = 8
n_data = 3**2 = 9
T = rounds
```

批量数据形状（N=50000, T=25）：
- `measurement`: `[50000, 25, 8]`
- `event`: `[50000, 25, 8]`
- `final_soft`: `[50000, 9]`
- `label`: `[50000]`
- `detection_events`: `[50000, 25, 8]` 或 `[50000, 200]`

### 11.2 d=5 时

```python
n_stab = 5**2 - 1 = 24
n_data = 5**2 = 25
```

批量数据形状（N=50000, T=25）：
- `measurement`: `[50000, 25, 24]`
- `event`: `[50000, 25, 24]`
- `final_soft`: `[50000, 25]`
- `label`: `[50000]`
- `detection_events`: `[50000, 25, 24]` 或 `[50000, 600]`

---

## 12. 扩展性说明

本规范的形式设计允许以下扩展：

1. **任意噪声模型**：只要输出字段满足形状和语义关系即可。
2. **任意量子纠错码**：只要 stabilizer 数量、data qubit 数量、`distance` 等元数据一致即可。
3. **多逻辑 qubit**：`label` 可扩展为 `[N, n_logical]`。
4. **多码族联合训练**：可增加 `family` 字段标识码族。
5. **不同 soft readout 范围**：`measurement` 不限于 [0, 1]，但需在训练前做归一化或调整模型输入层。
