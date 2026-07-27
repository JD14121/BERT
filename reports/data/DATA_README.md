# PAEMS 合成噪声数据说明报告（DATA README）

> 本文件是 `PAEMS-data/` 下合成数据的**说明报告**，描述数据的来源模型、组织方式、字段含义、生成流水线、质量指标与使用方法。
> 数据严格遵循 [`deliverables/data_specification/synthetic_data_spec.md`](../deliverables/data_specification/synthetic_data_spec.md)（v2.0）数据形式契约。

---

## 1. 数据概述

本数据集是使用 **PAEMS**（Precise and Adaptive Error Model for superconducting Quantum Processors，精确自适应超导量子处理器错误模型）生成的**表面码（Surface Code）量子纠错解码器训练数据**。PAEMS 是一个电路级随机错误模型，其特点是：

- **逐量子比特（per-qubit）异质参数化** —— 每个量子比特的 T1/T2/门保真度/泄漏率/串扰各不相同；
- **非对称退极化通道 ADC** —— 由 T1/T2 推导的 Pauli 概率满足 `P_X = P_Y ≠ P_Z`；
- **对称退极化通道 SDC** —— 由门保真度推导的去极化噪声；
- **SPAM 错误** —— 状态准备错误 `P_init` 与测量错误 `P_meas` 分离且不相等；
- **泄漏与回渗** —— `|1⟩→|2⟩` 泄漏（LP）+ `|2⟩→|1⟩` 回渗（SP），并在双比特门之间传播；
- **旁观者串扰** —— 由 `crosstalk_pairs` 指定的量子比对之间的去极化事件。

| 项 | 值 |
|---|---|
| 文件总数 | **22 个 `.pt` 文件** |
| 总体积 | **≈ 1.21 GB** |
| 码族 | 旋转表面码（rotated surface code），Z-basis memory 实验 |
| 码距离 | d = 3 与 d = 5 |
| 物理错误率 p_eff | d3 ≈ 0.12% · d5 ≈ 0.20%（2Q 去极化均值，远低于表面码阈值 ~1%） |
| 软读出 SNR / t | SNR = 10.0 · t = 0.01 |
| 文件格式 | PyTorch 张量字典 `.pt`（`torch.float32`） |
| 命名规范 | `{split}_d{distance}_r{rounds}_n{N}.pt` |

---

## 2. 目录结构

```
PAEMS-data/
├── DATA_README.md          # ← 本说明报告
├── README.md               # 项目说明（模型/复现指令）
├── QA_REPORT.md            # 自动生成的逐文件 QA 报告
├── code/                   # 生成器 + 校验器源码
│   ├── paems_noise_model.py
│   ├── generate_paems_data.py
│   └── validate_paems_data.py
├── params/                 # 共享的逐距离 PAEMS 校准参数（审计/复现用）
│   ├── paems_params_d3.json
│   └── paems_params_d5.json
├── train_d3_r25_n50000.pt   train_d5_r25_n50000.pt
├── val_d3_r25_n10000.pt     val_d5_r25_n10000.pt
├── test_d3_r25_n10000.pt    test_d5_r25_n10000.pt
├── ler_d3_r{3,6,9,12,15,18,21,25}_n2000.pt   （8 个）
└── ler_d5_r{3,6,9,12,15,18,21,25}_n2000.pt   （8 个）
```

> `params/paems_params_d{3,5}.json` 与所有同距离的 `train/val/test/ler` 文件**共享同一套设备校准**，即 train/val/test/LER 看到的是同一台“虚拟超导芯片”的不同实验，保证各 split 之间可比、可对齐 LER 曲线。

---

## 3. 数据清单（22 个文件）

### 3.1 训练 / 验证 / 测试集（6 个，统一 rounds=25）

| 文件名 | 码距 d | 轮次 T | 样本数 N | 体积 |
|---|---|---|---|---|
| `train_d3_r25_n50000.pt` | 3 | 25 | 50 000 | 202.0 MB |
| `train_d5_r25_n50000.pt` | 5 | 25 | 50 000 | 605.2 MB |
| `val_d3_r25_n10000.pt`   | 3 | 25 | 10 000 | 40.4 MB |
| `val_d5_r25_n10000.pt`   | 5 | 25 | 10 000 | 121.0 MB |
| `test_d3_r25_n10000.pt`  | 3 | 25 | 10 000 | 40.4 MB |
| `test_d5_r25_n10000.pt`  | 5 | 25 | 10 000 | 121.0 MB |

### 3.2 LER 扫描集（16 个，每码距 8 个轮次点，N=2000/文件）

用于绘制“逻辑错误率随纠错轮数 T”的 LER 曲线。轮次序列：`{3, 6, 9, 12, 15, 18, 21, 25}`。

| 码距 d | 检测器数随轮次（r=3→25） |
|---|---|
| d = 3 | 24 → 48 → 72 → 96 → 120 → 144 → 168 → 200 |
| d = 5 | 72 → 144 → 216 → 288 → 360 → 432 → 504 → 600 |

---

## 4. 字段定义与语义（spec §2.1 / §4.1）

每个 `.pt` 文件是一个 Python 字典，键与形状如下（`N`=样本数，`T`=轮数=rounds，`n_stab = d²−1`=稳定子数，`n_data = d²`=数据比特数）：

| 字段 | 形状 | 类型 | 语义 |
|---|---|---|---|
| `measurement` | `[N, T, n_stab]` | float32 | 每轮每个稳定子的**软**测量结果 P(1)。值域≈[0,1]；0=确定测得 0，1=确定测得 1。 |
| `event` | `[N, T, n_stab]` | float32 | 每轮每个稳定子的**软**检测事件 = soft-XOR(meas[t], meas[t-1])，首轮 = meas[0]。 |
| `final_soft` | `[N, n_data]` | float32 | 最后一轮对数据比特的软测量结果。 |
| `label` | `[N]` | float32 | 逻辑 observable 标签（0/1），即该次实验的逻辑错误翻转比特。 |
| `detection_events` | `[N, num_detectors]` | float32 | 按 DEM（detector error model）顺序排列的**硬**检测事件，供 MWPM 解码。 |
| `leakage` | `[N, T, n_stab]` | float32 | 每轮每个稳定子的泄漏受影响标记（faithful 泄漏建模；非零）。 |
| `event_leakage` | `[N, T, n_stab]` | float32 | leakage 的轮间 soft-XOR（泄漏事件变化）。 |
| `distance` | 标量 | int | 码距离（3 或 5）。 |
| `rounds` | 标量 | int | 纠错轮数 T。 |
| `p` | 标量 | float | 等效物理错误率元数据（d3≈0.00123，d5≈0.00195），仅记录不约束噪声模型。 |
| `snr` | 标量 | float | 软读出信噪比（10.0）。 |
| `_meta` | dict | — | seed、t、p_eff、include_leakage、num_detectors 等审计信息。 |

### 4.1 关键语义关系（必须满足，已校验通过）

- **measurement ↔ event（spec §3.1/§7.1）**：`event` 由 `measurement` 经 soft-XOR `a + b − 2ab` 重建，二者最大差 < `1e-5`（实测 = `0.0`）。
- **detection_events ↔ label（spec §3.2/§7.2）**：二者来自**同一次**底层错误采样。本文用 stimulated 采样保证：对含噪电路分别调用 measurement-sampler 与 detector-sampler，并使用**相同随机种子**，使每个 shot 的检测事件与 observable 标签严格对齐。校验方式：MWPM 同次序解码准确率 0.905–0.994（随机错配 ≈ 0.5）。
- **measurement/event/final_soft 同 shot（spec §3.3）**：三者均由同一 stim shot 的原始测量记录经软读出得到，天然一致。

> **泄漏处理方式（官方 PAEMS Option A）**：`detection_events` 与 `label` 来自 matched-seed 的 stim 采样（不含泄漏）；`measurement`/`event`/`final_soft` 在原始测量记录上叠加泄漏后处理翻转（50/50），再经软读出。这与官方 `run_sampling.py` 一致，同时满足规范全部条款。

---

## 5. PAEMS 噪声模型与生成流水线

### 5.1 噪声通道（注入到标准 stim 表面码电路，保留 DETECTOR/OBSERVABLE_INCLUDE）

| 机制 | Stim 通道 | 参数来源 |
|---|---|---|
| ADC（非对称退相干） | `PAULI_CHANNEL_1([P_X, P_Y, P_Z])`，`P_X=P_Y=(1−e^{−t/T₁})/4`，`P_Z=(1−e^{−t/T₂})/2 − P_X` | T1, T2, 门时长 t |
| SDC（门误差，对称去极化） | `DEPOLARIZE1(p1)` / `DEPOLARIZE2(p2)`，`p = d(F_E−F)/(d·F_E−1)` | 门保真度 F |
| SPAM（`P_init ≠ P_meas`） | `X_ERROR`（初始化 / 测量 / 重置前） | 逐比特 init/meas 错误率 |
| 泄漏 + 回渗 | 后处理：LP 让 `|1⟩→|2⟩`，SP 让 `|2⟩→|1⟩`，CX 按 `lp_propagation_prob` 传播；受影响测量 50/50 翻转 | LP, SP, 传播概率 |
| 串扰 | `DEPOLARIZE1`（旁观者） | `crosstalk_pairs` 强度 |

2Q 门总保真度遵循 PAEMS 公式：`F_total^CX = F_sqg,ctrl² · F_sqg,tgt² · F_CX`。

### 5.2 逐距离硬件参数（共享校准，实测）

| 参数 | d = 3 | d = 5 |
|---|---|---|
| 量子比特总数 | 17（数据 9 + ancilla 8） | 49（数据 25 + ancilla 24） |
| CX 门对数 | 24 | 80 |
| 串扰对数 | 3 | 5 |
| T1 范围 | 102–585 μs（均值 330 μs） | 89–593 μs（均值 334 μs） |
| T2/T1 均值 | 1.10 | 1.21 |
| 单比特门保真度 | 0.99910–0.99989 | 0.99901–0.99989 |
| 2Q 门保真度 | 0.99722–0.99887 | 0.99702–0.99898 |

> 这些值均落在 PAEMS / CMA-ES 文档给出的物理边界内（T1∈[10,600]μs、T2≤2T1、2Q 保真度 0.9–0.9999 等）。数据生成**无需** CMA-ES（其用途是用真实芯片数据校准参数，属于部署而非合成数据生成）。

### 5.3 生成流水线（每文件）

```
逐距离 PAEMS 参数生成（确定性种子） ──▶ params/paems_params_d{d}.json
        │
标准 stim 旋转表面码电路（noiseless） ──▶ 注入 PAEMS 噪声通道（保留 DETECTOR/OBSERVABLE）
        │
含噪电路 ──▶ 同种子 measurement-sampler + detector-sampler 采样
        │            ├─ measurement record [N, T·n_stab + n_data]
        │            ├─ detection_events [N, num_det]（DEM 序）
        │            └─ label [N]
        │
泄漏状态机（向量化、分批） ──▶ affected 掩码 ──▶ 50/50 翻转 measurement record
        │
拆分 + 软读出（SNR=10, t=0.01） ──▶ measurement/event/final_soft（float32）
        │
拼装字典 ──▶ torch.save(.pt)
```

---

## 6. 质量指标（QA，全部通过）

`validate_paems_data.py` 按 spec §3–§7 + PAEMS 物理保真度逐项校验，**22 个文件全部 PASS，退出码 0**。完整逐文件表格见 [`QA_REPORT.md`](QA_REPORT.md)。

### 6.1 代表性指标

| 指标 | 取值范围 | 说明 |
|---|---|---|
| 同次序 MWPM 准确率 | **0.905 – 0.994** | ≫0.55 ⟹ detection_events 与 label 来自同一 shot（spec §3.2 通过） |
| measurement↔event soft-XOR 最大差 | **0.0** | <1e-5 ⟹ 一致性通过（spec §7.1） |
| leakage↔event_leakage 一致性 | **0.0** | <1e-5 |
| 检测事件密度 | 0.0448 – 0.0633 | 落在物理合理的 QEC 区间（非全 0/全 1） |
| 数据类型 | 全 float32 | 满足 spec §9 |
| 命名规范 | 22/22 合法 | 满足 spec §4.3 |

### 6.2 LER 曲线（逻辑错误率随轮数上升，物理合理）

| 轮次 r | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 25 |
|---|---|---|---|---|---|---|---|---|
| d=3 label 率 | 0.062 | 0.106 | 0.146 | 0.184 | 0.208 | 0.243 | 0.276 | 0.298 |
| d=5 label 率 | 0.097 | 0.181 | 0.243 | 0.281 | 0.333 | 0.352 | 0.377 | 0.406 |

> label 率随轮数单调上升，符合“长记忆实验中逻辑错误累积”的物理直觉，是为 LER 评估所设计的数据形态。

### 6.3 PAEMS 模型保真度（确认非均匀噪声）

| 码距 | P_X（均值） | P_Z（均值） | 非对称 P_X≠P_Z | P_init | P_meas | P_init≠P_meas |
|---|---|---|---|---|---|---|
| d = 3 | 6.26e-5 | 5.96e-5 | ✅ yes | 0.00153 | 0.00929 | ✅ yes |
| d = 5 | 5.76e-5 | 4.78e-5 | ✅ yes | 0.00152 | 0.00920 | ✅ yes |

两处非对称性（ADC 的 `P_X≠P_Z` 与 SPAM 的 `P_init≠P_meas`）均成立，证明数据**确实来自 PAEMS 而非均匀 SI1000/去极化模型**。

---

## 7. 如何使用这些数据

### 7.1 通过既有训练接口加载（推荐）

数据可直接喂给既有消费者 `PTBatchDataset`（无需任何额外适配）：

```python
import sys; sys.path.insert(0, '.')  # alphaqubit 包根目录
from alphaqubit.data.pt_dataset import PTBatchDataset

ds = PTBatchDataset('PAEMS-data/train_d5_r25_n50000.pt')
print(ds.distance, ds.rounds, ds.p, ds.snr, len(ds))   # 5 25 0.00195 10.0 50000

batch = ds.get_batch(64)                                 # 随机批
# batch['measurement'] : [64, 25, 24]   batch['label'] : [64, 1]
# batch['final_soft']  : [64, 25]       batch['leakage'] : [64, 25, 24]

de = ds.get_detection_events(0)                          # [num_detectors] 供 MWPM
```

### 7.2 直接 torch.load

```python
import torch
d = torch.load('PAEMS-data/ler_d3_r15_n2000.pt', map_location='cpu', weights_only=False)
# d['measurement'], d['event'], d['final_soft'], d['label'],
# d['detection_events'], d['leakage'], d['event_leakage'],
# d['distance'], d['rounds'], d['p'], d['snr'], d['_meta']
```

### 7.3 MWPM 基线评估（spec §7.2）

```python
import pymatching, stim  # 需 conda base 环境
import sys; sys.path.insert(0, 'PAEMS-data/code')
import paems_noise_model as pnm
import torch

d = torch.load('PAEMS-data/test_d5_r25_n10000.pt', map_location='cpu', weights_only=False)
params = pnm.generate_paems_params(5, seed=5*7919+42)                    # 与生成同种子
noisy = pnm.build_paems_noisy_circuit(pnm._base_surface_code_circuit(5, 25), params, 25)
dem = noisy.detector_error_model()
mwpm = pymatching.Matching.from_detector_error_model(dem)
preds = mwpm.decode_batch((d['detection_events'].numpy() > 0.5).astype('uint8')).flatten()
acc = (preds == d['label'].numpy().astype(int)).mean()
print('MWPM accuracy:', acc)   # ≈ 0.958
```

---

## 8. 复现与环境

- **执行环境**：conda base 解释器 `D:/anaconda/python.exe`（Python 3.12）。依赖：stim 1.16、numpy 1.26、scipy 1.13、torch 2.6、pymatching 2.4（均已就位；`cma` 不需要）。
- **复现命令**：

```bash
# 重新生成全套 22 个文件（确定性，可逐字节复现）
D:/anaconda/python.exe PAEMS-data/code/generate_paems_data.py --manifest --chunk-size 5000

# 运行校验
D:/anaconda/python.exe PAEMS-data/code/validate_paems_data.py      # 退出码 0 = 全通过
```

- **确定性**：采样种子由 `(split, distance, rounds)` 推导；逐距离参数种子为 `d*7919+42`。因此重新生成同一配置会得到字节级一致的 `.pt` 文件。

---

## 9. 局限与说明

1. **参数为合成而非真实校准**：逐比特参数在物理合理区间内随机采样（未经 CMA-ES 对真实芯片数据校准）。这足以产生高质量、保真 PAEMS 特征的合成训练数据；若需匹配某真实平台分布，可替换 `params/paems_params_d{d}.json` 后重跑生成。
2. **泄漏仅在测量记录生效（Option A）**：`detection_events`/`label` 不含泄漏，`measurement`/`event`/`final_soft` 含泄漏——与官方 PAEMS 及数据规范一致。
3. **码距离与码族**：当前为 d3、d5 旋转表面码 Z-memory。扩展到更大距离或其他码族（XZZX/color code）只需在 `paems_noise_model.py` 替换基础电路，输出字段不变。
4. **逻辑错误率随轮数累积**：d3 在 25 轮时 label 率可达 ~30%，d5 ~41%，这是长记忆实验中 d 较小时逻辑错误累积的真实物理结果（非数据缺陷）。

---

*生成时间：2026-07-10 · 环境验证：conda base · 数据规范版本：v2.0 · 校验结果：22/22 PASS*
