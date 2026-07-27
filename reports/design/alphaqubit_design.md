# AlphaQubit 复现设计文档

## 1. 项目概述

### 1.1 目标
复现 Google DeepMind 的 AlphaQubit 神经网络解码器，用于 Surface Code 量子纠错。

### 1.2 范围
- 使用 **Stim** 库生成模拟数据（非真机数据）
- 使用 **PyTorch** 重新实现
- 支持 code distance 3, 5, 7, 9, 11
- 支持 soft readout 模拟

### 1.3 参考论文
- Bausch et al., "Learning high-accuracy error decoding for quantum processors", Nature 2024

---

## 2. 模型架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AlphaQubit 整体架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │  Stim 模拟数据   │                                                       │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ SoftReadout模拟  │  ← 可选：将binary转为soft概率值                        │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                    双轨制输入处理                             │          │
│   │  ┌─────────────────────┐    ┌─────────────────────────┐    │          │
│   │  │  通道A: Syndrome     │    │  通道B: Final Data       │    │          │
│   │  │  (中间轮次stabilizer)│    │  (最终轮次data qubit)    │    │          │
│   │  │  SyndromeEmbedder   │    │  FinalDataEmbedder      │    │          │
│   │  └──────────┬──────────┘    └───────────┬─────────────┘    │          │
│   └─────────────┼───────────────────────────┼──────────────────┘          │
│                 │                           │                              │
│                 ▼                           │                              │
│   ┌─────────────────────────┐               │                              │
│   │       RNN Core          │               │                              │
│   │  (循环处理每个时间步)     │               │                              │
│   │  ┌───────────────────┐  │               │                              │
│   │  │ SyndromeTransformer│ │               │                              │
│   │  │      × 3 层        │  │               │                              │
│   │  └───────────────────┘  │               │                              │
│   └───────────┬─────────────┘               │                              │
│               │                             │                              │
│               ▼                             ▼                              │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │                  Readout Network                         │              │
│   │  Scatter → Conv2D → Pool → ResNet(16层) → Linear → σ    │              │
│   └────────────────────────┬────────────────────────────────┘              │
│                            │                                               │
│                            ▼                                               │
│                     P(Logical Error)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 输入数据设计

### 3.1 数据来源
- 使用 Stim 库生成 SI1000 噪声模型的 Surface Code 电路
- Stim 输出 binary 测量结果，可选择性添加 soft readout 模拟

### 3.2 双轨制输入

#### 通道A: 中间轮次 Syndrome（离散/Soft）

| 输入字段 | 形状 | 类型 | 描述 |
|---------|------|------|------|
| `measurement` | (batch, rounds, n_stab) | float | Stabilizer测量值 (0/1 或 soft概率) |
| `event` | (batch, rounds, n_stab) | float | 检测事件 = meas_n ⊕ meas_{n-1} |
| `leakage` | (batch, rounds, n_stab) | float | 泄漏概率 (模拟中设为0) |
| `event_leakage` | (batch, rounds, n_stab) | float | 泄漏事件变化 (模拟中设为0) |
| `position_idx` | (n_stab,) | int | 每个stabilizer在2D网格中的位置索引 |

#### 通道B: 最终轮次 Data Qubit（连续/Soft）

| 输入字段 | 形状 | 类型 | 描述 |
|---------|------|------|------|
| `data_meas` | (batch, n_data) | float | Data qubit测量值 (soft readout) |
| `position_idx` | (n_data,) | int | 每个data qubit在2D网格中的位置索引 |

### 3.3 位置编码

Stabilizer 和 Data Qubit 共享同一个2D网格坐标系：

```
d=3 的布局示例 (4×4 网格):

    col  0   1   2   3
row  ┌───┬───┬───┬───┐
 0   │ S │ D │ S │ D │    S = Stabilizer (n_stab = d² - 1 = 8)
     ├───┼───┼───┼───┤    D = Data Qubit (n_data = d² = 9)
 1   │ D │ S │ D │ S │    
     ├───┼───┼───┼───┤    网格尺寸: (d+1) × (d+1)
 2   │ S │ D │ S │ D │    位置索引: pos_idx = row * (d+1) + col
     ├───┼───┼───┼───┤
 3   │ D │ S │ D │ S │
     └───┴───┴───┴───┘
```

---

## 4. 模块详细设计

### 4.1 Soft Readout 模拟器

**目的**: 将 Stim 输出的 binary 测量值转换为模拟真机 I/Q 信号的 soft 概率值

**参数**:
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `snr` | 10.0 | Signal-to-Noise Ratio |
| `t` | 0.01 | 归一化测量时间 t = t_meas / T1 (影响amplitude damping) |

**算法**:
```
1. 采样 I/Q 信号值 z:
   - |0⟩ 状态: z ~ N(0, 1/(2*SNR))
   - |1⟩ 状态: z ~ N(1 - t/2, 1/(2*SNR))  # 考虑amplitude damping

2. 计算后验概率:
   P(meas=1 | z) = P(z|1) / (P(z|0) + P(z|1))
```

**Soft Event 计算**:
```
soft_event = p_n * (1 - p_{n-1}) + (1 - p_n) * p_{n-1}
```

---

### 4.2 SyndromeEmbedder (通道A)

**目的**: 将中间轮次的 stabilizer 测量编码为 d_model 维向量

**结构**:
```
measurement ──→ Linear(1, d_model) ──┐
event       ──→ Linear(1, d_model) ──┼──→ (+) ──→ ResNetBlock ──→ S_n
leakage     ──→ Linear(1, d_model) ──┤
event_leak  ──→ Linear(1, d_model) ──┤
position    ──→ Embedding(n_pos, d_model)─┘
```

**输入**: 
- `measurement, event, leakage, event_leakage`: (batch, n_stab)
- `position_idx`: (n_stab,)

**输出**: (batch, n_stab, d_model)

---

### 4.3 FinalDataEmbedder (通道B)

**目的**: 将最终轮次的 data qubit soft readout 编码为 d_model 维向量

**结构**:
```
soft_meas ──→ MLP(1 → hidden → d_model) ──┐
                                          ├──→ (+) ──→ ResNetBlock ──→ 输出
position  ──→ Embedding(n_pos, d_model) ──┘
```

**MLP 结构**: Linear(1, 64) → ReLU → Linear(64, d_model)

**输入**:
- `data_meas`: (batch, n_data) soft readout 值
- `position_idx`: (n_data,)

**输出**: (batch, n_data, d_model)

---

### 4.4 ResNetBlock

**目的**: 特征混合和非线性变换

**结构** (Pre-LN, 等宽):
```
Input (x)
    │
    ├────────────────┐ (Skip Connection)
    ▼                │
LayerNorm(d_model)   │
    │                │
    ▼                │
Linear(d_model → d_model)
    │                │
    ▼                │
ReLU                 │
    │                │
    ▼                │
Dropout              │
    │                │
    ▼                │
Linear(d_model → d_model)  ← 零初始化
    │                │
    ▼                │
Dropout              │
    │                │
    ▼                │
   (+) ◄─────────────┘
    │
    ▼
Output (y)
```

**公式**: `y = x + Linear2(ReLU(Linear1(LayerNorm(x))))`

**设计要点**:
- 使用 Pre-LN 而非 Post-LN
- 等宽设计 (d_model → d_model)，不使用瓶颈结构
- Linear2 零初始化，使残差分支初始输出 ≈ 0

---

### 4.5 RNN Core

**目的**: 循环处理每个时间步的 syndrome，维护 decoder state

**结构**:
```
每个时间步 t:

    Decoder State_{t-1}     S_t (Syndrome Embedding)
          │                        │
          └──────────┬─────────────┘
                     │
                    (+)
                     │
                    (× 0.7)  ← 缩放因子 ≈ 1/√2
                     │
                     ▼
           ┌─────────────────┐
           │SyndromeTransformer│ × 3层 (串行)
           └────────┬────────┘
                    │
                    ▼
            Decoder State_t
```

**初始状态**: `h_0 = zeros(batch, n_stab, d_model)`

**缩放因子 0.7 ≈ 1/√2 的原因**:
- 假设 state 和 embed 都是零均值单位方差
- 相加后方差变为 2，标准差变为 √2
- 乘以 1/√2 使方差回到 1，保持信号幅度稳定

---

### 4.6 SyndromeTransformer

**目的**: 核心特征处理模块，捕捉全局和局部错误关联

**结构** (串行，每个子层都有残差连接和 Pre-LN):
```
Input
  │
  ├───────────────┐ (残差)
  ▼               │
Self-Attention    │  ← 捕捉长程依赖
(+ Attention Bias)│  ← 可选：基于空间距离的偏置
  │               │
 (+) ◄────────────┘
  │
  ├───────────────┐ (残差)
  ▼               │
Gated Dense Block │  ← GLU风格，筛选特征
  │               │
 (+) ◄────────────┘
  │
  ├───────────────┐ (残差)
  ▼               │
Scatter to 2D     │
  ▼               │
Dilated Convs ×3  │  ← 捕捉局部空间特征
  ▼               │
Gather from 2D    │
  │               │
 (+) ◄────────────┘
  │
  ▼
Output
```

#### 4.6.1 Self-Attention with Bias

**公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d + B) V
```

其中 B 是可学习的 (n_heads, n_stab, n_stab) 偏置矩阵，基于 stabilizer 间的空间关系。

#### 4.6.2 Gated Dense Block (GeGLU)

**公式**:
```
Output = (x @ W1) ⊙ GELU(x @ W_gate) @ W2
```

- 门控机制可以过滤噪声信号，放大关键错误特征
- expansion_factor = 4

#### 4.6.3 Dilated 2D Convolutions

**流程**:
1. **Scatter**: 将 (batch, n_stab, d_model) 排列到 (batch, d_model, H, W) 网格
2. **Dilated Conv2D**: 3层 3×3 卷积，dilation 根据层级变化 [1, 1, 2] 或 [1, 2, 4]
3. **Gather**: 从 2D 网格收集回 1D 序列

**作用**:
- 平移不变性，符合 Surface Code 的对称性
- 空洞卷积扩大感受野而不增加参数
- 捕捉局部几何纹理

---

### 4.7 Readout Network

**目的**: 将最终 decoder state 转换为逻辑错误概率

**结构**:
```
Decoder State (batch, n_stab, d_model)
        │
        ▼
┌───────────────┐
│ Scatter to 2D │
└───────┬───────┘
        │
        ▼
(batch, d_model, H_stab, W_stab)
        │
┌───────────────┐
│ Conv to Data  │  ← 2×2 Conv, stride=1
│ (Stab→Data)   │     将相邻4个stabilizer融合为1个data qubit
└───────┬───────┘
        │
        ▼
(batch, d_model, H_data, W_data)  where H_data = d
        │
┌───────────────┐
│   Project     │  ← Linear(d_model → d_readout)
└───────┬───────┘
        │
        ▼
(batch, H, W, d_readout)
        │
┌───────────────────────────────────────┐
│           Line Mean Pool              │
│  (沿 logical observable 方向池化)      │
│  论文发现垂直方向效果更好              │
└───────────────┬───────────────────────┘
                │
                ▼
        (batch, d_readout)
                │
         ┌──────┴──────┐
         │             │
         ▼             ▼
┌─────────────┐  ┌───────────┐
│ Mean Pooled │  │Cycle Embed│  ← 轮次编码
│   Feature   │  │  (n轮)    │
└──────┬──────┘  └─────┬─────┘
       │               │
       └───────┬───────┘
               │ (+)
               ▼
┌─────────────────────────┐
│     Deep ResNet         │  ← 16层 (Sycamore) / 4层 (Scaling)
│   (深度特征整合)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Linear(d_readout, 1)   │
│         ↓               │
│      Sigmoid            │
└───────────┬─────────────┘
            │
            ▼
      P(Logical Error)
```

---

## 5. 超参数配置

### 5.1 模型参数 (论文默认值)

| 参数 | Sycamore 实验 | Scaling 实验 | 描述 |
|------|---------------|--------------|------|
| `d_model` | 320 | 256 | 特征维度 |
| `n_heads` | 4 | 4 | 注意力头数 |
| `n_transformer_layers` | 3 | 3 | SyndromeTransformer 层数 |
| `n_readout_layers` | 16 | 4 | Readout ResNet 层数 |
| `d_readout` | 64 | 32 | Readout 特征维度 |
| `dropout` | 0.0 | 0.0 | Dropout 概率 |
| `conv_dilations` | [1, 1, 1] (d=3) | [1, 2, 4] | 各层卷积 dilation |
|  | [1, 1, 2] (d=5) | | |

### 5.2 Soft Readout 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `snr` | 10.0 | Signal-to-Noise Ratio |
| `t` | 0.01 | 归一化测量时间 |

### 5.3 根据 Code Distance 变化的参数

| Code Distance | n_stabilizers | n_data_qubits | grid_size | conv_dilations |
|---------------|---------------|---------------|-----------|----------------|
| 3 | 8 | 9 | 4×4 | [1, 1, 1] |
| 5 | 24 | 25 | 6×6 | [1, 1, 2] |
| 7 | 48 | 49 | 8×8 | [1, 2, 4] |
| 9 | 80 | 81 | 10×10 | [1, 2, 4] |
| 11 | 120 | 121 | 12×12 | [1, 2, 4] |

---

## 6. 数据流详解

### 6.1 完整前向传播流程

```python
def forward(syndrome_rounds, final_data_meas, n_rounds):
    """
    Args:
        syndrome_rounds: (batch, n_rounds, n_stab) 中间轮次测量
        final_data_meas: (batch, n_data) 最终轮次 data qubit 测量
        n_rounds: 轮次数
    
    Returns:
        prob: (batch,) 逻辑错误概率
    """
    
    # 1. 可选: Soft Readout 模拟
    if use_soft_readout:
        soft_syndrome = soft_simulator(syndrome_rounds)
        soft_event = compute_soft_event(soft_syndrome)
        soft_final = soft_simulator(final_data_meas)
    
    # 2. 初始化 decoder state
    state = zeros(batch, n_stab, d_model)
    
    # 3. 逐轮处理 syndrome
    for t in range(n_rounds):
        # Embedding
        s_t = syndrome_embedder(
            meas=soft_syndrome[:, t],
            event=soft_event[:, t],
            leak=zeros,
            event_leak=zeros,
            pos_idx=stab_pos_idx
        )
        
        # RNN Core 更新
        state = (state + s_t) * 0.7
        for transformer in syndrome_transformers:
            state = transformer(state, scatter_indices)
    
    # 4. 处理最终轮次
    final_embed = final_data_embedder(soft_final, data_pos_idx)
    # 将 final_embed 融合到 state (具体方式需确认)
    
    # 5. Readout
    prob = readout_network(state, scatter_indices, n_rounds)
    
    return prob
```

### 6.2 张量形状变化

```
输入:
  syndrome_rounds: (B, R, N_stab)     e.g., (32, 25, 24) for d=5
  final_data_meas: (B, N_data)        e.g., (32, 25) for d=5

Embedding 后:
  syndrome_embed: (B, R, N_stab, D)   e.g., (32, 25, 24, 256)
  final_embed: (B, N_data, D)         e.g., (32, 25, 256)

RNN Core 处理:
  decoder_state: (B, N_stab, D)       e.g., (32, 24, 256)

Readout:
  scatter_2d: (B, D, H, W)            e.g., (32, 256, 6, 6)
  conv_to_data: (B, D, H-1, W-1)      e.g., (32, 256, 5, 5)
  pooled: (B, D_readout)              e.g., (32, 64)
  output: (B,)                        e.g., (32,)
```

---

## 7. 训练配置

### 7.1 损失函数

**Binary Cross Entropy**:
```python
loss = F.binary_cross_entropy(pred_prob, target_label)
```

**可选: 多 Logical Observable Loss**:
```python
# 对所有 d 个等价 logical observable 计算 loss 并平均
loss = mean([BCE(pred[i], label) for i in range(d)])
```

### 7.2 优化器

| 阶段 | 优化器 | 学习率 | Weight Decay |
|------|--------|--------|--------------|
| Pretraining (Sycamore) | Lamb | 3.46e-4 | 1e-5 |
| Finetuning (Sycamore) | Lamb | 3.46e-4 | 0.08 (相对于预训练参数) |
| Scaling | Lion | varies by distance | 1e-7 |

### 7.3 训练策略

1. **两阶段训练**:
   - Pretraining: 在 SI1000 合成数据上训练 (数十亿样本)
   - Finetuning: 在目标数据上微调 (有限样本)

2. **Noise Curriculum**: 从低噪声逐渐过渡到高噪声

3. **Rounds Curriculum**: 从少轮次逐渐过渡到多轮次

4. **Intermediate Labels**: 利用模拟数据的中间状态作为辅助监督

---

## 8. 评估指标

### 8.1 Logical Error Rate (LER)

**定义**: 每轮的逻辑错误率

**计算**:
```python
# 对于 n 轮实验，逻辑错误概率 E(n) 满足:
# E(n) = 0.5 * (1 - (1 - 2ε)^n)
# 其中 ε 是 LER

# 从实验结果拟合 LER:
fidelity = 1 - 2 * error_rate  # 每轮的 fidelity
log_fidelity = log(fidelity)
# 线性拟合: log(F(n)) = log(F_0) + n * log(1 - 2ε)
# 从斜率得到 LER
```

### 8.2 Error Suppression Factor (Λ)

**定义**: 增加 code distance 带来的错误抑制比

```python
Λ = LER(d) / LER(d+2)
# 理想情况下 Λ > 1，表示增加 distance 有效
```

---

## 9. 待讨论/实现的内容

1. **数据加载器**: Stim 数据生成和批处理
2. **位置索引构建**: 从 Stim 获取 stabilizer/data qubit 坐标映射
3. **训练循环**: 完整的训练代码
4. **Baseline 对比**: PyMatching MWPM 解码器
5. **实验配置**: 具体的实验设置和超参数搜索

---

## 10. 文件结构规划

```
alphaqubit/
├── models/
│   ├── __init__.py
│   ├── embedder.py          # SyndromeEmbedder, FinalDataEmbedder
│   ├── resnet.py            # ResNetBlock
│   ├── transformer.py       # SyndromeTransformer, Attention, GLU, Conv
│   ├── rnn_core.py          # RNNCore
│   ├── readout.py           # ReadoutNetwork
│   └── alphaqubit.py        # 主模型
├── data/
│   ├── __init__.py
│   ├── stim_generator.py    # Stim 数据生成
│   ├── soft_readout.py      # Soft readout 模拟
│   ├── dataset.py           # PyTorch Dataset
│   └── coordinates.py       # 位置坐标处理
├── training/
│   ├── __init__.py
│   ├── trainer.py           # 训练循环
│   ├── loss.py              # 损失函数
│   └── metrics.py           # 评估指标
├── configs/
│   ├── model_config.py      # 模型配置
│   └── train_config.py      # 训练配置
└── scripts/
    ├── train.py             # 训练脚本
    └── evaluate.py          # 评估脚本
```
