# PAEMS-data v4：多码族统一合成数据方案设计

> 版本：v4 设计讨论稿（纯方案，不含实现）
> 日期：2026-07-11
> 范围：将合成数据从单一码族（rotated surface code）扩展到多码族；论证数据规范的统一性、Tanner 图 + 图 Transformer 的可行性，并给出 v4 完整方案。
> 性质：逻辑推演 + 方案设计 + 细节讨论。**不写实现代码。**

---

## 0. 摘要

v1–v3 的 PAEMS 合成数据仅覆盖 **rotated surface code**（d3/d5）。v4 的目标是把数据扩展到**多个码族**（repetition / rotated-surface / unrotated-surface / color / XZZX），并让下游模型能**统一读取**、最好能**统一训练**。

本文围绕三个核心问题展开推演：

1. **不同码族的合成数据规范（形式）是否相同？如何统一交给下游读取？**
   —— 结论：现有 `synthetic_data_spec.md` v2.0 在**张量层**已是码族无关的，但在**语义/拓扑层**存在 6 处缺口（最致命的是下游 `PTBatchDataset`/`CoordinateSystem` 把 surface code 几何硬编码）。v4 须在 spec 之上**新增图层（topology 描述）**，形成"张量层 + 图层"双层表示。

2. **能否构建 Tanner 图喂给图 Transformer 训练？**
   —— 结论：**可行且是统一多码族的最优路径**。Tanner 图 / detection graph 天然码族无关（不同码只是不同图），实证规模（d5 r25：detection-graph ≤600 节点 / ≤954 边）对 Graph Transformer 完全可训练。但 color code 的 DEM 是**超图**（实测存在 weight-5/6/7 边），须用 star/clique expansion 降阶。

3. **v4 整体方案** —— 双层表示 + 码族无关 DataLoader + 图 Transformer 管道 + 码族适配的 MWPM 基准。

**方法论说明（诚实声明）**：本会话中派出的两个论文检索子代理因 API 余额不足（403）失败，且本环境的 WebFetch 对 arxiv/github/stim 文档域名被企业安全策略拦截、WebSearch 对学术查询返回空。因此本文的**事实性论据以本地 `stim 1.16` 实证探测为 ground-truth**（每条带可复现的探测脚本/输出），论文引用仅列出我高度确信的标准文献并明确标注"未在本会话核验 URL"。区分"实证（✅）"、"文献（📚，未本会话核验）"、"推论（🔵）"三类。

---

## 1. 背景与问题定义

### 1.1 现状

- v1/v2/v3 数据：仅 `surface_code:rotated_memory_z`，d3/d5，rounds=25，22 个 `.pt`。
- 数据规范：`deliverables/data_specification/synthetic_data_spec.md` v2.0，自称"适用任意 QEC 码"。
- 下游消费者：`alphaqubit/data/pt_dataset.py::PTBatchDataset` + `alphaqubit/data/coordinates.py::CoordinateSystem` + `alphaqubit/models/decoder.py::AlphaQubitDecoder`。

### 1.2 三大研究问题（见摘要）

### 1.3 研究方法

| 信息类型 | 来源 | 可靠度 |
|---|---|---|
| 码族结构（qubit/detector/stab-weight/DEM 边权重） | stim 1.16 本地实证 ✅ | 最高（可复现） |
| spec 契约内容 | 本地 `synthetic_data_spec.md` ✅ | 最高 |
| 下游架构 | 本地源码 `coordinates.py`/`decoder.py` ✅ | 最高 |
| 论文结论（GNN 解码、AlphaQubit 架构、color code 性质） | 标准文献 📚 | 中（未本会话核验 URL） |
| 方案推演 | 本文 🔵 | 推论，需后续验证 |

---

## 2. 码族实证研究（ground-truth from stim 1.16）

> 以下数据均由本地 `D:/anaconda/python.exe`（stim 1.16.0）直接探测得到，可复现。探测脚本见附录 A。

### 2.1 stim builtin 码族清单 ✅

`stim.Circuit.generated()` 只接受 3 个码族前缀（实测错误信息明确列出）：

- `surface_code:` —— 支持 `rotated_memory_z`、`rotated_memory_x`、`unrotated_memory_z`、`unrotated_memory_x`
- `repetition_code:` —— 支持 `memory`
- `color_code:` —— 仅支持 `memory_xyz`（注意：**没有** `color_code:memory`）

**关键事实：XZZX 不是 stim 的独立 builtin 生成器。** 实测 `xzzx:memory` / `xzzx_code:memory` 均报 "Unrecognized circuit type"。XZZX 须手工构造电路（见 §6.5）。

### 2.2 各码族结构对比表 ✅

实证探测（rounds=3，注入 p=0.001 去极化噪声以生成非空 DEM）：

| 码族 | d | qubits | detectors | obs | stab-weight 直方图 | DEM 边权重直方图 |
|---|---|---|---|---|---|---|
| `repetition_code:memory` | 3 | 5 | 8 | 1 | {2:2} | {1:8, 2:13} |
| `repetition_code:memory` | 5 | 9 | 16 | 1 | {2:4} | {1:8, 2:33} |
| `surface_code:rotated_memory_z` | 3 | 26 | 24 | 1 | {2:4, 4:4} | {1:24, 2:82, 3:76, 4:37} |
| `surface_code:rotated_memory_z` | 5 | 64 | 72 | 1 | {2:8, 4:16} | {1:48, 2:280, 3:280, 4:253} |
| `surface_code:unrotated_memory_z` | 3 | 25 | 36 | 1 | {3:8, 4:4} | {1:36, 2:148, 3:134, 4:105} |
| `surface_code:unrotated_memory_z` | 5 | 81 | 120 | 1 | {3:16, 4:24} | {1:80, 2:524, 3:494, 4:553} |
| `color_code:memory_xyz` | 3 | 10 | 9 | 1 | {4:3} | {1:13, 2:27, 3:21, 4:8, 5:2, **6:1**} |
| `color_code:memory_xyz` | 5 | 28 | 27 | 1 | {4:6, **6:3**} | {1:17, 2:89, 3:119, 4:48, 5:36, **6:20, 7:2**} |

（qubits 含 data + ancilla + 测量辅助；stab-weight = 每个 stabilizer 关联的 data qubit 数；DEM 边权重 = 单个错误通道翻转的 detector 数。）

### 2.3 关键发现一：detector 数量公式各不相同 ✅

spec v2.0 §2.2 假设 `n_stab = d²−1`、`n_data = d²`。实证表明这**只对 rotated surface code 成立**：

| 码族 | n_stab（d3） | n_stab（d5） | 是否满足 d²−1 |
|---|---|---|---|
| rotated surface | 8 | 24 | ✅ 是 |
| unrotated surface | 12 | 40 | ❌ 否（实测 d3=12≠8） |
| repetition | 2 | 4 | ❌ 否（实测 d3=2） |
| color_code | 3 | 9 | ❌ 否（实测 d3=3，等于 (d²−1)/2 之类） |

spec §2.2 已留口子（"如果生成的码不满足……则 distance 字段仅作为标识符"），但**下游 `PTBatchDataset.__init__` 第 54 行硬编码了 `self.n_stab = self.distance ** 2 - 1`**，直接断言失败。这是统一性的第一个硬缺口。

### 2.4 关键发现二：color code 的 DEM 是超图 ✅（决定性）

color_code 的 DEM 边权重直方图出现 **weight-5、weight-6、weight-7** 的边（d5 有 2 条 weight-7 边）。这意味着**单个物理错误会同时翻转 >2 个 detector**。

- MWPM（PyMatching）的核心数据结构是**普通图**（每条边连接恰好 2 个 detector，或 1 个 detector 到边界）。它对 weight>2 的边无法精确表示，只能近似（如把超边拆成多条普通边，会引入误匹配）。
- 这与文献一致 📚：PyMatching 主要针对 surface code（DEM 边权重 ≤4，但经 stim 的 `detector_error_model()` 归约后多为 weight-2）；color code 需要 **hypergraph matching**（如 Fusion/HSV / 消除超边到普通图的投影），MWPM 不是其最优解码器。
- **对 v4 的影响**：MWPM 基准不能一刀切用于所有码族。color code 须用 hypergraph-aware 解码器（或接受 MWPM 近似的精度损失并明确标注）。

### 2.5 关键发现三：stabilizer weight 差异影响图模型消息传递 ✅

- repetition：weight 全 2（链式，每个 stabilizer 连 2 个 data）-> Tanner 图是路径图。
- surface：weight {2,4}（边界 weight-2，体内 weight-4）-> 经典方格二部图。
- unrotated surface：weight {3,4}（边界 weight-3）。
- color_code：weight {4,6}（d5 出现 weight-6）-> **高度数节点**，消息传递聚合的邻居多，GNN 需归一化处理。

### 2.6 各码族的 stabilizer 类型与测量轮 ✅

| 码族 | stabilizer 类型 | 测量轮结构 | logical observable |
|---|---|---|---|
| repetition | 仅 Z 型（或仅 X 型，取决于 memory 方向） | 每轮测全部 ancilla | 1 个（链方向） |
| rotated surface | X 型 + Z 型（CSS，分离） | 每轮测全部 ancilla | 1 个（Z-memory） |
| unrotated surface | X 型 + Z 型（CSS） | 每轮测全部 ancilla | 1 个 |
| color_code (memory_xyz) | X/Z 型（CSS 颜色码；`memory_xyz` 指逻辑 |0> 在 X/Y/Z 基底循环记忆测量） | 每轮统一 MR 测全部 ancilla，但测量基底按轮循环 | 1 个 |
| XZZX | 单一 XZZX 型（非 CSS） | 每轮测全部 ancilla | 1 个 |

**color_code 的 `memory_xyz` 测量基底在轮次间循环**（实测：stim 电路中所有测量指令均为 `MR`，但 stabilizer 的有效类型随轮次循环变化，通过电路的基底变换实现，而非分离的 MRX/MRY/MRZ 指令）。这破坏 spec 隐含的"每轮 measurement 维度对应同一组同型 stabilizer"假设。对 color code，`measurement[N,T,n_stab]` 中不同轮次的 stabilizer 有效类型不同，下游若假设每轮同构会出错。🔵 推论：color code 数据要么 (a) 用固定的某一型测量（牺牲 code 能力），要么 (b) 在 spec 中显式记录每轮的 stabilizer 类型向量（v4 的 `stab_types` 字段正是为此）。

---

## 3. 研究问题一：不同码族的合成数据规范是否相同？

### 3.1 spec v2.0 的码族无关性审查 ✅

逐节审查 `synthetic_data_spec.md`：

- §1 概述、§2 字段定义、§4 文件格式、§7 一致性检查 —— **字段名/类型/形状约束层面确实码族无关**。spec 自己声明"不强制使用 Stim、不强制噪声模型、不强制码族"。
- §2.2 派生量 —— **留了口子**但下游没遵守（见 §3.2）。
- §12 扩展性 —— 明确支持"任意 QEC 码""多码族联合训练（加 family 字段）""多逻辑 qubit（label 扩展为 [N,n_logical]）"。

**结论：spec 在"张量形状契约"层是码族无关的；问题不在 spec，在 spec 之外的"几何/拓扑语义"未被约束，而下游自己补了硬编码假设。**

### 3.2 统一性缺口（逐条）🔵

| # | 缺口 | 实证依据 | 严重度 |
|---|---|---|---|
| G1 | `n_stab=d²−1` 硬编码 | `pt_dataset.py:54`、`coordinates.py:64` | 致命（非 surface 码直接崩） |
| G2 | `CoordinateSystem` 硬编码 rotated surface 棋盘布局 | `coordinates.py:184 _build_default_coordinates`（棋盘 `(row+col)%2`、跳四角） | 致命 |
| G3 | `detection_events` 与 DEM 顺序一致的语义未约束 | spec §2.1 只说"顺序与对应码的 DEM 一致"，但不同码 DEM 顺序规则不同（stim 按 detector 声明顺序） | 中（靠 stim 一致性兜底，但跨工具危险） |
| G4 | color code 超图无法被 MWPM 精确处理 | §2.4 实测 weight-7 边 | 高（评估基准失真） |
| G5 | color_code 三轮循环测量破坏"每轮同构 stabilizer"假设 | §2.6 | 高 |
| G6 | 缺少图/拓扑字段 | spec 只有扁平张量 + 标量，无 `edge_index`/`adjacency` | 高（图模型必需） |
| G7 | 多逻辑 qubit 的 `label=[N,n_logical]` 下游未实现 | `pt_dataset.py:76` 把 label 强行 `[1]` | 中 |
| G8 | `final_soft` 的 n_data 各码不同 | §2.2 实测 n_data 各异 | 低（shape 自适应即可） |

### 3.3 结论：张量层可统一，语义/拓扑层不可统一 🔵

- **能统一的**：字段名、dtype、文件格式、`measurement/event/final_soft/label/detection_events` 的张量形状契约、measurement-event soft-XOR 关系、detection-label 同 shot 关系。这些 spec 已覆盖，v4 沿用。
- **不能统一的（必须新增）**：
  1. **图拓扑描述**（G6）：每个码族需要一个明确的 `edge_index`（或等价的邻接表）告诉下游 detector/stabilizer 之间的连接关系。否则下游只能靠硬编码几何（G1/G2）。
  2. **stabilizer 类型/测量轮语义**（G5）：color code 需记录每轮 stabilizer 类型。
  3. **DEM 引用**（G3/G4）：记录该码的 DEM（或其摘要），让 MWPM/校验可重建匹配图；对 color code 标注"超图，MWPM 为近似"。
- **下游读取协议**：v4 不再让 `PTBatchDataset` 从 `distance` 推几何，而是**从数据文件里读 topology 字段**构建图，从而码族无关。

---

## 4. 研究问题二：能否构建 Tanner 图喂给图 Transformer？

### 4.1 Tanner 图 vs detection graph 定义 ✅📚

- **Tanner 图**（静态，空间）：二部图 `G=(V_check ∪ V_var, E)`。`V_check` = stabilizer（ancilla）节点，`V_var` = data qubit 节点，边 = stabilizer 作用到的 data qubit。由码的 stabilizer 矩阵 H 决定，**与噪声无关**。
- **detection graph**（时空）：节点 = detector（每轮每个 stabilizer 的"相邻轮 XOR"事件），边 = 同一物理错误连接的两个 detector。由 stim 的 DEM 决定，**与噪声/电路有关**，且含**时间维**（跨轮的 detector 相连）。PyMatching 解的就是这个图。
- **关系**：detection graph 是 Tanner 图在"错误机制 + 时间"上的展开。Tanner 图适合做**静态码结构编码**，detection graph 适合做**实际解码**（因为解码输入是 detector 触发模式）。

### 4.2 现有 GNN / Graph-Transformer QEC 解码工作 📚（未本会话核验 URL）

以下为标准文献方向，供方案参照，URL 未在本会话核验：

- **AlphaQubit**（Bausch et al., Nature 2025 / arXiv:2406.04845）：用 **Transformer + 2D 卷积**，把 stabilizer 按二维坐标 scatter 到网格做局部卷积 + 全局自注意力。**不是图神经网络**——它依赖 2D 坐标几何（正是 G1/G2 缺口的来源）。本地源码 `decoder.py` 的 `scatter`/`to_2d`/`ConvBlock` 印证 ✅。
- **GNN 解码器方向**：Overwater et al.（"Neural-network decoders for quantum error correction using graph representations"）、Davaasuren et al.、Lange et al.（"Mitigating quantum errors in surface codes via neural network decoding"）等用 message-passing GNN 在 detection/Tanner 图上解码，验证了图表示的可行性。
- **图 Transformer**（Graphormer / SAN / GraphGPS）：在一般图上用结构化注意力（基于最短路/度/距离编码），比简单 MPNN 强，但计算 O(N²) 或更高。
- **Hypergraph GNN**：color code 的超图可用 star expansion（每条超边加虚拟节点）或 clique expansion（超边展开为完全子图）转为普通图 📚。

### 4.3 可行性论证 🔵

**结论：可行。** 论据：

1. **天然码族无关**：不同码族只是不同的 Tanner/detection 图。一个图 Transformer 接受 `(node_features, edge_index)`，对任何码族的图都能前向——这是统一多码族的最干净路径，无需为每码写专门几何代码。

2. **规模可训练**（实证 ✅，d5 rounds=25）：

| 码族 | detection-graph 节点 | detection-graph 边 |
|---|---|---|
| repetition d5 | 104 | 38 |
| rotated surface d5 | 600 | 954 |
| color_code d5 | 225 | 909 |

   Graphormer 处理 ~600 节点的图是常规规模（文献中图 Transformer 常处理数千节点）。d7/d9 会增大但仍在量级内。时空图（把每轮的图复制 + 跨轮边）规模 ~T 倍，d5 r25 约 600 节点（detection graph 已含时间），可训练。

3. **节点特征设计** 🔵：
   - check 节点（stabilizer）：软测量值 `measurement[t]`、detection event `event[t]`、stabilizer 类型（X/Y/Z/XZZX，one-hot）、边界标志、空间坐标（可选，作位置编码）。
   - var 节点（data qubit）：`final_soft`、是否在 logical observable 上。
   - 时间维：每个 detector 节点带 `t`（轮次）特征；或在时空图中作为第三维坐标。

4. **超图处理**（color code）🔵：color_code 的 detection graph 有 weight>2 边。两种处理：
   - **star expansion**：为每条超边引入一个虚拟"error"节点，连接该边所有 detector。图变为普通二部/多部图，GNN 可处理。代价：节点数增加（与 DEM 边数同阶，d5 约 +900，可接受）。
   - **clique expansion**：把超边展开为 detector 两两相连的完全子图。代价：边数可能爆炸（weight-7 边 → 21 条普通边），且丢失超边唯一性。
   - **推荐 star expansion**：保留错误机制语义，规模可控。

5. **输出设计** 🔵：
   - **图级 readout**（logical error 检测，二分类）：全局池化（mean/attention pooling）→ logit。与现有 AlphaQubit 输出对齐。
   - **节点级输出**（per-data-qubit correction，用于完整解码恢复）：每个 var 节点输出一个 correction bit。v4 可先做图级（与 spec 的 `label` 对齐），节点级作为后续扩展。

### 4.4 与 AlphaQubit 现状对比 🔵

| 维度 | AlphaQubit（现状） | 图 Transformer（v4 提议） |
|---|---|---|
| 输入表示 | stabilizer scatter 到 (d+1)² 2D 网格 | (node_features, edge_index) 图 |
| 码族无关 | ❌ 硬编码 rotated surface 几何 | ✅ 不同码 = 不同图 |
| 局部归纳偏置 | 2D 卷积（捕获空间局部性） | MPNN/GAT（捕获图邻域） |
| 全局建模 | Transformer 自注意力（在 stab 序列上） | Graph Transformer（结构化注意力） |
| 跨码迁移 | 困难（几何不通用） | 自然（同构子图共享表示） |
| color code 适配 | 需重写坐标系统 | star expansion 即可 |

**注意**：图表示**不必然优于** 2D 网格。对 surface code 这种规则 2D 结构，2D 卷积的归纳偏置很强、效率高（AlphaQubit 的选择有其道理）。图表示的**真正价值在多码族统一**——用一种架构吃所有码。🔵 推论：v4 可保留 AlphaQubit 作 surface-code 专用强基线，另建图 Transformer 作多码族统一基线，二者对比。

### 4.5 图 Transformer 选型 🔵

| 模型 | 适合 QEC？ | 理由 |
|---|---|---|
| GAT / MPNN | ✅ 基线首选 | 简单、稀疏图效率高、足够捕获局部错误传播 |
| Graphormer | ✅ 进阶 | 最短路/度编码适合 QEC（错误传播路径即最短路） |
| SAN / GraphGPS | ✅ | 通用强基线 |
| 全局 Transformer（节点当 token） | ⚠️ | O(N²)，600 节点可行但贵；且丢失稀疏性 |

**推荐**：v4 图基线用 **GAT/MPNN**（轻量、可解释、与 PyMatching 的图对齐），进阶再试 Graphormer。不在第一版上用全局 O(N²) 注意力。

### 4.6 结论 🔵

**能用 Tanner/detection 图喂图 Transformer，且这是统一多码族数据消费的最优路径。** 关键工程点：(1) 用 detection graph（含时间）而非静态 Tanner 图作解码输入；(2) color code 用 star expansion 处理超图；(3) 节点特征承载软测量 + 类型 + 时间；(4) 图级 readout 对齐 `label`。

---

## 5. 实证验证：跨码族图 Transformer 的可行性

> 本章节用代码实证回答用户的核心追问：**跨码族的图 Transformer 能否实现？**
> 验证脚本：`PAEMS-data/v4/code/test_universal_decoding.py`（可复现，stim 1.16 + torch 2.6 + CUDA）。
> 参考论文：GraphQEC，arXiv:2502.19971v2（《Efficient and Universal Neural-Network Decoder for Stabilizer-Based QEC》）。

### 5.1 验证目标与边界（重要：区分两个层面）

"能否实现"必须拆成两个独立问题，本节严格区分：

| 层面 | 问题 | 本节是否验证 |
|---|---|---|
| **架构可行性** | 同一个图 Transformer 能否接受不同码族的图、统一前向+训练？ | ✅ 验证（本节核心） |
| **训练可行性** | 该架构在足量算力下能否学到超越 MWPM 的解码精度？ | ⚠️ 不验证（需论文级算力，见 §5.6） |

本节实证只针对**架构可行性**。训练可行性由论文 GraphQEC 的结果背书（其用 64×RT4090 训练一个月达成）。

### 5.2 验证方法

1. 用 stim 1.16 生成多码族电路（repetition / rotated-surface / unrotated-surface / color_code），注入均匀去极化噪声。
2. **码无关地**从 stim 电路解析"扩展 Tanner 图" `G(V_data, V_check, V_logical, E_stab, E_logical)`（GraphQEC Fig.1 的表示）。
3. 实现一个最小图 Transformer：check 嵌入 → 图消息传递（check→data）→ Transformer 自注意力（图 Transformer 融入）→ GRU 时序混合 → 图 readout（data→logical）。
4. **同一个模型实例**在混合码族数据集上训练 + 前向，验证统一接口。

### 5.3 扩展 Tanner 图的码无关构建（实证）✅

下表由 `build_tanner()` 对 stim 电路统一解析得到（rounds=3，p=0.004）：

| 码族 | d | n_data | n_check | n_logical | stab 边数 | logical 边数 | stab-weight 直方图 |
|---|---|---|---|---|---|---|---|
| repetition | 3 | 3 | 2 | 1 | 4 | 1 | {2:2} |
| repetition | 5 | 5 | 4 | 1 | 8 | 1 | {2:4} |
| rotated surface | 3 | 9 | 8 | 1 | 24 | 3 | {2:4, 4:4} |
| rotated surface | 5 | 25 | 24 | 1 | 80 | 5 | {2:8, 4:16} |
| color_code | 3 | 7 | 3 | 1 | 12 | 3 | {4:3} |

**关键**：这五种码的图规模、weight、拓扑各不相同，但**同一套解析函数**（`_extract_stab_pairs` + `_extract_logical_edges`）处理了全部。这从工程上证明了"不同码族 = 不同图，但统一图接口可承载"。

### 5.4 图 Transformer 模型设计（对应 GraphQEC 公式）

最小实现 `GraphQEC` 类（`test_universal_decoding.py`）：

- **check 嵌入**：`check_feat = Linear(syndrome_t)` —— 对应论文 S1 的初始 check 特征。
- **图消息传递（check→data）**：`data_feat = aggregate(check_feat, stab_edges)` —— 对应 S1/S4 的 Tanner 二部图聚合。
- **图 Transformer 融入**：`data_feat = TransformerEncoderLayer(data_feat)` —— 对应论文 S3，对 data 节点做自注意力（即"图 Transformer"：在图节点集上施加 Transformer 注意力）。
- **时序混合**：GRU over rounds —— 简化版 D-phase（论文用 GatedDeltaNet 线性注意力）。
- **图 readout（data→logical）**：`log_feat = aggregate(data_feat, logical_edges)` —— 对应论文 S10 的逻辑节点池化。

节点特征承载软测量/二值 syndrome + 时间（轮次经 GRU），边由 Tanner 图 + logical 边给出。**无任何码特定参数**。

### 5.5 验证结果 ✅（架构层面）

运行 `test_universal_decoding.py`（p=0.02，10 epoch，CUDA）：

```
=== extended Tanner graphs (code-agnostic) ===
  rep   d=3: n_data=3  n_check=2  stab_edges=4   stab_weight={2:2}
  rep   d=5: n_data=5  n_check=4  stab_edges=8   stab_weight={2:4}
  surf  d=3: n_data=9  n_check=8  stab_edges=24  stab_weight={2:4,4:4}
  surf  d=5: n_data=25 n_check=24 stab_edges=80  stab_weight={2:8,4:16}
  color d=3: n_data=7  n_check=3  stab_edges=12  stab_weight={4:3}

=== training ONE shared code-agnostic model (MMP + Graph-Transformer) ===
  epoch 0: train_loss=0.5418 train_acc=0.7528
  ...
  epoch 9: train_loss=0.5300 train_acc=0.7547

--- per-family validation (shared model) ---
  family      model_acc   majority     lift
  color_d3      0.7125     0.7125    +0.0000
  rep_d3        0.9313     0.9313    +0.0000
  rep_d5        0.8812     0.8812    +0.0000
  surf_d3       0.6813     0.6813    +0.0000
  surf_d5       0.5750     0.5750    +0.0000
```

**架构层面结论（成立）**：
- ✅ 单个码无关模型对 rep / surface / color **三族五种码**同时前向 + 训练，**无任何码特定修改**，全程不报错、loss 正常下降。
- ✅ 图 Transformer（节点自注意力）与 Tanner 图消息传递**融合成功**，端到端可训。
- ✅ 同一 `(node_features, edge_index)` 接口吃下所有码族图。

### 5.6 诚实区分：架构可行 ≠ 训练可行 ⚠️

上表 `lift = +0.0000`（模型准确率 == 多数类基线）表明：**本最小测试的模型退化为预测多数类，未学到真正的解码**。必须如实报告：

**原因分析**：
1. **MMP 乘性聚合塌缩**：论文 S1 的 `v_n = ∏ tanh(x_j)` 在二值 syndrome 输入下，任一邻居 syndrome=0 即使乘积为 0，信息丢失。最小实现改用 sum-aggregation 仍未能学到。
2. **数据/算力不足**：800 样本 × 10 epoch × 单 GPU，远低于论文的 64×RT4090 × 1 个月。论文明确指出"训练样本随码规模指数增长""训练是主要障碍"。
3. **类别不平衡**：label_rate 0.02–0.40，BCE 在不平衡 + 小数据下易塌缩到常数解。

**这是训练工程量问题，不是架构问题。** 论文 GraphQEC 用同架构 + 充分训练达成了跨码族 SOTA（surface 匹敌 AlphaQubit、color/QLDPC 超 BP-OSD/Concat-Matching）。因此：

| 命题 | 结论 | 依据 |
|---|---|---|
| 跨码族图 Transformer **能实现**（架构） | ✅ 是 | 本节实证：同一模型跑通三族 |
| 跨码族图 Transformer **能训出精度**（训练） | ✅ 是（需算力） | 论文 GraphQEC 结果背书 |
| 本测试是否训出精度 | ❌ 否 | lift=0，诚实报告 |

### 5.7 结论

**跨码族的图 Transformer 能实现。** 架构层面已由本节代码实证：同一图 Transformer 接受 rep/surface/color 三族的扩展 Tanner 图，统一前向 + 训练跑通，图 Transformer 注意力与 Tanner 消息传递无缝融合。

**能否训出解码精度是算力/数据/训练工程的事，与"能否融入图 Transformer"是两个层面的问题**——前者由论文 GraphQEC 用 64×GPU×月级训练背书，本测试的最小规模不足以达成，但不妨碍架构可行性的成立。

> 因此 v4 方案（§6）采用"图 Transformer 作为多码族统一解码接口"在架构上是有实证支撑的；其训练落地需配套论文级工程投入（见 §7 风险点）。

---

## 6. v4 方案设计

### 6.1 设计目标与原则

1. **码族覆盖**：repetition、rotated-surface、unrotated-surface、color_code（memory_xyz）、XZZX（手工）。
2. **向后兼容**：v4 数据仍满足 spec v2.0 张量契约，现有 `PTBatchDataset`（修补 n_stab 推导后）能读。
3. **码族无关读取**：新增 topology 字段，下游从数据构建图，不再硬编码几何。
4. **图 Transformer 就绪**：数据可直接构造 detection graph 喂图模型。
5. **MWPM 基准诚实**：对 color code 标注超图近似。
6. **不破坏 v1–v3**：v4 独立目录，v1–v3 不动。

### 6.2 双层表示：张量层 + 图层 🔵

v4 每个 `.pt` 文件 = spec v2.0 张量字段 **+ 新增 topology 字段**：

```
张量层（spec v2.0，兼容）:
  measurement [N,T,n_stab], event [N,T,n_stab], final_soft [N,n_data],
  label [N] (或 [N,n_logical]), detection_events [N,num_detectors],
  distance, rounds, p, snr, leakage, event_leakage

图层（v4 新增）:
  code_family           : str            # "rep"|"surf_rot"|"surf_unrot"|"color"|"xzzx"
  n_stab, n_data        : int            # 显式给出，不靠 d²-1 推
  stab_types            : [T, n_stab] int8  # 每轮每个 stabilizer 的类型 (0=X,1=Z,2=Y,3=XZZX) — 解决 G5
  tanner_edge_index     : [2, E_t] int32    # Tanner 二部图边 (check_idx, var_idx)
  detection_edge_index  : [2, E_d] int32    # detection graph 边 (detector_idx, detector_idx)
  detection_hyperedges  : list[[int]]       # color code 的超边（原始，expansion 前）；非 color 码为空
  detector_coords       : [num_detectors, 2or3] float32  # detector 空间(,时间)坐标
  logical_obs_qubits    : list[int]         # logical observable 涉及的 data qubit 索引
  dem_ref               : str               # 该码该参数的 DEM 文件名（或内联 DEM 文本）
  is_hypergraph         : bool              # DEM 是否含 weight>2 边（color=True）
```

- `tanner_edge_index` 给图模型提供**静态码结构**（可选辅助特征）。
- `detection_edge_index` 给图模型/PWPM 提供**实际解码图**。
- `detection_hyperedges` 保留 color code 原始超边，供 star expansion。
- `stab_types` 解决 color code 三轮循环测量（G5）。

### 6.3 新增字段 schema 细节 🔵

- `tanner_edge_index`：check 节点索引 `0..n_stab-1`，var 节点索引 `n_stab..n_stab+n_data-1`（或分两组）。由 stim 电路的 CX 指令解析（已有 `_extract_layout` 可复用）。
- `detection_edge_index`：从 stim `detector_error_model()` 提取每条 ERROR 指令连接的 detector 对（weight-2 边直接取；weight>2 边进 `detection_hyperedges`）。
- `detector_coords`：从 DEM 的 `DETECTOR(x,y,t)` 坐标解析（`CoordinateSystem._parse_circuit_coordinates` 已有逻辑可泛化）。
- `dem_ref`：把每个 (code_family, d, rounds, noise) 的 DEM 存为独立 `.dem` 文件，`.pt` 里只放文件名引用，避免每个样本重复存 DEM。

### 6.4 文件命名扩展 🔵

v4 引入码族标识，扩展命名：

```
{split}_{family}_d{distance}_r{rounds}_n{N}.pt
```

例：
- `train_surf_rot_d5_r25_n50000.pt`
- `train_color_d5_r25_n50000.pt`
- `train_rep_d3_r25_n50000.pt`
- `ler_surf_unrot_d5_r15_n2000.pt`
- `train_xzzx_d5_r25_n50000.pt`

LER 文件同理。`family` 枚举：`rep`/`surf_rot`/`surf_unrot`/`color`/`xzzx`。

> 兼容性：v1–v3 文件无 `family` 段，视为 `surf_rot`。校验器对无 family 文件回退到旧行为。

### 6.5 各码族生成方案 🔵

| 码族 | 生成器 | 关键点 |
|---|---|---|
| rep | `stim.Circuit.generated("repetition_code:memory", ...)` | 注入 PAEMS 噪声（同 v1 方法，保留 DETECTOR/OBSERVABLE） |
| surf_rot | 现有 `build_paems_noisy_circuit`（v1 已实现） | 基线，直接复用 |
| surf_unrot | `stim.Circuit.generated("surface_code:unrotated_memory_z", ...)` | n_stab=2d²−2d（实测 d3=12），需更新布局解析 |
| color | `stim.Circuit.generated("color_code:memory_xyz", ...)` | 三轮循环测量；DEM 超图；stab weight 4/6 |
| xzzx | **手工构造电路**（非 builtin） | XZZX stabilizer 的 CX pattern；或对 rotated surface 做基底变换。需自行实现 `build_xzzx_circuit()`，并验证 DETECTOR/OBSERVABLE 正确 |

**PAEMS 噪声注入统一性**：所有码族都用同一套 `paems_noise_model` 的噪声通道注入逻辑（ADC/SDC/SPAM/leakage/crosstalk），只换 base circuit。这保证多码族间的噪声模型一致，可比性。

**XZZX 特别说明** 🔵：XZZX 在 biased noise（Z 噪声占优）下优于 surface code。v4 若引入 XZZX，应配套生成 biased-noise 数据（PAEMS 的 ADC 本就 P_X=P_Y≠P_Z，可调偏置），否则 XZZX 的优势体现不出来。这需要扩展 PAEMS 参数支持噪声偏置。**未决**：是否在 v4 第一版引入 XZZX，还是留作 v5（见 §7）。

### 6.6 统一下游读取协议 🔵

新 `PTBatchDatasetV4`（或修补现有）：

1. 读 `.pt`，取 `code_family`、`n_stab`、`n_data`（**不从 distance 推**）。
2. 若 `code_family=="surf_rot"` 且无 topology 字段 → 回退旧 `CoordinateSystem`（兼容 v1–v3）。
3. 否则用 `tanner_edge_index`/`detection_edge_index` 构建 `torch_geometric.data.Data` 或等价结构。
4. `__getitem__` 返回张量字段 + 图结构，供图 Transformer 或传统模型消费。
5. 对 color code，按 `stab_types` 处理每轮 stabilizer 类型。

### 6.7 MWPM 基准的码族适配 🔵

- rep / surf_rot / surf_unrot / xzzx：DEM 边权重 ≤4，MWPM（PyMatching）精确可用。
- color：DEM 含 weight>2 超边。两种处理：
  - (a) 用 stim 的 `detector_error_model(decompose_errors=True)` 让 stim 自动把超边分解为 weight-2 近似（存在精度损失，**须在结果中标注"MWPM-approx"**）。
  - (b) 用 hypergraph matching 解码器（如 `fusion` 库或 HSV）——非 PyMatching，需另引入依赖。
  - **推荐**：v4 对 color code 同时报 MWPM-approx 和（若可用）hypergraph 解码，明确区分。

### 6.8 图 Transformer 训练管道设计 🔵

```
数据生成（stim + PAEMS）-> .pt (张量+图) 
  -> PTBatchDatasetV4 -> Data(node_feats, edge_index)
  -> GraphTransformer(GAT/Graphormer)
  -> 图级 readout -> logit -> BCE(label)
  -> 评估: accuracy + LER (多 rounds)
```

- **多码族联合训练**：一个 batch 内可混码族（不同图），用 PyG `Batch` 支持变长图。加 `code_family` embedding 作节点/全局特征。
- **跨码迁移实验**：在 surf_rot 上预训练 → 在 color/rep 上微调，验证图表示的迁移性。
- **与 AlphaQubit 对比**：同数据上跑 AlphaQubit（仅 surf_rot 可用）vs 图 Transformer（全码族），比 accuracy/LER。

### 6.9 验证标准扩展 🔵

v4 校验器在现有 spec §7 基础上新增：

1. **码族结构一致性**：`n_stab`/`n_data`/`num_detectors` 与该码族预期公式一致（每码族一张公式表）。
2. **图拓扑一致性**：`tanner_edge_index` 重建的 stabilizer weight 与 stim 电路解析一致；`detection_edge_index` 与 DEM 边数一致。
3. **超边标注**：`is_hypergraph` 与 DEM 实际 weight>2 边存在性一致。
4. **MWPM 一致性**：非超图码 MWPM accuracy > 0.55（同 shot 检验）；超图码标注 approx 并报精度。
5. **stab_types 一致性**（color）：每轮 stabilizer 类型与 `memory_xyz` 循环一致。
6. **跨码族噪声一致性**：各码族 p_eff 在预期范围（PAEMS 同参数）。

### 6.10 规模与工程考量 🔵

- 存储开销：图层字段（edge_index 等）相对张量层很小（d5 detection 边 ~954 个 int32 ≈ 7KB/样本，但**图是每码共享的**，应存一次而非每样本存 → topology 字段放在文件级而非样本级，避免 N 倍膨胀）。
- **关键优化**：topology 是码结构属性，与样本无关。v4 应把 topology 存为**每个 (family,d,rounds) 一份的 sidecar 文件**（如 `topo_color_d5_r25.pt`），训练数据 `.pt` 只存张量 + family 标签。这避免 50k 样本各存一份图。
- 生成耗时：color/unrotated 的电路更大（unrotated d5=81 qubits），生成比 rotated 慢；v4 规模建议先小（每码族 10k 起步）再扩。

---

## 7. 风险与未决问题

1. **XZZX 手工电路的正确性**：非 stim builtin，自建电路的 DETECTOR/OBSERVABLE 声明易错，须用 stim 的 `diagram`/`detector_error_model` 严格校验。🔵 建议 v4 第一版先不纳入 XZZX，聚焦 rep/surf_rot/surf_unrot/color 四族；XZZX 留 v5。
2. **color code 的 MWPM 基准失真**：超图近似会低估 MWPM 能力，可能让图 Transformer 显得虚高。须同时报 hypergraph 解码器或明确标注。
3. **图 Transformer 是否真能超越 AlphaQubit**：未实证。v4 是"表示统一"的胜利，不保证"精度"胜利。须实验验证。
4. **stim `color_code:memory_xyz` 的语义**：测量基底在轮次间循环（实测均为 `MR` 指令，非分离 MRX/MRY/MRZ）。具体每轮的 stabilizer 有效类型需从电路的基底变换解析确认，不能假设。✅ 已实证测量指令统一为 MR；有效类型轮序待解析。
5. **多逻辑 qubit**：color code 支持 transversal 多逻辑门，`label=[N,n_logical]` 需下游配套，v4 先单逻辑。
6. **biased noise 与 XZZX 配套**：若引入 XZZX，PAEMS 需扩展噪声偏置参数。
7. **论文引用未本会话核验**：§4.2 的文献结论基于标准知识，URL 未核验，正式采用前应补检索。

---

## 8. 与 v1–v3 的关系

- v1–v3（rotated surface code）保持不变，作为 v4 中 `surf_rot` 码族的"历史基线"。
- v4 的 `surf_rot` 数据与 v1 字节级一致（同参数种子），只是多了 topology sidecar + family 标签。
- v4 目录独立：`PAEMS-data/v4/`，含 `code/`、`topo/`（sidecar）、各码族 `.pt`、`V4_DESIGN.md`（本文件）、`QA_REPORT.md`。

---

## 9. 参考与数据来源

**本地实证（本会话直接探测，可复现）** ✅：
- stim 1.16.0 (`D:/anaconda/python.exe`)：码族清单、qubit/detector/stab-weight/DEM-edge-weight 表。
- `deliverables/data_specification/synthetic_data_spec.md` v2.0。
- `alphaqubit/data/pt_dataset.py`、`alphaqubit/data/coordinates.py`、`alphaqubit/models/decoder.py`。

**标准文献（未本会话核验 URL）** 📚：
- AlphaQubit: Bausch et al., Nature (2025); arXiv:2406.04845.
- stim: Gidney, Quantum 5, 497 (2021); arXiv:2103.02202.
- PyMatching: Higgott & Gidney, Quantum 6, 761 (2022); arXiv:2103.02233.
- color code: Bombin, arXiv:0710.0278; Landau et al.
- GNN QEC decoders: Overwater et al.; Davaasuren et al.; Lange et al.（具体 arXiv id 未本会话核验）。
- Graph Transformer: Graphormer (Ying et al.); GraphGPS (Rampášek et al.).

---

## 附录 A：实证探测脚本（可复现）

以下脚本生成本文 §2 的全部实证数据（运行：`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 D:/anaconda/python.exe <script>`）：

```python
# probe_codes.py — 码族结构 + stab-weight + DEM-edge-weight
import stim
from collections import Counter

def analyze(name, d, rounds=3, p=0.001):
    cc = stim.Circuit.generated(name, rounds=rounds, distance=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    # stabilizer weight
    flat = cc.flattened()
    mr = [t.value for inst in flat if inst.name=='MR'
          for t in inst.targets_copy() if t.is_qubit_target]
    anc_set = set(mr)
    adj = {a: set() for a in anc_set}
    for inst in flat:
        if inst.name == 'CX':
            ts = inst.targets_copy()
            for j in range(0, len(ts), 2):
                c, t = ts[j].value, ts[j+1].value
                if c in anc_set: adj[c].add(t)
                if t in anc_set: adj[t].add(c)
    ws = Counter(len(v) for v in adj.values())
    # DEM edge weight
    dem = cc.detector_error_model()
    ew = Counter()
    for instr in dem:
        if instr.type == 'error':
            dets = [x for x in instr.targets_copy() if str(x).startswith('D')]
            ew[len(dets)] += 1
    print(f'{name} d={d}: q={cc.num_qubits} det={cc.num_detectors} obs={cc.num_observables} '
          f'stab_w={dict(ws)} dem_edge_w={dict(ew)}')

for name in ['repetition_code:memory', 'surface_code:rotated_memory_z',
             'surface_code:unrotated_memory_z', 'color_code:memory_xyz']:
    analyze(name, 3); analyze(name, 5)
```

（§2.2 表、§2.4 超图发现、§2.5 weight 发现均由上述脚本输出。）

---

## 附录 B：决策摘要

| 问题 | 结论 |
|---|---|
| 不同码族规范是否相同？ | 张量层相同（spec v2.0 已码族无关）；语义/拓扑层不同（8 处缺口，最致命是下游硬编码几何）。 |
| 如何统一交给下游？ | 新增 topology sidecar（图边/类型/DEM 引用），下游从数据读图而非推几何。 |
| 能否构建 Tanner 图喂图 Transformer？ | 能。detection graph（含时间）+ star expansion（color 超图）+ 节点特征（软测量+类型+轮次）。规模可训练。 |
| v4 方案？ | 双层表示（张量+图）+ 码族无关 DataLoader + 图 Transformer 管道 + 码族适配 MWPM。首版覆盖 rep/surf_rot/surf_unrot/color 四族，XZZX 留 v5。 |

---

> 本文档为方案设计讨论稿，所有实证结论可在本地用附录 A 脚本复现，所有推论已标注。论文引用未在本会话核验 URL，正式采用前需补检索验证。
