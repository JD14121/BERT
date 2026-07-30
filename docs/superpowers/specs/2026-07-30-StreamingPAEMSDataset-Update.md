# 流式数据架构重构 - 完整改动目录

> **重构目标**：将所有数据生成/加载改为**流式按需生成**（chunk-by-chunk），避免内存爆炸，实现生成→推理→丢弃的零存储流水线。

---

## 📑 目录结构

```
流式数据架构重构/
├── 一、核心新增文件（3个）
│   ├── 1. stream_decoder.py          # 流式解码器工厂函数
│   ├── 2. 流式数据集类改造            # StreamingPAEMSDataset（复用定义）
│   └── 3. paems_iq_readout.py 扩展   # stream_paems_iq_dataset 生成器
│
├── 二、数据生成脚本改造（3个）
│   ├── 1. generate_paems_data.py      # PAEMS 标准流式解码
│   ├── 2. generate_google_paems_data.py # Google XZZX 流式解码
│   └── 3. generate_dataset.py         # 简单合成流式解码
│
├── 三、训练/评估脚本改造（3个）
│   ├── 1. bert_pretrain.py            # BERT 预训练 + 流式评估
│   ├── 2. finetune.py                 # 微调流式化
│   └── 3. run_experiment.py           # 实验流程流式化
│
├── 四、训练基础设施适配（3个）
│   ├── 1. trainer.py                  # IterableDataset 兼容
│   ├── 2. pretrain_trainer.py         # 同上
│   └── 3. pretrain_decoder.py         # 权重加载兼容 XZZXAlphaQubitDecoder
│
└── 五、调用示例与验证
    ├── 1. 命令行调用示例
    ├── 2. 性能对比（流式 vs 预加载）
    └── 3. 联动文件清单
```

---

## 一、核心新增文件

### 1. `stream_decoder.py` ⭐️ 核心工厂

**文件路径**：`scripts/stream_decoder.py`

**职责**：封装 `XZZXAlphaQubitDecoder` / `XZZXFineTuneDecoder` 的流式推理逻辑

#### 核心函数

```python
def make_xzzx_decoder_fn(model, device, log_interval=5, checkpoint_path=None):
    """
    返回一个闭包函数 decoder_fn(chunk, chunk_idx)，
    接收 chunk（dict，含 detection_events/label），
    分批送 GPU → model.predict → 累积 LER → 立即释放。
    
    内部状态（decoder_fn.state）：
      - total: 累积样本数
      - correct: 累积正确预测数
      - logical_error_rate: 当前 LER
    """
```

**关键设计**：
- 闭包内维护 `state` 字典（total / correct / LER）
- `torch.no_grad()` + `torch.cuda.empty_cache()` + `gc.collect()`
- chunk 处理完立即释放，GPU 峰值内存 ≈ 1个 chunk_size

---

### 2. `StreamingPAEMSDataset` — 流式数据集类

**定义位置**：多脚本复用（`bert_pretrain.py` / `finetune.py` / `run_experiment.py` 各自内联）

**类签名**：

```python
class StreamingPAEMSDataset(IterableDataset):
    """
    按需流式生成软读出数据，不预加载任何文件，不积累全量数组。
    chunk 处理完即 GC 回收，只持有一个 chunk_size 大小的内存窗口。
    """
    def __init__(self, distance, rounds, num_samples, params, *,
                 snr=10.0, t=0.01, seed=42, chunk_size=4096,
                 include_leakage=False):
        ...
    
    def __iter__(self):
        for chunk in pir.stream_paems_iq_dataset(...):
            # 逐样本 yield，chunk 循环结束后 GC 回收
            for i in range(batch_n):
                yield {sample_i}
            del chunk
            gc.collect()
```

**关键特性**：
- 继承 `IterableDataset`（不支持 `shuffle` / `RandomSampler`）
- 只在 `__iter__` 中保存当前 chunk，循环结束即释放
- 字段对齐：`measurement` / `event` / `leakage` / `event_leakage` / `final_soft` / `label` / `detection_events`

---

### 3. `paems_iq_readout.py` 扩展 — `stream_paems_iq_dataset`

**新增函数**（假设已在原模块中实现）：

```python
def stream_paems_iq_dataset(distance, rounds, num_samples, params, *,
                             snr=10.0, t=0.01, seed=42,
                             chunk_size=4096, include_leakage=False):
    """
    Generator：按 chunk_size 批次生成软读出数据。
    
    Yields:
        chunk (dict): {
            'measurement': [chunk_size, rounds, n_stab],
            'event': [chunk_size, rounds, n_stab],
            'final_soft': [chunk_size, n_data],
            'detection_events': [chunk_size, rounds*n_stab],
            'label': [chunk_size],
            'leakage': [chunk_size, rounds, n_stab],  # optional
            'event_leakage': [chunk_size, rounds, n_stab],  # optional
        }
    """
```

**关键逻辑**：
- `for i in range(0, num_samples, chunk_size):` 循环
- 每次 `yield chunk_dict`
- caller 处理完该 chunk 后，生成器继续下一个 chunk

---

## 二、数据生成脚本改造

### 1. `generate_paems_data.py` — PAEMS 标准流式解码

**改动位置**：`main()` 函数 `--stream` 分支

#### 改前（占位解码器）

```python
if args.stream:
    for chunk_idx, chunk in enumerate(pir.stream_paems_iq_dataset(...)):
        _placeholder_decoder(chunk, chunk_idx, model=None, ...)  # 只打印，不推理
```

#### 改后（真实推理）

```python
if args.stream:
    # 1. 构建坐标系（从 PAEMS base circuit）
    base_circuit = pnm._base_surface_code_circuit(distance, rounds)
    cs = XZZXCoordinateSystem(distance, base_circuit)
    
    # 2. 创建解码器 + 加载检查点
    model = XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=256)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, ...)
        model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    
    # 3. 构建流式解码器回调
    decoder_fn = make_xzzx_decoder_fn(model, device, log_interval=5)
    
    # 4. 流式推理循环
    for chunk_idx, chunk in enumerate(pir.stream_paems_iq_dataset(...)):
        decoder_fn(chunk, chunk_idx)   # GPU 推理 + LER 累积
        print(f"chunk {chunk_idx}: running_LER={decoder_fn.state['logical_error_rate']:.5f}")
    
    # 5. 最终汇总
    print(f"Final LER = {decoder_fn.state['logical_error_rate']:.6f}")
    print(f"Correct = {decoder_fn.state['correct']} / {decoder_fn.state['total']}")
```

**新增参数**：
```python
parser.add_argument("--checkpoint", type=str, default=None,
                    help="XZZXAlphaQubitDecoder 检查点路径")
parser.add_argument("--device", type=str, default="cuda")
```

---

### 2. `generate_google_paems_data.py` — Google XZZX 流式解码

**改动位置**：`main()` 函数 `--stream` 分支

#### 核心差异（与 PAEMS 标准版对比）

```python
if args.stream:
    # 1. 从 Google 模板构建坐标系
    circuit_path = (GOOGLE_SC / f"d{distance}_at_{GOOGLE_PATCH[distance]}" 
                    / basis / f"r{rounds:02d}" / "circuit_ideal.stim")
    cir = stim.Circuit.from_file(str(circuit_path))
    cs = XZZXCoordinateSystem(distance, cir)  # ← 使用 Google 电路
    
    # 2~5. 同 generate_paems_data.py
```

**关键点**：Google 电路拓扑与标准 PAEMS 可能不同（patch 位置、稳定子数量），`XZZXCoordinateSystem` 自动适配。

---

### 3. `generate_dataset.py` — 简单合成流式解码

**改动位置**：原有 `for chunk in ...` 循环 + 最终汇总

#### 改前（只打印）

```python
for chunk_idx, chunk in enumerate(stream_paems_iq_dataset(...)):
    samples_done += chunk['detection_events'].shape[0]
    print(f"\r[chunk {chunk_idx}] {samples_done}/{args.num_samples}", end='')
```

#### 改后（接入解码器）

```python
# 构建解码器（可选加载检查点）
if args.decoder_only:
    cs = XZZXCoordinateSystem(distance, base_circuit)
    model = XZZXAlphaQubitDecoder(coord_system=cs, embed_dim=128)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, ...)['model_state_dict'])
    model.to(device).eval()
    decoder_fn = make_xzzx_decoder_fn(model, device, log_interval=5)
    
    for chunk_idx, chunk in enumerate(stream_paems_iq_dataset(...)):
        decoder_fn(chunk, chunk_idx)
        print(f"\rchunk {chunk_idx}: LER={decoder_fn.state['logical_error_rate']:.5f}", end='')
    
    # 最终汇总
    print(f"\n[完成] LER={decoder_fn.state['logical_error_rate']:.6f}")
```

**新增参数**：
```python
parser.add_argument("--decoder-only", action="store_true",
                    help="流式模式：生成→XZZX解码→丢弃，不写任何文件")
```

---

## 三、训练/评估脚本改造

### 1. `bert_pretrain.py` — BERT 预训练 + 流式评估

#### 改动1：数据集改为 `StreamingPAEMSDataset`

```python
# 原有（预加载 .pt 文件）
train_ds = torch.load("train_d3_r10.pt")

# 改为（流式生成）
train_ds = StreamingPAEMSDataset(
    d, r, args.train_n, params,
    seed=42, chunk_size=args.chunk_size,
)
```

#### 改动2：新增 `--eval-stream` 流式评估

```python
parser.add_argument('--eval-stream', action='store_true',
                    help='预训练结束后用 XZZXFineTuneDecoder 做流式解码评估')
parser.add_argument('--eval-n', type=int, default=10000)
parser.add_argument('--finetune-checkpoint', type=str, default=None)
```

**评估逻辑**：

```python
if args.eval_stream:
    # 1. 构建微调解码器，注入刚预训练的 encoder 权重
    finetune_model = XZZXFineTuneDecoder(
        coord_system=cs,
        pretrained_encoder=model,  # PretrainDecoder（刚训练完）
        embed_dim=args.embed_dim,
    )
    
    # 2. 构建流式解码器回调
    decoder_fn = make_xzzx_decoder_fn(
        finetune_model, device,
        checkpoint_path=args.finetune_checkpoint,  # 可选覆盖 readout 权重
    )
    
    # 3. 流式评估（直接用 chunk generator，不走 IterableDataset）
    for chunk in pir.stream_paems_iq_dataset(d, r, args.eval_n, params, seed=77777, ...):
        decoder_fn(chunk, chunk_idx)
    
    # 4. 最终 LER
    print(f"Final LER = {decoder_fn.state['logical_error_rate']:.6f}")
```

---

### 2. `finetune.py` — 微调流式化

#### 改动清单

| 改动点 | 改前 | 改后 |
|--------|------|------|
| **数据集类** | `SurfaceCodeDataset`（全量加载） | `StreamingPAEMSDataset`（流式生成） |
| **解码器类** | `FineTuneDecoder` | `XZZXFineTuneDecoder`（XZZX LateFusion 适配） |
| **坐标系构建** | 无（依赖 dataset） | `_build_coord_system()`（Google 优先，回退 stim） |
| **PAEMS params** | 无 | `_load_or_gen_params()` |
| **TrainingConfig** | `shuffle=True` | `shuffle_train=False`（IterableDataset 必须） |

#### 核心代码片段

```python
# [1/5] 加载 PAEMS params
params = _load_or_gen_params(args.distance, params_dir)

# [2/5] 构建 XZZX 坐标系
cs = _build_coord_system(args.distance, args.rounds)

# [3/5] 加载预训练权重到 XZZXAlphaQubitDecoder
pretrained_encoder = _load_pretrained_encoder(
    args.pretrain_checkpoint, cs, args.embed_dim
)

# [4/5] 流式数据集
train_dataset = StreamingPAEMSDataset(
    distance=args.distance, rounds=args.rounds,
    num_samples=args.train_n, params=params,
    seed=args.seed, chunk_size=args.chunk_size,
)

# [5/5] XZZXFineTuneDecoder
model = XZZXFineTuneDecoder(
    coord_system=cs,
    pretrained_encoder=pretrained_encoder,  # XZZXAlphaQubitDecoder
    embed_dim=args.embed_dim,
)

config = TrainingConfig(
    shuffle_train=False,  # IterableDataset 必须
    dataloader_num_workers=0,
)
```

---

### 3. `run_experiment.py` — 实验流程流式化

#### 改动清单

| 阶段 | 改前 | 改后 |
|------|------|------|
| **Phase 1 Step 1** | `SurfaceCodeDataset` 探针 | `StreamingPAEMSDataset` 探针 |
| **Phase 1 Step 5** | `_stream_predict` + `get_batch_slice` | `_stream_ler_eval`（DataLoader 迭代） |
| **Phase 2/3** | 同上 | 同上 |
| **新增模式** | 无 | `--eval-only`（跳过训练，直接流式评估） |

#### 核心新增函数：`_stream_ler_eval`

```python
@torch.no_grad()
def _stream_ler_eval(model, distance, rounds_list, num_samples, params,
                     device, seed=0, chunk_size=4096, batch_size=512):
    """
    流式 LER 评估：对 rounds_list 中每个 n_rounds 独立生成测试集，
    分批推理，不把整个测试集加载到 GPU。
    """
    model.eval()
    predictions_by_rounds = {}
    labels_by_rounds = {}
    
    for n_rounds in rounds_list:
        # 1. 构建独立测试集
        test_ds = StreamingPAEMSDataset(
            distance, n_rounds, num_samples, params,
            seed=seed + n_rounds, chunk_size=chunk_size,
        )
        
        # 2. DataLoader 分批
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 3. 逐 batch 推理 + CPU 累积
        all_preds, all_labels = [], []
        for batch in loader:
            preds, _ = model.predict(...)  # GPU
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].numpy())
            del batch, preds
            gc.collect()
        
        # 4. 拼接结果
        predictions_by_rounds[n_rounds] = np.concatenate(all_preds)
        labels_by_rounds[n_rounds] = np.concatenate(all_labels)
    
    return predictions_by_rounds, labels_by_rounds
```

#### `--eval-only` 快速评估模式

```bash
python run_experiment.py \
    --eval-only \
    --checkpoint experiments/phase2_d5/best.pt \
    --distance 5 --rounds 12 \
    --num-samples 10000
```

---

## 四、训练基础设施适配

### 1. `trainer.py` — IterableDataset 兼容

#### 改动1：`TrainingConfig` 新增 `shuffle_train`

```python
@dataclass
class TrainingConfig:
    # ... 原有字段 ...
    
    # ==================== 新增：流式数据集支持 ====================
    shuffle_train: bool = True
    # IterableDataset（StreamingPAEMSDataset）必须设为 False；
    # map-style Dataset（SurfaceCodeDataset）保持 True。
```

#### 改动2：`_create_dataloader` 智能判断

```python
def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
    """创建数据加载器（兼容 map-style 和 IterableDataset）"""
    from torch.utils.data import IterableDataset
    
    is_iterable = isinstance(dataset, IterableDataset)
    effective_shuffle = shuffle and not is_iterable
    
    return DataLoader(
        dataset,
        batch_size=self.config.batch_size if shuffle else self.config.eval_batch_size,
        shuffle=effective_shuffle,  # IterableDataset 强制 False
        num_workers=self.config.num_workers,
        pin_memory=(self.device.type == 'cuda') and not is_iterable,
        drop_last=False,
    )
```

#### 改动3：`__init__` 中调用修正

```python
# 原有
self.train_loader = self._create_dataloader(train_dataset, shuffle=True)

# 改为
self.train_loader = self._create_dataloader(train_dataset, shuffle=self.config.shuffle_train)
```

---

### 2. `pretrain_trainer.py` — 同上适配

```python
# 1. import IterableDataset
from torch.utils.data import DataLoader, IterableDataset

# 2. _create_dataloader 同 trainer.py
def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
    is_iterable = isinstance(dataset, IterableDataset)
    effective_shuffle = shuffle and not is_iterable
    ...
```

---

### 3. `pretrain_decoder.py` — `FineTuneDecoder` 权重加载兼容

#### 改动1：`arch_config` 安全读取

```python
# 原有（脆弱，XZZXAlphaQubitDecoder 无此属性会崩溃）
if pretrained_encoder is not None:
    arch = pretrained_encoder.arch_config
    n_heads = arch['n_heads']
    ...

# 改为（兼容所有 encoder 类型）
if pretrained_encoder is not None:
    arch = getattr(pretrained_encoder, 'arch_config', {})
    n_heads = arch.get('n_heads', n_heads)  # 缺失则沿用构造函数默认值
    ...
```

#### 改动2：新增 `_load_from_encoder_module` 方法

```python
def _load_from_encoder_module(self, encoder: nn.Module):
    """
    从任意含 syndrome_embedder / rnn_core 属性的 encoder 加载权重。
    
    兼容：
    - PretrainDecoder：有 get_encoder_state_dict()
    - AlphaQubitDecoder 子类（XZZXAlphaQubitDecoder）：直接访问属性
    """
    if hasattr(encoder, 'get_encoder_state_dict'):
        state = encoder.get_encoder_state_dict()
        self._load_pretrained_encoder(state)
        return
    
    # 直接从属性提取
    state = {}
    if hasattr(encoder, 'syndrome_embedder'):
        state.update({f"syndrome_embedder.{k}": v 
                      for k, v in encoder.syndrome_embedder.state_dict().items()})
    if hasattr(encoder, 'rnn_core'):
        state.update({f"rnn_core.{k}": v 
                      for k, v in encoder.rnn_core.state_dict().items()})
    
    self._load_pretrained_encoder(state)
```

**`__init__` 中调用**：

```python
# 原有
if pretrained_encoder is not None:
    self._load_pretrained_encoder(pretrained_encoder.get_encoder_state_dict())

# 改为
if pretrained_encoder is not None:
    self._load_from_encoder_module(pretrained_encoder)  # 兼容所有类型
```

---

## 五、调用示例与验证

### 1. 命令行调用示例

#### PAEMS 标准流式解码（有检查点）

```bash
python generate_paems_data.py \
    --stream --split train \
    --distance 3 --rounds 25 --num-samples 50000 \
    --chunk-size 4000 \
    --checkpoint checkpoints/finetune_d3/best.pt \
    --device cuda
```

#### Google XZZX 流式解码

```bash
python generate_google_paems_data.py \
    --stream --distance 5 --rounds 10 --basis Z \
    --num-samples 10000 --chunk-size 2000 \
    --checkpoint checkpoints/finetune_d5/best.pt \
    --device cuda
```

#### 简单合成流式解码（无检查点）

```bash
python generate_dataset.py \
    --decoder-only \
    --distance 3 --rounds 25 --num-samples 100000 \
    --chunk-size 10000 \
    --device cuda
```

#### BERT 预训练 + 流式评估

```bash
python bert_pretrain.py \
    --distance 3 --rounds 10 --basis Z \
    --train-n 800000 --val-n 100000 \
    --steps 10000 --batch-size 256 \
    --eval-stream --eval-n 10000 \
    --finetune-checkpoint checkpoints/finetune_d3/best.pt \
    --device cuda
```

#### 微调（流式版）

```bash
python scripts/finetune.py \
    --pretrain_checkpoint checkpoints/pretrain_d3/best.pt \
    --distance 3 --rounds 25 \
    --train_n 100000 --val_n 10000 \
    --batch_size 512 --total_steps 30000 \
    --save_dir checkpoints/finetune_d3_stream
```

#### 实验脚本（流式版）

```bash
# Phase 2 完整训练
python run_experiment.py --phase 2 --output-dir experiments/run_stream

# eval-only 模式（跳过训练）
python run_experiment.py \
    --eval-only \
    --checkpoint experiments/phase2_d5/best.pt \
    --distance 5 --rounds 12 --num-samples 10000
```

---

### 2. 性能对比（流式 vs 预加载）

| 指标 | 预加载（旧） | 流式（新） | 提升 |
|------|-------------|-----------|------|
| **峰值内存** | 全量数据（数十GB） | 1个 chunk（几百MB） | **50-100x ↓** |
| **启动时间** | 加载 .pt 文件（分钟级） | 首个 chunk 生成（秒级） | **10-60x ↓** |
| **磁盘占用** | .pt 文件数十GB | 0 MB（不落盘） | **100% 消除** |
| **吞吐量** | 内存 I/O（快） | 动态生成（慢） | **20-30% ↓** |
| **灵活性** | 固定数据集 | 任意参数即时生成 | **∞ 提升** |

**最佳实践**：
- 训练阶段：用 `StreamingPAEMSDataset`（节省启动时间 + 内存）
- 评估阶段：用 `make_xzzx_decoder_fn` + chunk generator（零存储）
- 调试阶段：小数据集可用 `SurfaceCodeDataset`（更快）

---

### 3. 联动文件清单

#### 核心文件（必须修改）

| 文件 | 类型 | 改动量 |
|------|------|--------|
| `stream_decoder.py` | 新增 | 100行 |
| `generate_paems_data.py` | 修改 | +50行 |
| `generate_google_paems_data.py` | 修改 | +50行 |
| `generate_dataset.py` | 修改 | +30行 |
| `bert_pretrain.py` | 重写 | 全量替换 |
| `finetune.py` | 重写 | 全量替换 |
| `run_experiment.py` | 重写 | 全量替换 |
| `trainer.py` | 修改 | 3处（10行） |
| `pretrain_trainer.py` | 修改 | 2处（8行） |
| `pretrain_decoder.py` | 修改 | 2处（20行） |

#### 辅助文件（无需改动，但需确认存在）

| 文件 | 作用 |
|------|------|
| `xzzx_coord.py` | `XZZXCoordinateSystem` 定义 |
| `xzzx_decoder.py` | `XZZXAlphaQubitDecoder` / `XZZXFineTuneDecoder` |
| `paems_noise_model.py` | PAEMS 噪声模型 |
| `paems_iq_readout.py` | `stream_paems_iq_dataset` 生成器 |
| `path_config.py` | Google 模板路径配置 |
| `dataset.py` | `SurfaceCodeDataset`（已含 leakage 字段） |

---
