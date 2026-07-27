"""
AlphaQubit: 量子纠错神经网络解码器

AlphaQubit是一个基于深度学习的量子纠错解码器，
专为Surface Code设计，实现了Google论文中描述的架构。

模块结构：
┌─────────────────────────────────────────────────────────────────┐
│                        AlphaQubit                               │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐    │
│  │    data     │   │   models    │   │     training        │    │
│  │             │   │             │   │                     │    │
│  │ - coords    │   │ - decoder   │   │ - trainer           │    │
│  │ - stim      │   │ - embedder  │   │ - scheduler         │    │
│  │ - soft read │   │ - rnn       │   │ - losses            │    │
│  │ - dataset   │   │ - conv      │   │                     │    │
│  └─────────────┘   │ - fusion    │   └─────────────────────┘    │
│                    └─────────────┘                              │
│                                                                 │
│  ┌─────────────┐   ┌───────────────────────────────────────┐    │
│  │ evaluation  │   │            experiments                │    │
│  │             │   │                                       │    │
│  │ - metrics   │   │ - multi_distance (多码距训练)           │    │
│  │ - bootstrap │   │ - baselines （MWPM等基线）              │    │
│  │ - calibrate │   │ - evaluator （完整评估流程)              │    │
│  │             │   │ - visualize （可视化）                  │    │
│  └─────────────┘   └───────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

快速开始：
```python
from alphaqubit.data import SurfaceCodeDataset
from alphaqubit.models import AlphaQubitDecoder, AlphaQubitDecoderConfig
from alphaqubit.training import Trainer, TrainingConfig
from alphaqubit.evaluation import compute_ler, bootstrap_ler
from alphaqubit.experiments import (
    MultiDistanceTrainer,
    MWPMBaseline,
    Evaluator,
    plot_ler_vs_noise,
)

# 创建数据集
train_ds = SurfaceCodeDataset(distance=5, rounds=12, p=0.005)
val_ds = SurfaceCodeDataset(distance=5, rounds=12, p=0.005, seed=123)

# 创建模型
model = AlphaQubitDecoderConfig.base(train_ds.coord_system)

# 训练
config = TrainingConfig(total_steps=30000)
trainer = Trainer(model, train_ds, val_ds, config)
history = trainer.train()

# 评估
evaluator = Evaluator(model, EvaluationConfig(distances=[5], noise_levels=[0.005]))
results = evaluator.evaluate_all()

# 可视化
plot_ler_vs_noise(results)
```
"""

__version__ = "0.1.0"

# ==================== 子模块 ====================
from . import data
from . import models
from . import training
from . import evaluation
from . import experiments
from . import configs

# ==================== 常用导出 ====================
# 数据
from .data import (
    SurfaceCodeDataset,
    CoordinateSystem,
    StimDataGenerator,
    SoftReadoutSimulator,
    NPZDataset,
)

# 模型
from .models import (
    AlphaQubitDecoder,
    AlphaQubitDecoderConfig,
)

# 训练
from .training import (
    Trainer,
    TrainingConfig,
    train_decoder,
)

# 评估
from .evaluation import (
    compute_ler,
    compute_error_rate,
    compute_fidelity,
    bootstrap_ler,
    LERResult,
    BootstrapResult,
)

# 实验
from .experiments import (
    MultiDistanceTrainer,
    MultiDistanceConfig,
    MWPMBaseline,
    Evaluator,
    EvaluationConfig,
    EvaluationResult,
    plot_ler_vs_noise,
    plot_ler_vs_distance,
    plot_comparison,
)

__all__ = [
    # 版本
    "__version__",
    # 子模块
    "data",
    "models",
    "training",
    "evaluation",
    "experiments",
    "configs",
    # 数据
    "SurfaceCodeDataset",
    "CoordinateSystem",
    "StimDataGenerator",
    "SoftReadoutSimulator",
    "NPZDataset",
    # 模型
    "AlphaQubitDecoder",
    "AlphaQubitDecoderConfig",
    # 训练
    "Trainer",
    "TrainingConfig",
    "train_decoder",
    # 评估
    "compute_ler",
    "compute_error_rate",
    "compute_fidelity",
    "bootstrap_ler",
    "LERResult",
    "BootstrapResult",
    # 实验
    "MultiDistanceTrainer",
    "MultiDistanceConfig",
    "MWPMBaseline",
    "Evaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "plot_ler_vs_noise",
    "plot_ler_vs_distance",
    "plot_comparison",
]
