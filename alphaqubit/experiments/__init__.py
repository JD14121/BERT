"""
实验模块 - AlphaQubit完整实验流程

这个模块提供Phase 3的核心功能：

1. 多码距训练 (Multi-Distance Training)
   - 联合训练多个码距的模型
   - 码距课程学习策略
   - 共享/独立参数配置

2. 基线比较 (Baseline Comparisons)
   - MWPM (Minimum Weight Perfect Matching) via PyMatching
   - 与神经网络解码器的性能对比

3. 完整评估流程 (Full Evaluation Pipeline)
   - 多噪声水平评估
   - 多码距评估
   - LER和Λ因子计算

4. 结果可视化 (Visualization)
   - LER vs 噪声强度曲线
   - LER vs 码距曲线
   - 校准图和置信区间

实验流程图：
┌─────────────────────────────────────────────────────────────────┐
│                     Experiment Pipeline                          │
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ 数据生成    │ → │ 模型训练      │ → │ 评估和比较        │    │
│  │             │    │              │    │                  │    │
│  │ - d=3,5,7,9│    │ - 单码距训练  │    │ - LER计算        │    │
│  │ - 多噪声p  │    │ - 多码距训练  │    │ - MWPM基线       │    │
│  │ - 多rounds │    │ - 课程学习    │    │ - Λ因子          │    │
│  │             │    │              │    │ - 可视化         │    │
│  └─────────────┘    └──────────────┘    └──────────────────┘    │
│                                                                   │
│  输出:                                                            │
│    - 训练好的模型检查点                                           │
│    - LER评估结果                                                  │
│    - 与MWPM的对比报告                                             │
│    - 可视化图表                                                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

典型使用：
```python
from alphaqubit.experiments import (
    MultiDistanceTrainer,
    MWPMBaseline,
    Evaluator,
    plot_ler_comparison,
)

# 1. 多码距训练
trainer = MultiDistanceTrainer(distances=[3, 5, 7])
trainer.train()

# 2. 评估
evaluator = Evaluator(model, distances=[3, 5, 7], noise_levels=[0.001, 0.005, 0.01])
results = evaluator.evaluate_all()

# 3. 基线比较
baseline = MWPMBaseline()
baseline_results = baseline.evaluate(distances=[3, 5, 7], noise_levels=[0.001, 0.005, 0.01])

# 4. 可视化
plot_ler_comparison(results, baseline_results)
```
"""

from .multi_distance import (
    MultiDistanceTrainer,
    MultiDistanceConfig,
    DistanceCurriculum,
)
from .baselines import (
    MWPMBaseline,
    BaselineResult,
)
from .evaluator import (
    Evaluator,
    EvaluationResult,
    EvaluationConfig,
)
from .visualize import (
    plot_ler_vs_noise,
    plot_ler_vs_distance,
    plot_lambda_factor,
    plot_comparison,
    create_summary_report,
)

__all__ = [
    # 多码距训练
    "MultiDistanceTrainer",
    "MultiDistanceConfig",
    "DistanceCurriculum",
    # 基线
    "MWPMBaseline",
    "BaselineResult",
    # 评估
    "Evaluator",
    "EvaluationResult",
    "EvaluationConfig",
    # 可视化
    "plot_ler_vs_noise",
    "plot_ler_vs_distance",
    "plot_lambda_factor",
    "plot_comparison",
    "create_summary_report",
]
