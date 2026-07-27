# BERT 量子纠错解码器 -- 交付代码包

> 项目：AlphaQubit BERT 自监督量子纠错解码器
> 版本：2026-07-23 ｜ 覆盖 2026-07-14 至 2026-07-23 全部实验
> 模型规模：1.64M → 5.5M → 12M → 55M（四轮扩容）
> 码距：d3 / d5 / d7（Google 105Q 真机 XZZX 噪声）

---

## 目录结构

```
E:\BERT\
├── alphaqubit/                 # 核心库（模型 + 训练 + 评估 + 数据加载）
│   ├── models/                 #   模型架构（embeddings, rnn_core, transformer, fusion, readout, decoder, pretrain_decoder）
│   ├── training/               #   训练器（trainer, pretrain_trainer, losses, scheduler）
│   ├── evaluation/             #   评估（metrics, bootstrap, calibration）
│   ├── data/                   #   数据加载（pt_dataset, soft_readout, stim_generator, coordinates, ...）
│   ├── experiments/            #   实验框架（baselines, evaluator, multi_distance, visualize）
│   ├── configs/                #   配置（data_config）
│   └── tests/                  #   测试
├── scripts/                    # 实验脚本
│   ├── *.py                    #   数据生成 + 训练 + 评估（见下方详细列表）
│   ├── patches/                #   代码补丁（加速优化, focal loss, 梯度累积, ...）
│   ├── analysis/               #   分析脚本（注意力分析, 对称性验证, 绘图）
│   ├── launch/                 #   一键启动脚本（55M, 两阶段）
│   └── (根目录)                #   数据生成 + 训练 + 评估主脚本
├── config/                     # 配置文件
│   ├── path_config.py          #   路径配置（云端/本地适配）
│   └── data_config.py          #   数据配置
├── data_specs/                 # 数据规范
│   ├── synthetic_data_spec.md  #   合成数据形式规范
│   └── BEAT_MWPM_DESIGN.md     #   BEAT_MWPM 实验设计
├── reports/                    # 实验报告（分门别类）
│   ├── main/                   #   主报告
│   │   ├── EXPERIMENT_REPORT.md        # 全实验报告（§3.1-3.8 + fig1-12）
│   │   ├── FULL_EXPERIMENT_SUMMARY.md  # 四轮扩容汇总 + 55M 展望
│   │   └── ARCHITECTURE_TECHNICAL_DOC.md # AQ vs BERT 架构详解（Mermaid + matplotlib）
│   ├── plans/                  #   工程计划书
│   │   ├── BEAT_MWPM_DESIGN.md
│   │   ├── BEAT_MWPM_ENGINEERING_PLAN.md
│   │   ├── D7_LER_ENGINEERING_PLAN.md
│   │   ├── D7_FINETUNE_OPTIMIZATION_PLAN.md
│   │   ├── FOCAL_FINETUNE_ABLATION_PLAN.md
│   │   └── EXPERIMENT_PLAN.md
│   ├── cloud_d7/               #   云端 d7 实验报告
│   │   ├── CLOUD_D7_EXPERIMENT_REPORT.md
│   │   ├── COMPLETE_EXPERIMENT_REPORT.md
│   │   └── FINAL_EXPERIMENT_REPORT.md
│   ├── attention_analysis/     #   注意力分析
│   │   ├── ATTENTION_ANALYSIS_EXPERIMENT_REPORT.md
│   │   ├── ATTENTION_ANALYSIS_REPORT.md
│   │   └── *.json              #   5 个 correlation JSON + summary
│   ├── innovation/             #   创新方向
│   │   └── qecGPT_INSPIRATION_AND_BERT_INNOVATION.md
│   ├── data/                   #   数据报告
│   │   ├── DATA_README.md
│   │   ├── QA_REPORT.md
│   │   └── synthetic_data_spec.md
│   ├── design/                 #   设计文档
│   │   ├── V4_DESIGN.md
│   │   └── alphaqubit_design.md
│   └── figures/                #   全部图表（24 张 PNG）
├── checkpoints/                # 模型检查点（需另行拷贝，见下方说明）
└── README.md                   # 本文件
```

---

## 脚本详细列表

### scripts/（根目录）-- 数据生成 + 训练 + 评估

| 文件 | 作用 | 阶段 |
|---|---|---|
| `generate_google_paems_data.py` | Google XZZX 合成数据生成（stim + 噪声注入 + 软读出）| 数据生成 |
| `calibrate_to_google.py` | PAEMS 噪声参数校准到 Google 真机 | 数据生成 |
| `generate_manifest.py` | 生成数据清单（manifest）| 数据生成 |
| `validate_google_paems_data.py` | 数据 QC 校验 | 数据生成 |
| `verify_sameshot.py` | same-shot 一致性验证 | 数据生成 |
| `generate_paems_data.py` | PAEMS 合成数据生成（旋转表面码）| 数据生成 |
| `paems_noise_model.py` | PAEMS 噪声模型核心 | 数据生成 |
| `validate_paems_data.py` | PAEMS 数据 QC | 数据生成 |
| `prepare_google_real.py` | Google 真机 .b8 -> .pt 转换（多 patch 汇总）| 数据生成 |
| `bert_pretrain.py` | BERT 自监督预训练（MSM 掩码预测）| 预训练 |
| `run_experiment.py` | 三模型对比实验（AQ + BERT + MWPM + LER）| 训练+评估 |
| `eval_ler.py` | LER 评估（多轮错误率拟合）| 评估 |
| `compare_baseline_paems.py` | PAEMS 三阶段对比实验 | 训练 |
| `local_train_5m.py` | 本地 5.5M 模型训练（预训练 + 微调 + 评估）| 训练 |
| `two_stage.py` | 两阶段模态微调（Stage1 合成 -> Stage2 真机）| 训练 |
| `two_stage_55m.py` | 55M 两阶段微调（支持 CLI 配置 + 梯度累积）| 训练 |
| `symmetry_augment.py` | 对称增强实验 A（C2 180° 旋转 + 验证门）| 训练 |
| `compressed_npy_dataset.py` | 压缩 npy 数据集加载器（memmap）| 数据加载 |
| `single_npy_dataset.py` | 单 npy 文件数据集 | 数据加载 |
| `xzzx_coord.py` | XZZX 坐标系（grid=2d-1, 归一化坐标）| 工具 |
| `xzzx_decoder.py` | XZZX 解码器（XZZXAlphaQubitDecoder + XZZXFineTuneDecoder + _patch_late_fusion）| 模型 |
| `mixed_msm.py` | MixedStructuredMSM 掩码策略（40%随机+30%空间+30%时序）| 预训练 |
| `plot_results.py` | 单实验可视化 | 绘图 |
| `plot_report.py` | 实验报告绘图 | 绘图 |
| `plot_d7_ler.py` | d7 LER 绘图 | 绘图 |
| `plot_cloud_d7.py` | 云端 d7 绘图 | 绘图 |
| `plot_final.py` | 最终报告绘图 | 绘图 |

### scripts/patches/ -- 代码补丁

| 文件 | 作用 |
|---|---|
| `patch_acceleration.py` | TF32 + cudnn.benchmark + torch.compile + 梯度累积（55M 加速）|
| `patch_bert_focal.py` | run_experiment.py 加 focal loss + --start-from + --ft-suffix |
| `patch_eval_ler_suffix.py` | eval_ler.py 加 --ft-suffix |
| `patch_real_suffix.py` | run_experiment.py 加 --real-suffix（对称增强）|
| `patch_focal.py` | AQ 阶段 focal loss（早期版本）|
| `patch_early_stop.py` | 早停参数调整 |
| `patch_resume.py` | 预训练 checkpoint 续训 |
| `fix_bp.py` | bert_pretrain.py TF32 修复 |
| `fix_patch2.py` | pretrain_trainer.py 梯度累积修复 |
| `final_tf32_fix.py` | 全文件 TF32 补齐 |
| `run_experiment_patched.py` | 已 patch 的 run_experiment.py 完整版 |
| `eval_ler_patched.py` | 已 patch 的 eval_ler.py 完整版 |
| `compress_d7_npy.py` | d7 数据 Plan C 压缩（bitpack + uint8）|
| `compress_d7_planC.py` | d7 Plan C 压缩 v2（边压边删）|
| `compress_d5_planC.py` | d5 Plan C 压缩 |

### scripts/analysis/ -- 分析 + 绘图

| 文件 | 作用 |
|---|---|
| `analyze_attention_v3.py` | 注意力与 DEM 关联分析（忠实 patch 版）|
| `check_symmetry.py` | 对称群验证（C4/D4 布局检验）|
| `check_label_symmetry.py` | label 保持性验证（rot180 corr 检验）|
| `plot_architecture.py` | AQ vs BERT 架构对比图 |
| `plot_latest_results.py` | 最新结果绘图（d5 opt/focal, d7 focal）|
| `plot_d5_focal_modelcard.py` | d5 focal 模型卡片图 |
| `plot_cloud_d7.py` | 云端 d7 绘图 |
| `plot_final.py` | 最终报告绘图 |

### scripts/launch/ -- 一键启动

| 文件 | 作用 |
|---|---|
| `launch_55m.py` | 55M 全流程启动（预训练 -> 两阶段 -> eval_ler，支持 d5/d7 自动参数）|
| `launch_experiment_c.py` | 实验 C 两阶段启动 |

---

## alphaqubit 核心库

### models/（模型架构）
| 文件 | 关键类 | 作用 |
|---|---|---|
| `embeddings.py` | SyndromeEmbedder, FinalDataEmbedder | 综合征嵌入（4 信号+位置）+ 数据比特嵌入 |
| `rnn_core.py` | RNNCore | 时序递推核心（0.7 缩放 + Transformer）|
| `transformer.py` | SyndromeTransformer, MultiHeadSelfAttention, SpatialAttentionBias, GatedDenseBlock, DilatedConvBlock | 双向 Transformer（注意力+GeGLU+空洞卷积）|
| `fusion.py` | LateFusion, StabToDataConv | 晚期融合（stab->data 卷积 + soft 嵌入）|
| `readout.py` | FullReadoutNetwork, ResNetBlock, CycleEmbedding, LineMeanPool | 深层 ResNet 读出（scatter+conv+pool+16层ResNet）|
| `decoder.py` | AlphaQubitDecoder | AQ 完整解码器（全监督）|
| `pretrain_decoder.py` | PretrainDecoder, FineTuneDecoder | BERT 预训练 + 微调解码器 |
| `conv_block.py` | ConvBlock | 卷积残差块 |

### training/（训练）
| 文件 | 关键类 | 作用 |
|---|---|---|
| `trainer.py` | Trainer, TrainingConfig | 通用训练器（AMP, 早停, 梯度累积, cosine decay）|
| `pretrain_trainer.py` | PretrainTrainer, PretrainConfig | BERT 预训练器（mask accuracy 监控）|
| `losses.py` | DecoderLoss, BalancedBCELoss | 损失函数（BCE + Focal + label smoothing）|
| `scheduler.py` | CurriculumManager, DistanceCurriculum | 学习率调度 + 课程学习 |

### data/（数据）
| 文件 | 关键类 | 作用 |
|---|---|---|
| `pt_dataset.py` | PTBatchDataset | .pt 格式数据加载 |
| `soft_readout.py` | SoftReadoutSimulator | 软读出噪声模拟 |
| `stim_generator.py` | StimDataGenerator | Stim 标准表面码电路生成 |
| `coordinates.py` | CoordinateSystem | 坐标系（scatter/gather/to_2d）|

### evaluation/（评估）
| 文件 | 关键函数 | 作用 |
|---|---|---|
| `metrics.py` | compute_ler, LERResult | LER 计算（E(n)=½(1-(1-2ε)ⁿ) 拟合）|
| `bootstrap.py` | BootstrapResult | 自举置信区间 |

---

## 数据集说明（不在交付包内，需另行准备）

| 数据集 | 位置 | 格式 | 大小 |
|---|---|---|---|
| Google XZZX 合成（d3/d5）| E:\Code\...\google_paems_data\data\d{3,5}\ | .pt (800k train, r=10, snr=10) | ~2-6GB |
| Google XZZX 合成（d7）| 云端 /root/data/d7/npy_compressed/ | .npy (125M, Plan C 压缩) | ~125GB |
| Google 真机（d3/d5/d7）| E:\Code\...\google_paems_data\data\real_d{3,5,7}\ | .pt (硬读出, r=10) | ~1.4GB |
| PAEMS 合成（d3/d5）| D:\Code\...\PAEMS-data\v3\ | .pt (500k, r=25, snr=10) | ~12GB |
| Google 真机原始 | E:\Code\...\google_paems_data\Google-data\ | .b8 + .stim | ~12GB |

---

## 环境配置

### 本地（RTX 4070 SUPER 12GB）
```bash
# 训练环境
conda activate quantum_env  # 或直接用 /d/condapy/quantum_env/python
# 依赖: torch 2.7+cu128, stim 1.16, numpy 1.26, pymatching, scipy, matplotlib

# 数据生成环境
D:/anaconda/python.exe  # conda base, stim 1.16, numpy 1.26
```

### 云端（V100-SXM2-32GB）
```bash
/root/miniconda3/envs/quantum_env/python  # torch 2.6+cu124, stim 1.16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8
```

### 加速优化（已 patch 云端）
- TF32: `torch.set_float32_matmul_precision('high')` -- V100 matmul ~3×
- cudnn.benchmark -- 固定形状自动调优
- torch.compile -- ~10-30%
- 梯度累积 -- bs64×4 或 bs128×2 = 有效 bs256
- AMP 混合精度 + 8 DataLoader workers + np.memmap

---

## 快速开始

### 本地 5.5M 训练（d3）
```bash
cd scripts
# 1. 准备真机数据（仅需一次）
python prepare_google_real.py --distance 3 --rounds 10

# 2. 预训练
python local_train_5m.py --distance 3 --stage pretrain --steps 10000

# 3. 微调
python local_train_5m.py --distance 3 --stage finetune --steps 3000

# 4. 评估
python local_train_5m.py --distance 3 --stage eval
```

### 云端 55M 训练（d7）
```bash
# 一键全流程：预训练 30k 步 -> 两阶段微调 -> eval_ler
python scripts/launch/launch_55m.py --distance 7
# d5:
python scripts/launch/launch_55m.py --distance 5
```

### 数据生成
```bash
# Google XZZX 合成数据
python generate_google_paems_data.py --distance 7 --manifest --out-dir /root/data/d7

# Google 真机 .b8 -> .pt
python prepare_google_real.py --distance 3 --rounds 10
```

---

## 关键实验结果

| 码距 | MWPM acc | 1.64M | 5.5M | 12M | 12M 两阶段 | MWPM LER | 12M 两阶段 LER |
|---|---|---|---|---|---|---|---|
| d3 | 0.9125 | 0.9027 | 0.9190 | **0.9330** | - | 0.0109 | - |
| d5 | 0.9428 | 0.7980 | 0.8721 | 0.9058 | - | 0.0035 | - |
| d7 | 0.9702 | 0.7438 | 0.7782 | 0.8664 | **0.8810** | 0.0027 | **0.007975** |

**里程碑**：
- d3 BERT 12M 超 MWPM（+2.05pp）-- 主胜
- d7 两阶段 80k 突破 0.87 墙（+1.46pp, LER -41.6%）-- PASS
- 55M 正在云端训练中

---

## 工程铁律

1. 前期规划：工程计划书文档化
2. 审查组与代码组分离（独立 subagent），逐阶段审查门
3. quantum_env 训练，conda base 数据生成
4. 禁数据造假/不合理假设/工程简化走捷径
5. 透明披露所有简化

---

## 联系信息

- 工作空间：`D:\Code\LZai\Ai for QEC\Alpha-qubit\code\alphaquibit-main\alphaquibit-main`
- 自进化记忆：工作空间根 `SELF_MEMORY.md`
- Claude 记忆：`~/.claude/projects/.../memory/`
- 云端服务器（最新）：180.127.11.177:24112 root/aiNg4ahp

---

*交付日期：2026-07-23 ｜ 121 个 Python 文件 ｜ 20 份报告 ｜ 24 张图表*
