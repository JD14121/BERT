"""
Phase 2测试 - 模型、训练和评估模块测试

这个文件包含Phase 2实现的所有测试：
1. 模型组件测试（嵌入、RNN、卷积、融合）
2. 完整模型测试（前向传播、形状、梯度）
3. 训练组件测试（损失、调度器）
4. 评估指标测试（LER、Bootstrap、Calibration）
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from alphaqubit.data import SurfaceCodeDataset, CoordinateSystem
from alphaqubit.models import (
    SyndromeEmbedder,
    PositionalEmbedding,
    SoftDataEmbedder,
    ConvBlock,
    ScatterConvGather,
    StabToDataConv,
    LateFusion,
    ReadoutHead,
    AlphaQubitDecoder,
)
from alphaqubit.models.decoder import AlphaQubitDecoderConfig
from alphaqubit.training import (
    WarmupCosineScheduler,
    RoundsCurriculum,
    DecoderLoss,
    TrainingConfig,
)
from alphaqubit.evaluation import (
    compute_error_rate,
    compute_fidelity,
    fit_ler,
    compute_ler,
    compute_lambda,
    bootstrap_ler,
    compute_calibration_curve,
    expected_calibration_error,
)


# ==================== 测试数据准备 ====================

@pytest.fixture
def coord_system():
    """创建测试用坐标系统"""
    return CoordinateSystem(distance=3)


@pytest.fixture
def sample_batch(coord_system):
    """创建测试用batch"""
    B, T = 4, 10
    n_stab = coord_system.n_stab
    n_data = coord_system.n_data

    return {
        'events': torch.rand(B, T, n_stab),
        'final_soft': torch.rand(B, n_data),
        'label': torch.randint(0, 2, (B, 1)).float(),
    }


# ==================== 嵌入层测试 ====================

class TestEmbeddings:
    """嵌入层测试"""

    def test_syndrome_embedder_shape(self, coord_system, sample_batch):
        """测试SyndromeEmbedder输出形状"""
        embed_dim = 32
        embedder = SyndromeEmbedder(
            embed_dim=embed_dim,
            n_stab=coord_system.n_stab,
        )

        events = sample_batch['events']
        B, T, n_stab = events.shape

        output = embedder(events)

        assert output.shape == (B, T, n_stab, embed_dim)

    def test_syndrome_embedder_time_embedding(self, coord_system):
        """测试时间嵌入"""
        embedder = SyndromeEmbedder(
            embed_dim=32,
            n_stab=coord_system.n_stab,
            use_time_embedding=True,
        )

        events = torch.rand(2, 10, coord_system.n_stab)
        output = embedder(events)

        # 时间嵌入应该使不同时间步有不同的表示
        assert output.shape[1] == 10

    def test_positional_embedding(self):
        """测试位置编码"""
        embed_dim = 32
        pos_embed = PositionalEmbedding(embed_dim)

        # 创建归一化坐标
        positions = torch.rand(8, 2) * 2 - 1  # [-1, 1]

        output = pos_embed(positions)

        assert output.shape == (8, embed_dim)

    def test_soft_data_embedder(self, coord_system):
        """测试数据比特嵌入器"""
        embed_dim = 16
        embedder = SoftDataEmbedder(
            embed_dim=embed_dim,
            n_data=coord_system.n_data,
        )

        final_soft = torch.rand(4, coord_system.n_data)
        output = embedder(final_soft)

        assert output.shape == (4, coord_system.n_data, embed_dim)


# ==================== 卷积层测试 ====================

class TestConv:
    """卷积层测试"""

    def test_conv_block(self):
        """测试ConvBlock"""
        block = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
        )

        x = torch.rand(4, 32, 4, 4)
        output = block(x)

        assert output.shape == (4, 64, 4, 4)

    def test_scatter_conv_gather(self, coord_system):
        """测试ScatterConvGather"""
        D = 32
        n_stab = coord_system.n_stab

        block = ScatterConvGather(
            coord_system=coord_system,
            in_channels=D,
            out_channels=D,
            num_conv_layers=2,
        )

        x = torch.rand(4, n_stab, D)
        output = block(x)

        assert output.shape == (4, n_stab, D)


# ==================== 融合层测试 ====================

class TestFusion:
    """融合层测试"""

    def test_stab_to_data_conv(self, coord_system):
        """测试Stab-to-Data卷积"""
        D = 32
        n_stab = coord_system.n_stab
        n_data = coord_system.n_data

        conv = StabToDataConv(
            in_channels=D,
            out_channels=D,
            coord_system=coord_system,
        )

        stab_features = torch.rand(4, n_stab, D)
        output = conv(stab_features)

        assert output.shape == (4, n_data, D)

    @pytest.mark.parametrize("fusion_type", ["concat", "add"])
    def test_late_fusion(self, coord_system, fusion_type):
        """测试Late Fusion"""
        D = 32
        n_stab = coord_system.n_stab
        n_data = coord_system.n_data

        fusion = LateFusion(
            stab_dim=D,
            soft_dim=16,
            output_dim=D,
            coord_system=coord_system,
            fusion_type=fusion_type,
        )

        stab_features = torch.rand(4, n_stab, D)
        final_soft = torch.rand(4, n_data)

        output = fusion(stab_features, final_soft)

        assert output.shape == (4, n_data, D)


# ==================== 读出头测试 ====================

class TestReadout:
    """读出头测试"""

    def test_readout_head(self):
        """测试ReadoutHead"""
        D = 32
        n_data = 9

        head = ReadoutHead(
            input_dim=D,
            hidden_dim=64,
        )

        x = torch.rand(4, n_data, D)
        output = head(x)

        assert output.shape == (4, 1)


# ==================== 完整模型测试 ====================

class TestDecoder:
    """完整解码器测试"""

    def test_decoder_forward(self, coord_system, sample_batch):
        """测试解码器前向传播"""
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        events = sample_batch['events']
        final_soft = sample_batch['final_soft']

        output = model(events, final_soft)

        assert output.shape == (events.shape[0], 1)

    def test_decoder_predict(self, coord_system, sample_batch):
        """测试解码器预测"""
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        events = sample_batch['events']
        final_soft = sample_batch['final_soft']

        predictions, probs = model.predict(events, final_soft)

        assert predictions.shape == (events.shape[0],)
        assert probs.shape == (events.shape[0],)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_decoder_backward(self, coord_system, sample_batch):
        """测试解码器梯度计算"""
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        events = sample_batch['events']
        final_soft = sample_batch['final_soft']
        labels = sample_batch['label']

        output = model(events, final_soft)
        loss = nn.functional.binary_cross_entropy_with_logits(output, labels)
        loss.backward()

        # 检查梯度是否计算
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"

    def test_decoder_configs(self, coord_system):
        """测试不同配置的解码器"""
        tiny = AlphaQubitDecoderConfig.tiny(coord_system)
        small = AlphaQubitDecoderConfig.small(coord_system)
        base = AlphaQubitDecoderConfig.base(coord_system)

        # 检查参数量递增
        tiny_params = sum(p.numel() for p in tiny.parameters())
        small_params = sum(p.numel() for p in small.parameters())
        base_params = sum(p.numel() for p in base.parameters())

        assert tiny_params < small_params < base_params

    def test_decoder_parameter_count(self, coord_system):
        """测试参数量统计"""
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        stats = model.get_num_parameters()

        assert 'total' in stats
        assert stats['total'] > 0


# ==================== 训练组件测试 ====================

class TestTraining:
    """训练组件测试"""

    def test_warmup_cosine_scheduler(self):
        """测试Warmup Cosine调度器"""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Warmup阶段
        lrs_warmup = []
        for _ in range(100):
            lrs_warmup.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # 学习率应该递增
        assert lrs_warmup[-1] > lrs_warmup[0]

        # Decay阶段
        lrs_decay = []
        for _ in range(100):
            lrs_decay.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # 学习率应该递减
        assert lrs_decay[-1] < lrs_decay[0]

    def test_rounds_curriculum(self):
        """测试轮数课程"""
        curriculum = RoundsCurriculum()

        assert curriculum.get_rounds(0) == 3
        assert curriculum.get_rounds(5000) == 6
        assert curriculum.get_rounds(15000) == 12
        assert curriculum.get_rounds(30000) == 12

    def test_decoder_loss(self):
        """测试解码器损失"""
        loss_fn = DecoderLoss()

        logits = torch.randn(32, 1)
        labels = torch.randint(0, 2, (32, 1)).float()

        loss, metrics = loss_fn(logits, labels)

        assert loss.item() > 0
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_decoder_loss_label_smoothing(self):
        """测试标签平滑"""
        loss_fn = DecoderLoss(label_smoothing=0.1)

        logits = torch.randn(32, 1)
        labels = torch.randint(0, 2, (32, 1)).float()

        loss, _ = loss_fn(logits, labels)

        assert loss.item() > 0

    def test_initial_loss(self, coord_system):
        """测试随机初始化时的loss应接近ln(2)"""
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )
        model.eval()

        B, T = 100, 10
        events = torch.rand(B, T, coord_system.n_stab)
        final_soft = torch.rand(B, coord_system.n_data)
        labels = torch.randint(0, 2, (B, 1)).float()

        with torch.no_grad():
            logits = model(events, final_soft)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        # 随机预测时loss应接近ln(2) ≈ 0.693
        assert 0.5 < loss.item() < 0.9


# ==================== 评估指标测试 ====================

class TestEvaluation:
    """评估指标测试"""

    def test_compute_error_rate(self):
        """测试错误率计算"""
        preds = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])

        error_rate = compute_error_rate(preds, labels)

        assert error_rate == 0.2  # 1/5 错误

    def test_compute_fidelity(self):
        """测试fidelity计算"""
        # E = 0 → F = 1
        assert compute_fidelity(0.0) == 1.0

        # E = 0.5 → F = 0
        assert compute_fidelity(0.5) == 0.0

        # E = 0.25 → F = 0.5
        assert compute_fidelity(0.25) == 0.5

    def test_fit_ler(self):
        """测试LER拟合"""
        # 模拟理想的指数衰减数据
        rounds = [3, 6, 9, 12]
        ler_true = 0.01  # 真实LER

        fidelities = []
        for n in rounds:
            f = (1 - 2 * ler_true) ** n
            fidelities.append(f)

        ler_fitted, r_squared, _, _, is_valid = fit_ler(rounds, fidelities)

        # 拟合应该接近真实值
        assert abs(ler_fitted - ler_true) < 0.001
        assert r_squared > 0.99
        assert is_valid

    def test_compute_lambda(self):
        """测试抑错因子计算"""
        ler_d3 = 0.02
        ler_d5 = 0.01

        lam = compute_lambda(ler_d3, ler_d5)

        assert lam == 2.0  # 0.02 / 0.01


class TestBootstrap:
    """Bootstrap测试"""

    def test_bootstrap_ler(self):
        """测试Bootstrap LER估计"""
        # 创建模拟数据
        np.random.seed(42)
        rounds_list = [3, 6, 9, 12]

        preds_by_rounds = {}
        labels_by_rounds = {}

        for n in rounds_list:
            n_samples = 1000
            # 模拟一定的错误率
            error_rate = 0.1 + n * 0.01
            labels = np.random.randint(0, 2, n_samples)
            preds = labels.copy()
            # 引入一些错误
            flip_mask = np.random.rand(n_samples) < error_rate
            preds[flip_mask] = 1 - preds[flip_mask]

            preds_by_rounds[n] = preds
            labels_by_rounds[n] = labels

        result = bootstrap_ler(
            preds_by_rounds,
            labels_by_rounds,
            num_bootstrap=50,
            seed=42,
        )

        assert result.ler_mean > 0
        assert result.ler_std > 0
        assert result.valid_rate > 0


class TestCalibration:
    """校准测试"""

    def test_compute_calibration_curve(self):
        """测试校准曲线计算"""
        np.random.seed(42)

        # 模拟完美校准的数据
        probs = np.random.rand(1000)
        labels = (np.random.rand(1000) < probs).astype(float)

        confidences, accuracies, counts, _ = compute_calibration_curve(
            probs, labels, num_bins=10
        )

        assert len(confidences) == 10
        assert len(accuracies) == 10

    def test_expected_calibration_error(self):
        """测试ECE计算"""
        np.random.seed(42)

        # 完美校准的数据应该有低ECE
        probs = np.random.rand(1000)
        labels = (np.random.rand(1000) < probs).astype(float)

        ece = expected_calibration_error(probs, labels)

        assert ece < 0.1  # 接近完美校准


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_training_step(self, coord_system):
        """测试完整的训练步"""
        # 创建模型
        model = AlphaQubitDecoder(
            coord_system=coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 创建数据
        B, T = 8, 10
        events = torch.rand(B, T, coord_system.n_stab)
        final_soft = torch.rand(B, coord_system.n_data)
        labels = torch.randint(0, 2, (B, 1)).float()

        # 训练步
        model.train()
        optimizer.zero_grad()

        logits = model(events, final_soft)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 检查loss有效
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_dataset_model_compatibility(self):
        """测试数据集和模型的兼容性"""
        # 创建数据集
        dataset = SurfaceCodeDataset(
            distance=3,
            rounds=10,
            num_samples=100,
        )

        # 创建模型
        model = AlphaQubitDecoder(
            coord_system=dataset.coord_system,
            embed_dim=32,
            hidden_dim=32,
        )

        # 获取batch
        batch = dataset.get_batch(8)

        # 前向传播
        logits = model(batch['events'], batch['final_soft'])

        assert logits.shape == (8, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
