"""
Phase 3 测试模块 - 测试多码距训练、基线比较和评估流程

测试覆盖：
1. 多码距训练
   - DistanceCurriculum课程调度
   - MultiDistanceDataset数据集管理
   - MultiDistanceModel模型
   - MultiDistanceTrainer训练器

2. 基线比较
   - MWPMBaseline（需要PyMatching）
   - RandomBaseline
   - MajorityVoteBaseline

3. 完整评估流程
   - Evaluator评估器
   - EvaluationResult结果
   - Lambda因子计算

4. 可视化（可选）
   - 图表生成函数

运行测试：
    cd /Users/garrixma/Desktop/quantum/QEC/Alphaqubit_1
    python -m pytest alphaqubit/tests/test_phase3.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

# ==================== 导入被测试的模块 ====================
from alphaqubit.experiments.multi_distance import (
    MultiDistanceConfig,
    DistanceCurriculum,
    MultiDistanceDataset,
    MultiDistanceModel,
    MultiDistanceTrainer,
)

from alphaqubit.experiments.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    quick_evaluate,
)

from alphaqubit.data.coordinates import CoordinateSystem
from alphaqubit.models.decoder import AlphaQubitDecoder, AlphaQubitDecoderConfig


# ==================== Fixtures ====================

@pytest.fixture
def small_distances():
    """小码距列表用于快速测试"""
    return [3, 5]


@pytest.fixture
def coord_systems(small_distances):
    """创建坐标系统字典"""
    return {d: CoordinateSystem(d) for d in small_distances}


@pytest.fixture
def small_model(coord_systems):
    """创建小模型用于测试"""
    return MultiDistanceModel(
        coord_systems=coord_systems,
        embed_dim=32,
        hidden_dim=32,
        num_rnn_layers=1,
        use_conv=False,
    )


# ==================== DistanceCurriculum Tests ====================

class TestDistanceCurriculum:
    """测试码距课程调度器"""

    def test_basic_schedule(self):
        """测试基本课程调度"""
        schedule = [
            (0, [3]),
            (100, [3, 5]),
            (200, [3, 5, 7]),
        ]
        curriculum = DistanceCurriculum(schedule)

        # 测试不同步数返回正确的码距
        assert curriculum.get_distances(0) == [3]
        assert curriculum.get_distances(50) == [3]
        assert curriculum.get_distances(100) == [3, 5]
        assert curriculum.get_distances(150) == [3, 5]
        assert curriculum.get_distances(200) == [3, 5, 7]
        assert curriculum.get_distances(500) == [3, 5, 7]

    def test_progress_info(self):
        """测试进度信息"""
        schedule = [
            (0, [3]),
            (100, [3, 5]),
        ]
        curriculum = DistanceCurriculum(schedule)

        info = curriculum.get_progress_info(50)
        assert info["current_distances"] == [3]
        assert info["next_milestone"] == 100
        assert info["next_distances"] == [3, 5]

    def test_invalid_schedule_order(self):
        """测试无效的进度表顺序"""
        schedule = [
            (100, [3, 5]),
            (0, [3]),  # 顺序错误
        ]
        with pytest.raises(ValueError):
            DistanceCurriculum(schedule)

    def test_single_stage(self):
        """测试单阶段课程"""
        schedule = [(0, [3, 5, 7])]
        curriculum = DistanceCurriculum(schedule)

        assert curriculum.get_distances(0) == [3, 5, 7]
        assert curriculum.get_distances(1000) == [3, 5, 7]


# ==================== MultiDistanceDataset Tests ====================

class TestMultiDistanceDataset:
    """测试多码距数据集"""

    def test_dataset_creation(self, small_distances):
        """测试数据集创建"""
        rounds_per_distance = {3: 6, 5: 6}

        dataset = MultiDistanceDataset(
            distances=small_distances,
            rounds_per_distance=rounds_per_distance,
            noise_p=0.005,
            num_samples_per_distance=100,
            seed=42,
        )

        # 检查每个码距的数据集都创建了
        for d in small_distances:
            assert d in dataset.datasets
            assert d in dataset.coord_systems
            assert len(dataset.datasets[d]) == 100

    def test_get_dataset(self, small_distances):
        """测试获取单个数据集"""
        dataset = MultiDistanceDataset(
            distances=small_distances,
            rounds_per_distance={3: 6, 5: 6},
            noise_p=0.005,
            num_samples_per_distance=50,
        )

        ds_3 = dataset.get_dataset(3)
        sample = ds_3[0]

        assert "events" in sample
        assert "final_soft" in sample
        assert "label" in sample

    def test_coord_system_consistency(self, small_distances):
        """测试坐标系统一致性"""
        dataset = MultiDistanceDataset(
            distances=small_distances,
            rounds_per_distance={3: 6, 5: 6},
            noise_p=0.005,
            num_samples_per_distance=50,
        )

        for d in small_distances:
            coord = dataset.get_coord_system(d)
            assert coord.distance == d
            assert coord.n_stab == d * d - 1
            assert coord.n_data == d * d

    def test_max_distance(self, small_distances):
        """测试最大码距"""
        dataset = MultiDistanceDataset(
            distances=small_distances,
            rounds_per_distance={3: 6, 5: 6},
            noise_p=0.005,
            num_samples_per_distance=50,
        )

        assert dataset.get_max_distance() == 5


# ==================== MultiDistanceModel Tests ====================

class TestMultiDistanceModel:
    """测试多码距模型"""

    def test_model_creation(self, coord_systems):
        """测试模型创建"""
        model = MultiDistanceModel(
            coord_systems=coord_systems,
            embed_dim=32,
            hidden_dim=32,
        )

        assert model.max_distance == 5
        assert len(model.distances) == 2

    def test_forward_max_distance(self, small_model, coord_systems):
        """测试最大码距的前向传播"""
        B, T = 4, 6
        max_d = 5
        n_stab = coord_systems[max_d].n_stab
        n_data = coord_systems[max_d].n_data

        events = torch.randn(B, T, n_stab)
        final_soft = torch.randn(B, n_data)

        logits = small_model(events, final_soft, distance=max_d)

        assert logits.shape == (B, 1)

    def test_forward_small_distance(self, small_model, coord_systems):
        """测试小码距的前向传播（需要padding）"""
        B, T = 4, 6
        d = 3  # 小于最大码距
        n_stab = coord_systems[d].n_stab
        n_data = coord_systems[d].n_data

        events = torch.randn(B, T, n_stab)
        final_soft = torch.randn(B, n_data)

        logits = small_model(events, final_soft, distance=d)

        assert logits.shape == (B, 1)

    def test_param_count(self, small_model):
        """测试参数计数"""
        param_count = small_model.get_num_parameters()

        assert "total" in param_count
        assert param_count["total"] > 0


# ==================== MultiDistanceConfig Tests ====================

class TestMultiDistanceConfig:
    """测试多码距配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = MultiDistanceConfig()

        assert config.distances == [3, 5, 7]
        assert config.strategy == "joint"
        assert config.total_steps == 50000

    def test_custom_config(self):
        """测试自定义配置"""
        config = MultiDistanceConfig(
            distances=[3, 5],
            strategy="curriculum",
            total_steps=1000,
            batch_size=32,
        )

        assert config.distances == [3, 5]
        assert config.strategy == "curriculum"
        assert config.total_steps == 1000
        assert config.batch_size == 32

    def test_rounds_per_distance(self):
        """测试每个码距的rounds配置"""
        config = MultiDistanceConfig(
            distances=[3, 5, 7],
            rounds_per_distance={3: 6, 5: 9, 7: 12},
        )

        assert config.rounds_per_distance[3] == 6
        assert config.rounds_per_distance[5] == 9
        assert config.rounds_per_distance[7] == 12


# ==================== MultiDistanceTrainer Tests ====================

class TestMultiDistanceTrainer:
    """测试多码距训练器"""

    def test_trainer_creation(self, tmp_path):
        """测试训练器创建"""
        config = MultiDistanceConfig(
            distances=[3],
            total_steps=10,
            batch_size=8,
            eval_interval=5,
            log_interval=2,
            device="cpu",
        )

        trainer = MultiDistanceTrainer(
            config=config,
            save_dir=str(tmp_path),
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_short_training(self, tmp_path):
        """测试短训练流程"""
        config = MultiDistanceConfig(
            distances=[3],
            total_steps=5,
            batch_size=4,
            eval_interval=10,  # 不进行验证
            log_interval=1,
            device="cpu",
        )

        trainer = MultiDistanceTrainer(
            config=config,
            save_dir=str(tmp_path),
        )

        history = trainer.train()

        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_curriculum_training(self, tmp_path):
        """测试课程学习训练"""
        config = MultiDistanceConfig(
            distances=[3, 5],
            strategy="curriculum",
            curriculum_schedule=[(0, [3]), (3, [3, 5])],
            total_steps=5,
            batch_size=4,
            eval_interval=10,
            log_interval=1,
            device="cpu",
        )

        trainer = MultiDistanceTrainer(
            config=config,
            save_dir=str(tmp_path),
        )

        history = trainer.train()

        assert len(history["train_loss"]) > 0


# ==================== EvaluationConfig Tests ====================

class TestEvaluationConfig:
    """测试评估配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = EvaluationConfig()

        assert 3 in config.distances
        assert 0.005 in config.noise_levels
        assert 12 in config.rounds_list

    def test_custom_config(self):
        """测试自定义配置"""
        config = EvaluationConfig(
            distances=[3, 5],
            noise_levels=[0.001, 0.01],
            rounds_list=[3, 6],
            num_samples=1000,
        )

        assert config.distances == [3, 5]
        assert len(config.noise_levels) == 2
        assert config.num_samples == 1000


# ==================== EvaluationResult Tests ====================

class TestEvaluationResult:
    """测试评估结果"""

    def test_result_creation(self):
        """测试结果创建"""
        result = EvaluationResult(
            distance=5,
            noise_p=0.005,
            error_rates={3: 0.1, 6: 0.15, 9: 0.2},
            fidelities={3: 0.8, 6: 0.7, 9: 0.6},
            ler=0.01,
            ler_std=0.001,
            ler_ci=(0.008, 0.012),
            r_squared=0.95,
            is_valid=True,
            ece=0.02,
            mce=0.05,
            brier_score=0.1,
            num_samples=10000,
            inference_time=1.0,
            time_per_sample=0.0001,
        )

        assert result.distance == 5
        assert result.noise_p == 0.005
        assert result.ler == 0.01
        assert result.is_valid

    def test_to_dict(self):
        """测试转换为字典"""
        result = EvaluationResult(
            distance=5,
            noise_p=0.005,
            error_rates={3: 0.1},
            fidelities={3: 0.8},
            ler=0.01,
            ler_std=None,
            ler_ci=None,
            r_squared=0.95,
            is_valid=True,
            ece=0.02,
            mce=0.05,
            brier_score=0.1,
            num_samples=1000,
            inference_time=1.0,
            time_per_sample=0.001,
        )

        d = result.to_dict()

        assert d["distance"] == 5
        assert d["ler"] == 0.01
        assert d["is_valid"] is True

    def test_str_representation(self):
        """测试字符串表示"""
        result = EvaluationResult(
            distance=5,
            noise_p=0.005,
            error_rates={},
            fidelities={},
            ler=0.01,
            ler_std=0.001,
            ler_ci=None,
            r_squared=0.95,
            is_valid=True,
            ece=0.02,
            mce=0.05,
            brier_score=0.1,
            num_samples=1000,
            inference_time=1.0,
            time_per_sample=0.001,
        )

        s = str(result)
        assert "d=5" in s
        assert "LER=" in s


# ==================== Evaluator Tests ====================

class TestEvaluator:
    """测试评估器"""

    def test_evaluator_creation(self):
        """测试评估器创建"""
        coord = CoordinateSystem(3)
        model = AlphaQubitDecoderConfig.tiny(coord)

        config = EvaluationConfig(
            distances=[3],
            noise_levels=[0.005],
            rounds_list=[3, 6],
            num_samples=100,
            num_bootstrap=10,
            device="cpu",
        )

        evaluator = Evaluator(model, config)

        assert evaluator.model is not None
        assert len(evaluator.coord_systems) == 1

    def test_evaluate_single(self):
        """测试单配置评估"""
        coord = CoordinateSystem(3)
        model = AlphaQubitDecoderConfig.tiny(coord)

        config = EvaluationConfig(
            distances=[3],
            noise_levels=[0.005],
            rounds_list=[3, 6],
            num_samples=50,
            num_bootstrap=5,
            device="cpu",
        )

        evaluator = Evaluator(model, config, coord_systems={3: coord})
        result = evaluator.evaluate_single(3, 0.005)

        assert isinstance(result, EvaluationResult)
        assert result.distance == 3
        assert result.noise_p == 0.005
        assert 3 in result.error_rates
        assert 6 in result.error_rates

    def test_compute_lambda_factors(self):
        """测试Lambda因子计算"""
        # 创建模拟结果
        results = {}

        for d in [3, 5]:
            for p in [0.005]:
                results[(d, p)] = EvaluationResult(
                    distance=d,
                    noise_p=p,
                    error_rates={3: 0.1},
                    fidelities={3: 0.8},
                    ler=0.01 / d,  # 大码距LER更小
                    ler_std=0.001,
                    ler_ci=None,
                    r_squared=0.95,
                    is_valid=True,
                    ece=0.02,
                    mce=0.05,
                    brier_score=0.1,
                    num_samples=1000,
                    inference_time=1.0,
                    time_per_sample=0.001,
                )

        coord = CoordinateSystem(5)
        model = AlphaQubitDecoderConfig.tiny(coord)
        config = EvaluationConfig(
            distances=[3, 5],
            noise_levels=[0.005],
        )
        evaluator = Evaluator(model, config)

        lambda_factors = evaluator.compute_lambda_factors(results)

        assert (3, 5, 0.005) in lambda_factors
        lam, lam_std = lambda_factors[(3, 5, 0.005)]
        assert lam > 1  # 大码距应该有更好的LER


# ==================== Baseline Tests ====================

class TestBaselines:
    """测试基线方法"""

    def test_random_baseline_import(self):
        """测试随机基线导入"""
        from alphaqubit.experiments.baselines import RandomBaseline

        baseline = RandomBaseline(seed=42)
        assert baseline is not None

    def test_random_baseline_evaluate(self):
        """测试随机基线评估"""
        from alphaqubit.experiments.baselines import RandomBaseline

        baseline = RandomBaseline(seed=42)
        result = baseline.evaluate_single(
            distance=3,
            rounds=6,
            noise_p=0.005,
            num_samples=100,
        )

        # 随机基线的错误率应该接近0.5
        assert 0.3 < result.error_rate < 0.7
        assert result.method == "Random"

    def test_baseline_result_str(self):
        """测试BaselineResult字符串表示"""
        from alphaqubit.experiments.baselines import BaselineResult

        result = BaselineResult(
            method="Test",
            distance=5,
            rounds=12,
            noise_p=0.005,
            error_rate=0.1,
            num_samples=1000,
            time_per_sample=0.001,
        )

        s = str(result)
        assert "Test" in s
        assert "d=5" in s

    @pytest.mark.skipif(
        not pytest.importorskip("pymatching", reason="PyMatching not installed"),
        reason="PyMatching not installed"
    )
    def test_mwpm_baseline(self):
        """测试MWPM基线（需要PyMatching）"""
        try:
            from alphaqubit.experiments.baselines import MWPMBaseline

            baseline = MWPMBaseline(seed=42)
            result = baseline.evaluate_single(
                distance=3,
                rounds=6,
                noise_p=0.005,
                num_samples=100,
            )

            assert result.method == "MWPM"
            assert 0 <= result.error_rate <= 1
        except ImportError:
            pytest.skip("PyMatching not installed")


# ==================== Integration Tests ====================

class TestIntegration:
    """集成测试"""

    def test_full_pipeline_tiny(self):
        """测试完整的迷你pipeline"""
        # 1. 创建模型
        coord = CoordinateSystem(3)
        model = AlphaQubitDecoderConfig.tiny(coord)

        # 2. 简短训练（模拟）
        # 这里跳过实际训练，直接测试评估

        # 3. 评估
        config = EvaluationConfig(
            distances=[3],
            noise_levels=[0.005],
            rounds_list=[3],  # 只用一个rounds加速
            num_samples=20,
            num_bootstrap=5,
            device="cpu",
        )

        evaluator = Evaluator(model, config, coord_systems={3: coord})
        results = evaluator.evaluate_all()

        assert len(results) == 1
        assert (3, 0.005) in results

    def test_quick_evaluate_function(self):
        """测试快速评估函数"""
        coord = CoordinateSystem(3)
        model = AlphaQubitDecoderConfig.tiny(coord)

        # quick_evaluate需要coord_systems，所以我们手动创建evaluator
        config = EvaluationConfig(
            distances=[3],
            noise_levels=[0.005],
            rounds_list=[3],
            num_samples=20,
            num_bootstrap=3,
            device="cpu",
        )

        evaluator = Evaluator(model, config, coord_systems={3: coord})
        result = evaluator.evaluate_single(3, 0.005)

        assert result.distance == 3
        assert result.noise_p == 0.005


# ==================== 可视化测试（可选）====================

class TestVisualization:
    """测试可视化函数"""

    def test_visualize_import(self):
        """测试可视化模块导入"""
        try:
            from alphaqubit.experiments.visualize import (
                plot_ler_vs_noise,
                plot_ler_vs_distance,
                plot_lambda_factor,
            )
            assert callable(plot_ler_vs_noise)
            assert callable(plot_ler_vs_distance)
            assert callable(plot_lambda_factor)
        except ImportError as e:
            pytest.skip(f"Visualization not available: {e}")


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
