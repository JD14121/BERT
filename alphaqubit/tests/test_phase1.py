"""Phase 1 tests for AlphaQubit data generation pipeline."""

import numpy as np
import pytest
import torch

from alphaqubit.configs import DataConfig
from alphaqubit.data import (
    CoordinateSystem,
    SoftReadoutSimulator,
    StimDataGenerator,
    SurfaceCodeDataset,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.distance == 3
        assert config.rounds == 12
        assert config.n_stab == 8
        assert config.n_data == 9
        assert config.grid_size == 4

    def test_distance_5_config(self):
        """Test configuration for distance 5."""
        config = DataConfig(distance=5)
        assert config.n_stab == 24
        assert config.n_data == 25
        assert config.grid_size == 6

    def test_invalid_distance(self):
        """Test that invalid distance raises error."""
        with pytest.raises(AssertionError):
            DataConfig(distance=4)  # Even distance
        with pytest.raises(AssertionError):
            DataConfig(distance=1)  # Too small


class TestCoordinateSystem:
    """Tests for CoordinateSystem."""

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_scatter_gather_roundtrip(self, distance: int):
        """Test that scatter → gather is identity."""
        coord = CoordinateSystem(distance)

        # Create random test data
        B, D = 4, 8
        original = torch.randn(B, coord.n_stab, D)

        # Round-trip
        scattered = coord.scatter(original)
        recovered = coord.gather(scattered)

        # Check equality
        assert torch.allclose(original, recovered), "Scatter-gather round-trip failed"

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_scatter_gather_roundtrip_no_feature(self, distance: int):
        """Test scatter/gather without feature dimension."""
        coord = CoordinateSystem(distance)

        B = 4
        original = torch.randn(B, coord.n_stab)

        scattered = coord.scatter(original)
        recovered = coord.gather(scattered)

        assert torch.allclose(original, recovered)

    def test_grid_mask(self):
        """Test that grid_mask correctly marks stabilizer positions."""
        coord = CoordinateSystem(3)

        # Number of True values should equal n_stab
        assert coord.grid_mask.sum() == coord.n_stab

    def test_scatter_idx_range(self):
        """Test that scatter indices are in valid range."""
        coord = CoordinateSystem(5)

        max_idx = coord.grid_size ** 2 - 1
        assert coord.scatter_idx.max() <= max_idx
        assert coord.scatter_idx.min() >= 0

    def test_to_2d_from_2d_roundtrip(self):
        """Test 2D reshape operations."""
        coord = CoordinateSystem(3)

        B, D = 2, 4
        original = torch.randn(B, coord.grid_size ** 2, D)

        # Round-trip through 2D
        as_2d = coord.to_2d(original)
        recovered = coord.from_2d(as_2d)

        assert torch.allclose(original, recovered)
        assert as_2d.shape == (B, D, coord.grid_size, coord.grid_size)

    def test_reset_padding(self):
        """Test padding reset functionality."""
        coord = CoordinateSystem(3)

        B, D = 2, 4
        lattice = torch.ones(B, coord.grid_size ** 2, D)
        pad_vec = torch.zeros(D)

        result = coord.reset_padding(lattice, pad_vec)

        # Stabilizer positions should still be 1
        stab_values = coord.gather(result)
        assert torch.allclose(stab_values, torch.ones_like(stab_values))

        # Padding positions should be 0
        padding_mask = ~coord.grid_mask
        padding_values = result[:, padding_mask, :]
        assert torch.allclose(padding_values, torch.zeros_like(padding_values))

    def test_validate(self):
        """Test the built-in validation method."""
        coord = CoordinateSystem(3)
        assert coord.validate(), "Coordinate system validation failed"


class TestStimDataGenerator:
    """Tests for StimDataGenerator."""

    def test_circuit_generation(self):
        """Test that circuit is generated correctly."""
        gen = StimDataGenerator(distance=3, rounds=5, p=0.001)

        assert gen.circuit is not None
        assert gen.n_stab == 8
        assert gen.n_data == 9

    def test_sample_shapes(self):
        """Test that sample outputs have correct shapes."""
        gen = StimDataGenerator(distance=3, rounds=5, p=0.001)

        num_shots = 100
        measurements, observables = gen.sample(num_shots)

        expected_meas_len = gen.rounds * gen.n_stab + gen.n_data
        assert measurements.shape == (num_shots, expected_meas_len)
        assert observables.shape == (num_shots,)

    def test_extract_ancilla_measurements(self):
        """Test ancilla measurement extraction."""
        gen = StimDataGenerator(distance=3, rounds=5)

        measurements, _ = gen.sample(50)
        ancilla = gen.extract_ancilla_measurements(measurements)

        assert ancilla.shape == (50, gen.rounds, gen.n_stab)

    def test_extract_final_data(self):
        """Test final data extraction."""
        gen = StimDataGenerator(distance=3, rounds=5)

        measurements, _ = gen.sample(50)
        final_data = gen.extract_final_data(measurements)

        assert final_data.shape == (50, gen.n_data)

    def test_detection_events_shape(self):
        """Test detection event computation."""
        gen = StimDataGenerator(distance=3, rounds=5)

        measurements, _ = gen.sample(50)
        ancilla = gen.extract_ancilla_measurements(measurements)
        events = gen.compute_detection_events(ancilla)

        assert events.shape == ancilla.shape
        assert events.dtype == bool

    def test_event_density_range(self):
        """Test that event density is in expected range."""
        gen = StimDataGenerator(distance=3, rounds=12, p=0.005)

        stats = gen.get_event_statistics(num_shots=5000)
        event_density = stats["event_density"]

        # With p=0.005, event density should be roughly 1-10%
        assert 0.001 < event_density < 0.2, f"Event density {event_density} out of range"

    def test_observable_flip_rate(self):
        """Test that observable flip rate is reasonable."""
        gen = StimDataGenerator(distance=3, rounds=12, p=0.005)

        stats = gen.get_event_statistics(num_shots=5000)
        flip_rate = stats["observable_flip_rate"]

        # Should be between 0 and 1, typically low for small p
        assert 0 <= flip_rate <= 1

    def test_generate_batch(self):
        """Test complete batch generation."""
        gen = StimDataGenerator(distance=3, rounds=5)

        events, final_data, labels = gen.generate_batch(100)

        assert events.shape == (100, gen.rounds, gen.n_stab)
        assert final_data.shape == (100, gen.n_data)
        assert labels.shape == (100,)

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        gen1 = StimDataGenerator(distance=3, rounds=5, seed=42)
        gen2 = StimDataGenerator(distance=3, rounds=5, seed=42)

        events1, _, labels1 = gen1.generate_batch(100)
        events2, _, labels2 = gen2.generate_batch(100)

        assert np.array_equal(events1, events2)
        assert np.array_equal(labels1, labels2)


class TestSoftReadoutSimulator:
    """Tests for SoftReadoutSimulator."""

    def test_soft_output_range(self):
        """Test that soft outputs are in [0, 1]."""
        sim = SoftReadoutSimulator(snr=10.0, t=0.01)

        binary = np.random.randint(0, 2, size=(1000,)).astype(bool)
        soft = sim.simulate(binary)

        assert np.all(soft >= 0) and np.all(soft <= 1)

    def test_bimodal_distribution(self):
        """Test that soft readout produces bimodal distribution."""
        sim = SoftReadoutSimulator(snr=10.0, t=0.01)

        stats = sim.get_distribution_stats(num_samples=10000)

        # Mean for |0⟩ should be near 0, for |1⟩ near 1
        assert stats["soft_given_0_mean"] < 0.3
        assert stats["soft_given_1_mean"] > 0.7

        # There should be clear separation
        assert stats["separation"] > 0.5

    def test_low_snr_more_noise(self):
        """Test that low SNR gives more spread out distributions."""
        sim_high = SoftReadoutSimulator(snr=20.0, t=0.01)
        sim_low = SoftReadoutSimulator(snr=2.0, t=0.01)

        stats_high = sim_high.get_distribution_stats(num_samples=10000)
        stats_low = sim_low.get_distribution_stats(num_samples=10000)

        # Low SNR should have less separation
        assert stats_low["separation"] < stats_high["separation"]

    def test_soft_event_computation(self):
        """Test soft XOR computation."""
        # Hard case: 0 XOR 0 = 0, 0 XOR 1 = 1
        soft_0 = np.array([0.0, 0.0, 1.0, 1.0])
        soft_1 = np.array([0.0, 1.0, 0.0, 1.0])

        soft_event = SoftReadoutSimulator.compute_soft_event(soft_0, soft_1)

        expected = np.array([0.0, 1.0, 1.0, 0.0])
        assert np.allclose(soft_event, expected)

    def test_soft_event_probabilistic(self):
        """Test soft XOR with probabilistic inputs."""
        # P(0 XOR 0.5) should be ~0.5
        soft_curr = np.array([0.0, 0.5, 1.0])
        soft_prev = np.array([0.5, 0.5, 0.5])

        soft_event = SoftReadoutSimulator.compute_soft_event(soft_curr, soft_prev)

        assert np.allclose(soft_event, [0.5, 0.5, 0.5])

    def test_detection_events_shape(self):
        """Test soft detection events computation."""
        sim = SoftReadoutSimulator(snr=10.0, t=0.01)

        ancilla_meas = np.random.randint(0, 2, size=(50, 10, 8)).astype(bool)
        soft_events = sim.simulate_detection_events(ancilla_meas)

        assert soft_events.shape == ancilla_meas.shape
        assert soft_events.dtype == np.float32


class TestSurfaceCodeDataset:
    """Tests for SurfaceCodeDataset."""

    def test_dataset_length(self):
        """Test that dataset reports correct length."""
        ds = SurfaceCodeDataset(distance=3, rounds=5, num_samples=1000)
        assert len(ds) == 1000

    def test_getitem_shapes(self):
        """Test that __getitem__ returns correct shapes."""
        ds = SurfaceCodeDataset(distance=3, rounds=5, num_samples=100)

        sample = ds[0]

        assert sample["events"].shape == (ds.rounds, ds.n_stab)
        assert sample["final_soft"].shape == (ds.n_data,)
        assert sample["label"].shape == (1,)
        assert sample["stab_pos_idx"].shape == (ds.n_stab,)

    def test_getitem_dtypes(self):
        """Test that __getitem__ returns correct dtypes."""
        ds = SurfaceCodeDataset(distance=3, rounds=5)

        sample = ds[0]

        assert sample["events"].dtype == torch.float32
        assert sample["final_soft"].dtype == torch.float32
        assert sample["label"].dtype == torch.float32
        assert sample["stab_pos_idx"].dtype == torch.int64

    def test_soft_readout_enabled(self):
        """Test soft readout produces non-binary values."""
        ds = SurfaceCodeDataset(
            distance=3,
            rounds=5,
            use_soft_readout=True,
            snr=10.0,
        )

        sample = ds[0]
        events = sample["events"].numpy()

        # Soft readout should produce non-binary values
        unique_values = len(np.unique(events))
        assert unique_values > 2, "Soft readout should produce continuous values"

    def test_soft_readout_disabled(self):
        """Test hard readout produces binary values."""
        ds = SurfaceCodeDataset(
            distance=3,
            rounds=5,
            use_soft_readout=False,
        )

        sample = ds[0]
        events = sample["events"].numpy()

        # Hard readout should be 0 or 1
        assert set(np.unique(events)).issubset({0.0, 1.0})

    def test_get_batch(self):
        """Test batch generation."""
        ds = SurfaceCodeDataset(distance=3, rounds=5)

        batch = ds.get_batch(32)

        assert batch["events"].shape == (32, ds.rounds, ds.n_stab)
        assert batch["final_soft"].shape == (32, ds.n_data)
        assert batch["label"].shape == (32, 1)

    def test_dataloader(self):
        """Test DataLoader creation."""
        ds = SurfaceCodeDataset(distance=3, rounds=5, num_samples=100)

        loader = ds.get_dataloader(batch_size=16, num_workers=0)

        batch = next(iter(loader))

        assert batch["events"].shape[0] == 16
        assert batch["label"].shape == (16, 1)

    def test_validate(self):
        """Test validation method."""
        ds = SurfaceCodeDataset(distance=3, rounds=5)

        metrics = ds.validate()

        assert "event_density" in metrics
        assert "scatter_gather_valid" in metrics
        assert metrics["scatter_gather_valid"]

    @pytest.mark.parametrize("distance", [3, 5])
    def test_multiple_distances(self, distance: int):
        """Test dataset works for different distances."""
        ds = SurfaceCodeDataset(distance=distance, rounds=5, num_samples=100)

        sample = ds[0]

        expected_n_stab = distance ** 2 - 1
        expected_n_data = distance ** 2

        assert sample["events"].shape[1] == expected_n_stab
        assert sample["final_soft"].shape[0] == expected_n_data


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline(self):
        """Test complete data generation pipeline."""
        # Config
        distance = 3
        rounds = 10
        p = 0.005

        # Generate data
        ds = SurfaceCodeDataset(
            distance=distance,
            rounds=rounds,
            p=p,
            use_soft_readout=True,
            snr=10.0,
            num_samples=500,
        )

        # Validate
        metrics = ds.validate()
        print(f"\nValidation metrics: {metrics}")

        # Check key metrics
        assert 0.001 < metrics["event_density"] < 0.2
        assert metrics["scatter_gather_valid"]

        # Test scatter/gather with actual data
        batch = ds.get_batch(32)
        events = batch["events"]  # [B, T, n_stab]

        # Flatten time dimension
        B, T, n_stab = events.shape
        events_flat = events.view(B * T, n_stab)

        # Test scatter
        coord = ds.coord_system
        scattered = coord.scatter(events_flat.unsqueeze(-1))

        assert scattered.shape == (B * T, coord.grid_size ** 2, 1)

        # Test gather round-trip
        recovered = coord.gather(scattered)
        assert torch.allclose(events_flat.unsqueeze(-1), recovered)

    def test_dataloader_training_loop(self):
        """Test simulated training loop."""
        ds = SurfaceCodeDataset(
            distance=3,
            rounds=5,
            num_samples=256,
        )

        loader = ds.get_dataloader(batch_size=32, num_workers=0)

        total_samples = 0
        for batch in loader:
            total_samples += batch["events"].shape[0]

            # Simulate forward pass
            events = batch["events"]
            labels = batch["label"]

            assert events.dim() == 3  # [B, T, n_stab]
            assert labels.dim() == 2  # [B, 1]

        assert total_samples == 256


def test_visualization():
    """Test coordinate system visualization (manual inspection)."""
    coord = CoordinateSystem(3)
    coord.visualize()

    coord5 = CoordinateSystem(5)
    coord5.visualize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
