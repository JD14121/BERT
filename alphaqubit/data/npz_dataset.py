"""
Disk-backed Dataset — Load pre-generated .npz data for fast training

This module provides a PyTorch Dataset that loads pre-generated training data
from a compressed .npz file (produced by scripts/generate_dataset.py).

Key advantages over online SurfaceCodeDataset:
1. No Stim CPU bottleneck — data loads from disk/SSD
2. Memory-mapped access — datasets larger than RAM work fine
3. Same output format — interchangeable with SurfaceCodeDataset

Usage:
    dataset = NPZDataset("data/d3_r12_n5000000.npz")
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
    model = AlphaQubitDecoderConfig.base(dataset.coord_system)
    ...
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .coordinates import CoordinateSystem


class NPZDataset(Dataset):
    """Load pre-generated Surface Code data from a .npz file.

    The .npz file is expected to contain:
        measurement:  [N, T, n_stab] float32  — soft readout probabilities
        event:        [N, T, n_stab] float32  — soft XOR events
        final_soft:   [N, n_data]    float32  — final data qubit measurements
        label:        [N]            float32  — observable value (0.0 or 1.0)
        distance:     scalar         int      — code distance
        rounds:       scalar         int      — number of rounds
        p:            scalar         float    — physical error rate
        snr:          scalar         float    — soft readout SNR

    leakage and event_leakage are NOT stored (always zero in simulation);
    they are reconstructed on-the-fly as zero tensors.
    """

    def __init__(
        self,
        npz_path: str,
        mmap_mode: Optional[str] = None,  # None = load into RAM (fast, no pickle issues)
    ):
        """Initialize the dataset.

        Args:
            npz_path: Path to the .npz file.
            mmap_mode: Memory-map mode for np.load.
                       'r' = read-only mmap (default, low RAM usage)
                       None = load everything into RAM (faster but uses more memory)
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        self.npz_path = npz_path

        # Load with optional memory mapping
        self._data = np.load(str(npz_path), mmap_mode=mmap_mode)

        # Read metadata
        self.distance = int(self._data['distance'])
        self.rounds = int(self._data['rounds'])
        self.p = float(self._data['p'])
        self.snr = float(self._data['snr'])

        # Derived quantities (same as SurfaceCodeDataset)
        self.n_stab = self.distance ** 2 - 1
        self.n_data = self.distance ** 2
        self.num_samples = len(self._data['label'])

        # Build coordinate system
        self._coord_system = CoordinateSystem(self.distance)

    @property
    def coord_system(self) -> CoordinateSystem:
        """Coordinate system for scatter/gather operations."""
        return self._coord_system

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a single sample.

        Returns dict matching SurfaceCodeDataset format:
            measurement, event, leakage, event_leakage, final_soft, label, stab_pos_idx
        """
        T = self.rounds

        # Load data slices from mmap'd arrays (or in-memory)
        # 使用 asarray 避免不必要的 copy；DataLoader 会自动 collate
        measurement = torch.from_numpy(
            np.asarray(self._data['measurement'][idx])
        )  # [T, n_stab]

        event = torch.from_numpy(
            np.asarray(self._data['event'][idx])
        )  # [T, n_stab]

        final_soft = torch.from_numpy(
            np.asarray(self._data['final_soft'][idx])
        )  # [n_data]

        label = torch.tensor(
            [self._data['label'][idx]], dtype=torch.float32
        )  # [1]

        # Reconstruct zero tensors (not stored to save space)
        leakage = torch.zeros(T, self.n_stab, dtype=torch.float32)
        event_leakage = torch.zeros(T, self.n_stab, dtype=torch.float32)

        return {
            'measurement': measurement,
            'event': event,
            'leakage': leakage,
            'event_leakage': event_leakage,
            'final_soft': final_soft,
            'label': label,
            'stab_pos_idx': self._coord_system.scatter_idx.clone(),
        }

    def get_batch(self, batch_size: int) -> Dict[str, Tensor]:
        """Get a random batch directly (for testing/debug)."""
        indices = np.random.choice(self.num_samples, batch_size, replace=False)
        batch_data = [self[i] for i in indices]

        return {
            'measurement': torch.stack([d['measurement'] for d in batch_data]),
            'event': torch.stack([d['event'] for d in batch_data]),
            'leakage': torch.stack([d['leakage'] for d in batch_data]),
            'event_leakage': torch.stack([d['event_leakage'] for d in batch_data]),
            'final_soft': torch.stack([d['final_soft'] for d in batch_data]),
            'label': torch.stack([d['label'] for d in batch_data]),
            'stab_pos_idx': self._coord_system.scatter_idx.clone(),
        }

    def validate(self) -> Dict[str, float]:
        """Compute basic statistics for data quality check."""
        # Sample a subset to avoid loading everything into RAM
        n_sample = min(5000, self.num_samples)
        indices = np.random.choice(self.num_samples, n_sample, replace=False)
        indices.sort()

        events_sample = self._data['event'][indices]
        labels_sample = self._data['label'][indices]
        final_sample = self._data['final_soft'][indices]

        return {
            'num_samples': self.num_samples,
            'distance': self.distance,
            'rounds': self.rounds,
            'p': self.p,
            'snr': self.snr,
            'n_stab': self.n_stab,
            'n_data': self.n_data,
            'event_density': float(np.mean(events_sample > 0.5)),
            'event_mean': float(np.mean(events_sample)),
            'final_soft_mean': float(np.mean(final_sample)),
            'label_flip_rate': float(np.mean(labels_sample)),
            'scatter_gather_valid': self._coord_system.validate(),
        }

    def close(self):
        """Close the memory-mapped file handle."""
        if hasattr(self._data, 'close'):
            self._data.close()

    def __del__(self):
        self.close()
