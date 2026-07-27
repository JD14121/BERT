"""
Torch 张量格式数据集 - PTBatchDataset 的改进版

直接使用 .pt 文件保存数据，避免 NPZ 的 numpy 转换开销。
"""

from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .coordinates import CoordinateSystem


class PTBatchDataset(Dataset):
    """Load pre-generated Surface Code data from a .pt file.

    The .pt file is expected to contain:
        measurement:  [N, T, n_stab] float32
        event:        [N, T, n_stab] float32
        final_soft:   [N, n_data]    float32
        label:        [N]            float32
        distance:     scalar         int
        rounds:       scalar         int
        p:            scalar         float
        snr:          scalar         float
        detection_events: [N, T, n_stab] float32 (optional, for MWPM)
    """

    def __init__(self, pt_path: str):
        """Initialize the dataset.

        Args:
            pt_path: Path to the .pt file.
        """
        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pt_path}")

        self.pt_path = pt_path

        # Load data
        self._data = torch.load(str(pt_path), map_location='cpu', weights_only=False)

        # Read metadata
        self.distance = int(self._data['distance'])
        self.rounds = int(self._data['rounds'])
        self.p = float(self._data['p'])
        self.snr = float(self._data['snr'])

        # Derived quantities
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
        """Get a single sample."""
        T = self.rounds

        measurement = self._data['measurement'][idx]  # [T, n_stab]
        event = self._data['event'][idx]              # [T, n_stab]
        final_soft = self._data['final_soft'][idx]    # [n_data]
        label = torch.tensor([self._data['label'][idx]], dtype=torch.float32)  # [1]

        # Reconstruct zero tensors
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
        """Get a random batch directly."""
        import numpy as np
        indices = np.random.choice(self.num_samples, batch_size, replace=False)

        return {
            'measurement': self._data['measurement'][indices],
            'event': self._data['event'][indices],
            'leakage': torch.zeros(batch_size, self.rounds, self.n_stab),
            'event_leakage': torch.zeros(batch_size, self.rounds, self.n_stab),
            'final_soft': self._data['final_soft'][indices],
            'label': self._data['label'][indices].unsqueeze(-1),
            'stab_pos_idx': self._coord_system.scatter_idx.clone(),
        }

    def get_detection_events(self, idx: int) -> Tensor:
        """Get detection events for a single sample (for MWPM)."""
        if 'detection_events' in self._data:
            return self._data['detection_events'][idx]
        return None

    def close(self):
        """Release data."""
        self._data = None

    def __del__(self):
        self.close()
