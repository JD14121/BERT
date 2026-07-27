"""compressed_npy_dataset.py: 读取方案C压缩后的 d7 数据 (bitpack + uint8)。
fork-safe (np.memmap), 支持 num_workers>0。
解压在 __getitem__ 中进行 (unpackbits + uint8->float32)。
"""
import numpy as np, torch, json
from pathlib import Path
from torch.utils.data import Dataset

class CompressedNpyDataset(Dataset):
    def __init__(self, comp_dir, distance=7, rounds=10, p=0.0, snr=10.0):
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.snr = snr
        self.n_stab = distance * distance - 1
        self.n_data = distance * distance
        d = Path(comp_dir)
        meta = json.load(open(d / "meta.json"))
        self.N = meta["N"]
        self.num_samples = self.N
        num_det = meta["num_det"]
        packed_det = num_det // 8

        # mmap 压缩文件
        self._det_packed = np.memmap(str(d / "detection_events_packed.npy"), dtype=np.uint8, mode="r", shape=(self.N, packed_det))
        self._label = np.memmap(str(d / "label.npy"), dtype=np.uint8, mode="r", shape=(self.N,))
        self._meas = np.memmap(str(d / "measurement.npy"), dtype=np.uint8, mode="r", shape=(self.N, rounds, self.n_stab))
        self._event = np.memmap(str(d / "event.npy"), dtype=np.uint8, mode="r", shape=(self.N, rounds, self.n_stab))
        self._final = np.memmap(str(d / "final_soft.npy"), dtype=np.uint8, mode="r", shape=(self.N, self.n_data))

        from alphaqubit.data.coordinates import CoordinateSystem
        self._coord_system = CoordinateSystem(distance)

    @property
    def coord_system(self):
        return self._coord_system

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.rounds
        # 解压二值字段
        det_packed = self._det_packed[idx]  # [60] uint8
        det = np.unpackbits(det_packed).astype(np.float32)  # [480] 0/1
        label = float(self._label[idx])  # 0 or 1
        # 解压软值字段 (uint8 -> float32 / 255)
        meas = self._meas[idx].astype(np.float32) / 255.0  # [10, 48]
        event = self._event[idx].astype(np.float32) / 255.0  # [10, 48]
        final = self._final[idx].astype(np.float32) / 255.0  # [49]

        return {
            "measurement": torch.from_numpy(meas),
            "event": torch.from_numpy(event),
            "final_soft": torch.from_numpy(final),
            "label": torch.tensor([label], dtype=torch.float32),
            "leakage": torch.zeros(T, self.n_stab, dtype=torch.float32),
            "event_leakage": torch.zeros(T, self.n_stab, dtype=torch.float32),
            "stab_pos_idx": self._coord_system.scatter_idx.clone(),
        }

    def get_detection_events(self, idx):
        det_packed = self._det_packed[idx]
        return torch.from_numpy(np.unpackbits(det_packed).astype(np.float32))


def load_compressed_npy(d, r, basis, data_dir):
    return CompressedNpyDataset(Path(data_dir) / f"d{d}" / "npy_compressed", distance=d, rounds=r)
