"""XZZXCoordinateSystem: XZZX 表面码坐标系（Google 真实拓扑）。

CoordinateSystem 默认旋转码 (d+1)² 网格 + _stim_to_grid(x//2,y//2) 对 Google XZZX
坐标（间距 1）会塌缩（d3: 8 stab -> 6 unique），回退错误旋转布局。

本类：grid_size=2d-1（XZZX 真实网格），坐标归一化 (x-min_x, y-min_y) 保证全唯一。
设 self.distance=2d-2 使 decoder 的 (distance+1)²=(2d-1)² 自动匹配，无需改 decoder。
n_stab=d²-1, n_data=d² 保持真值。scatter/gather/to_2d 等继承自 CoordinateSystem。
"""
from typing import Dict, List, Tuple, Optional
import torch
from alphaqubit.data.coordinates import CoordinateSystem


class XZZXCoordinateSystem(CoordinateSystem):
    def __init__(self, distance: int, circuit):
        # 先执行必要的局部属性设置
        self.code_distance = int(distance)
        self.distance = 2 * distance - 2
        self.n_stab = distance ** 2 - 1
        self.n_data = distance ** 2
        self.grid_size = 2 * distance - 1
        self._parse_xzzx_coords(circuit)
        # 再调用父类（父类 __init__ 若依赖上述属性，顺序在此）
        super().__init__()  # 或按父类签名传参# 父类 __init__ 已调用 _build_indices，此处无需重复调用

    def _parse_xzzx_coords(self, circuit) -> None:
        import stim
        qubit_coords: Dict[int, Tuple[float, float]] = {}
        measurement_targets: List[int] = []
        observable_data_indices: List[int] = []
        for inst in circuit.flattened():
            if inst.name == "QUBIT_COORDS":
                a = inst.gate_args_copy()
                for t in inst.targets_copy():
                    if t.is_qubit_target:
                        qubit_coords[t.value] = (a[0], a[1])
            elif inst.name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
                for t in inst.targets_copy():
                    if t.is_qubit_target:
                        measurement_targets.append(t.value)
            elif inst.name == "OBSERVABLE_INCLUDE":
                for t in inst.targets_copy():
                    if t.is_measurement_record_target:
                        observable_data_indices.append(t.value)

        ancilla = measurement_targets[:self.n_stab]
        data = measurement_targets[-self.n_data:]
        all_qs = [q for q in ancilla + data if q in qubit_coords]
        xs = [qubit_coords[q][0] for q in all_qs]
        ys = [qubit_coords[q][1] for q in all_qs]
        min_x, min_y = min(xs), min(ys)

        self.stab_positions: Dict[int, Tuple[int, int]] = {}
        stab_pos_list: List[Tuple[int, int]] = []
        for q in ancilla:
            x, y = qubit_coords[q]
            stab_pos_list.append((int(y - min_y), int(x - min_x)))   # (row, col)
        # 唯一性校验
        if len(set(stab_pos_list)) != self.n_stab:
            raise ValueError(f"XZZX stab 位置非唯一: {len(set(stab_pos_list))} != {self.n_stab}")
        for idx, pos in enumerate(stab_pos_list):
            self.stab_positions[idx] = pos

        self.data_positions: List[Tuple[int, int]] = []
        for q in data:
            x, y = qubit_coords[q]
            self.data_positions.append((int(y - min_y), int(x - min_x)))
        if len(self.data_positions) != self.n_data:
            raise ValueError(f"XZZX data 位置数 {len(self.data_positions)} != {self.n_data}")

        self.observable_data_indices = observable_data_indices
