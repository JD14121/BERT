"""
坐标系统模块 - 稳定子到网格位置的映射

这个模块实现了Surface Code稳定子坐标系统，主要功能：
1. 解析Stim电路的detector坐标，获取稳定子在网格中的位置
2. 实现scatter操作：将稀疏的稳定子状态分散到稠密网格
3. 实现gather操作：从网格中收集稳定子位置的值
4. 支持2D reshape，便于CNN处理

为什么需要坐标系统？
- Surface Code的稳定子在物理上排列成特定的2D图案
- 神经网络（特别是CNN）需要规则的网格输入
- scatter/gather操作允许我们在稀疏表示和稠密表示之间转换

网格布局示例（d=3）：
    0 1 2 3
  0 . Z . .     '.' = padding（空位）
  1 . Z Z X     'Z' = Z稳定子
  2 Z Z X .     'X' = X稳定子
  3 . . Z .

n_stab = 8个稳定子嵌入到4×4=16的网格中
"""

from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from torch import Tensor


class CoordinateSystem:
    """Surface Code稳定子坐标系统

    将稳定子索引映射到(d+1)×(d+1)网格上的位置，
    实现scatter/gather操作用于神经网络处理。

    Rotated Surface Code的稳定子位置由Stim电路的DETECTOR坐标决定。

    主要属性：
        distance: 码距离
        n_stab: 稳定子数量 = d² - 1
        n_data: 数据比特数量 = d²
        grid_size: 网格边长 = d + 1
        scatter_idx: 稳定子索引到网格flat索引的映射
        grid_mask: 标记网格中哪些位置有稳定子
        stab_positions: 稳定子索引到(row, col)的映射字典

    主要方法：
        scatter: 稀疏 → 稠密（将稳定子状态分散到网格）
        gather: 稠密 → 稀疏（从网格收集稳定子状态）
        to_2d/from_2d: flat网格 ↔ 2D网格转换
    """

    def __init__(self, distance: int, circuit: Optional["stim.Circuit"] = None):
        """初始化坐标系统

        Args:
            distance: 码距离（必须是>=3的奇数）
            circuit: 可选的Stim电路，用于提取detector坐标
                     如果为None，使用默认的rotated surface code布局
        """
        self.distance = distance
        self.n_stab = distance ** 2 - 1      # 稳定子数量
        self.n_data = distance ** 2          # 数据比特数量
        self.grid_size = distance + 1        # 网格边长

        # 从电路解析坐标，或使用默认坐标
        if circuit is not None:
            self._parse_circuit_coordinates(circuit)
        else:
            self._build_default_coordinates()

        # 构建scatter/gather索引张量
        self._build_indices()

    def _parse_circuit_coordinates(self, circuit: "stim.Circuit") -> None:
        """从Stim电路提取detector坐标

        Stim电路中的DETECTOR指令包含坐标信息：
        - DETECTOR(x, y, t) rec[-1]
        - (x, y)是空间坐标，t是时间坐标（轮数）

        我们只需要空间坐标(x, y)来确定稳定子在网格中的位置。
        每个唯一的(x, y)对应一个稳定子。

        Args:
            circuit: Stim电路对象
        """
        import stim

        # 存储稳定子位置：稳定子索引 → (row, col)
        self.stab_positions: Dict[int, Tuple[int, int]] = {}

        # 存储observable涉及的数据比特索引
        self.observable_data_indices: List[int] = []
        self.data_positions: List[Tuple[int, int]] = []
        qubit_coords: Dict[int, Tuple[float, float]] = {}
        measurement_targets: List[int] = []

        # 备选：从 DETECTOR 指令解析到的 stabilizer 位置（顺序=首次出现）
        detector_positions: List[Tuple[int, int]] = []
        detector_seen: Set[Tuple[int, int]] = set()

        # 遍历电路中的所有指令
        for instruction in circuit.flattened():
            if instruction.name == "QUBIT_COORDS":
                coords = instruction.gate_args_copy()
                for t in instruction.targets_copy():
                    if t.is_qubit_target:
                        qubit_coords[t.value] = (coords[0], coords[1])
            elif instruction.name == "DETECTOR":
                # 获取detector的坐标参数
                coords = instruction.gate_args_copy()
                if len(coords) >= 2:
                    x, y = coords[0], coords[1]
                    # 将Stim坐标转换为网格坐标
                    row, col = self._stim_to_grid(x, y)
                    pos = (row, col)

                    if pos not in detector_seen:
                        detector_seen.add(pos)
                        detector_positions.append(pos)

            elif instruction.name == "OBSERVABLE_INCLUDE":
                # 解析observable涉及的数据比特
                # OBSERVABLE_INCLUDE指令告诉我们哪些测量结果构成logical observable
                targets = instruction.targets_copy()
                for t in targets:
                    if t.is_measurement_record_target:
                        self.observable_data_indices.append(t.value)
            elif instruction.name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
                for t in instruction.targets_copy():
                    if t.is_qubit_target:
                        measurement_targets.append(t.value)

        # ========= 优先使用“测量顺序”来确定 stabilizer 的索引顺序 =========
        # ancilla 测量顺序 == measurement record 顺序（每轮 n_stab 个）
        if measurement_targets and qubit_coords:
            ancilla_qubits = measurement_targets[:self.n_stab]
            stab_positions: List[Tuple[int, int]] = []
            missing = False
            for q in ancilla_qubits:
                if q not in qubit_coords:
                    missing = True
                    break
                x, y = qubit_coords[q]
                row, col = self._stim_to_grid(x, y)
                stab_positions.append((row, col))

            if (not missing) and len(stab_positions) == self.n_stab:
                # 确保位置唯一（防止异常电路导致重复）
                if len(set(stab_positions)) == self.n_stab:
                    self.stab_positions = {
                        idx: pos for idx, pos in enumerate(stab_positions)
                    }

        # 解析 final data 测量顺序，并映射到网格坐标
        if measurement_targets and qubit_coords:
            data_qubits = measurement_targets[-self.n_data:]
            data_positions: List[Tuple[int, int]] = []
            for q in data_qubits:
                if q in qubit_coords:
                    x, y = qubit_coords[q]
                    row, col = self._stim_to_grid(x, y)
                    data_positions.append((row, col))

            if len(data_positions) == self.n_data:
                self.data_positions = data_positions

        # ========= 如果测量顺序解析失败，回退到 detector 顺序 =========
        if len(self.stab_positions) < self.n_stab and detector_positions:
            for idx, pos in enumerate(detector_positions[:self.n_stab]):
                self.stab_positions[idx] = pos

        # 如果没有找到足够的detector/measurement，使用默认坐标
        if len(self.stab_positions) < self.n_stab:
            self._build_default_coordinates()

        # 如果 data 位置解析失败，使用默认布局
        if len(self.data_positions) != self.n_data:
            self._build_default_data_positions()

    def _build_default_coordinates(self) -> None:
        """构建默认的Rotated Surface Code坐标

        当没有提供Stim电路时，使用这个方法生成默认的稳定子位置。

        Rotated Surface Code的布局规则：
        - 网格大小为(d+1)×(d+1)
        - 四个角落没有稳定子
        - 稳定子位于(row+col)为偶数的位置（棋盘图案）
        """
        self.stab_positions: Dict[int, Tuple[int, int]] = {}
        self.observable_data_indices: List[int] = []

        d = self.distance
        stab_idx = 0

        # 遍历网格，按棋盘图案放置稳定子
        # Rotated Surface Code中，稳定子在(row+col)为偶数的位置
        for row in range(d + 1):
            for col in range(d + 1):
                # 跳过四个角落（没有稳定子）
                if (row, col) in [(0, 0), (0, d), (d, 0), (d, d)]:
                    continue

                # 检查是否是有效的稳定子位置
                # 在rotated布局中，(row+col)为奇数的位置是数据比特
                if (row + col) % 2 == 1:
                    continue

                # 添加稳定子位置
                if stab_idx < self.n_stab:
                    self.stab_positions[stab_idx] = (row, col)
                    stab_idx += 1

        # 如果还没有足够的稳定子，填充剩余位置
        if len(self.stab_positions) < self.n_stab:
            self._fill_remaining_positions()

        # 默认的observable：Z-memory对应第一行的数据比特
        self.observable_data_indices = list(range(d))
        self._build_default_data_positions()

    def _build_default_data_positions(self) -> None:
        """构建默认 data qubit 的网格位置（与final data测量顺序一致）"""
        d = self.distance
        self.data_positions = [(row, col) for row in range(d) for col in range(d)]

    def _fill_remaining_positions(self) -> None:
        """填充剩余的稳定子位置

        这是一个后备方法，用于处理默认布局不足以覆盖所有稳定子的情况。
        """
        d = self.distance
        used_positions = set(self.stab_positions.values())
        stab_idx = len(self.stab_positions)

        # 遍历所有可用位置
        for row in range(d + 1):
            for col in range(d + 1):
                if stab_idx >= self.n_stab:
                    break
                pos = (row, col)
                if pos not in used_positions:
                    # 跳过角落
                    if pos not in [(0, 0), (0, d), (d, 0), (d, d)]:
                        self.stab_positions[stab_idx] = pos
                        used_positions.add(pos)
                        stab_idx += 1

    def _stim_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """将Stim坐标转换为网格坐标

        Stim使用的坐标系统中，相邻的qubit间距为2。
        我们需要将其转换为网格索引（从0开始的整数）。

        转换公式：
            row = round(y / 2)
            col = round(x / 2)

        Args:
            x: Stim的x坐标
            y: Stim的y坐标

        Returns:
            (row, col) 网格位置
        """
        row = int(y // 2)
        col = int(x // 2)
        return row, col

    def _build_indices(self) -> None:
        """构建scatter/gather索引张量

        创建两个关键的索引结构：
        1. scatter_idx: [n_stab] 张量，稳定子索引 → 网格flat索引
        2. grid_mask: [(d+1)²] 布尔张量，标记哪些网格位置有稳定子

        这些索引使scatter/gather操作可以高效地使用PyTorch的
        scatter_和gather函数实现。
        """
        d = self.distance

        # scatter_idx: 稳定子索引 → 网格flat索引
        # flat索引计算公式: flat_idx = row * grid_size + col
        scatter_idx = torch.zeros(self.n_stab, dtype=torch.long)
        for stab_idx, (row, col) in self.stab_positions.items():
            flat_idx = row * self.grid_size + col
            scatter_idx[stab_idx] = flat_idx

        self.scatter_idx = scatter_idx

        # 稳定子位置张量（供 attention bias 使用）
        stab_pos_list = [self.stab_positions[i] for i in range(self.n_stab)]
        self.stab_positions_tensor = torch.tensor(stab_pos_list, dtype=torch.float32)

        # data qubit 在 (d+1)x(d+1) 网格中的 flat 索引（与final data测量顺序一致）
        data_grid_indices = torch.zeros(self.n_data, dtype=torch.long)
        for idx, (row, col) in enumerate(self.data_positions):
            data_grid_indices[idx] = row * self.grid_size + col
        self.data_grid_indices = data_grid_indices

        # grid_mask: 标记网格中哪些位置有稳定子
        # True表示该位置有稳定子，False表示padding
        grid_mask = torch.zeros((self.grid_size ** 2,), dtype=torch.bool)
        for row, col in self.stab_positions.values():
            flat_idx = row * self.grid_size + col
            grid_mask[flat_idx] = True

        self.grid_mask = grid_mask

        # 构建反向映射（用于gather，虽然实际上我们用scatter_idx就够了）
        self._gather_idx = torch.zeros((self.grid_size ** 2,), dtype=torch.long)
        for stab_idx, (row, col) in self.stab_positions.items():
            flat_idx = row * self.grid_size + col
            self._gather_idx[flat_idx] = stab_idx

    def scatter(self, states: Tensor) -> Tensor:
        """将稳定子状态分散到网格位置（稀疏 → 稠密）

        这是神经网络前向传播的关键操作。将n_stab个稳定子的状态
        放置到(d+1)²的网格中的对应位置，其余位置填充为0。

        可视化示例（d=3, n_stab=8）：
            输入: [B, 8, D] - 8个稳定子的D维特征
            输出: [B, 16, D] - 4×4网格的flat表示

            稳定子0 → 网格位置8
            稳定子1 → 网格位置5
            ...

        Args:
            states: 形状为[B, n_stab, D]或[B, n_stab]的张量
                    B = batch size
                    n_stab = 稳定子数量
                    D = 特征维度（可选）
                    B个batch，每个有n_stab个稳定子，每个D维

        Returns:
            形状为[B, (d+1)², D]或[B, (d+1)²]的张量
            非稳定子位置填充为0
        """
        # 处理没有特征维度的情况
        has_feature_dim = states.dim() == 3
        if not has_feature_dim:
            states = states.unsqueeze(-1)  # [B, n_stab] → [B, n_stab, 1]

        B, n_stab, D = states.shape
        assert n_stab == self.n_stab, f"期望{self.n_stab}个稳定子，实际{n_stab}个"

        # 初始化输出为全0（padding值）
        output = torch.zeros(
            B, self.grid_size ** 2, D,
            dtype=states.dtype, device=states.device
        )

        # 使用scatter_将states放到对应位置
        # scatter_(dim, index, src): output[i][index[i][j]][k] = src[i][j][k]
        scatter_idx = self.scatter_idx.to(states.device)
        # 扩展索引以匹配states的形状 [n_stab] → [B, n_stab, D]
        scatter_idx_expanded = scatter_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        output.scatter_(1, scatter_idx_expanded, states)

        # 如果输入没有特征维度，移除添加的维度
        if not has_feature_dim:
            output = output.squeeze(-1)

        return output

    def gather(self, lattice: Tensor) -> Tensor:
        """从网格位置收集稳定子状态（稠密 → 稀疏）

        这是scatter的逆操作。从(d+1)²的网格中提取n_stab个
        稳定子位置的值。

        Args:
            lattice: 形状为[B, (d+1)², D]或[B, (d+1)²]的张量

        Returns:
            形状为[B, n_stab, D]或[B, n_stab]的张量
        """
        # 处理没有特征维度的情况
        has_feature_dim = lattice.dim() == 3
        if not has_feature_dim:
            lattice = lattice.unsqueeze(-1)

        B, grid_flat, D = lattice.shape
        assert grid_flat == self.grid_size ** 2, \
            f"期望网格大小{self.grid_size ** 2}，实际{grid_flat}"

        # 使用gather从网格中提取稳定子位置的值
        # gather(dim, index): output[i][j][k] = input[i][index[i][j][k]][k]
        scatter_idx = self.scatter_idx.to(lattice.device)
        scatter_idx_expanded = scatter_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        output = lattice.gather(1, scatter_idx_expanded)

        if not has_feature_dim:
            output = output.squeeze(-1)

        return output

    def reset_padding(self, lattice: Tensor, pad_vec: Tensor) -> Tensor:
        """重置padding位置的值

        因为这里用的 Grid 只放 Stabilizer，不放 Data Qubit。 DAta Qubit只在最后一轮做测量，那么棋盘上就会出现空位。
        
        在某些神经网络层后，padding位置可能被污染。
        这个方法将padding位置重置为指定的值。

        Args:
            lattice: 形状为[B, (d+1)², D]的张量
            pad_vec: 形状为[D]的张量，指定padding值

        Returns:
            padding位置被重置后的张量
        """
        B, grid_flat, D = lattice.shape
        assert grid_flat == self.grid_size ** 2

        grid_mask = self.grid_mask.to(lattice.device)
        # 取反得到padding位置的mask
        padding_mask = ~grid_mask

        # 扩展mask和padding向量以进行广播
        padding_mask = padding_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)
        pad_vec = pad_vec.unsqueeze(0).unsqueeze(0).expand(B, grid_flat, -1)

        # 使用where操作：padding位置用pad_vec，其他位置保持原值
        output = lattice.clone()
        output = torch.where(padding_mask, pad_vec.to(lattice.dtype), output)

        return output

    def to_2d(self, lattice_flat: Tensor) -> Tensor:
        """将flat网格reshape为2D空间格式

        将[B, (d+1)², D]转换为[B, D, d+1, d+1]，
        便于使用2D卷积等操作。

        Args:
            lattice_flat: 形状为[B, (d+1)², D]的张量

        Returns:
            形状为[B, D, d+1, d+1]的张量（PyTorch图像格式：NCHW）
        """
        B, grid_flat, D = lattice_flat.shape
        assert grid_flat == self.grid_size ** 2

        # 先reshape为[B, H, W, D]，再转置为[B, D, H, W]
        lattice_2d = lattice_flat.view(B, self.grid_size, self.grid_size, D)
        lattice_2d = lattice_2d.permute(0, 3, 1, 2)  # NHWD → NDHW

        return lattice_2d

    def from_2d(self, lattice_2d: Tensor) -> Tensor:
        """将2D空间格式reshape为flat网格

        这是to_2d的逆操作。

        Args:
            lattice_2d: 形状为[B, D, d+1, d+1]的张量

        Returns:
            形状为[B, (d+1)², D]的张量
        """
        B, D, H, W = lattice_2d.shape
        assert H == W == self.grid_size

        # 先转置为[B, H, W, D]，再flatten为[B, H*W, D]
        lattice_flat = lattice_2d.permute(0, 2, 3, 1)  # NDHW → NHWD
        lattice_flat = lattice_flat.reshape(B, self.grid_size ** 2, D)

        return lattice_flat

    def validate(self) -> bool:
        """验证scatter/gather的round-trip正确性

        测试：original → scatter → gather → recovered
        应该有：recovered == original

        Returns:
            True 如果round-trip成功
        """
        # 创建随机测试数据
        B, D = 4, 8
        original = torch.randn(B, self.n_stab, D)

        # Round-trip测试
        scattered = self.scatter(original)
        recovered = self.gather(scattered)

        # 检查是否相等（使用allclose处理浮点误差）
        return torch.allclose(original, recovered)

    def visualize(self) -> None:
        """可视化网格布局

        打印ASCII图显示稳定子在网格中的位置。
        'X'表示X稳定子，'Z'表示Z稳定子，'.'表示空位。
        """
        d = self.distance

        # 初始化网格为空位
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 填入稳定子位置
        for stab_idx, (row, col) in self.stab_positions.items():
            # 使用简单启发式区分X和Z稳定子
            # 实际上应该从电路中解析，这里用位置奇偶性近似
            symbol = "X" if (row + col) % 4 == 0 else "Z"
            grid[row][col] = symbol

        # 打印网格
        print(f"Surface Code d={d} 网格布局:")
        print("  " + " ".join(str(c) for c in range(self.grid_size)))
        for row in range(self.grid_size):
            print(f"{row} " + " ".join(grid[row]))
        print(f"\n稳定子数 = {self.n_stab}, 网格大小 = {self.grid_size}×{self.grid_size}")
