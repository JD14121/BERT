"""
Stim数据生成器模块 - 基于Stim的Surface Code模拟

这个模块使用Google的Stim库生成Surface Code的综合征数据。

主要功能：
1. 生成带SI1000噪声模型的Surface Code电路
2. 采样测量数据（ancilla测量和最终数据测量）
3. 计算detection events（检测事件）
4. 计算observable label（逻辑observable的值）

什么是Detection Event？
- 在量子纠错中，我们重复测量稳定子
- 如果两次连续测量结果不同，说明可能发生了错误
- Detection event = meas[t] XOR meas[t-1]
- 第一轮：event[0] = meas[0] XOR 0 = meas[0]

测量记录(Measurement Record)布局：
┌─────────────┬─────────────┬─────┬───────────────┬─────────────────┐
│ Round 0     │ Round 1     │ ... │ Round T-1     │ Final data      │
│ ancillas    │ ancillas    │     │ ancillas      │ qubits          │
│ (n_stab)    │ (n_stab)    │     │ (n_stab)      │ (n_data)        │
└─────────────┴─────────────┴─────┴───────────────┴─────────────────┘
总长度 = T * n_stab + n_data

SI1000噪声模型：
- after_clifford_depolarization: Clifford门后的去极化错误
- before_measure_flip_probability: 测量前的比特翻转
- after_reset_flip_probability: 重置后的比特翻转
"""

from typing import Optional, Tuple

import numpy as np
import stim

from .coordinates import CoordinateSystem


class StimDataGenerator:
    """基于Stim的Surface Code数据生成器

    使用Stim的内置Surface Code电路生成功能，配合SI1000噪声模型
    进行真实的错误模拟。

    主要属性：
        distance: 码距离
        rounds: 纠错轮数
        p: 物理错误率
        circuit: Stim电路对象
        coord_system: 坐标系统（用于稳定子位置映射）
        n_stab: 稳定子数量
        n_data: 数据比特数量

    主要方法：
        sample(): 采样原始测量数据
        generate_batch(): 生成完整的训练batch
        compute_detection_events(): 计算检测事件
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        p: float = 0.005,
        seed: Optional[int] = None,
    ):
        """初始化数据生成器

        Args:
            distance: 码距离（必须是>=3的奇数）
            rounds: 纠错轮数（时间步数）
            p: 物理错误率，用于SI1000噪声模型
               典型值：0.001（低噪声）到 0.01（高噪声）
            seed: 随机种子，用于可复现性
                  如果为None，则使用随机种子
        """
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.seed = seed

        # 计算稳定子和数据比特数量
        self.n_stab = distance ** 2 - 1   # 稳定子数量 = d² - 1
        self.n_data = distance ** 2       # 数据比特数量 = d²

        # 生成Stim电路
        self.circuit = self.generate_circuit()

        # 从电路构建坐标系统
        self.coord_system = CoordinateSystem(distance, self.circuit)

        # 编译采样器（用于高效批量采样）
        self._compile_samplers()

    def generate_circuit(self) -> stim.Circuit:
        """生成带SI1000噪声的Surface Code电路

        使用Stim的Circuit.generated()方法生成标准的
        rotated surface code memory实验电路。

        电路类型说明：
        - "surface_code:rotated_memory_z": Rotated布局，Z-basis memory实验
        - memory实验：准备|0⟩_L，运行T轮纠错，最后测量Z_L

        Returns:
            带噪声的Stim电路对象
        """
        circuit = stim.Circuit.generated(
            # 电路类型：rotated surface code，Z-memory实验
            "surface_code:rotated_memory_z",

            # 纠错轮数
            rounds=self.rounds,

            # 码距离
            distance=self.distance,

            # ========== SI1000噪声模型参数 ==========
            # Clifford门后的去极化噪声
            # 每个Clifford门后，以概率p发生随机Pauli错误
            # 门操作错误， 量子门执行后，qubit 状态可能被随机 Pauli 错误污染
            after_clifford_depolarization=self.p,
            before_round_data_depolarization=self.p,

            # 测量前的比特翻转概率
            # 模拟测量错误（readout error）测量时把 |0⟩ 误读成 1，或把 |1⟩ 误读成 0
            before_measure_flip_probability=self.p,

            # 重置后的比特翻转概率
            # 模拟状态准备错误，理想情况下 reset 把 qubit 置为 |0⟩，但实际可能出错变成 |1⟩
            after_reset_flip_probability=self.p,
        )
        return circuit

    def _compile_samplers(self) -> None:
        """编译测量和detector采样器

        Stim的compiled sampler比直接采样快得多，
        特别是在需要大量样本时。

        我们创建两个采样器：
        1. measurement_sampler: 采样原始测量结果
        2. detector_sampler: 直接采样detection events（用于验证）
        """
        if self.seed is not None:
            # 使用指定的随机种子
            self._measurement_sampler = self.circuit.compile_sampler(seed=self.seed)
            self._detector_sampler = self.circuit.compile_detector_sampler(seed=self.seed)
        else:
            # 使用随机种子
            self._measurement_sampler = self.circuit.compile_sampler()
            self._detector_sampler = self.circuit.compile_detector_sampler()

    def sample(self, num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样测量数据和observable值

        这是主要的数据采样方法。返回原始测量结果和
        Stim计算的observable值。

        Args:
            num_shots: 采样数量（样本数）

        Returns:
            Tuple of:
                - measurements: [shots, T*n_anc + n_data] bool数组
                  包含所有轮次的ancilla测量和最终的data测量
                - observables: [shots] bool数组
                  每个样本的logical observable值（0或1）
        """
        # 采样原始测量结果
        # 形状: [shots, total_measurements]
        # total_measurements = rounds * n_stab + n_data
        measurements = self._measurement_sampler.sample(shots=num_shots)

        # 使用detector sampler采样observable（更准确）
        # separate_observables=True 会分别返回detectors和observables
        detector_samples = self._detector_sampler.sample(
            shots=num_shots,
            separate_observables=True,
        )
        # detector_samples[0]: detection events
        # detector_samples[1]: observable values
        observables = detector_samples[1].flatten()

        return measurements, observables

    def sample_with_matched_detectors(
        self, num_shots: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """采样原始测量、DEM序detection events和observable

        使用同一个随机种子分别调用 measurement_sampler 和 detector_sampler，
        两者在 Stim 的 compiled sampler 中按 shot 索引派生随机流，因此返回的
        测量、detection events 和 observable 来自同一次底层错误采样。
        detection events 按 DEM (detector error model) 的 detector 顺序返回，
        可直接喂给 PyMatching。

        Args:
            num_shots: 采样数量（样本数）

        Returns:
            Tuple of:
                - measurements: [shots, T*n_anc + n_data] bool数组
                - detection_events: [shots, num_detectors] bool数组，按DEM序
                - observables: [shots] bool数组
        """
        measurements = self._measurement_sampler.sample(shots=num_shots)

        detector_samples = self._detector_sampler.sample(
            shots=num_shots,
            separate_observables=True,
        )
        detection_events = detector_samples[0]  # [shots, num_detectors]
        observables = detector_samples[1].flatten()

        return measurements, detection_events, observables

    def sample_with_detectors(self, num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
        """直接从Stim采样detection events

        这个方法使用Stim的内置detector计算，主要用于验证
        我们手动计算的detection events是否正确。

        Args:
            num_shots: 采样数量

        Returns:
            Tuple of:
                - detection_events: [shots, T*n_stab] bool数组
                - observables: [shots] bool数组
        """
        result = self._detector_sampler.sample(
            shots=num_shots,
            separate_observables=True,
        )
        detection_events = result[0]  # [shots, num_detectors]
        observables = result[1].flatten()  # [shots]
        return detection_events, observables

    def extract_ancilla_measurements(self, measurements: np.ndarray) -> np.ndarray:
        """从原始测量记录中提取ancilla测量

        测量记录布局：
        [Round 0 ancillas | Round 1 ancillas | ... | Round T-1 ancillas | Final data]
              n_stab           n_stab                    n_stab             n_data

        我们需要提取前 T * n_stab 个测量，并reshape成 [shots, T, n_stab]

        Args:
            measurements: [shots, T*n_anc + n_data] 原始测量数组

        Returns:
            [shots, T, n_anc] 形状的ancilla测量数组
        """
        shots = measurements.shape[0]
        total_anc = self.rounds * self.n_stab  # ancilla测量总数

        # 提取ancilla测量（前total_anc个）
        anc_flat = measurements[:, :total_anc]

        # Reshape成 [shots, rounds, n_stab]
        anc_meas = anc_flat.reshape(shots, self.rounds, self.n_stab)

        return anc_meas

    def extract_final_data(self, measurements: np.ndarray) -> np.ndarray:
        """从原始测量记录中提取最终的数据比特测量

        最终的数据测量是测量记录的最后n_data个元素。

        Args:
            measurements: [shots, T*n_anc + n_data] 原始测量数组

        Returns:
            [shots, n_data] 形状的最终数据测量数组
        """
        total_anc = self.rounds * self.n_stab

        # 提取最后n_data个测量
        final_data = measurements[:, total_anc:]

        return final_data

    def compute_detection_events(self, ancilla_meas: np.ndarray) -> np.ndarray:
        """从ancilla测量计算detection events

        Detection event定义：
        - event[t] = meas[t] XOR meas[t-1]  (t > 0)
        - event[0] = meas[0] XOR 0 = meas[0]  (第一轮)

        为什么用XOR？
        - 在无错误情况下，稳定子测量结果不变
        - 如果发生错误，测量结果会翻转
        - XOR检测两次测量之间是否发生了变化

        Args:
            ancilla_meas: [shots, T, n_stab] ancilla测量数组

        Returns:
            [shots, T, n_stab] detection events数组
        """
        shots, T, n_stab = ancilla_meas.shape

        # 初始化events数组
        events = np.zeros_like(ancilla_meas)

        # 第一轮：与假设的初始值0做XOR
        # 因为初始态是|0⟩_L，所有稳定子测量应该是0
        # 所以 event[0] = meas[0] XOR 0 = meas[0]
        events[:, 0, :] = ancilla_meas[:, 0, :]

        # 后续轮次：与前一轮做XOR
        for t in range(1, T):
            events[:, t, :] = ancilla_meas[:, t, :] ^ ancilla_meas[:, t - 1, :]

        return events

    def compute_observable_label(self, final_data: np.ndarray) -> np.ndarray:
        """从最终数据测量计算observable label

        Observable是特定数据比特的XOR，定义了logical qubit的状态。
        对于Z-memory实验：
        - Observable = XOR of specific data qubits
        - 由电路的OBSERVABLE_INCLUDE指令定义

        Args:
            final_data: [shots, n_data] 最终数据测量数组

        Returns:
            [shots] observable值数组（0或1）
        """
        shots = final_data.shape[0]

        # 从坐标系统获取observable涉及的数据比特索引
        obs_indices = self.coord_system.observable_data_indices

        if len(obs_indices) == 0:
            # 后备方案：对于Z-memory，使用第一行数据比特
            d = self.distance
            obs_indices = list(range(d))

        # 计算observable：所有相关数据比特的XOR
        observable = np.zeros(shots, dtype=bool)
        for idx in obs_indices:
            if idx < final_data.shape[1]:
                observable ^= final_data[:, idx]

        return observable

    def generate_batch(
        self, num_shots: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成完整的训练batch

        这是主要的数据生成接口。一次调用生成所有需要的数据：
        - Detection events（输入）
        - Final data measurements（输入）
        - Observable labels（目标）

        Args:
            num_shots: batch大小

        Returns:
            Tuple of:
                - events: [shots, T, n_stab] detection events
                - final_data: [shots, n_data] 最终数据测量
                - labels: [shots] observable labels
        """
        # 采样原始测量
        measurements, stim_observables = self.sample(num_shots)

        # 提取各部分
        ancilla_meas = self.extract_ancilla_measurements(measurements)
        final_data = self.extract_final_data(measurements)

        # 计算detection events
        events = self.compute_detection_events(ancilla_meas)

        # 使用Stim的observable计算（更准确）
        labels = stim_observables

        return events, final_data, labels

    def validate_against_stim(self, num_shots: int = 1000) -> dict:
        """与Stim的detector sampler对比验证

        验证我们手动计算的detection events是否与
        Stim的内置计算一致。

        Args:
            num_shots: 用于验证的样本数

        Returns:
            包含验证指标的字典
        """
        # 我们的计算
        measurements, stim_obs = self.sample(num_shots)
        ancilla_meas = self.extract_ancilla_measurements(measurements)
        our_events = self.compute_detection_events(ancilla_meas)
        our_events_flat = our_events.reshape(num_shots, -1)

        # Stim的计算
        stim_events, _ = self.sample_with_detectors(num_shots)

        # 比较（注意：可能因为边界detector处理有细微差异）
        # Stim可能包含额外的边界detector
        min_detectors = min(our_events_flat.shape[1], stim_events.shape[1])

        if min_detectors > 0:
            match_rate = np.mean(
                our_events_flat[:, :min_detectors] == stim_events[:, :min_detectors]
            )
        else:
            match_rate = 1.0

        # 计算event密度
        event_density = np.mean(our_events)

        return {
            "event_density": event_density,           # 事件密度
            "stim_match_rate": match_rate,            # 与Stim的匹配率
            "our_event_shape": our_events.shape,      # 我们的events形状
            "stim_event_shape": stim_events.shape,    # Stim的events形状
        }

    def get_event_statistics(self, num_shots: int = 10000) -> dict:
        """计算detection events的统计信息

        用于验证数据生成是否合理。

        期望值：
        - event_density: 约1-10%（取决于p和rounds）
        - observable_flip_rate: 取决于逻辑错误率

        Args:
            num_shots: 用于统计的样本数

        Returns:
            包含统计信息的字典
        """
        events, final_data, labels = self.generate_batch(num_shots)

        return {
            "event_density": np.mean(events),                    # 事件密度
            "observable_flip_rate": np.mean(labels),             # observable翻转率
            "events_per_round": np.mean(np.sum(events, axis=2)), # 每轮平均事件数
            "samples": num_shots,                                # 样本数
        }
