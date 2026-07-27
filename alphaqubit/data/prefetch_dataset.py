"""
多进程预取数据集 - 支持在线数据生成的高性能数据加载

这个模块解决了Windows上DataLoader多进程无法pickle Stim sampler的问题。
通过在每个worker进程中独立创建Stim generator，实现真正的多进程在线数据生成。

设计思路：
┌─────────────────────────────────────────────────────────────┐
│                    Main Process (GPU)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Prefetch Queue (生成好的batch)          │    │
│  │    [batch1] [batch2] [batch3] [batch4] ...          │    │
│  └─────────────────────────────────────────────────────┘    │
│                         ↑                                    │
│         ┌───────────────┼───────────────┐                   │
│         ↑               ↑               ↑                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Worker 0 │    │ Worker 1 │    │ Worker 2 │  ...         │
│  │ (Stim)   │    │ (Stim)   │    │ (Stim)   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│     CPU Core        CPU Core        CPU Core                │
└─────────────────────────────────────────────────────────────┘

优点：
1. 完全绕过pickle问题 - 每个worker独立创建generator
2. 充分利用多核CPU - 9950X3D的16核可以并行生成
3. GPU不等待 - 预取队列保证数据随时可用
4. 在线生成 - 无限数据，无需预存储
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty
from typing import Dict, Optional, Tuple
import threading
import time
import numpy as np
import torch

# Windows需要这个
if hasattr(mp, 'set_start_method'):
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # 已经设置过了


def worker_process(
    worker_id: int,
    output_queue: Queue,
    stop_event,
    distance: int,
    rounds: int,
    p: float,
    use_soft_readout: bool,
    snr: float,
    t: float,
    batch_size: int,
    seed: int,
):
    """Worker进程函数 - 在独立进程中生成数据

    每个worker有自己的Stim generator，完全独立运行。
    生成的数据放入共享队列，供主进程使用。
    """
    # 在worker进程内部import，避免pickle问题
    from alphaqubit.data.stim_generator import StimDataGenerator
    from alphaqubit.data.soft_readout import SoftReadoutSimulator
    from alphaqubit.data.coordinates import CoordinateSystem

    # 设置随机种子（每个worker不同）
    worker_seed = seed + worker_id * 10000
    np.random.seed(worker_seed)

    # 创建该worker的Stim generator
    generator = StimDataGenerator(
        distance=distance,
        rounds=rounds,
        p=p,
        seed=worker_seed,
    )

    # 创建软读出模拟器（如果启用）
    soft_simulator = None
    if use_soft_readout:
        soft_simulator = SoftReadoutSimulator(snr=snr, t=t)

    # 获取坐标系统
    coord_system = generator.coord_system
    scatter_idx = coord_system.scatter_idx.numpy()

    # 持续生成数据直到收到停止信号
    while not stop_event.is_set():
        try:
            # 检查队列是否已满（避免内存溢出）
            if output_queue.qsize() > 16:  # 最多缓存16个batch（减少内存占用）
                time.sleep(0.01)
                continue

            # 生成一个batch的数据
            raw_measurements, stim_observables = generator.sample(batch_size)

            # 提取各部分
            ancilla_meas = generator.extract_ancilla_measurements(raw_measurements)
            final_data = generator.extract_final_data(raw_measurements)

            # 计算detection events
            events = generator.compute_detection_events(ancilla_meas)
            labels = stim_observables

            # 软读出转换
            if soft_simulator is not None:
                # 只采样一次I/Q噪声
                soft_meas = soft_simulator.simulate(ancilla_meas)

                # 从soft_meas计算soft events（不重新采样）
                shots, T, n_stab = soft_meas.shape
                soft_events = np.zeros_like(soft_meas)
                soft_events[:, 0, :] = soft_meas[:, 0, :]

                for t in range(1, T):
                    # 使用静态方法计算软XOR
                    soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                        soft_meas[:, t, :],
                        soft_meas[:, t - 1, :],
                    )

                final_data = soft_simulator.simulate(final_data.astype(bool))
                measurement = soft_meas
                event = soft_events
            else:
                measurement = ancilla_meas.astype(np.float32)
                event = events.astype(np.float32)
                final_data = final_data.astype(np.float32)

            # Leakage（模拟中为0）
            leakage = np.zeros_like(measurement)
            event_leakage = np.zeros_like(event)

            # 打包数据
            batch_data = {
                'measurement': measurement,
                'event': event,
                'leakage': leakage,
                'event_leakage': event_leakage,
                'final_soft': final_data,
                'label': labels.astype(np.float32),
                'scatter_idx': scatter_idx,
            }

            # 放入队列
            output_queue.put(batch_data, timeout=1.0)

        except Exception as e:
            if not stop_event.is_set():
                print(f"Worker {worker_id} error: {e}")
            continue


class PrefetchDataLoader:
    """多进程预取数据加载器

    使用多个后台进程并行生成数据，通过队列传递给主进程。
    完全绕过DataLoader的pickle限制。

    使用示例：
        loader = PrefetchDataLoader(
            distance=3,
            rounds=25,
            p=0.005,
            batch_size=2048,
            num_workers=8,  # 使用8个CPU核心
        )

        for batch in loader:
            # batch已经是GPU上的tensor
            loss = model(batch)
            ...

        loader.shutdown()
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        p: float = 0.005,
        use_soft_readout: bool = True,
        snr: float = 10.0,
        t: float = 0.01,
        batch_size: int = 512,
        num_workers: int = 4,
        num_batches: int = 1000,  # 每个epoch的batch数
        seed: int = 42,
        device: str = 'cuda',
        pin_memory: bool = True,
    ):
        """初始化预取数据加载器

        Args:
            distance: 码距离
            rounds: 纠错轮数
            p: 物理错误率
            use_soft_readout: 是否使用软读出
            snr: 信噪比
            t: 归一化测量时间
            batch_size: 每个batch的样本数
            num_workers: 后台worker进程数（建议设为CPU核心数的一半）
            num_batches: 每个epoch的batch数量
            seed: 随机种子
            device: 目标设备
            pin_memory: 是否使用pinned memory（GPU训练时推荐True）
        """
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.use_soft_readout = use_soft_readout
        self.snr = snr
        self.t = t
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pin_memory = pin_memory and self.device.type == 'cuda'

        # 进程管理
        self.workers = []
        self.output_queue = None
        self.stop_event = None
        self._started = False

        # 获取scatter_idx（主进程需要一份）
        from alphaqubit.data.stim_generator import StimDataGenerator
        temp_gen = StimDataGenerator(distance, rounds, p, seed=seed)
        self.scatter_idx = temp_gen.coord_system.scatter_idx.clone()
        del temp_gen

    def start(self):
        """启动后台worker进程"""
        if self._started:
            return

        # 创建队列和停止事件
        self.output_queue = Queue(maxsize=64)
        self.stop_event = mp.Event()

        # 启动worker进程
        for i in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    i,
                    self.output_queue,
                    self.stop_event,
                    self.distance,
                    self.rounds,
                    self.p,
                    self.use_soft_readout,
                    self.snr,
                    self.t,
                    self.batch_size,
                    self.seed,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        self._started = True

        # 等待第一个batch生成（预热）
        time.sleep(0.5)

    def shutdown(self):
        """关闭所有worker进程"""
        if not self._started:
            return

        self._started = False

        # 发送停止信号
        try:
            if self.stop_event is not None:
                self.stop_event.set()
        except:
            pass

        # 清空队列（防止worker阻塞在put上）
        try:
            if self.output_queue is not None:
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        break
        except:
            pass

        # 等待所有worker结束
        for p in self.workers:
            try:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=0.5)
            except:
                pass

        self.workers = []
        self.output_queue = None
        self.stop_event = None

    def __len__(self):
        """返回每个epoch的batch数"""
        return self.num_batches

    def __iter__(self):
        """迭代器接口"""
        if not self._started:
            self.start()

        for _ in range(self.num_batches):
            # 从队列获取数据
            try:
                batch_data = self.output_queue.get(timeout=30.0)
            except Empty:
                raise RuntimeError("Data generation timeout - workers may have crashed")

            # 转换为PyTorch tensor并移动到目标设备
            batch = {}
            for key, value in batch_data.items():
                if key == 'scatter_idx':
                    batch['stab_pos_idx'] = torch.from_numpy(value).long()
                elif key == 'label':
                    batch[key] = torch.from_numpy(value).unsqueeze(-1)
                else:
                    batch[key] = torch.from_numpy(value)

            # Pin memory for faster GPU transfer (skip stab_pos_idx)
            if self.pin_memory:
                for key in batch:
                    if key != 'stab_pos_idx':
                        batch[key] = batch[key].pin_memory()

            yield batch

    def __del__(self):
        """析构时确保关闭worker"""
        self.shutdown()


class PrefetchDataset:
    """封装PrefetchDataLoader，提供与SurfaceCodeDataset兼容的接口

    这个类可以直接替换SurfaceCodeDataset使用。
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        p: float = 0.005,
        use_soft_readout: bool = True,
        snr: float = 10.0,
        t: float = 0.01,
        num_samples: int = 100000,
        seed: int = 42,
        num_workers: int = 4,
    ):
        self.distance = distance
        self.rounds = rounds
        self.p = p
        self.use_soft_readout = use_soft_readout
        self.snr = snr
        self.t = t
        self.num_samples = num_samples
        self.seed = seed
        self.num_workers = num_workers

        # 计算派生量
        self.n_stab = distance ** 2 - 1
        self.n_data = distance ** 2

        # 获取坐标系统
        from alphaqubit.data.stim_generator import StimDataGenerator
        temp_gen = StimDataGenerator(distance, rounds, p, seed=seed)
        self.coord_system = temp_gen.coord_system
        del temp_gen

    def get_prefetch_loader(
        self,
        batch_size: int,
        device: str = 'cuda',
    ) -> PrefetchDataLoader:
        """创建预取数据加载器

        Args:
            batch_size: 批大小
            device: 目标设备

        Returns:
            PrefetchDataLoader实例
        """
        num_batches = self.num_samples // batch_size

        return PrefetchDataLoader(
            distance=self.distance,
            rounds=self.rounds,
            p=self.p,
            use_soft_readout=self.use_soft_readout,
            snr=self.snr,
            t=self.t,
            batch_size=batch_size,
            num_workers=self.num_workers,
            num_batches=num_batches,
            seed=self.seed,
            device=device,
        )

    def validate(self) -> Dict:
        """验证数据生成"""
        from alphaqubit.data.stim_generator import StimDataGenerator
        from alphaqubit.data.soft_readout import SoftReadoutSimulator

        generator = StimDataGenerator(
            self.distance, self.rounds, self.p, seed=self.seed
        )

        soft_simulator = None
        if self.use_soft_readout:
            soft_simulator = SoftReadoutSimulator(snr=self.snr, t=self.t)

        # 生成测试数据
        raw_measurements, labels = generator.sample(1000)
        ancilla_meas = generator.extract_ancilla_measurements(raw_measurements)
        events = generator.compute_detection_events(ancilla_meas)

        if soft_simulator:
            # 只采样一次I/Q噪声
            soft_meas = soft_simulator.simulate(ancilla_meas)

            # 从soft_meas计算soft events（不重新采样）
            shots, T, n_stab = soft_meas.shape
            soft_events = np.zeros_like(soft_meas)
            soft_events[:, 0, :] = soft_meas[:, 0, :]

            for t in range(1, T):
                soft_events[:, t, :] = SoftReadoutSimulator.compute_soft_event(
                    soft_meas[:, t, :],
                    soft_meas[:, t - 1, :],
                )

            event_density = float(np.mean(soft_events > 0.5))
        else:
            event_density = float(np.mean(events))

        return {
            'event_density': event_density,
            'scatter_gather_valid': self.coord_system.validate(),
        }
