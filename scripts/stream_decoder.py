# stream_decoder.py
"""
流式解码器工厂
将XZZXAlphaQubitDecoder / XZZXFineTuneDecoder 包装为符合
stream_and_decode(chunk, chunk_idx) -> None 签名的回调函数。
"""
from __future__ import annotations
import gc
from typing import Optional

import numpy as np
import torch

from scripts.xzzx_decoder import XZZXAlphaQubitDecoder, XZZXFineTuneDecoder

def make_xzzx_decoder_fn(
    model: torch.nn.Module,
    device: str = "cuda",
    log_interval: int = 10,
    checkpoint_path: Optional[str] = None,
):
    """
    工厂函数：返回符合 stream_and_decode 回调签名的解码器函数。

    参数
    ----
    model           : 已实例化的 XZZXAlphaQubitDecoder 或 XZZXFineTuneDecoder
    device          : 推理设备
    log_interval    : 每隔多少 chunk 打印一次累积LER
    checkpoint_path : 可选，自动加载检查点权重

    返回
    ----
    decoder_fn(chunk: dict, chunk_idx: int) -> None
        chunk键：measurement, event, final_soft, label,detection_events[, leakage, event_leakage]
        函数内部不保存任何数据，只累积统计量。

    使用方法
    --------decoder_fn = make_xzzx_decoder_fn(model, device='cuda')
        stream_and_decode(..., decoder_fn=decoder_fn, ...)print(f"最终 LER = {decoder_fn.state['logical_error_rate']:.6f}")
    """
    #── 加载权重 ─────────────────────────────────────────────────────────────
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        #兼容 {'model_state_dict': ...} 和直接 state_dict 两种格式
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[decoder] loaded checkpoint: {checkpoint_path}"
              f"step={ckpt.get('global_step', 'unknown')}")

    model = model.to(device)
    model.eval()

    #跨chunk 累积状态（外部可通过 decoder_fn.state 访问）
    state = {
        "correct":0,
        "total":              0,
        "logical_error_rate": 1.0,
        "chunk_lrs":          [],# 每个 chunk 的 LER，可用于后续绘图
    }

    @torch.no_grad()
    def decoder_fn(chunk: dict, chunk_idx: int) -> None:
        """
        流式解码回调：1. 将 numpy chunk 转为 tensor并送至 device
          2. 调用 model.predict() 得到逻辑比特预测
          3. 与 label 比对，更新累积 LER
          4. 不保存任何张量到外部，函数退出后GC 可回收
        """
        B= chunk["label"].shape[0]
        rounds = chunk["measurement"].shape[1]
        n_stab = chunk["measurement"].shape[2]

        # ── 组装模型输入 ──────────────────────────────────────────────────────
        measurement = torch.from_numpy(chunk["measurement"]).to(device)# [B,T,n_stab]
        event       = torch.from_numpy(chunk["event"]).to(device)# [B,T,n_stab]
        final_soft  = torch.from_numpy(chunk["final_soft"]).to(device)   # [B,n_data]
        labels      = torch.from_numpy(chunk["label"]).bool()# [B],留CPU

        # leakage 可选（generate_dataset.py 的 chunk无此键）
        if "leakage" in chunk and chunk["leakage"] is not None:
            leakage = torch.from_numpy(
                chunk["leakage"].astype(np.float32)).to(device)          # [B,T,n_stab]else:
            leakage = torch.zeros(B, rounds, n_stab, device=device)

        if "event_leakage" in chunk and chunk["event_leakage"] is not None:
            event_leakage = torch.from_numpy(
                chunk["event_leakage"].astype(np.float32)).to(device)
        else:
            event_leakage = torch.zeros(B, rounds, n_stab, device=device)

        # ── 推理 ──────────────────────────────────────────────────────────────
        preds, _ = model.predict(measurement, event, leakage, event_leakage, final_soft)
        # preds: [B] bool tensor on device

        # ── 累积统计 ──────────────────────────────────────────────────────────
        correct= (preds.cpu() == labels).sum().item()
        chunk_lr  = 1.0 - correct / B

        state["correct"] += correct
        state["total"]   += B
        state["logical_error_rate"] =1.0 - state["correct"] / state["total"]
        state["chunk_lrs"].append(chunk_lr)

        # ── 日志 ──────────────────────────────────────────────────────────────
        if chunk_idx % log_interval == 0:
            print(
                f"    [xzzx_decoder] chunk={chunk_idx:4d}  "
                f"batch={B}  "
                f"chunk_LR={chunk_lr:.5f}  "
                f"running_LR={state['logical_error_rate']:.5f}  "
                f"total={state['total']:,}"
            )

        # ── 显式释放 GPU 张量 ────────────────────────────────────────────────
        del measurement, event, final_soft, leakage, event_leakage, preds
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    #把状态挂到函数上，方便外部在流式结束后读取最终LER
    decoder_fn.state = state
    return decoder_fn