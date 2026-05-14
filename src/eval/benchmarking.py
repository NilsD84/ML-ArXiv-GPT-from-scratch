"""Throughput and latency benchmarking."""
import time
import torch
import torch.nn as nn


def benchmark_throughput(model: nn.Module, seq_len: int, batch_size: int, device: str = "cuda", n_iters: int = 20) -> dict:
    model.eval()
    dummy = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # warmup
    for _ in range(3):
        with torch.no_grad():
            model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0

    tokens_per_sec = (n_iters * batch_size * seq_len) / elapsed
    return {"tokens_per_sec": tokens_per_sec, "ms_per_iter": elapsed / n_iters * 1000}
