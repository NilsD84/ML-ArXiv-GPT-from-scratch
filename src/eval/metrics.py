"""Evaluation metrics."""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def perplexity(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)
