"""
LoRA (Low-Rank Adaptation) implementation.

Wraps nn.Linear with two small trainable matrices A and B.
During fine-tuning, the original weight W is frozen.
The adapter computes: output = W(x) + (B @ A)(x) * scale

Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank  # scaling factor applied to adapter output

        in_features = linear.in_features
        out_features = linear.out_features

        # Freeze the original weight
        self.weight = linear.weight
        self.weight.requires_grad = False
        self.bias = linear.bias  # usually None in our model

        # LoRA adapter matrices
        # A: initialized with Gaussian (provides initial variation)
        # B: initialized to zero (so adapter starts as identity — no change at step 0)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen linear
        base = nn.functional.linear(x, self.weight, self.bias)
        # LoRA adapter: dropout → A → B, scaled
        adapter = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scale
        return base + adapter

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base weight and return a plain nn.Linear.
        Used before saving a final merged checkpoint for clean inference."""
        merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scale
        linear = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        linear.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.clone())
        return linear


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> nn.Module:
    """
    Replace target linear layers in the GPT model with LoRALinear.
    Targets: qkv and proj in every CausalSelfAttention block.
    All other parameters are frozen.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into attention projections
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha, dropout=dropout)

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters back into base weights for clean inference."""
    for block in model.blocks:
        block.attn.qkv = block.attn.qkv.merge()
        block.attn.proj = block.attn.proj.merge()
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights — tiny file (~2MB)."""
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights into an already-injected model."""
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected[:5]}")
    return model
