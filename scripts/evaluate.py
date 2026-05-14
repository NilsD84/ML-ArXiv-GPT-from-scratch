"""
Evaluate a trained GPT checkpoint.
Computes perplexity on the validation set and generates sample texts.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/small/best.pt
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gpt import GPT
from src.data.dataloader import make_dataloaders
from src.data.tokenizer import load_tokenizer
from src.training.checkpointing import load_checkpoint


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_perplexity(model, loader, device, max_batches=200):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


SAMPLE_PROMPTS = [
    "We propose a novel",
    "In this paper, we introduce",
    "Recent advances in deep learning have",
    "Our experiments demonstrate that",
    "The attention mechanism allows",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--max_batches", type=int, default=200, help="Batches to use for perplexity (more = more accurate)")
    parser.add_argument("--gen_tokens", type=int, default=150, help="Tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    print("=" * 60)
    print("GPT EVALUATION")
    print("=" * 60)

    print(f"\nLoading model from {args.checkpoint}...")
    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=0.0,
    )
    ckpt = load_checkpoint(args.checkpoint, model)
    model = model.to(args.device)
    model.eval()

    print(f"Checkpoint: step {ckpt['step']} | val_loss during training: {ckpt['val_loss']:.4f}")
    print(f"Parameters: {model.num_params() / 1e6:.1f}M | Device: {args.device}")

    print(f"\nLoading validation data...")
    _, val_loader = make_dataloaders(
        token_file=data_cfg["token_file"],
        seq_len=model_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        val_fraction=data_cfg.get("val_fraction", 0.05),
        num_workers=2,
    )

    print(f"Computing perplexity on up to {args.max_batches} validation batches...")
    avg_loss, ppl = compute_perplexity(model, val_loader, args.device, args.max_batches)
    print(f"\n{'─' * 40}")
    print(f"  Validation loss:       {avg_loss:.4f}")
    print(f"  Perplexity:            {ppl:.2f}")
    print(f"{'─' * 40}")

    print(f"\n{'=' * 60}")
    print("TEXT GENERATION SAMPLES")
    print(f"  temperature={args.temperature} | top_k={args.top_k} | tokens={args.gen_tokens}")
    print(f"{'=' * 60}")

    tokenizer = load_tokenizer(args.tokenizer)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    for prompt in SAMPLE_PROMPTS:
        prompt_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=args.device)

        with torch.no_grad():
            output = model.generate(
                idx,
                max_new_tokens=args.gen_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        generated_ids = output[0].tolist()[1:]  # skip BOS
        # Stop at EOS if present
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]
        text = tokenizer.decode(generated_ids)

        print(f"\nPrompt: \"{prompt}\"")
        print(f"Output:\n{text}")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
