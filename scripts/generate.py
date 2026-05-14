"""
Generate text from a trained GPT checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/small/best.pt \
                                --prompt "We propose a novel" \
                                --max_tokens 200 \
                                --temperature 0.8 \
                                --top_k 50
"""
import argparse
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gpt import GPT
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import load_tokenizer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", default="configs/small.yaml", help="Config used to train the model")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--prompt", default="We propose a novel", help="Text prompt to start generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower=more focused, higher=more random)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--n_samples", type=int, default=1, help="Number of independent samples to generate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Loading model from {args.checkpoint}...")
    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=0.0,  # no dropout at inference
    )
    ckpt = load_checkpoint(args.checkpoint, model)
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})")
    print(f"Params: {model.num_params() / 1e6:.1f}M | Device: {args.device}")
    print(f"Temperature: {args.temperature} | Top-k: {args.top_k}")
    print("-" * 60)

    bos_id = tokenizer.token_to_id("[BOS]")
    prompt_ids = tokenizer.encode(args.prompt).ids
    prompt_tensor = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=args.device)

    for i in range(args.n_samples):
        if args.n_samples > 1:
            print(f"\n--- Sample {i + 1} ---")

        with torch.no_grad():
            output = model.generate(
                prompt_tensor.clone(),
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        # Decode — skip the BOS token
        generated_ids = output[0].tolist()[1:]
        text = tokenizer.decode(generated_ids)
        print(f"\nPrompt: {args.prompt}")
        print(f"\nGenerated:\n{text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
