"""
Load checkpoints at multiple training steps and generate text from each.
Shows how the model's language improves through training.

Usage:
    python scripts/sample_outputs.py --checkpoint_dir checkpoints/small \
                                      --prompt "We propose a novel attention mechanism"
"""
import argparse
import re
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gpt import GPT
from src.data.tokenizer import load_tokenizer
from src.training.checkpointing import load_checkpoint


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path: str, model_cfg: dict, device: str) -> tuple:
    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=0.0,
    )
    ckpt = load_checkpoint(ckpt_path, model)
    model = model.to(device)
    model.eval()
    return model, ckpt


def generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float, top_k: int, device: str) -> str:
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    prompt_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    generated_ids = out[0].tolist()[1:]
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/small")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--prompt", default="We propose a novel attention mechanism that")
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default=best_device())
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    tokenizer = load_tokenizer(args.tokenizer)
    ckpt_dir = Path(args.checkpoint_dir)

    # Collect all step checkpoints sorted by step number
    step_ckpts = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))
    )
    best_ckpt = ckpt_dir / "best.pt"

    checkpoints = step_ckpts
    if best_ckpt.exists():
        checkpoints = checkpoints + [best_ckpt]

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    print("=" * 65)
    print("MODEL PROGRESSION — Text Quality Over Training")
    print("=" * 65)
    print(f"Prompt: \"{args.prompt}\"\n")

    for ckpt_path in checkpoints:
        model, ckpt = load_model(str(ckpt_path), model_cfg, args.device)
        step = ckpt["step"]
        val_loss = ckpt.get("val_loss", float("nan"))

        label = f"BEST (step {step})" if ckpt_path.name == "best.pt" else f"Step {step:,}"
        print(f"{'─' * 65}")
        print(f"  {label}  |  val_loss: {val_loss:.4f}")
        print(f"{'─' * 65}")

        text = generate(model, tokenizer, args.prompt, args.max_tokens,
                        args.temperature, args.top_k, args.device)
        print(text)
        print()

    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
