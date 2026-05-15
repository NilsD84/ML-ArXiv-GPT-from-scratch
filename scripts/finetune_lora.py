"""
LoRA fine-tuning entry point.

Loads pretrained GPT, injects LoRA adapters, fine-tunes on
(abstract → JSON) pairs, saves adapter weights.

Usage:
    python scripts/finetune_lora.py --config configs/lora.yaml
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.gpt import GPT
from src.model.lora import inject_lora, save_lora_weights, count_trainable_params, merge_lora
from src.data.lora_dataset import make_lora_dataloaders
from src.data.tokenizer import load_tokenizer
from src.training.checkpointing import load_checkpoint
from src.training.scheduler import cosine_with_warmup


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_loss(model, loader, device, max_batches=30) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # Ignore padding (-100) in loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            total += loss.item()
            count += 1
    model.train()
    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--device", default=best_device())
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    print("=" * 60)
    print("LoRA Fine-Tuning — Structured Abstract Extraction")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model from {cfg['base_checkpoint']}...")
    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    load_checkpoint(cfg["base_checkpoint"], model)

    # Inject LoRA
    print(f"Injecting LoRA (rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']})...")
    model = inject_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"])
    model = model.to(args.device)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Load tokenizer and data
    print(f"\nLoading tokenizer and data from {data_cfg['train_file']}...")
    tokenizer = load_tokenizer("data/tokenizer.json")
    train_loader, val_loader = make_lora_dataloaders(
        jsonl_path=data_cfg["train_file"],
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        val_fraction=data_cfg["val_fraction"],
    )

    # Optimizer — only trainable (LoRA) parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )
    scheduler = cosine_with_warmup(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 50),
        total_steps=train_cfg["total_steps"],
    )

    # Optional W&B
    wandb = None
    if train_cfg.get("wandb", False):
        try:
            import wandb as wb
            wb.init(
                project=train_cfg.get("wandb_project", "gpt-arxiv"),
                name=train_cfg.get("wandb_run_name", "lora-run"),
                config=cfg,
            )
            wandb = wb
            print("W&B logging enabled.")
        except ImportError:
            print("wandb not installed — skipping.")

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/lora"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_every = train_cfg.get("log_every", 20)
    eval_every = train_cfg.get("eval_every", 100)
    save_every = train_cfg.get("save_every", 200)

    print(f"\nStarting fine-tuning for {train_cfg['total_steps']} steps...")
    print("-" * 60)

    model.train()
    step = 0
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(100):
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)

            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                t0 = time.time()
                if wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

            if step % eval_every == 0:
                val = eval_loss(model, val_loader, args.device)
                print(f"  val_loss {val:.4f}")
                if wandb:
                    wandb.log({"val/loss": val}, step=step)
                if val < best_val:
                    best_val = val
                    save_lora_weights(model, str(ckpt_dir / "best_lora.pt"))
                    print(f"  Saved best LoRA weights (val_loss={val:.4f})")

            if step % save_every == 0:
                save_lora_weights(model, str(ckpt_dir / f"lora_step_{step:05d}.pt"))

            if step >= train_cfg["total_steps"]:
                break
        if step >= train_cfg["total_steps"]:
            break

    print("\n" + "=" * 60)
    print(f"Fine-tuning complete. Best val loss: {best_val:.4f}")

    # Save merged model for clean inference
    print("Merging LoRA weights into base model...")
    model = model.cpu()
    merged = merge_lora(model)
    torch.save({"model_state": merged.state_dict(), "val_loss": best_val}, str(ckpt_dir / "merged_model.pt"))
    print(f"Merged model saved to {ckpt_dir}/merged_model.pt")

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
