"""Training loop with gradient accumulation and optional W&B logging."""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .scheduler import cosine_with_warmup
from .checkpointing import save_checkpoint


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.step = 0
        self.accum_steps = config.get("grad_accum_steps", 1)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.1),
            betas=(0.9, 0.95),
        )
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=config.get("warmup_steps", 100),
            total_steps=config["total_steps"],
        )
        self._use_amp = device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Optional W&B logging
        self._wandb = None
        if config.get("wandb", False):
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "gpt-arxiv"),
                    name=config.get("wandb_run_name", None),
                    config=config,
                )
                self._wandb = wandb
                print("W&B logging enabled.")
            except ImportError:
                print("wandb not installed — run `pip install wandb` to enable logging.")

    def _accumulate_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single forward+backward without optimizer step. Returns unscaled loss."""
        x, y = x.to(self.device), y.to(self.device)
        with torch.cuda.amp.autocast(enabled=self._use_amp):
            logits = self.model(x)
            # Divide loss by accum_steps so gradients average over the full accumulation window
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / self.accum_steps
        self.scaler.scale(loss).backward()
        return loss.item() * self.accum_steps  # return un-divided loss for logging

    def train_step(self, batches: list[tuple]) -> float:
        """Run gradient accumulation over `batches`, then optimizer step."""
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for x, y in batches:
            total_loss += self._accumulate_step(x, y)

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        return total_loss / len(batches)

    @torch.no_grad()
    def eval_loss(self, loader: DataLoader, max_batches: int = 50) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(1, count)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        cfg = self.config
        ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        log_every = cfg.get("log_every", 50)
        eval_every = cfg.get("eval_every", 500)
        save_every = cfg.get("save_every", 1000)
        best_val = float("inf")

        effective_batch = cfg.get("batch_size", 8) * self.accum_steps
        print(f"Gradient accumulation: {self.accum_steps} steps "
              f"(effective batch size: {effective_batch})")

        self.model.train()
        t0 = time.time()
        accum_buffer = []

        for epoch in range(cfg.get("epochs", 1)):
            for x, y in train_loader:
                accum_buffer.append((x, y))

                # Only take an optimizer step when we have enough micro-batches
                if len(accum_buffer) < self.accum_steps:
                    continue

                loss = self.train_step(accum_buffer)
                accum_buffer = []

                if self.step % log_every == 0:
                    elapsed = time.time() - t0
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"step {self.step:6d} | loss {loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                    t0 = time.time()
                    if self._wandb:
                        self._wandb.log({"train/loss": loss, "train/lr": lr}, step=self.step)

                if self.step % eval_every == 0:
                    val_loss = self.eval_loss(val_loader)
                    print(f"  val_loss {val_loss:.4f}")
                    if self._wandb:
                        self._wandb.log({"val/loss": val_loss}, step=self.step)
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(str(ckpt_dir / "best.pt"), self.model, self.optimizer, self.step, val_loss)

                if self.step % save_every == 0:
                    save_checkpoint(str(ckpt_dir / f"step_{self.step:07d}.pt"), self.model, self.optimizer, self.step, loss)

                if self.step >= cfg["total_steps"]:
                    if self._wandb:
                        self._wandb.finish()
                    return
