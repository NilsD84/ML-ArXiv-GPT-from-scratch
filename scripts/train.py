"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/small.yaml
"""
import argparse
from pathlib import Path

import yaml
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import make_dataloaders
from src.model.gpt import GPT
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    def best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    parser.add_argument("--device", default=best_device())
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    train_loader, val_loader = make_dataloaders(
        token_file=data_cfg["token_file"],
        seq_len=model_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        val_fraction=data_cfg.get("val_fraction", 0.05),
        num_workers=data_cfg.get("num_workers", 4),
    )

    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg.get("dropout", 0.1),
    )
    print(f"Model parameters: {model.num_params() / 1e6:.1f}M")

    trainer = Trainer(model, train_cfg, device=args.device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
