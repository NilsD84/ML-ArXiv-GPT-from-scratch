"""
Download CShorten/ML-ArXiv-Papers, filter to cs.LG + cs.CL,
train a BPE tokenizer, tokenize all texts, and save a binary token file.

Usage:
    python scripts/prepare_data.py --output_dir data/ --vocab_size 32000
"""
import argparse
import os
import numpy as np
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.tokenizer import train_bpe_tokenizer


def iter_texts(dataset) -> list[str]:
    texts = []
    for ex in dataset:
        abstract = ex.get("abstract", "") or ""
        title = ex.get("title", "") or ""
        text = (title + "\n" + abstract).strip()
        if text:
            texts.append(text)
    return texts


def tokenize_and_save(texts: list[str], tokenizer: Tokenizer, out_path: str) -> int:
    all_ids = []
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    for text in texts:
        enc = tokenizer.encode(text)
        all_ids.append(bos_id)
        all_ids.extend(enc.ids)
        all_ids.append(eos_id)

    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)
    return len(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split=args.split)
    print(f"Total examples before filter: {len(ds)}")

    # Dataset is already curated ML papers — no category field to filter on
    texts = iter_texts(ds)
    print(f"Texts extracted: {len(texts)}")

    tokenizer_path = str(out_dir / "tokenizer.json")
    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
    tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size, save_path=tokenizer_path)

    token_path = str(out_dir / "tokens_train.bin")
    print("Tokenizing and saving binary token file...")
    n_tokens = tokenize_and_save(texts, tokenizer, token_path)
    print(f"Saved {n_tokens:,} tokens to {token_path}")
    print(f"File size: {Path(token_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
