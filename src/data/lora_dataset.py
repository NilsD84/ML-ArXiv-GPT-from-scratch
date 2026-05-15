"""Dataset for LoRA fine-tuning on (abstract → JSON) pairs."""
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader


INSTRUCTION = (
    "Extract structured information from this machine learning abstract. "
    "Return only valid JSON with keys: main_contribution, methods, "
    "datasets_evaluated, baselines_compared, claimed_improvement, task_domain.\n\n"
    "Abstract:\n"
)


def format_example(abstract: str, output: dict) -> str:
    """Format one (abstract, JSON) pair into the instruction-following template."""
    return (
        f"[INST] {INSTRUCTION}{abstract.strip()} [/INST]\n"
        f"{json.dumps(output, indent=2)}"
    )


class LoRADataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512, val_fraction: float = 0.1, split: str = "train", seed: int = 42):
        examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        random.seed(seed)
        random.shuffle(examples)
        val_size = max(1, int(len(examples) * val_fraction))

        if split == "train":
            examples = examples[val_size:]
        else:
            examples = examples[:val_size]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []

        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")

        skipped = 0
        for ex in examples:
            text = format_example(ex["input"], ex["output"])
            ids = tokenizer.encode(text).ids
            ids = [bos_id] + ids + [eos_id]

            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                skipped += 1

            self.data.append(ids)

        if skipped:
            print(f"  [{split}] {skipped} examples truncated to {max_seq_len} tokens")
        print(f"  [{split}] {len(self.data)} examples loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.data[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_fn(batch: list, pad_id: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences in a batch to the same length."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_padded = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id) for x in xs])
    y_padded = torch.stack([torch.nn.functional.pad(y, (0, max_len - y.size(0)), value=-100) for y in ys])
    return x_padded, y_padded


def make_lora_dataloaders(jsonl_path: str, tokenizer, max_seq_len: int, batch_size: int, val_fraction: float = 0.1) -> tuple[DataLoader, DataLoader]:
    train_ds = LoRADataset(jsonl_path, tokenizer, max_seq_len, val_fraction, split="train")
    val_ds = LoRADataset(jsonl_path, tokenizer, max_seq_len, val_fraction, split="val")

    pad_id = tokenizer.token_to_id("[PAD]")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_id))
    return train_loader, val_loader
