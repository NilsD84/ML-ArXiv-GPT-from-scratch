# GPT Project

A from-scratch GPT trained on ML ArXiv papers (cs.LG + cs.CL).

## Repo structure

```
src/
  data/        tokenizer, dataset, dataloader
  model/       attention, transformer blocks, GPT
  training/    trainer loop, LR scheduler, checkpointing
  eval/        perplexity, throughput benchmarking
scripts/
  prepare_data.py   download + filter + tokenize → binary token file
  train.py          main training entry point
configs/
  small.yaml        ~85M params, single GPU
  medium.yaml       ~350M params
notebooks/          exploratory notebooks
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Prepare data (~few minutes)
python scripts/prepare_data.py --output_dir data/ --vocab_size 32000

# 2. Train
python scripts/train.py --config configs/small.yaml
```


## Design notes

- BPE tokenizer trained on the filtered corpus via HuggingFace `tokenizers`
- Tokens stored as `uint16` memmap — efficient, zero-copy loading
- `F.scaled_dot_product_attention` → uses Flash Attention when available
- Weight tying between embedding and LM head
- Cosine LR schedule with linear warmup
- Mixed-precision (AMP) + gradient clipping
