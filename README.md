# GPT from Scratch — Trained on ML ArXiv Papers

A GPT language model built entirely from scratch in PyTorch, trained on 117,592 machine learning research paper abstracts. Built as a learning project to deeply understand transformer architecture, training pipelines, and modern ML engineering practices.

> *Built with AI-assisted development (Claude Code) as a learning project.*

---

## Architecture

```
Input tokens
     │
     ▼
Token Embedding [32000 × 512]
     +
Position Embedding [512 × 512]
     │
     ▼
┌─────────────────────────┐
│   Transformer Block ×6  │
│  ┌───────────────────┐  │
│  │   Layer Norm      │  │
│  │   Causal MHA      │  │  ← 8 heads, Flash Attention
│  │   (residual +)    │  │
│  ├───────────────────┤  │
│  │   Layer Norm      │  │
│  │   MLP (4× expand) │  │  ← GELU activation
│  │   (residual +)    │  │
│  └───────────────────┘  │
└─────────────────────────┘
     │
     ▼
Layer Norm → LM Head [512 × 32000]
     │
     ▼
Logits over 32,000 tokens
```

**Model size:** 35.5M parameters (weight-tied embeddings)

---

## Dataset

| Property | Value |
|----------|-------|
| Source | `CShorten/ML-ArXiv-Papers` (HuggingFace) |
| Papers | 117,592 titles + abstracts |
| Tokenizer | BPE, 32,000 vocab, trained on corpus |
| Total tokens | 27.8M |
| File size | 55.6 MB (uint16 binary memmap) |

---

## Training

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 3e-4 (cosine decay + 200 step warmup) |
| Batch size | 8 (MPS) / 64 with grad accumulation |
| Sequence length | 512 tokens |
| Steps | 20,000 |
| Hardware | Apple M-series (MPS) |
| Training time | ~3.7 hours |

---

## Results

| Step | Train Loss | Val Loss | Perplexity |
|------|-----------|----------|------------|
| 0 | ~10.4 | — | ~32,000 |
| 500 | ~5.6 | 5.61 | ~273 |
| 1,000 | ~5.1 | 5.15 | ~173 |
| 10,000 | ~3.9 | — | ~49 |
| 20,000 | TBD | TBD | TBD |

---

## Project Structure

```
gpt-project/
├── src/
│   ├── data/           tokenizer, dataset, dataloader
│   ├── model/          attention, transformer block, GPT
│   ├── training/       trainer, LR scheduler, checkpointing
│   └── eval/           perplexity, benchmarking
├── scripts/
│   ├── prepare_data.py download + tokenize + save binary
│   ├── train.py        training entry point
│   ├── evaluate.py     perplexity + text generation eval
│   ├── generate.py     interactive text generation
│   ├── sample_outputs.py  compare outputs across checkpoints
│   └── plot_loss.py    visualize training curves
├── configs/
│   ├── small.yaml      35M params (this run)
│   └── medium.yaml     ~350M params (cloud GPU)
└── notebooks/
    └── explore_tokenizer.ipynb
```

---

## Quickstart

```bash
git clone https://github.com/NilsD84/ML-ArXiv-GPT-from-scratch
cd ML-ArXiv-GPT-from-scratch
pip install -r requirements.txt

# 1. Prepare data
python scripts/prepare_data.py --output_dir data/ --vocab_size 32000

# 2. Train
python scripts/train.py --config configs/small.yaml

# 3. Generate text
python scripts/generate.py --checkpoint checkpoints/small/best.pt \
    --prompt "We propose a novel attention mechanism that"

# 4. Evaluate (perplexity + sample outputs)
python scripts/evaluate.py --checkpoint checkpoints/small/best.pt

# 5. Plot loss curve
python scripts/plot_loss.py

# 6. Resume interrupted training
python scripts/train.py --config configs/small.yaml \
    --resume checkpoints/small/best.pt
```

---

## Key Design Decisions

- **Flash Attention** via `F.scaled_dot_product_attention` — automatic on PyTorch 2.x
- **Weight tying** between token embedding and LM head — saves 16.4M parameters
- **uint16 memmap** for token storage — half the size of int32, zero-copy loading
- **Gradient accumulation** — simulates large batches without extra memory
- **Pre-norm architecture** (LayerNorm before attention/MLP) — more stable than post-norm
- **Cosine LR schedule** with linear warmup — standard for transformers

---

## Roadmap

- [x] BPE tokenizer + binary data pipeline
- [x] GPT model (attention, transformer, generation)
- [x] Training loop with AMP, grad clipping, checkpointing
- [x] MPS (Apple Silicon) support
- [x] Gradient accumulation
- [x] W&B logging support
- [x] Evaluation + generation scripts
- [ ] Loss curve visualization (post-training)
- [ ] LoRA fine-tuning
- [ ] Cloud GPU training (medium config)
- [ ] Rotary positional embeddings (RoPE)
