# GPT Project — Full Documentation
**Date:** May 14, 2026  
**Author:** Nils Depner  
**Purpose:** Learning ML and software engineering fundamentals by building a GPT language model from scratch.

---

## Table of Contents

1. [What Are We Building and Why?](#1-what-are-we-building-and-why)
2. [The Big Picture: How a GPT Works](#2-the-big-picture-how-a-gpt-works)
3. [Project Structure Explained](#3-project-structure-explained)
4. [Phase 1: The Data Pipeline](#4-phase-1-the-data-pipeline)
   - [The Dataset](#41-the-dataset)
   - [Tokenization and BPE](#42-tokenization-and-bpe)
   - [Binary Token File](#43-binary-token-file)
   - [Dataset and DataLoader](#44-dataset-and-dataloader)
5. [Phase 2: The Model](#5-phase-2-the-model)
   - [Embeddings](#51-embeddings)
   - [Attention](#52-attention-the-heart-of-gpt)
   - [The Transformer Block](#53-the-transformer-block)
   - [The Full GPT](#54-the-full-gpt)
6. [Phase 3: Training](#6-phase-3-training)
   - [The Loss Function](#61-the-loss-function)
   - [Backpropagation](#62-backpropagation)
   - [The Optimizer](#63-the-optimizer-adamw)
   - [Learning Rate Schedule](#64-learning-rate-schedule)
   - [Checkpointing](#65-checkpointing)
   - [The Training Loop](#66-the-training-loop)
7. [Configuration System](#7-configuration-system)
8. [Key ML Concepts Glossary](#8-key-ml-concepts-glossary)
9. [What the Numbers Mean](#9-what-the-numbers-mean)
10. [What Happens Next](#10-what-happens-next)

---

## 1. What Are We Building and Why?

We are building a **GPT** (Generative Pre-trained Transformer) — the same fundamental architecture behind ChatGPT, Claude, and most modern AI assistants. Our version is much smaller, but the ideas are identical.

### What does a GPT actually do?

At its core, a GPT does one thing: **given some text, predict what word comes next**. That's it. If you give it:

> "The neural network learns to"

It tries to predict that the next word might be "recognize" or "classify" or "generalize". If you keep feeding it its own predictions, it can generate entire paragraphs.

This sounds simple, but the magic is that to predict the next word well, the model must actually *understand* the text — it must learn grammar, facts, reasoning patterns, and context. Prediction is just the training mechanism; understanding is what emerges.

### Why train on ML ArXiv papers?

The dataset (`CShorten/ML-ArXiv-Papers`) contains 117,592 titles and abstracts from machine learning research papers. We chose it because:
- It's a well-defined, clean domain (no noise from the entire internet)
- It's small enough to train on a laptop
- The outputs are interpretable — we can judge if generated text sounds like a real ML abstract

### Why build from scratch instead of using a library?

You could download a pretrained model in 3 lines of code. But you would learn nothing about *why* it works. Building from scratch means you understand every single number that flows through the system.

---

## 2. The Big Picture: How a GPT Works

Here is the full journey of text through our model, from input to output:

```
Input text: "attention mechanisms are"
     |
     v
[Tokenizer] — splits text into subword tokens
     |
     v
Token IDs: [4231, 892, 17]  ← integers representing each token
     |
     v
[Token Embedding] — converts each integer into a vector of 512 numbers
     |
     v
[Positional Embedding] — adds information about WHERE each token is
     |
     v
[6x Transformer Block] — each block refines the representation
  ├── [Layer Norm]
  ├── [Causal Self-Attention] — tokens look at each other
  ├── [Layer Norm]
  └── [MLP] — each token processes what it learned
     |
     v
[Final Layer Norm]
     |
     v
[LM Head] — projects to vocabulary size (32,000 numbers)
     |
     v
[Softmax] — converts to probabilities
     |
     v
Output: probability distribution over 32,000 possible next tokens
```

---

## 3. Project Structure Explained

```
gpt-project/
├── src/                        # All reusable Python code (the "library")
│   ├── data/
│   │   ├── tokenizer.py        # Train and save the BPE tokenizer
│   │   ├── dataset.py          # PyTorch Dataset — reads from the binary file
│   │   └── dataloader.py       # Wraps Dataset into batches for training
│   ├── model/
│   │   ├── attention.py        # Multi-head causal self-attention
│   │   ├── transformer.py      # One transformer block (attention + MLP)
│   │   └── gpt.py              # Full GPT model (stacks transformer blocks)
│   ├── training/
│   │   ├── trainer.py          # The training loop
│   │   ├── scheduler.py        # Learning rate schedule
│   │   └── checkpointing.py    # Save/load model weights
│   └── eval/
│       ├── metrics.py          # Perplexity calculation
│       └── benchmarking.py     # Speed benchmarking
├── scripts/
│   ├── prepare_data.py         # Run once: download → tokenize → save binary
│   └── train.py                # Run to start training
├── configs/
│   ├── small.yaml              # Hyperparameters for a ~35M param model
│   └── medium.yaml             # Hyperparameters for a ~350M param model
├── data/                       # Created after running prepare_data.py
│   ├── tokenizer.json          # The trained BPE tokenizer
│   └── tokens_train.bin        # 27.8M tokens as raw integers (55.6 MB)
├── checkpoints/                # Created during training
│   └── small/
│       ├── best.pt             # Best model so far (lowest val loss)
│       └── step_0002000.pt     # Periodic snapshots
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt            # Python packages this project needs
└── DOCUMENTATION.md            # This file
```

### Why separate `src/` from `scripts/`?

- `src/` contains **reusable building blocks** — functions and classes you can import anywhere
- `scripts/` contains **one-off runnable programs** — you execute these from the terminal

This is standard software engineering practice. If you later write a notebook to visualize training, you can `import` from `src/` without copy-pasting code.

---

## 4. Phase 1: The Data Pipeline

Before the model sees any text, we need to convert raw text into numbers. This entire pipeline lives in `scripts/prepare_data.py` and `src/data/`.

### 4.1 The Dataset

**File:** `scripts/prepare_data.py`

We use HuggingFace's `datasets` library to download the data:

```python
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
```

This gives us a table with 117,592 rows, each with a `title` and `abstract`. We concatenate them:

```python
text = title + "\n" + abstract
```

This creates one long string per paper, which becomes the training material.

**Why titles + abstracts?** An abstract already contains the key ideas of a paper compressed into ~200 words. It's dense, structured, and consistent — ideal for a small model.

### 4.2 Tokenization and BPE

**File:** `src/data/tokenizer.py`

#### Why can't we just feed text directly to the model?

Neural networks only understand numbers. We need to map text → integers.

#### The naive approach: character-level

We could assign an integer to every letter: `a=1, b=2, ...`. But then the model has to learn from scratch that `c`, `a`, `t` together mean "cat". It has to work very hard at the character level before it can even start learning meaning.

#### The word-level approach

Map every word to an integer: `"attention"=4231`. But English has hundreds of thousands of words, and scientific text invents new ones constantly ("self-supervised", "cross-lingual"). Your vocabulary would need to be enormous, and any new word is just `[UNKNOWN]`.

#### BPE: the best of both worlds

**Byte-Pair Encoding (BPE)** is a middle ground. It starts with individual characters and repeatedly merges the most frequent pairs:

```
Start:  a t t e n t i o n
Step 1: at t e n t i o n     (merged "a"+"t" → "at", very common)
Step 2: at te n t i o n      (merged "t"+"e" → "te")
Step 3: at ten t i o n       (merged "te"+"n" → "ten")
Step 4: atten t i o n        ...
Step 5: attention             eventually becomes one token
```

The result: common words become single tokens, rare words get split into recognizable pieces. "self-supervised" might become `["self", "-", "super", "vised"]`. The model can handle any word, even ones it's never seen.

**Our tokenizer:** 32,000 vocabulary size, trained on our corpus. Special tokens:
- `[BOS]` — Beginning of Sequence (marks where a paper starts)
- `[EOS]` — End of Sequence (marks where a paper ends)
- `[PAD]` — Padding (for batching sequences of different lengths)
- `[UNK]` — Unknown token (fallback for truly unrecognizable characters)

**Code breakdown** (`src/data/tokenizer.py`):

```python
def train_bpe_tokenizer(texts, vocab_size=32000, save_path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(...)  # split on bytes first
    tokenizer.decoder = decoders.ByteLevel()                  # reverse the encoding
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2,    # a pair must appear at least twice to be merged
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)
```

### 4.3 Binary Token File

**File:** `scripts/prepare_data.py` → `data/tokens_train.bin`

After tokenizing, we have something like:

```
Paper 1: [BOS, 4231, 892, 17, 6610, ..., EOS]
Paper 2: [BOS, 771, 2204, 55, ..., EOS]
...
```

We concatenate all of these into **one giant flat array** of 27.8 million integers and save it as a binary file.

```python
arr = np.array(all_ids, dtype=np.uint16)  # uint16 = 2 bytes per token
arr.tofile(out_path)
```

**Why `uint16`?** Our vocab size is 32,000. The maximum value of `uint16` is 65,535 — enough to hold any token ID. And `uint16` uses only 2 bytes per token instead of 4 (int32) or 8 (int64), halving storage and memory usage.

**Why one flat array instead of separate files per paper?** Training efficiency. During training, we read random fixed-size chunks from this array. If it's all one file, we can load it into memory once using `np.memmap` (memory-mapped file — the OS loads pages on demand, we never load the whole 55 MB at once). This is much faster than opening thousands of small files.

**Result:** 27,809,478 tokens in 55.6 MB.

### 4.4 Dataset and DataLoader

**File:** `src/data/dataset.py` and `src/data/dataloader.py`

#### The Dataset

```python
class TokenDataset(Dataset):
    def __init__(self, token_file, seq_len):
        self.seq_len = seq_len
        data = np.memmap(token_file, dtype=np.uint16, mode="r")
        self.data = torch.from_numpy(data.astype(np.int64))

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]       # input tokens
        y = self.data[start + 1 : start + self.seq_len + 1]  # target tokens
        return x, y
```

**The crucial shift by 1:** Notice that `y` is `x` shifted one position to the right. This is how language model training works:

```
x (input):  [The, cat, sat, on]
y (target): [cat, sat, on, the]
```

The model sees "The" and must predict "cat". It sees "The cat" and must predict "sat". Each position in the sequence is simultaneously an input AND a label. One sequence of 512 tokens gives us 512 training examples for free.

**With 27.8M tokens and seq_len=512:** We get about 54,300 non-overlapping sequences in the dataset.

#### The DataLoader

```python
def make_dataloaders(token_file, seq_len, batch_size, val_fraction=0.05, ...):
    dataset = TokenDataset(token_file, seq_len)
    val_size = max(1, int(len(dataset) * val_fraction))  # 5% held out for validation
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, ...)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, ...)
    return train_loader, val_loader
```

**What is a DataLoader?** It wraps the Dataset and provides:
- **Batching:** groups multiple sequences together (batch_size=8 means 8 sequences at once, shape `[8, 512]`)
- **Shuffling:** randomizes the order each epoch so the model doesn't memorize sequence order
- **Parallel loading:** `num_workers=4` means 4 CPU processes pre-fetch data while the GPU trains

**Train vs. validation split:** 95% of sequences go to training, 5% to validation. The validation set is data the model never trains on — we use it to measure how well the model generalizes to unseen text. If train loss keeps dropping but val loss rises, the model is overfitting (memorizing training data instead of learning patterns).

---

## 5. Phase 2: The Model

All model code lives in `src/model/`. The three files build on each other:
`attention.py` → `transformer.py` → `gpt.py`

### 5.1 Embeddings

**File:** `src/model/gpt.py`

Before attention can happen, each token ID needs to become a vector — a list of numbers the model can do math on.

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)  # 32000 × 512 lookup table
self.pos_emb = nn.Embedding(seq_len, d_model)      # 512 × 512 lookup table
```

**Token embedding:** A lookup table. Token ID `4231` → row 4231 of a 32,000×512 matrix → a vector of 512 numbers. Initially random, these numbers are learned during training. Similar words end up with similar vectors.

**Positional embedding:** The same token in different positions should be treated differently. "Cat sat on the mat" — the model needs to know "cat" is position 0 and "mat" is position 4. We add a learned vector for each position 0–511.

```python
pos = torch.arange(T, device=idx.device)          # [0, 1, 2, ..., T-1]
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # add token + position info
```

The result: every token is represented as a 512-dimensional vector that encodes both *what* the token is and *where* it appears.

**d_model = 512** is the "width" of the model — how many numbers represent each token throughout the entire network. Larger d_model = more expressive but slower and more memory-hungry.

### 5.2 Attention: The Heart of GPT

**File:** `src/model/attention.py`

Attention is what makes transformers powerful. It lets each token "look at" other tokens and decide what context is relevant.

#### The intuition

Consider: "The bank by the river was steep."

When processing "bank", how do we know it means a riverbank and not a financial institution? The word "river" is the clue. Attention lets "bank" look at "river", realize it's relevant, and pull information from it.

#### Queries, Keys, and Values

Every token produces three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information will I share if selected?"

Attention score between token A (query) and token B (key):
```
score(A, B) = dot_product(Q_A, K_B) / sqrt(head_dim)
```

A high dot product means "token A finds token B relevant". We then softmax all scores so they sum to 1 (making them probabilities), and use them to take a weighted average of all Values.

```
output_A = sum over B of: softmax(score(A,B)) * V_B
```

Token A's output is a blend of all other tokens' Values, weighted by how relevant they are.

#### Why divide by sqrt(head_dim)?

Raw dot products between high-dimensional vectors can get very large, pushing softmax into regions where gradients vanish. Dividing by `sqrt(head_dim)` keeps the scale stable.

#### Multi-Head Attention

Instead of one set of Q, K, V, we run **8 parallel attention operations** (heads), each with its own weights. Each head can specialize:
- Head 1 might learn syntactic relationships (subject-verb)
- Head 2 might learn semantic similarity
- Head 3 might focus on nearby context

The outputs of all 8 heads are concatenated and projected back to d_model.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.n_heads = n_heads           # 8
        self.head_dim = d_model // n_heads  # 512 / 8 = 64 per head
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # one matrix for Q, K, V
        self.proj = nn.Linear(d_model, d_model, bias=False)      # recombine heads

    def forward(self, x):
        B, T, C = x.shape  # batch=8, tokens=512, channels=512
        
        # Compute Q, K, V for all heads at once
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # PyTorch's optimized attention (uses Flash Attention on supported hardware)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)
        
        # Merge heads and project
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)
```

#### `is_causal=True` — the crucial constraint

This is what makes it a *language model* rather than just a model. Each token can only attend to tokens **before** it (including itself), never future tokens.

```
Token 0: can see [0]
Token 1: can see [0, 1]
Token 2: can see [0, 1, 2]
Token 3: can see [0, 1, 2, 3]
```

This is enforced by a mask that sets future attention scores to `-infinity` before softmax (so they become 0 after softmax). Without this, the model could "cheat" during training by reading ahead — and then at inference time it couldn't work, because there's no future text to look at.

#### Flash Attention

`F.scaled_dot_product_attention` in PyTorch 2.x automatically uses Flash Attention when available. Flash Attention is a mathematically identical but much more memory-efficient implementation — instead of materializing the full `T×T` attention matrix (512×512=262,144 numbers per head), it computes attention in tiles that fit in fast GPU cache. This gives 2-4x speedup and uses far less memory.

### 5.3 The Transformer Block

**File:** `src/model/transformer.py`

A Transformer Block wraps attention with two other key ingredients: a feed-forward network (MLP) and Layer Normalization.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.mlp(self.ln2(x))    # MLP sub-layer
        return x
```

#### Layer Normalization

Before attention and MLP, we apply LayerNorm. It normalizes the 512 numbers for each token independently to have mean≈0 and std≈1. This prevents any single dimension from dominating and keeps gradients flowing smoothly during training. Without normalization, deep networks often fail to train at all.

#### Residual Connections (`x + ...`)

Notice that we do `x = x + self.attn(...)` rather than `x = self.attn(x)`. This is a **residual connection** (also called a skip connection).

**Why?** In a 6-layer network, gradients must flow backward through every layer during backpropagation. Without residual connections, gradients shrink exponentially as they travel backward (the "vanishing gradient problem") and the early layers learn nothing. With residual connections, gradients have a "highway" to travel directly from the loss to early layers without passing through the transformations, making deep networks trainable.

The intuition: each block adds a small *correction* to x, rather than completely rewriting it. The original signal is always preserved.

#### The MLP

```python
class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # expand: 512 → 2048
            nn.GELU(),                           # non-linearity
            nn.Linear(4 * d_model, d_model),   # contract: 2048 → 512
            nn.Dropout(dropout),
        )
```

**Role of the MLP:** Attention is about routing information between tokens (which tokens talk to which). The MLP is about processing that information *within* each token independently. After attention lets "bank" collect context from "river", the MLP processes that combined representation to update "bank"'s meaning.

**4× expansion:** A common heuristic — the hidden layer is 4 times wider than d_model. This gives the network enough capacity to represent complex transformations without making the model too wide at every layer.

**GELU:** The non-linearity. Without non-linear activation functions, stacking linear layers is mathematically equivalent to a single linear layer — you gain no depth. GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU that tends to work slightly better for transformers.

**Dropout:** During training, randomly sets 10% of values to zero. This forces the network to learn redundant representations — no single neuron can be relied upon — which reduces overfitting.

### 5.4 The Full GPT

**File:** `src/model/gpt.py`

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, n_layers, dropout=0.1):
        self.tok_emb = nn.Embedding(vocab_size, d_model)    # token lookup
        self.pos_emb = nn.Embedding(seq_len, d_model)       # position lookup
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )                                                    # 6 transformer blocks
        self.ln_f = nn.LayerNorm(d_model)                  # final layer norm
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # project to vocab
        
        self.tok_emb.weight = self.head.weight              # weight tying!
```

#### Weight Tying

`self.tok_emb.weight = self.head.weight` makes the token embedding table and the final output projection share the **same** weight matrix. This means the same 32,000×512 matrix is used both to look up input token representations AND to score output token probabilities.

**Why?** Two reasons:
1. **Parameter efficiency:** saves 32,000×512×2 = 32.7M parameters (almost the size of the whole model)
2. **Coherence:** tokens that are similar in the input space should also score similarly in the output space. Sharing weights enforces this symmetry.

#### Weight Initialization

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
```

All weights start as random Gaussian values with std=0.02. Why not zeros? A network of zeros would produce identical gradients for all neurons — they'd all learn the same thing and the network would never gain capacity. Random initialization breaks this symmetry.

#### Text Generation

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.seq_len:]           # trim to context window
        logits = self(idx_cond)[:, -1, :]           # only care about last position
        logits = logits / temperature               # temperature scaling
        # top_k: zero out all but the 50 most likely tokens
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)       # convert to probabilities
        next_tok = torch.multinomial(probs, num_samples=1)  # sample
        idx = torch.cat([idx, next_tok], dim=1)     # append to sequence
    return idx
```

**Temperature:** Dividing logits by temperature before softmax controls randomness. Temperature=1.0 is default. Temperature<1 (e.g. 0.5) makes the distribution sharper — the model picks the most likely tokens more consistently. Temperature>1 (e.g. 1.5) flattens the distribution — more random, creative output.

**Top-K sampling:** Instead of sampling from all 32,000 tokens, we keep only the 50 most likely and zero out the rest. This prevents the model from occasionally generating completely nonsensical tokens that have tiny but non-zero probability.

---

## 6. Phase 3: Training

All training code lives in `src/training/`. The scripts entry point is `scripts/train.py`.

### 6.1 The Loss Function

We use **cross-entropy loss**, the standard loss for classification tasks (and language modeling is just classification: "which of the 32,000 tokens comes next?").

For each position in the sequence, the model outputs a probability distribution over 32,000 tokens. Cross-entropy loss measures how surprised the model was by the actual next token:

```
loss = -log(probability assigned to the correct token)
```

If the model assigned 90% probability to the correct token: loss = -log(0.9) = 0.105 (low, good)
If the model assigned 1% probability to the correct token: loss = -log(0.01) = 4.6 (high, bad)

We average this over all positions in all sequences in the batch. At the start of training, the model assigns roughly equal probability to all tokens: 1/32000 ≈ 0.00003, so loss ≈ -log(0.00003) ≈ **10.4**. A well-trained model might reach **3.0–4.0** on this dataset.

### 6.2 Backpropagation

After computing the loss, we need to update all the weights to make the loss smaller. **Backpropagation** computes the gradient of the loss with respect to every single parameter in the model.

A gradient is just a derivative — "if I increase this weight slightly, does the loss go up or down, and by how much?" With 35.5M parameters, PyTorch automatically computes 35.5M gradients in one backward pass using the chain rule of calculus.

```python
loss.backward()   # PyTorch computes all 35.5M gradients automatically
```

This is why deep learning became practical — automatic differentiation means you define the forward computation, and gradients come for free.

### 6.3 The Optimizer: AdamW

```python
self.optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,            # learning rate
    weight_decay=0.1,     # L2 regularization
    betas=(0.9, 0.95),    # momentum coefficients
)
```

**SGD (naive approach):** Update each weight by: `weight -= lr * gradient`. Simple, but slow — learning rate must be the same for all parameters.

**Adam** (Adaptive Moment Estimation) improves on SGD in two ways:
1. **Momentum:** keeps a running average of past gradients, so it doesn't change direction abruptly with each noisy batch
2. **Adaptive learning rate:** each parameter gets its own effective learning rate, scaled by how large its gradients have historically been. Parameters with consistently large gradients get smaller steps; parameters with small gradients get larger steps.

**AdamW** adds **weight decay** — a small penalty proportional to the weight's magnitude. This prevents weights from growing arbitrarily large and encourages the model to spread learning across many parameters rather than relying on a few very large ones (regularization against overfitting).

**betas=(0.9, 0.95):** The momentum coefficients. 0.9 means gradient momentum decays slowly (90% of previous momentum is kept). 0.95 means the adaptive scaling is based on a longer history. These are standard transformer values from the GPT-3 paper.

### 6.4 Learning Rate Schedule

**File:** `src/training/scheduler.py`

The learning rate isn't fixed throughout training — we use a schedule:

```
                   peak lr (0.0003)
                  /\
                 /  \
                /    \_______________
               /                     \___  min lr (10% of peak)
              /
  warmup     /  cosine decay
  (200 steps)
```

```python
def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps          # linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π * progress))
    return LambdaLR(optimizer, lr_lambda)
```

**Linear warmup (steps 0–200):** At the very start, weights are random and gradients are chaotic. A large learning rate would cause huge, harmful updates. We start small and ramp up over 200 steps, by which point the model has stabilized.

**Cosine decay (steps 200–20,000):** We gradually reduce the learning rate following a cosine curve. Early in training, we want larger steps to explore the loss landscape quickly. Late in training, we want smaller steps to fine-tune within a good region without overshooting.

**Why cosine instead of linear decay?** Cosine spends more time near the peak (fast learning) and slows down gracefully near the end, rather than decaying linearly. Empirically, cosine produces better final models.

### 6.5 Checkpointing

**File:** `src/training/checkpointing.py`

```python
def save_checkpoint(path, model, optimizer, step, val_loss):
    torch.save({
        "step": step,
        "val_loss": val_loss,
        "model_state": model.state_dict(),      # all 35.5M weights
        "optimizer_state": optimizer.state_dict(),  # Adam's momentum buffers
    }, path)
```

A checkpoint saves everything needed to resume training: model weights, optimizer state (Adam needs its momentum history), and training step.

We save:
- `best.pt` — overwritten whenever validation loss improves (always the best model)
- `step_XXXXXXX.pt` — every 2,000 steps (a full history you can roll back to)

**Why save optimizer state?** If you resume from a checkpoint with only the model weights, Adam's momentum buffers are reset to zero. The first few steps after resuming would behave poorly (the optimizer "forgot" what direction it was moving). Saving optimizer state ensures seamless resumption.

### 6.6 The Training Loop

**File:** `src/training/trainer.py`

```python
def fit(self, train_loader, val_loader):
    for epoch in range(cfg["epochs"]):
        for x, y in train_loader:          # iterate over all batches
            loss = self.train_step(x, y)   # forward + backward + optimizer step

            if self.step % 50 == 0:        # log every 50 steps
                print(f"step {self.step} | loss {loss:.4f} | lr {lr:.2e}")

            if self.step % 500 == 0:       # evaluate on validation set
                val_loss = self.eval_loss(val_loader)
                if val_loss < best_val:
                    save_checkpoint("best.pt", ...)

            if self.step >= cfg["total_steps"]:
                return                     # stop after 20,000 steps
```

**One step** = one batch of 8 sequences × 512 tokens = 4,096 tokens processed.

**One epoch** = one full pass through all 54,300 sequences in the training set.

**20,000 total steps** = about 3 epochs at batch_size=8 (20,000 steps × 8 sequences / 54,300 sequences per epoch ≈ 3 epochs).

#### Inside `train_step`

```python
def train_step(self, x, y):
    x, y = x.to(self.device), y.to(self.device)   # move to MPS/GPU
    
    logits = self.model(x)                          # forward pass
    loss = cross_entropy(logits, y)                 # compute loss
    
    self.optimizer.zero_grad(set_to_none=True)      # clear old gradients
    loss.backward()                                  # compute new gradients
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # clip
    self.optimizer.step()                            # update weights
    self.scheduler.step()                            # update learning rate
    
    return loss.item()
```

**Gradient clipping** (`clip_grad_norm_(..., 1.0)`): If the gradient vector has a magnitude larger than 1.0, scale it down so its magnitude is exactly 1.0. This prevents rare "exploding gradient" events where a single bad batch causes enormous weight updates that destroy training progress.

**`zero_grad(set_to_none=True)`:** PyTorch accumulates gradients by default. We must clear them before each step. `set_to_none=True` is slightly faster than zeroing — it deallocates the gradient tensors rather than filling them with zeros.

---

## 7. Configuration System

**Files:** `configs/small.yaml`, `configs/medium.yaml`

Rather than hard-coding hyperparameters in Python, we store them in YAML files. This lets you run different experiments without changing code:

```bash
python scripts/train.py --config configs/small.yaml   # 35M params, fast
python scripts/train.py --config configs/medium.yaml  # 350M params, slow
```

**Current `small.yaml`:**
```yaml
model:
  vocab_size: 32000   # how many unique tokens
  seq_len: 512        # how many tokens the model sees at once
  d_model: 512        # width of the model
  n_heads: 8          # number of attention heads
  n_layers: 6         # number of transformer blocks stacked
  dropout: 0.1        # 10% dropout rate

data:
  token_file: data/tokens_train.bin
  val_fraction: 0.05  # 5% for validation
  batch_size: 8       # sequences per step (limited by Mac RAM)
  num_workers: 4      # CPU processes for data loading

training:
  lr: 0.0003          # peak learning rate
  weight_decay: 0.1
  warmup_steps: 200
  total_steps: 20000
  epochs: 10
  log_every: 50
  eval_every: 500
  save_every: 2000
  checkpoint_dir: checkpoints/small
```

---

## 8. Key ML Concepts Glossary

| Term | Definition |
|------|-----------|
| **Parameter** | A learnable number in the model (a weight or bias). Our model has 35.5M of them. |
| **Tensor** | A multi-dimensional array of numbers. PyTorch's fundamental data structure. Shape `[8, 512, 512]` means 8 batches × 512 tokens × 512 features. |
| **Forward pass** | Running input through the model to get output (and loss). |
| **Backward pass** | Computing gradients via backpropagation. |
| **Gradient** | How much the loss changes if you nudge a parameter. Points uphill; we go downhill. |
| **Learning rate** | How big a step we take in the gradient direction per update. Too large = unstable. Too small = slow. |
| **Batch** | A group of sequences processed together. Bigger batches = more stable gradients but more memory. |
| **Epoch** | One full pass through the entire training dataset. |
| **Overfitting** | Model memorizes training data instead of learning general patterns. Validation loss rises while train loss falls. |
| **Perplexity** | `exp(loss)`. A more interpretable version of loss. "On average, how many tokens was the model choosing between?" Lower is better. |
| **Embedding** | A dense vector representation of a discrete object (token, position). |
| **Logits** | Raw unnormalized scores before softmax. The model's output before converting to probabilities. |
| **Softmax** | Converts a vector of logits to a probability distribution (all positive, sums to 1). |
| **LayerNorm** | Normalizes activations within each token to have mean≈0, std≈1. Stabilizes training. |
| **Residual connection** | Adding the input of a layer to its output (`x + f(x)`). Allows gradients to flow through deep networks. |
| **MPS** | Metal Performance Shaders — Apple Silicon's GPU API. PyTorch uses it as a CUDA alternative on Mac. |
| **Checkpoint** | A saved snapshot of model weights (and optionally optimizer state) at a specific training step. |
| **Hyperparameter** | A configuration choice you make before training (learning rate, model size, etc.) — as opposed to parameters, which are learned. |

---

## 9. What the Numbers Mean

### Model size: 35.5M parameters

The actual parameter count (weight tying reduces it from the ~85M estimate):
- Token embedding: 32,000 × 512 = **16.4M** (shared with output head)
- Position embedding: 512 × 512 = **0.26M**
- Per transformer block (×6):
  - QKV projection: 512 × 1,536 = 0.79M
  - Output projection: 512 × 512 = 0.26M
  - MLP expand: 512 × 2,048 = 1.05M
  - MLP contract: 2,048 × 512 = 1.05M
  - LayerNorm ×2: negligible
- Total per block: ~3.15M × 6 = **18.9M**
- **Total: ~35.5M**

For reference: GPT-2 small = 117M params, GPT-3 = 175B params.

### Training speed: ~674ms/step on MPS

- Per step: 8 sequences × 512 tokens = 4,096 tokens
- Throughput: ~6,000 tokens/second
- 20,000 steps total ≈ **3.7 hours**

### Data: 27.8M tokens

The ArXiv papers, once tokenized, produce 27.8M tokens. 
- For comparison, GPT-3 was trained on ~300B tokens — about 10,000× more.
- This means our model will learn ML paper structure and jargon well, but won't have deep knowledge or reasoning ability.

### Expected loss trajectory

| Step | Expected Loss | Perplexity | Interpretation |
|------|--------------|------------|----------------|
| 0 | ~10.4 | ~32,000 | Random (uniform over vocab) |
| 500 | ~5–6 | ~150–400 | Learning basic structure |
| 2,000 | ~4–5 | ~55–150 | Learning domain vocabulary |
| 10,000 | ~3–4 | ~20–55 | Generating plausible-sounding text |
| 20,000 | ~2.5–3.5 | ~12–33 | Decent ML-flavored text generation |

---

## 10. What Happens Next

**Phase 2 is in progress** — the training run is active right now on your Mac's MPS GPU.

Once training completes, here's what's planned:

### Immediate next steps

1. **Evaluate the model** — check final perplexity, look at the loss curve
2. **Generate text** — prompt the model with an abstract opening and see what it produces
3. **Visualize training** — plot train/val loss curves over time (we'll add this to a notebook)

### Future phases

| Phase | Focus |
|-------|-------|
| Phase 3 | Evaluation — perplexity on held-out data, qualitative text generation |
| Phase 4 | Scaling — try medium config, experiment with hyperparameters |
| Phase 5 | Improvements — rotary position embeddings (RoPE), better tokenizer, gradient accumulation |
| Phase 6 | Cloud training — move to a GPU instance for larger model runs |

### Things you can experiment with now

- **Change `seq_len`** in the config: shorter sequences train faster but the model has less context
- **Change `n_layers`**: more layers = deeper model, better at reasoning, slower to train
- **Change `dropout`**: increase to 0.2 if you see overfitting, decrease to 0.0 for a faster but potentially overfit model
- **Change `lr`**: if loss is unstable/diverging, try 0.0001; if loss is plateauing too early, try 0.0005

---

*This documentation covers the project as of May 14, 2026 — end of Phase 1 (data pipeline) and start of Phase 2 (training). The model is currently training.*
