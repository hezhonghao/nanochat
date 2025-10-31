# Nanochat Architecture & Implementation Guide

> A comprehensive deep-dive into transformer structure, algorithms, and bug-prone areas in the nanochat codebase.

**Author**: Study guide for understanding transformer internals before training runs
**Date**: October 2025
**Codebase**: [karpathy/nanochat](https://github.com/karpathy/nanochat)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Transformer Architecture Deep Dive](#2-transformer-architecture-deep-dive)
3. [Training Pipeline](#3-training-pipeline)
4. [RL Training Deep Dive](#4-rl-training-deep-dive)
5. [Data Loading](#5-data-loading)
6. [Tokenizer](#6-tokenizer)
7. [Inference Engine](#7-inference-engine)
8. [Common Bug Hotspots](#8-common-bug-hotspots)
9. [Learning Path](#9-learning-path)
10. [Quick Reference](#10-quick-reference)

---

## 1. Project Structure

```
nanochat/
├── nanochat/          # Core library
│   ├── gpt.py         # Transformer model architecture ⭐
│   ├── engine.py      # Inference engine with KV cache ⭐
│   ├── tokenizer.py   # BPE tokenizer (rustbpe + tiktoken)
│   ├── dataloader.py  # Streaming data loader
│   ├── muon.py        # Muon optimizer for matrices
│   ├── adamw.py       # AdamW for embeddings
│   ├── common.py      # Distributed utilities
│   ├── checkpoint_manager.py
│   ├── core_eval.py
│   ├── loss_eval.py
│   ├── execution.py
│   └── report.py
├── scripts/           # Training & evaluation scripts
│   ├── base_train.py  # Pretraining script ⭐
│   ├── mid_train.py   # Midtraining script
│   ├── chat_sft.py    # Supervised fine-tuning
│   ├── chat_rl.py     # RL with GRPO/REINFORCE ⭐
│   ├── chat_web.py    # Web UI server
│   ├── chat_cli.py    # CLI interface
│   └── tok_train.py   # Tokenizer training
├── tasks/             # Evaluation tasks (GSM8K, etc.)
├── tests/             # Unit tests
└── rustbpe/           # Rust BPE tokenizer (fast training)
```

**Total**: ~8,300 lines of code across 44 files

---

## 2. Transformer Architecture Deep Dive

**File**: [nanochat/gpt.py](../nanochat/gpt.py) (323 lines)

### 2.1 Key Architectural Features

The model uses modern transformer improvements that differ from the original GPT-2:

#### **1. Rotary Position Embeddings (RoPE)**
- **Lines**: 41-49, 201-215
- **No learned positional embeddings** - positions encoded via rotation in Q/K space
- Applied to both queries and keys before attention

```python
def apply_rotary_emb(x, cos, sin):
    """Rotate pairs of dimensions in x using precomputed cos/sin"""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split into two halves
    y1 = x1 * cos + x2 * sin         # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)     # re-assemble
    out = out.to(x.dtype)            # ensure input/output dtypes match
    return out
```

**Precomputation** (gpt.py:201-215):
```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
    channel_range = torch.arange(0, head_dim, 2)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos, sin = freqs.cos(), freqs.sin()
    # Shape: (1, seq_len, 1, head_dim/2) for broadcasting
    return cos[None, :, None, :], sin[None, :, None, :]
```

**Why RoPE?**
- Relative positional encoding (not absolute)
- Extrapolates better to longer sequences
- No extra parameters to learn

---

#### **2. QK Normalization**
- **Line**: 90
- Normalizes queries and keys after applying RoPE

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)  # QK norm for stability
```

**Why QK norm?**
- Stabilizes training for deep models
- Prevents attention scores from exploding
- Used in modern LLMs (Gemini, etc.)

---

#### **3. RMSNorm without learnable params**
- **Lines**: 36-38
- Simpler than LayerNorm (no bias/scale)

```python
def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))
```

**Why no params?**
- Fewer parameters = faster training
- Still provides normalization benefits
- Modern trend in LLM design

---

#### **4. Untied Embeddings**
- **Lines**: 159, 162
- Separate weights for input embedding vs output projection

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(vocab_size, n_embd),  # Input embedding
    "h": nn.ModuleList([...]),                 # Transformer blocks
})
self.lm_head = nn.Linear(n_embd, vocab_size)  # Output projection (separate!)
```

**Why untied?**
- More expressiveness for output layer
- Common in modern LLMs
- Small memory cost for better performance

---

#### **5. ReLU² Activation in MLP**
- **Line**: 137
- `F.relu(x).square()` instead of GELU/SwiGLU

```python
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()  # ReLU²
    x = self.c_proj(x)
    return x
```

**Why ReLU²?**
- Simpler than GELU
- Works well empirically
- Faster to compute

---

#### **6. Multi-Query Attention (MQA)**
- **Lines**: 31-32, 52-61, 99-101
- Multiple query heads share same key/value heads

```python
@dataclass
class GPTConfig:
    n_head: int = 6      # Number of query heads
    n_kv_head: int = 6   # Number of key/value heads (for MQA)
```

```python
def repeat_kv(x, n_rep):
    """Expand KV heads to match number of query heads"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
```

**In attention** (gpt.py:99-101):
```python
# Apply MQA: replicate key/value heads for each query head
nrep = self.n_head // self.n_kv_head
k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
```

**Why MQA?**
- **Inference speedup**: Much smaller KV cache
- **Memory savings**: For n_head=12, n_kv_head=1 → 12x smaller cache!
- Used in: Llama 2, GPT-4 (reportedly), PaLM

**Example**:
- 12 query heads, 1 KV head → 12:1 ratio (Grouped Query Attention)
- Default in nanochat: 1:1 ratio (standard multi-head attention)

---

#### **7. Logits Softcapping**
- **Lines**: 278, 283, 290
- Prevents extreme logit values

```python
softcap = 15
logits = softcap * torch.tanh(logits / softcap)  # Soft clamp to [-15, 15]
```

**Why softcapping?**
- Numerical stability
- Prevents saturation in softmax
- Used in Gemini models

---

### 2.2 Critical Code Sections

#### **Attention Mechanism** (gpt.py:79-126)

The attention function handles **three different cases**:

```python
def forward(self, x, cos_sin, kv_cache):
    B, T, C = x.size()

    # Project input to Q, K, V
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # Apply RoPE + QK norm
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # Apply KV cache if provided
    if kv_cache is not None:
        k, v = kv_cache.insert_kv(self.layer_idx, k, v)
    Tq = q.size(2)  # Number of queries
    Tk = k.size(2)  # Number of keys (cache + current)

    # Expand KV heads to match Q heads (MQA)
    nrep = self.n_head // self.n_kv_head
    k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

    # Three cases for attention:
    if kv_cache is None or Tq == Tk:
        # Case 1: Training (no cache) OR full prefill
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    elif Tq == 1:
        # Case 2: Autoregressive decoding (single token)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        # Case 3: Chunked inference (multiple new tokens with prefix cache)
        attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
        prefix_len = Tk - Tq
        if prefix_len > 0:
            attn_mask[:, :prefix_len] = True  # Attend to all prefix
        # Causal mask within the new chunk
        attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # Reassemble and project
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y
```

**🔴 Bug-prone area**: Case 3 attention mask (lines 113-121)
- `prefix_len = Tk - Tq` must be calculated correctly
- Mask must combine: (1) attend to all cached keys, (2) causal within new chunk
- Off-by-one errors can cause incorrect masking

---

#### **Model Forward Pass** (gpt.py:259-291)

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Get rotary embeddings for current sequence
    assert T <= self.cos.size(1), "Sequence too long for RoPE cache"
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # Forward through transformer
    x = self.transformer.wte(idx)
    x = norm(x)  # Norm after embedding
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)
    x = norm(x)  # Final norm

    # Compute logits
    logits = self.lm_head(x)
    logits = 15 * torch.tanh(logits / 15)  # Softcap
    logits = logits.float()  # Use FP32 for numerics

    if targets is not None:
        # Training: compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,  # Ignore padding/masked tokens
            reduction=loss_reduction
        )
        return loss
    else:
        # Inference: return logits
        return logits
```

**Key points**:
- Line 267: Rotary embedding offset by cache position
- Line 272: Norm applied after token embedding (uncommon)
- Line 285: `ignore_index=-1` critical for masking

---

### 2.3 Model Sizing

**From base_train.py:74-82**:

```python
num_layers = depth                          # e.g., depth=20
model_dim = depth * 64                      # e.g., 20*64=1280 (aspect ratio 64)
num_heads = max(1, (model_dim + 127) // 128)  # e.g., ceil(1280/128)=10 heads
num_kv_heads = num_heads                    # 1:1 ratio (standard MHA)
```

**Aspect ratio**: `model_dim / depth = 64`
- Smaller models: ratio ~64
- Larger models: ratio can go to 128

**Head dimension**: Always ~128
- `head_dim = model_dim / num_heads ≈ 128`

**Example sizes**:
| depth | model_dim | num_heads | params (M) | Chinchilla tokens (B) |
|-------|-----------|-----------|------------|-----------------------|
| 12    | 768       | 6         | ~85        | 1.7                   |
| 20    | 1280      | 10        | ~240       | 4.8                   |
| 26    | 1664      | 13        | ~400       | 8.0                   |
| 32    | 2048      | 16        | ~600       | 12.0                  |

---

## 3. Training Pipeline

**File**: [scripts/base_train.py](../scripts/base_train.py) (340 lines)

### 3.1 Dual Optimizer Setup

**Critical insight**: nanochat uses **TWO different optimizers**!

**From gpt.py:228-257**:

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2,
                     matrix_lr=0.02, weight_decay=0.0):
    model_dim = self.config.n_embd

    # Separate parameters into 3 groups
    matrix_params = list(self.transformer.h.parameters())      # All transformer blocks
    embedding_params = list(self.transformer.wte.parameters()) # Token embedding
    lm_head_params = list(self.lm_head.parameters())          # Output projection

    # Scale LR by 1/√(d_model/768)
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # AdamW for embeddings and lm_head
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_optimizer = DistAdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10,
                                weight_decay=weight_decay)

    # Muon for transformer linear layers
    muon_optimizer = DistMuon(matrix_params, lr=matrix_lr, momentum=0.95)

    return [adamw_optimizer, muon_optimizer]
```

**Why two optimizers?**
1. **Muon** (Momentum Orthogonalized by Newton): For weight matrices
   - Preconditioned gradient descent
   - Better for large linear layers
   - See [nanochat/muon.py](../nanochat/muon.py) for implementation

2. **AdamW**: For embeddings and output head
   - Adaptive learning rates work well for embedding layers
   - Standard choice for sparse parameters

**🔴 Bug-prone area**: Learning rate scaling (line 238)
```python
dmodel_lr_scale = (model_dim / 768) ** -0.5
```
- If you change model size, LRs automatically scale
- Tuned for 768-dim models, scales for others
- **Example**: For 1280-dim model, LR × 0.77

---

### 3.2 Gradient Accumulation

**From base_train.py:86-92**:

```python
# Calculate gradient accumulation steps
tokens_per_fwdbwd = device_batch_size * max_seq_len  # Per GPU
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # All GPUs
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

**Example**:
- `device_batch_size=32`, `max_seq_len=2048`, `ddp_world_size=8`
- `tokens_per_fwdbwd = 32 * 2048 = 65,536`
- `world_tokens_per_fwdbwd = 65,536 * 8 = 524,288`
- `total_batch_size = 524,288` → `grad_accum_steps = 1` (no accumulation needed)

**If you have less GPUs**:
- `ddp_world_size=1` → `grad_accum_steps = 8` (8 forward/backward passes per step)

**🔴 Common bug**:
```python
assert total_batch_size % world_tokens_per_fwdbwd == 0
```
If assertion fails, `total_batch_size` not divisible → adjust batch sizes!

---

### 3.3 Training Loop

**From base_train.py:254-280**:

```python
for step in range(num_iterations):
    # Gradient accumulation loop
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:  # bfloat16 mixed precision
            loss = model(x, y)

        train_loss = loss.detach()  # For logging
        loss = loss / grad_accum_steps  # ⚠️ CRITICAL: Normalize BEFORE backward!
        loss.backward()

        x, y = next(train_loader)  # Prefetch next batch (async)

    # Gradient clipping (optional)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

    # Update learning rates
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Update Muon momentum
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    # Step optimizers
    for opt in optimizers:
        opt.step()

    model.zero_grad(set_to_none=True)
```

**🔴 Bug-prone areas**:

1. **Loss normalization** (line 261):
   ```python
   loss = loss / grad_accum_steps  # MUST divide BEFORE .backward()!
   ```
   - Each `.backward()` **accumulates** gradients
   - Without division, gradients are `grad_accum_steps` times too large!

2. **Data prefetching** (line 263):
   ```python
   x, y = next(train_loader)  # Overlaps CPU/GPU work
   ```
   - Starts loading next batch while GPU computes backward pass
   - Requires `pin_memory=True` in dataloader

3. **Gradient clipping order** (lines 265-266):
   ```python
   torch.nn.utils.clip_grad_norm_(...)  # BEFORE optimizer step!
   ```
   - Must clip after all `.backward()` calls
   - Before any `.step()` calls

4. **Zero gradients** (line 280):
   ```python
   model.zero_grad(set_to_none=True)  # Faster than zero_grad()
   ```
   - `set_to_none=True` saves memory

---

### 3.4 Learning Rate Scheduling

**From base_train.py:148-163**:

```python
warmup_ratio = 0.0      # No warmup by default
warmdown_ratio = 0.2    # Decay last 20% of training
final_lr_frac = 0.0     # Decay to 0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # Linear warmup
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # Constant LR
        return 1.0
    else:
        # Linear decay
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**Schedule visualization**:
```
LR
 │
1.0 ─────────────────┐
 │                   │
 │                   └─────────╲
 │                              ╲
0.0 ──────────────────────────────╲──
    │         │                    │
    0%       80%                 100%
    warmup   constant            decay
```

**Muon momentum scheduling** (base_train.py:160-163):
```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)  # Ramp up over first 300 steps
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
```
- Starts at 0.85, ramps to 0.95 over 300 iterations
- Helps with early training stability

---

### 3.5 Training Horizon Calculation

**From base_train.py:108-126**:

Three ways to specify training length (in order of precedence):

```python
# Option 1: Explicit number of iterations
if num_iterations > 0:
    # Use as-is
    pass

# Option 2: Target FLOPs
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))

# Option 3: Chinchilla ratio (default)
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
```

**Chinchilla optimal**: `target_param_data_ratio = 20`
- For 100M param model: 2B training tokens
- For 1B param model: 20B training tokens

---

### 3.6 Evaluation & Checkpointing

**Validation loss** (base_train.py:177-192):
```python
if step % eval_every == 0:
    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Validation bpb: {val_bpb:.4f}")  # bits per byte
```

**CORE metric** (base_train.py:196-207):
```python
if step % core_metric_every == 0:
    results = evaluate_model(orig_model, tokenizer, device,
                           max_per_task=core_metric_max_per_task)
    print0(f"CORE metric: {results['core_metric']:.4f}")
```
- CORE = aggregate score across multiple benchmarks
- See [nanochat/core_eval.py](../nanochat/core_eval.py)

**Sampling** (base_train.py:211-228):
```python
if step % sample_every == 0:
    prompts = ["The capital of France is", ...]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, max_tokens=16, temperature=0)
        print0(tokenizer.decode(sample[0]))
```

---

## 4. RL Training Deep Dive

**File**: [scripts/chat_rl.py](../scripts/chat_rl.py) (332 lines)

### 4.1 Simplified GRPO/REINFORCE

From the docstring (lines 2-10):

> "GRPO" in quotes because we simplify it significantly:
> 1. Delete trust region → no KL regularization to reference model
> 2. On-policy → no PPO ratio clipping needed
> 3. Token-level advantage normalization (GAPO style)
> 4. Advantage = (r - μ) instead of (r - μ)/σ

**Standard GRPO** (Group Relative Policy Optimization):
```
Loss = -E[log π(a|s) * A(s,a)] + β * KL(π || π_ref)
       ↑                          ↑
     policy gradient           trust region
```

**Nanochat's simplified version**:
```
Loss = -E[log π(a|s) * A(s,a)]
       ↑
     just REINFORCE!
```

**Why simplify?**
- On-policy: sample from current model, use immediately
- No reference model needed → saves memory
- Still works well for GSM8K math problems

---

### 4.2 Rollout Generation

**From chat_rl.py:79-140**:

```python
@torch.no_grad()
def get_batch():
    for example_idx in itertools.cycle(rank_indices):
        # Get conversation from training set
        conversation = train_task[example_idx]

        # Tokenize, removing last assistant message
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples rollouts (in batches to avoid OOM)
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size

        for sampling_step in range(num_sampling_steps):
            # ⚠️ CRITICAL: Vary seed for each batch!
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF

            with autocast_ctx:
                generated_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            generated_token_sequences.extend(generated_batch)
            masks.extend(masks_batch)

        # Calculate rewards for each rollout
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad sequences to same length
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]

        # Convert to tensors
        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # Create autoregressive inputs/targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # ⚠️ Must clone!
        targets[mask_ids[:, 1:] == 0] = -1  # Mask padding/prompt/tools

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # Calculate advantages (r - mean)
        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_token_sequences, inputs, targets, rewards, advantages
```

**🔴 Bug-prone areas**:

1. **Seed variation** (line 98):
   ```python
   seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
   ```
   - Must change seed for each sampling batch
   - Without this: all batches generate identical samples!
   - `& 0x7FFFFFFF` ensures positive int32

2. **Clone before in-place modification** (line 131):
   ```python
   targets = ids[:, 1:].clone()  # MUST clone!
   targets[mask_ids[:, 1:] == 0] = -1  # In-place modification
   ```
   - Forgetting `.clone()` modifies `ids` tensor
   - Subtle bug: targets and inputs become misaligned

3. **Mask semantics** (line 132):
   ```python
   targets[mask_ids[:, 1:] == 0] = -1
   ```
   - `mask=0` → not supervised (prompt, padding, tool outputs)
   - `mask=1` → supervised (assistant generated text)
   - `-1` → ignored by cross-entropy loss

4. **Padding token choice** (line 124):
   ```python
   seq + [assistant_end] * (max_length - len(seq))
   ```
   - Uses `<|assistant_end|>` for padding
   - Safe because these positions get `target=-1` anyway

---

### 4.3 Policy Gradient Calculation

**From chat_rl.py:241-275**:

```python
for example_step in range(examples_per_rank):
    # Get one batch (num_samples rollouts for one question)
    sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

    model.train()

    # Loop over mini-batches (to not exceed device_batch_size)
    num_passes = inputs_all.size(0) // device_batch_size
    for pass_idx in range(num_passes):
        b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
        inputs = inputs_all[b0:b1]
        targets = targets_all[b0:b1]
        advantages = advantages_all[b0:b1]

        with autocast_ctx:
            # Calculate negative log-likelihoods (NLL = -log p)
            logp = -model(inputs, targets, loss_reduction='none')  # (B, T)

        # Policy gradient objective: E[log π(a|s) * A(s,a)]
        pg_obj = (logp * advantages.unsqueeze(-1)).sum()

        # Normalize by number of valid tokens
        num_valid = (targets >= 0).sum().clamp(min=1)
        pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)

        # Loss to minimize (negative of objective)
        loss = -pg_obj
        loss.backward()

        print0(f"loss: {loss.item():.6f} | avg reward: {rewards_all.mean().item()}")
```

**Key points**:

1. **Log probabilities** (line 261):
   ```python
   logp = -model(inputs, targets, loss_reduction='none')
   ```
   - Model returns NLL (negative log-likelihood)
   - Negate to get log probabilities

2. **Policy gradient** (line 263):
   ```python
   pg_obj = (logp * advantages.unsqueeze(-1)).sum()
   ```
   - `advantages`: (B,) tensor
   - `.unsqueeze(-1)`: (B, 1) for broadcasting
   - Result: weighted sum of log probs

3. **Triple normalization** (line 266):
   ```python
   pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
   ```
   - `num_valid`: number of supervised tokens
   - `num_passes`: mini-batch splits
   - `examples_per_rank`: gradient accumulation
   - Must normalize correctly for gradient accumulation!

4. **No PPO clipping** (line 267 comment):
   > "No need to add PPO ratio+clip because we are on policy"

   **Standard PPO**:
   ```python
   ratio = π_new(a|s) / π_old(a|s)
   clipped_ratio = clip(ratio, 1-ε, 1+ε)
   loss = -min(ratio * A, clipped_ratio * A)
   ```

   **On-policy (nanochat)**:
   ```python
   loss = -log π(a|s) * A  # No ratio needed!
   ```

---

### 4.4 RL Training Configuration

**Key hyperparameters** (chat_rl.py:31-47):

```python
device_batch_size = 8       # Max batch size (OOM limit)
examples_per_step = 16      # Questions per training step
num_samples = 16            # Rollouts per question
max_new_tokens = 256        # Max generation length
temperature = 1.0           # Sampling temperature
top_k = 50                  # Top-k sampling

# Learning rates (same as base training)
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 0.05         # Start at 5% of base LR

num_epochs = 1              # Passes through GSM8K
```

**Memory usage**:
- `examples_per_step=16`, `num_samples=16` → 256 sequences per step
- Each sequence up to 256 tokens
- Total: ~65K tokens per step (similar to pretraining batch)

**Training length**:
```python
train_task = GSM8K(subset="main", split="train")  # 7473 examples
num_steps = (len(train_task) // examples_per_step) * num_epochs
# = (7473 // 16) * 1 = 467 steps
```

---

### 4.5 Evaluation

**Pass@k metric** (chat_rl.py:219-239):

```python
if step % eval_every == 0:
    passk = torch.zeros(device_batch_size, device=device)

    records_iter = run_gsm8k_eval(
        val_task,
        tokenizer,
        engine,
        num_samples=device_batch_size,  # Generate k samples
        max_examples=eval_examples,
        temperature=1.0
    )
    records = list(records_iter)

    # Calculate pass@k for k=1..device_batch_size
    for k in range(1, device_batch_size + 1):
        # Example passes if ANY of first k samples are correct
        passk[k-1] = sum(
            any(o["is_correct"] for o in r["outcomes"][:k])
            for r in records
        )

    # Reduce across ranks
    if ddp:
        dist.all_reduce(passk, op=dist.ReduceOp.SUM)
    passk = passk / num_records  # Normalize
```

**Example output**:
```
Step 0 | Pass@1: 0.0250, Pass@2: 0.0450, Pass@4: 0.0750, Pass@8: 0.1200
```

**Pass@k interpretation**:
- **Pass@1**: Accuracy with greedy decoding
- **Pass@4**: % examples with ≥1 correct answer in 4 samples
- Higher k → higher success rate (more chances)

---

## 5. Data Loading

**File**: [nanochat/dataloader.py](../nanochat/dataloader.py) (50 lines)

### 5.1 Streaming Architecture

```python
def tokenizing_distributed_data_loader(B, T, split,
                                       tokenizer_threads=4,
                                       tokenizer_batch_size=128):
    """Stream text from parquet files, tokenize, yield batches."""

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 for target at last position

    # Get tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # Token buffer (deque for efficient pop/append)
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    # Infinite iterator over parquet files
    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split,
                                              start=ddp_rank,
                                              step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]

    batches = document_batches()

    # Main loop
    while True:
        # Accumulate tokens until we have enough
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch,
                                          prepend=bos_token,
                                          num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)

        # Move tokens from deque to scratch buffer
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()

        # Create inputs/targets
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]

        # Move to GPU (async)
        inputs = inputs_cpu.view(B, T).to(device="cuda", dtype=torch.int32,
                                         non_blocking=True)
        targets = targets_cpu.view(B, T).to(device="cuda", dtype=torch.int64,
                                           non_blocking=True)

        yield inputs, targets
```

---

### 5.2 Key Design Decisions

1. **Streaming (no epochs)**:
   - Infinite loop via `while True`
   - Data never "runs out"
   - Training length controlled by `num_iterations`

2. **Token buffer (deque)**:
   - Efficient append/pop operations
   - Handles variable-length documents
   - Documents concatenated seamlessly

3. **BOS prepending**:
   ```python
   token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
   ```
   - Every document starts with `<|bos|>`
   - Signals start of new sequence to model

4. **Distributed data sharding**:
   ```python
   for batch in parquets_iter_batched(split=split,
                                      start=ddp_rank,      # Offset by rank
                                      step=ddp_world_size): # Skip by world size
   ```
   - Each rank reads different data
   - No overlap between ranks
   - Ensures diverse batches

5. **Async GPU transfer**:
   ```python
   inputs = inputs_cpu.view(B, T).to(device="cuda", non_blocking=True)
   ```
   - `non_blocking=True` allows CPU/GPU overlap
   - Requires `pin_memory=True` (line 19)

---

### 5.3 Bug-Prone Areas

1. **BOS token everywhere** (line 36):
   ```python
   token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
   ```
   - Training data: every doc has BOS
   - Evaluation: must also prepend BOS
   - **Mismatch = distribution shift!**

2. **needed_tokens calculation** (line 13):
   ```python
   needed_tokens = B * T + 1  # Off-by-one!
   ```
   - Need `B*T` inputs + `B*T` targets
   - Targets shifted by 1 → need 1 extra token
   - **Example**: `B=2, T=4` → need 9 tokens:
     ```
     tokens:  [0, 1, 2, 3, 4, 5, 6, 7, 8]
     inputs:  [0, 1, 2, 3 | 4, 5, 6, 7]     (8 tokens)
     targets: [1, 2, 3, 4 | 5, 6, 7, 8]     (8 tokens)
     ```

3. **Pinned memory requirement** (line 19):
   ```python
   scratch = torch.empty(needed_tokens, pin_memory=True)
   ```
   - Required for `non_blocking=True` transfer
   - Without pinning: slower transfer, no async benefit

4. **Dtype mismatch** (lines 44-45):
   ```python
   inputs_cpu = scratch[:-1].to(dtype=torch.int32)   # int32
   targets_cpu = scratch[1:]                          # int64
   ```
   - **Inputs**: int32 (saves memory, sufficient for vocab size)
   - **Targets**: int64 (required by `F.cross_entropy`)
   - Then converted on GPU (lines 47-48)

---

## 6. Tokenizer

**File**: [nanochat/tokenizer.py](../nanochat/tokenizer.py) (396 lines)

### 6.1 Two Implementations

1. **HuggingFace Tokenizer** (lines 39-148):
   - Training + inference
   - Slower but more flexible
   - Used for initial development

2. **RustBPE + tiktoken** (lines 150-257):
   - **Training**: rustbpe (Rust, fast)
   - **Inference**: tiktoken (C++, faster)
   - Production choice ⭐

**Usage**:
```python
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()  # Returns RustBPETokenizer
```

---

### 6.2 Special Tokens

**From tokenizer.py:13-25**:

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of sequence (document delimiter)
    "<|user_start|>",       # User message start
    "<|user_end|>",         # User message end
    "<|assistant_start|>",  # Assistant message start
    "<|assistant_end|>",    # Assistant message end
    "<|python_start|>",     # Tool use: Python REPL
    "<|python_end|>",       # Tool use end
    "<|output_start|>",     # Tool output start
    "<|output_end|>",       # Tool output end
]
```

**Chat format example**:
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>Let me calculate that.<|python_start|>2+2<|python_end|><|output_start|>4<|output_end|> The answer is 4.<|assistant_end|>
```

---

### 6.3 Conversation Rendering

**From tokenizer.py:258-342**:

```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    Tokenize a conversation for SFT training.
    Returns:
        ids: list[int] - token IDs
        mask: list[int] - 1=train on, 0=ignore
    """
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # Handle system messages (merge with first user message)
    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[1]["role"] == "user"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    else:
        messages = conversation["messages"]

    # Get special tokens
    bos = self.get_bos_token_id()
    user_start, user_end = self.encode_special("<|user_start|>"), ...
    assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), ...
    python_start, python_end = self.encode_special("<|python_start|>"), ...
    output_start, output_end = self.encode_special("<|output_start|>"), ...

    # Tokenize
    add_tokens(bos, 0)  # BOS not supervised
    for i, message in enumerate(messages):
        if message["role"] == "user":
            value_ids = self.encode(message["content"])
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)  # User messages not supervised
            add_tokens(user_end, 0)

        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            content = message["content"]

            if isinstance(content, str):
                # Simple text response
                value_ids = self.encode(content)
                add_tokens(value_ids, 1)  # ⭐ Supervised!

            elif isinstance(content, list):
                # Structured content (text + tool calls)
                for part in content:
                    value_ids = self.encode(part["text"])

                    if part["type"] == "text":
                        add_tokens(value_ids, 1)  # Supervised

                    elif part["type"] == "python":
                        # <|python_start|>expression<|python_end|>
                        add_tokens(python_start, 1)
                        add_tokens(value_ids, 1)  # Supervised
                        add_tokens(python_end, 1)

                    elif part["type"] == "python_output":
                        # <|output_start|>result<|output_end|>
                        add_tokens(output_start, 0)
                        add_tokens(value_ids, 0)  # NOT supervised
                        add_tokens(output_end, 0)

            add_tokens(assistant_end, 1)  # Supervised

    # Truncate to max_tokens
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]

    return ids, mask
```

---

### 6.4 Mask Logic

**Supervision rules**:

| Token Type | Mask | Why |
|------------|------|-----|
| `<|bos|>` | 0 | Document delimiter |
| User message content | 0 | Don't train to predict user input |
| `<|user_start|>`, `<|user_end|>` | 0 | Markup tokens |
| `<|assistant_start|>` | 0 | Markup token |
| Assistant text | **1** | ⭐ Main training signal |
| `<|python_start|>`, `<|python_end|>` | **1** | Tool invocation |
| Python expression | **1** | Tool invocation |
| `<|output_start|>`, `<|output_end|>` | 0 | Tool results (external) |
| Python output | 0 | Not generated by model |
| `<|assistant_end|>` | **1** | End-of-turn marker |

**Training loss**:
```python
# In training loop
ids, mask = tokenizer.render_conversation(conversation)
inputs = torch.tensor(ids[:-1])
targets = torch.tensor(ids[1:])
targets[mask[1:] == 0] = -1  # Ignore index

loss = F.cross_entropy(logits.view(-1, vocab_size),
                      targets.view(-1),
                      ignore_index=-1)  # Skips -1 targets
```

---

### 6.5 Rendering for RL Completion

**From tokenizer.py:356-374**:

```python
def render_for_completion(self, conversation):
    """
    Render conversation priming assistant for completion.
    Used in RL training - we want model to generate the last assistant message.
    """
    # Remove last assistant message
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant"
    messages.pop()  # Remove last message

    # Tokenize remaining conversation
    ids, mask = self.render_conversation(conversation)

    # Append assistant_start to prime for completion
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)

    return ids  # No mask needed for RL
```

**Example**:
```python
# Original conversation
conversation = {
    "messages": [
        {"role": "user", "content": "What is 5*7?"},
        {"role": "assistant", "content": "35"}  # <-- Remove this
    ]
}

# After render_for_completion
tokens = [<|bos|>, <|user_start|>, ..., <|user_end|>, <|assistant_start|>]
#                                                      ↑
#                                                   Model completes from here
```

---

### 6.6 Bug-Prone Areas

1. **System message assumptions** (lines 274-281):
   ```python
   if conversation["messages"][0]["role"] == "system":
       assert messages[1]["role"] == "user", "System message must be followed by user"
   ```
   - Assumes system message followed by user message
   - Crashes if system message is last!

2. **Alternating user/assistant** (lines 297-299):
   ```python
   must_be_from = "user" if i % 2 == 0 else "assistant"
   assert message["role"] == must_be_from
   ```
   - Enforces strict alternation
   - Can't have consecutive user or assistant messages

3. **Truncation loses alignment** (lines 339-341):
   ```python
   ids = ids[:max_tokens]
   mask = mask[:max_tokens]
   ```
   - Silently truncates long conversations
   - Can cut mid-message → mask misaligned with targets!
   - **Better**: Reject conversations > max_tokens

4. **Tool output not supervised** (lines 328-332):
   ```python
   elif part["type"] == "python_output":
       add_tokens(output_start, 0)
       add_tokens(value_ids, 0)  # mask=0
       add_tokens(output_end, 0)
   ```
   - Tool outputs come from Python, not model
   - Model should predict tool **invocation**, not results
   - Training on results = data leak!

---

## 7. Inference Engine

**File**: [nanochat/engine.py](../nanochat/engine.py) (344 lines)

### 7.1 KV Cache Implementation

**From engine.py:56-124**:

```python
class KVCache:
    """
    Maintains key/value cache for efficient autoregressive generation.
    Works hand-in-hand with GPT model.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Shape: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        #        ↑           ↑   ↑           ↑          ↑        ↑
        #        layers      K/V batch       heads      time     dim
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None  # Lazy initialization
        self.pos = 0          # Current position in cache

    def reset(self):
        """Reset position to start of cache"""
        self.pos = 0

    def get_pos(self):
        return self.pos

    def insert_kv(self, layer_idx, k, v):
        """Insert new keys/values and return full cache view"""
        # Lazy init (need to know dtype/device from first call)
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add

        # Dynamic growth if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # Buffer of 1024
            t_needed = (t_needed + 1023) & ~1023  # Round up to multiple of 1024
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)

        # Insert k, v
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k  # Keys
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v  # Values

        # Return views of full cache
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

        # Increment position after last layer
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1

        return key_view, value_view
```

**Key design decisions**:

1. **Lazy initialization** (lines 103-104):
   - Defers allocation until first use
   - Allows meta device initialization
   - Gets dtype/device from first tensor

2. **Dynamic growth** (lines 109-114):
   - Cache grows automatically if needed
   - Grows in chunks of 1024 for efficiency
   - Prevents OOM from fixed-size allocation

3. **Position tracking** (lines 122-123):
   - `.pos` incremented after last layer
   - Ensures all layers process before advancing
   - Used for RoPE offset in model.forward()

4. **Views, not copies** (lines 119-120):
   - Returns views into cache
   - No memory copying
   - Efficient even for large caches

---

### 7.2 Two-Phase Generation

**From engine.py:164-268**:

```python
@torch.inference_mode()
def generate(self, tokens, num_samples=1, max_tokens=None,
             temperature=1.0, top_k=None, seed=42):
    """
    Efficient batched generation with KV cache.
    1. Prefill: Process prompt once (batch=1)
    2. Decode: Replicate cache and generate num_samples in parallel
    """

    # Setup
    device = self.model.get_device()
    rng = torch.Generator(device=device).manual_seed(seed)

    # Get special tokens
    python_start = self.tokenizer.encode_special("<|python_start|>")
    python_end = self.tokenizer.encode_special("<|python_end|>")
    output_start = self.tokenizer.encode_special("<|output_start|>")
    output_end = self.tokenizer.encode_special("<|output_end|>")
    assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
    bos = self.tokenizer.get_bos_token_id()

    # --- Phase 1: Prefill (batch=1) ---
    m = self.model.config
    kv_cache_prefill = KVCache(
        batch_size=1,
        seq_len=len(tokens),
        num_heads=m.n_kv_head,
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
    )
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
    logits = logits[:, -1, :]  # Last token logits
    next_ids = sample_next_token(logits, rng, temperature, top_k)
    sampled_tokens = next_ids[:, 0].tolist()

    # --- Phase 2: Decode (batch=num_samples) ---
    kv_length_hint = len(tokens) + max_tokens if max_tokens else m.sequence_len
    kv_cache_decode = KVCache(
        batch_size=num_samples,
        seq_len=kv_length_hint,
        num_heads=m.n_kv_head,
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
    )
    kv_cache_decode.prefill(kv_cache_prefill)  # Copy prefill cache
    del kv_cache_prefill  # Free memory

    # Initialize per-row state
    row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

    # Main generation loop
    num_generated = 0
    first_iteration = True
    while True:
        # Stop conditions
        if max_tokens and num_generated >= max_tokens:
            break
        if all(state.completed for state in row_states):
            break

        # Get sampled tokens
        if first_iteration:
            # Broadcast first token to all samples
            sampled_tokens = [sampled_tokens[0]] * num_samples
            # TODO: should sample different token for each row
            first_iteration = False
        else:
            # Forward and sample
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)
            logits = logits[:, -1, :]
            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

        # Process each row
        token_column = []
        token_masks = []
        for i, state in enumerate(row_states):
            # Choose next token (forced or sampled)
            is_forced = len(state.forced_tokens) > 0
            token_masks.append(0 if is_forced else 1)
            next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
            token_column.append(next_token)

            # Update state
            state.current_tokens.append(next_token)

            # Check for completion
            if next_token == assistant_end or next_token == bos:
                state.completed = True

            # Tool use state machine
            if next_token == python_start:
                state.in_python_block = True
                state.python_expr_tokens = []
            elif next_token == python_end and state.in_python_block:
                state.in_python_block = False
                if state.python_expr_tokens:
                    expr = self.tokenizer.decode(state.python_expr_tokens)
                    result = use_calculator(expr)  # Evaluate
                    if result is not None:
                        result_tokens = self.tokenizer.encode(str(result))
                        # Force inject output
                        state.forced_tokens.append(output_start)
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(output_end)
                state.python_expr_tokens = []
            elif state.in_python_block:
                state.python_expr_tokens.append(next_token)

        # Yield token column
        yield token_column, token_masks
        num_generated += 1

        # Prepare for next iteration
        ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
```

---

### 7.3 Why Two Phases?

**Without prefill optimization**:
```
Forward 1: [token_1, token_2, ..., token_N]         (N tokens)
Forward 2: [token_1, ..., token_N, new_token_1]     (N+1 tokens)
Forward 3: [token_1, ..., token_N, new_1, new_2]    (N+2 tokens)
...
Total: N + (N+1) + (N+2) + ... ≈ O(N²) operations
```

**With prefill + KV cache**:
```
Prefill:   [token_1, token_2, ..., token_N]         (N tokens, cache filled)
Decode 1:  [new_token_1]                            (1 token, use cache)
Decode 2:  [new_token_2]                            (1 token, use cache)
...
Total: N + 1 + 1 + ... ≈ O(N + M) operations (M = generated tokens)
```

**Speedup**: ~10-100x for long prompts!

---

### 7.4 Tool Use State Machine

**From engine.py:148-156, 246-261**:

```python
class RowState:
    """Per-row state during generation"""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()      # Queue of tokens to inject
        self.in_python_block = False      # Inside <|python_start|>...<|python_end|>
        self.python_expr_tokens = []      # Tokens of expression
        self.completed = False            # Hit <|assistant_end|> or <|bos|>
```

**State transitions**:
```
Normal generation
    ↓ (sample <|python_start|>)
Collecting expression tokens
    ↓ (sample <|python_end|>)
Evaluate expression → inject <|output_start|>result<|output_end|>
    ↓
Force inject output tokens
    ↓
Resume normal generation
```

**Example**:
```
Prompt: "What is 123 * 456?"
Model samples: Let me calculate <|python_start|>123*456<|python_end|>
                                                  ↑
                                          Collected: [123, *, 456]
Engine evaluates: 123*456 = 56088
Engine injects: <|output_start|>56088<|output_end|>
Model continues: The answer is 56088.
```

**Calculator function** (engine.py:46-53):
```python
def use_calculator(expr):
    """Safely evaluate math expressions"""
    expr = expr.replace(",", "")
    # Only allow: 0-9, +, -, *, /, ., (, ), space
    if any([x not in "0123456789*+-/.() " for x in expr]):
        return None
    if "**" in expr:  # Disallow power (can be expensive)
        return None
    return eval_with_timeout(expr, max_time=3)
```

---

### 7.5 Bug-Prone Areas

1. **First token broadcast** (lines 219-222):
   ```python
   if first_iteration:
       sampled_tokens = [sampled_tokens[0]] * num_samples
       # TODO: should sample different token for each row
   ```
   - **Known limitation**: All samples get same first token!
   - Reduces diversity in batch generation
   - Fix: Sample `num_samples` tokens from prefill logits

2. **Forced token masking** (lines 236-238):
   ```python
   is_forced = len(state.forced_tokens) > 0
   token_masks.append(0 if is_forced else 1)
   next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
   ```
   - Must track which tokens were forced vs sampled
   - Mask=0 for forced → not trained on in RL
   - Deque empty check prevents crash

3. **Completion tracking** (lines 242-244):
   ```python
   if next_token == assistant_end or next_token == bos:
       state.completed = True
   ```
   - Must stop generation on special tokens
   - If model never emits these, runs to `max_tokens`
   - Can waste compute on already-complete sequences

4. **Tool evaluation safety** (engine.py:46-53):
   ```python
   if any([x not in "0123456789*+-/.() " for x in expr]):
       return None  # Reject unsafe expressions
   ```
   - **Security**: Never `eval()` arbitrary code!
   - Whitelist approach: only allow math operators
   - Timeout prevents infinite loops

5. **Cache device/dtype** (engine.py:94, 104):
   ```python
   # Lazy init gets dtype/device from first tensor
   self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
   ```
   - Cache must match model dtype (bfloat16)
   - Must be on same device as model
   - Mismatch → runtime error or silent failures

---

## 8. Common Bug Hotspots

### 8.1 Device/Dtype Mismatches

**Critical locations**:

1. **Rotary embeddings** (gpt.py:265):
   ```python
   assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be bfloat16"
   assert idx.device == self.cos.device  
   ```

NEP Let me guess: it defaults float32, but we want less computational cost here (in sacrifice of precise), so we go to float16, but we still need the range of float32, so we choose bfloat16. True?

2. **KV cache** (engine.py:94):
   ```python
   self.kv_cache = torch.empty(..., dtype=k.dtype, device=k.device)
   ```
NE

3. **Dataloader dtypes** (dataloader.py:44-48):
   ```python
   inputs = inputs_cpu.to(device="cuda", dtype=torch.int32)   # int32 for inputs
   targets = targets_cpu.to(device="cuda", dtype=torch.int64)  # int64 for targets
   ```

**Common errors**:
- FP32 rotary embeddings → precision mismatch
- CPU tensors in forward pass → slow/crash
- int64 inputs → unnecessary memory usage

---

### 8.2 Off-by-One Errors

**Critical locations**:

1. **Dataloader** (dataloader.py:13):
   ```python
   needed_tokens = B * T + 1  # +1 for autoregressive target!
   ```

2. **Autoregressive targets** (base_train.py:259, chat_rl.py:131):
   ```python
   inputs = ids[:, :-1]   # All but last token
   targets = ids[:, 1:]   # All but first token
   ```

3. **Attention mask** (gpt.py:116):
   ```python
   prefix_len = Tk - Tq  # Can be 0!
   if prefix_len > 0:
       attn_mask[:, :prefix_len] = True
   ```

4. **RoPE offset** (gpt.py:267):
   ```python
   T0 = 0 if kv_cache is None else kv_cache.get_pos()
   cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
   ```

**Testing strategy**:
- Edge case: `T=1` (single token)
- Edge case: `prefix_len=0` (no cached prefix)
- Verify shapes match at each step

---

### 8.3 Gradient Accumulation

**Critical locations**:

1. **Loss normalization** (base_train.py:261):
   ```python
   loss = loss / grad_accum_steps  # BEFORE .backward()!
   ```
   - **Wrong**: `loss.backward(); loss = loss / N`
   - **Right**: `loss = loss / N; loss.backward()`

2. **Clipping timing** (base_train.py:265-266):
   ```python
   # After all .backward() calls, before .step()
   if grad_clip > 0.0:
       torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   ```

3. **Zero gradients** (base_train.py:277):
   ```python
   model.zero_grad(set_to_none=True)  # After .step(), before next iteration
   ```

**Common errors**:
- Forgetting to divide loss
- Clipping before all gradients accumulated
- Not zeroing gradients between steps

---

### 8.4 Distributed Training

**Critical locations**:

1. **Data sharding** (dataloader.py:25, chat_rl.py:81):
   ```python
   for batch in parquets_iter_batched(start=ddp_rank, step=ddp_world_size):
   ```
   - Each rank sees different data
   - No overlap → important for batch diversity

2. **Reductions** (chat_rl.py:229-231):
   ```python
   if ddp:
       dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
       dist.all_reduce(passk, op=dist.ReduceOp.SUM)
   ```
   - Evaluation metrics must be reduced across ranks
   - Training loss already averaged by DDP

3. **Print statements** (common.py:59-62):
   ```python
   def print0(s="", **kwargs):
       ddp_rank = int(os.environ.get('RANK', 0))
       if ddp_rank == 0:  # Only master rank prints
           print(s, **kwargs)
   ```

4. **Checkpointing** (base_train.py:231):
   ```python
   if master_process and last_step:  # Only master saves
       save_checkpoint(...)
   ```

**Common errors**:
- All ranks printing → spammy logs
- All ranks saving → file conflicts
- Forgetting to reduce eval metrics

---

### 8.5 Memory Management

**Critical locations**:

1. **KV cache growth** (engine.py:108-114):
   ```python
   if t1 > self.kv_cache.size(4):
       t_needed = t1 + 1024
       t_needed = (t_needed + 1023) & ~1023  # Round up
       self.kv_cache.resize_(current_shape)
   ```
   - Grows dynamically → can OOM
   - Must call `.reset()` between generations

2. **Torch compile** (base_train.py:102):
   ```python
   model = torch.compile(model)
   ```
   - Compilation overhead: ~1-2 GB
   - First forward pass: graph tracing
   - Subsequent passes: faster

3. **Gradient checkpointing** (not implemented):
   - nanochat doesn't use it
   - Trade compute for memory
   - Consider for large models

4. **Autocast context** (base_train.py:62):
   ```python
   autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
   ```
   - Saves memory (bfloat16 vs fp32)
   - Maintains numerical stability (fp32 for reductions)

**OOM debugging**:
```python
# Check peak memory
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Reset peak tracker
torch.cuda.reset_peak_memory_stats()

# Monitor during training
torch.cuda.memory_summary()
```

---

### 8.6 Tokenization Edge Cases

**Critical locations**:

1. **System messages** (tokenizer.py:274-281):
   ```python
   if conversation["messages"][0]["role"] == "system":
       assert messages[1]["role"] == "user"  # Can crash!
   ```

2. **Alternation** (tokenizer.py:298-299):
   ```python
   must_be_from = "user" if i % 2 == 0 else "assistant"
   assert message["role"] == must_be_from  # Strict!
   ```

3. **Truncation** (tokenizer.py:339-341):
   ```python
   ids = ids[:max_tokens]    # Silent truncation
   mask = mask[:max_tokens]  # Can lose alignment
   ```

4. **Tool output supervision** (tokenizer.py:328-332):
   ```python
   elif part["type"] == "python_output":
       add_tokens(value_ids, 0)  # mask=0 → not supervised
   ```

**Testing strategy**:
- Empty conversations
- Single-message conversations
- Very long conversations (> max_tokens)
- Malformed tool calls

---

## 9. Learning Path

### Phase 1: Core Architecture (2-3 hours)

**Goal**: Understand the transformer from scratch

1. **Read [nanochat/gpt.py](../nanochat/gpt.py)** (323 lines)
   - Start with `GPTConfig` dataclass
   - Trace `forward()` method step-by-step
   - Understand `CausalSelfAttention.forward()`
   - Study RoPE implementation

2. **Hands-on exercise**:
   ```python
   from nanochat.gpt import GPT, GPTConfig
   import torch

   # Create tiny model
   config = GPTConfig(n_layer=2, n_head=2, n_embd=128, vocab_size=100)
   model = GPT(config)
   model.init_weights()

   # Forward pass
   idx = torch.randint(0, 100, (1, 10))  # Batch=1, seq_len=10
   logits = model(idx)
   print(f"Logits shape: {logits.shape}")  # Should be (1, 10, 100)

   # With targets
   targets = torch.randint(0, 100, (1, 10))
   loss = model(idx, targets)
   print(f"Loss: {loss.item()}")
   ```

3. **Read [nanochat/engine.py](../nanochat/engine.py)** (344 lines)
   - Focus on `KVCache` class
   - Understand `insert_kv()` and dynamic growth
   - Trace `Engine.generate()` two-phase approach

4. **Hands-on exercise**:
   ```python
   from nanochat.engine import KVCache
   import torch

   # Create cache
   cache = KVCache(batch_size=2, num_heads=4, seq_len=100,
                   head_dim=32, num_layers=2)

   # Simulate insertion
   k = torch.randn(2, 4, 5, 32)  # 5 new keys
   v = torch.randn(2, 4, 5, 32)

   for layer_idx in range(2):
       k_full, v_full = cache.insert_kv(layer_idx, k, v)
       print(f"Layer {layer_idx}: cache now has {k_full.size(2)} keys")

   print(f"Cache position: {cache.get_pos()}")  # Should be 5
   ```

---

### Phase 2: Training Loop (2-3 hours)

**Goal**: Understand how training works end-to-end

1. **Read [scripts/base_train.py](../scripts/base_train.py)** (340 lines)
   - Understand model sizing (lines 74-82)
   - Study dual optimizer setup (lines 129-131)
   - Trace gradient accumulation loop (lines 257-263)
   - Understand learning rate scheduling (lines 148-163)

2. **Read [nanochat/dataloader.py](../nanochat/dataloader.py)** (50 lines)
   - Understand streaming architecture
   - Trace token buffer logic
   - See how BOS tokens are prepended

3. **Hands-on exercise**:
   ```python
   # Simulate gradient accumulation
   model = ...  # Your model
   optimizer = ...
   grad_accum_steps = 4

   for step in range(num_steps):
       for micro_step in range(grad_accum_steps):
           x, y = next(data_loader)
           loss = model(x, y)
           loss = loss / grad_accum_steps  # Normalize!
           loss.backward()

       optimizer.step()
       model.zero_grad(set_to_none=True)
   ```

4. **Calculate FLOPs**:
   ```python
   # From base_train.py
   num_params = sum(p.numel() for p in model.parameters())
   num_flops_per_token = model.estimate_flops()
   total_flops = num_flops_per_token * total_tokens

   print(f"Model size: {num_params/1e6:.1f}M params")
   print(f"FLOPs per token: {num_flops_per_token:e}")
   print(f"Total training FLOPs: {total_flops:e}")
   ```

---

### Phase 3: Advanced Topics (3-4 hours)

**Goal**: Understand RL, tokenization, and tools

1. **Read [scripts/chat_rl.py](../scripts/chat_rl.py)** (332 lines)
   - Understand rollout generation (lines 79-140)
   - Study policy gradient calculation (lines 259-269)
   - See how advantages are computed (lines 136-138)

2. **Read [nanochat/tokenizer.py](../nanochat/tokenizer.py)** (396 lines)
   - Understand `render_conversation()` (lines 258-342)
   - Study mask logic for different content types
   - See `render_for_completion()` for RL (lines 356-374)

3. **Hands-on exercise**:
   ```python
   from nanochat.tokenizer import get_tokenizer

   tokenizer = get_tokenizer()

   # Example conversation
   conversation = {
       "messages": [
           {"role": "user", "content": "What is 2+2?"},
           {"role": "assistant", "content": "The answer is 4."}
       ]
   }

   # Render and visualize
   ids, mask = tokenizer.render_conversation(conversation)
   print(f"Num tokens: {len(ids)}")
   print(f"Supervised tokens: {sum(mask)}")

   # Visualize (colors in terminal)
   viz = tokenizer.visualize_tokenization(ids, mask)
   print(viz)
   ```

4. **Study tool use**:
   - Read calculator implementation (engine.py:46-53)
   - Trace state machine (engine.py:246-261)
   - Test with Python expressions

---

### Phase 4: Debugging & Experimentation (ongoing)

**Goal**: Develop intuition through hands-on debugging

1. **Set breakpoints**:
   ```python
   # In gpt.py:forward()
   import pdb; pdb.set_trace()

   # Inspect shapes
   print(f"x.shape: {x.shape}")
   print(f"cos_sin[0].shape: {cos_sin[0].shape}")
   ```

2. **Visualize attention**:
   ```python
   # In CausalSelfAttention.forward()
   # After computing attention scores
   attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(head_dim), dim=-1)

   import matplotlib.pyplot as plt
   plt.imshow(attn_weights[0, 0].cpu().detach().numpy())
   plt.colorbar()
   plt.title("Attention weights (head 0)")
   plt.show()
   ```

3. **Check gradient flow**:
   ```python
   # After loss.backward()
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
       else:
           print(f"{name}: NO GRADIENT!")
   ```

4. **Profile performance**:
   ```python
   import torch.profiler as profiler

   with profiler.profile(
       activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
       record_shapes=True,
   ) as prof:
       for step in range(10):
           loss = model(x, y)
           loss.backward()

   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   ```

---

## 10. Quick Reference

### 10.1 File Purposes

| File | Purpose | LOC | Key Functions |
|------|---------|-----|---------------|
| [gpt.py](../nanochat/gpt.py) | Transformer model | 323 | `GPT.forward()`, `CausalSelfAttention.forward()` |
| [engine.py](../nanochat/engine.py) | Inference + KV cache | 344 | `KVCache.insert_kv()`, `Engine.generate()` |
| [base_train.py](../scripts/base_train.py) | Pretraining | 340 | Main training loop |
| [chat_rl.py](../scripts/chat_rl.py) | RL training | 332 | `get_batch()`, policy gradient |
| [tokenizer.py](../nanochat/tokenizer.py) | BPE tokenizer | 396 | `render_conversation()`, `render_for_completion()` |
| [dataloader.py](../nanochat/dataloader.py) | Streaming data | 50 | `tokenizing_distributed_data_loader()` |
| [common.py](../nanochat/common.py) | Utilities | 137 | `compute_init()`, `get_dist_info()` |
| [muon.py](../nanochat/muon.py) | Muon optimizer | 243 | `Muon`, `DistMuon` |
| [adamw.py](../nanochat/adamw.py) | AdamW optimizer | 88 | `DistAdamW` |

**Total**: ~8,300 LOC across 44 files

---

### 10.2 Model Configurations

**Speedrun (d20)**:
```python
depth = 20
model_dim = 1280  # 20 * 64
num_heads = 10    # ceil(1280 / 128)
params = ~240M
training_tokens = 4.8B  # 20x Chinchilla
cost = ~$100 (4 hours on 8xH100)
```

**d26 (GPT-2 grade)**:
```python
depth = 26
model_dim = 1664
num_heads = 13
params = ~400M
training_tokens = 8B
cost = ~$300 (12 hours)
```

**d32 (hosted demo)**:
```python
depth = 32
model_dim = 2048
num_heads = 16
params = ~1.9B
training_tokens = 38B
cost = ~$800 (33 hours)
```

---

### 10.3 Hyperparameter Cheat Sheet

**Pretraining (base_train.py)**:
```python
# Model
depth = 20
max_seq_len = 2048

# Batch sizes
device_batch_size = 32      # Per GPU (tune to not OOM)
total_batch_size = 524288   # Total tokens per step

# Learning rates
embedding_lr = 0.2          # AdamW for embeddings
unembedding_lr = 0.004      # AdamW for lm_head
matrix_lr = 0.02            # Muon for transformer blocks

# Training length
target_param_data_ratio = 20  # Chinchilla optimal

# Scheduling
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0

# Regularization
weight_decay = 0.0
grad_clip = 1.0
```

**RL (chat_rl.py)**:
```python
# Sampling
device_batch_size = 8
examples_per_step = 16      # Questions per step
num_samples = 16            # Rollouts per question
max_new_tokens = 256
temperature = 1.0
top_k = 50

# Learning rates (fraction of base)
init_lr_frac = 0.05         # Start at 5%
# Then linear decay to 0

# Training
num_epochs = 1              # Through GSM8K (7473 examples)
```

---

### 10.4 Common Commands

**Training**:
```bash
# Single GPU
python -m scripts.base_train

# 8 GPUs
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# With custom config
python -m scripts.base_train -- --depth=26 --device_batch_size=16
```

**Inference**:
```bash
# CLI
python -m scripts.chat_cli

# Web UI
python -m scripts.chat_web
# Then visit http://localhost:8000
```

**Evaluation**:
```bash
# Base model CORE metric
python -m scripts.base_eval

# GSM8K pass@k
python -m scripts.chat_eval
```

---

### 10.5 Debugging Checklist

Before training:
- [ ] Data downloaded/tokenized
- [ ] Tokenizer trained and saved
- [ ] GPU memory sufficient (check with small model first)
- [ ] Distributed setup correct (RANK, WORLD_SIZE env vars)

During training:
- [ ] Loss decreasing (not NaN/inf)
- [ ] Validation BPB improving
- [ ] MFU > 30% (model FLOPs utilization)
- [ ] No CUDA OOM errors
- [ ] Checkpoints saving correctly

After training:
- [ ] Model generates coherent text
- [ ] CORE metric reasonable (>0.2 for speedrun)
- [ ] Checkpoint loadable for fine-tuning

---

### 10.6 Key Insights

**Architecture**:
- RoPE > learned positional embeddings
- QK norm stabilizes deep models
- MQA critical for inference speed
- ReLU² works as well as GELU
- Untied embeddings worth the memory

**Training**:
- Dual optimizers (Muon + AdamW) work best
- Gradient accumulation = parallel → sequential
- Chinchilla ratio (20:1 tokens:params) is good baseline
- No warmup needed, but warmdown helps
- BFloat16 saves memory without quality loss

**Inference**:
- KV cache is essential (10-100x speedup)
- Prefill once, decode many (batch generation)
- Dynamic cache growth prevents OOM
- Tool use via state machine (not separate model)

**RL**:
- Simplified GRPO (no reference model) works
- On-policy = no PPO clipping needed
- Advantage = (r - μ) sufficient (no division by σ)
- Token-level normalization better than sequence-level

**General**:
- Streaming data > epochs (never runs out)
- Distributed sharding important for diversity
- BOS token everywhere (train/eval consistency)
- Mask supervision carefully (tool outputs = no supervision)

---

## Appendix: Additional Resources

### Papers Referenced

1. **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. **QK Norm**: [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)
3. **MQA/GQA**: [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150), [GQA](https://arxiv.org/abs/2305.13245)
4. **Muon**: [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
5. **Chinchilla**: [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)
6. **GRPO**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

### Related Codebases

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Pretraining only
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) - Speedrun pretraining
- [tiktoken](https://github.com/openai/tiktoken) - BPE tokenizer

### Discussions

- [Introducing nanochat](https://github.com/karpathy/nanochat/discussions/1) - Speedrun walkthrough
- [nanochat d32 demo](https://github.com/karpathy/nanochat/discussions/8) - Hosted model

---

**End of Guide**

This guide covers the essential architecture, algorithms, and bug-prone areas in nanochat. Use it as a reference while exploring the codebase and before running training runs.

For questions or corrections, please open an issue in the nanochat repository.

Happy training! 🚀
