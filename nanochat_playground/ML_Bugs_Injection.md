# 🐛 ML Bug Injection Practice - Complete Guide

MUST DO: 
WHEN INJECT A NEW BUG, PLEAE DO IT SILIENTLY. DO NOT LEAVE A COMMENT SAYING THIS IS A BUG. JUST EDIT, DELETE, ADD A BUG A TIME AS YOU LIKE, FROM THIS LIST. AND DO NOT TELL ANYWHERE WHAT BUG YOU INJECTED.

## Overview

This guide contains **20 bugs** organized by difficulty. Each bug is injected **blindly** (you won't know where it is) to simulate real interview conditions.

## Two types of coding debugging interview 
### **Type 1: Coding Review (60% of interviews)**

Interviewer: "Here's the training code. Find the bug."
[Shows 20-30 lines of code]
You: [Read code, spot bug by inspection]
Time: 5-10 min

NO logs provided - pure code reading.


### **Type 2: Production debugging (40% of interviews)**

Interviewer: "Training loss became NaN at step 347. Here are the logs:"
[Shows basic logs: loss values, maybe learning rate]

You: "I need to see gradient norms"
Interviewer: [Provides grad norms]
You: "I see grad norm spiked. Let me check attention scores"
Interviewer: [Provides attention diagnostics]

You propose what to check, they give you the info.
Time: 15-20 min

---

## Setup (On GPU Machine)

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_USERNAME/nanochat.git
cd nanochat

# 2. Setup environment
source .venv/bin/activate  # or create new venv

# 3. Run baseline (IMPORTANT!)
bash speedrun.sh
# Or shorter: python scripts/base_train.py --num_iterations=1000

# 4. Save baseline metrics
cat > neplocal/baseline.txt << EOF
Step 1000:
- Training loss: [copy from terminal]
- Validation bpb: [copy from terminal]
- CORE metric: [copy from terminal]
- Sample quality: [good/bad]
EOF
```

---

## Bug List (20 Bugs)

# Complete Bug List for Nanochat (Balanced Distribution)

**Total: 54 bugs across all categories**

Distribution target:
- 40% Training/Optimization (20 bugs)
- 30% Model Architecture (15 bugs)
- 20% Data Pipeline (10 bugs)
- 10% Inference/Evaluation (5 bugs)

Priority distribution:
- Priority 1 (Must Know): 60% of bugs (30 bugs)
- Priority 2 (Should Know): 30% of bugs (15 bugs)
- Priority 3 (Good to Know): 10% of bugs (5 bugs)

---

## **TRAINING & OPTIMIZATION BUGS (20 bugs - 40%)**

### **Already Covered (8 bugs):**
1. Forgot `optimizer.zero_grad()` - P1, ⭐
2. Wrong gradient accumulation scaling - P1, ⭐⭐
3. Model in eval mode during training - P1, ⭐
8. Data not shuffled - P1, ⭐
13. Learning rate not decayed - P2, ⭐⭐
14. Validation gradients not disabled - P1, ⭐
12. Loss not averaged in distributed training - P2, ⭐⭐⭐
16. Memory leak from loss accumulation - P2, ⭐⭐⭐

### **NEW Training/Optimization Bugs (12 bugs):**

---

### **Bug #21: Gradient clipping applied after optimizer step**
- **Category**: Gradient/Optimization
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: `scripts/base_train.py`
- **Location**: After `optimizer.step()`
- **Change**: Move `clip_grad_norm_()` to after `optimizer.step()`
- **Expected Symptom**:
  - Gradients not actually clipped
  - Training unstable, occasional spikes
  - Clipping has no effect
- **Debugging Hint**: Check order of operations in training loop
```python
# WRONG:
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Too late!

# CORRECT:
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### **Bug #22: Learning rate not scaled for distributed training**
- **Category**: Gradient/Optimization
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Optimizer initialization
- **Change**: Don't scale LR by world_size
- **Expected Symptom**:
  - Single GPU training works fine
  - Multi-GPU training diverges or converges slower
  - Effective LR is wrong
- **Debugging Hint**: Compare single vs multi-GPU learning curves
```python
# Should be:
lr = base_lr * world_size  # or adjust batch size instead
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
```

---

### **Bug #23: Optimizer state not reset after loading checkpoint**
- **Category**: Gradient/Optimization
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Checkpoint loading
- **Change**: Load model weights but forget to load optimizer state
- **Expected Symptom**:
  - Training restarts with wrong momentum
  - Loss jumps after checkpoint load
  - Slower convergence after resume
- **Debugging Hint**: Check if optimizer state dict is loaded
```python
# WRONG:
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
# Missing: optimizer.load_state_dict(checkpoint['optimizer'])

# CORRECT:
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

---

### **Bug #24: Mixed precision loss scaling not updated**
- **Category**: Gradient/Optimization
- **Priority**: 2
- **Difficulty**: ⭐⭐⭐ Hard
- **File**: `scripts/base_train.py`
- **Location**: Mixed precision training loop
- **Change**: Forget to call `scaler.update()`
- **Expected Symptom**:
  - Gradients underflow in FP16
  - Training extremely slow or not learning
  - Loss scale never adjusts
- **Debugging Hint**: Check scaler state, monitor loss scale value
```python
# WRONG:
scaler.scale(loss).backward()
scaler.step(optimizer)
# Missing: scaler.update()

# CORRECT:
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### **Bug #25: Warmup steps applied to weight decay**
- **Category**: Gradient/Optimization
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Learning rate scheduler
- **Change**: Apply LR warmup to weight decay parameter
- **Expected Symptom**:
  - Model underfits during warmup
  - Different behavior early vs late training
- **Debugging Hint**: Check if weight decay is constant
```python
# Weight decay should typically stay constant, not follow LR schedule
# WRONG: Scaling weight decay with LR warmup
```

---

### **Bug #26: Global gradient norm computed incorrectly**
- **Category**: Gradient/Optimization  
- **Priority**: 3
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Gradient norm computation
- **Change**: Sum norms instead of sqrt of sum of squares
- **Expected Symptom**:
  - Gradient norm values wrong
  - Clipping threshold incorrect
- **Debugging Hint**: Check gradient norm formula
```python
# WRONG:
total_norm = sum(p.grad.norm() for p in model.parameters())

# CORRECT:
total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.parameters()))
```

---

### **Bug #27: Scheduler stepped before optimizer**
- **Category**: Training Loop
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: `scripts/base_train.py`
- **Location**: Training loop
- **Change**: Call `scheduler.step()` before `optimizer.step()`
- **Expected Symptom**:
  - Learning rate schedule off by one step
  - Subtle performance degradation
- **Debugging Hint**: Log LR and check timing
```python
# WRONG:
loss.backward()
scheduler.step()
optimizer.step()

# CORRECT:
loss.backward()
optimizer.step()
scheduler.step()
```

---

### **Bug #28: Training metrics not reset between epochs**
- **Category**: Training Loop
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: `scripts/base_train.py`
- **Location**: Epoch loop
- **Change**: Don't reset loss accumulator
- **Expected Symptom**:
  - Reported loss grows each epoch
  - Metrics meaningless
- **Debugging Hint**: Check if counters reset
```python
# WRONG:
total_loss = 0  # Outside epoch loop
for epoch in range(num_epochs):
    for batch in train_loader:
        total_loss += loss  # Keeps accumulating!

# CORRECT:
for epoch in range(num_epochs):
    total_loss = 0  # Reset each epoch
    for batch in train_loader:
        total_loss += loss
```

---

### **Bug #29: Checkpoint saved at wrong frequency**
- **Category**: Training Loop
- **Priority**: 3
- **Difficulty**: ⭐ Easy
- **File**: `scripts/base_train.py`
- **Location**: Checkpoint saving logic
- **Change**: Save every step instead of every N steps
- **Expected Symptom**:
  - Disk fills up
  - Training very slow (I/O bound)
- **Debugging Hint**: Check checkpoint directory size
```python
# WRONG:
for step in range(max_steps):
    train_step()
    save_checkpoint()  # Every step!

# CORRECT:
if step % checkpoint_interval == 0:
    save_checkpoint()
```

---

### **Bug #30: Early stopping checks training loss instead of validation**
- **Category**: Training Loop
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Early stopping logic
- **Change**: Check train_loss instead of val_loss
- **Expected Symptom**:
  - Never early stops (train loss keeps decreasing)
  - Or stops too early
  - Model overfits
- **Debugging Hint**: Check which loss is used for early stopping
```python
# WRONG:
if train_loss < best_loss:  # Should use val_loss!
    patience_counter = 0
else:
    patience_counter += 1
```

---

### **Bug #31: Model parameters frozen accidentally**
- **Category**: Training Loop
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: Model setup
- **Change**: Set `requires_grad=False` for all parameters
- **Expected Symptom**:
  - Loss doesn't decrease at all
  - Gradients are all None or zero
  - Model outputs constant
- **Debugging Hint**: Check `param.requires_grad` for model params
```python
# WRONG:
for param in model.parameters():
    param.requires_grad = False  # Freezes everything!

# CORRECT:
# Only freeze specific layers if needed
for param in model.embedding.parameters():
    param.requires_grad = False  # Only freeze embeddings
```

---

### **Bug #32: Training loop doesn't sync across GPUs**
- **Category**: Distributed Training
- **Priority**: 2
- **Difficulty**: ⭐⭐⭐ Hard
- **File**: `scripts/base_train.py`
- **Location**: Distributed training loop
- **Change**: Missing `dist.barrier()` at critical points
- **Expected Symptom**:
  - GPUs get out of sync
  - Random crashes
  - Inconsistent behavior
- **Debugging Hint**: Add barriers and check GPU synchronization
```python
# Need barriers before collective operations:
dist.barrier()  # Sync all GPUs
loss = model(batch)
```

---

## **MODEL ARCHITECTURE BUGS (15 bugs - 30%)**

### **Already Covered (7 bugs):**
4. Causal mask wrong diagonal - P1, ⭐⭐
5. Attention scores not scaled - P1, ⭐⭐
6. Using -1e9 instead of -inf in mask - P1, ⭐⭐
7. Position indices not on correct device - P1, ⭐
9. KV cache position not incremented - P2, ⭐⭐⭐
10. RoPE not applied to Q and K - P2, ⭐⭐⭐
11. QK normalization missing - P2, ⭐⭐

### **NEW Model Architecture Bugs (8 bugs):**

---

### **Bug #33: Attention mask shape doesn't broadcast**
- **Category**: Model Architecture
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/model.py`
- **Location**: Attention mask application
- **Change**: Mask is `(seq_len, seq_len)` instead of `(1, 1, seq_len, seq_len)`
- **Expected Symptom**:
  - Broadcasting error
  - Or silent wrong attention (if shapes happen to match)
- **Debugging Hint**: Print shapes of mask and scores
```python
# WRONG:
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # (seq_len, seq_len)
scores = scores + mask  # Doesn't broadcast correctly over batch and heads!

# CORRECT:
mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

---

### **Bug #34: Residual connection added before projection**
- **Category**: Model Architecture
- **Priority**: 2
- **Difficulty**: ⭐⭐⭐ Hard
- **File**: `nanochat/model.py`
- **Location**: Transformer block
- **Change**: Add residual before output projection
- **Expected Symptom**:
  - Subtle: model trains but slower
  - Different gradient flow
  - Harder to train deep models
- **Debugging Hint**: Trace tensor flow through block
```python
# WRONG:
def forward(self, x):
    attn_out = self.attention(x)
    x = x + attn_out  # Added too early!
    return self.output_proj(x)

# CORRECT:
def forward(self, x):
    attn_out = self.attention(x)
    attn_out = self.output_proj(attn_out)
    return x + attn_out
```

---

### **Bug #35: Layer normalization applied after residual**
- **Category**: Model Architecture
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/model.py`
- **Location**: Transformer block
- **Change**: Post-norm instead of pre-norm
- **Expected Symptom**:
  - Training instability (especially for deep models)
  - Harder to train
  - Gradient flow issues
- **Debugging Hint**: Check norm placement (before or after residual)
```python
# Modern transformers use PRE-NORM (more stable):
def forward(self, x):
    x = x + self.attention(self.norm1(x))  # Norm BEFORE attention
    x = x + self.ffn(self.norm2(x))        # Norm BEFORE FFN
    return x

# POST-NORM (less stable, but original Transformer used this):
def forward(self, x):
    x = self.norm1(x + self.attention(x))  # Norm AFTER residual
    x = self.norm2(x + self.ffn(x))
    return x
```

---

### **Bug #36: Dropout not applied in MLP**
- **Category**: Model Architecture
- **Priority**: 2
- **Difficulty**: ⭐ Easy
- **File**: `nanochat/model.py`
- **Location**: FFN/MLP layer
- **Change**: Comment out dropout layer
- **Expected Symptom**:
  - Model overfits more
  - Train/val gap larger
  - No regularization
- **Debugging Hint**: Check if dropout exists in FFN
```python
# Should have dropout:
def forward(self, x):
    x = self.fc1(x)
    x = self.activation(x)
    x = self.dropout(x)  # Missing in bug!
    x = self.fc2(x)
    return x
```

---

### **Bug #37: Position embeddings exceed max length**
- **Category**: Model Architecture
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: `nanochat/model.py`
- **Location**: Position embedding lookup
- **Change**: Don't check if sequence length exceeds max_position
- **Expected Symptom**:
  - IndexError when seq_len > max_position
  - Crash during inference with long sequences
- **Debugging Hint**: Check max_position vs actual sequence length
```python
# Need to check:
assert seq_len <= self.max_position, f"Sequence length {seq_len} exceeds max position {self.max_position}"

# Or handle gracefully:
positions = torch.arange(min(seq_len, self.max_position))
```

---

### **Bug #38: Multi-head attention heads not properly separated**
- **Category**: Model Architecture
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/model.py`
- **Location**: Attention head splitting
- **Change**: Wrong reshape/view operation
- **Expected Symptom**:
  - Heads are not independent
  - Model doesn't learn properly
  - Shape errors
- **Debugging Hint**: Print shapes after head splitting
```python
# WRONG:
Q = self.W_q(x).view(B, T, self.n_heads * self.d_head)  # Not split!

# CORRECT:
Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
```

---

### **Bug #39: Output projection missing in attention**
- **Category**: Model Architecture
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/model.py`
- **Location**: Attention module
- **Change**: Comment out `self.W_o` projection
- **Expected Symptom**:
  - Model trains but performance poor
  - Limited capacity (no mixing across heads)
- **Debugging Hint**: Check if attention has output projection
```python
# WRONG:
def forward(self, x):
    # ... compute attention ...
    out = attn @ V
    return out.transpose(1, 2).contiguous().view(B, T, C)
    # Missing: return self.W_o(out)

# CORRECT:
out = out.transpose(1, 2).contiguous().view(B, T, C)
return self.W_o(out)
```

---

### **Bug #40: Bias added in linear layers when using RMSNorm**
- **Category**: Model Architecture
- **Priority**: 3
- **Difficulty**: ⭐ Easy
- **File**: `nanochat/model.py`
- **Location**: Linear layer initialization
- **Change**: `bias=True` in Linear layers
- **Expected Symptom**:
  - Slight performance degradation
  - Unnecessary parameters
  - Bias made redundant by norm
- **Debugging Hint**: Check Linear layer bias parameter
```python
# Should be:
self.W_q = nn.Linear(d_model, d_model, bias=False)

# Not:
self.W_q = nn.Linear(d_model, d_model, bias=True)  # Redundant with norm!
```

---

## **DATA PIPELINE BUGS (10 bugs - 20%)**

### **Already Covered (2 bugs):**
8. Data not shuffled - P1, ⭐
15. Padding tokens included in loss - P1, ⭐⭐

### **NEW Data Pipeline Bugs (8 bugs):**

---

### **Bug #41: Train/validation split has data leakage**
- **Category**: Data Pipeline
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/dataset.py` or preprocessing script
- **Location**: Data splitting
- **Change**: Normalize/process data before splitting
- **Expected Symptom**:
  - Validation loss artificially low
  - Model performance overstated
  - Poor generalization to test set
- **Debugging Hint**: Check order of preprocessing vs splitting
```python
# WRONG:
all_data = load_data()
normalized = normalize(all_data)  # Uses stats from val set!
train, val = split(normalized)

# CORRECT:
all_data = load_data()
train, val = split(all_data)  # Split first
train = normalize(train)
val = normalize(val, stats=train.stats)  # Use train stats only
```

---

### **Bug #42: Batch collation doesn't pad sequences**
- **Category**: Data Pipeline
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/dataset.py`
- **Location**: DataLoader collate function
- **Change**: Don't pad sequences to same length
- **Expected Symptom**:
  - Shape mismatch error when batching
  - Can't stack tensors
- **Debugging Hint**: Print shapes of batch items
```python
# Need collate function:
def collate_fn(batch):
    # Pad all sequences to max length in batch
    max_len = max(len(item) for item in batch)
    padded = [pad_to_length(item, max_len) for item in batch]
    return torch.stack(padded)

train_loader = DataLoader(dataset, collate_fn=collate_fn)
```

---

### **Bug #43: Tokenizer encoding/decoding mismatch**
- **Category**: Data Pipeline
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/dataset.py` or tokenizer
- **Location**: Tokenization
- **Change**: Different special tokens in encode vs decode
- **Expected Symptom**:
  - Generated text has weird tokens
  - Can't reconstruct original text
  - Evaluation metrics wrong
- **Debugging Hint**: Test `decode(encode(text)) == text`
```python
# Test round-trip:
text = "Hello world"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert decoded == text, f"Mismatch: {decoded} != {text}"
```

---

### **Bug #44: Data loader workers cause memory leak**
- **Category**: Data Pipeline
- **Priority**: 3
- **Difficulty**: ⭐⭐ Medium
- **File**: `scripts/base_train.py`
- **Location**: DataLoader initialization
- **Change**: Too many workers or workers not cleaned up
- **Expected Symptom**:
  - Memory usage grows over time
  - System RAM fills up
  - Slower over time
- **Debugging Hint**: Monitor memory with different num_workers
```python
# For small datasets, use fewer workers:
train_loader = DataLoader(dataset, num_workers=0)  # Or 2-4

# Not:
train_loader = DataLoader(dataset, num_workers=16)  # Overkill!
```

---

### **Bug #45: Duplicate examples in train and validation sets**
- **Category**: Data Pipeline
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: Data splitting logic
- **Location**: Train/val split
- **Change**: Random split without checking uniqueness
- **Expected Symptom**:
  - Validation loss too low
  - Model "cheating" by memorizing
- **Debugging Hint**: Check for overlaps between sets
```python
# Check for duplicates:
train_ids = set(train_dataset.get_ids())
val_ids = set(val_dataset.get_ids())
overlap = train_ids.intersection(val_ids)
assert len(overlap) == 0, f"Found {len(overlap)} duplicate examples!"
```

---

### **Bug #46: Data augmentation applied to validation set**
- **Category**: Data Pipeline
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: `nanochat/dataset.py`
- **Location**: Transform definition
- **Change**: Use same augmentation for train and val
- **Expected Symptom**:
  - Validation metrics noisy
  - Not reproducible
  - Wrong performance estimates
- **Debugging Hint**: Check if val uses deterministic transforms
```python
# WRONG:
train_transform = RandomCrop()
val_transform = RandomCrop()  # Should be deterministic!

# CORRECT:
train_transform = RandomCrop()
val_transform = CenterCrop()  # Deterministic
```

---

### **Bug #47: Class imbalance not handled**
- **Category**: Data Pipeline
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: Training script or loss computation
- **Location**: Loss function
- **Change**: Don't weight classes
- **Expected Symptom**:
  - Model predicts majority class for everything
  - High accuracy but useless (predicts all one class)
- **Debugging Hint**: Check class distribution and loss weights
```python
# For imbalanced data (e.g., 90% class 0, 10% class 1):
class_weights = torch.tensor([1.0, 9.0])  # Inverse frequency
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

### **Bug #48: Wrong input/target alignment in language modeling**
- **Category**: Data Pipeline
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/dataset.py`
- **Location**: Sequence creation
- **Change**: Wrong offset for targets
- **Expected Symptom**:
  - Model can't learn (predicting wrong token)
  - Loss doesn't decrease
- **Debugging Hint**: Print input and target sequences side by side
```python
# WRONG:
inputs = tokens[:-1]
targets = tokens[:-1]  # Same as inputs!

# CORRECT (next-token prediction):
inputs = tokens[:-1]   # [0, 1, 2, 3]
targets = tokens[1:]   # [1, 2, 3, 4]
```

---

## **INFERENCE & EVALUATION BUGS (9 bugs - 18%)**

### **Already Covered (1 bug):**
9. KV cache position not incremented - P2, ⭐⭐⭐

### **NEW Inference/Evaluation Bugs (8 bugs):**

---

### **Bug #49: Temperature = 0 causes division by zero**
- **Category**: Inference
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/engine.py`
- **Location**: Sampling function
- **Change**: Don't handle temperature=0 case
- **Expected Symptom**:
  - NaN or Inf when temperature=0
  - Crash during sampling
- **Debugging Hint**: Check special case handling for temperature
```python
# WRONG:
def sample(logits, temperature):
    probs = F.softmax(logits / temperature, dim=-1)  # Div by zero!
    return torch.multinomial(probs, 1)

# CORRECT:
def sample(logits, temperature):
    if temperature == 0:
        return logits.argmax(dim=-1)  # Greedy
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)
```

---

### **Bug #50: Top-k sampling with k > vocab_size**
- **Category**: Inference
- **Priority**: 3
- **Difficulty**: ⭐ Easy
- **File**: `nanochat/engine.py`
- **Location**: Top-k sampling
- **Change**: Don't clamp k to vocab_size
- **Expected Symptom**:
  - Error when k > vocab_size
  - Crash during generation
- **Debugging Hint**: Check k vs vocab_size
```python
# WRONG:
top_k_logits = logits.topk(k)  # Error if k > vocab_size!

# CORRECT:
k = min(k, logits.size(-1))
top_k_logits = logits.topk(k)
```

---

### **Bug #51: Evaluation metrics computed on wrong dimension**
- **Category**: Evaluation
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: Evaluation scripts
- **Location**: Accuracy computation
- **Change**: `argmax(dim=0)` instead of `dim=-1`
- **Expected Symptom**:
  - Wrong accuracy numbers
  - Shape errors
  - Metrics don't make sense
- **Debugging Hint**: Print prediction shapes
```python
# WRONG:
predictions = logits.argmax(dim=0)  # Wrong dimension!

# CORRECT:
predictions = logits.argmax(dim=-1)  # Last dimension
```

---

### **Bug #52: BatchNorm statistics not updated during eval**
- **Category**: Inference
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: Evaluation script
- **Location**: Model mode setting
- **Change**: Never called `model.train()` during training
- **Expected Symptom**:
  - BatchNorm uses wrong statistics
  - Eval performance worse than expected
  - Inconsistent results
- **Debugging Hint**: Check if running stats were updated
```python
# During training, running stats should be updated:
model.train()  # This allows BatchNorm to update running stats
for batch in train_loader:
    ...

# During eval, use the updated stats:
model.eval()  # Use running stats, don't update them
```

---

### **Bug #53: Loss function doesn't match task**
- **Category**: Loss Functions
- **Priority**: 1
- **Difficulty**: ⭐ Easy
- **File**: Training script
- **Location**: Loss definition
- **Change**: MSELoss for classification
- **Expected Symptom**:
  - Model doesn't learn
  - Wrong optimization objective
- **Debugging Hint**: Check loss function vs task type
```python
# WRONG for classification:
criterion = nn.MSELoss()

# CORRECT:
criterion = nn.CrossEntropyLoss()
```

---

### **Bug #54: Softmax applied before CrossEntropyLoss**
- **Category**: Loss Functions
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: Training or model forward
- **Location**: Before loss computation
- **Change**: Apply softmax to logits before CrossEntropyLoss
- **Expected Symptom**:
  - Double softmax
  - Loss too small
  - Model doesn't learn well
- **Debugging Hint**: Check if logits are already softmaxed
```python
# WRONG:
logits = model(x)
probs = F.softmax(logits, dim=-1)  # Don't do this!
loss = F.cross_entropy(probs, targets)  # CrossEntropy applies softmax internally

# CORRECT:
logits = model(x)
loss = F.cross_entropy(logits, targets)  # Pass raw logits
```

---

## **ADDITIONAL BUGS TO REACH 54**

### **Bug #17: Wrong dimension for argmax** (moved from Tier 3)
- **Category**: Evaluation/Metrics
- **Priority**: 1
- **Difficulty**: ⭐⭐ Medium
- **File**: Evaluation code
- **Location**: Prediction/accuracy computation
- **Change**: `logits.argmax(dim=-1)` → `logits.argmax(dim=0)`
- **Expected Symptom**:
  - Wrong predictions
  - Accuracy metrics nonsensical
  - Shape errors if batch size ≠ vocab size
- **Debugging Hint**: Print prediction shapes
```python
# Prediction:
predictions = logits.argmax(dim=0)  # ← SHOULD BE: dim=-1
```

---

### **Bug #18: Softmax before CrossEntropyLoss** (duplicate of #54, removing)

### **Bug #19: Residual connection in wrong place** (duplicate of #34, removing)

### **Bug #20: Division by zero in normalization**
- **Category**: Numerical Stability
- **Priority**: 2
- **Difficulty**: ⭐⭐ Medium
- **File**: `nanochat/model.py`
- **Location**: RMSNorm implementation
- **Change**: Remove `eps` (epsilon) from denominator
- **Expected Symptom**:
  - NaN when input has zero variance
  - Rare but catastrophic
  - Usually happens with specific inputs
- **Debugging Hint**: Check for NaN after normalization
```python
# RMSNorm:
def forward(self, x):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    return x / rms  # ← SHOULD BE: x / (rms + eps)
```

---

## **SUMMARY TABLE**

| Category | Total Bugs | P1 | P2 | P3 | % of Total |
|----------|-----------|----|----|----|----|
| **Training/Optimization** | 20 | 10 | 7 | 3 | 37% |
| **Model Architecture** | 15 | 6 | 7 | 2 | 28% |
| **Data Pipeline** | 10 | 7 | 2 | 1 | 19% |
| **Inference/Evaluation** | 9 | 4 | 3 | 2 | 17% |
| **TOTAL** | **54** | **27** | **19** | **8** | **100%** |

---

## **COMPLETE BUG INDEX BY PRIORITY**

### **Priority 1 Bugs (Must Know) - 27 bugs:**

**Training/Optimization (10):**
- Bug #1: Forgot `optimizer.zero_grad()`
- Bug #2: Wrong gradient accumulation scaling
- Bug #3: Model in eval mode during training
- Bug #14: Validation gradients not disabled
- Bug #21: Gradient clipping after optimizer step
- Bug #27: Scheduler stepped before optimizer
- Bug #28: Training metrics not reset
- Bug #30: Early stopping checks training loss

**Model Architecture (6):**
- Bug #4: Causal mask wrong diagonal
- Bug #5: Attention scores not scaled
- Bug #6: Using -1e9 instead of -inf
- Bug #7: Position indices wrong device
- Bug #33: Attention mask shape doesn't broadcast
- Bug #37: Position embeddings exceed max length
- Bug #39: Output projection missing

**Data Pipeline (7):**
- Bug #8: Data not shuffled
- Bug #15: Padding tokens in loss
- Bug #41: Data leakage in train/val split
- Bug #42: Batch collation doesn't pad
- Bug #45: Duplicate examples in train/val
- Bug #46: Data augmentation on validation
- Bug #48: Wrong input/target alignment

**Inference/Evaluation (4):**
- Bug #17: Wrong dimension for argmax
- Bug #51: Evaluation metrics wrong dimension
- Bug #53: Loss function doesn't match task
- Bug #54: Softmax before CrossEntropyLoss

---

### **Priority 2 Bugs (Should Know) - 19 bugs:**

**Training/Optimization (7):**
- Bug #12: Loss not averaged in distributed
- Bug #13: Learning rate not decayed
- Bug #16: Memory leak from loss accumulation
- Bug #22: LR not scaled for distributed
- Bug #23: Optimizer state not loaded
- Bug #24: Mixed precision loss scaling
- Bug #25: Warmup applied to weight decay
- Bug #31: Model parameters frozen
- Bug #32: Training loop doesn't sync GPUs

**Model Architecture (7):**
- Bug #9: KV cache position not incremented
- Bug #10: RoPE not applied to Q and K
- Bug #11: QK normalization missing
- Bug #34: Residual added before projection
- Bug #35: Layer norm after residual
- Bug #36: Dropout not applied in MLP
- Bug #38: Attention heads not separated

**Data Pipeline (2):**
- Bug #43: Tokenizer encode/decode mismatch
- Bug #47: Class imbalance not handled

**Inference/Evaluation (3):**
- Bug #20: Division by zero in normalization
- Bug #49: Temperature = 0 division
- Bug #52: BatchNorm statistics not updated

---

### **Priority 3 Bugs (Good to Know) - 8 bugs:**

**Training/Optimization (3):**
- Bug #26: Gradient norm computed incorrectly
- Bug #29: Checkpoint saved at wrong frequency

**Model Architecture (2):**
- Bug #40: Bias in linear layers with RMSNorm

**Data Pipeline (1):**
- Bug #44: Data loader workers memory leak

**Inference/Evaluation (2):**
- Bug #50: Top-k sampling k > vocab_size

---


### **Practice Progress Tracking**

✅ = Completed | ⬜ = Not yet practiced

### **Tier 1: Essential Bugs (Must Practice)**

| ID | Bug Name | Category | Difficulty | Time | Files | Status |
|----|----------|----------|------------|------|-------|--------|
| 1 | Forgot optimizer.zero_grad() | Gradient | ⭐ Easy | 10min | base_train.py | ✅ |
| 2 | Wrong gradient accumulation | Gradient | ⭐⭐ Med | 15min | base_train.py | ✅ |
| 3 | Model in eval mode | Training | ⭐ Easy | 5min | base_train.py | ✅ |
| 4 | Causal mask wrong diagonal | Architecture | ⭐⭐ Med | 20min | gpt.py | ✅ |
| 5 | Attention not scaled | Architecture | ⭐⭐ Med | 15min | gpt.py | ⬜ |
| 6 | Using -1e9 instead of -inf | Numerical | ⭐⭐ Med | 20min | gpt.py | ⬜ |
| 7 | Device mismatch | Architecture | ⭐ Easy | 5min | gpt.py | ✅ |
| 8 | Data not shuffled | Data | ⭐ Easy | 10min | dataloader.py | ⬜ |

### **Tier 2: Important Bugs (Should Practice)**

| ID | Bug Name | Category | Difficulty | Time | Files | Status |
|----|----------|----------|------------|------|-------|--------|
| 9 | KV cache position not incremented | Inference | ⭐⭐⭐ Hard | 30min | engine.py | ⬜ |
| 10 | RoPE not applied | Architecture | ⭐⭐⭐ Hard | 30min | gpt.py | ✅ |
| 11 | QK normalization missing | Architecture | ⭐⭐ Med | 20min | gpt.py | ⬜ |
| 12 | Loss not averaged in DDP | Distributed | ⭐⭐⭐ Hard | 25min | base_train.py | ⬜ |
| 13 | LR not decayed | Training | ⭐⭐ Med | 15min | base_train.py | ✅ |
| 14 | Validation gradients enabled | Training | ⭐ Easy | 10min | base_train.py | ✅ |
| 15 | Padding in loss | Loss | ⭐⭐ Med | 20min | base_train.py | ⬜ |

### **Tier 3: Advanced Bugs (Optional)**

| ID | Bug Name | Category | Difficulty | Time | Files | Status |
|----|----------|----------|------------|------|-------|--------|
| 16 | Memory leak from loss | Memory | ⭐⭐⭐ Hard | 30min | base_train.py | ✅ (OOM in validation) |
| 17 | Wrong argmax dimension | Metrics | ⭐⭐ Med | 15min | eval code | ⬜ |
| 18 | Softmax before CrossEntropy | Loss | ⭐⭐ Med | 15min | gpt.py | ✅ |
| 19 | Residual connection wrong | Architecture | ⭐⭐⭐ Hard | 35min | gpt.py | ⬜ |
| 20 | Division by zero in norm | Numerical | ⭐⭐ Med | 20min | gpt.py | ⬜ |

---

### **Summary: 9/20 Completed (45%)**

**Completed bugs:**
- ✅ Bug #1: Forgot optimizer.zero_grad()
- ✅ Bug #2: Wrong gradient accumulation
- ✅ Bug #3: Model in eval mode
- ✅ Bug #4: Causal mask wrong diagonal
- ✅ Bug #7: Device mismatch
- ✅ Bug #10: RoPE not applied
- ✅ Bug #13: LR not decayed
- ✅ Bug #14: Validation gradients enabled
- ✅ Bug #16: Memory leak (OOM in validation)
- ✅ Bug #18: Softmax before CrossEntropy

**Remaining bugs:** 11 (5 Easy/Med in Tier 1, 3 in Tier 2, 3 in Tier 3)

---

## How to Use This Guide

### **Method 1: I Inject Bugs for You (Recommended)**

In a new Claude Code session:

```
You: "Inject bug #1 blindly (don't tell me where)"
Claude: [modifies files]
You: [debug it]
You: "Done, next bug"
Claude: [injects bug #2]
...
```

### **Method 2: Self-Directed (For Review)**

Use the bug patches below to inject yourself:

```bash
# Apply bug patch
git apply neplocal/bug_practice/patches/bug_01.patch

# Debug it
python scripts/base_train.py --num_iterations=50

# Restore
git checkout .
```

---

## Bug Details & Patches

### **Bug #1: Forgot optimizer.zero_grad()**

**Symptoms:**
- Loss decreases too fast initially
- Then becomes unstable/oscillates
- Gradients accumulate across batches

**Patch:**
```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -285,7 +285,7 @@
     for opt in optimizers:
         opt.step()
-    model.zero_grad(set_to_none=True)
+    # model.zero_grad(set_to_none=True)
```

**Debug hints:**
- Monitor gradient norms
- Check if they keep growing
- Look after optimizer.step()

---

### **Bug #2: Wrong gradient accumulation**

**Symptoms:**
- Loss explodes or becomes NaN quickly
- Effective LR is grad_accum_steps × too high
- Model diverges

**Patch:**
```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -270,7 +270,7 @@
         train_loss = loss.detach()
-        loss = loss / grad_accum_steps
+        # loss = loss / grad_accum_steps
         loss.backward()
```

**Debug hints:**
- Check grad_accum_steps value
- Monitor gradient magnitudes
- Compare to baseline convergence

---

### **Bug #3: Model in eval mode**

**Symptoms:**
- Loss barely decreases
- Model doesn't learn
- Dropout/normalization in wrong mode

**Patch:**
```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -175,7 +175,7 @@
     # Training loop
-    model.train()
+    model.eval()
     for step in range(num_iterations):
```

**Debug hints:**
- Check model.training flag
- Look before training loop starts

---

### **Bug #4: Causal mask wrong diagonal**

**Symptoms:**
- Model can see future positions
- Training loss decreases but validation is bad
- Model learns wrong patterns

**Patch:**
```diff
--- a/nanochat/gpt.py
+++ b/nanochat/gpt.py
@@ -100,7 +100,7 @@
         attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool)
-        attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq)), diagonal=1)
+        attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq)), diagonal=0)
```

**Debug hints:**
- Visualize attention patterns
- Check if upper triangle has non-zero values
- Compare train vs validation loss gap

---

### **Bug #5: Attention not scaled**
**Note:** This bug is tricky in nanochat since it uses `F.scaled_dot_product_attention()`. Skip this one or modify to implement manual attention.

---

### **Bug #6: Using -1e9 instead of -inf**

**Symptoms:**
- Training fine initially
- Loss becomes NaN after 100-1000 steps
- Happens when attention scores get large

**Patch:**
```diff
--- a/nanochat/gpt.py
+++ b/nanochat/gpt.py
@@ -99,7 +99,7 @@
-        attn_mask = torch.where(mask_bool, 0.0, float('-inf'))
+        attn_mask = torch.where(mask_bool, 0.0, -1e9)
```

**Debug hints:**
- Check when NaN first appears
- Monitor attention score magnitudes
- Look at masking logic

---

### **Bug #7: Device mismatch**

**Symptoms:**
- Immediate crash
- Error: "Expected tensor on cuda:0 but got cpu"

**Patch:**
```diff
--- a/nanochat/gpt.py
+++ b/nanochat/gpt.py
@@ -206,7 +206,7 @@
-        t = torch.arange(seq_len, device=device)
+        t = torch.arange(seq_len)
```

**Debug hints:**
- Read error message carefully
- Look for tensor creation without .to(device)

---

### **Bug #8: Data not shuffled**

**Symptoms:**
- Model sees data in same order each epoch
- Validation worse than expected
- If data sorted, learns spurious patterns

**Patch:**
```diff
--- a/nanochat/dataloader.py
+++ b/nanochat/dataloader.py
# This is tricky - nanochat uses streaming, not DataLoader
# Need to modify parquets_iter_batched to not shuffle
```

**Debug hints:**
- Check if batches repeat patterns
- Look at dataloader creation

---

### **Bug #13: LR not decayed**

**Symptoms:**
- Loss plateaus early
- Model doesn't reach optimal performance
- No crash, just suboptimal

**Patch:**
```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -280,7 +280,7 @@
         for group in opt.param_groups:
-            group["lr"] = group["initial_lr"] * lrm
+            # group["lr"] = group["initial_lr"] * lrm
```

**Debug hints:**
- Log and plot LR over time
- Check scheduler.step() calls

---

### **Bug #14: Validation gradients enabled**

**Symptoms:**
- Validation is slower
- Memory usage grows during validation
- May OOM on large validation sets

**Patch:**
```diff
--- a/scripts/base_train.py
+++ b/scripts/base_train.py
@@ -180,7 +180,7 @@
     if step % eval_every == 0:
         model.eval()
-        with torch.no_grad():
+        # with torch.no_grad():
             val_bpb = evaluate_bpb(model, val_loader, eval_steps)
```

**Debug hints:**
- Check memory usage during validation
- Monitor GPU memory growth

---

## Debugging Workflow

### **1. Run Training (Short)**
```bash
python scripts/base_train.py --num_iterations=50 --run=bug_test
```

### **2. Observe Symptoms**
- Does it crash? → Read error carefully
- Loss explodes? → Gradient issue
- Loss doesn't decrease? → Training mode, data, or architecture
- NaN after some steps? → Numerical stability
- Subtle/slow learning? → Harder bugs (mask, normalization, etc.)

### **3. Form Hypothesis**
Based on symptoms, guess the category:
- **Crash immediately** → Device, dtype, shape mismatch
- **Loss explodes** → Gradient accumulation, LR, normalization
- **Loss stuck** → Model in eval, data issues, architecture broken
- **Loss → NaN** → Numerical stability (inf, division by zero)
- **Trains but bad** → Subtle architecture bugs

### **4. Add Debugging**
```python
# Gradient monitoring
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.4f}")

# Check model mode
print(f"Model training: {model.training}")

# Check loss values
print(f"Loss: {loss.item():.6f}, NaN: {torch.isnan(loss)}, Inf: {torch.isinf(loss)}")
```

### **5. Search Code**
- Start with `scripts/base_train.py` (most bugs here)
- Then `nanochat/gpt.py` (architecture bugs)
- Check recent git changes: `git diff`

### **6. Fix & Verify**
```bash
# After fix
python scripts/base_train.py --num_iterations=50 --run=bug_fixed

# Compare to baseline
# Should behave normally now
```

---

## Recommended Practice Order

### **Week 1: Core Bugs (Priority 1)**
Day 1: Bugs #1, #3, #7 (30 min)
Day 2: Bugs #2, #8 (30 min)
Day 3: Bugs #13, #14 (30 min)

### **Week 2: Architecture Bugs**
Day 1: Bug #4 (30 min)
Day 2: Bug #6 (30 min)
Day 3: Bug #11 (30 min)

### **Week 3: Advanced**
Day 1: Bug #10 (45 min)
Day 2: Bug #9 (45 min)
Day 3: Bugs #15, #16 (1 hour)

---

## Tips for Success

✅ **Do:**
- Run SHORT training (50-100 steps) to see bugs quickly
- Document symptoms before looking at code
- Form hypothesis about category
- Add strategic debugging prints
- Understand WHY it breaks, not just HOW to fix

❌ **Don't:**
- Look at git diff immediately
- Skip straight to hints
- Fix without understanding root cause
- Run full 1000-step training (waste of time)

---

## Tracking Progress

```bash
# Create log
cat > neplocal/bug_practice/my_progress.md << EOF
# My Bug Debugging Log

## Bug #1: [Name]
- Date: $(date)
- Symptoms observed:
- Time to find:
- Key learning:

## Bug #2: ...
EOF
```

---

## Reference: Common Debugging Commands

```bash
# Check what changed
git diff

# Find string in code
grep -r "zero_grad" scripts/

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check if file was modified
git status

# See recent commits
git log --oneline -5
```

---

## Interview Simulation

When ready, practice under interview conditions:
1. **Time limit**: 20 minutes per bug
2. **No hints**: Don't look at solutions
3. **Talk out loud**: Explain your reasoning
4. **Move on if stuck**: Real interviews move on after 20 min

---

## Resources

- **nanochat docs**: [README.md](../../README.md)
- **Architecture guide**: [neplocal/ARCHITECTURE_GUIDE.md](../ARCHITECTURE_GUIDE.md)
- **PyTorch docs**: https://pytorch.org/docs/

---

**Total Practice Time**: ~8 hours for all 20 bugs
**Essential Bugs Only**: ~3 hours for Tier 1

**Good luck! This will make you 10x better at debugging ML code.** 🚀
