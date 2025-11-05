```markdown
# Realistic ML Debugging Interview - Complete Case Study

---

## **PART 1: THE INTERVIEW (One Complete Case)**

---

### **[00:00] INTERVIEWER OPENS**

**Interviewer:** "Hi! Thanks for joining. Today we'll work through a debugging scenario together. I'll present a problem, and I'd like you to walk me through how you'd approach finding and fixing it. Feel free to think out loud - I want to understand your debugging process. Ready?"

**Candidate:** "Yes, ready!"

---

### **[00:30] SCENARIO PRESENTATION**

**Interviewer:** "Great. Here's the situation:

You're training a 500M parameter transformer language model. Training has been running smoothly for about 2000 steps with loss steadily decreasing. Then suddenly at step 2147, the loss becomes NaN and stays NaN for all subsequent steps.

Here's what we know:

**Environment:**
- 4 GPUs, batch size 16 per GPU
- Learning rate: 3e-4 with cosine decay
- Mixed precision (FP16) enabled
- Model: 24 layers, 1024 hidden dim

**Loss values (last 10 steps before NaN):**
```
Step 2138: 2.234
Step 2139: 2.221
Step 2140: 2.209
Step 2141: 2.197
Step 2142: 2.184
Step 2143: 2.173
Step 2144: 2.389  ← sudden jump
Step 2145: 3.672  ← bigger jump
Step 2146: 8.231  ← explosion
Step 2147: NaN
```

Walk me through how you'd debug this."

---

### **[02:00] CANDIDATE FORMS HYPOTHESES**

**Candidate:** "Okay, let me think through this systematically. The pattern shows the loss was stable and decreasing, then suddenly jumped and exploded to NaN. This suggests something that was accumulating or growing crossed a threshold rather than a one-time error.

My initial hypotheses are:
1. Gradient explosion - something caused gradients to become very large
2. Attention score overflow - in mixed precision, large attention logits can overflow
3. Numerical instability in a specific operation
4. Data-related issue - a corrupted or unusual batch

Given that it happened at a specific step and wasn't gradual, I'd start by checking gradient norms right before the explosion. Can I see the gradient norms for steps 2143-2147?"

---

### **[03:00] INTERVIEWER PROVIDES REQUESTED INFO**

**Interviewer:** "Good starting point. Here are the gradient norms:

```
Step 2143: grad_norm = 2.3
Step 2144: grad_norm = 8.7  ← spike
Step 2145: grad_norm = 45.2
Step 2146: grad_norm = 327.8
Step 2147: grad_norm = NaN
```

The gradient norms are clearly exploding."

---

### **[03:30] CANDIDATE NARROWS DOWN**

**Candidate:** "Okay, so gradient explosion is confirmed. The spike at 2144 suggests something specific triggered it. In transformers with mixed precision, the most common cause of sudden gradient explosions is attention score overflow.

Can I see the layer-wise gradient breakdown for step 2144? I want to see which layer first shows the spike."

**Interviewer:** "Sure. Here's the gradient norm per layer at step 2144:

```
Embedding: 1.2
Layer 0-10: 0.8 - 1.5 (normal range)
Layer 11 attention: 47.3  ← spike here!
Layer 12-23: NaN (propagated from layer 11)
Output head: NaN
```

Layer 11 attention is where it starts."

---

### **[04:30] CANDIDATE REQUESTS CODE**

**Candidate:** "Perfect. Layer 11 attention is the culprit. Before we look at the code, can you tell me - are attention scores being scaled by sqrt(d_k)? And are we using any attention score normalization like QK normalization?"

**Interviewer:** "Good questions. Let me show you the attention implementation:

```python
# 15 lines of code
class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1)  # (B, n_heads, T, T)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
```

Take a look and tell me what you see."

---

### **[05:30] CANDIDATE IDENTIFIES BUG**

**Candidate:** "I found it. Line 23 - the attention scores are not being scaled by sqrt(d_head). 

The line should be:
```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
```

Without this scaling, as training progresses and Q and K weights grow, the dot products become very large. With d_head = 64 in this model, we should be dividing by sqrt(64) = 8.

In mixed precision FP16, large numbers overflow easily. My guess is that by step 2144, the attention scores in layer 11 reached values that caused exp() in the softmax to overflow to infinity, producing NaN."

---

### **[06:30] INTERVIEWER PROBES DEEPER**

**Interviewer:** "Excellent catch. Two follow-up questions:

1. Why did this happen at step 2144 specifically and not earlier?
2. Why layer 11 and not other layers?"

---

### **[07:00] CANDIDATE EXPLAINS**

**Candidate:** "Good questions.

For question 1 - why step 2144:
The weights start small (random initialization) so early attention scores are also small. As training progresses, the weights grow through gradient updates. The scores increase gradually until they cross the overflow threshold for FP16. Step 2144 is simply when layer 11's attention scores first exceeded that threshold - probably around 88.7 (the FP16 max before overflow in exp).

For question 2 - why layer 11:
Different layers train at different rates depending on:
- The gradient flow through the network
- The data patterns they're learning
- Random initialization differences

Layer 11 happened to have weights that grew slightly faster than other layers, so it hit the overflow threshold first. If we continued training without fixing this, other layers would eventually hit it too.

I'd also add that the sudden jump in loss at 2144 rather than gradual increase confirms this is a threshold-crossing issue, not a gradual accumulation problem."

---

### **[08:00] INTERVIEWER ASKS ABOUT FIX**

**Interviewer:** "Great analysis. How would you fix this, and how would you verify the fix works?"

**Candidate:** "The fix is straightforward - add the scaling:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
```

To verify:
1. **Before deploying:** Add logging to track max attention score values across all layers. They should stay well below 88 (FP16 safe range).

2. **Add an assertion** in the attention code as a safety check:
```python
assert scores.abs().max() < 80, f"Attention scores too large: {scores.abs().max()}"
```

3. **Restart training from a checkpoint** before step 2144 (like step 2000) with the fixed code. Monitor:
   - Max attention scores per layer (should be < 20 typically with scaling)
   - Gradient norms (should stay stable)
   - Loss curve (should continue smooth decrease)

4. **Run for at least 500 more steps** to ensure stability past where it previously failed.

I'd also consider adding gradient clipping as an additional safety measure, though with proper scaling it shouldn't be necessary."

---

### **[09:30] INTERVIEWER FINAL QUESTION**

**Interviewer:** "One last thing - you mentioned this is more likely in mixed precision. Would this bug also cause issues in full FP32 training?"

**Candidate:** "Yes, but it would take much longer to manifest. FP32 has a much larger range before overflow (max ~10^38 vs FP16's ~65,000). 

Without scaling, the attention scores would still grow unboundedly, but it might take 50,000 or 100,000 steps instead of 2,000 steps to overflow. So in FP32 training, you might not notice this bug in short runs, but it would still eventually cause problems in longer training runs.

This is actually a common pattern - bugs that are latent in FP32 become acute in FP16, which is why it's good practice to test with mixed precision even if your production training uses FP32."

**Interviewer:** "Excellent. That's all I have for this scenario. Do you have any questions for me?"

---

### **[10:30] END OF CASE**

**Total time: ~10 minutes**

---

---

## **PART 2: INTERVIEWER GUIDELINES**

*[This section is for the interviewer - what the candidate doesn't see]*

---

## **A. Scenario Design Principles**

### **1. Symptom Selection**

**Good symptoms (pick one):**
- Loss becomes NaN at specific step
- Training loss decreases, validation loss increases
- Model outputs are all the same token
- Memory usage grows over time (OOM)
- Gradients vanish (all near zero)
- Training is extremely slow (performance bug)

**Bad symptoms (avoid):**
- "Something is wrong" (too vague)
- Multiple unrelated symptoms (confusing)
- Symptoms that require domain expertise (dataset-specific)

**Rule:** Pick ONE clear symptom with objective measurement.

---

### **2. Code Snippet Length**

**Recommended lengths by format:**

| Interview Type | Lines of Code | Rationale |
|----------------|---------------|-----------|
| Quick code review | 15-25 lines | Can read in 2-3 minutes |
| Standard debugging | 30-50 lines | One function or class |
| Complex scenario | 50-80 lines | Multiple related functions |
| Architecture review | 80-120 lines | Full model class |

**Rule:** Candidate should be able to read the code in 3-5 minutes.

---

### **3. Bug Difficulty Calibration**

**Easy (5-7 minutes to solve):**
- Missing `optimizer.zero_grad()`
- `model.eval()` instead of `model.train()`
- Wrong dimension in `argmax()`

**Medium (8-12 minutes to solve):**
- Missing attention score scaling
- Causal mask wrong diagonal
- Data normalization before split

**Hard (15-20 minutes to solve):**
- Subtle gradient accumulation error
- KV cache position not incremented
- Interaction between multiple components

**Rule:** For 45-minute interviews, use 1 easy + 1 medium, or 1 hard scenario.

---

## **B. Information Release Strategy**

### **0. Interview Type Selection (Choose Before Starting)**

**Type 1: Code Review Format (60% of interviews)**
- **When to use:** Testing code reading ability, pattern recognition, familiarity with common bugs
- **Structure:** Provide complete code snippet upfront, minimal context
- **Candidate task:** Read code and identify the bug by inspection
- **Time:** 5-15 minutes per bug
- **Example opening:** "Here's a training loop that causes gradient explosion. Take a look and tell me what's wrong."

**Type 2: Production Debugging Format (40% of interviews)**
- **When to use:** Testing systematic debugging methodology, hypothesis-driven investigation
- **Structure:** Start with symptoms only, candidate requests information progressively
- **Candidate task:** Form hypotheses, request diagnostics, narrow down, then see code
- **Time:** 15-25 minutes per bug
- **Example opening:** "Training loss became NaN at step 347. Here are the logs. How would you debug this?"

**Key Difference:**
- **Type 1:** Code is provided **immediately** (tests inspection skills)
- **Type 2:** Code is provided **after narrowing down** (tests debugging process)

**IMPORTANT for Type 1 - Code Snippet Completeness:**

When showing code in Type 1 format, provide **complete functional context**, not isolated snippets:

✅ **Good - Complete context:**
```python
# Show the entire gradient accumulation loop
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        loss = model(x, y)
    train_loss = loss.detach()
    loss.backward()  # BUG: Missing division
    x, y = next(train_loader)

for opt in optimizers:
    opt.step()
model.zero_grad(set_to_none=True)
```

❌ **Bad - Isolated snippet:**
```python
loss.backward()  # Where's grad_accum_steps? Where's the loop?
```

**Required context for common scenarios:**
- **Training loop bugs:** Show full loop (data loading → forward → backward → optimizer step)
- **Attention bugs:** Show full attention forward method (Q/K/V projection → scores → softmax → output)
- **Loss bugs:** Show logits computation → loss calculation → backward
- **Gradient bugs:** Show full optimization step including accumulation and zero_grad

**Rule:** Candidate should be able to understand data flow without asking "where does X come from?"

---

### **1. Initial Information (Always Provide)**

**Must include:**
- Clear symptom with measurements
- Environment details (GPU, batch size, model size)
- Timeline (when did it happen?)
- Basic metrics (loss values, step numbers)

**Example:**
```
"Loss becomes NaN at step 347.
Model: 500M params, 24 layers
Environment: 4 GPUs, FP16, batch size 64
Loss: 2.5 → 2.3 → ... → 2.1 → NaN"
```

**Don't include initially (Type 2 only):**
- Gradient norms (make them request this)
- Layer-wise breakdown (they should narrow down first)
- Code (they should form hypotheses first)

**Note:** In Type 1 format, code IS included initially - see section B.0 above.

---

### **2. When Candidate Requests Information**

**Always provide when requested:**
- Gradient norms (standard debugging info)
- Layer-wise metrics (after they ask for gradients)
- Learning rate values
- Attention statistics (if relevant)
- Memory usage

**Provide code when:**
- They've formed clear hypotheses
- They've asked specific questions about implementation
- They've narrowed down to a component

**Example good flow:**
```
Candidate: "Can I see gradient norms?"
You: [Provide]

Candidate: "Gradients are exploding in layer 11. Can I see the attention code?"
You: [Provide code]
```

**Example bad flow:**
```
Candidate: "Can I see all the code?"
You: "Let's start with hypotheses. What do you think might cause this?"
```

---

### **3. Hints and Timing Flexibility**

**Default Timing for Hints (Type 2 Production Debugging format):**

| Time Elapsed | Candidate Status | Hint Level |
|--------------|-----------------|------------|
| 0-5 min | Exploring | None - let them work |
| 5-8 min | Stuck on wrong path | Gentle: "Have you considered X?" |
| 8-12 min | Making no progress | Direct: "Let's look at the attention layer" |
| 12-15 min | Still stuck | Strong: "Notice line 23 has no scaling" |

**IMPORTANT: Timing Flexibility Based on Difficulty and Progress**

The above timing is a **baseline for medium difficulty bugs**. Adjust based on:

**1. Bug Difficulty Tier (from 20MLBugs.md):**

| Difficulty | Expected Time | When to Give First Hint |
|-----------|---------------|------------------------|
| ⭐ Easy (5-10 min) | Model eval, Device mismatch, zero_grad | 3-5 min if stuck |
| ⭐⭐ Medium (15-20 min) | Grad accumulation, Causal mask, QK norm | 8-12 min if stuck |
| ⭐⭐⭐ Hard (25-35 min) | KV cache, RoPE, DDP loss averaging | 15-20 min if stuck |

**2. Candidate Progress:**

✅ **Extend "no hint" period indefinitely if:**
- Candidate is making steady progress
- They're forming reasonable hypotheses
- They're requesting appropriate information
- They're narrowing down systematically

**Even if it takes 15+ minutes, no hints needed if they're on the right track.**

❌ **Give earlier hints if:**
- Completely stuck with no ideas (stuck at hypothesis formation)
- Going in circles on wrong path for 5+ minutes
- Silent for 3+ minutes without explanation
- Showing signs of frustration or giving up

**3. Interview Format:**

| Format | Hint Strategy |
|--------|---------------|
| **Type 1 (Code Review)** | Minimal hints - code is already visible. If stuck >8 min on easy bug or >15 min on hard bug, make small suggestions|
| **Type 2 (Production Debugging)** | Hints can help with methodology: "What would you check next?" or "Have you narrowed down to a component?" |

---

**Good Hints (guide without solving):**
- "Have you considered checking gradient norms?"
- "What could cause sudden explosions rather than gradual growth?"
- "In mixed precision, what operations are sensitive to large values?"
- "You've been looking at the data pipeline for a while. What other components could cause this symptom?"

**Bad Hints (give away answer):**
- "The bug is in the attention scaling"
- "You're missing sqrt(d_k)"
- "Line 128 is wrong"

**Rule:** Hints should redirect thinking, not provide solutions.

---

**Special Case: Strong Candidates on Hard Bugs**

If a strong candidate is systematically working through a hard bug (⭐⭐⭐):
- Let them work for 20-25 minutes even without finding it yet
- Their process is more valuable than speed
- Only intervene if they're completely stuck, not just slow
- Hard bugs are designed to take 25-35 minutes - that's expected

**Example:**
```
Bug: RoPE not applied (Hard, 30 min expected)
Candidate at 18 min: "I've checked the attention mechanism, the masking looks correct, now examining the positional encoding logic..."
Interviewer: [Stay quiet, they're making progress]

Candidate at 22 min: "Ah, I see - the apply_rotary_emb call is missing"
Interviewer: "Excellent. Why would this cause issues with long-range dependencies?"
```

This candidate took 22 minutes but demonstrated excellent systematic debugging. No hints were needed.

---

### **4. What NOT to Tell Candidates**

**Never reveal:**
- The exact line number of the bug
- The exact fix
- Whether their hypothesis is right before they check

**Always make them:**
- Form hypotheses first
- Request specific information
- Explain their reasoning
- Propose the fix themselves

---

## **C. Question Asking Guidelines**

### **1. Mandatory Follow-up Questions (CRITICAL for Evaluation)**

**After bug is found, ALWAYS ask 2-3 follow-up questions. This is critical for:**
- Distinguishing **understanding** from **lucky guesses**
- Assessing **depth of knowledge** vs surface pattern matching
- Evaluating **communication skills** and ability to explain technical concepts
- Testing whether they truly understand the **mechanism** of the bug

**Standard follow-up questions (always ask 2-3 of these):**

- "Why did this happen at this specific step/time?"
- "How would you verify the fix works?"
- "What would happen if we didn't fix this?"
- "How would you prevent this in the future?"
- "Would this affect [related component]?"

---

**Follow-up Question Bank by Bug Category:**

Use these category-specific questions to probe deeper understanding:

**Gradient Bugs (zero_grad, accumulation, clipping):**
- "Why didn't the problem manifest immediately at step 1?"
- "What would happen in fp32 vs bfloat16?"
- "How does gradient accumulation interact with this bug?"
- "What would the loss curve look like over time?"
- "At what point do the weights become unrecoverable?"

**Architecture Bugs (attention, masking, normalization):**
- "How would this affect training vs inference differently?"
- "Would this bug impact all sequence lengths equally?"
- "What would the attention patterns look like visually?"
- "How would this interact with different batch sizes?"
- "Would you see this in validation metrics or only training?"

**Numerical Stability Bugs (overflow, underflow, precision):**
- "Why does this cause NaN and not just a large number?"
- "What's the specific threshold or condition that triggers this?"
- "How does the choice of dtype (fp16/bf16/fp32) affect this?"
- "Would gradient clipping help or just hide the symptom?"
- "What operations are most sensitive to this issue?"

**Data Pipeline Bugs (shuffling, leakage, preprocessing):**
- "How would you detect this bug if you didn't see the code?"
- "What would the learning curves tell you?"
- "Would this affect train and validation equally?"
- "What statistical test could catch this?"
- "How severe is this compared to other data issues?"

**Training Loop Bugs (train/eval mode, lr decay, validation):**
- "What observable metrics would this affect?"
- "How much would this slow down convergence?"
- "Would this cause immediate failure or gradual degradation?"
- "What PyTorch mechanisms are involved here?"
- "How do you verify the model is in the right mode?"

---

**Example Deep Probing Sequence:**

**Scenario: Candidate finds missing `loss / grad_accum_steps`**

❌ **Shallow interview (no follow-ups):**
```
Candidate: "You didn't scale down the loss by grad_accum_steps."
Interviewer: "Correct! Next bug..."
```
**Result:** Can't tell if they understood or just pattern-matched.

✅ **Deep interview (with follow-ups):**
```
Candidate: "You didn't scale down the loss by grad_accum_steps."
Interviewer: "Correct. Why did it explode at step 50 instead of step 1?"
Candidate: "Because weights start small... [explains compounding]"
Interviewer: "How would this differ in fp32 vs bfloat16?"
Candidate: "bfloat16 overflows sooner... [explains precision]"
Interviewer: "If grad_accum_steps=4, how many times too large are gradients?"
Candidate: "4 times too large because each of 4 micro-batches contributes full gradient..."
```
**Result:** Now you know they deeply understand gradient accumulation, numerical precision, and training dynamics.

---

**Red Flags in Follow-up Answers:**

These responses indicate shallow understanding:
- "I'm not sure, I just noticed the pattern from similar code"
- "That's what the documentation says"
- "I'd have to test it to know"
- Long silence followed by vague answer
- Circular reasoning ("It explodes because the gradients are too large")

**Strong signals in follow-up answers:**
- Explains mechanism step-by-step with specific values
- References relevant concepts unprompted (e.g., "bfloat16 has 8 exponent bits...")
- Draws connections to related issues
- Proposes multiple ways to verify
- Anticipates edge cases

---

**Timing for follow-ups:**
- Immediate (after they state the bug) - don't let them move on
- 2-4 minutes total for follow-up questioning
- Cut earlier if they clearly demonstrate deep understanding
- Probe longer if answers seem memorized or shallow

**Purpose:** This is often the highest-signal part of the interview. A candidate who finds bugs quickly but struggles with follow-ups likely has pattern recognition but weak fundamentals. A candidate who takes longer but answers follow-ups deeply has strong fundamentals.

---

### **2. Clarification Questions (Strict No-Hinting Policy)**

**When candidate asks clarifying questions, provide:**

| Question Type | Response |
|--------------|----------|
| "What's the model architecture?" | Provide high-level specs |
| "Are we using gradient clipping?" | Answer directly |
| "What's the batch size?" | Answer directly |
| "Can I see the full codebase?" | "Let's focus on [relevant part]" |
| "Is there a bug in PyTorch?" | "Assume PyTorch is correct" |

**Rule:** Answer factual questions directly. Guide away from distractions.

---

**CRITICAL: Avoid Accidental Hints in Factual Answers**

Even when answering factual clarification questions, avoid language that **directs toward the solution**:

✅ **Good - Pure factual answers:**

| Candidate Question | Good Answer (No hint) |
|-------------------|----------------------|
| "What do you mean by long-range dependencies?" | "Long-range dependencies means the model's ability to relate tokens that are far apart in the sequence, like understanding that a pronoun in position 50 refers to a name in position 5." |
| "What's bfloat16's range?" | "bfloat16 has a maximum value of approximately 3.4 × 10^38, same exponent range as fp32 but with less precision." |
| "What does grad_accum_steps do?" | "grad_accum_steps splits a large batch into smaller micro-batches. You accumulate gradients across micro-batches before doing one optimizer step." |
| "What is QK normalization?" | "QK normalization normalizes the query and key vectors before computing attention scores. It's a technique to improve training stability." |

❌ **Bad - Hinting answers (connect question to bug):**

| Candidate Question | Bad Answer (Contains hint) |
|-------------------|---------------------------|
| "What do you mean by long-range dependencies?" | "Long-range dependencies... **This suggests the model isn't effectively using positional information**, which could explain the poor generation quality." ← Hints at RoPE bug |
| "What's bfloat16's range?" | "bfloat16 can overflow around 10^38, **which is relevant when values grow unbounded during training**." ← Hints at scaling bug |
| "What does grad_accum_steps do?" | "It accumulates gradients across micro-batches. **Without proper scaling, this can cause gradients to be too large**." ← Hints at the bug |
| "What is QK normalization?" | "It normalizes Q and K vectors. **Missing this can cause attention score instability**, especially in deeper layers." ← Hints at the bug |

**The Difference:**
- **Good answer:** Explains the concept objectively, as if teaching in a classroom
- **Bad answer:** Connects the concept to the current debugging scenario or symptom

**Test yourself:** Would this answer be appropriate in a textbook definition? If yes → factual. If it references the current bug → hint.

**Examples in Context:**

**Scenario: User is debugging RoPE bug**

User asks: "What are positional embeddings?"

❌ **Bad (hints):** "Positional embeddings encode position information in the sequence. Without them, the model can't distinguish between positions, **which could explain why your model isn't learning position-dependent patterns**."

✅ **Good (factual):** "Positional embeddings encode position information in the sequence. Common approaches include sinusoidal embeddings (absolute position), learned embeddings, or rotary embeddings (relative position through rotation)."

---

**Scenario: User is debugging gradient accumulation bug**

User asks: "Why would explosion happen at step 50 instead of immediately?"

❌ **Bad (hints):** "Gradients accumulate their effects over time. **If gradients are too large, the weights grow until they cross the overflow threshold**. That's why you see delayed explosion."

✅ **Good (factual):** "When training is unstable, problems can manifest after some steps rather than immediately because: (1) Weights start small from initialization and grow over time, (2) Numerical issues compound across steps, (3) Mixed precision formats have specific overflow thresholds that get crossed at specific values. Many instabilities show this delayed pattern."

**Key principle:** Answer explains general ML concepts, not specific to this bug's mechanism.

---

**When in doubt:**
- Would you give this exact answer if they asked during a ML course lecture? → Factual
- Does your answer analyze or connect to their specific debugging scenario? → Hint

---

### **3. Probing Questions (Test Understanding)**

**Good probing questions:**
- "Walk me through what happens during this computation"
- "What's the shape of this tensor at this point?"
- "Why does this operation cause the symptom?"
- "What assumptions is this code making?"

**Purpose:** Verify they understand, not just guessed.

---

## **D. Evaluation Rubric**

### **What to Score (Internal Checklist)**

**Debugging Process (40%):**
- [ ] Forms multiple hypotheses before diving in
- [ ] Requests information in logical order
- [ ] Uses binary search / narrowing approach
- [ ] Doesn't jump to conclusions
- [ ] Adapts when hypotheses proven wrong

**Technical Knowledge (30%):**
- [ ] Understands gradients, loss, optimization
- [ ] Knows common failure modes
- [ ] Can read code fluently
- [ ] Explains technical concepts clearly

**Communication (20%):**
- [ ] Thinks out loud
- [ ] Explains reasoning clearly
- [ ] Asks good clarifying questions
- [ ] Organizes thoughts logically

**Speed & Efficiency (10%):**
- [ ] Finds bug within reasonable time
- [ ] Doesn't waste time on dead ends
- [ ] Recovers quickly from wrong paths

---

### **Red Flags (Automatic Concerns)**

**Major red flags:**
- Silent for >3 minutes without explanation
- Can't explain why their solution works
- Gives up without trying multiple approaches
- Blames external factors (PyTorch, hardware)
- Can't read Python/PyTorch code fluently

**Minor red flags:**
- Takes >15 min on easy bug
- Needs multiple hints
- Misses obvious issues initially
- Doesn't verify solution

---

## **E. Common Interviewer Mistakes**

### **1. Giving Too Much Info Upfront**

**❌ Bad:**
```
"Here's the code with gradient norms, attention stats, 
learning rate, and layer-wise breakdown. Find the bug."
```

**✅ Good:**
```
"Here's the symptom and basic environment. 
What would you check first?"
```

**Why:** You want to see HOW they debug, not just if they can spot bugs.

---

### **2. Jumping to Code Too Fast**

**❌ Bad:**
```
Candidate: "Loss is NaN"
You: "Here's the attention code"
```

**✅ Good:**
```
Candidate: "Loss is NaN"
You: "What are your initial hypotheses?"
Candidate: "Could be gradients or attention..."
You: "What would you check to narrow it down?"
Candidate: "Gradient norms per layer"
You: [Provides gradient info]
[After they narrow to attention]
You: [Shows code]
```

**Why:** Tests debugging methodology, not just code reading.

---

### **3. Not Probing Deep Enough**

**❌ Bad:**
```
Candidate: "Missing sqrt(d_k) scaling"
You: "Correct! Next scenario..."
```

**✅ Good:**
```
Candidate: "Missing sqrt(d_k) scaling"
You: "Why would that cause NaN at step 2144 specifically?"
Candidate: [Explains threshold crossing]
You: "How would you verify the fix?"
```

**Why:** Distinguish lucky guesses from real understanding.

---

### **4. Making Scenarios Unrealistic**

**❌ Bad scenarios:**
- Bug that would never pass code review (syntax error)
- Requires PhD-level math to understand
- Needs knowledge of obscure library internals
- Multiple unrelated bugs simultaneously

**✅ Good scenarios:**
- Realistic bugs from production systems
- Testable with standard debugging tools
- Clear symptoms
- Single root cause (though multiple symptoms OK)

---

## **F. Time Management**

### **Typical 45-Minute Interview Structure:**

```
00:00 - 02:00  Introduction, explain format
02:00 - 15:00  Scenario 1 (Easy-Medium bug)
15:00 - 18:00  Follow-up questions on Scenario 1
18:00 - 35:00  Scenario 2 (Medium-Hard bug)
35:00 - 40:00  Follow-up questions on Scenario 2
40:00 - 45:00  Candidate questions, wrap-up
```

**Time intervention points:**

| Time on Bug | Action |
|-------------|--------|
| 0-5 min | Let them work |
| 5-8 min | Check: "How's it going?" |
| 8-10 min | Small hint if stuck |
| 10-12 min | Redirect if on wrong path |
| 12-15 min | Strong hint or move on |
| 15+ min | Cut and move to next scenario |

**Rule:** Don't let candidates spend >15 min on one scenario without making progress.

---

## **G. Scenario Variations by Level**

### **Junior/New Grad (Focus: Basics)**

**Scenarios:**
- Missing `zero_grad()`
- Train/eval mode confusion
- Wrong loss function
- Simple shape mismatches

**Code length:** 15-30 lines

**Expected solve time:** 5-10 minutes

---

### **Mid-Level (Focus: Methodology)**

**Scenarios:**
- Attention bugs (scaling, masking)
- Data pipeline issues (leakage, shuffling)
- Gradient accumulation errors
- Numerical stability issues

**Code length:** 30-60 lines

**Expected solve time:** 10-15 minutes

---

### **Senior (Focus: Systems Thinking)**

**Scenarios:**
- Distributed training bugs
- Interaction between components
- Performance/memory issues
- Complex edge cases

**Code length:** 50-100 lines

**Expected solve time:** 15-25 minutes

---

## **H. Example Decision Tree for Interviewer**

```
Candidate says: "Loss is NaN"
    ↓
You ask: "What are your hypotheses?"
    ↓
    ├─ If they form 2-3 hypotheses → ✅ Good, provide gradient info
    │
    ├─ If they immediately say "show me code" → ❌ Redirect: "Let's think about what could cause this first"
    │
    └─ If they're silent → Give hint: "What are common causes of NaN in training?"
    
After they request gradient norms:
    ↓
You provide: Gradients exploding
    ↓
    ├─ If they say "it's layer 11" → Ask: "How do you know?"
    │
    ├─ If they request layer-wise breakdown → ✅ Good, provide it
    │
    └─ If they request code immediately → Guide: "Where would you look first?"
    
After they identify layer 11 attention:
    ↓
You provide: Attention code (30 lines)
    ↓
    ├─ If they find bug in 2-3 min → ✅ Excellent
    │
    ├─ If scanning for 5+ min → Hint: "Focus on the attention score computation"
    │
    └─ If still stuck at 8 min → Strong hint: "What scaling should attention scores have?"
```

---

## **I. Post-Interview Evaluation Template**

```
Candidate: [Name]
Position: [Level]
Date: [Date]

SCENARIO 1: [Description]
Time to solve: [X minutes]
Hints needed: [None / Light / Moderate / Heavy]

Debugging Process:
[ ] Formed clear hypotheses
[ ] Systematic information gathering
[ ] Logical narrowing approach
[ ] Recovered from wrong paths

Technical Depth:
[ ] Found bug correctly
[ ] Explained root cause
[ ] Proposed correct fix
[ ] Understood implications

Communication:
[ ] Clear explanation
[ ] Good questions
[ ] Organized thinking

SCENARIO 2: [Description]
[Repeat above]

OVERALL RECOMMENDATION:
[ ] Strong Hire - Excellent debugging, would hire immediately
[ ] Hire - Solid performance, meets bar
[ ] Weak Hire - Barely meets bar, border case
[ ] No Hire - Significant gaps in debugging or communication

SPECIFIC FEEDBACK:
Strengths:
- [Examples]

Areas for Improvement:
- [Examples]

COMPARISON TO BAR:
[How does this compare to typical candidates at this level?]
```

---

## **J. Sample Scenarios Bank (Quick Reference)**

### **Scenario Template:**

```markdown
**Symptom:** [One clear issue with metrics]
**Environment:** [GPU, model, batch size, etc.]
**Timeline:** [When it happened]
**Code:** [15-50 lines with bug]
**Expected solve time:** [X-Y minutes]
**Key learning:** [What concept this tests]
```

### **10 Ready-to-Use Scenarios:**

1. **Missing zero_grad** (Easy, 5-7 min)
2. **Attention scaling missing** (Medium, 8-12 min) ← Used in example above
3. **Causal mask wrong diagonal** (Medium, 8-12 min)
4. **Data leakage in normalization** (Medium, 10-15 min)
5. **KV cache position bug** (Hard, 15-20 min)
6. **Gradient accumulation scaling** (Medium, 10-12 min)
7. **Softmax before CrossEntropy** (Easy, 5-8 min)
8. **Distributed training loss averaging** (Hard, 15-20 min)
9. **Memory leak from loss accumulation** (Medium, 12-15 min)
10. **BatchNorm in eval during training** (Easy, 5-7 min)

---

**END OF INTERVIEWER GUIDE**

---

This guide provides both:
1. A realistic example interaction (Part 1)
2. Comprehensive principles for conducting interviews (Part 2)

The interviewer should study Part 2 to understand HOW to run effective debugging interviews, not just WHAT to ask.
```