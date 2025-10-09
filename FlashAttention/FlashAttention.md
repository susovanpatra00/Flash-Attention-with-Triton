# FlashAttention — what it is, why we need it, and a step-by-step derivation

*(Markdown, detailed, with numeric examples you can copy/paste)*

---

## Quick summary

* **FlashAttention** is not a different mathematical attention — it computes the same scaled dot-product attention.
* The difference: **FlashAttention is an I/O-aware, blocked implementation** that fuses `QK^T → softmax → (softmax·V)` into a streaming kernel that **never materializes the full $n\times n$** attention matrix.
* This reduces memory from $O(n^2)$ (naïve) down to $O(n\cdot d)$ (plus small tile buffers) and greatly reduces GPU DRAM traffic, so it runs much faster for long sequences while producing numerically identical outputs.

---

## 1. Recap: attention formula

Scaled dot-product attention (single head) for queries $Q\in\mathbb{R}^{n\times d}$, keys $K\in\mathbb{R}^{n\times d}$, values $V\in\mathbb{R}^{n\times d_v}$:

$$
\mathrm{Attention}(Q,K,V) \;=\; \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)\,V
$$

If we form $S = \frac{QK^\top}{\sqrt{d}}$, then `softmax` is applied row-wise (each query attends to all keys), producing an $n\times n$ matrix $A$ where $A_{i,*} = \mathrm{softmax}(S_{i,*})$, and final output is $A V$.

---

## 2. Why FlashAttention (motivation)

* **Naïve implementation** computes $S\in\mathbb{R}^{n\times n}$, stores it, applies softmax, then does $S'V$. Storing $S$ costs $O(n^2)$ memory and reading/writing it repeatedly causes high DRAM traffic.
* **FlashAttention**: compute attention in blocks that fit in fast on-chip memory (shared SRAM), fuse the sequence:

  * compute a tile of logits $L = Q_{\text{tile}} K_{\text{tile}}^\top$,
  * apply numerically-stable softmax logic **streamingly** (update running max and running sum),
  * immediately accumulate weighted V contributions,
  * move to next key tile — **never write the full $n\times n$ matrix to DRAM**.

**Result:** same mathematical output, much lower memory footprint and much less DRAM IO → faster in practice especially when $n$ is large.

---

## 3. Stable Softmax (Why and How)

Naïve softmax:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Problem: if $x = [1000, 999, 995]$, exponentials overflow.

### Trick

Subtract maximum value $M = \max(x_j)$:

$$
\text{softmax}(x_i) = \frac{e^{x_i - M}}{\sum_j e^{x_j - M}}
$$

This makes numbers small and safe.

---

## 4. Example with [3, 2, 5, 1]

$$
x = [3, 2, 5, 1]
$$

* Naïve: $[e^3, e^2, e^5, e^1]$ → large.
* Stable: subtract max = 5 → $[e^{-2}, e^{-3}, e^{0}, e^{-4}]$.
* Denominator = $e^{-2} + e^{-3} + 1 + e^{-4}$.
* Numerically stable and correct.

---

## 5. On-the-Fly Correction (From Your Screenshots)

When processing in **blocks**, we don’t see the whole vector at once.
We update incrementally:

### Step 1:

For $[3]$:
$\max_1 = 3,\; l_1 = e^{3-3} = 1$.

### Step 2:

Now include $[3,2]$:
$\max_2 = 3,\; l_2 = l_1 + e^{2-3} = 1 + e^{-1}$.

### Step 3:

Now include $[3,2,5]$:
$\max_3 = 5$.
Naïve update: $l_3 = l_2 + e^{5-5}$.
But this is WRONG, because previous terms were computed w.r.t max=3, not 5.

### Correction:

$$
l_3 = l_2 \cdot e^{3-5} + e^{5-5}
$$

This rescales old terms to the new maximum.
General rule:

$$
l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{x_j - m_{\text{new}}}
$$

This is the **streaming softmax** trick that FlashAttention uses.

---

## 6. Normal vs Flash Attention

**Normal Attention**

1. Compute $QK^T$ ($n \times n$).
2. Apply stable softmax.
3. Multiply with $V$.

* Needs $O(n^2)$ memory.

**FlashAttention**

1. Split $Q,K$ into blocks (e.g., $128 \times 128$).
2. For each block pair:

   * Compute partial dot product.
   * Update running max and denominator (streaming softmax).
   * Multiply with $V$.
3. Never store full matrix.

* Needs only $O(n \cdot d)$ memory.

---

## 7. Block Matrix Multiplication Example

Suppose $A(4 \times 4)$ and $B(4 \times 4)$.
Naïve: compute all 16 outputs, accessing memory a lot.

Blocked (tile size $2 \times 2$):

* Split into submatrices:

$$
A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}, \quad
B = \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$

* Compute:

$$
C_{11} = A_{11}B_{11} + A_{12}B_{21}
$$

and so on.

* Each tile fits in SRAM → faster, less IO.

---

## 8. Step-by-Step FlashAttention Formula

Let $Q \in \mathbb{R}^{n \times d}, K \in \mathbb{R}^{n \times d}, V \in \mathbb{R}^{n \times d}$.

**Naïve attention:**

$$
O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

**FlashAttention derivation:**

1. Partition $Q, K, V$ into blocks.
2. For each query block $Q_b$:

   * Initialize running max $m = -\infty$, running denominator $l = 0$, running output $o = 0$.
3. For each key block $K_b, V_b$:

   * Compute scores: $S = Q_b K_b^T / \sqrt{d}$.
   * Update running max: $m' = \max(m, \max(S))$.
   * Update denominator (with correction):

     $$
     l' = l \cdot e^{m - m'} + \sum e^{S - m'}
     $$
   * Update output:

     $$
     o' = \frac{o \cdot e^{m - m'} + \sum e^{S - m'} V_b}{l'}
     $$
   * Set $m = m', l = l', o = o'$.
4. Final output for block = $o$.

This ensures:

* Softmax is stable.
* Only blockwise computation is stored.
* Exact same result as normal attention.

---

## 9. Vectorized / matrix block formulas

Let $L = Q_i K_j^\top \;(\,b_q \times b_k\,)$. Let $M^{(t-1)}$ be the vector of previous row-maxes (length $b_q$), $Z^{(t-1)}$ the previous denominators (length $b_q$), and $Y^{(t-1)}$ the previous numerators (shape $b_q\times d_v$).

Compute per block:

* `block_max` = rowwise max of $L$: $m_{\text{block}} = \max_{c} L[:,c]$ (vector length $b_q$).
* `m_new` = elementwise max: $m^{(t)} = \max(M^{(t-1)}, m_{\text{block}})$.
* `exp_factor` = $\exp(M^{(t-1)} - m^{(t)})$ (vector broadcastable).
* `z_old_scaled` = $exp\_factor \odot Z^{(t-1)}$.
* `y_old_scaled` = $exp\_factor[:, None] \odot Y^{(t-1)}$.
* `L_shifted` = $L - m^{(t)}[:, \text{None}]$ (broadcast rowwise).
* `E` = $\exp(L_{\text{shifted}})$ (shape $b_q\times b_k$).
* `z_block` = rowwise sum of `E` (length $b_q$).
* `y_block` = `E @ V_j` (result $b_q \times d_v$).

Then:
* Z^(t) = z_old_scaled + z_block
* Y^(t) = y_old_scaled + y_block


Finish with final outputs $Y^{(T)} \oslash Z^{(T)}$ (rowwise division).

This is exactly what FlashAttention kernels implement, but done carefully in fused GPU kernels to avoid extra reads/writes to DRAM.

---

## 10. Pseudocode (high level)

```python
# Q: (n x d), K: (n x d), V: (n x dv)
# choose block sizes Bq (queries) and Bk (keys)
for i in range(0, n, Bq):            # query blocks
    Q_i = Q[i:i+Bq, :]               # load into fast memory
    M = -inf * ones(Bq)              # running max (per row)
    Z = zeros(Bq)                    # running denom (per row)
    Y = zeros(Bq, dv)                # running numerators
    for j in range(0, n, Bk):        # iterate key blocks
        K_j = K[j:j+Bk, :]           # load keys
        V_j = V[j:j+Bk, :]           # load values
        L = (Q_i @ K_j.T) / sqrt(d)  # bq x bk logits
        m_block = rowwise_max(L)
        m_new = maximum(M, m_block)
        factor = exp(M - m_new)      # shape (bq,)
        Z_old_scaled = factor * Z
        Y_old_scaled = factor[:,None] * Y
        L_shifted = L - m_new[:,None]    # broadcast
        E = exp(L_shifted)               # bq x bk
        Z_block = rowwise_sum(E)         # (bq,)
        Y_block = E @ V_j                # (bq x dv)
        Z = Z_old_scaled + Z_block
        Y = Y_old_scaled + Y_block
        M = m_new
    output_block = Y / Z[:,None]   # final bq x dv outputs
    write output_block out
```

---

## 11. Complexity and practical notes

* **Arithmetic complexity**: FlashAttention does the same arithmetic (no asymptotic reduction): computing all pairwise logits is still $O(n^2 d)$ FLOPs in the dense attention case (unless you change the attention algorithm).
* **Memory**: Naïve stores $O(n^2)$ intermediate logits; FlashAttention stores only $O(n d)$ (Q,K,V and output) plus tile buffers. This is the big win for long sequences.
* **Practical speed**: because FlashAttention reduces DRAM traffic and improves locality, the kernel runs much faster, particularly when n grows and the naive approach causes memory thrashing or OOMs.
* **Correctness**: outputs are *numerically identical* (up to floating-point associativity/rounding) to the standard attention if implemented correctly.

---

## 12. Short example showing the streaming softmax intuition

Using our `[3,2,5,1]` example but imagining it arrives in two blocks: `[3,2]` then `[5,1]`.

Process block 1:

* $m^{(1)} = \max(-\infty, \max(3,2)) = 3$
* $Z^{(1)} = e^{3-3} + e^{2-3} = 1 + e^{-1} \approx 1 + 0.3678794412 = 1.3678794412$
* $Y^{(1)} = e^{0} V_3 + e^{-1} V_2$ (here V are placeholders)

Process block 2 (logits [5,1]):

* block max = 5, so new max = 5.
* scale old accumulators: factor = exp(3-5)=e^{-2}=0.135335...
* scaled Z_old = 0.135335... * 1.3678794412 ≈ 0.185...
* compute new block contributions: exp(5-5)=1, exp(1-5)=e^{-4}=0.0183156 → Z_block ≈ 1.0183156
* Z_new = scaled_Z_old + Z_block ≈ 0.185 + 1.0183 ≈ 1.203438... (same denominator as earlier shifted approach)
* similarly update Y with scaled old and new block numerator → final Y / Z gives identical result to full softmax.

This is the same algebra as the earlier single-shot stable softmax — just done incrementally for blocks.

---

## 13. TL;DR / takeaways

* FlashAttention = **fused, blocked, numerically stable implementation** of the standard softmax attention.
* It **does not change the math** — it changes the *order of operations* to exploit SRAM and avoid storing the full $n\times n$ attention matrix.
* Benefits: **lower memory**, **less DRAM IO**, **faster** for long contexts, and **numerically stable** because it uses the `max` / `log-sum-exp` trick in a streaming way.

---
