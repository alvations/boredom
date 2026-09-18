# Formal statements

Everything here is stated so it can be attacked. Where a result is standard I say so and give a
proof sketch only; where I'm claiming something is provable I give the proof. Where I can't prove
it, it's labelled a conjecture and stays that way.

## 1. Notation

A decoder-only model `M` with `L` blocks. Residual stream at position `t`, layer `ℓ`:

```
h^(0) = E(x)                         embedding
h^(ℓ) = h^(ℓ−1) + F_ℓ(h^(ℓ−1))       ℓ = 1..L        (pre-norm residual block)
p_ℓ   = softmax( W_U · N(h^(ℓ)) )                    (logit lens at layer ℓ)
```

`W_U ∈ R^{V×d}` is the unembedding, `N` the final norm, `V = |vocabulary|`. The model's actual
output distribution is `p_L`. Write `σ` for softmax, `TV(p,q) = ½‖p−q‖₁`, and
`‖W‖₂,∞ = max_i ‖W_{i,:}‖₂` (largest row norm).

The two cost regimes matter and get conflated constantly:

- **Compute-bound**: cost ∝ FLOPs ∝ (positions × layers). Large batch, prefill.
- **Memory-bound**: cost ∝ bytes of weights streamed from HBM, roughly *independent of how many
  positions you push through*. Batch-1 autoregressive decode lives here, and it is the regime the
  entire speculative-decoding argument depends on.

---

## 2. Early exit as the draft

### Lemma 2.1 (softmax is TV-Lipschitz in sup-norm)

For `u, v ∈ R^V` with `ε = ‖u−v‖_∞`:

```
TV( σ(u), σ(v) )  ≤  1 − e^{−2ε}   ≤  2ε
```

*Proof.* For each `i`,

```
σ(u)_i / σ(v)_i = e^{u_i − v_i} · ( Σ_j e^{v_j} ) / ( Σ_j e^{u_j} ).
```

Since `|u_j − v_j| ≤ ε` for all `j`, we have `e^{−ε} Σ_j e^{v_j} ≤ Σ_j e^{u_j} ≤ e^{ε} Σ_j e^{v_j}`,
and `e^{u_i−v_i} ∈ [e^{−ε}, e^{ε}]`. Hence `σ(u)_i / σ(v)_i ∈ [e^{−2ε}, e^{2ε}]`, so in particular
`σ(u)_i ≥ e^{−2ε} σ(v)_i`. Now use `TV(p,q) = Σ_i (q_i − p_i)_+`:

```
TV(σ(u),σ(v)) = Σ_i ( σ(v)_i − σ(u)_i )_+ ≤ Σ_i σ(v)_i (1 − e^{−2ε}) = 1 − e^{−2ε}.
```

The final inequality is `1 − e^{−x} ≤ x`. ∎

### Theorem 2.2 (exactness — the draft is unconstrained)

Let `p` be the target distribution and `q` *any* distribution on the same support. Draw `x ~ q`,
accept with probability `min(1, p(x)/q(x))`, and on rejection draw from the normalised residual
`(p − q)_+ / ‖(p − q)_+‖₁`. The result is distributed exactly as `p`. Acceptance probability is
`Σ_x min(p(x), q(x)) = 1 − TV(p,q)`.

*Proof.* Standard (modified rejection sampling; Leviathan et al. 2023). For any `x`:
`Pr[output x] = q(x)·min(1, p(x)/q(x)) + Pr[reject]·(p(x)−q(x))_+ / ‖(p−q)_+‖₁`. The first term is
`min(p(x),q(x))`. Summing, `Pr[reject] = 1 − Σ_x min(p,q) = ‖(p−q)_+‖₁`, so the second term is
exactly `(p(x)−q(x))_+`, and `min(p,q) + (p−q)_+ = p(x)`. ∎

> **This is the load-bearing result for complaint 1.** Correctness imposes *no* condition on `q`.
> `q` may be a smaller model, an early exit of the same model, an n-gram table, or noise. There is
> therefore no correctness argument for maintaining a separate draft network — the only question is
> the acceptance-to-cost ratio. Any claim that draft models are principled has to be a claim about
> that ratio, and §2.4 says the ratio favours the self-draft in the regime that matters.

### Theorem 2.3 (acceptance is lower-bounded by residual tail energy)

Let `T_ℓ = ‖ Σ_{k=ℓ+1}^{L} F_k(h^(k−1)) ‖₂ = ‖h^(L) − h^(ℓ)‖₂` be the tail energy, and let `N` be
`κ`-Lipschitz on the region containing `h^(ℓ), h^(L)`. Then the acceptance rate of an early-exit
draft at layer `ℓ` satisfies

```
α_ℓ  =  1 − TV(p_ℓ, p_L)  ≥  exp( −2 ‖W_U‖₂,∞ · κ · T_ℓ ).
```

*Proof.* The logit gap is `u − v = W_U( N(h^(ℓ)) − N(h^(L)) )`. Coordinate `i` of that vector is
`⟨ (W_U)_{i,:}, N(h^(ℓ)) − N(h^(L)) ⟩`, so by Cauchy–Schwarz
`ε = ‖u−v‖_∞ ≤ ‖W_U‖₂,∞ · ‖N(h^(ℓ)) − N(h^(L))‖₂ ≤ ‖W_U‖₂,∞ · κ · T_ℓ`. Apply Lemma 2.1 and
`1 − (1 − e^{−2ε}) = e^{−2ε}`. ∎

Remarks, because this bound is more useful as a measuring stick than as a guarantee:

- It is **monotone in exactly the right thing**: acceptance is controlled by how much work the model
  has left to do after layer `ℓ`. That is the formal content of "the model already knows the easy
  tokens by layer 12".
- It will be **numerically vacuous** on a real model (`ε` of order 10 gives `e^{−2ε} ≈ 0`), and
  `exp1` is built to measure *how* vacuous. The looseness is not an accident: `‖·‖₂,∞` assumes the
  worst row of `W_U` aligns with the tail direction, whereas `h^(L) − h^(ℓ)` empirically concentrates
  in a low-dimensional subspace. A rank-restricted refinement — replacing `‖W_U‖₂,∞` with
  `max_i ‖P (W_U)_{i,:}‖₂` for `P` the projector onto the top-`r` tail subspace — is the obvious
  next step and is measurable with the same forward passes.
- For RMSNorm with gain `g`, `κ ≤ √d · max|g| / ‖h‖₂` locally, so the bound tightens as the residual
  stream grows — consistent with the observed fact that late layers are more decodable.

### Proposition 2.4 (self-draft dominance in the memory-bound regime)

Measure cost in units of one full target forward pass. Under `γ` draft steps and i.i.d. acceptance
`α`, expected accepted tokens per round is `E_γ(α) = (1 − α^{γ+1}) / (1 − α)`.

- External draft of relative cost `c_d`: `C_ext = γ·c_d + 1`, **and** a second weight stream.
- Early exit at layer `ℓ`, `ρ = ℓ/L`: `C_self = γ·ρ + 1`, no second stream, and the draft's KV
  entries for layers `1..ℓ` are bit-identical to the ones verification needs, so they are computed
  once (Remark below).

Hence `S = E_γ(α) / C`, and the self-draft dominates whenever `ρ ≤ c_d` and `α_ℓ ≥ α_d`.

*Proof.* Immediate from the cost accounting; `E_γ` is the standard geometric-series expectation
(accept `j` tokens with probability `α^j(1−α)` for `j<γ`, all `γ+1` with probability `α^γ·1`). ∎

*KV-reuse remark.* This is the structural point and it has no analogue for an external draft. When
drafting position `t+i` the model computes `h^(1..ℓ)` and writes KV for layers `1..ℓ`. Verification
needs `h^(1..L)` at those same positions with the same prefix, and since layers `1..ℓ` depend only on
the prefix and the position's own input, those cached tensors are exactly correct — verification
resumes at layer `ℓ+1`. The draft is not speculative work thrown away on rejection; it is a *prefix
of the verification itself*. An external draft model's forward passes are pure loss on rejection.

*The honest counterweight.* `α_ℓ` for an untrained early exit is poor, because `W_U` was fit against
`N(h^(L))` alone and `N(h^(ℓ))` is off-distribution for it. This is a statement about the training
objective, not about the architecture, and it is what LayerSkip's early-exit loss plus layer dropout
fixes. So complaint 1 resolves to: **the hack is not speculative decoding; the hack is training a
model whose intermediate layers are not decodable and then needing a second network to read them.**

---

## 3. Latent reasoning

### Theorem 3.1 (serial compute is necessary — the strong complaint is false)

A log-precision transformer of fixed depth with `poly(n)` width computes only functions in uniform
`TC⁰`. The same model equipped with `poly(n)` chain-of-thought steps decides exactly `P`.
Consequently, if `TC⁰ ≠ P`, there are problems solvable with CoT and **not** solvable by any
fixed-depth forward pass of that model, however well trained.

*Proof.* Merrill & Sabharwal (2024); the `TC⁰` upper bound is Merrill & Sabharwal (2023) via
uniform threshold-circuit simulation of log-precision attention, and the lower bound is by
simulating a Turing machine step per emitted token. ∎

> So "a model that had really learnt to reason wouldn't need to spit tokens" is **false as stated**.
> Serial depth has to come from somewhere. What follows is the part of the complaint that survives.

### Theorem 3.2 (bandwidth separation — the medium is the hack)

Consider `n` steps of computation.

1. *CoT.* The full internal state at step `n` — every activation, every KV entry — is a
   deterministic function of the emitted token sequence `y_{1..n}` (and the fixed prompt). Hence the
   reachable set of states has cardinality at most `|V|^n`, i.e. at most `n·log₂|V|` bits.
2. *Latent recurrence.* `n` steps of `s_{i+1} = G(s_i)` with `s ∈ R^d` at `b` effective bits per
   coordinate reach up to `2^{n·d·b}` states, i.e. `n·d·b` bits.

Therefore any task requiring the model to distinguish `K` intermediate configurations at step `n`
needs `n ≥ log₂K / log₂|V|` CoT steps, but only `n ≥ log₂K / (d·b)` latent steps.

*Proof.* Part 1 is a counting argument: the map `y_{1..n} ↦ (activations)` is a function, and a
function's image is no larger than its domain. The KV cache carries no information beyond the
tokens that produced it — it is expensive to recompute, not informative. Part 2 is the cardinality
of the discretised state space. ∎

For Qwen-class models: `log₂|V| = log₂(151936) ≈ 17.2` bits per CoT step. Against `d = 1024`, even at
a deliberately pessimistic 4 effective bits per coordinate, a latent step writes ~4,000 bits.
**Two to three orders of magnitude.** The nominal bf16 figure (~16k bits) is not the honest one to
quote, since the low mantissa bits are below the noise floor of a trained network; the effective
figure is itself an empirical question, and `exp2` is designed to measure it rather than assume it.

### Theorem 3.3 (latent recurrence subsumes CoT step-for-step)

Any `n`-step CoT computation of a model `M` can be simulated by an `n`-step latent-recurrent model
of the same width and depth.

*Proof sketch.* Take the latent update `G` to be: run `M`'s stack on the current state, apply the
unembedding, take `argmax` (or a sampled one-hot), re-embed via `E`, and write the result into the
state. This is a composition of the maps `M` already applies, so it is within the function class of
a latent-recurrent model with the same blocks; it reproduces the CoT trajectory exactly. ∎

Combined with 3.2, the inclusion is **strict**: CoT is the special case of latent recurrence in
which the state is projected onto the `|V|`-point codebook `{E(v)}` at every step.

### Conjecture 3.4 (the bandwidth–drift trade-off) — *open, and the interesting part*

That projection is not only a loss. Rounding onto a finite codebook every step is an
**error-correcting** operation: it prevents small representational errors from compounding across
steps, at the cost of `d·b − log₂|V|` bits per step. Latent recurrence keeps the bits and loses the
correction, which is a candidate explanation for why looped/latent models are hard to train to
large depth while CoT scales to thousands of tokens nearly for free.

Conjecturally, with per-step noise `η` and `n` steps, useful-information-through-the-chain behaves
like `min(bandwidth per step, capacity of a noisy channel with drift η^n)`, so there is a crossover
depth `n*(η)` below which latent wins and above which discretisation wins. I have no proof, and I
don't know whether the crossover is at `n = 5` or `n = 5000`. **A hybrid that discretises every `k`
steps — a learnt codebook, not the vocabulary — is the obvious design this suggests, and `k` is the
knob nobody has swept.** If any single thing in this directory is worth doing, it's that sweep.

---

## 4. Transformers are gated RNNs

### Theorem 4.1 (exact RNN form — not an analogy)

A decoder-only transformer is an RNN. Define the state `S_t = (K_t, V_t)` where `K_t, V_t` are the
per-layer key/value caches over positions `1..t`. Then

```
S_t = S_{t−1} ⊕ (k_t, v_t)            (concatenation along the position axis)
o_t = Attn(q_t, S_t)
```

which is a recurrence `S_t = f(S_{t−1}, x_t)`, `o_t = g(S_t, x_t)`. The state dimension grows as
`O(t · d · L)`.

*Proof.* Causal masking means position `t`'s output depends on positions `≤ t` only through their
keys and values, which is precisely what `S_t` stores. ∎

The only thing separating a transformer from a "plain RNN" is therefore that its state **grows**
instead of being compressed into fixed size. That is the whole difference. Complaint 3 is correct
about the form; §4.4 says what the growth buys.

### Theorem 4.2 (linear attention = fixed-state gated RNN)

With a feature map `φ` replacing `exp(q·k)`, attention becomes

```
S_t = S_{t−1} + φ(k_t) v_tᵀ ∈ R^{d_k × d_v},     z_t = z_{t−1} + φ(k_t)
o_t = ( φ(q_t)ᵀ S_t ) / ( φ(q_t)ᵀ z_t )
```

a fixed-size RNN with `d_k·d_v` state. Adding a data-dependent gate gives
`S_t = Diag(a_t) S_{t−1} + φ(k_t) v_tᵀ`, which is gated linear attention; Mamba-2's SSD duality
shows the same object is a structured masked attention. Standard (Katharopoulos et al. 2020; Yang
et al. 2024; Dao & Gu 2024).

### Theorem 4.3 (a constrained gated linear RNN *is* the HMM forward algorithm)

Let the state be constrained non-negative, `s_t ∈ R^m_{≥0}`, with row-stochastic transition
`A ∈ R^{m×m}` and emission gate `b(o_t) ∈ R^m_{≥0}`. Then the recursion

```
s_t = ( A ᵀ s_{t−1} ) ⊙ b(o_t)
```

computes exactly the HMM forward variables `α_t(j) = P(o_{1..t}, z_t = j)`, and the max-product
variant `s_t = ( max_i A_{ij} s_{t−1}(i) ) ⊙ b(o_t)` is Viterbi.

*Proof.* The HMM forward recursion is `α_t(j) = [ Σ_i α_{t−1}(i) A_{ij} ] · B_j(o_t)`. The bracket
is coordinate `j` of `Aᵀ α_{t−1}`; the factor is coordinate `j` of `b(o_t)` with `b_j = B_j(o_t)`.
Identical term for term, by induction from `α_0 = π`. The max-product claim follows by replacing
the sum-product semiring `(+, ×)` with `(max, ×)`, under which the same induction gives
`s_t(j) = max_{z_{1..t−1}} P(o_{1..t}, z_{1..t−1}, z_t = j)`. ∎

So "gated RNN with HMM-like decoding steps" is not loose talk: **non-negativity plus stochasticity
is exactly the constraint that turns the gate into a transition matrix.** The normalisation in
softmax attention is a weak echo of this — the output is a convex combination of value vectors,
i.e. a posterior-mean readout over positions.

`experiments/exp3_rnn_is_hmm.py` verifies this numerically, both semirings, no dependencies.

### Theorem 4.4 (fixed state has a provable recall ceiling)

Any model that can, for arbitrary inputs, reproduce an arbitrary `n`-token substring from its
context must carry at least `n·log₂|V|` bits in its state at the moment it begins reproducing.
Hence a fixed-state RNN with `m` bits of state fails for `n > m / log₂|V|`, while a KV-cache
transformer (state growing with `t`) does not.

*Proof.* Information-theoretic / pigeonhole: the map from the `|V|^n` possible substrings to states
must be injective on the relevant inputs, so the state space needs `≥ |V|^n` elements. ∎

This is the known failure mode of pure linear-attention and SSM models on associative recall and
copying (Arora et al. 2023; Jelassi et al. 2024). It is also why the field settled on hybrids:
a few full-attention layers restore exact recall, and the rest of the stack can be recurrent.

**Consequence for complaint 3, stated plainly:** the complaint is right that these are gated RNNs
and right that RNN-side lessons transfer, but "a decent model with fixed state" has a hard ceiling
you cannot optimise your way past. The open engineering question isn't *whether* to compress the
state, it's **how few exact-recall layers you need**, and that is measurable — which is what the
window-attention probe in `exp3` is for.

---

## 5. What would make this one model

Lay the three results side by side and they describe one object rather than three patches:

```
state:  s_t = Diag(a_t) s_{t−1} + k_t v_tᵀ         learnt, bounded, gated        (Thm 4.2/4.3)
        + a few exact-recall layers                 because of                    (Thm 4.4)
time:   iterate the block r times without emitting  keeping d·b bits/step         (Thm 3.2/3.3)
        discretise every k steps onto a learnt codebook, k swept                  (Conj 3.4)
depth:  halt when the tail energy T_ℓ is small; the halted state is the draft     (Thm 2.3/2.4)
```

The pleasing part is that the depth axis serves two masters: the same early-exit signal that says
"this token is easy, emit it now" is the signal that produces a free speculative draft for the
tokens that aren't. Adaptive computation time and self-speculation are the same mechanism read in
two directions.

I have not proved anything about the combination, and I'd rather run `exp1`–`exp3` than add
another theorem.
