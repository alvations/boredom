# Three hacks, one complaint

**Boredomspiration**: three things that have been annoying me about how we build and run LLMs.

1. Speculative decoding with a separate draft model is a hack. Why can't the model just exit
   early in its own layers and produce the speculation itself?
2. Reasoning-by-token-spitting and test-time-compute-as-context is a hack. It just means the
   model never learnt to reason *latently*.
3. Decoder-only transformers are just gated RNNs with a silly-large state. Take what we learnt
   about optimizers, residuals and policy learning, point it at plain RNNs with big historical
   state, and you should get HMM-like decoding steps and a perfectly decent model.

The thing I noticed writing these down: **they are the same complaint three times.** Each one is
about where a model is allowed to spend *serial* computation, and how it carries state while
doing it.

| # | Complaint | Axis | The hack | What it should be |
|---|-----------|------|----------|-------------------|
| 1 | Draft models | **depth** | a second network guesses for you | the network's own early layers guess |
| 2 | Chain of thought | **time** | serial compute squeezed through the vocabulary | serial compute in the residual stream |
| 3 | KV cache | **state** | state that grows without bound, never compressed | a learnt, bounded, gated state |

So it is one project: *a model whose depth can halt early, whose time steps need not emit a token,
and whose state is learnt rather than accumulated.* Depth, time, state.

The point of this directory is to do the two things that make it more than a rant:
**(a)** state each claim precisely enough that it is either provable or refutable, and
**(b)** kill each one cheaply with a small model before spending anything on a big one.

- Formal statements and proofs: [`theory.md`](theory.md)
- Experiments: [`experiments/`](experiments/)

---

## Scoreboard (what the math actually says)

I wrote the proofs first, expecting to confirm all three. The math only partly cooperated, which
is the useful part.

### 1. Early-exit drafting — **the complaint is right, and for a sharper reason than I had**

Speculative sampling returns *exactly* the target model's distribution for **any** draft
distribution `q` whatsoever (Theorem 2.1). Correctness places no constraint on the draft at all.
So there is no correctness argument for a separate draft model — the entire design question is the
ratio of acceptance to cost, and nothing else.

That makes the external draft model look strictly silly in the single-stream memory-bound regime,
where decode time is dominated by *streaming weights from HBM*, not by FLOPs. A separate draft
model is a second set of weights to stream. Early exit reuses weights that are already in flight,
and the draft's KV entries for layers `1..ℓ` are the *same tensors* verification needs
(Proposition 2.4). The self-draft's marginal cost is `ℓ/L`; the external draft's is
`(its size)/(target size)` plus a second weight stream, plus VRAM, plus tokenizer alignment.

And there is a real bound to check. Because the residual stream is additive, the gap between the
layer-`ℓ` distribution and the final one is controlled by the *tail energy* of the remaining
blocks, giving (Theorem 2.3):

```
α_ℓ  ≥  exp( −2 · ‖W_U‖₂,∞ · κ · T_ℓ )        T_ℓ = ‖ Σ_{k>ℓ} F_k(h^(k−1)) ‖₂
```

Acceptance rate is lower-bounded by how little work the model has left to do. Every quantity on
the right is measurable on a 0.6B model in an afternoon.

**The catch, stated honestly:** this is not a new idea. LayerSkip, Draft&Verify, Kangaroo, Medusa
and EAGLE all live here. And the reason the naive version underperforms is well known — intermediate
layers were never *trained* to be decodable, so `p_ℓ` is off-distribution for the unembedding.
Which relocates the hack: **the hack isn't speculative decoding, it's that we don't train for
decodable intermediate layers.** That's an objective-function problem, not an architecture problem,
and it is cheap to test.

### 2. Latent reasoning — **strong form is provably false, refined form is provably true**

The strong claim ("CoT is only a hack, a model that had learnt to reason would not need it") is
**false under standard complexity assumptions**, and I think this is worth internalising rather
than arguing with. A fixed-depth, log-precision transformer computes only functions in uniform
`TC⁰`. With `poly(n)` chain-of-thought steps the same model decides everything in `P`
(Merrill & Sabharwal). So if `TC⁰ ≠ P`, *no* amount of latent cleverness at fixed depth replaces
serial steps. Serial compute is not a hack. It is load-bearing.

But the medium is negotiable, and that is where the complaint survives — with teeth. A CoT
trajectory's entire state is a deterministic function of the emitted tokens, so after `n` steps it
can occupy at most `|V|ⁿ` distinguishable configurations: **≤ log₂|V| ≈ 17 bits written per step**
for Qwen's vocabulary. A latent step writes a `d`-dimensional vector. Nominally `d·b` bits; even at
a pessimistic ~4 effective bits per coordinate on a `d=1024` model that is ~4,000 bits
(Theorem 3.2). Two to three orders of magnitude per step. And anything CoT can do in `n` steps,
latent recurrence can do in `n` steps (Theorem 3.3), so the inclusion is strict.

**Refined thesis, which I believe and can defend:** serial compute is necessary; *discretising it
through the vocabulary* is the hack.

**The open crux, which I can't yet prove either way:** discretisation may be doing real work as an
*error-correcting code*. Rounding to a token projects the state onto a finite codebook every step
and stops drift from compounding. Latent recurrence has no such projection, which is plausibly why
looped models are hard to train deep. The honest formulation is a trade-off between bandwidth and
drift, and I don't know the shape of that curve. That is the most interesting thing in this
directory (Conjecture 3.4).

### 3. Transformers are gated RNNs — **literally true, with a provable ceiling**

Not a metaphor and not even hard: define the state as the KV cache and a decoder-only transformer
*is* an RNN, exactly, with state growing as `O(t·d·L)` (Theorem 4.1). Linear attention is the same
recursion with the state pinned to a fixed `d_k×d_v` matrix, and the gated variants are
`S_t = Diag(a_t)·S_{t−1} + k_t v_tᵀ` (Theorem 4.2). Katharopoulos et al. said this in 2020 and
Mamba-2's SSD duality made it a formal correspondence.

The HMM intuition is also literally right, under constraints: **constrain a gated linear RNN's
state to be non-negative with row-stochastic transitions and its recursion becomes the HMM forward
algorithm, term for term** (Theorem 4.3). Decoding becomes the forward recursion; the max-product
variant is Viterbi. `experiments/exp3_rnn_is_hmm.py` checks this numerically to floating-point
tolerance, and it runs in plain Python with no dependencies.

The ceiling is the part the complaint has to survive, and it is a theorem, not a vibe: exact recall
of `n` tokens requires `Ω(n log|V|)` bits of state, so a **fixed**-state RNN provably cannot copy or
retrieve beyond its state capacity, while a growing KV cache can (Theorem 4.4). This is why pure
linear-attention models fail associative recall, and why every serious system has converged on
hybrids — a mostly-recurrent stack with a few full-attention layers. So the honest version of
thesis 3 is: *yes, they're gated RNNs; the unbounded state is not decoration, it buys exact recall,
and the design question is how few full-attention layers you can get away with.*

---

## Validation plan — small models first, kill criteria up front

Every experiment below has a **pre-registered prediction** and a **kill criterion**. Nothing gets
scaled up and nothing gets trained until the free version passes. Sizes are Qwen3-0.6B and
Qwen2.5-0.5B; two of the three need no training at all.

| Exp | Question | Cost | Kill criterion |
|-----|----------|------|----------------|
| [`exp1`](experiments/exp1_early_exit.py) | Is `α_ℓ` high enough at small `ℓ/L` to pay for itself? | forward passes only, no training | best `S(ℓ,γ) ≤ 1.0` even after the tuning of §1b |
| [`exp2`](experiments/exp2_latent_vs_cot.py) | Does the bits-per-step bound predict where CoT breaks? | ~10M params from scratch, CPU-feasible | CoT accuracy does **not** fall off at `log₂(states) > log₂\|V\|` |
| [`exp3`](experiments/exp3_rnn_is_hmm.py) + [`exp3b`](experiments/exp3b_recall_capacity.py) | Is the HMM equivalence real, and where does fixed state break? | pure Python + forward passes | equivalence fails, or recall break-point is independent of state size |

**exp1 — early-exit acceptance profile.** Pure inference. For every layer `ℓ` of Qwen3-0.6B, apply
the final norm and unembedding to `h^(ℓ)`, and measure acceptance `α_ℓ = 1 − TV(p_ℓ, p_L)`, tail
energy `T_ℓ`, and the Theorem 2.3 bound. Then compute the speedup surface
`S(ℓ,γ) = (1−α_ℓ^{γ+1}) / ((1−α_ℓ)(γ·ℓ/L + 1))`.

> Prediction: the raw bound will be **vacuous** (`ε_ℓ` is large, so `exp(−2ε)` underflows toward 0)
> while measured `α_ℓ` is respectable — I expect a large gap, because `W_U` only reads a
> low-rank-ish subspace of the residual stream and the `‖·‖₂,∞` bound ignores that. *Quantifying
> that gap is the actual result*, and a rank-restricted bound is the follow-up.
> Second prediction: untuned `α_ℓ` at `ℓ/L = 0.5` lands too low to beat `S=1`, and the LayerSkip-style
> early-exit LoRA (§1b) is what moves it. If it doesn't, thesis 1 dies cheaply.

**exp2 — the bandwidth prediction, made falsifiable.** The trick is picking a task with a state
whose entropy I control exactly: **composition of permutations in `S_n`**. Composing `t`
permutations has an intermediate state of exactly `log₂(n!)` bits, and nothing smaller suffices.
Train a ~10M-parameter model from scratch two ways — (a) CoT, emitting the running permutation as
tokens from a `V`-symbol vocabulary, (b) looped-latent, `r` passes over one block with no emission —
at matched FLOPs, sweeping `n` and `V` independently.

> Prediction from Theorem 3.2: CoT accuracy collapses precisely when `log₂(n!) > log₂|V|` per step,
> and *the collapse point moves when you change `V` alone, holding the task fixed*. Latent looping
> should be flat across that boundary until it hits its own `d`-dependent limit. If CoT sails past
> its bandwidth bound, the theorem is wrong or the task leaks state, and thesis 2 dies.
> This is the experiment I most want to run, because `V` is a knob on the *theory*, not on the task.

**exp3 — HMM equivalence and the capacity ceiling.** Two parts. The equivalence check builds a
random HMM and the corresponding gated non-negative linear RNN and asserts identical outputs — it
validates Theorem 4.3 directly, runs in-container, no dependencies. The capacity probe needs no
training either (`exp3b`): restrict Qwen3-0.6B's attention to a sliding window of `w` tokens, which
is exactly a fixed-size-state proxy, and find where associative recall breaks as a function of `w`.

> Prediction: the break-point tracks the information bound of Theorem 4.4 — recall survives while
> the needed span fits in the window and falls off a cliff after, rather than degrading gently.

## Status

Nothing has been run. The container this was written in has no torch, no numpy and no GPU, so
`exp1` and the `exp2`/`exp3` model halves are **written but unexecuted**; only the pure-Python HMM
equivalence check in `exp3` has been verified to run. Treat every number above as a prediction,
not a result.

## Reading that got here first

Worth being clear that almost none of the *mechanisms* are new — the contribution I'm after is the
unification and the falsifiable bounds, not priority.

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)
- Elhoushi et al., *LayerSkip* (2024); Zhang et al., *Draft & Verify* (2023); Cai et al., *Medusa* (2024); Li et al., *EAGLE* (2024)
- nostalgebraist, *interpreting GPT: the logit lens* (2020)
- Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought* (2024)
- Dehghani et al., *Universal Transformers* (2018); Geiping et al., *latent recurrent depth* (2025)
- Hao et al., *Coconut: Chain of Continuous Thought* (2024)
- Katharopoulos et al., *Transformers are RNNs* (2020); Dao & Gu, *Mamba-2 / SSD duality* (2024); Yang et al., *Gated Linear Attention* (2024)
- Arora et al., *Zoology* (2023); Jelassi et al., *Repeat After Me* (2024)
