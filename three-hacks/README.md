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

I wrote the proofs expecting to confirm all three, then had each section audited
adversarially. Two of the three headline claims did not survive, and the corrected
versions are more interesting than what I started with. Full formal treatment is in
[`paper/`](paper/); the errata are in [`theory.md`](theory.md).

### 1. Early-exit drafting — **stands, with a cost ceiling I had missed**

Speculative sampling returns *exactly* the target distribution for **any** draft `q`
whatsoever — no support condition, no constraint at all. Correctness therefore never
justified a separate draft network; the whole question is acceptance against cost.
Under a cost model that interpolates between memory- and compute-bound decoding,
self-drafting strictly dominates any external draft of comparable relative cost, in
*every* regime, and the strictness comes from cache reuse letting verification skip
blocks `1..ℓ` entirely — i.e. skip *streaming their weights*, which is precisely what
the memory-bound regime charges for.

**What the audit caught:** every draft step must also stream the unembedding `W_U` to
sample from `p_ℓ`. For Qwen3-0.6B that is ~26% of all parameters, paid on every draft
step. This imposes a hard ceiling

```
S  ≤  (γ+1) / (γu + 1)   =  2.45  at γ=4, u=0.26
```

and inflates naive speedup estimates by up to 1.39× at `ρ=0.25` — largest exactly where
early exit looks most attractive. Since the pre-registered kill criterion is `max S ≤ 1`,
this can invert the verdict. The way out is a *shortlisted* draft head, which the
exactness theorem legitimises precisely because it needs no support condition: restrict
the draft to a pre-chosen `S ⊆ V`, cut head cost to `u|S|/V`, and lose at most
`p_L(V∖S)` acceptance.

Still not novel as a mechanism — LayerSkip, Draft&Verify, Medusa, EAGLE. The honest
relocation stands: **the hack isn't speculative decoding, it's training models whose
intermediate layers aren't decodable.**

### 2. Latent reasoning — **my headline was backwards**

The strong form is false under `TC⁰ ≠ P`: serial steps buy power fixed depth cannot
recover. I had that right. What I got **wrong** was the bandwidth argument, and the
error reverses the conclusion.

I claimed `n` latent steps reach `2^(n·d·b)` states against CoT's `|V|ⁿ`. That is false:
iterating a deterministic map cannot grow cardinality (`|G(A)| ≤ |A|`), so `n` steps of a
fixed-size latent state reach at most `2^(d·b)` states — no `n` in the exponent at all.
The right comparison is **accumulation**, not per-step width:

| medium | per-step width | accumulates? | state after `n` steps |
|---|---|---|---|
| chain of thought | `log₂V ≈ 17` bits | yes — tokens persist, attention re-reads them | `n · log₂V` |
| fixed-size latent loop | `d·b ≈ 4096` bits | **no** — each step overwrites | `d·b` |
| appended latent thoughts | `d·b` bits | yes | `n · d·b` |

So a token chain **overtakes** a fixed-size latent loop at `n* = d·b/log₂V ≈ 238` steps —
i.e. throughout the thousands-of-tokens regime that makes CoT interesting. My §3 and §4
were contradicting each other: this is just the fixed-state recall ceiling seen from the
other side.

**Corrected thesis, which I believe:** discretisation is a real per-step cost, and it is
worth paying only where the medium accumulates. *Latent reasoning should append, not
overwrite* — and an appending latent medium does dominate CoT at every horizon.

A separation is available unconditionally only under a **streaming read schedule**: with
full attention the model re-reads the prompt, and with unbounded per-step compute a
one-step machine solves the task outright. Worse, in the full-attention setting any
theorem of the form "this needs ≥2 CoT steps" would imply `TC⁰ ≠ NC¹` by Barrington — so
it is out of reach, not merely open. The experiment has to *enforce* the schedule.

### 3. Transformers are gated RNNs — **true, but the payoff claim is false**

The recurrent form is real and nearly vacuous: *every* causal model admits one, and the
trivial state (`S_t = x_{1:t}`) is ~10⁴× **smaller** than the KV cache, so a transformer
does not compress its history — it expands it. The content is incrementality: the cache
is appended to and never rewritten, costing `Θ(Ld² + LtH_kv d_h)` per step against
`Θ(Lt²d)` to recompute.

The HMM identity is real: `s_t = (Aᵀ s_{t−1}) ⊙ b(o_t)` **is** the forward algorithm, and
max-product **is** Viterbi. Verified per `(t,j)` against brute-force path enumeration,
with a negative control confirming non-negativity is load-bearing for Viterbi only.

**But the bridge I claimed does not exist, and cannot.** Gated linear attention updates
`S_t = Diag(a_t)S_{t−1} + φ(k_t)v_tᵀ` — a *diagonal* action. The forward algorithm needs
`Diag(b)Aᵀ` — a *dense* one. Two independent impossibility proofs, both confirmed
numerically:

- **Support monotonicity.** `Φ(S) − Φ(S') = Diag(a)(S−S')`, so the rows on which two
  trajectories differ can only shrink. Under `Ψ` one differing coordinate spreads to all
  of them in a single step.
- **Simultaneous diagonalisability.** Conjugacy by any injective linear map would force
  every `Ψ_o` diagonal in one basis, hence commuting. They don't
  (`‖Ψ₀Ψ₁−Ψ₁Ψ₀‖ ≈ 4.5e-2`).

A gate *is* a diagonal matrix; constraining it non-negative and stochastic gives the
identity, not a transition matrix. **The transition has to be added to the architecture,
not constrained out of a gate that is already diagonal** — and diagonality is exactly
what makes these models parallelise as an associative scan. So "constrain a gated linear
RNN and get HMM-like decoding" is false as stated. Dense-transition models (the delta
rule's `I − βkkᵀ`) are the ones where an HMM reading is even a candidate.

The capacity ceiling also needed repair: it is **false** without a finite-precision
hypothesis, since `s_t = s_{t−1}/2 + x_t/2` stores unbounded history in one real. With
`m`-bit statefulness it holds, `m ≥ n log₂|V|`, and the Fano version predicts *graceful*
degradation past threshold rather than the cliff I had pre-registered.

## Validation plan — small models first, kill criteria up front

Every experiment has a **pre-registered prediction** and a **kill criterion**. Nothing
scales up until the free version passes. Sizes are Qwen3-0.6B; three of the four need no
training, and one runs with no dependencies at all.

| Exp | Question | Cost | Kill criterion |
|-----|----------|------|----------------|
| [`exp1`](experiments/exp1_early_exit.py) | Is `α_ℓ` high enough at small `ℓ/L` to beat the head-cost ceiling? | forward passes only | best `S ≤ 1.0` even after early-exit tuning and shortlisting |
| [`exp2`](experiments/exp2_latent_vs_cot.py) | Does interface width bind at `log₂(n!)`? | ~1M params from scratch, CPU-feasible | accuracy stays high below threshold *with the read schedule enforced* |
| [`exp3`](experiments/exp3_rnn_is_hmm.py) | Is the HMM identity real, and is the bridge really impossible? | stdlib only | identity fails, or support/commutator tests come out the other way |
| [`exp3b`](experiments/exp3b_recall_capacity.py) | Where does bounded state break? | forward passes only | break-point does not move with `w` (falsifies the proxy, not the theorem) |

**exp1** measures `α_ℓ`, tail energy, and the acceptance bound per layer, then the speedup
surface under the *corrected* cost including the head fraction `u`. It reports the bound's
looseness split into its Cauchy–Schwarz and softmax stages, and the shortlisted-head
variant with the measured `p_L(S^c)`.

> Prediction: the bound is vacuous and the split says the `D_U` step is responsible.
> Untuned `α_ℓ` at `ρ=0.5` will not clear `S=1` once the head is charged; whether
> shortlisting plus early-exit tuning rescues it is the actual open question.

**exp2** is a step machine under a **streaming Markov schedule** — step `i` sees generator
`g_i` and the previous interface, nothing else — with the interface content *unsupervised*
and trained through a straight-through Gumbel-softmax. That makes `V` a pure channel width
rather than a corrupted label, and it closes the re-read escape that no amount of depth
tuning closes on its own.

> Prediction: accuracy collapses exactly where `log₂V < log₂(n!)`, and the collapse point
> moves with `V` alone at fixed task and model.

**exp3** verifies the HMM identity per `(t,j)` against brute-force path enumeration (the
only non-circular check available — an earlier version compared a loop against its own
rewrite, which cannot fail), with a negative control, and then verifies the impossibility
of the bridge directly.

**exp3b** uses window attention as a bounded-state proxy. Note the Fano bound predicts
*graceful* degradation, so the original "sharp cliff" prediction was wrong and has been
retracted.

## Status

`exp3` **runs and passes**, both parts, including the negative control and both
impossibility checks. `exp1`, `exp2` and `exp3b` are written but **unrun**: no torch, no
numpy and no GPU in the container this was written in. Every number attributed to them
above is a prediction.

The paper in [`paper/`](paper/) is written but **not compiled** — no LaTeX toolchain here
either. Cross-references and environment nesting are checked by script; typesetting is not.

## Provenance

The three sections were each audited adversarially after I wrote them. That process
killed my thesis-2 headline outright (the cardinality error), killed the thesis-3 payoff
(the bridge), found the omitted head cost in thesis 1, and found a circular test in
`exp3`. [`theory.md`](theory.md) records what was wrong and where each correction lives.
Claims that survived are marked as such; nothing here should be read as confirmed until
the experiments run.

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
