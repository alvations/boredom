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

## Results

Everything below was run. The environment could not reach `huggingface.co` (the network
policy denies it, as it does `download.pytorch.org`), so the language-model measurements
are on a 1.71M-parameter model trained here that satisfies the paper's Definition 2.1
exactly. It is weak — validation perplexity 431 against a 2048 vocab — which inflates
acceptance rates and gives only six choices of exit depth. Full write-up with tables:
[`paper/main.pdf`](paper/main.pdf), §6.

| Claim | Result |
|---|---|
| Exactness for any draft (Thm 3.1) | **Confirmed.** Indistinguishable from direct sampling; α matches `1−TV(p_ℓ,p_L)` to 10⁻³ |
| Acceptance ↔ tail energy (Thm 3.7) | **Shape confirmed, bound useless.** 46× too small even fed the measured spread |
| Round cost (Prop 3.15) | **Was wrong.** Omitted the bonus position; gave S=1.735 where S=1 is forced |
| Cache reuse (old Rmk) | **Retracted.** Worse when memory-bound, better only when compute-bound |
| Head-cost ceiling (Cor 3.17) | **Confirmed, and decisive** |
| Yield vs geometric (Cor 3.12) | **My remark was backwards.** Decays *faster* than geometric |
| HMM identity (Thm 5.6) | **Confirmed** against a brute-force oracle, per `(t,j)`, with a negative control |
| No embedding (Thm 5.8) | **Confirmed**, both obstructions |
| Recall ceiling (Thm 5.12) | **Confirmed**, 19/20 cells, no violation |
| Interface bandwidth (Thm 4.10) | **Confirmed**, monotone in width, no violation |
| Early-exit training (Rmk 3.22) | **Rescues thesis 1**: S=1.044 at Qwen's head fraction, zero quality cost |
| Discretisation as error correction (OP 4.14) | **Mis-posed.** No lock-in, no interior `k`; binary switch on rate vs drift |
| Thm 5.8 practical bite | **Confirmed** on a cyclic HMM: 10× excess-NLL gap, depth and width don't close it |

### The headline number

Untuned, at Qwen3-0.6B's head fraction `u=0.26`, **no early-exit layer beats 1.0** — the
optimum is the degenerate ρ=1, and the unembedding cost alone decides it. Then 600 steps of
auxiliary early-exit loss, early-stopped on final-layer validation (which *improved* by
0.03): **S = 1.044** at `u=0.26`, a real exit at layer 2. Tail energy fell at every layer —
the mechanism Thm 3.7 names. Without early stopping, 1.072 at +0.08 loss.

So thesis 1 survives on this model as an engineering claim, at zero quality cost, by 4%.
That is the predicted sign and the predicted mechanism; it is not a margin to build on,
and the model's weakness inflates every acceptance rate in it.

### Thesis 3's theorem is not empty

Thm 5.8 says a diagonal gate can't *exactly* embed the HMM forward algorithm. That could
have been true and irrelevant. On a **cyclic** HMM — belief entropy 0.92 of max 1.39, so
the hidden state has to be tracked through a rotating transition — a single dense layer
captures 94% of the learnable headroom and a single diagonal layer captures 44%. Three
diagonal layers: 53%. Four times the width: 62%. Neither closes it.

The first HMM tried said the opposite and was wrong: sharply peaked rows collapse the
belief after a symbol, no mixing is ever needed, and a "disproof" printed. The script now
measures belief entropy and refuses to rule unless mixing is required.

Also worth recording: at layer 1, sampling-mode acceptance is 0.327 but greedy top-1
agreement is only **0.052**. Those govern different deployment modes, and a 6× gap means
an acceptance rate that reads as tolerable for sampled decoding is near-useless for greedy.

### Four things I got wrong, found by running it

1. **The cost model omitted a forward pass.** Caught by the ρ→1 sanity check: drafting with
   the full model *is* ordinary decoding, so S must be 1. It returned 1.735. Recommended as
   a standing check on any speculative-decoding cost model — it survived derivation *and*
   adversarial review.
2. **Cache reuse doesn't help when memory-bound.** I defended this against the audit and was
   wrong. Reuse splits verification across two depths, and a memory-bound pass costs a whole
   weight stream regardless of position count, so the lower blocks get paid for twice.
3. **The bound's looseness isn't mainly Cauchy–Schwarz.** I predicted it was. The softmax
   stage alone is 46× off; Cauchy–Schwarz merely compounds it. A useful bound needs a
   different proof strategy, not a tighter constant.
4. **Acceptance decays faster than geometric, not slower.** I'd argued from Jensen that
   estimating from a mean acceptance rate is conservative. Measured: 1.320 against 1.481
   predicted — it *overstates* yield by 12%. The Jensen argument is about variation between
   contexts; within a round each accepted draft moves the model onto its own continuation,
   where a shallow draft agrees less.

### Open Problem 4.14 was the wrong question

Exact closed forms: a projected bit flips back with the same probability it flipped, so
the state is a symmetric two-state chain and **projection never locks errors in**. With a
rate-sufficient codebook, `k=1` — discretise every step — wins at every noise level. With
an insufficient codebook the trade-off is real but **binary**: never project until drift
`σ√T` exceeds the rounding error, then every step. No clear interior-`k` winner in any of
140 cells at 20k trials. The question isn't "how often"; it's "does the rate clear the
task's precision", and the learned-codebook version is the only part still open.

### Three times an optimisation artefact impersonated a capacity result

This is the methodological lesson, and it cost more time than the theory did.

- **Undertraining.** A sweep at 800 steps showed seven failures *with sufficient capacity*,
  including 4 bits failing to store 2. At 3000 steps that cell hits 1.000.
- **Parameterisation.** A one-hot codebook over `2^b` symbols never cleared threshold at
  n≥3, because the optimiser must *discover* an injective code through a biased
  straight-through gradient. Rewriting the same `b` bits as `b` independent binary units —
  identical capacity — produced the exact predicted diagonal.
- **Bottleneck depth, and this one flattered the thesis.** At T=12 the discrete interface
  scored ~0 at *every* width while continuous scored 1.000 — which reads as a textbook
  confirmation of the bandwidth argument. It isn't: a bound binding at 6.91 bits cannot
  explain failure at 20 bits. The real constraint was credit assignment through twelve
  stacked quantisations. At T=4 the genuine threshold appears cleanly.

The general check: before reading a discrete-bottleneck failure as a capacity bound, verify
the failure is monotone in the bottleneck width and absent well above the claimed threshold.
And report violations — successes *below* threshold — separately from match rates, since
only that direction can refute a lower bound.

## Reproducing

```
pip install torch                                  # pypi works; pytorch.org is blocked here
python3 experiments/exp3_rnn_is_hmm.py             # stdlib only, ~10s
python3 experiments/exp0_pretrain.py --steps 1500 --vocab 2048 --d 128 --layers 6 --ctx 64 --bs 16
python3 experiments/exp1_local.py --rounds 200
python3 experiments/exp3b_recall_capacity.py --steps 2500 --d 96 --ns 2 3 4 5 --bits 1 2 3 4 5
python3 experiments/exp2_latent_vs_cot.py --n 5 --T 4 --steps 4000 --seeds 2 --sweep-v --bit-list 3 4 5 6 7 8 15
```

Build the paper with `cd paper && latexmk -pdf main.tex`. `neurips_2024.sty` is unavailable
here (media.neurips.cc is blocked), so the preamble falls back to a close approximation;
drop the real style file in and flip one line to switch.

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
