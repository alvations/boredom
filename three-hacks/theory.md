# Errata, and where the formal treatment lives

The formal development now lives in [`paper/`](paper/) (NeurIPS format,
`paper/main.tex` plus `paper/sections/`). This file used to hold the proofs. It was
audited section by section, adversarially, and enough of it was wrong that keeping a
second divergent copy would be worse than useless. What follows is the record of what
was wrong, because the corrections are the most useful thing here.

## Errata

### Old §2 (depth / early exit) — now paper §3

| Was | Actually | Fixed in |
|---|---|---|
| Exactness stated with a support hypothesis on `q` | No hypothesis is needed at all. `q(x)=0<p(x)` is supplied by the residual branch with exactly the right mass. The spurious hypothesis also undercut the very claim the theorem exists to support | `Thm 3.1` |
| Cost `C_self = γ·ℓ/L + 1` | **Omits the unembedding.** Every draft step must stream `W_U` to sample from `p_ℓ` — ~26% of Qwen3-0.6B. Inflates speedup by up to 1.39× at `ρ=0.25` and caps it at `(γ+1)/(γu+1)` | `Def 3.14`, `Prop 3.15`, `Cor 3.17` |
| `TV ≤ 1 − e^{−2ε}`, `ε = ‖u−v‖_∞` | Softmax is shift-invariant, `‖·‖_∞` is not — slack by up to 2×. Use the spread `max−min` | `Lemma 3.3` |
| `κ ≤ √d·max|g|/‖h‖` for RMSNorm | Missing a factor 2, and "Lipschitz on a region containing two points" is vacuous — a two-point set admits any constant. Replaced by an exact two-point inequality needing no Lipschitz hypothesis | `Lemma 3.5` |
| `‖W_U‖_{2,∞}` in the acceptance bound | Only *differences* of unembedding rows matter to a softmax, so the diameter `D_U = max_ij‖w_i−w_j‖` is the right constant | `Thm 3.7` |
| Suggested rank-restricted refinement | **Not valid unconditionally.** Needs `B_P/A_P ≲ η/2` — a joint condition on `W_U` *and* the tail, essentially unsatisfiable. Replaced by an unconditional two-term bound | `Prop 3.9`, `Rmk 3.10` |
| Yield formula asserted with an i.i.d. gloss | The parenthetical derivation mixes accepted drafts with emitted tokens and is short by `1−α^γ`. The exact identity `E[Y] = 1 + Σ_j P(A_j)` needs no independence | `Prop 3.11` |

### Old §3 (time / latent reasoning) — now paper §4

| Was | Actually | Fixed in |
|---|---|---|
| "`n` latent steps reach `2^{n·d·b}` states" | **False, and it reverses the conclusion.** Iterating a deterministic map cannot grow cardinality: `n` steps of a fixed-size latent state reach `2^{d·b}`, with no `n` in the exponent. The real axis is *accumulation*, and a token chain overtakes a fixed-size latent loop at `n* = d·b/log₂V ≈ 238` | `Thm 4.4`, `Cor 4.5` |
| §3 and §4.4 | Were contradicting each other — the bandwidth claim and the recall ceiling are the same fact seen from opposite sides | `Rmk 4.6` |
| Counting corollary ("distinguish `K` configurations") | **False.** Intermediate states of a deterministic task are functions of the prompt: a prompt with two elements of `S₃₀` induces `30!` configurations distinguished at `n=0` by attention. It also combined two *lower* bounds as though one were an upper bound | `Rmk 4.9` |
| "decides exactly `P`" | Category error — a model decides a language, `P` is a class. Also drops the uniformity hypothesis (a condition on the *model*) and generalized pre-norm | `Thm 4.1`, `Rmk 4.2` |
| Simulation via hard `argmax` | `argmax` is discontinuous. Repaired with temperature `τ`, a uniform logit margin `δ`, and Lipschitz blocks — at the price of `Ω(n log λ)` precision, which contradicts the log-precision hypothesis of the complexity result. Both are stated | `Thm 4.7`, `Rmk 4.8` |
| Strictness of the inclusion | Does not follow. Available unconditionally only under a streaming read schedule; in the full-attention setting it would imply `TC⁰ ≠ NC¹` (Barrington) | `Thm 4.10`, `Rmk 4.13` |
| Conjecture "noisy channel with drift `η^n`" | Not well-posed — `η^n` has the wrong monotonicity, and quantisation *amplifies* error near a cell boundary | `Open Problem 4.14` |

### Old §4 (state / gated RNNs) — now paper §5

| Was | Actually | Fixed in |
|---|---|---|
| "constrain a gated linear RNN and its recursion becomes the HMM forward algorithm" | **Impossible.** Gated linear attention acts *diagonally*; the forward algorithm needs a *dense* transition. Two independent proofs (support monotonicity; simultaneous diagonalisability), both confirmed numerically. A gate constrained non-negative and stochastic gives the identity, not a transition matrix | `Thm 5.8` |
| "transformers are RNNs with growing state — that is the whole difference" | True but vacuous — every causal model admits such a form, and the trivial state is ~10⁴× *smaller* than the KV cache, so transformers expand rather than compress history. The content is incrementality and per-step cost | `Thm 5.2`, `Rmk 5.1` |
| Forward-algorithm base case "induction from `α_0 = π`" | Applies one spurious transition. The code always implemented the correct base case, so the script was verifying a recursion the proof did not state | `Thm 5.6` |
| Hypotheses of the HMM identity | Row-stochasticity is unused for the identity (pure algebra); non-negativity is load-bearing for Viterbi only. Now split into three parts | `Thm 5.6` |
| Recall ceiling by pigeonhole | **False** without finite precision: `s_t = s_{t−1}/2 + x_t/2` stores unbounded history in one real. Needs `m`-bit statefulness, determinism, and an adversary-order clause | `Def 5.11`, `Thm 5.12` |
| "falls off a cliff" prediction for exp3b | The Fano form predicts *graceful* degradation. Retracted | `Rmk 5.13` |
| Mamba-2 SSD duality as stated | Holds for *scalar* gates; the vector-gated form written alongside it has no structured-mask dual | `Rmk 5.5` |

### Experiments

| Was | Actually |
|---|---|
| `exp3` check 1 | **Circular** — compared a nested loop against its own list-comprehension rewrite. Replaced by per-`(t,j)` brute-force path enumeration, plus a negative control, plus direct tests of the impossibility result |
| `exp2` confound argument | Used the wrong quantity: composition is associative, so recomputation is a balanced tree of depth `log₂T`, not `T`. `T=24` against 4 layers left the escape open. Redesigned around an enforced streaming schedule with an unsupervised interface |
| `exp2` `rank(perm) % V` | Tests routing around a corrupted training signal, not a bandwidth limit |
| `exp1` speedup | Missing the head cost; kill criterion is `max S ≤ 1`, so the omission could invert the verdict |

## Errata round 2: found by running the experiments

The audits were done before anything was run. Running it found four more, three of which
invert a claim rather than tighten it.

| Was | Actually | Fixed in |
|---|---|---|
| Round cost charges drafting and verification | **Omits the bonus position's full-depth pass.** Drafting touches positions `t..t+γ-1`; verification also needs full depth at `t+γ`. At `ρ=1` the broken model returns `S=1.735`, but `ρ=1` means the draft *is* the target, so `S=1` is forced | `Prop 3.15`, `Rmk 3.16` |
| Cache reuse is the structural advantage of self-drafting | **Backwards when memory-bound.** Reuse splits verification across two depths, and a memory-bound pass costs a whole weight stream however many positions it covers, so the lower blocks are paid twice: 2.25 vs 1.50 at `ρ=0.25, γ=2`. Helps only when compute-bound. I defended this against the audit and was wrong | `Prop 3.19`, `Rmk 3.20` |
| Acceptance-bound looseness is mainly Cauchy–Schwarz | **The softmax stage alone is fatal** — 46× too small at `ℓ=5` even when fed the measured spread. Cauchy–Schwarz adds a further 2.5×. A useful bound needs a different proof strategy, not a tighter constant | `Rmk 3.10`, §6.3 |
| Geometric yield at mean α under-estimates (Jensen) | **It over-estimates**, by 12%: 1.320 measured against 1.481. Jensen governs variation *between* contexts; within a round each accepted draft moves the model onto its own continuation, where a shallow draft agrees less | §6.6 |
| Bounded state fails off a cliff | **Graceful degradation**, as the Fano form predicts: 0.83 / 0.76 / 0.71 at `b=n-1`. Pre-registration withdrawn | `Rmk 5.13`, §6.8 |

### Experiment errata

| Was | Actually |
|---|---|
| Corpus globs `**/*.py` | The experiment sources live in this repo, so **editing an experiment changed its own evaluation data** — measured α drifted 0.189 → 0.249 across two runs of one script. Corpus now excludes `experiments/`, and the validation split is frozen into the checkpoint |
| exp3b at 800 steps | Undertrained. Seven cells failed *with sufficient capacity*, including 4 bits failing to hold 2 bits; at 3000 steps that cell reaches 1.000 |
| exp3b/exp2 one-hot interface over `2^b` symbols | Same `b` bits, but the optimiser must **discover** an injective code through a biased straight-through gradient, and largely cannot. `b` independent binary units make the natural solution directly representable and reproduce the predicted threshold |
| exp2 at T=12 | Discrete scored ~0 at every width, continuous 1.000 — which reads as confirming the bandwidth thesis. **It doesn't**: a bound binding at 6.91 bits cannot explain failure at 20 bits. The constraint was credit assignment through twelve stacked quantisations. At T=4 the real threshold appears |
| "cells match prediction" as the summary statistic | Weights both error directions equally, when only **success below threshold** can refute a lower bound. Violations now reported separately |

## Errata round 3: the three open questions

| Was | Actually | Where |
|---|---|---|
| Conjecture: discretisation *locks in* errors, so there is a bandwidth-vs-drift trade-off with an interior optimum `k*` | **No lock-in exists.** A projected bit flips back with the same probability it flipped, so the projected state is a symmetric two-state chain and `k=1` is optimal at every noise level when the codebook rate suffices — exact closed form, Monte Carlo at 20k trials. With an *insufficient* codebook the trade-off is real but **binary** (never vs. every step), switching on `σ√T` against the rounding error, with no clear interior-`k` winner in any cell. Open Problem 4.14 asked the wrong question | `Prop 4.15`, §6.7 |
| exp7 on a peaked random HMM printed "DISPROOF: Thm 5.8 is practically empty" | **Unearned.** A sharply peaked HMM's belief collapses after a symbol or two, so no mixing is ever needed and a diagonal recurrence matching a dense one is *expected*. The script now measures belief entropy and refuses to rule on bite unless the hidden state must actually be tracked. The cyclic HMM — belief rotates, complex eigenvalues, the case Thm 5.8(b) names — is the test that can decide it | `exp7 --hmm cyclic` |
| Thesis 1 dead at Qwen's head fraction (`S = 1.000`, degenerate `ρ=1`) | **Rescued by training, marginally.** 600 steps of auxiliary early-exit loss: `S = 1.072` at `u=0.26`, `1.146` at `u=0.153`, with tail energy falling at every layer — the mechanism Thm 3.7 names. +0.08 final-layer val loss. A 7% margin on an inflated-acceptance model is not something to build on, but the sign is the predicted one | §6.6 |

## What survived unchanged

- Speculative sampling is exact for an arbitrary draft, and acceptance is `1 − TV(p,q)`.
- The acceptance bound is monotone in residual tail energy (the useful shape; the
  constants needed work).
- Self-drafting dominates an external draft of comparable relative cost — now proved in
  every regime `w`, not just the memory-bound endpoint.
- `TC⁰ ≠ P` kills the strong form of the latent-reasoning complaint.
- The HMM forward identity and its Viterbi variant, verified against a brute-force oracle
  per `(t,j)`, with a negative control confirming the hypothesis split.
- The no-embedding impossibility, confirmed numerically on both obstructions.
- Exactness for an arbitrary draft, confirmed at the sampling-noise floor — this is what
  makes the shortlisted draft head legitimate.
- Both capacity bounds: no cell below either threshold ever succeeded.
