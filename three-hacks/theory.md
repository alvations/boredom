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

## What survived unchanged

- Speculative sampling is exact for an arbitrary draft, and acceptance is `1 − TV(p,q)`.
- The acceptance bound is monotone in residual tail energy (the useful shape; the
  constants needed work).
- Self-drafting dominates an external draft of comparable relative cost — now proved in
  every regime `w`, not just the memory-bound endpoint.
- `TC⁰ ≠ P` kills the strong form of the latent-reasoning complaint.
- The HMM forward identity and its Viterbi variant, now verified against an oracle.
