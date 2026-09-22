# Search: iterating a depth/time/state architecture

A theory-guided architecture search over the three axes the paper argues about. The
proposer is the theory (and me reading results); the harness is what keeps it honest.

## The architecture — `dts.py`

Every claim the paper made is a switch, next to its naive alternative:

| axis | switch | naive | theory says | source |
|---|---|---|---|---|
| state | block type per layer | `diag` | `dense` mixing + a few `attn` for recall | Thm 5.8, 5.12; exp7, exp3b |
| time | `r` latent loops, `time_mode` | `overwrite` | `append` (accumulate) | Cor 4.5 |
| time | codebook projection every `proj_k` | never | `k=1` if the rate suffices | Prop 4.15 |
| depth | early-exit loss `ee_lambda` | 0 | > 0 | exp5 |

## The benchmark — one task per axis, one shared vocabulary

| task | what it probes | score |
|---|---|---|
| RECALL | state capacity: 6 key/value pairs then a query | accuracy |
| TRACK | state mixing: cyclic-HMM next symbol, vs the Bayes-optimal forward algorithm | fraction of headroom captured |
| COMPOSE | serial time: 8 generators of S₅ → the product | exact-match accuracy |
| DEPTH | early-exit speedup at an imposed head fraction u=0.26, corrected cost model | `clip(2(S−0.9), 0, 1)` **gated by task quality** |

`F = state + time + depth ∈ [0, 3]`, `state = (recall + track)/2`, `time = compose`.

The depth gate matters: an untrained model has every layer near-uniform, so α≈1 and S is
maximal — a model that predicts nothing would score full depth credit. Depth credit scales
with `min(1, state + time)`.

## Protocol — pre-registered before round 1

- **Budgets.** Params ≤ 1.25× and compute ≤ 3.5× the round-0 reference. Over-budget
  candidates are written as rejected and never scored. "Make it bigger" is not a move.
- **Seeds.** 2 per candidate; 3 for the baseline. Held-out eval on fresh procedural samples.
- **Noise band τ** = the baseline's across-seed std of F, fixed *before* any candidate runs.
- **Acceptance.** A candidate replaces the incumbent iff `mean F(cand) − mean F(inc) > τ`.
- **Attribution.** Every candidate is tagged with the axis it mutates and the theorem or
  result that motivates it. Rounds 1–3 are the three theory-predicted moves, one per axis,
  applied to the naive baseline, so each axis's contribution is measured in isolation.
- **Final check.** Incumbent vs round-0 vs the three-axis theory config on fresh seeds.

## Ledger

[`LEDGER.md`](LEDGER.md) — every round: config, axis, rationale, result, accept/reject.
Raw per-round JSON in [`rounds/`](rounds/).
