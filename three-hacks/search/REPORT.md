# Search report — 20 rounds on the depth / time / state axes

## The result

Starting from a model naive on all three axes, a theory-guided search with a pre-registered
acceptance rule found a configuration **79% better on held-out seeds** (F 1.560 ± 0.056 vs
0.873 ± 0.032): dense-transition mixing with one attention layer, plus an early-exit auxiliary
loss, no latent loops. Two of the three axes moved; the third did not, twice.

| axis | theory said | search found | held-out contribution |
|---|---|---|---|
| **depth** | early-exit loss lifts self-drafting over the head ceiling (exp5) | accepted in v1 *and* v2 at the identical +0.302, task scores untouched | **+0.319** |
| **state** | dense mixing tracks hidden state where diagonal can't (Thm 5.8, exp7) | accepted in v2; TRACK 0.56 → 0.64; keep the attention layer | **+0.201** |
| **time** | append beats overwrite (Cor 4.5) | loops worse at every budget and encoding; **overwrite beat append twice** in matched tests | — |

Depth and state combine slightly super-additively (+0.687 together on held-out).

## What the twenty rounds were

**v1, rounds 0–13.** Naive baseline, τ=0.097. Round 3 (early-exit loss) was the only accepted
move. Ten further rounds across all three axes and two learning rates produced nothing above
the band. Diagnostics — not assumptions — showed why: RECALL sat at 0.33–0.36 for *every*
architecture including four attention layers trained on it alone at 4× budget, and a localizer
pinned the failure to the first key match (a one-pair copy trains to 1.000; two pairs drop to
0.55). Two-hop retrieval does not form in this model at this budget. Architecture cannot
register on a task no architecture can learn.

**v2, rounds 14–20.** Same model, same rule, same COMPOSE and TRACK, one change: RECALL as one
token per pair (one-hop, identical semantics), gated first — attention now climbs (0.27→0.66),
diagonal doesn't (0.375). τ₂=0.015. Depth +0.302, state +0.139, both loops worse, combined
+0.579 → incumbent. Round 20 (drop the attention layer) tied on search seeds and lost on
held-out.

## What the search says about the theory

- The **depth** prediction is the most robust thing in the project: same magnitude in two
  searches on two encodings, orthogonal to every task score. It is also the cheapest move.
- The **state** prediction holds exactly where Thm 5.8 locates it — the mixing task — and
  nowhere else. RECALL never moved with architecture in the three-task mix.
- The **time** prediction did not hold as tested. Cor 4.5 is a capacity claim about long
  horizons; at r=1 on a model that cannot yet use one loop, capacity is not what binds. The
  search recorded the loss both times rather than explaining it away. What it would take to
  test the corollary properly: a task whose serial depth exceeds what four layers can do
  without loops, and enough budget for the loops to train — neither available here.

## Rounds 21–40: autonomous, three seeds, held-out before promotion

**Result: one promotion.** Learning rate 4e-3 (round 21) took the incumbent from held-out
1.560 to **1.806** — the largest single move of the project, and a *recipe*, not an
architecture. v1 had seen it (+0.057 under τ=0.097) and could not accept it. Nothing else in
twenty rounds cleared τ; the stacked near-misses of rounds 39–40 landed at +0.127 and +0.187
against τ=0.20.

**The time axis was never measured before round 36.** Five loop configurations had lost
across both searches, and the ledger recorded each as evidence against Cor 4.5. Round 32 —
placeholder slots inserted, *zero* loops — lost by the same −0.537. A dense recurrence with a
spectrally-normed transition and gates below one contracts over four steps of constant input:
filler tokens erase its memory. Attention wouldn't care. One line (`skip_slots`: hold recurrent
state at placeholder positions) removed the penalty exactly — round 36 matches the incumbent —
and rounds 37–38 became the first fair loop tests. Both lose by ~0.48. Append ties overwrite
(+0.018), so the corollary's *direction* survives the one honest test it has had; loops as a
mechanism, on this benchmark at this budget, do not.

**The state axis found its mechanism in a losing round.** RECALL sat at 0.25–0.36 under every
layout for thirty-four rounds. Round 34 put attention at both ends and RECALL jumped to 0.84
— the first movement ever — while TRACK and COMPOSE collapsed. The follow-up (attention
first, three dense after) confirmed it: RECALL 0.86, COMPOSE 0.11. Content lookup needs
attention on raw token embeddings before the recurrence's normalised mixing erases key
identity; composition needs dense mixing on those same raw tokens. **The first layer is
contested**, and four blocks have one. Attention-last never helped (rounds 27, 30, 35) for
exactly this reason.

**Recipe mapped.** lr 6e-3 turns over (−0.062); 3e-3 is worse than 4e-3 (−0.024). λ=0.5 is
near-optimal; λ=0.75 helps COMPOSE (0.49) but not enough overall.

### What the second twenty taught about the first

The v1 protocol's wide τ (0.097, from one outlier baseline seed) hid the single largest
lever. The v1/v2 time-axis "verdicts" were artefacts of a sequence-layout choice. And the
state-axis null result on RECALL was a first-layer conflict, not an absence of effect. All
three were found by controls and diagnostics inside the search — the `slots_only` round, the
attention-ends round, the three-seed rule — not by assumption.

## What kept it honest

- τ fixed from the baseline before any candidate ran; acceptance by margin, not by max.
- Budgets enforced before compute was spent; "make it bigger" never a legal move.
- Depth credit gated on task quality, after an untrained model scored full marks.
- Two gates before restarting on a new encoding; the first (bigger budget) **failed**.
- One silent no-op edit caught by a crashing diagnostic, not by review.
- The final claim made on seeds the selection never touched — which reversed round 20.

## Files

`dts.py` architecture + benchmark · `run_round.py` one round under budget · `ledger.py`
verdicts · `holdout.py` fresh-seed comparison · `LEDGER.md` every round · `rounds/` every
log and JSON, including the diagnostics that changed the plan.
