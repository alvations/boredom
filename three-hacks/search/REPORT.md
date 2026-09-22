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
