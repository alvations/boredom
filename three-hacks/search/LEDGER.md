# Search ledger

`F = state + time + depth`. τ = baseline across-seed std of F, fixed after round 0.
Accept iff `F(cand) − F(inc) > τ` on 2 seeds. Budgets: params ≤ 1.25×, compute ≤ 3.5× round 0.

| round | name | axis | mutation | F | Δ vs inc | verdict |
|---|---|---|---|---|---|---|
| 0 | r00_naive | baseline | diag×4, r=0, λ=0 (3 seeds) | 0.813±0.097 | — | **incumbent**; τ=0.097 |
| 1 | r01_state | state | dense,dense,attn,dense | 0.897±0.034 | +0.084 | no change (< τ). track 0.51→0.57 as exp7 predicts; recall flat |
| 2 | r02_time | time | append r=2, 4 slots | 0.677±0.031 | −0.136 | **worse**. recall 0.36→0.20, compose 0.13→0.07 at 3× compute. Loops hurt at this budget; append-vs-overwrite still untested |
| 3 | r03_depth | depth | ee_lambda=0.5 | 1.115±0.016 | **+0.302** | **ACCEPT → incumbent**. S 1.116→1.666, state/time unchanged — exp5 exactly |

**Note after batch 1.** COMPOSE is near floor for every config (0.07–0.15). The time axis
contributes almost nothing to F until the model can learn it; the benchmark stays fixed
(no mid-search changes), so time-axis credit has to be *earned* by an architecture that
makes COMPOSE learnable within budget. Learnability diagnostic: `rounds/_diag_compose.log`.
