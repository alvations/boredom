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

**Diagnostic result.** COMPOSE-only training, full 1500-step budget, naive model: exact-match
0.125 → 0.234 → 0.266 at steps 500/1000/1500, per-token 0.82. Learnable, slowly. The naive
architecture's single-task ceiling at this budget is ~0.27; in the three-task mix it gets a
third of the steps and reaches 0.13. The time axis is live and hard, not dead — credit on it
must come from an architecture that composes better than diagonal blocks. Benchmark unchanged.
| 4 | r04_state_on_depth | state | dense,dense,attn,dense + λ=0.5 | 1.119±0.081 | +0.003 | no change. track 0.52→0.55, recall 0.36→0.33 |
| 5 | r05_append_r1 | time | append r=1 + λ=0.5 | 0.894±0.022 | −0.222 | worse. recall 0.24, compose 0.07 |
| 6 | r06_overwrite_r1 | time | overwrite r=1 + λ=0.5 | 0.998±0.043 | −0.117 | worse — but **beats append by 0.104 (> τ)**: matched test goes against Cor 4.5 here |
| 7 | r07_lambda1 | depth | λ=1.0 | 1.104±0.023 | −0.011 | no change. S 1.67→1.70, recall dips; plateau |
| 8 | r08_two_attn | state | attn,dense,attn,dense + λ=0.5 | 0.948±0.007 | −0.168 | worse. track 0.52→0.42 (a mixing layer lost); recall **unchanged at 0.35** |

**Note after batch 2.** Incumbent unchanged (r03, F=1.115). Only the depth axis has ever
moved. RECALL is 0.33–0.36 for *every* layout including two attention layers, which should
solve associative recall outright; COMPOSE is 0.07–0.15 everywhere. The state and time tasks
are bottlenecked by trainability at this budget, not by architecture, so architecture
mutations cannot register on them. Batch 3 targets that: two *recipe* moves within budget
(tagged separately from the three axes), and axis moves that follow from batch 2.
Diagnostic: RECALL-only, all-attention — `rounds/_diag_recall.log`.
