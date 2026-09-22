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

**Diagnostic result (RECALL).** All-attention model, RECALL only, full 1500-step budget:
0.203 → 0.328 → **0.359**; at lr 5e-3, 0.312. Four attention layers on a six-pair lookup
should approach 1.0. The task needs two-hop retrieval, which forms after a phase transition,
and 1500 × 32 examples is short of it. **Two axes are pinned by the training budget, not by
architecture.** That is a harness design flaw, found in batch 3. Gating test for a revised
budget (3000 steps × bs 64): `rounds/_diag_recall_v2budget.log`. If it clears, the search
restarts as v2 with the same tasks and a larger fixed budget, fully disclosed; if not, it is
a bug hunt.

**Gate FAILED.** At 3000 steps × bs 64 (4× the examples) all-attention RECALL is 0.328 — flat,
not slow. A six-pair lookup four attention layers cannot learn from 192k examples is a broken
harness, not a hard task. ~0.3 is what "emit any value seen in context, ignore the key"
scores. **No v2 restart until the cause is found.** Localizer sweeping n_pairs from 1 (pure
copy) to 6: `rounds/_diag_recall_bug.log`.

**Localizer result.** All-attention, RECALL only, 1200 steps × bs 64, sweeping n_pairs:
`1 → 1.000 (loss 0.001)`, `2 → 0.550`, `3 → 0.444`, `6 → 0.400`. The pipeline is sound (a pure
copy trains perfectly); the failure begins at the first point a key match is required. That is
the induction cliff — two-hop retrieval needs a previous-token head feeding a content-match
head, and it does not form in this model at this budget for any n ≥ 2. **Not a bug; a task
encoding that no candidate can learn.** v2 encodes each pair as one token (one-hop content
match, same semantics), behind `recall_mode="pairs"`; v1 rounds keep their interleaved
record. Gate before restart: `rounds/_diag_recall_pairs.log`.

**v2 gate: PASS, on the criterion that matters.** One-hop RECALL, 1200 single-task steps:
all-attention `0.272 → 0.423 → 0.661` and still rising steeply; diagonal state `0.375`.
Under the interleaved encoding both sat at ~0.35 whatever the architecture; now block types
*separate* and the curve is not flat. It does not saturate at this budget, so v2 measures
state and time in a partial-learning regime where architecture appears as slope, not as
ceiling. Stated as such.
| 9 | r09_lr4e3 | recipe | lr 4e-3 | 1.173±0.013 | +0.057 | no change. recall 0.35 regardless |
| 10 | r10_lr1e3 | recipe | lr 1e-3 | 0.994±0.001 | −0.122 | worse |
| 11 | r11_all_dense | state | dense×4 + λ=0.5 | 1.136±0.049 | +0.021 | no change. track 0.56, best on the mixing task |
| 12 | r12_deep_exits | depth | λ=0.5 on layers 2,3 only | 1.118±0.063 | +0.003 | no change. S 1.70→1.35, no quality gain |
| 13 | r13_overwrite_proj | time | overwrite r=1 + proj every step | 0.996±0.019 | −0.119 | worse; **0.996 vs 0.998 unprojected** — projection neither helps nor hurts, as Prop 4.15 says for a noise-free chain |

**v1 closed: 13 rounds, one accepted move (depth, round 3), incumbent r03 F=1.115.** Every
state and time mutation was a wash or worse because RECALL was unlearnable under the
interleaved encoding and COMPOSE sat near floor; the two recipe moves confirmed that lr is
not the bottleneck. The one theory prediction that reproduced cleanly is the depth axis. The
matched time test went against Cor 4.5 (overwrite > append by 0.104 at r=1).

## Search v2 — one-hop RECALL, 2400 steps, same COMPOSE/TRACK, same rule

τ₂ is set from the v2 baseline (3 seeds) before any v2 candidate is scored. Rounds 14–19 are
pre-committed (baseline; depth, state, and the matched time pair in isolation; depth+state
combined). Round 20 is chosen from their results. Then a held-out comparison on fresh seeds.

| round | name | axis | mutation | F | Δ vs inc | verdict |
|---|---|---|---|---|---|---|
