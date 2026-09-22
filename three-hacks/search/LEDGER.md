# Search ledger

`F = state + time + depth`. τ = baseline across-seed std of F, fixed after round 0.
Accept iff `F(cand) − F(inc) > τ` on 2 seeds. Budgets: params ≤ 1.25×, compute ≤ 3.5× round 0.

| round | name | axis | mutation | F | Δ vs inc | verdict |
|---|---|---|---|---|---|---|
