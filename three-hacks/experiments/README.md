# Experiments

Ordered by cost. Nothing here trains anything bigger than ~10M parameters, and two of
the four need no training at all. Each has a kill criterion in its docstring; the point
is to spend an afternoon disproving a thesis, not a week confirming one.

| file | needs | trains? | tests |
|------|-------|---------|-------|
| `exp3_rnn_is_hmm.py` | nothing (stdlib) | no | Thm 5.6 — the HMM identity; **and Thm 5.8 — the bridge to linear attention is impossible** |
| `exp1_early_exit.py` | torch, transformers, Qwen3-0.6B | no | Thm 3.7 / 3.16 / Cor 3.17 — acceptance, speedup surface, head-cost ceiling |
| `exp3b_recall_capacity.py` | torch, transformers, Qwen3-0.6B | no | Thm 5.12 — where bounded state breaks, via window attention |
| `exp2_latent_vs_cot.py` | torch | ~1M from scratch | Thm 4.10 — interface bandwidth, under an enforced streaming schedule |

```
pip install -r requirements.txt
python3 exp3_rnn_is_hmm.py                                  # runs anywhere, ~1s
python3 exp1_early_exit.py  --model Qwen/Qwen3-0.6B
python3 exp3b_recall_capacity.py --model Qwen/Qwen3-0.6B
python3 exp2_latent_vs_cot.py --n 5 --sweep-v
```

## Status

`exp3_rnn_is_hmm.py` runs and passes: the HMM identity checked per `(t,j)` against
brute-force path enumeration, a negative control confirming non-negativity matters for
Viterbi only, and both impossibility checks for the bridge. The other three are written
but **unrun** — no torch, no numpy, no GPU in this container.

An earlier version of `exp3` had a **circular** first check (a nested loop compared
against its own list-comprehension rewrite, which cannot fail). It was replaced by the
brute-force oracle. Worth remembering when a test passes on the first try.

## Order to run them in

1. **`exp3_rnn_is_hmm.py`** — free, and it carries the section's main negative result.
   (Runs; both parts pass.)
2. **`exp1_early_exit.py`** — free apart from forward passes, and the single most
   informative number in the project is the measured `alpha_l` curve against the bound.
   Expect the bound to be vacuous; the useful output is the `slack` column saying which
   inequality threw the information away, and whether `S > 1` survives the head cost.
3. **`exp3b_recall_capacity.py`** — free, and it either validates the window-as-state
   proxy or tells you the proxy is wrong before anyone distils a linear-attention model.
4. **`exp2_latent_vs_cot.py`** — the only one that trains, and the only one whose
   result would be genuinely new. Run it last, run it longest. Its read schedule is
   load-bearing: without it the task is solvable in one step and the experiment is void.

## What would change my mind

- `exp1` best speedup `<= 1.0` even after an early-exit LoRA → thesis 1 is an
  aesthetic preference, not an engineering claim.
- `exp2` accuracy flat across `log2(V) < log2(n!)` **with the streaming schedule
  enforced** → the bandwidth bound does not bind, and thesis 2's last unconditional
  result dies. Without the schedule enforced the experiment says nothing either way.
- `exp3b` break-point independent of `w` → the window proxy is wrong; redo it with an
  actual distilled bounded-state model. Note the Fano bound predicts *graceful* decay,
  so the earlier "sharp cliff" prediction has been retracted — a smooth fall is now the
  expected outcome, not a refutation.
