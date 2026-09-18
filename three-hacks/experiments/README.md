# Experiments

Ordered by cost. Nothing here trains anything bigger than ~10M parameters, and two of
the four need no training at all. Each has a kill criterion in its docstring; the point
is to spend an afternoon disproving a thesis, not a week confirming one.

| file | needs | trains? | tests |
|------|-------|---------|-------|
| `exp3_rnn_is_hmm.py` | nothing (stdlib) | no | Theorem 4.3 — gated non-negative linear RNN **is** the HMM forward algorithm |
| `exp1_early_exit.py` | torch, transformers, Qwen3-0.6B | no | Theorems 2.2–2.4 — early-exit acceptance rate and the speedup surface |
| `exp3b_recall_capacity.py` | torch, transformers, Qwen3-0.6B | no | Theorem 4.4 — where fixed state breaks, via window attention |
| `exp2_latent_vs_cot.py` | torch | ~10M from scratch | Theorem 3.2 — the bits-per-step bound, with `V` as the knob |

```
pip install -r requirements.txt
python3 exp3_rnn_is_hmm.py                                  # runs anywhere, ~1s
python3 exp1_early_exit.py  --model Qwen/Qwen3-0.6B
python3 exp3b_recall_capacity.py --model Qwen/Qwen3-0.6B
python3 exp2_latent_vs_cot.py --n 6 --sweep-v
```

## Status

Only `exp3_rnn_is_hmm.py` has been executed — it passes, including against a
brute-force enumeration oracle rather than just a termwise comparison. The other
three are written but **unrun**: the container they were written in has no torch,
no numpy and no GPU. Every prediction in the parent README is therefore a
prediction and should be read as one.

## Order to run them in

1. **`exp3_rnn_is_hmm.py`** — free, and if Theorem 4.3 were wrong the whole framing
   of thesis 3 would be off. (It isn't.)
2. **`exp1_early_exit.py`** — free apart from forward passes, and the single most
   informative number in the project is the measured `alpha_l` curve against the
   Theorem 2.3 bound. Expect the bound to be vacuous; the useful output is the
   `slack` column saying which inequality threw the information away.
3. **`exp3b_recall_capacity.py`** — free, and it either validates the window-as-state
   proxy or tells you the proxy is wrong before anyone distils a linear-attention model.
4. **`exp2_latent_vs_cot.py`** — the only one that trains, and the only one whose
   result would be genuinely new. Run it last, run it longest.

## What would change my mind

- `exp1` best speedup `<= 1.0` even after an early-exit LoRA → thesis 1 is an
  aesthetic preference, not an engineering claim.
- `exp2` CoT accuracy flat across `log2(V) < log2(n!)` → the bandwidth bound does not
  bind on real trained models, and the interesting version of thesis 2 dies.
- `exp3b` smooth decay rather than a cliff → the window proxy is wrong; redo it with
  an actual distilled fixed-state model before believing anything about capacity.
