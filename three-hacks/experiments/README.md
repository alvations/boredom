# Experiments

Every script used in this project, what it tests, and what it found. Theorem numbers
refer to [`../paper/main.pdf`](../paper/main.pdf); the write-up is §6.

## Environment

This container could not reach `huggingface.co` or `download.pytorch.org` — the agent
proxy denies both — so **no public checkpoint was used anywhere**. `pypi.org` is reachable,
so torch installs normally. All language-model measurements are on `exp0`'s model, trained
here, which satisfies the paper's Definition 2.1 exactly (pre-norm residual blocks,
RMSNorm, unembedding).

```
pip install torch          # NOT --index-url download.pytorch.org: blocked
```

## Main experiments

| script | needs | tests | result |
|---|---|---|---|
| [`exp3_rnn_is_hmm.py`](exp3_rnn_is_hmm.py) | stdlib only | Thm 5.6 HMM identity, **Thm 5.8 no-embedding** | both confirmed |
| [`exp0_pretrain.py`](exp0_pretrain.py) | torch | trains the model everything else measures | val ppl 431, u=0.153 |
| [`exp1_local.py`](exp1_local.py) | torch | Thm 3.1 exactness, 3.7 bound, 3.11 yield, 3.21 speedup | exactness confirmed; **cost model was wrong** |
| [`exp3b_recall_capacity.py`](exp3b_recall_capacity.py) | torch | Thm 5.12 recall ceiling | 19/20 cells, no violation |
| [`exp2_latent_vs_cot.py`](exp2_latent_vs_cot.py) | torch | Thm 4.10 interface bandwidth | threshold confirmed at T=4 |
| [`exp1_early_exit.py`](exp1_early_exit.py) | torch, **HF access** | same as `exp1_local` but on Qwen3-0.6B | **unrun** — network blocked |

`exp1_early_exit.py` is kept because it is the version that should be run once a real
checkpoint is reachable; it is the only script here that has never executed.

## Diagnostics

These exist because three separate optimisation artefacts imitated capacity results during
this project. Each script isolates one. They are small, and `d1` is the most valuable thing
in this directory.

| script | needs | what it isolates |
|---|---|---|
| [`d1_cost_model_sanity.py`](diagnostics/d1_cost_model_sanity.py) | stdlib, ~1s | **the ρ→1 check that caught a bug in our own theory** |
| [`d2_undertraining.py`](diagnostics/d2_undertraining.py) | torch | training budget imitating a capacity ceiling |
| [`d3_interface_parameterisation.py`](diagnostics/d3_interface_parameterisation.py) | torch | same capacity, two encodings, opposite conclusions |
| [`d4_bottleneck_depth.py`](diagnostics/d4_bottleneck_depth.py) | torch | the artefact that **flattered** our hypothesis |

### d1 — run this one

At ρ=1 the early-exit draft *is* the full model, so speculative decoding degenerates to
ordinary decoding and the speedup must be exactly 1. Our original cost model returned
**1.735**, because it charged drafting and verification but omitted the full-depth pass at
position `t+γ` that supplies the bonus token — a position drafting never touches.

That error survived derivation *and* an adversarial review pass. It did not survive a
three-line check. We recommend the ρ→1 limit as a standing check on any
speculative-decoding cost model.

d1 also settles cache reuse (worse when memory-bound, better when compute-bound — reuse
splits verification across two depths and a memory-bound pass costs a whole weight stream
regardless of position count) and prints the head-cost ceiling table.

### d4 — the dangerous one

At T=12 the discrete interface scored ~0 at *every* width while continuous scored 1.000.
That reads as textbook confirmation of the bandwidth thesis. It is not: a bound binding at
6.91 bits cannot explain failure at 20 bits. The constraint was credit assignment through
twelve stacked quantisations.

The two failure modes separate cleanly: **optimisation failure is monotone in T, bandwidth
failure is monotone in b.** Before reading any discrete-bottleneck failure as a capacity
bound, check that it is monotone in the bottleneck width and absent well above the claimed
threshold.

## Reproducing

```bash
python3 exp3_rnn_is_hmm.py                      # stdlib, ~10s, both parts pass
python3 diagnostics/d1_cost_model_sanity.py     # stdlib, ~1s

python3 exp0_pretrain.py --steps 1500 --vocab 2048 --d 128 --layers 6 --ctx 64 --bs 16
python3 exp1_local.py --rounds 200
python3 exp3b_recall_capacity.py --steps 2500 --d 96 --ns 2 3 4 5 --bits 1 2 3 4 5
python3 exp2_latent_vs_cot.py --n 5 --T 4 --steps 4000 --seeds 2 --sweep-v \
        --bit-list 3 4 5 6 7 8 15
```

CPU-only, 4 threads: `exp0` ~6 min, `exp3b` sweep ~25 min, `exp2` sweep ~15 min.

## Results

Raw logs from the runs reported in the paper are in [`results/`](results/). Files marked
`_ARTEFACT` are the misleading runs, kept deliberately — they are the evidence for the
methodology section, and deleting them would leave the corrections unsupported.

| log | what it shows |
|---|---|
| `exp3_hmm_identity.log` | HMM identity vs brute-force oracle, negative control, both impossibility checks |
| `exp1_local_results.json` | per-layer acceptance, tail energy, bound looseness, speedup surface |
| `exp3b_capacity_sweep.log` | the 19/20 capacity table |
| `exp2_bandwidth_T4.log` | the monotone bandwidth curve, threshold at 6.91 bits |
| `exp3b_800steps_ARTEFACT.log` | undertraining imitating a capacity ceiling |
| `exp3b_onehot_ARTEFACT.log` | one-hot parameterisation failing at ample capacity |
| `exp2_bandwidth_T12_ARTEFACT.log` | the artefact whose naive reading confirmed our thesis |

## Two design rules these experiments taught us

1. **Report violations separately from match rates.** Only a success *below* a claimed
   threshold can refute a lower bound. A symmetric "fraction of cells matching" score
   weights both error directions equally and hides which one you are looking at.
2. **Check that a test can fail.** `exp3`'s original first check compared a nested loop
   against its own list-comprehension rewrite — it could not fail, and it passed. It was
   replaced by brute-force path enumeration, which promptly exposed a base-case error in
   the proof it was supposedly verifying.
