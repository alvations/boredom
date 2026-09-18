# boredom

When bored, code.

**Boredomspiration**:

 - (n): Inspiration from reading something that makes you code something up


## How this works

Every new topic gets:

1. A dated entry in the list below (newest first), `YYYY-MM-DD`.
2. Its own directory, named after the topic, with a `README.md` inside saying
   where the boredomspiration came from.

---

## Topics

### 2026-09-18 — [Three hacks, one complaint](three-hacks/)

- Boredomspiration: three things that annoy me about how we run LLMs, which turned out to be
  the same complaint about depth, time and state
- Speculative decoding should be the model drafting for itself; CoT is serial compute squeezed
  through a 17-bit vocabulary; decoder-only transformers are gated RNNs with unbounded state
- Proofs in [`three-hacks/theory.md`](three-hacks/theory.md), falsification-first experiments on
  Qwen3-0.6B in [`three-hacks/experiments/`](three-hacks/experiments/)

### 2023-12-31 — [The economics of...](the-economics-of/)

- Boredomspiration: the WSJ "The Economics of ..." series on YouTube, playing for hours in the background
- Corpus linguistics on YouTube auto-caption `.srt` files, which are gloriously unparseable
- Subs pulled from https://downsub.com/

### 2023-12-13 — [Paper blitz 2023](paper-blitz-2023/)

- Boredomspiration: https://twitter.com/alvations/status/1734781934998577198
- Live-blitzing ACL + EMNLP 2023 papers, scoped down to "evaluation"

### 2019-01-10 — [Evolutionary NN](evo-nets/)

- Boredomspiration: http://homepages.inf.ed.ac.uk/pkoehn/publications/gann94.pdf

### 2018-12-04 — [An NMT story, till...](nmt-story-till.pdf)

- A bored story, written out as a PDF

### 2018-11-23 — [Super-convergence](super-convergence/)

- Boredomspiration: https://www.fast.ai/2018/07/02/adam-weight-decay/
- adamW code from https://github.com/egg-west/AdamW-pytorch
- Cyclic LR code from https://github.com/ahirner/pytorch-retraining/blob/master/CLR_preview.py

(Most updated code on: https://github.com/alvations/pytorch_challenge/blob/master/train-flower.py)
