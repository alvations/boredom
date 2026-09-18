# Paper

NeurIPS-format write-up of the formal results. **Not compiled** — no LaTeX toolchain in
the container this was written in. Cross-references, duplicate labels and environment
nesting are checked by script; typesetting is not.

```
main.tex              preamble, abstract, bibliography
sections/intro.tex    the three complaints, one axis framing
sections/prelim.tex   definitions: residual-stream model, cost model, media, recurrent form
sections/depth.tex    early exit  (Thm 3.1 exactness, 3.7 acceptance bound, 3.16 dominance)
sections/time.tex     latent reasoning (Thm 4.1 TC0, 4.4 capacity, 4.10 streaming separation)
sections/state.tex    gated RNNs (Thm 5.6 HMM identity, 5.8 no-embedding, 5.12 recall)
sections/unified.tex  the combined object, protocol, limitations
```

To build, drop `neurips_2024.sty` beside `main.tex`, swap the `\usepackage` line flagged
in the preamble, and delete the `geometry` fallback:

```
latexmk -pdf main.tex
```

## Load-bearing results

- **Theorem 3.1** — speculative sampling is exact for *any* draft, with no support
  hypothesis. This is what makes the shortlisted draft head of Corollary 3.19 legitimate.
- **Corollary 3.17** — the unembedding must be streamed on every draft step, capping
  self-drafting speedup at `(γ+1)/(γu+1)`.
- **Corollary 4.5** — a token chain overtakes a fixed-size latent loop at
  `n* = d·b/log₂V`. This reverses the usual bandwidth argument.
- **Theorem 5.8** — no injective linear map carries the HMM forward algorithm into a
  diagonally-gated linear-attention recurrence. Verified numerically by
  `../experiments/exp3_rnn_is_hmm.py`.

## Health warning

Sections 3–5 were each audited adversarially after drafting, and the audits changed two
of the three conclusions. `../theory.md` records what was wrong. No experiment in the
paper has been run.
