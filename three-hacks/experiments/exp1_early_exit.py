"""
Experiment 1 -- can the model draft for itself?

Measures, for every layer of a <=1B Qwen, the early-exit acceptance rate
    alpha_l = 1 - TV(p_l, p_L)
i.e. exactly the probability that speculative sampling accepts a token drafted
by exiting at layer l. No training, no draft model, forward passes only.

Two corrections relative to the naive version, both of which change conclusions:

  * COST. Every draft step must stream the unembedding W_U to sample from p_l,
    not just the first l blocks. For Qwen3-0.6B W_U is ~26% of all parameters,
    so omitting it inflates speedup by up to 1.39x at rho=0.25 -- largest
    exactly where early exit looks most attractive -- and imposes a hard ceiling
    S <= (gamma+1)/(gamma*u + 1). The kill criterion is max S <= 1.0, so this
    can invert the verdict.
  * BOUND. Softmax is shift-invariant but ||.||_inf is not, so the sup-norm
    bound is slack by up to 2x. We use the spread (max - min) of the logit
    difference instead, which is the shift-invariant quantity.

It also measures how loose the acceptance bound actually is, decomposed into its
two sources of slack, because that decomposition is the real deliverable:

    TV(p_l, p_L)   <=   1 - exp(-spread)             <- softmax lemma
    spread         <=   D_U * ||dN||_2               <- Cauchy-Schwarz on rows

where D_U = max_ij ||w_i - w_j|| is the unembedding diameter. Computing D_U
exactly is O(V^2); we use the valid upper bound D_U <= 2*||W_U||_2,inf, which
only weakens the bound and so keeps it a bound.

If the bound is vacuous (expected) the interesting question is *which* of those
two inequalities threw the information away. My prediction is the second: W_U
only reads a low-dimensional slice of the residual stream, but ||.||_2,inf
assumes its worst row aligns with the tail direction.

Finally it computes the speedup surface
    S(l, gamma) = (1 - alpha^(gamma+1)) / ((1 - alpha) * (gamma * l/L + 1))
under the memory-bound cost model of Proposition 2.4.

KILL CRITERION: if max_l,gamma S <= 1.0 both here and after the early-exit
LoRA of exp1b, self-drafting does not pay for itself and thesis 1 is dead.

    pip install torch transformers
    python3 exp1_early_exit.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The key insight behind speculative decoding is that verification of several tokens",
    "In a transformer, the residual stream accumulates contributions from each block, so",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n",
    "The capital of France is Paris, and the capital of Japan is",
    "To prove that the sum of two even integers is even, let a = 2m and b = 2n. Then",
    "She had never seen the sea before, and when the road finally turned east she",
]


def locate_final_norm(model):
    """Find the final norm applied before the LM head. Qwen puts it at model.model.norm."""
    for path in ("model.norm", "model.model.norm", "transformer.ln_f", "model.final_layernorm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate the final norm; inspect the model and pass it manually")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32",
                    help="float32 keeps the TV numbers trustworthy; bf16 adds noise "
                         "at exactly the scale we are trying to measure")
    ap.add_argument("--max-gamma", type=int, default=8)
    ap.add_argument("--out", default="exp1_results.json")
    ap.add_argument("--shortlist", type=int, default=0,
                    help="if >0, also report the shortlisted-head variant with "
                         "|S|=k: head cost falls to u*k/V, acceptance loses at "
                         "most p_L(S^c), which is measured")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    norm = locate_final_norm(model)
    head = model.get_output_embeddings()
    W_U = head.weight                                  # [V, d]
    wu_row_max = W_U.norm(dim=-1).max().item()         # ||W_U||_2,inf
    D_U = 2.0 * wu_row_max                             # upper bound on diameter

    # head fraction u: share of a single-position forward pass spent on W_U
    n_total = sum(p.numel() for p in model.parameters())
    n_head = W_U.numel()
    if model.config.tie_word_embeddings:
        # tied: the embedding matrix is counted once in parameters() but is
        # streamed by both the embedding lookup and the head; the head still
        # accounts for n_head of the bytes a draft step must read.
        pass
    u = n_head / n_total
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"{args.model}: L={L} layers, d={d}, V={W_U.shape[0]}, "
          f"||W_U||_2,inf={wu_row_max:.3f}, device={device}")
    print(f"head fraction u = {n_head}/{n_total} = {u:.3f}  "
          f"(ceiling S <= (g+1)/(g*u+1), = {(args.max_gamma+1)/(args.max_gamma*u+1):.2f} "
          f"at gamma={args.max_gamma})")

    # accumulators, one slot per layer 1..L
    acc = {k: [0.0] * (L + 1) for k in
           ("alpha", "top1", "tail", "eps_actual", "eps_bound", "dnorm")}
    n_tok = 0
    pL_sum = torch.zeros(W_U.shape[0], device=device)   # for the shortlist study

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").to(device)
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states                          # tuple len L+1: embeddings + each block
        h_final = hs[-1]
        p_L = torch.softmax(head(norm(h_final)).float(), dim=-1)[0]      # [T, V]
        T = p_L.shape[0]
        n_tok += T
        pL_sum += p_L.sum(0)

        for l in range(1, L + 1):
            h_l = hs[l]
            n_l, n_L = norm(h_l), norm(h_final)
            logits_l = head(n_l).float()[0]
            p_l = torch.softmax(logits_l, dim=-1)

            # acceptance == overlap == 1 - TV. This is the quantity in Theorem 2.2.
            alpha = torch.minimum(p_l, p_L).sum(-1)                      # [T]
            top1 = (p_l.argmax(-1) == p_L.argmax(-1)).float()            # greedy-mode accept

            tail = (h_final - h_l)[0].norm(dim=-1)                       # T_l
            dnorm = (n_L - n_l)[0].float().norm(dim=-1)                  # ||dN||_2
            dlogit = logits_l - head(n_L).float()[0]
            # shift-invariant spread, not sup-norm: softmax ignores constants
            eps_act = dlogit.max(-1).values - dlogit.min(-1).values
            eps_bnd = D_U * dnorm

            for key, val in (("alpha", alpha), ("top1", top1), ("tail", tail),
                             ("eps_actual", eps_act), ("eps_bound", eps_bnd),
                             ("dnorm", dnorm)):
                acc[key][l] += val.sum().item()

    for key in acc:
        acc[key] = [v / n_tok for v in acc[key]]

    # ---- report -------------------------------------------------------------
    print(f"\n{n_tok} token positions\n")
    print(f"{'l':>3} {'rho':>5} {'alpha':>7} {'top1':>6} {'T_l':>8} "
          f"{'eps_act':>8} {'eps_bnd':>9} {'bound(a)':>9} {'slack':>7}")
    print("-" * 74)
    rows = []
    for l in range(1, L + 1):
        rho = l / L
        a = acc["alpha"][l]
        # Theorem 2.3, with the two stages separated
        bound_from_actual = math.exp(-acc["eps_actual"][l])
        bound_full = math.exp(-acc["eps_bound"][l])
        slack = acc["eps_bound"][l] / max(acc["eps_actual"][l], 1e-9)
        rows.append(dict(layer=l, rho=rho, alpha=a, top1=acc["top1"][l],
                         tail=acc["tail"][l], eps_actual=acc["eps_actual"][l],
                         eps_bound=acc["eps_bound"][l],
                         bound_from_actual=bound_from_actual, bound_full=bound_full))
        print(f"{l:>3} {rho:>5.2f} {a:>7.4f} {acc['top1'][l]:>6.3f} "
              f"{acc['tail'][l]:>8.2f} {acc['eps_actual'][l]:>8.3f} "
              f"{acc['eps_bound'][l]:>9.3f} {bound_from_actual:>9.2e} {slack:>7.1f}x")

    print("\n'bound(a)' is the bound fed the *measured* spread -- pure softmax slack.")
    print("'slack' is eps_bound/eps_actual -- what the D_U Cauchy-Schwarz step gives away.")

    # ---- speedup surface, Proposition 2.4 ------------------------------------
    print(f"\nSpeedup S(l, gamma), memory-bound (w=1), head cost included:")
    print(f"  C_self = gamma*[(1-u)*rho + u] + [(1-u)*(1-rho) + u]\n")
    gammas = list(range(1, args.max_gamma + 1))
    print("  l  rho  " + "".join(f"  g={g:<4}" for g in gammas))
    best = (0.0, None, None)
    for l in range(1, L + 1):
        rho, a = l / L, min(acc["alpha"][l], 1 - 1e-9)
        cells = []
        for g in gammas:
            exp_tok = (1 - a ** (g + 1)) / (1 - a)
            # memory-bound: D = w + (1-w)*gamma = 1 at w=1
            cost = g * ((1 - u) * rho + u) + ((1 - u) * (1 - rho) + u)
            s = exp_tok / cost
            cells.append(s)
            if s > best[0]:
                best = (s, l, g)
        print(f"{l:>3} {rho:>4.2f}  " + "".join(f"{c:>7.3f}" for c in cells))

    # ---- shortlisted draft head ---------------------------------------------
    if args.shortlist:
        k = min(args.shortlist, pL_sum.numel())
        topk = pL_sum.topk(k).indices
        # mean_x p_L(S) under the same token distribution that produced pL_sum
        mass_in = pL_sum[topk].sum().item() / n_tok
        leak = 1.0 - mass_in
        u_s = u * k / W_U.shape[0]
        print(f"\nShortlisted head, |S|={k}: head fraction {u:.3f} -> {u_s:.4f}, "
              f"acceptance loses at most p_L(S^c) = {leak:.4f}")
        best_s = (0.0, None, None)
        for l in range(1, L + 1):
            rho = l / L
            a = min(max(acc["alpha"][l] - leak, 1e-9), 1 - 1e-9)
            for g in gammas:
                exp_tok = (1 - a ** (g + 1)) / (1 - a)
                cost = g * ((1 - u_s) * rho + u_s) + ((1 - u_s) * (1 - rho) + u_s)
                if exp_tok / cost > best_s[0]:
                    best_s = (exp_tok / cost, l, g)
        print(f"  best S = {best_s[0]:.3f} at layer {best_s[1]}, gamma={best_s[2]} "
              f"(vs {best[0]:.3f} unshortlisted)")

    print(f"\nBest S = {best[0]:.3f} at layer {best[1]} (rho={best[1]/L:.2f}), gamma={best[2]}")
    if best[0] <= 1.0:
        print("=> Untuned self-drafting does NOT pay for itself. Proceed to exp1b")
        print("   (early-exit LoRA); if it still fails there, thesis 1 is dead.")
    else:
        print(f"=> Untuned self-drafting already pays for itself: {best[0]:.2f}x")
        print("   Next: confirm on the wall clock, then compare against Qwen3-0.6B")
        print("   drafting for a larger target.")

    with open(args.out, "w") as f:
        json.dump(dict(model=args.model, L=L, d=d, wu_row_max=wu_row_max,
                       head_fraction=u, D_U=D_U,
                       n_tok=n_tok, best_speedup=best[0], best_layer=best[1],
                       best_gamma=best[2], rows=rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
