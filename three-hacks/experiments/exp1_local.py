"""
Experiment 1 (local model) -- measuring the depth-axis theorems on a real
trained transformer.

huggingface.co is unreachable from this container (the agent proxy denies
CONNECT), so this runs against the model trained by exp0_pretrain.py, which
satisfies Definition 2.1 exactly: pre-norm residual blocks, RMSNorm, an
unembedding. The theorems are architecture-general and none of them mention
Qwen; what changes is scale, and where scale matters we say so.

Four measurements:

  A. EXACTNESS (Thm 3.1). Speculative sampling with an early-exit draft must
     return EXACTLY the target distribution. Tested by drawing many samples and
     comparing to the sampling-noise floor -- the TV distance between p_L and an
     equal number of direct draws from p_L. Comparing to zero would be wrong:
     any finite sample has TV > 0.
  B. ACCEPTANCE PROFILE (Thm 3.7). alpha_l, tail energy T_l, the bound, and its
     looseness split into the Cauchy-Schwarz and softmax stages.
  C. YIELD (Prop 3.11). Real speculative decoding runs, comparing measured E[Y]
     against the assumption-free identity 1 + sum_j P(A_j) and against the
     geometric approximation at the mean acceptance rate. This tests the
     correlated-acceptance remark directly.
  D. SPEEDUP (Thm 3.16, Cor 3.17), with the head cost included, reported both
     at this model's u and at Qwen3-0.6B's u = 0.26.

    python3 exp1_local.py
"""

import argparse
import json
import math
import os

import torch

from exp0_pretrain import TinyLM, load_corpus, CKPT


def load_model(device, path=CKPT):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TinyLM(ck["V"], ck["d"], ck["L"], ctx=ck["ctx"]).to(device).eval()
    m.load_state_dict(ck["state"])
    return m, ck


@torch.no_grad()
def layer_dists(model, idx):
    """p_l for every layer l, plus the residual stream, at every position."""
    hs = model.hidden_states(idx)
    outs = []
    for h in hs:
        outs.append(torch.softmax(model.head(model.norm(h)).float(), dim=-1))
    return outs, hs


# ------------------------------------------------------- A. exactness

@torch.no_grad()
def test_exactness(model, idx, layer, n_samples, device, rng):
    ps, _ = layer_dists(model, idx)
    p_L = ps[-1][0, -1]                      # target at the last position
    q = ps[layer][0, -1]                     # early-exit draft

    # speculative step, vectorised over n_samples
    y = torch.multinomial(q, n_samples, replacement=True)
    u = torch.rand(n_samples, generator=rng, device=device)
    accept = u < (p_L[y] / q[y].clamp_min(1e-30)).clamp(max=1.0)
    n_rej = int((~accept).sum())
    out = y.clone()
    if n_rej > 0:
        resid = (p_L - q).clamp_min(0)
        resid = resid / resid.sum().clamp_min(1e-30)
        out[~accept] = torch.multinomial(resid, n_rej, replacement=True)

    emp_spec = torch.bincount(out, minlength=p_L.numel()).float() / n_samples
    direct = torch.multinomial(p_L, n_samples, replacement=True)
    emp_direct = torch.bincount(direct, minlength=p_L.numel()).float() / n_samples

    tv_spec = 0.5 * (emp_spec - p_L).abs().sum().item()
    tv_direct = 0.5 * (emp_direct - p_L).abs().sum().item()
    alpha_pred = torch.minimum(p_L, q).sum().item()
    alpha_obs = accept.float().mean().item()
    return tv_spec, tv_direct, alpha_pred, alpha_obs


# --------------------------------------------- C. real speculative decoding

@torch.no_grad()
def run_decoding(model, prompt, layer, gamma, n_rounds, device, rng):
    """Actual speculative decoding with an early-exit draft. Returns the
    per-round yields and the run of acceptance indicators."""
    ctx = model.ctx
    seq = prompt.clone()
    yields, prefix_hits = [], [0] * (gamma + 1)
    for _ in range(n_rounds):
        cur = seq[:, -ctx:]
        drafts = []
        work = cur
        for _ in range(gamma):                       # sequential drafting
            hs = model.hidden_states(work[:, -ctx:])
            q = torch.softmax(model.head(model.norm(hs[layer]))[0, -1].float(), -1)
            tok = torch.multinomial(q, 1, generator=rng)
            drafts.append((tok.item(), q))
            work = torch.cat([work, tok.view(1, 1)], dim=1)

        # one batched verification pass over the drafted positions
        wt = work[:, -ctx:]
        ps, _ = layer_dists(model, wt)
        p_full = ps[-1][0]
        Lw = wt.shape[1]
        n_acc = 0
        bonus = None
        for i, (tok, q) in enumerate(drafts):
            p_i = p_full[Lw - gamma - 1 + i]
            r = torch.rand(1, generator=rng, device=device).item()
            if r < min(1.0, (p_i[tok] / q[tok].clamp_min(1e-30)).item()):
                n_acc += 1
            else:
                resid = (p_i - q).clamp_min(0)
                resid = resid / resid.sum().clamp_min(1e-30)
                bonus = torch.multinomial(resid, 1, generator=rng)
                break
        if bonus is None:                       # all gamma accepted
            bonus = torch.multinomial(p_full[Lw - 1], 1, generator=rng)
        for j in range(n_acc + 1):
            prefix_hits[j] += 1                 # P(A_j) counters, j=0 is trivial
        yields.append(n_acc + 1)                # n_acc accepted + 1 bonus
        accepted = torch.tensor([[t for t, _ in drafts[:n_acc]]],
                                dtype=torch.long, device=device)
        seq = torch.cat([seq, accepted, bonus.view(1, 1)], dim=1)
        if seq.shape[1] > ctx * 4:
            seq = seq[:, -ctx:]
    return yields, prefix_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400000)
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--max-gamma", type=int, default=8)
    ap.add_argument("--out", default="exp1_local_results.json")
    ap.add_argument("--ckpt", default=CKPT,
                    help="checkpoint to measure; exp5 writes an early-exit-trained one")
    args = ap.parse_args()

    device = "cpu"
    torch.manual_seed(0)
    rng = torch.Generator(device=device); rng.manual_seed(0)
    model, ck = load_model(device, args.ckpt)
    print(f"checkpoint: {args.ckpt}")
    L, V, d = model.L, model.V, model.d

    n_all = sum(p.numel() for p in model.parameters())
    u = model.head.weight.numel() / n_all
    W_U = model.head.weight
    D_U_bound = 2.0 * W_U.norm(dim=-1).max().item()
    g_inf = max(b.n1.g.abs().max().item() for b in model.blocks)
    g_inf = max(g_inf, model.norm.g.abs().max().item())
    print(f"model: L={L} d={d} V={V} params={n_all/1e6:.2f}M  u={u:.3f}")
    print(f"||W_U||-diameter bound D_U <= {D_U_bound:.2f}, ||g||_inf={g_inf:.2f}\n")

    # frozen validation split, saved at training time -- NOT re-globbed, so
    # editing these scripts cannot change the evaluation data
    val = ck["val"]
    idx = val[: model.ctx].unsqueeze(0).to(device)

    # ---------------- A. exactness -----------------------------------------
    print("A. EXACTNESS (Theorem 3.1) -- speculative sampling must return p_L exactly")
    print(f"{'layer':>6} {'TV(spec,p_L)':>13} {'TV(direct,p_L)':>15} "
          f"{'ratio':>7} {'alpha_pred':>11} {'alpha_obs':>10}")
    exact_rows = []
    for layer in sorted({1, L // 2, L - 1}):
        tv_s, tv_d, a_p, a_o = test_exactness(model, idx, layer, args.samples,
                                              device, rng)
        ratio = tv_s / max(tv_d, 1e-12)
        exact_rows.append(dict(layer=layer, tv_spec=tv_s, tv_direct=tv_d,
                               ratio=ratio, alpha_pred=a_p, alpha_obs=a_o))
        print(f"{layer:>6} {tv_s:>13.5f} {tv_d:>15.5f} {ratio:>7.3f} "
              f"{a_p:>11.4f} {a_o:>10.4f}")
    print("  ratio ~ 1.0 => indistinguishable from direct sampling, i.e. exact.")
    print("  alpha_obs should match alpha_pred = 1 - TV(p_l, p_L).\n")

    # ---------------- B. acceptance profile --------------------------------
    print("B. ACCEPTANCE PROFILE (Theorem 3.7)")
    n_pos = 0
    acc = {k: [0.0] * (L + 1) for k in ("alpha", "top1", "tail", "spread",
                                        "spread_bnd", "dnorm", "hnorm")}
    n_chunks = max(1, args.positions // 8)
    for c in range(n_chunks):
        s = c * model.ctx
        if s + model.ctx > len(val):
            break
        chunk = val[s:s + model.ctx].unsqueeze(0).to(device)
        ps, hs = layer_dists(model, chunk)
        p_L = ps[-1][0]
        h_L = hs[-1][0]
        nL = model.norm(h_L)
        logits_L = model.head(nL).float()
        T = p_L.shape[0]; n_pos += T
        for l in range(1, L + 1):
            p_l = ps[l][0]
            h_l = hs[l][0]
            n_l = model.norm(h_l)
            logits_l = model.head(n_l).float()
            dl = logits_l - logits_L
            spread = (dl.max(-1).values - dl.min(-1).values)
            dn = (n_l - nL).norm(dim=-1)
            acc["alpha"][l] += torch.minimum(p_l, p_L).sum(-1).sum().item()
            acc["top1"][l] += (p_l.argmax(-1) == p_L.argmax(-1)).float().sum().item()
            acc["tail"][l] += (h_L - h_l).norm(dim=-1).sum().item()
            acc["spread"][l] += spread.sum().item()
            acc["spread_bnd"][l] += (D_U_bound * dn).sum().item()
            acc["dnorm"][l] += dn.sum().item()
            acc["hnorm"][l] += (h_l.norm(dim=-1) + h_L.norm(dim=-1)).sum().item()
    for k in acc:
        acc[k] = [v / max(n_pos, 1) for v in acc[k]]

    print(f"{'l':>3} {'rho':>5} {'alpha':>7} {'top1':>6} {'T_l':>8} {'spread':>8} "
          f"{'bound_sp':>9} {'e^-spr':>9} {'CS slack':>9}")
    rows = []
    for l in range(1, L + 1):
        sp, spb = acc["spread"][l], acc["spread_bnd"][l]
        rows.append(dict(layer=l, rho=l / L, alpha=acc["alpha"][l],
                         top1=acc["top1"][l], tail=acc["tail"][l],
                         spread=sp, spread_bound=spb,
                         bound_from_measured=math.exp(-sp)))
        print(f"{l:>3} {l/L:>5.2f} {acc['alpha'][l]:>7.4f} {acc['top1'][l]:>6.3f} "
              f"{acc['tail'][l]:>8.2f} {sp:>8.2f} {spb:>9.1f} "
              f"{math.exp(-min(sp,700)):>9.2e} {spb/max(sp,1e-9):>8.1f}x")
    print("  'e^-spr' is the bound fed the MEASURED spread: pure softmax slack.")
    print("  'CS slack' is bound_spread/spread: what the D_U step alone gives away.\n")

    # ---------------- C. yield ---------------------------------------------
    best_layer = max(range(1, L), key=lambda l: acc["alpha"][l] / (l / L))
    print(f"C. YIELD (Prop 3.11) -- real decoding, layer {best_layer}, "
          f"gamma={args.gamma}, {args.rounds} rounds")
    prompt = val[:32].unsqueeze(0).to(device)
    yields, hits = run_decoding(model, prompt, best_layer, args.gamma,
                                args.rounds, device, rng)
    measured = sum(yields) / len(yields)
    pA = [h / args.rounds for h in hits[1:]]
    exact_pred = 1 + sum(pA)
    a_bar = acc["alpha"][best_layer]
    geom = (1 - a_bar ** (args.gamma + 1)) / (1 - a_bar)
    print(f"  measured E[Y]          = {measured:.3f}")
    print(f"  identity 1 + sum P(A_j) = {exact_pred:.3f}   (CIRCULAR: same run, "
          f"bookkeeping check only)")
    print(f"  geometric at alpha_bar  = {geom:.3f}   (alpha_bar={a_bar:.3f})")
    print(f"  P(A_j) = {['%.3f' % x for x in pA]}")
    print(f"  independence would give {['%.3f' % (a_bar**(j+1)) for j in range(args.gamma)]}\n")

    # ---------------- D. speedup -------------------------------------------
    # CORRECTED COST (see paper Prop 3.15 erratum). Drafting computes layers
    # 1..l at positions t..t+gamma-1. Verification needs FULL depth at position
    # t+gamma for the bonus token -- a position drafting never touched. Two
    # verification strategies:
    #   reuse:    skip blocks 1..l at the gamma drafted positions, but that
    #             splits verification into two passes at different depths
    #   no-reuse: one batched full-depth pass over all gamma+1 positions
    # In the memory-bound regime a pass costs a weight stream regardless of how
    # many positions it covers, so splitting pays twice and reuse LOSES.
    def cost_reuse(rho, g, w, uu):
        Dg, Dg1 = w + (1 - w) * g, w + (1 - w) * (g + 1)
        return (g * ((1 - uu) * rho + uu)                     # drafting
                + (1 - uu) * (1 - rho) * Dg                   # upper blocks, drafted pos
                + (1 - uu)                                    # full depth, bonus position
                + uu * Dg1)                                   # head everywhere
    def cost_noreuse(rho, g, w, uu):
        Dg1 = w + (1 - w) * (g + 1)
        return g * ((1 - uu) * rho + uu) + Dg1
    def best_S(w, uu, reuse):
        f = cost_reuse if reuse else cost_noreuse
        best = (0.0, None, None)
        for l in range(1, L + 1):
            a = min(max(acc["alpha"][l], 1e-9), 1 - 1e-9)
            for gmm in gammas:
                yld = (1 - a ** (gmm + 1)) / (1 - a)
                s_ = yld / f(l / L, gmm, w, uu)
                if s_ > best[0]:
                    best = (s_, l, gmm)
        return best

    gammas = list(range(1, args.max_gamma + 1))
    print("D. SPEEDUP (Thm 3.16, Cor 3.17) with the CORRECTED cost model")
    print("   sanity: at rho=1 drafting IS ordinary decoding, so S must be 1.000")
    a_top = 1 - 1e-9
    for w in (1.0, 0.0):
        c = min(cost_reuse(1.0, 1, w, u), cost_noreuse(1.0, 1, w, u))
        print(f"     w={w}: S(rho=1,gamma=1) = {2.0/c:.3f}")
    print()
    print(f"{'regime':>10} {'u':>6} {'strategy':>10} {'best S':>8} {'layer':>6} {'gamma':>6}")
    speed_rows = []
    for w, wname in ((1.0, "mem-bound"), (0.0, "compute")):
        for uu, uname in ((u, f"{u:.3f}"), (0.26, "0.260")):
            for reuse in (False, True):
                b = best_S(w, uu, reuse)
                speed_rows.append(dict(w=w, u=uu, reuse=reuse, S=b[0],
                                       layer=b[1], gamma=b[2]))
                print(f"{wname:>10} {uname:>6} {'reuse' if reuse else 'no-reuse':>10} "
                      f"{b[0]:>8.3f} {b[1]:>6} {b[2]:>6}")
    best_overall = max(r["S"] for r in speed_rows if r["w"] == 1.0)
    print(f"\n  KILL CRITERION (max S <= 1.0, memory-bound): "
          f"best S = {best_overall:.3f} -> "
          f"{'THESIS 1 FAILS on this model' if best_overall <= 1.0 else 'survives'}")
    print("  Note: untuned early exit, no early-exit auxiliary loss (Rmk 3.22).")

    json.dump(dict(L=L, d=d, V=V, u=u, D_U_bound=D_U_bound,
                   exactness=exact_rows, profile=rows,
                   yield_measured=measured, yield_identity=exact_pred,
                   yield_geometric=geom, pA=pA, best_layer=best_layer,
                   speedup=speed_rows),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
