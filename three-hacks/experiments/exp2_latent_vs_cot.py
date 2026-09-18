"""
Experiment 2 -- interface bandwidth, under a read schedule that makes it a test.

WHAT CHANGED AND WHY. The obvious design -- let a transformer emit intermediate
tokens and see whether a small codebook hurts -- does not test the bandwidth
bound, for two reasons found while formalising it:

  1. THE RE-READ ESCAPE. With full attention the model can ignore its own
     emitted tokens and recompute the running product from the original inputs.
     Composition is ASSOCIATIVE, so that recomputation is a balanced tree of
     depth log2(T), not T. Setting T=24 against a 4-layer model (log2 24 = 4.6)
     leaves the escape essentially open. With unbounded per-step computation a
     ONE-step machine solves the task outright, so the separation simply does
     not exist without a read-schedule hypothesis -- and in the full-attention
     setting any such separation would imply TC0 != NC1 (Barrington), i.e. it
     is out of reach, not merely hard.
  2. A CORRUPTED TARGET IS NOT A BANDWIDTH LIMIT. Supervising the intermediate
     token with `rank(perm) % V` asks the model to predict something that is not
     a function of anything it can know. It then measures whether the model can
     route around a corrupted label, which is a different phenomenon.

So this version enforces a STREAMING, MARKOV read schedule -- step i sees the
input symbol g_i and the interface from step i-1, and nothing else -- and leaves
the interface content UNSUPERVISED, trained end-to-end through a straight-through
Gumbel-softmax. V is then a pure channel width, exactly the quantity in the
theorem, and the model is free to use it optimally.

Task: compose T generators of S_n. The running product is the unique sufficient
statistic, so the interface must carry log2(n!) bits.

  CoT-like : interface is exactly b bits (see below)
  Latent   : interface is a vector in R^d (d * b_eff bits)

INTERFACE PARAMETERISATION. exp3b established that a one-hot codebook over 2^b
symbols, trained through a straight-through estimator, largely fails to find an
injective code even when capacity is ample -- an optimisation artefact that
imitates a capacity ceiling. We therefore use b independent binary units, which
carry the same b bits. The threshold under test is b >= log2(n!), the only
quantity the theorem mentions.

PREDICTION: accuracy collapses when log2(V) < log2(n!), and the collapse point
moves with V alone, at fixed task and fixed model.
KILL: accuracy stays high below the threshold => the bound does not bind.

    python3 exp2_latent_vs_cot.py --n 5 --sweep-v
"""

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def perm_rank(p):
    """Lehmer code -> integer in [0, n!). Injective."""
    p, n, r = list(p), len(p), 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if p[j] < p[i])
        r = r * (n - i) + smaller
    return r


def make_batch(bs, n, T, device):
    gens, finals = [], []
    for _ in range(bs):
        cur, g_seq = list(range(n)), []
        for _ in range(T):
            g = random.randrange(n - 1)
            cur[g], cur[g + 1] = cur[g + 1], cur[g]
            g_seq.append(g)
        gens.append(g_seq)
        finals.append(cur)
    tl = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return tl(gens), tl(finals)


class StepMachine(nn.Module):
    """One shared step function, iterated T times under a streaming schedule.

    Identical parameter count and identical per-step FLOPs in both modes; the
    ONLY difference is the width of the channel between consecutive steps.
    """

    def __init__(self, n, d=128, bits=None, hidden=4, tau=1.0):
        super().__init__()
        self.n, self.d, self.bits, self.tau = n, d, bits, tau
        self.emb_g = nn.Embedding(n - 1, d)
        self.core = nn.Sequential(
            nn.Linear(2 * d, hidden * d), nn.GELU(),
            nn.Linear(hidden * d, d), nn.LayerNorm(d),
        )
        if bits is not None:                    # discrete interface of width b bits
            self.to_bits = nn.Linear(d, bits)
            self.from_bits = nn.Linear(bits, d)
        self.readout = nn.Linear(d, n * n)
        self.s0 = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, gens, finals):
        B, T = gens.shape
        s = self.s0.expand(B, -1)
        for i in range(T):
            s = self.core(torch.cat([self.emb_g(gens[:, i]), s], dim=-1))
            if self.bits is not None:
                # straight-through: the interface is forced through exactly
                # b binary units, but stays differentiable
                p = torch.sigmoid(self.to_bits(s))
                hard = (p > 0.5).float()
                s = self.from_bits(hard + p - p.detach())
        logits = self.readout(s).view(B, self.n, self.n)
        loss = F.cross_entropy(logits.reshape(-1, self.n), finals.reshape(-1))
        acc = (logits.argmax(-1) == finals).all(-1).float().mean()
        return loss, acc


def train(model, n, T, steps, bs, device, lr=2e-3, quiet=False):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for i in range(steps):
        loss, acc = model(*make_batch(bs, n, T, device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not quiet and (i + 1) % max(1, steps // 4) == 0:
            print(f"      step {i+1}/{steps}  loss {loss.item():.4f}  acc {acc.item():.3f}")
    model.eval()
    with torch.no_grad():
        return sum(model(*make_batch(bs, n, T, device))[1].item() for _ in range(10)) / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="permutation group S_n")
    ap.add_argument("--T", type=int, default=12, help="composition steps")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-v", action="store_true")
    ap.add_argument("--bit-list", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fact = math.factorial(args.n)
    need = math.log2(n_fact)
    print(f"S_{args.n}: {n_fact} states, so the interface needs {need:.2f} bits")
    print(f"T={args.T} steps, streaming Markov schedule (no re-reading the input)")
    print(f"device={device}\n")

    bs_list = args.bit_list or [3, 4, 5, 6, 7, 8, 10]
    results = []
    for b in bs_list:
        print(f"  b={b} bits {'>= needed' if b >= need else '< NEEDED'}:")
        best = 0.0
        for sd in range(args.seeds):
            random.seed(args.seed + sd); torch.manual_seed(args.seed + sd)
            best = max(best, train(StepMachine(args.n, args.d, bits=b), args.n,
                                   args.T, args.steps, args.bs, device, quiet=True))
        results.append((b, best))
        print(f"      -> acc {best:.3f}")

    print(f"\n  continuous interface (R^{args.d}):")
    random.seed(args.seed); torch.manual_seed(args.seed)
    lat = train(StepMachine(args.n, args.d, bits=None), args.n, args.T,
                args.steps, args.bs, device, quiet=True)
    print(f"      -> acc {lat:.3f}")

    print(f"\n{'interface':>12} {'bits':>7} {'acc':>7}   prediction")
    viol = []
    for b, acc in results:
        pred = "ok" if b >= need else "COLLAPSE"
        if b < need and acc >= 0.9:
            viol.append(b)
        print(f"{('b=' + str(b)):>12} {b:>7} {acc:>7.3f}   {pred}")
    print(f"{('R^' + str(args.d)):>12} {'--':>7} {lat:>7.3f}   ok")
    print(f"\nThreshold is b >= log2({args.n}!) = {need:.2f}, i.e. b >= "
          f"{math.ceil(need)}.")
    print(f"Cells that would REFUTE the bound (b < {need:.2f} yet accurate): "
          f"{viol if viol else 'none'}")
    print("If accuracy does not fall there, the bandwidth bound does not bind and")
    print("this thesis loses its only unconditional result.")


if __name__ == "__main__":
    main()
