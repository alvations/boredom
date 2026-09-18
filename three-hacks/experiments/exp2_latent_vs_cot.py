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

  CoT-like : interface is one symbol from an alphabet of size V (log2 V bits)
  Latent   : interface is a vector in R^d (d * b_eff bits)

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

    def __init__(self, n, d=128, V=None, hidden=4, tau=1.0):
        super().__init__()
        self.n, self.d, self.V, self.tau = n, d, V, tau
        self.emb_g = nn.Embedding(n - 1, d)
        self.core = nn.Sequential(
            nn.Linear(2 * d, hidden * d), nn.GELU(),
            nn.Linear(hidden * d, d), nn.LayerNorm(d),
        )
        if V is not None:                       # discrete interface of width log2 V
            self.to_sym = nn.Linear(d, V)
            self.emb_sym = nn.Embedding(V, d)
        self.readout = nn.Linear(d, n * n)
        self.s0 = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, gens, finals):
        B, T = gens.shape
        s = self.s0.expand(B, -1)
        for i in range(T):
            s = self.core(torch.cat([self.emb_g(gens[:, i]), s], dim=-1))
            if self.V is not None:
                # straight-through Gumbel-softmax: the interface is forced
                # through exactly one of V symbols, but stays differentiable
                logits = self.to_sym(s)
                onehot = F.gumbel_softmax(logits, tau=self.tau, hard=True)
                s = onehot @ self.emb_sym.weight
        logits = self.readout(s).view(B, self.n, self.n)
        loss = F.cross_entropy(logits.reshape(-1, self.n), finals.reshape(-1))
        acc = (logits.argmax(-1) == finals).all(-1).float().mean()
        return loss, acc


def train(model, n, T, steps, bs, device, lr=1e-3, quiet=False):
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
    ap.add_argument("--T", type=int, default=32, help="composition steps")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-v", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fact = math.factorial(args.n)
    need = math.log2(n_fact)
    print(f"S_{args.n}: {n_fact} states, so the interface needs {need:.2f} bits")
    print(f"T={args.T} steps, streaming Markov schedule (no re-reading the input)")
    print(f"device={device}\n")

    vs = [2, 4, 8, 16, 32, 64, 128, 256] if args.sweep_v else [n_fact]
    results = []
    for V in vs:
        print(f"  V={V} ({math.log2(V):.2f} bits) "
              f"{'>= needed' if math.log2(V) >= need else '< NEEDED'}:")
        random.seed(args.seed); torch.manual_seed(args.seed)
        acc = train(StepMachine(args.n, args.d, V=V), args.n, args.T,
                    args.steps, args.bs, device, quiet=True)
        results.append((V, acc))
        print(f"      -> acc {acc:.3f}")

    print(f"\n  continuous interface (R^{args.d}):")
    random.seed(args.seed); torch.manual_seed(args.seed)
    lat = train(StepMachine(args.n, args.d, V=None), args.n, args.T,
                args.steps, args.bs, device, quiet=True)
    print(f"      -> acc {lat:.3f}")

    print(f"\n{'interface':>12} {'bits':>7} {'acc':>7}   prediction")
    for V, acc in results:
        pred = "ok" if math.log2(V) >= need else "COLLAPSE"
        print(f"{('V=' + str(V)):>12} {math.log2(V):>7.2f} {acc:>7.3f}   {pred}")
    print(f"{('R^' + str(args.d)):>12} {'--':>7} {lat:>7.3f}   ok")
    print(f"\nThreshold sits between V={2**math.floor(need)} and V={2**math.ceil(need)}.")
    print("If accuracy does not fall there, the bandwidth bound does not bind and")
    print("this thesis loses its only unconditional result.")


if __name__ == "__main__":
    main()
