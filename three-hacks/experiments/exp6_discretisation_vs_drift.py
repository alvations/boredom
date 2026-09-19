"""
Experiment 6 -- is discretisation error correction? (Open Problem 4.14)

The refined thesis 2 says discretising through the vocabulary costs bandwidth
per step. The counter-argument, left open in the paper, is that discretisation
is also ERROR CORRECTION: projecting the state onto a codebook every k steps
stops drift from compounding. Nobody has swept k. This does, twice.

PART A -- exact simulation. Strip the question to its mechanism. n bits are
stored as signs of n coordinates in [-1, 1]. Each of T steps adds Gaussian
noise sigma. Every k steps the state is optionally projected back to the
nearest codeword (sign). Read out at the end.

  Without projection, drift accumulates but never locks in: the readout at the
  end sees sign(1 + sum of T noises).
  With projection every k steps, drift is reset every k steps -- but a sign
  flip inside an interval is LOCKED IN by the projection and can never be
  undone. Errors accumulate across the T/k intervals.

So there is a genuine trade-off with an interior optimum k*(sigma, T), which
is exactly the shape Open Problem 4.14 conjectures. Part A computes the curve
by Monte Carlo and checks it against the analytic approximation

  P(bit survives | project every k) ~= [1 - Phi(-1 / (sigma sqrt k))]^(T/k)

PART B -- learned version. The same task inside a small trained recurrent
network: read n bits, idle for T noisy steps with an optional b-bit projection
every k steps, answer a query. Tests whether the idealised trade-off survives
contact with a learned codebook and a learned dynamics.

PREDICTION: at sigma=0 projection never helps (pure bandwidth cost); as sigma
grows, some finite k beats no projection, and k* shrinks as sigma grows.
DISPROOF: projection never helps at any sigma -- then the error-correction
premise is false and thesis 2's bandwidth argument stands unqualified.

    python3 exp6_discretisation_vs_drift.py            # part A, ~seconds
    python3 exp6_discretisation_vs_drift.py --learned  # part B, minutes
"""

import argparse
import math
import random

import torch


# =========================================================== PART A: exact

def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---- A1: sufficient rate. Bits stored as signs; the codebook is {-1, +1}.

def a1_exact(T, sigma, k):
    """P(one bit correct). No clamp, so these are exact.

    none:   final = 1 + N(0, T sigma^2); correct iff > 0.
    k>=1:   each interval starts at exactly +-1 after projection and flips iff
            N(0, k sigma^2) crosses 1. A flipped bit can flip BACK with the same
            probability -- flips are symmetric, not locked in. Two-state chain
            with flip prob p per interval, M = T/k intervals:
                P(correct) = (1 + (1 - 2p)^M) / 2.
    """
    if sigma == 0:
        return 1.0
    if not k:
        return 1 - Phi(-1 / (sigma * math.sqrt(T)))
    p = Phi(-1 / (sigma * math.sqrt(k)))
    return (1 + (1 - 2 * p) ** (T / k)) / 2


def a1_montecarlo(T, sigma, k, trials, rng):
    s = torch.ones(trials)
    for t in range(1, T + 1):
        s = s + sigma * torch.randn(trials, generator=rng)
        if k and t % k == 0:
            s = torch.where(s >= 0, 1.0, -1.0)
    return (s > 0).float().mean().item()


# ---- A2: rate-limited. An analog value must be recovered to precision eps;
#          the codebook is a grid of L levels, so projection ROUNDS.

def a2_montecarlo(T, sigma, k, L, eps, trials, rng):
    v = torch.rand(trials, generator=rng) * 2 - 1
    s = v.clone()
    step = 2.0 / (L - 1)
    for t in range(1, T + 1):
        s = s + sigma * torch.randn(trials, generator=rng)
        if k and t % k == 0:
            s = (torch.round((s + 1) / step) * step - 1).clamp(-1, 1)
    return ((s - v).abs() < eps).float().mean().item()


def part_a(args):
    rng = torch.Generator().manual_seed(args.seed)
    ks = [1, 2, 4, 8, 16, 32, 0]
    hdr = "".join(f"{('k='+str(k)) if k else 'none':>8}" for k in ks)

    # ------------------------------------------------------------ A1
    print(f"A1 -- SUFFICIENT rate: {args.n} bits as signs, codebook {{-1,+1}}, "
          f"T={args.T}. Exact closed form; Monte Carlo in brackets for one row.")
    print("cells: P(all bits correct). k=0 means never project.\n")
    print(f"{'sigma':>6} {hdr}   best")
    print("-" * (8 + 8 * len(ks) + 8))
    a1_best = {}
    for sigma in args.sigmas:
        row = [a1_exact(args.T, sigma, k) ** args.n for k in ks]
        b = ks[max(range(len(ks)), key=lambda i: row[i])]
        a1_best[sigma] = b
        print(f"{sigma:>6.2f} " + "".join(f"{v:>8.3f}" for v in row)
              + f"   {('k='+str(b)) if b else 'none'}")
    sig = args.sigmas[len(args.sigmas) // 2]
    mc = [a1_montecarlo(args.T, sig, k, args.trials, rng) ** args.n for k in ks]
    print(f"{'MC':>6} " + "".join(f"{v:>8.3f}" for v in mc) + f"   (sigma={sig})")

    # ------------------------------------------------------------ A2
    print(f"\nA2 -- RATE-LIMITED: one analog value in [-1,1], recover to within "
          f"eps={args.eps}, T={args.T}.")
    print("Codebook = grid of L levels; projection rounds to the nearest level.")
    print("Half-spacing above eps means projection alone breaks the task.\n")
    a2_best, a2_rows = {}, {}
    for L in args.levels:
        half = 1.0 / (L - 1)
        print(f"L={L:>3} levels ({math.log2(L):.0f} bits), half-spacing {half:.3f} "
              f"{'> eps: rate INSUFFICIENT' if half > args.eps else '< eps: rate sufficient'}")
        print(f"{'sigma':>6} {hdr}   best")
        for sigma in args.sigmas_a2:
            row = [a2_montecarlo(args.T, sigma, k, L, args.eps, args.trials, rng) for k in ks]
            b = ks[max(range(len(ks)), key=lambda i: row[i])]
            a2_best[(L, sigma)] = b
            a2_rows[(L, sigma)] = row
            print(f"{sigma:>6.3f} " + "".join(f"{v:>8.3f}" for v in row)
                  + f"   {('k='+str(b)) if b else 'none'}")
        print()

    # ------------------------------------------------------------ reading
    # Monte Carlo SE at 5000 trials is ~0.007, so anything within TIE of the
    # top cell is a tie, and reporting a 'winner' there is narrating noise.
    TIE = 0.02
    def verdict(row):
        top = max(row)
        winners = [ks[i] for i, v in enumerate(row) if top - v < TIE]
        if len(winners) == len(ks):
            return "all tied"
        if len(winners) > 1:
            return "tie: " + ",".join(("k=%d" % w) if w else "none" for w in winners)
        w = winners[0]
        return ("k=%d" % w) if w else "none"

    print("Reading (ties within %.2f are reported as ties):" % TIE)
    noisy = [s for s in args.sigmas if s > 0]
    if all(a1_best[s] == 1 for s in noisy):
        print("  A1  rate-sufficient bits: k=1 wins at EVERY noise level, no ties.")
        print("      A flipped bit can flip back, so projection does not lock")
        print("      errors in; the lock-in penalty conjectured in the paper does")
        print("      not exist in this model.")
    interior_clear = []
    for L in args.levels:
        half = 1.0 / (L - 1)
        tag = "INSUFFICIENT" if half > args.eps else "sufficient  "
        picks = []
        for sigma in args.sigmas_a2:
            v = verdict(a2_rows[(L, sigma)])
            picks.append(v)
            if v.startswith("k=") and v not in ("k=1",) and "tie" not in v:
                interior_clear.append((L, sigma, v))
        print(f"  A2  L={L:>3} rate {tag}: " + " | ".join(picks))

    print()
    if interior_clear:
        print(f"  Interior k won CLEARLY in {len(interior_clear)} cell(s): {interior_clear}")
        print("  -> an interior optimum exists in the rate-limited regime; investigate.")
    else:
        print("  No cell has a clear interior-k winner. Wherever a winner is clear it")
        print("  is k=1 or none. The trade-off is real but BINARY, and the switch is")
        print("  in the noise level, not in k: project every step once drift")
        print("  sigma*sqrt(T) exceeds the codebook's rounding error, never before.")
        print("  Open Problem 4.14 asked for an optimal k; the simplest model says")
        print("  the question is whether to discretise at all, decided by rate vs drift.")
    return a1_best


# ======================================================== PART B: learned

def part_b(args):
    import torch.nn as nn
    import torch.nn.functional as F

    class NoisyMemory(nn.Module):
        def __init__(self, n, d, b, k, sigma, T):
            super().__init__()
            self.n, self.d, self.b, self.k, self.sigma, self.T = n, d, b, k, sigma, T
            self.emb_val = nn.Embedding(2, d)
            self.emb_q = nn.Embedding(n, d)
            self.write = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            # Idle dynamics are the IDENTITY plus noise, exactly Part A's model.
            # A first version used tanh(G(s)) with G learned: 64 tanh steps
            # contract the state to zero and kill the gradient, and every cell
            # -- including sigma=0 with no projection -- sat at chance. That
            # was an optimisation failure, not a result about discretisation.
            self.to_bits, self.from_bits = nn.Linear(d, b), nn.Linear(b, d)
            self.read = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2))
            self.s0 = nn.Parameter(torch.zeros(d))

        def project(self, s):
            p = torch.sigmoid(self.to_bits(s))
            return self.from_bits((p > 0.5).float() + p - p.detach())

        def forward(self, vals, q):
            B = vals.shape[0]
            s = self.s0.expand(B, -1)
            for i in range(self.n):                                   # write phase
                s = s + self.write(torch.cat([self.emb_val(vals[:, i]), s], -1))
            for t in range(1, self.T + 1):                            # noisy idle
                s = (s + self.sigma * torch.randn_like(s)).clamp(-3, 3)
                if self.k and t % self.k == 0:
                    s = self.project(s)
            logits = self.read(torch.cat([self.emb_q(q), s], -1))
            tgt = vals.gather(1, q.unsqueeze(1)).squeeze(1)
            return F.cross_entropy(logits, tgt), (logits.argmax(-1) == tgt).float().mean()

    def run(k, sigma):
        random.seed(args.seed); torch.manual_seed(args.seed)
        m = NoisyMemory(args.n, 32, 2 * args.n, k, sigma, args.T)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps)
        for _ in range(args.steps):
            vals = torch.randint(0, 2, (128, args.n)); q = torch.randint(0, args.n, (128,))
            loss, _ = m(vals, q); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            return sum(m(torch.randint(0, 2, (256, args.n)),
                         torch.randint(0, args.n, (256,)))[1].item() for _ in range(8)) / 8

    ks = [1, 4, 16, 0]
    TIE = 0.03
    print(f"\nPART B -- learned: {args.n} bits, d=32, b={2*args.n}-bit projection, "
          f"T={args.T}, {args.steps} steps, identity idle dynamics\n")
    print(f"{'sigma':>6} " + "".join(f"{('k='+str(k)) if k else 'none':>8}" for k in ks) + "   verdict")
    print("-" * 52)
    rows = {}
    for sigma in args.sigmas_b:
        row = [run(k, sigma) for k in ks]
        rows[sigma] = row
        top = max(row)
        winners = [ks[i] for i, v in enumerate(row) if top - v < TIE]
        v = ("all tied" if len(winners) == len(ks) else
             "tie: " + ",".join(("k=%d" % w) if w else "none" for w in winners)
             if len(winners) > 1 else (("k=%d" % winners[0]) if winners[0] else "none"))
        print(f"{sigma:>6.2f} " + "".join(f"{x:>8.3f}" for x in row) + f"   {v}")

    # GUARD: the reference cell is sigma=0, no projection -- the easiest
    # configuration there is. If it cannot be learned, nothing else here
    # carries information and reporting winners would narrate noise.
    ref = rows[args.sigmas_b[0]][ks.index(0)] if args.sigmas_b[0] == 0 else None
    if ref is not None and ref < 0.9:
        print(f"\n  UNDERTRAINED: reference cell (sigma=0, none) scored {ref:.3f}.")
        print("  No comparison above is interpretable. Raise --steps.")
        return
    print("\n  Reading against Part A: with a rate-sufficient codebook, projection")
    print("  should never hurt at sigma=0 and should win once noise is present.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ap.add_argument("--eps", type=float, default=0.05, help="A2 precision target")
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 16, 64, 256])
    ap.add_argument("--sigmas-a2", type=float, nargs="+",
                    default=[0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--sigmas-b", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    part_a(args)
    if args.learned:
        part_b(args)


if __name__ == "__main__":
    main()
