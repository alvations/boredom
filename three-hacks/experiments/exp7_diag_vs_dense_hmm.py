"""
Experiment 7 -- does Theorem 5.8 have practical bite?

Theorem 5.8 proves no injective linear map carries the HMM forward algorithm
into a diagonally-gated linear-attention recurrence. That is an EXACT
embedding result. It says nothing about approximation, and if a diagonal-gated
recurrence can approximate HMM predictions to within noise, the theorem is
true and irrelevant. This experiment asks that question directly.

Data: sequences from a random HMM (m hidden states, |O| symbols). The
Bayes-optimal next-symbol predictor is the forward algorithm, so the optimal
NLL is computable exactly and every model is scored by its EXCESS over it.

Models, all input-controlled affine recurrences in the sense of Prop 5.9,
state normalised each step (a scalar rescaling; does not affect mixing):

  diag-1   s_t = a(o_t) (*) s_{t-1} + c(o_t)        one layer, DIAGONAL action
  dense-1  s_t = M(o_t)    s_{t-1} + c(o_t)         one layer, DENSE action
  diag-L   L diagonal layers with an MLP between    mixing only via depth

PREDICTION (Thm 5.8 with bite): dense-1 reaches ~0 excess NLL; diag-1 has an
irreducible gap at matched state size; diag-L closes it as L grows.
DISPROOF: diag-1 reaches ~0 excess at matched state size. Then exact
non-embeddability is real but practically empty, and thesis 3's negative
result should be reported as such.

    python3 exp7_diag_vs_dense_hmm.py
"""

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ HMM data

def make_hmm(m, n_obs, rng, kind="peaked", peaked=3.0, delta=0.1, q=0.5):
    """Row-stochastic A, B.

    peaked: random, sharply peaked rows. States are quickly identifiable from
            recent symbols, so the belief collapses and little MIXING is ever
            needed -- a diagonal recurrence can do well here without
            contradicting Thm 5.8, which is about exact embedding.
    cyclic: A = (1-delta) * cyclic permutation + delta * uniform, and each
            state prefers its own symbol only mildly (prob q). The belief
            ROTATES through the states; the transition operator has complex
            eigenvalues, which a real diagonal gate cannot represent. This is
            the case Thm 5.8's proof (b) singles out, and the honest stress
            test of whether the theorem has practical bite.
    """
    if kind == "peaked":
        def rows(r, c):
            M = torch.rand(r, c, generator=rng) ** peaked + 1e-3
            return M / M.sum(1, keepdim=True)
        return rows(m, m), rows(m, n_obs), torch.full((m,), 1.0 / m)
    P = torch.zeros(m, m)
    for i in range(m):
        P[i, (i + 1) % m] = 1.0
    A = (1 - delta) * P + delta / m
    B = torch.full((m, n_obs), (1 - q) / max(n_obs - 1, 1))
    for i in range(m):
        B[i, i % n_obs] = q
    B = B / B.sum(1, keepdim=True)
    return A, B, torch.full((m,), 1.0 / m)


@torch.no_grad()
def belief_entropy(A, B, pi, obs):
    """Mean entropy (nats) of the normalised forward posterior. Near 0 means the
    belief collapses -- the task needs no mixing. Near log(m) means the hidden
    state is genuinely uncertain and must be TRACKED through transitions."""
    n, T = obs.shape
    alpha = pi.expand(n, -1) * B[:, obs[:, 0]].T
    alpha = alpha / alpha.sum(1, keepdim=True)
    H = 0.0
    for t in range(1, T):
        alpha = (alpha @ A) * B[:, obs[:, t]].T
        alpha = alpha / alpha.sum(1, keepdim=True)
        H += -(alpha * torch.log(alpha + 1e-12)).sum(1).mean().item()
    return H / (T - 1)


def sample_hmm(A, B, pi, T, n_seq, rng):
    m = A.shape[0]
    z = torch.multinomial(pi.expand(n_seq, -1), 1, generator=rng).squeeze(1)
    obs = torch.empty(n_seq, T, dtype=torch.long)
    for t in range(T):
        obs[:, t] = torch.multinomial(B[z], 1, generator=rng).squeeze(1)
        z = torch.multinomial(A[z], 1, generator=rng).squeeze(1)
    return obs


@torch.no_grad()
def optimal_nll(A, B, pi, obs):
    """Exact forward-algorithm next-symbol NLL, averaged over positions 1..T-1."""
    n, T = obs.shape
    alpha = pi.expand(n, -1) * B[:, obs[:, 0]].T
    alpha = alpha / alpha.sum(1, keepdim=True)
    tot = 0.0
    for t in range(1, T):
        pred = (alpha @ A) @ B                       # P(o_t | o_<t)
        tot += -torch.log(pred.gather(1, obs[:, t:t+1])).sum().item()
        alpha = (alpha @ A) * B[:, obs[:, t]].T
        alpha = alpha / alpha.sum(1, keepdim=True)
    return tot / (n * (T - 1))


# -------------------------------------------------------------------- models

class AffineLayer(nn.Module):
    """One input-controlled affine recurrence: s_t = M(o_t) s_{t-1} + c(o_t)."""

    def __init__(self, n_obs, d, dense):
        super().__init__()
        self.dense, self.d = dense, d
        if dense:
            self.M = nn.Parameter(torch.eye(d).repeat(n_obs, 1, 1)
                                  + 0.05 * torch.randn(n_obs, d, d))
        else:
            self.a = nn.Parameter(torch.ones(n_obs, d) + 0.05 * torch.randn(n_obs, d))
        self.c = nn.Embedding(n_obs, d)
        self.s0 = nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, obs, drive=None):
        B, T = obs.shape
        s = self.s0.expand(B, -1)
        out = []
        for t in range(T):
            o = obs[:, t]
            inp = self.c(o) if drive is None else self.c(o) + drive[:, t]
            if self.dense:
                s = torch.bmm(self.M[o], s.unsqueeze(-1)).squeeze(-1) + inp
            else:
                s = self.a[o] * s + inp
            s = s / (s.norm(dim=-1, keepdim=True) + 1e-6)     # scalar rescale
            out.append(s)
        return torch.stack(out, 1)


class Stack(nn.Module):
    def __init__(self, n_obs, d, layers, dense):
        super().__init__()
        self.layers = nn.ModuleList([AffineLayer(n_obs, d, dense) for _ in range(layers)])
        self.mix = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d))
                                  for _ in range(layers - 1)])
        self.head = nn.Linear(d, n_obs)

    def forward(self, obs):
        h = self.layers[0](obs)
        for lyr, mx in zip(self.layers[1:], self.mix):
            h = lyr(obs, drive=mx(h))
        return self.head(h)                            # logits for o_{t+1}

    def nll(self, obs):
        logits = self(obs)[:, :-1]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               obs[:, 1:].reshape(-1))


def train(model, data, steps, bs, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = data.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        loss = model.nll(data[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()


@torch.no_grad()
def evaluate(model, data):
    return model.nll(data).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4, help="hidden states")
    ap.add_argument("--obs", type=int, default=3, help="observation symbols")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="state sizes to try; default: m and 4m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmm", default="peaked", choices=("peaked", "cyclic"))
    ap.add_argument("--delta", type=float, default=0.1, help="cyclic: transition noise")
    ap.add_argument("--q", type=float, default=0.5, help="cyclic: own-symbol emission prob")
    args = ap.parse_args()

    rng = torch.Generator().manual_seed(args.seed)
    A, B, pi = make_hmm(args.m, args.obs, rng, kind=args.hmm, delta=args.delta, q=args.q)
    train_d = sample_hmm(A, B, pi, args.T, 4000, rng)
    test_d = sample_hmm(A, B, pi, args.T, 2000, rng)
    opt_nll = optimal_nll(A, B, pi, test_d)
    uniform = math.log(args.obs)
    H = belief_entropy(A, B, pi, test_d)
    print(f"HMM ({args.hmm}): m={args.m} states, {args.obs} symbols, T={args.T}")
    print(f"optimal (forward algorithm) NLL = {opt_nll:.4f}   "
          f"uniform = {uniform:.4f}   headroom = {uniform - opt_nll:.4f}")
    print(f"mean belief entropy = {H:.3f} nats of max {math.log(args.m):.3f}  "
          f"({'belief collapses: little mixing needed' if H < 0.3 * math.log(args.m) else 'hidden state must be tracked: mixing needed'})\n")

    dims = args.dims or [args.m, 4 * args.m]
    configs = [("dense-1", 1, True), ("diag-1", 1, False),
               ("diag-2", 2, False), ("diag-3", 3, False)]
    print(f"{'model':>9} {'d':>4} {'test NLL':>9} {'excess':>8} {'% headroom':>11}")
    print("-" * 48)
    results = {}
    for d in dims:
        for name, L, dense in configs:
            random.seed(args.seed); torch.manual_seed(args.seed)
            model = Stack(args.obs, d, L, dense)
            train(model, train_d, args.steps, args.bs)
            nll = evaluate(model, test_d)
            ex = nll - opt_nll
            results[(name, d)] = ex
            print(f"{name:>9} {d:>4} {nll:>9.4f} {ex:>8.4f} "
                  f"{100 * ex / (uniform - opt_nll):>10.1f}%")
        print()

    d0 = dims[0]
    gap_diag = results[("diag-1", d0)]
    gap_dense = results[("dense-1", d0)]
    print(f"At matched state size d={d0}:")
    print(f"  dense-1 excess {gap_dense:.4f}   diag-1 excess {gap_diag:.4f}")
    # GUARD. If the belief collapses, the task never needs mixing and NO result
    # here bears on whether Thm 5.8 has bite -- a diagonal recurrence matching
    # a dense one on such an HMM is expected and uninformative. Only a
    # high-entropy HMM, where the hidden state must be tracked through the
    # transition structure, can distinguish the two. The peaked random HMM
    # produced a confident 'DISPROOF' before this guard existed.
    needs_mixing = H >= 0.3 * math.log(args.m)
    if not needs_mixing:
        print(f"  belief entropy {H:.3f} is low: the task needs no mixing, so this")
        print("  HMM cannot test the theorem's bite either way. Run --hmm cyclic.")
    elif gap_diag < 0.02 and gap_dense < 0.02:
        print("  => DISPROOF of practical bite: on an HMM that REQUIRES mixing, a")
        print("     single diagonal layer approximates the optimum as well as a")
        print("     dense one. Thm 5.8 is exact-only and practically empty.")
    elif gap_dense < 0.02 and gap_diag > 3 * max(gap_dense, 0.005):
        print("  => Thm 5.8 has bite: dense reaches the optimum, diagonal cannot,")
        print("     on an HMM whose belief must be tracked through transitions.")
        deeper = [results[(f"diag-{L}", d0)] for L in (2, 3)]
        print(f"     diag-2 {deeper[0]:.4f}   diag-3 {deeper[1]:.4f}  -- "
              f"{'depth closes the gap' if deeper[1] < gap_diag / 2 else 'depth does not fully close it'}")
    else:
        print("  => inconclusive at this budget; see the d=4m rows and raise --steps")

if __name__ == "__main__":
    main()
