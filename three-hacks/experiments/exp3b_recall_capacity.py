"""
Experiment 3b -- the recall ceiling (Theorem 5.12), tested directly.

The earlier version of this file used sliding-window attention on Qwen as a
proxy for bounded state. Two problems: huggingface.co is unreachable from this
container, and more importantly the proxy conflates "bounded state" with
"bounded context", which is not what the theorem is about. Theorem 5.12 says an
m-bit stateful model cannot reproduce an n-token block when m < n*log2|V|. So we
build a model with an EXPLICIT m-bit state and vary m.

Task: read n values v_1..v_n in {0,1}, one per step, through an interface of
exactly b = log2(V) bits; then read a query index q; then emit v_q. The query
arrives AFTER the values, so the state must carry all of them -- exactly the
quantifier order the theorem requires. Storing n binary values needs n bits.

OPTIMISATION BUDGET MATTERS, AND MISLED US ONCE. At 800 steps this sweep
produced seven cells that failed WITH SUFFICIENT CAPACITY -- including b=4, n=2,
where 16 symbols must hold 2 bits. That cannot be a capacity effect, since a
4-bit interface strictly contains a 3-bit one and b=3, n=2 scored 1.00 in the
same sweep. It was undertraining: at 3000 steps b=4, n=2 reaches 1.000 on every
seed tested. The discrete bottleneck trains slowly, so an underpowered run looks
exactly like a capacity ceiling. Default steps raised accordingly, and --seeds
takes the best of several runs, which is the right estimator for an achievability
claim ("a b-bit machine CAN do this") as opposed to an impossibility claim.

PREDICTION (Theorem 5.12): success iff b >= n, i.e. the diagonal of the table.
KILL: high accuracy at b < n, which would refute the bound rather than the
proxy. Note the Fano form predicts graceful, not cliff-like, decay just below
threshold, so a soft edge is expected; accuracy near 1.0 well below the
diagonal is what would falsify it.

    python3 exp3b_recall_capacity.py
"""

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallMachine(nn.Module):
    """Streaming step machine with a hard b-bit interface between steps."""

    def __init__(self, n, V, d=96):
        super().__init__()
        self.n, self.V = n, V
        self.emb_val = nn.Embedding(2, d)
        self.emb_q = nn.Embedding(n, d)
        self.phase = nn.Embedding(2, d)
        self.core = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.GELU(),
                                  nn.Linear(4 * d, d), nn.LayerNorm(d))
        self.to_sym = nn.Linear(d, V)
        self.emb_sym = nn.Embedding(V, d)
        self.readout = nn.Linear(d, 2)
        self.s0 = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, vals, q, tau=1.0):
        B = vals.shape[0]
        s = self.s0.expand(B, -1)
        for i in range(self.n):                      # read the values
            x = self.emb_val(vals[:, i]) + self.phase(torch.zeros_like(q))
            s = self.core(torch.cat([x, s], -1))
            s = F.gumbel_softmax(self.to_sym(s), tau=tau, hard=True) @ self.emb_sym.weight
        x = self.emb_q(q) + self.phase(torch.ones_like(q))   # then the query
        s = self.core(torch.cat([x, s], -1))
        logits = self.readout(s)
        target = vals.gather(1, q.unsqueeze(1)).squeeze(1)
        return F.cross_entropy(logits, target), (logits.argmax(-1) == target).float().mean()


def batch(bs, n, device):
    vals = torch.randint(0, 2, (bs, n), device=device)
    q = torch.randint(0, n, (bs,), device=device)
    return vals, q


def run(n, V, d, steps, bs, device, lr=2e-3):
    model = RecallMachine(n, V, d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for i in range(steps):
        tau = max(0.5, 2.0 * (1 - i / steps))        # anneal the relaxation
        loss, _ = model(*batch(bs, n, device), tau=tau)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        return sum(model(*batch(bs, n, device), tau=0.5)[1].item()
                   for _ in range(10)) / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=1,
                    help="report the best over this many seeds")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cpu"

    print("Theorem 5.12: an m-bit state cannot recall n binary values when m < n.")
    print("Rows = interface width b (bits), columns = n values to retain.")
    print("Prediction: ~1.0 on and above the diagonal b >= n, chance (0.5) below.\n")
    header = "  b \\ n " + "".join(f"{n:>7}" for n in args.ns)
    print(header); print("-" * len(header))

    table = {}
    for b in args.bits:
        V = 2 ** b
        cells = []
        for n in args.ns:
            acc = 0.0
            for sd in range(args.seeds):
                random.seed(args.seed + sd); torch.manual_seed(args.seed + sd)
                acc = max(acc, run(n, V, args.d, args.steps, args.bs, device))
            table[(b, n)] = acc
            cells.append(acc)
        print(f"{b:>5}  " + "".join(f"{c:>7.2f}" for c in cells))

    print("\nVerdict per cell (predicted vs observed, threshold 0.9):")
    ok = bad = 0
    for (b, n), acc in sorted(table.items()):
        pred_ok = b >= n
        obs_ok = acc >= 0.9
        if pred_ok == obs_ok:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH b={b} n={n}: predicted "
                  f"{'success' if pred_ok else 'failure'}, observed acc={acc:.2f}")
    print(f"  {ok}/{ok+bad} cells match the prediction.")
    viol = [(b, n) for (b, n), a in table.items() if b < n and a >= 0.9]
    print(f"  Cells that would REFUTE the bound (b < n yet accurate): "
          f"{viol if viol else 'none'}")
    print("  A failure at b >= n is an optimisation result, not a capacity one;")
    print("  only a success at b < n would contradict Theorem 5.12.")
    if bad == 0:
        print("  Theorem 5.12's threshold reproduced exactly.")


if __name__ == "__main__":
    main()
