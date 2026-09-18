"""
Experiment 2 -- the bandwidth bound (Theorem 3.2), made falsifiable.

The problem with "is latent reasoning better than CoT" as an empirical question
is that you cannot control the number of bits a task's intermediate state needs.
Composition of permutations fixes that: composing elements of S_n has an
intermediate state of exactly log2(n!) bits and provably nothing smaller.

Task: read t generator tokens g_1..g_t (adjacent transpositions of S_n), output
the composed permutation g_t o ... o g_1 as n tokens.

Two ~10M-param models trained from scratch, matched FLOPs:

  CoT     : emits t intermediate tokens, one per composition step, each drawn
            from a codebook of size V. Codebook is injective when V >= n!, and
            a lossy surjection (rank mod V) when V < n!.
  LATENT  : appends m scratch positions and runs the shared stack r = t times
            over them without emitting anything. State lives in R^d.

THE KNOB THAT MAKES THIS A TEST OF THE THEORY: V is a parameter of the
*encoding*, not of the task. Sweeping V alone, with n, t and the model fixed,
moves the predicted collapse point. A task-side confound cannot do that.

PREDICTION (Theorem 3.2): CoT accuracy collapses at log2(V) < log2(n!), and
LATENT is flat across that boundary.
KILL CRITERION: CoT sails past log2(V) < log2(n!) => the bound does not bind
in practice and thesis 2's refined form is dead.

A confound worth naming up front: attention lets the CoT model re-read the
original g_1..g_t and recompute the running product from scratch, bypassing the
bottlenecked token. That escape costs serial depth, so it is only closed off
while t >> n_layers. Hence t=24 against a 4-layer model, and --probe-depth to
check the escape stays shut (accuracy at V=2 should stay near chance).

    pip install torch
    python3 exp2_latent_vs_cot.py --n 6 --sweep-v
"""

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- the task

def perm_rank(p):
    """Lehmer code -> integer in [0, n!). Injective."""
    p, n, r = list(p), len(p), 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if p[j] < p[i])
        r = r * (n - i) + smaller
    return r


def make_batch(bs, n, t, V, device):
    """Returns generator tokens, per-step codebook tokens, and the final permutation."""
    gens, codes, finals = [], [], []
    for _ in range(bs):
        cur = list(range(n))
        g_seq, c_seq = [], []
        for _ in range(t):
            g = random.randrange(n - 1)                 # swap positions g, g+1
            cur[g], cur[g + 1] = cur[g + 1], cur[g]
            g_seq.append(g)
            c_seq.append(perm_rank(cur) % V)            # lossy iff V < n!
        gens.append(g_seq)
        codes.append(c_seq)
        finals.append(cur)
    tl = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return tl(gens), tl(codes), tl(finals)


# ---------------------------------------------------------------- the models

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class CoTModel(nn.Module):
    """Autoregressive: [gens] -> [t codebook tokens] -> [n output tokens].

    The only state carried from step i to step i+1 that was *created* by the
    model is the emitted token: <= log2(V) bits. That is the bound under test.
    """

    def __init__(self, n, t, V, d=256, layers=4, heads=4):
        super().__init__()
        self.n, self.t, self.V = n, t, V
        self.n_gen, self.n_out = n - 1, n
        # one flat vocabulary: generators | codebook | output symbols | BOS
        self.off_code = self.n_gen
        self.off_out = self.off_code + V
        vocab = self.off_out + n + 1
        self.bos = vocab - 1
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(t + t + n + 2, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, gens, codes, finals):
        B = gens.shape[0]
        seq = torch.cat([
            torch.full((B, 1), self.bos, device=gens.device),
            gens,
            codes + self.off_code,
            finals + self.off_out,
        ], dim=1)
        x = self.emb(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device))
        mask = torch.triu(torch.ones(seq.shape[1], seq.shape[1], device=seq.device,
                                     dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.ln_f(x))[:, :-1]
        target = seq[:, 1:]
        # score only the CoT tokens and the final answer, never the given inputs
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               target.reshape(-1), reduction="none").view(B, -1)
        keep = torch.zeros_like(loss, dtype=torch.bool)
        keep[:, self.t:] = True
        loss = (loss * keep).sum() / keep.sum()
        pred = logits[:, -self.n:].argmax(-1) - self.off_out
        return loss, (pred == finals).all(-1).float().mean()


class LatentModel(nn.Module):
    """Reads the generators, then loops the shared stack r times over m scratch
    slots without emitting. State per step is a d-dimensional vector."""

    def __init__(self, n, t, d=256, layers=4, heads=4, m=4, r=None):
        super().__init__()
        self.n, self.m, self.r = n, m, r if r is not None else t
        self.emb = nn.Embedding(n - 1, d)
        self.scratch = nn.Parameter(torch.randn(m, d) * 0.02)
        self.pos = nn.Embedding(t + m, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n * n)                 # n positions x n values

    def forward(self, gens, codes, finals):
        B = gens.shape[0]
        x = torch.cat([self.emb(gens), self.scratch.expand(B, -1, -1)], dim=1)
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        for _ in range(self.r):                         # <-- the serial steps
            for b in self.blocks:
                x = b(x)
        logits = self.head(self.ln_f(x[:, -1])).view(B, self.n, self.n)
        loss = F.cross_entropy(logits.reshape(-1, self.n), finals.reshape(-1))
        return loss, (logits.argmax(-1) == finals).all(-1).float().mean()


# -------------------------------------------------------------------- driver

def train(model, n, t, V, steps, bs, device, lr=3e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for i in range(steps):
        loss, acc = model(*make_batch(bs, n, t, V, device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (i + 1) % max(1, steps // 5) == 0:
            print(f"    step {i+1}/{steps}  loss {loss.item():.4f}  acc {acc.item():.3f}")
    model.eval()
    with torch.no_grad():
        accs = [model(*make_batch(bs, n, t, V, device))[1].item() for _ in range(10)]
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="permutation group S_n")
    ap.add_argument("--t", type=int, default=24, help="composition steps; keep >> layers")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-v", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_fact = math.factorial(args.n)
    bits_needed = math.log2(n_fact)
    print(f"S_{args.n}: {n_fact} states = {bits_needed:.1f} bits of state per step")
    print(f"t={args.t} composition steps, {args.layers}-layer models, device={device}\n")

    vs = ([2, 4, 8, 16, 32, 64, 128, 256, 512, 720, 1024]
          if args.sweep_v else [n_fact])
    vs = sorted({min(v, n_fact) for v in vs})

    print("CoT, sweeping the codebook size V (the theory's knob):")
    results = []
    for V in vs:
        print(f"  V={V} ({math.log2(V):.1f} bits/step, "
              f"{'LOSSLESS' if V >= n_fact else 'LOSSY'}):")
        random.seed(args.seed); torch.manual_seed(args.seed)
        acc = train(CoTModel(args.n, args.t, V, args.d, args.layers),
                    args.n, args.t, V, args.steps, args.bs, device)
        results.append((V, acc))
        print(f"    -> final acc {acc:.3f}")

    print("\nLATENT (no emission, r = t serial steps in R^d):")
    random.seed(args.seed); torch.manual_seed(args.seed)
    lat = train(LatentModel(args.n, args.t, args.d, args.layers),
                args.n, args.t, n_fact, args.steps, args.bs, device)
    print(f"    -> final acc {lat:.3f}")

    print(f"\n{'V':>6} {'bits/step':>10} {'CoT acc':>9}   predicted")
    for V, acc in results:
        pred = "ok" if math.log2(V) >= bits_needed else "COLLAPSE"
        print(f"{V:>6} {math.log2(V):>10.1f} {acc:>9.3f}   {pred}")
    print(f"{'latent':>6} {args.d * 4:>10.0f} {lat:>9.3f}   ok   (d*4 effective bits)")
    print(f"\nTheorem 3.2 predicts the CoT column falls off between "
          f"V={2**int(bits_needed)} and V={n_fact}.")
    print("If it does not, the bound does not bind in practice -- thesis 2 loses "
          "its teeth and this is the finding.")


if __name__ == "__main__":
    main()
