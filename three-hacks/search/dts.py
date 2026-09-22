"""
DTS -- a depth / time / state architecture, and the benchmark that scores each axis.

Everything the paper argued for is a switch here, next to the naive option:

  STATE  block type per layer   diag  | dense | attn
         Thm 5.8 / exp7: dense mixing is needed to track a hidden state;
         Thm 5.12: fixed state has a recall ceiling, so keep some attention.
  TIME   r latent loops over thought slots, mode overwrite | append,
         optional codebook projection every k loops.
         Cor 4.5: append (accumulate), never overwrite. Prop 4.15: k=1 if the
         rate suffices.
  DEPTH  auxiliary early-exit loss lambda on a shared head.
         exp5: this is what lifts self-drafting over the head-cost ceiling.

The benchmark is three procedurally generated tasks on one vocabulary, one per
axis, plus the corrected-cost early-exit speedup measured on all of them:

  RECALL   n key/value pairs, then a query      -> state (capacity)
  TRACK    cyclic-HMM next-symbol prediction,
           scored against the Bayes-optimal
           forward algorithm                    -> state (mixing)
  COMPOSE  T generators of S_5 -> the product   -> time (serial compute)
  DEPTH    alpha_l profile -> speedup S at an
           imposed head fraction u=0.26         -> depth

Fitness F = state + time + depth in [0, 3], with
  state = (recall_acc + track_headroom_captured) / 2
  time  = compose_acc
  depth = clip(2 (S - 0.9), 0, 1)
and hard budgets on parameters and compute so 'make it bigger' is not a move.

Held-out evaluation is on fresh procedurally generated samples.
"""

import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ vocabulary
K0, NK = 0, 16          # recall keys
V0, NV = 16, 8          # recall values
H0, NH = 24, 4          # hmm symbols
G0, NG = 28, 4          # S_5 adjacent transpositions
P0, NP = 32, 5          # permutation values
SEP, Q, THOUGHT, ANS = 37, 38, 39, 40
PAIR0 = 41              # pair tokens (k, v) -> PAIR0 + k*NV + v, for one-hop recall
VOCAB = 41 + NK * NV

IMPOSED_U = 0.26        # Qwen3-0.6B's head fraction; the synthetic vocab makes
                        # the model's own u ~1%, which would make depth free


@dataclass
class Cfg:
    name: str = "v0"
    d: int = 64
    L: int = 4
    heads: int = 2
    blocks: tuple = ("diag", "diag", "diag", "diag")
    # time axis
    r: int = 0
    time_mode: str = "overwrite"       # overwrite | append
    n_slots: int = 4
    proj_k: int = 0                    # project carried slots every k loops; 0 = never
    proj_bits: int = 16
    # depth axis
    ee_lambda: float = 0.0
    ee_layers: tuple = ()              # empty = 1..L-1
    # training
    steps: int = 1500
    lr: float = 2e-3
    bs: int = 32
    # provenance
    axis: str = "baseline"
    rationale: str = "naive on every axis"

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["blocks"] = tuple(d["blocks"])
        d["ee_layers"] = tuple(d.get("ee_layers", ()))
        return Cfg(**d)


# ------------------------------------------------------------------- blocks

def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class DiagRec(nn.Module):
    """s_t = a(x_t) (*) s_{t-1} + c(x_t): gated linear attention's action, diagonal."""
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d); self.c = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        a = torch.sigmoid(self.a(x)); c = self.c(x)
        s = torch.zeros(B, d, device=x.device); out = []
        for t in range(T):
            s = rms(a[:, t] * s + c[:, t])
            out.append(s)
        return self.o(torch.stack(out, 1))


class DenseRec(nn.Module):
    """s_t = (A s_{t-1}) (*) b(x_t) + c(x_t): the HMM forward-algorithm shape
    (Thm 5.6), a static dense transition with a per-token multiplicative gate.
    A is spectrally normalised for stability."""
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.b = nn.Linear(d, d); self.c = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        A = self.A / (torch.linalg.matrix_norm(self.A, ord=2) + 1e-6)
        b = torch.sigmoid(self.b(x)); c = self.c(x)
        s = torch.zeros(B, d, device=x.device); out = []
        for t in range(T):
            s = rms((s @ A.T) * b[:, t] + c[:, t])
            out.append(s)
        return self.o(torch.stack(out, 1))


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.mha(x, x, x, attn_mask=mask, need_weights=False)[0]


class Block(nn.Module):
    def __init__(self, kind, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mix = {"diag": DiagRec(d), "dense": DenseRec(d),
                    "attn": Attn(d, heads)}[kind]
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


# -------------------------------------------------------------------- model

class DTS(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(256, d)
        self.blocks = nn.ModuleList([Block(k, d, cfg.heads) for k in cfg.blocks])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.carry = nn.Linear(d, d)                 # slot state -> next-loop embedding
        if cfg.proj_k:
            self.to_bits = nn.Linear(d, cfg.proj_bits)
            self.from_bits = nn.Linear(cfg.proj_bits, d)
        self.compute = 0                             # block-applications, reset per forward

    def project(self, s):
        p = torch.sigmoid(self.to_bits(s))
        return self.from_bits((p > 0.5).float() + p - p.detach())

    def stack(self, e):
        """Run the blocks over an embedded sequence; return all layer outputs."""
        T = e.shape[1]
        x = e + self.pos(torch.arange(T, device=e.device))
        hs = [x]
        for b in self.blocks:
            x = b(x); hs.append(x)
        self.compute += len(self.blocks) * T
        return hs

    def forward(self, tokens, n_slots_here):
        """tokens: [B, N] with THOUGHT placeholders already inserted for pass 0.
        Returns per-layer hidden states at the final pass and the slot span."""
        cfg = self.cfg
        self.compute = 0
        e = self.emb(tokens)
        B, N = tokens.shape
        slot_idx = (tokens[0] == THOUGHT).nonzero().flatten()
        hs = self.stack(e)
        if cfg.r == 0 or len(slot_idx) == 0:
            return hs
        # positions of the (first) slot span and everything after it
        s0, s1 = int(slot_idx[0]), int(slot_idx[-1]) + 1
        head_e, tail_e = e[:, :s0], e[:, s1:]
        slot_e = e[:, s0:s1]
        for i in range(1, cfg.r + 1):
            carried = self.carry(self.norm(hs[-1][:, s0:s0 + (s1 - s0)]))
            if cfg.proj_k and i % cfg.proj_k == 0:
                carried = self.project(carried)
            if cfg.time_mode == "append":
                slot_e = torch.cat([slot_e, carried], 1)       # accumulate
            else:
                slot_e = carried                                # overwrite
            e2 = torch.cat([head_e, slot_e, tail_e], 1)
            hs = self.stack(e2)
            # for append, the NEW slots are the last n written; carry from them
            s0 = head_e.shape[1] + slot_e.shape[1] - cfg.n_slots
            s1 = head_e.shape[1] + slot_e.shape[1]
        return hs

    def logits_from(self, h):
        return self.head(self.norm(h))


# -------------------------------------------------------------------- tasks

def perm_rank(p):
    p, n, r = list(p), len(p), 0
    for i in range(n):
        r = r * (n - i) + sum(1 for j in range(i + 1, n) if p[j] < p[i])
    return r


class Tasks:
    def __init__(self, cfg: Cfg, seed=0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        g = torch.Generator().manual_seed(1234)          # fixed HMM for all runs
        m, delta, q = NH, 0.1, 0.5
        P = torch.zeros(m, m)
        for i in range(m):
            P[i, (i + 1) % m] = 1
        self.A = (1 - delta) * P + delta / m
        self.B = torch.full((m, m), (1 - q) / (m - 1))
        for i in range(m):
            self.B[i, i] = q
        self.pi = torch.full((m,), 1.0 / m)
        self.T_track = 24
        self.n_pairs = 6
        self.T_comp = 8
        self.opt_nll = self._optimal_nll(self.sample_track(512, torch.Generator().manual_seed(7)))

    # ---- RECALL: k1 v1 ... kn vn Q kj [slots] ANS -> vj
    def sample_recall(self, bs):
        n = self.n_pairs
        toks, tgt = [], []
        for _ in range(bs):
            keys = self.rng.sample(range(NK), n)
            vals = [self.rng.randrange(NV) for _ in range(n)]
            j = self.rng.randrange(n)
            seq = []
            for k, v in zip(keys, vals):
                if self.cfg.recall_mode == "pairs":
                    seq.append(PAIR0 + k * NV + v)
                else:
                    seq += [K0 + k, V0 + v]
            seq += [Q, K0 + keys[j]] + [THOUGHT] * (self.cfg.n_slots if self.cfg.r else 0) + [ANS]
            toks.append(seq); tgt.append(V0 + vals[j])
        return torch.tensor(toks), torch.tensor(tgt)

    # ---- TRACK: streaming next-symbol on the cyclic HMM (no slots)
    def sample_track(self, bs, gen=None):
        gen = gen or torch.Generator().manual_seed(self.rng.randrange(1 << 30))
        z = torch.multinomial(self.pi.expand(bs, -1), 1, generator=gen).squeeze(1)
        obs = torch.empty(bs, self.T_track, dtype=torch.long)
        for t in range(self.T_track):
            obs[:, t] = torch.multinomial(self.B[z], 1, generator=gen).squeeze(1)
            z = torch.multinomial(self.A[z], 1, generator=gen).squeeze(1)
        return obs + H0

    @torch.no_grad()
    def _optimal_nll(self, obs):
        o = obs - H0
        n, T = o.shape
        al = self.pi.expand(n, -1) * self.B[:, o[:, 0]].T
        al = al / al.sum(1, keepdim=True); tot = 0.0
        for t in range(1, T):
            pred = (al @ self.A) @ self.B
            tot += -torch.log(pred.gather(1, o[:, t:t + 1])).sum().item()
            al = (al @ self.A) * self.B[:, o[:, t]].T
            al = al / al.sum(1, keepdim=True)
        return tot / (n * (T - 1))

    # ---- COMPOSE: g1..gT [slots] SEP p1..p4 -> p1..p5 (teacher-forced)
    def sample_compose(self, bs):
        toks, tgt = [], []
        for _ in range(bs):
            cur, gs = list(range(NP)), []
            for _ in range(self.T_comp):
                g = self.rng.randrange(NG)
                cur[g], cur[g + 1] = cur[g + 1], cur[g]
                gs.append(G0 + g)
            slots = [THOUGHT] * (self.cfg.n_slots if self.cfg.r else 0)
            toks.append(gs + slots + [SEP] + [P0 + p for p in cur[:-1]])
            tgt.append([P0 + p for p in cur])
        return torch.tensor(toks), torch.tensor(tgt)


# ------------------------------------------------------------ loss / scoring

def task_forward(model, task, toks, tgt):
    """Returns (per-layer logits at scored positions, targets flat)."""
    cfg = model.cfg
    hs = model(toks, cfg.n_slots)
    if task == "track":
        pos = slice(0, toks.shape[1] - 1); tg = toks[:, 1:]
    elif task == "recall":
        pos = slice(-1, None); tg = tgt.unsqueeze(1)
    else:  # compose: predict at SEP..p4, i.e. the last 5 positions
        pos = slice(-NP, None); tg = tgt
    per_layer = [model.logits_from(h[:, pos]) for h in hs[1:]]
    return per_layer, tg.reshape(-1)


def losses(model, task, toks, tgt):
    per_layer, tg = task_forward(model, task, toks, tgt)
    ce = lambda lg: F.cross_entropy(lg.reshape(-1, VOCAB), tg)
    main = ce(per_layer[-1])
    cfg = model.cfg
    if cfg.ee_lambda > 0:
        layers = cfg.ee_layers or tuple(range(1, cfg.L))
        aux = torch.stack([ce(per_layer[l - 1]) for l in layers]).mean()
        return main + cfg.ee_lambda * aux, main
    return main, main


@torch.no_grad()
def evaluate(model, tasks, n_batches=12, bs=64):
    model.eval()
    out = {}
    alpha_sum = torch.zeros(model.cfg.L); n_pos = 0
    # recall
    acc = 0.0
    for _ in range(n_batches):
        toks, tgt = tasks.sample_recall(bs)
        pl, tg = task_forward(model, "recall", toks, tgt)
        acc += (pl[-1].argmax(-1).reshape(-1) == tg).float().mean().item()
        a, n = _alpha(pl); alpha_sum += a; n_pos += n
    out["recall_acc"] = acc / n_batches
    # track
    nll = 0.0
    for _ in range(n_batches):
        toks = tasks.sample_track(bs)
        pl, tg = task_forward(model, "track", toks, None)
        nll += F.cross_entropy(pl[-1].reshape(-1, VOCAB), tg).item()
        a, n = _alpha(pl); alpha_sum += a; n_pos += n
    nll /= n_batches
    uniform = math.log(NH)
    out["track_nll"] = nll; out["track_opt"] = tasks.opt_nll
    out["track_score"] = max(0.0, min(1.0, (uniform - nll) / (uniform - tasks.opt_nll)))
    # compose
    acc = 0.0
    for _ in range(n_batches):
        toks, tgt = tasks.sample_compose(bs)
        pl, tg = task_forward(model, "compose", toks, tgt)
        pred = pl[-1].argmax(-1)
        acc += (pred == tgt).all(-1).float().mean().item()
        a, n = _alpha(pl); alpha_sum += a; n_pos += n
    out["compose_acc"] = acc / n_batches
    # depth
    alpha = (alpha_sum / n_pos).tolist()
    out["alpha"] = alpha
    out["speedup"], out["exit_layer"], out["gamma"] = speedup(alpha, IMPOSED_U)
    out["state"] = 0.5 * out["recall_acc"] + 0.5 * out["track_score"]
    out["time"] = out["compose_acc"]
    # DEPTH IS GATED ON FINAL-LAYER QUALITY. An untrained model has every
    # layer near-uniform, so p_l ~ p_L, alpha ~ 1 and S is maximal: a model
    # that predicts nothing would score full depth credit. Speculative yield
    # is only worth anything if the final model is worth running, so depth
    # credit scales with task quality (saturating once the model is half-good).
    out["depth_raw"] = max(0.0, min(1.0, 2 * (out["speedup"] - 0.9)))
    out["depth"] = out["depth_raw"] * min(1.0, out["state"] + out["time"])
    out["fitness"] = out["state"] + out["time"] + out["depth"]
    model.train()
    return out


def _alpha(per_layer):
    """Sum over positions of overlap between each layer's readout and the last."""
    pL = torch.softmax(per_layer[-1].float(), -1).reshape(-1, VOCAB)
    a = torch.zeros(len(per_layer))
    for l, lg in enumerate(per_layer):
        pl = torch.softmax(lg.float(), -1).reshape(-1, VOCAB)
        a[l] = torch.minimum(pl, pL).sum(-1).sum()
    return a, pL.shape[0]


def speedup(alpha, u, max_gamma=6):
    """Corrected memory-bound cost (Prop 3.15, no-reuse): C = g[(1-u)rho + u] + 1."""
    L = len(alpha); best = (0.0, None, None)
    for l in range(1, L):                       # genuine early exits only
        rho = l / L; a = min(max(alpha[l - 1], 1e-9), 1 - 1e-9)
        for g in range(1, max_gamma + 1):
            yld = (1 - a ** (g + 1)) / (1 - a)
            S = yld / (g * ((1 - u) * rho + u) + 1)
            if S > best[0]:
                best = (S, l, g)
    return best


# ------------------------------------------------------------- one training run

def n_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_compute(model, tasks):
    """Block-applications per example, averaged over the three tasks."""
    with torch.no_grad():
        tot = 0
        for task in ("recall", "track", "compose"):
            toks = (tasks.sample_recall(1)[0] if task == "recall" else
                    tasks.sample_track(1) if task == "track" else tasks.sample_compose(1)[0])
            model(toks, model.cfg.n_slots); tot += model.compute
    return tot / 3


def run(cfg: Cfg, seed: int, quiet=False):
    random.seed(seed); torch.manual_seed(seed)
    tasks = Tasks(cfg, seed=seed)
    model = DTS(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps)
    t0 = time.time()
    for i in range(cfg.steps):
        task = ("recall", "track", "compose")[i % 3]
        if task == "recall":
            toks, tgt = tasks.sample_recall(cfg.bs)
        elif task == "track":
            toks, tgt = tasks.sample_track(cfg.bs), None
        else:
            toks, tgt = tasks.sample_compose(cfg.bs)
        loss, _ = losses(model, task, toks, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not quiet and (i + 1) % max(1, cfg.steps // 4) == 0:
            print(f"    step {i+1}/{cfg.steps} loss {loss.item():.3f}", flush=True)
    # held-out: fresh task generator with a different seed
    ev = evaluate(model, Tasks(cfg, seed=10_000 + seed))
    ev["params"] = n_params(model)
    ev["compute"] = measure_compute(model, tasks)
    ev["train_seconds"] = time.time() - t0
    ev["seed"] = seed
    return ev


def run_config(cfg: Cfg, seeds=(0, 1), quiet=False):
    per = [run(cfg, s, quiet) for s in seeds]
    keys = ["fitness", "state", "time", "depth", "depth_raw", "recall_acc",
            "track_score", "compose_acc", "speedup"]
    agg = {k: sum(p[k] for p in per) / len(per) for k in keys}
    agg.update({k + "_std": (sum((p[k] - agg[k]) ** 2 for p in per) / max(len(per) - 1, 1)) ** 0.5
                for k in keys})
    return {"config": cfg.to_dict(), "seeds": per, "mean": agg,
            "params": per[0]["params"], "compute": per[0]["compute"]}
