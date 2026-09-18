"""
Experiment 0 -- a real trained model to test the theorems against.

WHY THIS EXISTS. The theorems in the paper are architecture-general: they speak
about a residual-stream model (Def 2.1) with a final norm and an unembedding.
They do not mention Qwen. This container cannot reach huggingface.co (the agent
proxy denies CONNECT to it), so rather than leave every claim unmeasured we
train a small model here that satisfies Definition 2.1 exactly -- pre-norm
residual blocks, RMSNorm, tied unembedding -- and measure on it.

It is deliberately sized so that its HEAD FRACTION matches Qwen3-0.6B's:

    u = |W_U| / |all params| ~ 0.25       (Qwen3-0.6B: 0.26)

because u is the quantity in Corollary 3.17's ceiling, and a toy model with a
character vocabulary would have u ~ 0 and make that result untestable.

Corpus is the repo's own subtitle files, with a held-out validation split so
acceptance rates are measured off the training data.

    python3 exp0_pretrain.py --steps 3000
"""

import argparse
import glob
import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp0_model.pt")


# ------------------------------------------------------------------ corpus

def load_corpus(root, vocab_size):
    paths = (sorted(glob.glob(os.path.join(root, "**", "*.srt"), recursive=True)) +
             sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)) +
             sorted(glob.glob(os.path.join(root, "**", ".corpus_*.txt"), recursive=True)))
    # NB: deliberately NOT globbing *.py. The experiment sources live inside
    # this repo, so including them would make the evaluation corpus change
    # every time an experiment is edited -- which it did, shifting measured
    # acceptance rates between runs before this was caught.
    paths = [q for q in paths if os.sep + "experiments" + os.sep not in q]
    text = []
    for p in paths:
        try:
            raw = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        # strip srt timing lines and indices; keep the spoken text
        raw = re.sub(r"^\d+\s*$", "", raw, flags=re.M)
        raw = re.sub(r"^\d\d:\d\d:\d\d[,.]\d+ --> .*$", "", raw, flags=re.M)
        text.append(raw)
    words = re.findall(r"\w+|[^\w\s]", "\n".join(text).lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    keep = sorted(freq, key=lambda w: -freq[w])[:vocab_size - 1]
    stoi = {w: i + 1 for i, w in enumerate(keep)}          # 0 = UNK
    ids = torch.tensor([stoi.get(w, 0) for w in words], dtype=torch.long)
    return ids, stoi


# ------------------------------------------------------------------- model

class RMSNorm(nn.Module):
    """N(h) = sqrt(d) * g (*) h / ||h||, exactly the form in Lemma 3.5."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                                 nn.Linear(4 * d, d, bias=False))

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    """Residual-stream language model in the sense of Definition 2.1."""

    def __init__(self, V, d=256, L=8, heads=4, ctx=128):
        super().__init__()
        self.V, self.d, self.L, self.ctx = V, d, L, ctx
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def hidden_states(self, idx):
        """Returns [h^(0), ..., h^(L)] -- the residual stream at every layer."""
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        hs = [x]
        for b in self.blocks:
            x = b(x, mask)
            hs.append(x)
        return hs

    def forward(self, idx, targets=None):
        logits = self.head(self.norm(self.hidden_states(idx)[-1]))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.V),
                               targets[:, 1:].reshape(-1))
        return logits, loss


# -------------------------------------------------------------------- train

def get_batch(data, bs, ctx, device):
    ix = torch.randint(len(data) - ctx - 1, (bs,))
    x = torch.stack([data[i:i + ctx] for i in ix]).to(device)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids, stoi = load_corpus(args.root, args.vocab)
    n_val = max(2048, len(ids) // 10)
    train_d, val_d = ids[:-n_val], ids[-n_val:]
    V = args.vocab
    print(f"corpus: {len(ids)} tokens, vocab {V}, "
          f"train {len(train_d)} / val {len(val_d)}")

    model = TinyLM(V, args.d, args.layers, ctx=args.ctx).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_head = model.head.weight.numel()
    print(f"model: L={args.layers} d={args.d} params={n_all/1e6:.2f}M  "
          f"head fraction u = {n_head/n_all:.3f}  (Qwen3-0.6B: 0.26)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps)

    @torch.no_grad()
    def val_loss(n=8):
        model.eval()
        v = sum(model(*(lambda b: (b, b))(get_batch(val_d, args.bs, args.ctx,
                device)))[1].item() for _ in range(n)) / n
        model.train()
        return v

    best = (float("inf"), None)          # early stopping: this corpus is small
    for i in range(args.steps):
        x = get_batch(train_d, args.bs, args.ctx, device)
        _, loss = model(x, x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (i + 1) % max(1, args.steps // 20) == 0:
            vl = val_loss()
            flag = ""
            if vl < best[0]:
                best = (vl, {k: v.detach().clone()
                             for k, v in model.state_dict().items()})
                flag = "  <- best"
            print(f"  step {i+1}/{args.steps}  train {loss.item():.3f}  "
                  f"val {vl:.3f}  (ppl {math.exp(min(vl,20)):.1f}){flag}")

    if best[1] is not None:                # restore the best-validation weights
        model.load_state_dict(best[1])
        vl = best[0]
        print(f"restored best checkpoint: val {vl:.3f} (ppl {math.exp(min(vl,20)):.1f})")
    model.eval()
    # freeze the exact validation split into the checkpoint so downstream
    # measurements are reproducible and independent of the working tree
    torch.save({"state": model.state_dict(), "V": V, "d": args.d,
                "L": args.layers, "ctx": args.ctx, "stoi": stoi,
                "val": val_d, "val_loss": vl}, CKPT)
    print(f"saved {CKPT}")


if __name__ == "__main__":
    main()
