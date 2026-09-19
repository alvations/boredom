"""
Experiment 5 -- does early-exit TRAINING rescue thesis 1?

Everything so far leaves thesis 1 in one place. Speculative sampling is exact
for any draft (Thm 3.1), self-drafting weakly dominates an external draft
(Thm 3.21), but with the corrected cost model the untuned early-exit draft does
not pay for itself: best memory-bound S = 1.026 at this model's head fraction,
and exactly 1.000 -- the degenerate rho=1 -- at Qwen3-0.6B's u=0.26.

Remark 3.22 says the obstruction is the training objective: W_U was fit against
N(h^L) only, so N(h^l) is off-distribution for it. The remedy in LayerSkip and
Draft&Verify is an auxiliary loss that makes intermediate layers decodable.
This experiment applies exactly that and re-measures.

    loss = CE(head(norm(h^L)))  +  lambda * mean_{l<L} CE(head(norm(h^l)))

Same shared head, same data, starting from the exp0 checkpoint. Then exp1_local
measures the result with --ckpt.

PREDICTION: alpha_l rises at every l < L. The question is by how much.
KILL: if even after early-exit training the best memory-bound S at u=0.26 is
<= 1.0, thesis 1 fails on this model as an engineering claim -- the only
remaining escape would be scale, which cannot be tested here.
SECONDARY: final-layer validation loss must not degrade materially, or the
speedup is bought with model quality.

    python3 exp5_early_exit_training.py --steps 600
    python3 exp1_local.py --ckpt exp5_model.pt
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

from exp0_pretrain import TinyLM, CKPT, get_batch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp5_model.pt")


def early_exit_loss(model, x, lam, layers):
    hs = model.hidden_states(x)
    tgt = x[:, 1:].reshape(-1)
    ce = lambda h: F.cross_entropy(
        model.head(model.norm(h))[:, :-1].reshape(-1, model.V), tgt)
    final = ce(hs[-1])
    aux = torch.stack([ce(hs[l]) for l in layers]).mean()
    return final + lam * aux, final, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="a step is kept if final-layer val loss is within this "
                         "of the best so far; lets the auxiliary loss trade a "
                         "little final quality for decodable intermediates")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="which intermediate layers get the auxiliary loss; "
                         "default: all of 1..L-1")
    args = ap.parse_args()

    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyLM(ck["V"], ck["d"], ck["L"], ctx=ck["ctx"])
    model.load_state_dict(ck["state"])
    val = ck["val"]
    layers = args.layers or list(range(1, model.L))

    # The training split comes from the checkpoint, never from re-globbing the
    # working tree. An earlier version re-globbed and asserted the val split
    # still matched; the assertion fired, because README edits since training
    # had changed the corpus. The guard did its job; this removes the hazard.
    if "train" not in ck:
        raise SystemExit("checkpoint has no 'train' split; retrain with the "
                         "current exp0_pretrain.py or reconstruct it from git")
    train_d = ck["train"]

    @torch.no_grad()
    def val_final_loss(n=8):
        model.eval()
        v = 0.0
        for _ in range(n):
            xv = get_batch(val, args.bs, model.ctx, "cpu")
            v += model(xv, xv)[1].item()
        model.train()
        return v / n

    v0 = val_final_loss()
    print(f"before: final-layer val loss {v0:.3f} (ppl {math.exp(v0):.1f})")
    print(f"auxiliary loss on layers {layers}, lambda={args.lam}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps)
    # Early stopping on the FINAL layer's validation loss. The base model was
    # early-stopped on a 47k-token corpus; continued training overfits it
    # within a few hundred steps whatever the auxiliary loss does, and a
    # speedup measured on an overfit model is bought with quality.
    best = (v0, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0)
    model.train()
    for i in range(args.steps):
        x = get_batch(train_d, args.bs, model.ctx, "cpu")
        loss, fin, aux = early_exit_loss(model, x, args.lam, layers)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (i + 1) % max(1, args.steps // 12) == 0:
            vl = val_final_loss(6)
            flag = ""
            if vl < best[0] + args.tolerance:
                best = (vl, {k: v.detach().clone()
                             for k, v in model.state_dict().items()}, i + 1)
                flag = "  <- kept"
            print(f"  step {i+1}/{args.steps}  final {fin.item():.3f}  "
                  f"aux {aux.item():.3f}  val {vl:.3f}{flag}")

    if best[2] == 0:
        print("\nno step improved final-layer val loss within tolerance; keeping "
              "the LAST state anyway so the auxiliary loss has had some effect, "
              "but the quality warning below applies")
    else:
        print(f"\nrestoring step {best[2]} (val {best[0]:.3f})")
        model.load_state_dict(best[1])

    v1 = val_final_loss()
    print(f"\nafter:  final-layer val loss {v1:.3f} (ppl {math.exp(v1):.1f})  "
          f"delta {v1 - v0:+.3f}")
    if v1 - v0 > 0.1:
        print("  WARNING: final-layer quality degraded; any speedup below is")
        print("  partly bought with model quality and must be read as such.")

    model.eval()
    ck["state"] = model.state_dict()
    ck["val_loss"] = v1
    ck["early_exit_trained"] = dict(steps=args.steps, lam=args.lam, layers=layers,
                                    val_before=v0, val_after=v1)
    torch.save(ck, OUT)
    print(f"saved {OUT}\nnow run: python3 exp1_local.py --ckpt {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
