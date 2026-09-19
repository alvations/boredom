"""
Diagnostic 4 -- the artefact that FLATTERED our own hypothesis.

Artefact 3 of 3, and the most dangerous, because its naive reading confirms the
thesis instead of contradicting it.

The first bandwidth sweep used T=12 composition steps and found the discrete
interface scoring ~0 at EVERY width from 3 to 10 bits while the continuous
interface scored 1.000. Read quickly, that is a textbook confirmation of the
bandwidth argument: discrete channel bad, continuous channel good.

It is nothing of the sort. The threshold for S_5 is log2(120) = 6.91 bits, and a
bound that binds at 6.91 bits cannot explain failure at 20 bits. The real
constraint at T=12 is credit assignment through twelve stacked hard
quantisations, whose straight-through gradients degrade with the number of
bottlenecks on the path.

The two failure modes are separable, and that is the point of this script:

    optimisation failure  is monotone in T   (worse with more bottlenecks)
    bandwidth failure     is monotone in b   (worse with a narrower channel)

It also shows a secondary effect worth knowing: at T=4, b=7 (just above the
entropy bound) scores well below b=15, the "natural" code of 5 positions at 3
bits each. Gradient descent reaches the natural code, not the dense Lehmer code
that saturates the entropy bound.

    python3 d4_bottleneck_depth.py
"""

import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp2_latent_vs_cot import StepMachine, train                    # noqa: E402

import argparse

N, D, STEPS = 5, 96, 3000
NEED = math.log2(math.factorial(N))          # 6.91 bits for S_5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--d", type=int, default=D)
    args = ap.parse_args()

    print(f"S_{N}: the interface needs log2({N}!) = {NEED:.2f} bits "
          f"({args.steps} steps)\n")
    print(f"{'T':>4} {'b':>4} {'b >= need?':>11} {'acc':>8}   reading")
    print("-" * 58)
    cells = [(4, 7), (4, 15), (12, 7), (12, 15), (12, 20)]
    got = {}
    for T, b in cells:
        random.seed(0); torch.manual_seed(0)
        acc = train(StepMachine(N, args.d, bits=b), N, T, args.steps, 64,
                    "cpu", quiet=True)
        got[(T, b)] = acc
        note = ""
        if b >= NEED and acc < 0.5:
            note = "fails DESPITE ample bits -> not bandwidth"
        elif b >= NEED and acc >= 0.85:
            note = "succeeds, as bandwidth predicts"
        print(f"{T:>4} {b:>4} {'yes' if b >= NEED else 'no':>11} {acc:>8.3f}   {note}")

    # Guard: if even the shallow, wide cell fails, the run is undertrained and
    # NOTHING here is interpretable -- including the T-vs-b comparison. This
    # script exists to warn against reading undertrained failures as capacity
    # results, so it must not commit that error itself.
    reference = got.get((4, 15), 0.0)
    if reference < 0.5:
        print(f"\n  UNDERTRAINED: the reference cell (T=4, b=15 -- shallowest")
        print(f"  bottleneck, widest channel) scored {reference:.3f}. If that cell")
        print("  cannot be learned, no comparison below it means anything. Raise")
        print("  --steps (3000 is the default for a reason) and see d2. Treating")
        print("  these numbers as evidence would repeat the exact mistake this")
        print("  script documents.")
        return

    print("\n  Holding b fixed and raising T destroys accuracy:")
    for b in (7, 15):
        a4, a12 = got.get((4, b)), got.get((12, b))
        if a4 is not None and a12 is not None:
            print(f"    b={b:>2}: T=4 -> {a4:.3f}   T=12 -> {a12:.3f}")
    print("\n  That is optimisation, not capacity: the channel width never changed.")
    print("  Run the real sweep at T=4, where the optimiser succeeds and the")
    print("  bandwidth bound is the binding constraint:")
    print("    python3 ../exp2_latent_vs_cot.py --n 5 --T 4 --steps 4000 "
          "--seeds 2 --sweep-v --bit-list 3 4 5 6 7 8 15")


if __name__ == "__main__":
    main()
