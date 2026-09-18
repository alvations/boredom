"""
Diagnostic 2 -- undertraining imitates a capacity ceiling.

Artefact 1 of 3. The first capacity sweep ran 800 steps per cell and produced
SEVEN cells that failed with sufficient capacity, including b=4, n=2: sixteen
interface symbols to hold two bits. That cannot be a capacity effect, and the
same sweep proves it internally -- b=3, n=2 scored 1.00, and a 4-bit interface
strictly contains a 3-bit one. Capacity is monotone; the results were not.

The cause is training budget. This script isolates it by holding the cell fixed
and varying only the step count and the seed.

Expected: ~0.5 (chance) at 800 steps on most seeds, 1.000 at 3000 on all of them.

    python3 d2_undertraining.py
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp3b_recall_capacity import run                                # noqa: E402

N, B, D = 2, 4, 64          # 16 symbols to store 2 bits: capacity is ample


def main():
    print(f"cell: n={N} values, b={B} bits ({2**B} symbols) -- capacity is ample")
    print("only the training budget varies\n")
    print(f"{'seed':>6} {'800 steps':>12} {'3000 steps':>12}")
    print("-" * 32)
    short, long = [], []
    for seed in (0, 1, 2):
        row = []
        for steps in (800, 3000):
            random.seed(seed); torch.manual_seed(seed)
            row.append(run(N, B, D, steps, 128, "cpu", mode="onehot"))
        short.append(row[0]); long.append(row[1])
        print(f"{seed:>6} {row[0]:>12.3f} {row[1]:>12.3f}")

    print(f"\n  mean at  800 steps: {sum(short)/len(short):.3f}")
    print(f"  mean at 3000 steps: {sum(long)/len(long):.3f}")
    print("\n  A capacity ceiling would not move with the step count. This one does,")
    print("  so the 800-step failures said nothing about Theorem 5.12.")


if __name__ == "__main__":
    main()
