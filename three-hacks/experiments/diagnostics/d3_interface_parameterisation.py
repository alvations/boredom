"""
Diagnostic 3 -- two encodings of the SAME capacity train completely differently.

Artefact 2 of 3. A b-bit interface can be written as:

  onehot  one symbol drawn from a codebook of size 2^b
  bits    b independent binary units, straight-through

Both carry exactly b bits, which is the only quantity Theorem 5.12 mentions. But
`onehot` makes the optimiser DISCOVER an injective code through a biased
straight-through gradient, and at n>=3 it never cleared threshold even at 7000
steps with two seeds. `bits` makes the natural solution -- "unit i holds value
i" -- directly representable.

Reporting only the onehot numbers would have looked like evidence against the
theorem's tightness, when it is an artefact of how the bottleneck is written.

Expected: bits reaches 1.000 on the diagonal b=n; onehot does not, at n>=3.

    python3 d3_interface_parameterisation.py
"""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp3b_recall_capacity import run                                # noqa: E402

D, STEPS = 96, 3000


def main():
    print("Same capacity (b bits), two parameterisations.\n")
    print(f"{'n':>4} {'b':>4} {'predicted':>11} {'onehot':>9} {'bits':>9}")
    print("-" * 42)
    for n, b in ((2, 2), (3, 3), (4, 4), (3, 2), (4, 3)):
        row = {}
        for mode in ("onehot", "bits"):
            random.seed(0); torch.manual_seed(0)
            row[mode] = run(n, b, D, STEPS, 128, "cpu", mode=mode)
        pred = "success" if b >= n else "FAIL"
        print(f"{n:>4} {b:>4} {pred:>11} {row['onehot']:>9.3f} {row['bits']:>9.3f}")

    print("\n  Capacity is identical down each pair of columns; only trainability")
    print("  differs. The b=n rows are where the two diverge, and those are")
    print("  exactly the rows that decide whether the bound looks tight.")
    print("  Neither parameterisation ever succeeds below the diagonal, which is")
    print("  the only direction that could refute a lower bound.")


if __name__ == "__main__":
    main()
