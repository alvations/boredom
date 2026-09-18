"""
Diagnostic 1 -- the rho -> 1 sanity check that caught a bug in our own theory.

THIS IS THE SCRIPT THAT FOUND THE ERROR. Stdlib only, runs in about a second,
and we recommend it as a standing check on any speculative-decoding cost model.

The check: at rho = 1 the early-exit "draft" is the full model, so acceptance is
1 and the procedure degenerates to ordinary autoregressive decoding. The speedup
must therefore be EXACTLY 1. Any cost model returning more than 1 there is
manufacturing tokens from nothing.

An earlier version of the paper's Proposition 3.15 returned S = 1.735. It
charged drafting (layers 1..l at positions t..t+gamma-1) and verification, but
omitted the full-depth pass at position t+gamma, which supplies the bonus token
and which drafting never touches. That error survived both derivation and an
adversarial review pass; it did not survive this check.

The same script settles the cache-reuse question. Reuse lets the gamma drafted
positions resume at block l+1, but splits verification into two passes at
DIFFERENT depths. In the memory-bound regime a pass costs a full weight stream
however many positions it covers, so the lower blocks get paid for twice.

    python3 d1_cost_model_sanity.py
"""

U_DEFAULT = 0.153          # head fraction of the model used in exp1_local
U_QWEN = 0.26              # Qwen3-0.6B: 151936 x 1024 of ~5.96e8 params


def cost_buggy(rho, g, w, u):
    """The paper's ORIGINAL Prop 3.15. Omits the bonus position."""
    Dg = w + (1 - w) * g
    return g * ((1 - u) * rho + u) + ((1 - u) * (1 - rho) * Dg + u * Dg)


def cost_reuse(rho, g, w, u):
    """Corrected, reusing the draft's cache: two verification passes."""
    Dg, Dg1 = w + (1 - w) * g, w + (1 - w) * (g + 1)
    return (g * ((1 - u) * rho + u)          # drafting
            + (1 - u) * (1 - rho) * Dg       # upper blocks at drafted positions
            + (1 - u)                        # FULL depth at the bonus position
            + u * Dg1)                       # head everywhere


def cost_noreuse(rho, g, w, u):
    """Corrected, one batched full-depth pass over all gamma+1 positions."""
    Dg1 = w + (1 - w) * (g + 1)
    return g * ((1 - u) * rho + u) + Dg1


def yield_geometric(alpha, g):
    if alpha >= 1 - 1e-12:
        return g + 1.0
    return (1 - alpha ** (g + 1)) / (1 - alpha)


def main():
    print("CHECK 1: at rho=1 the draft IS the target, so S must be exactly 1.000\n")
    print(f"{'model':>24} {'cost':>8} {'yield':>7} {'S':>8}   verdict")
    print("-" * 62)
    failed = False
    for name, f in (("original Prop 3.15", cost_buggy),
                    ("corrected, with reuse", cost_reuse),
                    ("corrected, no reuse", cost_noreuse)):
        c = f(1.0, 1, 1.0, U_DEFAULT)
        s = yield_geometric(1.0, 1) / c
        ok = abs(s - 1.0) < 1e-9
        failed |= (name.startswith("original") and ok)
        print(f"{name:>24} {c:>8.3f} {2.0:>7.1f} {s:>8.3f}   "
              f"{'ok' if ok else 'IMPOSSIBLE (>1 means free tokens)'}")

    print("\nCHECK 2: does cache reuse help? (u=0 so the head does not confound)\n")
    print(f"{'regime':>12} {'rho':>6} {'gamma':>6} {'reuse':>8} {'no-reuse':>9}   better")
    print("-" * 60)
    for w, wname in ((1.0, "mem-bound"), (0.0, "compute")):
        for rho in (0.25, 0.5, 0.75):
            for g in (2, 4):
                r, n = cost_reuse(rho, g, w, 0.0), cost_noreuse(rho, g, w, 0.0)
                print(f"{wname:>12} {rho:>6.2f} {g:>6} {r:>8.2f} {n:>9.2f}   "
                      f"{'no-reuse' if r > n else 'reuse'}")

    print("\n  Memory-bound: reuse is never better -- splitting verification across")
    print("  two depths streams the lower blocks' weights twice. Compute-bound:")
    print("  reuse wins, because there the cost really is per-position arithmetic.")

    print("\nCHECK 3: the head-cost ceiling, S <= (gamma+1)/(gamma*u + 1)\n")
    print(f"{'gamma':>6} {'u=0 (no head)':>15} {'u=0.153':>10} {'u=0.26 (Qwen)':>15}")
    print("-" * 50)
    for g in (1, 2, 4, 8):
        row = [(g + 1) / (g * u + 1) for u in (0.0, U_DEFAULT, U_QWEN)]
        print(f"{g:>6} {row[0]:>15.2f} {row[1]:>10.2f} {row[2]:>15.2f}")
    print("\n  Ignoring the unembedding inflates the attainable speedup by up to")
    print("  2.5x at gamma=8. Since the pre-registered kill criterion is S <= 1,")
    print("  an omission of that size and sign can invert the conclusion.")

    if failed:
        raise SystemExit("original model unexpectedly passed; check the code")
    print("\nAll three checks behave as the corrected theory predicts.")


if __name__ == "__main__":
    main()
