"""
Theorem 4.3, checked numerically: a gated non-negative linear RNN *is* the HMM
forward algorithm, and its max-product variant *is* Viterbi.

Deliberately dependency-free (no numpy, no torch) so it runs anywhere, including
the container this was written in. Small dimensions so the brute-force oracle is
tractable -- the point is correctness, not speed.

Three checks:

  1. textbook HMM forward recursion  ==  gated linear RNN recursion
  2. sum of forward variables        ==  brute-force sum over all state paths
  3. max-product RNN recursion       ==  brute-force best path (Viterbi)

Check 2 is the one that matters: it pins the RNN's output to the *semantics*
(a marginal likelihood), not merely to another way of writing the same loop.

    python3 exp3_rnn_is_hmm.py
"""

import random
from itertools import product

M = 4       # hidden states
NOBS = 3    # observation alphabet
T = 6       # sequence length (brute force is M**T)
TOL = 1e-9


def rand_stochastic(rows, cols, rng):
    """Row-stochastic matrix."""
    out = []
    for _ in range(rows):
        r = [rng.random() + 1e-3 for _ in range(cols)]
        s = sum(r)
        out.append([v / s for v in r])
    return out


# ---------------------------------------------------------------- textbook HMM

def hmm_forward(pi, A, B, obs):
    """Classic nested-loop forward algorithm. alpha[t][j] = P(o_0..o_t, z_t=j)."""
    alpha = [[pi[j] * B[j][obs[0]] for j in range(M)]]
    for t in range(1, len(obs)):
        prev, cur = alpha[-1], []
        for j in range(M):
            acc = 0.0
            for i in range(M):
                acc += prev[i] * A[i][j]
            cur.append(acc * B[j][obs[t]])
        alpha.append(cur)
    return alpha


# ------------------------------------------------- gated non-negative linear RNN

def gated_rnn(pi, A, B, obs, semiring="sum"):
    """s_t = (A^T s_{t-1}) (*) b(o_t), with (*) elementwise.

    This is written as a state-update -- gate, then transition, then elementwise
    modulation -- with no reference to probability. `semiring="max"` swaps
    (+, x) for (max, x), which is the only change needed to get Viterbi.
    """
    combine = (lambda vals: sum(vals)) if semiring == "sum" else (lambda vals: max(vals))

    s = [pi[j] * B[j][obs[0]] for j in range(M)]      # initial state
    traj = [s]
    for t in range(1, len(obs)):
        gate = [B[j][obs[t]] for j in range(M)]        # data-dependent gate b(o_t)
        transported = [combine([s[i] * A[i][j] for i in range(M)]) for j in range(M)]
        s = [transported[j] * gate[j] for j in range(M)]   # elementwise gating
        traj.append(s)
    return traj


# ------------------------------------------------------------------ brute force

def brute_force(pi, A, B, obs, reduce_="sum"):
    """Enumerate every hidden path; sum or max its joint probability."""
    best, total = 0.0, 0.0
    for path in product(range(M), repeat=len(obs)):
        p = pi[path[0]] * B[path[0]][obs[0]]
        for t in range(1, len(obs)):
            p *= A[path[t - 1]][path[t]] * B[path[t]][obs[t]]
        total += p
        best = max(best, p)
    return total if reduce_ == "sum" else best


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main():
    rng = random.Random(0)
    failures = 0

    for trial in range(20):
        pi_raw = [rng.random() + 1e-3 for _ in range(M)]
        pi = [v / sum(pi_raw) for v in pi_raw]
        A = rand_stochastic(M, M, rng)          # transition
        B = rand_stochastic(M, NOBS, rng)       # emission
        obs = [rng.randrange(NOBS) for _ in range(T)]

        alpha = hmm_forward(pi, A, B, obs)
        traj = gated_rnn(pi, A, B, obs, semiring="sum")

        # 1. term-for-term identity, every timestep, every coordinate
        for t in range(T):
            for j in range(M):
                if not close(alpha[t][j], traj[t][j]):
                    print(f"  FAIL[{trial}] recursion t={t} j={j}: "
                          f"{alpha[t][j]!r} vs {traj[t][j]!r}")
                    failures += 1

        # 2. the RNN's final state really is the marginal likelihood
        if not close(sum(traj[-1]), brute_force(pi, A, B, obs, "sum")):
            print(f"  FAIL[{trial}] likelihood: {sum(traj[-1])} vs "
                  f"{brute_force(pi, A, B, obs, 'sum')}")
            failures += 1

        # 3. max-product variant is Viterbi
        vit = gated_rnn(pi, A, B, obs, semiring="max")
        if not close(max(vit[-1]), brute_force(pi, A, B, obs, "max")):
            print(f"  FAIL[{trial}] viterbi: {max(vit[-1])} vs "
                  f"{brute_force(pi, A, B, obs, 'max')}")
            failures += 1

    if failures:
        print(f"\nTheorem 4.3 NOT verified: {failures} failure(s)")
        raise SystemExit(1)

    print(f"Theorem 4.3 verified on {20} random HMMs "
          f"(M={M} states, |O|={NOBS}, T={T}):")
    print("  - gated non-negative linear RNN  ==  HMM forward algorithm (termwise)")
    print("  - its final state sums to the exact marginal likelihood")
    print("  - swapping (+,x) -> (max,x) gives exactly Viterbi")
    print("\nSo 'gated RNN with HMM-like decoding' is an identity, not an analogy.")
    print("The constraints that buy it: state >= 0, transitions row-stochastic,")
    print("gate multiplicative. Drop non-negativity and the HMM reading is gone.")


if __name__ == "__main__":
    main()
