"""
Experiment 3 -- the HMM identity, and the bridge that does not exist.

Dependency-free (stdlib only) so it runs anywhere. Two independent parts:

PART A (Theorem: gated dense recurrence == HMM forward algorithm)
  Verified against brute-force enumeration over all M**T hidden paths, PER
  (t, j), not merely on aggregates of the final state. An earlier version of
  this script compared a nested loop against its own list-comprehension
  rewrite, which is circular and cannot fail; the brute-force oracle below is
  the only non-circular check here, so it is the one that carries the claim.
  Includes a NEGATIVE CONTROL: drop non-negativity and confirm the Viterbi
  (max-product) reading breaks, since (max, x) distributes only over R_{>=0}.

PART B (Theorem: no embedding of the forward algorithm into gated linear attention)
  The load-bearing negative result. Gated linear attention updates the state as
      Phi(S) = Diag(a) S + C            (diagonal action, additive input)
  while the HMM forward algorithm updates it as
      Psi_o(s) = Diag(b(o)) A^T s       (DENSE action, multiplicative input)
  These are not the same object and neither embeds in the other. Two checks:

    B1. Support monotonicity. Phi(S) - Phi(S') = Diag(a)(S - S'), so the set of
        rows on which two trajectories differ can only SHRINK. Under Psi it
        GROWS: one differing coordinate spreads to all of them in one step.
    B2. Simultaneous diagonalisability. Conjugacy by any injective linear map
        would force every Psi_o to be diagonal in one fixed basis, hence to
        commute. They do not.

  Either check alone refutes the bridge.

    python3 exp3_rnn_is_hmm.py
"""

import random
from itertools import product

M = 4       # hidden states
NOBS = 3    # observation alphabet
T = 6       # sequence length (brute force is M**T)
TOL = 1e-9


# ------------------------------------------------------------ tiny linalg

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def hadamard(x, y):
    return [a * b for a, b in zip(x, y)]


def rand_stochastic(rows, cols, rng):
    out = []
    for _ in range(rows):
        r = [rng.random() + 1e-3 for _ in range(cols)]
        s = sum(r)
        out.append([v / s for v in r])
    return out


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


# ------------------------------------------------- textbook HMM forward

def hmm_forward(pi, A, B, obs):
    """Classic nested-loop forward algorithm, alpha[t][j] = P(o_0..o_t, z_t=j)."""
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


# ------------------------------------- gated dense recurrence (state form)

def gated_recurrence(pi, A, B, obs, semiring="sum"):
    """s_1 = pi (*) b(o_1);  s_t = (A^T s_{t-1}) (*) b(o_t).

    Written as a state update using matvec/transpose primitives: transport the
    state through a dense transition, then apply a multiplicative gate. Note
    the base case applies the gate WITHOUT a transition -- initialising with
    s_0 = pi and transitioning at t=1 would apply one spurious extra step.
    """
    At = transpose(A)
    s = hadamard(pi, [B[j][obs[0]] for j in range(M)])
    traj = [s]
    for t in range(1, len(obs)):
        gate = [B[j][obs[t]] for j in range(M)]
        if semiring == "sum":
            transported = matvec(At, s)
        else:
            transported = [max(A[i][j] * s[i] for i in range(M)) for j in range(M)]
        s = hadamard(transported, gate)
        traj.append(s)
    return traj


# ------------------------------------------------- brute-force oracle

def brute_force_alpha(pi, A, B, obs, t, j):
    """P(o_0..o_t, z_t = j) by enumerating every hidden path. Independent of
    both implementations above."""
    total = 0.0
    for path in product(range(M), repeat=t + 1):
        if path[t] != j:
            continue
        p = pi[path[0]] * B[path[0]][obs[0]]
        for k in range(1, t + 1):
            p *= A[path[k - 1]][path[k]] * B[path[k]][obs[k]]
        total += p
    return total


def brute_force_best(pi, A, B, obs, t, j):
    """max over hidden paths ending in state j at time t."""
    best = 0.0
    for path in product(range(M), repeat=t + 1):
        if path[t] != j:
            continue
        p = pi[path[0]] * B[path[0]][obs[0]]
        for k in range(1, t + 1):
            p *= A[path[k - 1]][path[k]] * B[path[k]][obs[k]]
        best = max(best, p)
    return best


# =========================================================== PART A

def part_a(rng):
    print("PART A -- gated dense recurrence == HMM forward algorithm")
    failures = 0
    for trial in range(10):
        pi_raw = [rng.random() + 1e-3 for _ in range(M)]
        pi = [v / sum(pi_raw) for v in pi_raw]
        A = rand_stochastic(M, M, rng)
        B = rand_stochastic(M, NOBS, rng)
        obs = [rng.randrange(NOBS) for _ in range(T)]

        alpha = hmm_forward(pi, A, B, obs)
        traj = gated_recurrence(pi, A, B, obs, "sum")
        vit = gated_recurrence(pi, A, B, obs, "max")

        # the non-circular check: every (t, j) against brute force
        for t in range(T):
            for j in range(M):
                oracle = brute_force_alpha(pi, A, B, obs, t, j)
                if not close(traj[t][j], oracle):
                    print(f"  FAIL[{trial}] sum-product t={t} j={j}: "
                          f"{traj[t][j]:.12g} vs oracle {oracle:.12g}")
                    failures += 1
                if not close(alpha[t][j], oracle):
                    print(f"  FAIL[{trial}] textbook t={t} j={j}")
                    failures += 1
                best = brute_force_best(pi, A, B, obs, t, j)
                if not close(vit[t][j], best):
                    print(f"  FAIL[{trial}] max-product t={t} j={j}: "
                          f"{vit[t][j]:.12g} vs oracle {best:.12g}")
                    failures += 1

    print(f"  sum-product and max-product match brute force at every (t, j): "
          f"{'OK' if failures == 0 else str(failures) + ' FAILURES'}")

    # --- negative control: non-negativity is load-bearing for Viterbi only ---
    pi = [0.25] * M
    A = [[(-1.0) ** (i + j) * (0.2 + 0.1 * ((i + j) % 3)) for j in range(M)]
         for i in range(M)]                                  # signed transition
    B = rand_stochastic(M, NOBS, rng)
    obs = [rng.randrange(NOBS) for _ in range(T)]
    sum_ok = all(
        close(gated_recurrence(pi, A, B, obs, "sum")[t][j],
              brute_force_alpha(pi, A, B, obs, t, j))
        for t in range(T) for j in range(M))
    max_ok = all(
        close(gated_recurrence(pi, A, B, obs, "max")[t][j],
              brute_force_best(pi, A, B, obs, t, j))
        for t in range(T) for j in range(M))
    print(f"  negative control, signed A: sum-product still exact = {sum_ok} "
          f"(expected True -- the identity is pure algebra)")
    print(f"                              max-product exact      = {max_ok} "
          f"(expected False -- (max,x) needs non-negativity)")
    return failures == 0 and sum_ok and not max_ok


# =========================================================== PART B

def part_b(rng):
    print("\nPART B -- the bridge from gated linear attention to the HMM")

    # --- B1: support monotonicity -------------------------------------------
    d_k, d_v = M, 2
    a = [rng.random() for _ in range(d_k)]                 # diagonal gate
    C = [[rng.random() for _ in range(d_v)] for _ in range(d_k)]   # input term
    S = [[rng.random() for _ in range(d_v)] for _ in range(d_k)]
    Sp = [row[:] for row in S]
    Sp[1] = [v + 1.0 for v in Sp[1]]                        # differ in row 1 only

    def phi(X):
        return [[a[i] * X[i][j] + C[i][j] for j in range(d_v)] for i in range(d_k)]

    def rowsupp(X, Y):
        return {i for i in range(len(X))
                if any(abs(X[i][j] - Y[i][j]) > TOL for j in range(len(X[0])))}

    before = rowsupp(S, Sp)
    after = rowsupp(phi(S), phi(Sp))
    gla_ok = after <= before
    print(f"  B1 gated linear attention: differing rows {sorted(before)} "
          f"-> {sorted(after)}  (non-increasing = {gla_ok})")

    A = rand_stochastic(M, M, rng)
    B = rand_stochastic(M, NOBS, rng)
    At = transpose(A)
    s = [0.0] * M
    sp = [0.0] * M
    sp[1] = 1.0                                             # differ in coord 1

    def psi(x, o):
        return hadamard(matvec(At, x), [B[j][o] for j in range(M)])

    supp_before = {i for i in range(M) if abs(s[i] - sp[i]) > TOL}
    d_after = [psi(s, 0)[j] - psi(sp, 0)[j] for j in range(M)]
    supp_after = {j for j in range(M) if abs(d_after[j]) > TOL}
    hmm_grows = supp_after > supp_before
    print(f"  B1 HMM forward algorithm : differing coords {sorted(supp_before)} "
          f"-> {sorted(supp_after)}  (strictly grows = {hmm_grows})")

    # --- B2: the Psi_o do not commute ---------------------------------------
    Psi = []
    for o in range(2):
        g = [B[j][o] for j in range(M)]
        Psi.append([[g[i] * At[i][j] for j in range(M)] for i in range(M)])
    P01, P10 = matmul(Psi[0], Psi[1]), matmul(Psi[1], Psi[0])
    comm = max(abs(P01[i][j] - P10[i][j]) for i in range(M) for j in range(M))
    noncommuting = comm > 1e-6
    print(f"  B2 ||Psi_0 Psi_1 - Psi_1 Psi_0||_max = {comm:.3e} "
          f"(non-commuting = {noncommuting})")

    ok = gla_ok and hmm_grows and noncommuting
    print("\n  => A diagonal gate cannot mix state coordinates; a transition")
    print("     matrix must. No injective linear map carries one recurrence")
    print("     into the other, so 'constrain a gated linear RNN and get an")
    print("     HMM' is false. The transition has to be ADDED, not constrained")
    print("     out of a gate that is already diagonal.")
    return ok


def main():
    rng = random.Random(0)
    a_ok = part_a(rng)
    b_ok = part_b(rng)
    print("\n" + "=" * 68)
    if a_ok and b_ok:
        print("Part A verified (identity holds, against an independent oracle).")
        print("Part B verified (the bridge to linear attention does not exist).")
    else:
        print(f"UNEXPECTED: part A ok={a_ok}, part B ok={b_ok}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
