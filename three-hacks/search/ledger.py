"""
Turn rounds/*.json into the ledger table: F, delta vs the incumbent, verdict.

    python3 ledger.py                       # table over all rounds, in order
    python3 ledger.py --incumbent r00_naive # deltas against a named round
    python3 ledger.py --markdown >> LEDGER.md

tau is read from rounds/_tau.json, written once from the baseline's std.
Acceptance: mean F(cand) - mean F(inc) > tau. Ties and sub-tau gains are
'no change', which is a verdict, not a failure.
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = os.path.join(HERE, "rounds")


def load(tag=""):
    rows = []
    pat = f"{tag}_r*.json" if tag else "r*.json"
    for p in sorted(glob.glob(os.path.join(ROUNDS, pat))):
        r = json.load(open(p))
        r["_name"] = os.path.basename(p)[:-5]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", help="round name to diff against; default: running best")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--set-tau-from", help="round name whose F std becomes tau")
    ap.add_argument("--tag", default="", help="scope tau and the round glob, e.g. v2")
    args = ap.parse_args()

    tau_file = os.path.join(ROUNDS, f"_tau_{args.tag}.json" if args.tag else "_tau.json")
    if args.set_tau_from:
        r = json.load(open(os.path.join(ROUNDS, args.set_tau_from + ".json")))
        tau = r["mean"]["fitness_std"]
        json.dump({"tau": tau, "from": args.set_tau_from}, open(tau_file, "w"), indent=2)
        print(f"tau = {tau:.4f} (from {args.set_tau_from})")
    tau = json.load(open(tau_file))["tau"] if os.path.exists(tau_file) else None

    rows = load(args.tag)
    inc = None
    if args.incumbent:
        inc = next(r for r in rows if r["_name"] == args.incumbent)
    fmt = ("| {name} | {axis} | {mut} | {F} | {d} | {v} |" if args.markdown
           else "{name:<14} {axis:<9} {F:>7} {d:>8}  {v:<10} {mut}")
    if not args.markdown:
        print(f"{'round':<14} {'axis':<9} {'F':>7} {'dF':>8}  {'verdict':<10} mutation"
              f"   (tau={tau if tau is None else round(tau, 3)})")
    best = None
    for r in rows:
        cfg = r["config"]
        if "rejected_by_budget" in r:
            print(fmt.format(name=r["_name"], axis=cfg["axis"], mut=cfg["rationale"][:60],
                             F="--", d="--", v="over budget"))
            continue
        F = r["mean"]["fitness"]
        ref = inc if inc is not None else best
        if ref is None:
            d, v = 0.0, "baseline"
        else:
            d = F - ref["mean"]["fitness"]
            v = ("ACCEPT" if (tau is not None and d > tau) else
                 "no change" if (tau is None or abs(d) <= tau) else "worse")
        print(fmt.format(name=r["_name"], axis=cfg["axis"], mut=cfg["rationale"][:60],
                         F=f"{F:.3f}±{r['mean']['fitness_std']:.3f}", d=f"{d:+.3f}", v=v))
        if inc is None and (best is None or F - best["mean"]["fitness"] > (tau or 0)):
            best = r
    if inc is None and best is not None and not args.markdown:
        m = best["mean"]
        print(f"\nincumbent: {best['_name']}  F={m['fitness']:.3f}  "
              f"state={m['state']:.3f} time={m['time']:.3f} depth={m['depth']:.3f}")


if __name__ == "__main__":
    main()
