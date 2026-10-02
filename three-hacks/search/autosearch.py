"""
Autonomous search driver for rounds 21+.

Stricter than the hand-driven loop, because it runs unattended:

  * THREE seeds per candidate (the round-19 incumbent spread +-0.224 on two).
  * tau is the INCUMBENT's own across-seed std of F, recomputed when the
    incumbent changes -- not the baseline's, which was 0.015 and would accept
    noise at this point in the search.
  * A candidate that beats the incumbent by more than tau is NOT promoted yet:
    it is re-trained on the held-out seeds and promoted only if it also beats
    the incumbent's held-out mean. Round 20 is why: +0.012 on search seeds,
    -0.100 held-out.
  * Hard deadline. A candidate is skipped if its estimated duration would
    cross it. Nothing is ever truncated.
  * Every round appends its own ledger row; the log is the audit trail.

    python3 autosearch.py --incumbent v2_r19_depth_state --menu menus/batchA.json \
        --tag v2 --round-start 21 --deadline "2026-10-03 01:00"
"""

import argparse
import json
import os
import time
from datetime import datetime

from dts import Cfg, run, DTS, Tasks, n_params, measure_compute

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = os.path.join(HERE, "rounds")
LEDGER = os.path.join(HERE, "LEDGER.md")
KEYS = ["fitness", "state", "time", "depth", "depth_raw", "recall_acc",
        "track_score", "compose_acc", "speedup"]


def log(msg):
    line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(ROUNDS, "_autosearch.log"), "a") as f:
        f.write(line + "\n")


def agg(per):
    m = {k: sum(p[k] for p in per) / len(per) for k in KEYS}
    m.update({k + "_std": (sum((p[k] - m[k]) ** 2 for p in per) / max(len(per) - 1, 1)) ** 0.5
              for k in KEYS})
    return m


def load_round(name):
    return json.load(open(os.path.join(ROUNDS, name + ".json")))


def save_round(name, rec):
    json.dump(rec, open(os.path.join(ROUNDS, name + ".json"), "w"), indent=2)


def ensure_seeds(name, seeds):
    """Make sure a round has results for every seed; run the missing ones."""
    rec = load_round(name)
    cfg = Cfg.from_dict(rec["config"])
    have = {p["seed"] for p in rec["seeds"]}
    for s in seeds:
        if s not in have:
            log(f"{name}: running missing seed {s}")
            rec["seeds"].append(run(cfg, s, quiet=True))
    rec["seeds"] = sorted(rec["seeds"], key=lambda p: p["seed"])
    rec["mean"] = agg([p for p in rec["seeds"] if p["seed"] in seeds])
    save_round(name, rec)
    return rec


def ensure_holdout(name, hseeds):
    rec = load_round(name)
    if "holdout" in rec and {p["seed"] for p in rec["holdout"]["seeds"]} >= set(hseeds):
        return rec["holdout"]
    cfg = Cfg.from_dict(rec["config"])
    log(f"{name}: held-out on seeds {hseeds}")
    per = [run(cfg, s, quiet=True) for s in hseeds]
    rec["holdout"] = {"seeds": per, "mean": agg(per)}
    save_round(name, rec)
    return rec["holdout"]


def over_budget(cfg, tag):
    bf = os.path.join(ROUNDS, f"_budget_{tag}.json" if tag else "_budget.json")
    if not os.path.exists(bf):
        return []
    b = json.load(open(bf)); m = DTS(cfg); t = Tasks(cfg)
    P, C = n_params(m), measure_compute(m, t)
    over = []
    if P > b["max_params_ratio"] * b["params"]:
        over.append(f"params {P} > {b['max_params_ratio']}x{b['params']}")
    if C > b["max_compute_ratio"] * b["compute"]:
        over.append(f"compute {C:.0f} > {b['max_compute_ratio']}x{b['compute']:.0f}")
    return over


def ledger_row(rnd, name, axis, mut, F, Fstd, d, verdict):
    with open(LEDGER, "a") as f:
        f.write(f"| {rnd} | {name} | {axis} | {mut} | {F:.3f}±{Fstd:.3f} | {d:+.3f} | {verdict} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--menu", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--round-start", type=int, required=True)
    ap.add_argument("--deadline", required=True, help="UTC, 'YYYY-MM-DD HH:MM'")
    args = ap.parse_args()
    deadline = datetime.strptime(args.deadline, "%Y-%m-%d %H:%M")

    inc_name = args.incumbent
    inc = ensure_seeds(inc_name, args.seeds)
    inc_h = ensure_holdout(inc_name, args.holdout_seeds)
    tau = inc["mean"]["fitness_std"]
    log(f"incumbent {inc_name}: F={inc['mean']['fitness']:.3f}±{tau:.3f} on seeds {args.seeds}, "
        f"held-out {inc_h['mean']['fitness']:.3f}; tau={tau:.3f}")

    menu = json.load(open(args.menu))
    base = Cfg().to_dict()
    sec_per_seed = None
    rnd = args.round_start
    for item in menu:
        d = dict(base); d.update(inc["config"]); d.update(item)
        cfg = Cfg.from_dict(d)
        name = cfg.name
        if os.path.exists(os.path.join(ROUNDS, name + ".json")):
            log(f"round {rnd} {name}: already exists, skipping"); rnd += 1; continue
        over = over_budget(cfg, args.tag)
        if over:
            log(f"round {rnd} {name}: REJECTED BY BUDGET {over}")
            save_round(name, {"config": cfg.to_dict(), "rejected_by_budget": over})
            ledger_row(rnd, name, cfg.axis, cfg.rationale[:60], 0, 0, 0, "over budget")
            rnd += 1; continue
        # deadline: estimate from the last measured seed time scaled by compute
        c_ratio = max(1.0, measure_compute(DTS(cfg), Tasks(cfg)) / max(inc["seeds"][0]["compute"], 1))
        est = (sec_per_seed or 120) * c_ratio * len(args.seeds)
        if datetime.utcnow().timestamp() + est > deadline.timestamp():
            log(f"round {rnd} {name}: would cross deadline (est {est/60:.0f} min), stopping"); break

        log(f"round {rnd} {name} [{cfg.axis}]: {cfg.rationale}")
        t0 = time.time()
        per = [run(cfg, s, quiet=True) for s in args.seeds]
        sec_per_seed = (time.time() - t0) / len(args.seeds) / c_ratio
        m = agg(per)
        rec = {"config": cfg.to_dict(), "seeds": per, "mean": m,
               "params": per[0]["params"], "compute": per[0]["compute"], "round": rnd}
        delta = m["fitness"] - inc["mean"]["fitness"]
        log(f"  F={m['fitness']:.3f}±{m['fitness_std']:.3f}  d={delta:+.3f} vs tau={tau:.3f}  "
            f"(recall {m['recall_acc']:.2f} track {m['track_score']:.2f} "
            f"compose {m['compose_acc']:.2f} S {m['speedup']:.3f})")
        if delta > tau:
            save_round(name, rec)
            h = ensure_holdout(name, args.holdout_seeds)
            hd = h["mean"]["fitness"] - inc_h["mean"]["fitness"]
            log(f"  candidate above tau; held-out {h['mean']['fitness']:.3f}±{h['mean']['fitness_std']:.3f} "
                f"vs incumbent held-out {inc_h['mean']['fitness']:.3f}  (d={hd:+.3f})")
            if hd > 0:
                verdict = f"**ACCEPT → incumbent** (held-out {h['mean']['fitness']:.3f} vs {inc_h['mean']['fitness']:.3f})"
                inc_name, inc, inc_h = name, load_round(name), h
                tau = inc["mean"]["fitness_std"]
                log(f"  PROMOTED. new tau={tau:.3f}")
            else:
                verdict = f"above τ on search seeds but **loses held-out** ({h['mean']['fitness']:.3f} vs {inc_h['mean']['fitness']:.3f}); not promoted"
        else:
            save_round(name, rec)
            verdict = "no change" if abs(delta) <= tau else "worse"
        ledger_row(rnd, name, cfg.axis, cfg.rationale[:60], m["fitness"], m["fitness_std"], delta, verdict)
        rnd += 1
    log(f"menu done. incumbent {inc_name} F={inc['mean']['fitness']:.3f} held-out {inc_h['mean']['fitness']:.3f}")
    json.dump({"incumbent": inc_name, "next_round": rnd}, open(os.path.join(ROUNDS, "_autosearch_state.json"), "w"))
    with open(os.path.join(ROUNDS, "_autosearch.log"), "a") as f:
        f.write("MENUDONE\n")


if __name__ == "__main__":
    main()
