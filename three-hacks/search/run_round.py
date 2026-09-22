"""
Run one search round: train + evaluate a config on N seeds, write rounds/<name>.json.

    python3 run_round.py --config '{"name":"v1", "blocks":["dense","dense","attn","dense"], ...}'
    python3 run_round.py --file configs/v1.json --seeds 0 1

Budgets are enforced here, not in the model: a candidate exceeding them is
written with "rejected_by_budget" and not scored, so 'make it bigger' can never
win a round.
"""

import argparse
import json
import os
import sys

from dts import Cfg, run_config, DTS, Tasks, n_params, measure_compute

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = os.path.join(HERE, "rounds")
BUDGET_FILE = os.path.join(ROUNDS, "_budget.json")   # overridden by --tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="inline JSON overriding Cfg defaults")
    ap.add_argument("--file", help="JSON file with the config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--set-budget", action="store_true",
                    help="record this config's params/compute as the budget reference")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="", help="scope the budget file, e.g. v2 -> _budget_v2.json")
    args = ap.parse_args()
    global BUDGET_FILE
    if args.tag:
        BUDGET_FILE = os.path.join(ROUNDS, f"_budget_{args.tag}.json")

    raw = json.loads(args.config) if args.config else json.load(open(args.file))
    base = Cfg().to_dict(); base.update(raw)
    cfg = Cfg.from_dict(base)
    os.makedirs(ROUNDS, exist_ok=True)

    # budget check before spending compute
    model = DTS(cfg); tasks = Tasks(cfg)
    P, C = n_params(model), measure_compute(model, tasks)
    if args.set_budget:
        json.dump({"params": P, "compute": C, "max_params_ratio": 1.25,
                   "max_compute_ratio": 3.5}, open(BUDGET_FILE, "w"), indent=2)
        print(f"budget reference set: params={P} compute={C:.0f}")
    elif os.path.exists(BUDGET_FILE):
        b = json.load(open(BUDGET_FILE))
        over = []
        if P > b["max_params_ratio"] * b["params"]:
            over.append(f"params {P} > {b['max_params_ratio']}x{b['params']}")
        if C > b["max_compute_ratio"] * b["compute"]:
            over.append(f"compute {C:.0f} > {b['max_compute_ratio']}x{b['compute']:.0f}")
        if over:
            out = {"config": cfg.to_dict(), "rejected_by_budget": over}
            json.dump(out, open(os.path.join(ROUNDS, cfg.name + ".json"), "w"), indent=2)
            print(f"{cfg.name}: REJECTED BY BUDGET -- " + "; ".join(over))
            return

    print(f"{cfg.name} [{cfg.axis}]: {cfg.rationale}")
    print(f"  blocks={list(cfg.blocks)} r={cfg.r} mode={cfg.time_mode} proj_k={cfg.proj_k} "
          f"ee_lambda={cfg.ee_lambda}  params={P} compute={C:.0f}")
    res = run_config(cfg, seeds=tuple(args.seeds), quiet=args.quiet)
    m = res["mean"]
    print(f"  F={m['fitness']:.3f}±{m['fitness_std']:.3f}  state={m['state']:.3f} "
          f"time={m['time']:.3f} depth={m['depth']:.3f}  "
          f"(recall {m['recall_acc']:.2f} track {m['track_score']:.2f} "
          f"compose {m['compose_acc']:.2f} S {m['speedup']:.3f})")
    json.dump(res, open(os.path.join(ROUNDS, cfg.name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
