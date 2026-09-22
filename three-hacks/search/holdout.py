"""
Held-out comparison: re-train named rounds' configs on FRESH seeds and score them.

The search picked its incumbent on seeds 0 and 1. A winner chosen on those seeds
may have been lucky on those seeds. This re-trains each named config on seeds
the search never saw and reports mean +- std, so the final claim rests on data
the selection did not touch.

    python3 holdout.py v2_r14_naive v2_r19_depth_state ... --seeds 5 6 7
"""

import argparse
import json
import os

from dts import Cfg, run

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", nargs="+", help="round names under rounds/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--out", default="rounds/_holdout.json")
    args = ap.parse_args()

    results = {}
    print(f"held-out seeds {args.seeds}\n")
    print(f"{'round':<22} {'F':>13} {'state':>7} {'time':>7} {'depth':>7} {'recall':>7} "
          f"{'track':>7} {'compose':>8} {'S':>7}")
    print("-" * 96)
    for name in args.rounds:
        cfg = Cfg.from_dict(json.load(open(os.path.join(HERE, "rounds", name + ".json")))["config"])
        per = [run(cfg, s, quiet=True) for s in args.seeds]
        keys = ["fitness", "state", "time", "depth", "recall_acc", "track_score", "compose_acc", "speedup"]
        mean = {k: sum(p[k] for p in per) / len(per) for k in keys}
        std = {k: (sum((p[k] - mean[k]) ** 2 for p in per) / max(len(per) - 1, 1)) ** 0.5 for k in keys}
        results[name] = {"config": cfg.to_dict(), "seeds": per, "mean": mean, "std": std}
        print(f"{name:<22} {mean['fitness']:>6.3f}±{std['fitness']:<5.3f} {mean['state']:>7.3f} "
              f"{mean['time']:>7.3f} {mean['depth']:>7.3f} {mean['recall_acc']:>7.3f} "
              f"{mean['track_score']:>7.3f} {mean['compose_acc']:>8.3f} {mean['speedup']:>7.3f}",
              flush=True)
    json.dump(results, open(os.path.join(HERE, args.out), "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
