"""
Experiment 3b -- where fixed state breaks (Theorem 4.4).

A sliding window of w tokens is the cheapest honest proxy for a fixed-size
recurrent state: it caps how much of the past the model can condition on, with
no training and no architecture surgery. If Theorem 4.4 describes something
real, associative recall should survive exactly while the needed span fits in
the window and then fall off a CLIFF -- not degrade gently.

Task: k key->value pairs, then "key_j ->" and the model must produce value_j.
The distance from the query back to pair j is the controlled variable.

PREDICTION: accuracy ~1 for distance < w, ~chance for distance > w, with the
transition sharp (a couple of tokens wide), because the bound is a counting
argument and counting arguments do not degrade gracefully.
KILL CRITERION: if accuracy degrades smoothly and the break-point does not move
with w, the window is not behaving like a state bound and this proxy is wrong.

Note: passing a 4D additive attention mask is version-sensitive; verified shape
is (batch, 1, q_len, kv_len). If your transformers build ignores it, the w=inf
and w=8 rows will be identical -- that is the sanity check, not a result.

    python3 exp3b_recall_capacity.py --model Qwen/Qwen3-0.6B
"""

import argparse
import random
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(rng, n_pairs, query_idx):
    words = ["".join(rng.choice(string.ascii_lowercase) for _ in range(5))
             for _ in range(2 * n_pairs)]
    keys, vals = words[:n_pairs], words[n_pairs:]
    lines = [f"{k} = {v}" for k, v in zip(keys, vals)]
    prompt = "\n".join(lines) + f"\n{keys[query_idx]} ="
    return prompt, vals[query_idx]


def window_mask(seq_len, w, device, dtype):
    """Causal mask additionally restricted to the last w tokens."""
    idx = torch.arange(seq_len, device=device)
    allowed = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64, 128, 256, 0],
                    help="0 means unrestricted (full KV cache)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device).eval()
    dtype = next(model.parameters()).dtype

    # bucket query distance (in tokens) into bands so the cliff is visible
    bands = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 10**6)]
    print(f"{args.model}, {args.pairs} pairs, {args.trials} trials per cell\n")
    header = "window  " + "".join(f"{f'd<{hi}':>10}" for _, hi in bands[:-1]) + f"{'d>=256':>10}"
    print(header)
    print("-" * len(header))

    for w in args.windows:
        hits = [[0, 0] for _ in bands]
        rng = random.Random(0)
        for _ in range(args.trials):
            for qi in (0, args.pairs // 4, args.pairs // 2, 3 * args.pairs // 4, args.pairs - 1):
                prompt, gold = make_prompt(rng, args.pairs, qi)
                ids = tok(prompt, return_tensors="pt").to(device)
                seq_len = ids.input_ids.shape[1]

                # distance in tokens from the end back to where pair qi was stated
                head = "\n".join(prompt.splitlines()[:qi])
                dist = seq_len - len(tok(head).input_ids)

                kwargs = dict(input_ids=ids.input_ids)
                if w:
                    kwargs["attention_mask"] = window_mask(seq_len, w, device, dtype)
                out = model(**kwargs)
                pred = tok.decode(out.logits[0, -1].argmax()).strip()
                ok = gold.startswith(pred) and len(pred) > 0

                for b, (lo, hi) in enumerate(bands):
                    if lo <= dist < hi:
                        hits[b][0] += int(ok); hits[b][1] += 1
                        break

        cells = "".join(f"{(h/t if t else float('nan')):>10.2f}" for h, t in hits)
        print(f"{(w or 'full'):>6}  {cells}")

    print("\nRead down each column: accuracy should stay high while the band's")
    print("distance fits inside the window and collapse once it does not.")
    print("A smooth decay instead of a cliff falsifies the proxy, not the theorem.")


if __name__ == "__main__":
    main()
