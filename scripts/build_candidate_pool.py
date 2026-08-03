"""Build a first-token candidate pool from a measured distribution.

The all-fail group rescue forces alternatives to the model's own top choices, so
the candidates are taken from the measured distribution *below* the dominant
ranks: they are in-distribution (cheap to lift) yet different from what the model
would pick on its own.

Input is the report written by ``measure_first_token_distribution.py``.
"""

import argparse
import json
import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dist_path",
        default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_distribution_chemistry.json",
    )
    p.add_argument(
        "--out_path",
        default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_chemistry.json",
    )
    p.add_argument("--skip_top", type=int, default=2, help="skip the dominant ranks the model already uses")
    p.add_argument("--take", type=int, default=6, help="how many candidates to keep")
    args = p.parse_args()

    dist = json.load(open(args.dist_path))
    ranked = sorted(dist["frequent_tokens"], key=lambda t: -t["prob_mean"])
    skipped = ranked[: args.skip_top]
    chosen = ranked[args.skip_top : args.skip_top + args.take]

    target_nll = -math.log(0.05)
    tokens = [
        {
            "token_id": t["token_id"],
            "token_text": t["token_text"],
            "prob_mean": t["prob_mean"],
            "nll": -math.log(t["prob_mean"]),
        }
        for t in chosen
    ]
    payload = {
        "source_dist": args.dist_path,
        "skipped_top": [{"token_text": t["token_text"], "prob_mean": t["prob_mean"]} for t in skipped],
        "unique_tokens": len(tokens),
        "tokens": tokens,
    }
    with open(args.out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out_path}")
    print(f"skipped top-{args.skip_top}: {[t['token_text'] for t in skipped]}")
    print(f"\n{'token':<12}{'p':<12}{'-log p':<9}{'lift to 0.05 (nats)':<20}")
    for t in tokens:
        print(f"{t['token_text']!r:<12}{t['prob_mean']:<12.3e}{t['nll']:<9.2f}{max(0.0, t['nll'] - target_nll):<20.2f}")


if __name__ == "__main__":
    main()
