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
import re


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
    p.add_argument("--slots_per_round", type=int, default=8, help="rescue slots per round (== rollout.n - n_keep)")
    p.add_argument(
        "--min_coverage",
        type=float,
        default=0.0,
        help="require the token in the top-k of at least this fraction of prompts. prob_mean in the "
        "report is conditional on appearing in the top-k, so a proper noun from one word problem "
        "scores like a generic opening; coverage is what separates them.",
    )
    p.add_argument("--alpha_only", dest="alpha_only", action="store_true", default=True)
    p.add_argument("--no_alpha_only", dest="alpha_only", action="store_false")
    args = p.parse_args()

    dist = json.load(open(args.dist_path))
    n_prompts = dist["n_prompts"]
    ranked = sorted(dist["frequent_tokens"], key=lambda t: -t["prob_mean"])
    skipped = ranked[: args.skip_top]

    word_re = re.compile(r"^[A-Za-z]+$")
    chosen, rejected, seen_lower = [], [], set()
    for t in ranked[args.skip_top :]:
        if len(chosen) >= args.take:
            break
        text = t["token_text"]
        coverage = t["appear_in_topk"] / n_prompts
        if args.alpha_only and not word_re.match(text):
            rejected.append((text, "not a bare word"))
            continue
        if len(text) < 2:
            rejected.append((text, "single character"))
            continue
        if coverage < args.min_coverage:
            rejected.append((text, f"prompt-specific (in top-k for only {coverage:.1%} of prompts)"))
            continue
        if text.lower() in seen_lower:
            rejected.append((text, "case variant of an earlier token"))
            continue
        seen_lower.add(text.lower())
        chosen.append(t)

    target_nll = -math.log(0.05)
    tokens = [
        {
            "token_id": t["token_id"],
            "token_text": t["token_text"],
            "prob_mean": t["prob_mean"],
            "nll": -math.log(t["prob_mean"]) if t["prob_mean"] > 0 else float("inf"),
        }
        for t in chosen
    ]
    n_rounds = math.ceil(len(tokens) / args.slots_per_round) if tokens else 0
    payload = {
        "source_dist": args.dist_path,
        "skipped_top": [{"token_text": t["token_text"], "prob_mean": t["prob_mean"]} for t in skipped],
        "unique_tokens": len(tokens),
        "slots_per_round": args.slots_per_round,
        "n_rounds": n_rounds,
        "tokens": tokens,
    }
    with open(args.out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out_path}")
    print(f"skipped top-{args.skip_top}: {[t['token_text'] for t in skipped]}")
    print(f"kept {len(tokens)} of {args.take} requested -> n_rounds={n_rounds} at {args.slots_per_round} slots/round")
    if rejected:
        print(f"\nfiltered out ({len(rejected)}):")
        for text, why in rejected:
            print(f"  {text!r:<14} {why}")
    print(f"\n{'token':<12}{'p':<12}{'-log p':<9}{'lift to 0.05 (nats)':<20}")
    for t in tokens:
        print(f"{t['token_text']!r:<12}{t['prob_mean']:<12.3e}{t['nll']:<9.2f}{max(0.0, t['nll'] - target_nll):<20.2f}")


if __name__ == "__main__":
    main()
