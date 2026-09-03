"""Cross-epoch survival analysis of FILL-rescued prompts.

fill_rescued.jsonl records one row per ALL-FAIL group, every step: winners carry
rescued=true plus the winning response, still-dead groups carry rescued=false.

With total_epochs>1 every prompt is revisited, and the dump only ever contains all-fail
groups -- so a prompt's ABSENCE from a later epoch means that when the policy saw it again
it got at least 1 of 8 rollouts correct. Comparing that survival rate between epoch-1
rescued=true and rescued=false rows is a control group at zero extra cost: both arms were
all-fail at the same point in training, and only one got a FILL gradient.

Join on the prompt text. uid is a fresh uuid4 per batch and is not stable across epochs.
"""

import argparse
import json
import math
from collections import defaultdict


def two_proportion_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return None, None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None, None
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="fill_rescued.jsonl")
    ap.add_argument("--steps_per_epoch", type=int, required=True,
                    help="train rows // train_batch_size, e.g. 7498//32 = 234")
    ap.add_argument("--train_size", type=int, default=0,
                    help="prompts in the train set; used to bound the drop_last confound")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.path) if line.strip()]
    if not rows:
        print("empty dump")
        return

    for r in rows:
        r["epoch"] = (int(r["step"]) - 1) // args.steps_per_epoch

    max_step = max(int(r["step"]) for r in rows)
    n_epochs_seen = max(r["epoch"] for r in rows) + 1
    print("rows %d  steps 1..%d  epochs present %d  steps/epoch %d"
          % (len(rows), max_step, n_epochs_seen, args.steps_per_epoch))

    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[r["epoch"]].append(r)
    for e in sorted(by_epoch):
        sub = by_epoch[e]
        resc = sum(1 for r in sub if r.get("rescued"))
        print("  epoch %d: %5d dead groups, %4d rescued (%.1f%%), unique prompts %d"
              % (e, len(sub), resc, 100 * resc / len(sub), len(set(r["prompt"] for r in sub))))

    # A prompt can legitimately appear twice inside one epoch only if the dataset has
    # duplicate statements; collapse to "was it dead at all, was it ever rescued".
    dead_in = defaultdict(dict)
    for r in rows:
        cur = dead_in[r["prompt"]].get(r["epoch"])
        if cur is None:
            dead_in[r["prompt"]][r["epoch"]] = dict(rescued=bool(r.get("rescued")), round=r.get("round"))
        elif r.get("rescued"):
            cur["rescued"] = True

    print()
    for e in range(n_epochs_seen - 1):
        nxt = e + 1
        if nxt not in by_epoch:
            print("epoch %d never reached; no survival comparison possible" % nxt)
            break
        arms = {True: [0, 0], False: [0, 0]}  # rescued -> [solved_later, total]
        for prompt, per_epoch in dead_in.items():
            if e not in per_epoch:
                continue
            arm = per_epoch[e]["rescued"]
            arms[arm][1] += 1
            if nxt not in per_epoch:
                arms[arm][0] += 1
        print("epoch %d -> %d : was all-fail in epoch %d, then NOT all-fail in epoch %d"
              % (e, nxt, e, nxt))
        for arm, label in ((True, "FILL rescued  "), (False, "still dead    ")):
            k, n = arms[arm]
            if n:
                print("  %s  %4d/%4d = %5.1f%%" % (label, k, n, 100 * k / n))
            else:
                print("  %s  (no rows)" % label)
        kt, nt = arms[True]
        kf, nf = arms[False]
        z, p = two_proportion_z(kt, nt, kf, nf)
        if z is not None:
            print("  difference %+.1f pp   z = %+.2f   two-sided p = %.4f"
                  % (100 * (kt / nt - kf / nf), z, p))
        if args.train_size:
            unseen = args.train_size - args.steps_per_epoch * args.batch_size
            print("  confound: drop_last leaves %d of %d prompts (%.2f%%) unseen per epoch, so that "
                  "fraction of 'absent' is dropout rather than solved"
                  % (unseen, args.train_size, 100 * unseen / args.train_size))

    # Repeat offenders: dead in every epoch they could be dead in.
    always = [p for p, pe in dead_in.items() if len(pe) == n_epochs_seen]
    print()
    print("prompts all-fail in every one of the %d epochs: %d of %d distinct dead prompts"
          % (n_epochs_seen, len(always), len(dead_in)))
    ever_resc = sum(1 for p in always if any(v["rescued"] for v in dead_in[p].values()))
    if always:
        print("  of those, %d were FILL-rescued at least once (%.1f%%) -- a rescue that did not stick"
              % (ever_resc, 100 * ever_resc / len(always)))


if __name__ == "__main__":
    main()
