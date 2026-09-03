"""Standing monitor for the MATH-only 3-epoch FILL run.

Prints a compact per-step table plus the aggregates that matter for the plan's open
questions, and tests the claim that lambda_grpo < 1 is driven by response length (longer
responses -> fewer sequences per dynamic micro-batch -> more chance an entire micro-batch
is dead-group tokens, whose gradient is then pure FILL).
"""

import argparse
import math

KEYS = (
    "perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb",
    "response_length/mean", "actor/entropy", "actor/grad_norm",
    "critic/score/mean", "srpo/lambda_grpo", "srpo/lambda_sdpo",
    "srpo/sdpo_sample_frac", "self_distillation/empty_target_batch",
    "srpo/fill_loss", "srpo/fill_token_cnt",
    "rescue/n_groups", "rescue/n_dead_groups", "rescue/n_all_pass_groups",
    "rescue/all_pass_group_frac", "rescue/n_forced_rollouts",
    "rescue/n_rescued_rollouts", "rescue/n_revived_groups", "rescue/n_rounds_run",
    "timing_s/step", "first_token/entropy", "first_token/top1_frac",
)


def parse(path):
    rows, vals = [], []
    for line in open(path, errors="ignore"):
        if "step:" not in line:
            continue
        s = line[line.index("step:"):]
        d = {}
        for kv in s.split(" - "):
            if ":" in kv:
                k, v = kv.rsplit(":", 1)
                d[k.strip()] = v.strip()
        if "timing_s/step" in d:
            rows.append(d)
    return rows


def num(d, k, default=float("nan")):
    try:
        return float(d[k])
    except Exception:
        return default


def pearson(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pts))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pts))
    if sx == 0 or sy == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / (sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tail", type=int, default=15, help="rows to print")
    ap.add_argument("--steps_per_epoch", type=int, default=234)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print("no step lines yet")
        return
    print("steps logged: %d (last step:%s)" % (len(rows), rows[-1].get("step")))

    hdr = ["step", "ep", "alloc", "resv", "resp", "ent", "gnorm", "score",
           "lamG", "lamS", "dead", "apF", "att", "ok", "rev", "fill_L", "s/step"]
    print(" ".join("%7s" % h for h in hdr))
    for d in rows[-args.tail:]:
        st = int(float(d.get("step", 0)))
        print(" ".join("%7s" % c for c in [
            st, (st - 1) // args.steps_per_epoch,
            "%.1f" % num(d, "perf/max_memory_allocated_gb"),
            "%.1f" % num(d, "perf/max_memory_reserved_gb"),
            "%.0f" % num(d, "response_length/mean"),
            "%.3f" % num(d, "actor/entropy"),
            "%.3f" % num(d, "actor/grad_norm"),
            "%.3f" % num(d, "critic/score/mean"),
            "%.3f" % num(d, "srpo/lambda_grpo"),
            "%.3f" % num(d, "srpo/lambda_sdpo"),
            "%.0f" % num(d, "rescue/n_dead_groups"),
            "%.2f" % num(d, "rescue/all_pass_group_frac"),
            "%.0f" % num(d, "rescue/n_forced_rollouts"),
            "%.0f" % num(d, "rescue/n_rescued_rollouts"),
            "%.0f" % num(d, "rescue/n_revived_groups"),
            "%.3f" % num(d, "srpo/fill_loss"),
            "%.0f" % num(d, "timing_s/step"),
        ]))

    tot = lambda k: sum(num(d, k, 0.0) for d in rows)
    g, dead, ap_, att, ok, rev = (tot("rescue/n_groups"), tot("rescue/n_dead_groups"),
                                  tot("rescue/n_all_pass_groups"), tot("rescue/n_forced_rollouts"),
                                  tot("rescue/n_rescued_rollouts"), tot("rescue/n_revived_groups"))
    print()
    print("groups %.0f | dead %.0f (%.2f%%) | all-pass %.0f (%.1f%%)"
          % (g, dead, 100 * dead / max(g, 1), ap_, 100 * ap_ / max(g, 1)))
    print("attempts %.0f | correct %.0f (p_hat %.3f%%) | revived %.0f/%.0f = %.1f%%"
          % (att, ok, 100 * ok / max(att, 1), rev, dead, 100 * rev / max(dead, 1)))
    for r in range(3):
        gi, wi = tot("rescue/round%d/n_groups_in" % r), tot("rescue/round%d/n_winners" % r)
        if gi:
            print("  round%d %.0f in / %.0f won = %.2f%%" % (r, gi, wi, 100 * wi / gi))

    alloc = [num(d, "perf/max_memory_allocated_gb") for d in rows]
    print()
    print("GATE memory: max allocated %.2f (must stay < 85), max reserved %.2f"
          % (max(alloc), max(num(d, "perf/max_memory_reserved_gb") for d in rows)))
    bad = [d.get("step") for d in rows
           if num(d, "srpo/lambda_sdpo", 0) != 0 or num(d, "srpo/sdpo_sample_frac", 0) != 0
           or num(d, "self_distillation/empty_target_batch", 1) != 1]
    print("GATE sdpo isolation (lambda_sdpo / sdpo_sample_frac / empty_target_batch):",
          "breached at %s" % bad[:5] if bad else "clean on all steps")

    lam = [num(d, "srpo/lambda_grpo") for d in rows]
    rl = [num(d, "response_length/mean") for d in rows]
    dg = [num(d, "rescue/n_dead_groups") for d in rows]
    sub1 = sum(1 for x in lam if x < 1.0)
    print("lambda_grpo < 1.0 on %d of %d steps (min %.3f); pure-FILL micro-batches exist there"
          % (sub1, len(lam), min(lam)))
    for name, xs in (("response_length", rl), ("n_dead_groups", dg)):
        r = pearson(xs, lam)
        if r is not None:
            print("  corr(lambda_grpo, %s) = %+.3f" % (name, r))

    ent = [num(d, "actor/entropy") for d in rows]
    print("entropy: first %.3f  last %.3f  max %.3f  (baseline all-time max 0.39)"
          % (ent[0], ent[-1], max(ent)))


if __name__ == "__main__":
    main()
