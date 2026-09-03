import sys

LOG = sys.argv[1]

rows = []
for line in open(LOG, errors="ignore"):
    if "step:" not in line:
        continue
    s = line[line.index("step:"):]
    d = {}
    for kv in s.split(" - "):
        if ":" in kv:
            k, v = kv.rsplit(":", 1)
            d[k.strip()] = v.strip()
    if "timing_s/gen" in d:
        rows.append(d)


def g(d, k, default=None):
    try:
        return float(d[k])
    except Exception:
        return default


print("steps parsed:", len(rows))
hdr = ("step", "alloc", "resv", "gen", "upd", "step_s", "resp", "groups", "dead", "allpass",
       "att", "ok", "rev", "rounds", "ent", "gnorm", "lam", "fill_loss")
print(" ".join("%8s" % h for h in hdr))
missing = []
for d in rows:
    st = d.get("step", "?")
    for k in ("rescue/all_pass_group_frac", "rescue/n_all_pass_groups"):
        if k not in d:
            missing.append((st, k))
    print(("%5s" + "%8.2f" * 5 + "%7.0f" + "%7.0f" * 2 + "%9.4f" + "%6.0f" * 3 + "%7.0f" + "%7.4f" + "%6.1f" + "%9.4f") % (
        st,
        g(d, "perf/max_memory_allocated_gb", 0), g(d, "perf/max_memory_reserved_gb", 0),
        g(d, "timing_s/gen", 0), g(d, "timing_s/update_actor", 0), g(d, "timing_s/step", 0),
        g(d, "response_length/mean", 0),
        g(d, "rescue/n_groups", 0), g(d, "rescue/n_dead_groups", 0),
        g(d, "rescue/all_pass_group_frac", -1),
        g(d, "rescue/n_forced_rollouts", 0), g(d, "rescue/n_forced_correct", 0),
        g(d, "rescue/n_revived_groups", 0), g(d, "rescue/n_rounds_run", 0),
        g(d, "actor/entropy", 0), g(d, "actor/grad_norm", 0), g(d, "srpo/lambda_grpo", -1),
        g(d, "srpo/fill_loss", 0),
    ))

alloc = [g(d, "perf/max_memory_allocated_gb", 0) for d in rows]
resv = [g(d, "perf/max_memory_reserved_gb", 0) for d in rows]
ap = [g(d, "rescue/all_pass_group_frac", 0) for d in rows]
gr = [g(d, "rescue/n_groups", 0) for d in rows]
apn = [g(d, "rescue/n_all_pass_groups", 0) for d in rows]
dead = [g(d, "rescue/n_dead_groups", 0) for d in rows]
att = [g(d, "rescue/n_forced_rollouts", 0) for d in rows]
ok = [g(d, "rescue/n_forced_correct", 0) for d in rows]
rev = [g(d, "rescue/n_revived_groups", 0) for d in rows]
print()
print("GATE max_memory_allocated_gb  max %.2f" % max(alloc))
print("GATE max_memory_reserved_gb   max %.2f  (must be < 75)" % max(resv))
print("GATE all_pass keys missing:", missing or "none")
print("all_pass: %.0f of %.0f groups = %.4f" % (sum(apn), sum(gr), sum(apn) / max(sum(gr), 1)))
print("dead: %.0f  attempts %.0f  correct %.0f  revived %.0f" % (sum(dead), sum(att), sum(ok), sum(rev)))
if sum(att):
    print("  p_hat = %.4f%%   revival = %.1f%%" % (100 * sum(ok) / sum(att), 100 * sum(rev) / max(sum(dead), 1)))
for r in range(4):
    gi = sum(g(d, "rescue/round%d/n_groups_in" % r, 0) for d in rows)
    wi = sum(g(d, "rescue/round%d/n_winners" % r, 0) for d in rows)
    if gi:
        print("  round %d: %.0f in / %.0f won = %.2f%%" % (r, gi, wi, 100 * wi / gi))
