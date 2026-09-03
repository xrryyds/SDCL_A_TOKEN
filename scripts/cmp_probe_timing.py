import sys


def parse(path, limit):
    rows = []
    for line in open(path, errors="ignore"):
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
        if len(rows) >= limit:
            break
    return rows


def col(rows, key):
    out = []
    for d in rows:
        try:
            out.append(float(d[key]))
        except Exception:
            pass
    return out


for path in sys.argv[1:]:
    rows = parse(path, 10)
    if not rows:
        print(path, "no step lines")
        continue
    gen = col(rows, "timing_s/gen")
    step = col(rows, "timing_s/step")
    rl = col(rows, "response_length/mean")
    ua = col(rows, "timing_s/update_actor")
    ma = col(rows, "perf/max_memory_allocated_gb")
    mr = col(rows, "perf/max_memory_reserved_gb")
    print(path)
    print("  n=%d  gen %.1f  update_actor %.1f  step %.1f  resp_len %.0f"
          % (len(gen), sum(gen) / len(gen), sum(ua) / len(ua), sum(step) / len(step), sum(rl) / len(rl)))
    print("  gen per step:", [round(x, 1) for x in gen])
    print("  max_alloc %.2f  max_reserved %.2f" % (max(ma), max(mr)))
