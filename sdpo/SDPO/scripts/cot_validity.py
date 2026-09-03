#!/usr/bin/env python3
"""Raw accuracy (paper metric) + CoT validity report for SciKnowEval val dumps.

Two sources are merged:
  1. run.log            -- val-core/<ds>/acc/mean@16, i.e. the paper's avg@16,
                           plus per-step time for wall-clock budget mapping
  2. <dump_dir>/*.jsonl  -- dumped val responses (fields: output, score, pred,
                           gts, incorrect_format), classified for CoT validity

Columns:
  step        training step (0 = val_before_train, the untrained model)
  wall_h      cumulative training wall-clock hours (excludes validation pauses)
  avg@16      PAPER METRIC: mean accuracy over 16 val rollouts
  best@16     pass@16, for reference
  dump_acc    accuracy of the single dumped rollout per question (sanity check)
  n_ok        number of correct dumped responses (the CoT denominator)

Of those n_ok correct responses:
  reason%     has a substantive reasoning block (>=200 chars, >=2 sentences)
  nocontra%   reasoning does not argue against its own final answer
  deriv%      contains a load-bearing derivation: arithmetic, quantitative
              comparison, or an explicit computation/derivation verb
  nonguess%   does NOT admit guessing/inability ("cannot be determined",
              "without further context", "reasonable estimate", ...)
  valid%      all four hold -- correct AND the CoT is actually doing work

The final block maps the paper protocol: peak avg@16 within 1h / 5h / 10h.

Usage:
  python scripts/cot_validity.py <dump_dir> [run.log]
"""

import json
import re
import sys
from pathlib import Path

REASONING_RE = re.compile(r"<reasoning>(.*?)(?:</reasoning>|<answer>|\Z)", re.S | re.I)

# Reasoning that argues for / against a specific option.
EVAL_RE = re.compile(
    r"(?:option\s+)?\b([A-D])\b(?:\s*[:.)-]|\s+)?[^.;\n]{0,80}?"
    r"(?:is|are|would be|must be|should be|seems|appears|looks)\b[^.;\n]{0,60}",
    re.I,
)
NEG_RE = re.compile(
    r"\b(not|n't|never|no longer|least|incorrect|wrong|unsupported|unlikely|"
    r"implausible|eliminat\w*|exclud\w*|rule[sd]? out|inconsistent|invalid|"
    r"contradict\w*|does not|cannot|can't)\b",
    re.I,
)
CORRECT_CLAIM_RE = re.compile(
    r"\b(correct|right|best|answer|most likely|most plausible|most reasonable)\b",
    re.I,
)

DERIV_RES = [
    re.compile(r"\d+(?:\.\d+)?\s*[+\-*/×÷]\s*\d+"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b", re.I),
    re.compile(
        r"\b(?:more|less|higher|lower|greater|fewer|larger|smaller|stronger|"
        r"weaker|faster|slower|closer)\b[^.\n]{0,60}\bthan\b",
        re.I,
    ),
    re.compile(
        r"\b(calculat\w+|comput\w+|deriv\w+|subtract\w+|multipl\w+|divid\w+|"
        r"ratio\b|difference between|sum of|net change|convert\w+|"
        r"therefore\s+\d|equals?\b)\b",
        re.I,
    ),
]

# Explicit hedging. Rather than enumerate phrasings, detect the two structures
# that mark a guess on this dataset: (a) conceding the value cannot be obtained
# from the prompt alone, (b) picking an option by plausibility, not derivation.
GUESS_RES = [
    # "without X, (we) cannot / it is not possible ..."   /   "cannot be determined"
    re.compile(r"\bwithout\b[^.\n]{0,80}?\b(cannot|can't|can not|unable|"
               r"not possible|no way|impossible|difficult)\b", re.I),
    re.compile(r"\b(cannot|can't|can not|unable to|not possible to|"
               r"no way to)\b[^.\n]{0,60}?\b(determin\w+|calculat\w+|comput\w+|"
               r"know|derive|obtain|establish|verify|assess)\b", re.I),
    re.compile(r"\b(cannot be|can't be|could not be) (determined|calculated|"
               r"computed|derived|known|established|verified)\b", re.I),
    # needs an external tool / data the prompt does not supply
    re.compile(r"\b(would|will|we would|one would|needs? to be|must be)\b"
               r"[^.\n]{0,40}\b(use|run|input|consult|employ|apply)\b"
               r"[^.\n]{0,60}\b(tool|software|algorithm|program|simulation|"
               r"database|method|model)\b", re.I),
    re.compile(r"\b(such as|e\.g\.,?)\s+(FoldX|Rosetta|DSSP|BLAST|AlphaFold)",
               re.I),
    re.compile(r"\bno\b[^.\n]{0,30}\b(tool|calculation|data|information|"
               r"context|value)\b[^.\n]{0,40}\b(provided|given|available|"
               r"supplied)\b", re.I),
    # selection by plausibility instead of derivation
    re.compile(r"\bmost\s+(reasonable|plausible|likely|probable|sensible)\b"
               r"[^.\n]{0,30}\b(value|option|choice|answer|estimate|guess)\b",
               re.I),
    re.compile(r"\b(reasonable|educated|best|rough)\s+(estimate|guess|"
               r"approximation|assumption)\b", re.I),
    re.compile(r"\b(guess|assume|arbitrar\w+|randomly|by elimination alone)\b",
               re.I),
    re.compile(r"\bbased on\s+(typical|general|common|standard)\s+"
               r"(values|knowledge|patterns|ranges)\b", re.I),
]


def is_guess(reason):
    return any(rx.search(reason) for rx in GUESS_RES)


def reasoning_of(output):
    m = REASONING_RE.search(output or "")
    return (m.group(1) if m else (output or "")).strip()


def has_contradiction(reason, answer):
    if not answer:
        return True
    argued_for, argued_against = set(), set()
    for m in EVAL_RE.finditer(reason):
        letter, window = m.group(1).upper(), m.group(0)
        if NEG_RE.search(window):
            argued_against.add(letter)
        elif CORRECT_CLAIM_RE.search(window):
            argued_for.add(letter)
    # Contradiction: the answer was explicitly argued against and never for.
    return answer in argued_against and answer not in argued_for


def classify(row):
    reason = reasoning_of(row.get("output", ""))
    answer = (row.get("pred") or "").strip().upper()[:1]
    reason_ok = len(reason) >= 200 and len(re.findall(r"[.!?](?:\s|$)", reason)) >= 2
    nocontra = not has_contradiction(reason, answer)
    deriv = any(rx.search(reason) for rx in DERIV_RES)
    nonguess = not is_guess(reason)
    return reason_ok, nocontra, deriv, nonguess


def analyze_file(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    step = rows[0].get("step")
    if step is None:
        step = int(Path(path).stem) if Path(path).stem.isdigit() else -1
    correct = [r for r in rows if (r.get("score") or 0) >= 1.0]
    res = {"step": int(step), "n": len(rows), "correct_n": len(correct),
           "dump_acc": len(correct) / len(rows)}
    keys = ("reason", "nocontra", "deriv", "nonguess", "valid")
    if not correct:
        res.update({k: 0.0 for k in keys})
        return res
    tally = dict.fromkeys(keys, 0)
    for r in correct:
        a, b, c, d = classify(r)
        for k, v in zip(keys[:4], (a, b, c, d)):
            tally[k] += int(v)
        tally["valid"] += int(a and b and c and d)
    res.update({k: tally[k] / len(correct) for k in keys})
    return res


def parse_log(log_path):
    """Return {step: (avg16, best16)}, {step: cumulative_train_seconds}."""
    val, cum, total = {}, {}, 0.0
    if not log_path or not Path(log_path).exists():
        return val, cum
    for line in open(log_path, errors="ignore"):
        if "perf/time_per_step:" in line:
            t = re.search(r"perf/time_per_step:([0-9.]+)", line)
            s = re.search(r"training/global_step:(\d+)", line)
            if t and s:
                total += float(t.group(1))
                cum[int(s.group(1))] = total
        if "acc/mean@16" in line:
            sm = re.search(r"step:(\d+) -", line)
            step = int(sm.group(1)) if sm else 0
            am = re.search(r"acc/mean@16[:'\s]*([0-9.]+)", line)
            bm = re.search(r"acc/best@16/mean[:'\s]*([0-9.]+)", line)
            if am:
                val[step] = (float(am.group(1)),
                             float(bm.group(1)) if bm else None)
    return val, cum


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_gen_bio-grpo")
    log_path = sys.argv[2] if len(sys.argv) > 2 else None
    files = [target] if target.is_file() else sorted(
        target.glob("*.jsonl"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not files:
        sys.exit(f"no jsonl dumps under {target}")
    val, cum = parse_log(log_path)

    hdr = (f"{'step':>5s} {'wall_h':>6s} {'test_acc':>8s} {'pass@16':>7s} "
           f"{'peak':>6s} {'n_ok':>5s} {'reason%':>7s} {'nocon%':>7s} "
           f"{'deriv%':>7s} {'nongue%':>7s} {'VALID%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    running_peak = 0.0
    for f in files:
        r = analyze_file(f)
        if not r:
            continue
        avg, best = val.get(r["step"], (None, None))
        # test_acc IS paper Table 1 metric (avg@16 on held-out test.parquet).
        # Falls back to dump_acc (1 rollout per question) when log has not yet
        # printed the mean@16 line for that step.
        acc = avg if avg is not None else r["dump_acc"]
        if acc is not None and acc > running_peak:
            running_peak = acc
        t = cum.get(r["step"], 0.0)
        f7 = lambda x: f"{x*100:7.2f}" if x is not None else f"{'-':>7s}"
        f8 = lambda x: f"{x*100:8.2f}" if x is not None else f"{'-':>8s}"
        print(f"{r['step']:5d} {t/3600:6.2f} {f8(acc)} {f7(best)} "
              f"{running_peak*100:6.2f} {r['correct_n']:5d} "
              f"{r['reason']*100:7.1f} {r['nocontra']*100:7.1f} "
              f"{r['deriv']*100:7.1f} {r['nonguess']*100:7.1f} "
              f"{r['valid']*100:7.1f}")

    # Paper Table 1 reference for Qwen3-8B Biology (SciKnowEval)
    ref = {"Qwen3-8B (base)": (30.5, 30.5, 30.5),
           "+GRPO":          (46.9, 68.1, 70.6),
           "+SDPO":          (52.1, 58.5, 58.5),
           "+SRPO":          (55.8, 68.3, 72.8)}
    print("\nPaper Table 1 Biology reference (avg@16 on held-out test):")
    print(f"  {'method':<20s} {'1h':>6s} {'5h':>6s} {'10h':>6s}")
    for name, (a, b, c) in ref.items():
        print(f"  {name:<20s} {a:6.1f} {b:6.1f} {c:6.1f}")

    if val:
        print("\nOur peak test_acc within wall-clock budget:")
        for b in (1, 5, 10):
            inb = [(s, a) for s, (a, _) in val.items() if cum.get(s, 0) <= b * 3600]
            if inb:
                sb, ab = max(inb, key=lambda x: x[1])
                print(f"  {b:>2d}h: {ab*100:6.2f}  (step {sb})")
            else:
                print(f"  {b:>2d}h: not reached")


if __name__ == "__main__":
    main()
