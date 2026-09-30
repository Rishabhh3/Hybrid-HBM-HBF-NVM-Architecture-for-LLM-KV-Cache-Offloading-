#!/usr/bin/env python3
"""PHASE 12 STEP 1 (re-do of STEP C) -- continuations measured in TOKENS.

Same match rule as STEP C, on the RAW 512-token hash ids: request R with >= 2
hash ids has stem R.hash_ids[:-1]; a later request R' continues R when
R'.hash_ids[:len(stem)] == stem. The last id is excluded because it usually
covers a partial block that keeps growing as R generates.

For each (R, R') this reports delta = R'.input_length - R.input_length against
R.output_length. See the findings entry for the assumption this does NOT
establish: the trace cannot show that delta *begins with* R's output.
"""
import json, statistics, sys
from collections import defaultdict


def load(path, count=None):
    rows = []
    with open(path) as f:
        for line in f:
            if count is not None and len(rows) >= count:
                break
            rows.append(json.loads(line))
    return rows


def pairs(rows):
    by_prefix = defaultdict(list)
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        for n in range(1, len(h) + 1):
            by_prefix[tuple(h[:n])].append(i)
    out = []
    eligible = 0
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        if len(h) < 2:
            continue
        eligible += 1
        later = [j for j in by_prefix.get(tuple(h[:-1]), ()) if j > i]
        if later:
            out.append((r, rows[min(later)]))
    return eligible, out


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


def report(label, rows):
    eligible, ps = pairs(rows)
    print(f"\n  {label}: {len(rows):,} requests, {eligible:,} eligible, "
          f"{len(ps):,} continued ({len(ps)/eligible*100:.2f}%)")
    if not ps:
        return
    zero_out = [(r, rr) for r, rr in ps if r["output_length"] == 0]
    scored = [(r, rr) for r, rr in ps if r["output_length"] > 0]
    ratios = [(rr["input_length"] - r["input_length"]) / r["output_length"]
              for r, rr in scored]
    deltas = [rr["input_length"] - r["input_length"] for r, rr in scored]
    ge = sum(1 for r, rr in scored
             if rr["input_length"] - r["input_length"] >= r["output_length"])
    lt = len(scored) - ge
    neg = sum(1 for d in deltas if d < 0)
    print(f"    R.output_length == 0 (ratio undefined, excluded): {len(zero_out):,}")
    print(f"    scored continuations: {len(scored):,}")
    print(f"    delta = R'.input_length - R.input_length (tokens): "
          f"p10 {pct(deltas,0.10):,}  p50 {pct(deltas,0.50):,}  "
          f"p90 {pct(deltas,0.90):,}  mean {statistics.mean(deltas):,.1f}")
    print(f"    delta / R.output_length: "
          f"p10 {pct(ratios,0.10):.3f}  p50 {pct(ratios,0.50):.3f}  "
          f"p90 {pct(ratios,0.90):.3f}  mean {statistics.mean(ratios):.3f}")
    print(f"    delta >= R.output_length: {ge:,} / {len(scored):,} = "
          f"{ge/len(scored)*100:.2f}%")
    print(f"    delta <  R.output_length: {lt:,} / {len(scored):,} = "
          f"{lt/len(scored)*100:.2f}%   (of which delta < 0: {neg:,} = "
          f"{neg/len(scored)*100:.2f}%)")


for name, src in (("Conversation", "conversation_trace.jsonl"),
                  ("Tool&Agent", "toolagent_trace.jsonl")):
    full = load(f"{sys.argv[1]}/{src}")
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    report("W = 5,000 window", full[:5000])
    report("full trace", full)
