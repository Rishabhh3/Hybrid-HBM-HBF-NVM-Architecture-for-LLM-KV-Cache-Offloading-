#!/usr/bin/env python3
"""PHASE 12 STEP C -- option-2 feasibility, trace analysis only.

Works on the RAW FAST25 512-token hash ids, not the expanded 16-token sub-ids.

For each request R with >= 2 hash ids, R's "stem" is hash_ids[:-1]: the last id
usually covers a partial block that keeps growing as the request generates, so a
continuation turn shares the stem, not the whole prompt. A later request L
matches R when L.hash_ids[:len(stem)] == stem.

Reported per trace, over the W=5000 window and over the full trace:
  1. fraction of requests with at least one such later match
  2. for matched requests: time gap to the first match (p50/p90), and the number
     of extra 512-token hashes the match carries beyond the stem, against
     ceil(output_length/512) -- what the generated KV would have been worth
"""
import json, math, statistics, sys
from collections import defaultdict

def load(path, count=None):
    rows = []
    with open(path) as f:
        for line in f:
            if count is not None and len(rows) >= count:
                break
            rows.append(json.loads(line))
    return rows

def analyse(rows, ts_scale):
    """ts_scale converts the raw timestamp field to seconds."""
    # index by stem so the scan is linear, not quadratic
    by_prefix = defaultdict(list)
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        # a later request matches R's stem if its own id chain starts with it;
        # index every prefix length that could be some other request's stem
        for n in range(1, len(h) + 1):
            by_prefix[tuple(h[:n])].append(i)

    eligible = matched = 0
    gaps, extras, expected, ratios = [], [], [], []
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        if len(h) < 2:
            continue
        eligible += 1
        stem = tuple(h[:-1])
        later = [j for j in by_prefix.get(stem, ()) if j > i]
        if not later:
            continue
        matched += 1
        j = min(later)
        gaps.append((rows[j]["timestamp"] - r["timestamp"]) * ts_scale)
        extra = len(rows[j]["hash_ids"]) - len(stem)
        exp = math.ceil(r["output_length"] / 512)
        extras.append(extra)
        expected.append(exp)
        if exp:
            ratios.append(extra / exp)
    return eligible, matched, gaps, extras, expected, ratios

def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((len(xs) - 1) * q)))
    return xs[k]

def report(label, rows, ts_scale):
    eligible, matched, gaps, extras, expected, ratios = analyse(rows, ts_scale)
    print(f"\n{label}: {len(rows):,} requests, {eligible:,} with >=2 hash ids")
    frac = matched / eligible if eligible else 0.0
    print(f"  (1) requests whose stem is extended by a LATER request: "
          f"{matched:,} / {eligible:,} = {frac*100:.2f}%")
    if not matched:
        return
    print(f"  (2) gap to first such later request (s): "
          f"p50 {pct(gaps,0.50):,.1f}  p90 {pct(gaps,0.90):,.1f}  "
          f"mean {statistics.mean(gaps):,.1f}")
    print(f"      extra 512-tok hashes past the stem:  "
          f"p50 {pct(extras,0.50)}  p90 {pct(extras,0.90)}  "
          f"mean {statistics.mean(extras):.2f}")
    print(f"      ceil(output_length/512) for the same requests: "
          f"p50 {pct(expected,0.50)}  p90 {pct(expected,0.90)}  "
          f"mean {statistics.mean(expected):.2f}")
    print(f"      extra / expected ratio: p50 {pct(ratios,0.50):.2f}  "
          f"mean {statistics.mean(ratios):.2f}")

for name, src in (("Conversation", "conversation_trace.jsonl"),
                  ("Tool&Agent", "toolagent_trace.jsonl")):
    full = load(f"{sys.argv[1]}/{src}")
    print(f"\n{'='*66}\n{name}\n{'='*66}")
    report("W=5000 window", full[:5000], 1e-3)   # raw timestamps are ms
    report("full trace", full, 1e-3)
