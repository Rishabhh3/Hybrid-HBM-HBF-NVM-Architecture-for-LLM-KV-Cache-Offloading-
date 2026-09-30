#!/usr/bin/env python3
"""PHASE 13 STEP 1b -- how big is generated KV, as a share of all KV?

Every token carries the same KV bytes (size_per_token is a model/TP constant,
cache_config.py), so the token share IS the byte share. Trace analysis only.
"""
import json, statistics, sys


def stats(rows):
    ins = [r["input_length"] for r in rows]
    outs = [r["output_length"] for r in rows]
    tin, tout = sum(ins), sum(outs)
    return ins, outs, tin, tout


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


def report(label, rows):
    ins, outs, tin, tout = stats(rows)
    print(f"  {label}: {len(rows):,} requests")
    print(f"    input  tokens total {tin:>14,}   mean {statistics.mean(ins):>9,.1f}  "
          f"median {pct(ins,0.5):>8,}")
    print(f"    output tokens total {tout:>14,}   mean {statistics.mean(outs):>9,.1f}  "
          f"median {pct(outs,0.5):>8,}  p90 {pct(outs,0.9):>8,}")
    print(f"    generated share  out/(in+out) = {tout/(tin+tout)*100:.2f}%")


for name, src in (("Conversation", "conversation_trace.jsonl"),
                  ("Tool&Agent", "toolagent_trace.jsonl")):
    rows = [json.loads(l) for l in open(f"{sys.argv[1]}/{src}")]
    print(f"\n{name}")
    report("W = 5,000", rows[:5000])
    report("full trace", rows)
