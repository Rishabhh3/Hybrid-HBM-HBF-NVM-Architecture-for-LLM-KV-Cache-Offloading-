#!/usr/bin/env python3
"""PHASE 13 STEP 2 -- how much of a predecessor's prompt a continuation cannot reach.

The FAST25 traces hash at 512 tokens; the simulator blocks at 16. The match rule
excludes R's last raw id (a partial chunk that grows), so R and R' agree only on
sub-indices [0, s) with s = 32*(len(R.raw)-1), while R holds p = R.input//16 full
prompt blocks. Where s < p, R's blocks in [s, p) carry sub-ids R' will never ask
for -- and since both plan_reuse (kv_cache_manager.py:58-64) and store.lookup
(store.py:79-93) stop at the first miss, everything after them is unreachable too.

Trace analysis only.
"""
import json, sys
from collections import defaultdict


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * q)))]


for name, src in (("Conversation", "conversation_trace.jsonl"),
                  ("Tool&Agent", "toolagent_trace.jsonl")):
    rows = []
    for line in open(f"{sys.argv[1]}/{src}"):
        if len(rows) >= 5000:
            break
        rows.append(json.loads(line))

    idx = defaultdict(list)
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        for n in range(1, len(h) + 1):
            idx[tuple(h[:n])].append(i)

    total_prompt_blocks = sum(r["input_length"] // 16 for r in rows)
    distinct = set()
    for r in rows:
        n = r["input_length"] // 16
        distinct.update(
            [32 * h + j for h in r["hash_ids"] for j in range(32)][:n]
        )

    pairs = affected = 0
    tails = []
    for i, r in enumerate(rows):
        h = r["hash_ids"]
        if len(h) < 2:
            continue
        if not [j for j in idx.get(tuple(h[:-1]), ()) if j > i]:
            continue
        pairs += 1
        s = 32 * (len(h) - 1)
        p = r["input_length"] // 16
        if s < p:
            affected += 1
            tails.append(p - s)

    print(f"\n{name} (W = 5,000)")
    print(f"  continuation pairs: {pairs:,}")
    print(f"  affected (chains diverge before R's last full prompt block): "
          f"{affected:,} = {affected/pairs*100:.1f}%")
    print(f"  unreachable prompt blocks per affected pair: "
          f"p50 {pct(tails,0.5)}  p90 {pct(tails,0.9)}  max {max(tails)}  "
          f"total {sum(tails):,}")
    print(f"  as % of all prompt-block instances in the window "
          f"({total_prompt_blocks:,}): {sum(tails)/total_prompt_blocks*100:.2f}%")
    print(f"  as % of distinct prompt blocks ({len(distinct):,}): "
          f"{sum(tails)/len(distinct)*100:.2f}%")
