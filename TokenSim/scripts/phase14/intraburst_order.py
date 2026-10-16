#!/usr/bin/env python3
"""How often request-mode jitter puts a continuation before its predecessor.

    intraburst_order.py WORK

For each trace (W = 5,000, 30 QPS) and jitter_seed 1..5 at j = 0.5 ms in
arrival_jitter_mode "request", replays the loader exactly as benchmark.py does
and counts continuation pairs (the PHASE 12 STEP C rule on the raw 512-token
hash ids: R' continues R when R'.ids start with R.ids[:-1], first such later
R') whose continuation arrives before its predecessor in replay order (stable
sort by arrival time, as util/request.py does). Seed 0 / j = 0 is the stock
order.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TokenSim.workload.loaders import load_workload  # noqa: E402

W = 5000


def pairs(raw_path):
    rows = [json.loads(line) for _, line in zip(range(W), open(raw_path))]
    out = []
    for i, r in enumerate(rows):
        ids = r["hash_ids"]
        if len(ids) < 2:
            continue
        stem = ids[:-1]
        for j in range(i + 1, len(rows)):
            if rows[j]["hash_ids"][: len(stem)] == stem:
                out.append((i, j, rows[j]["timestamp"] == r["timestamp"]))
                break
    return out


def replay_positions(built_path, jitter_ms, seed):
    args = SimpleNamespace(
        workload_type="qwen_jsonl", dataset_path=built_path, request_count=W,
        dataset_skip_count=0, trace_timestamp_scale=None, trace_target_qps=30.0,
        random_seed=0, arrival_jitter_ms=jitter_ms, jitter_seed=seed,
        arrival_jitter_mode="request",
    )
    reqs = load_workload(args, SimpleNamespace(moe_config=SimpleNamespace(num_experts=0)))
    order = sorted(range(len(reqs)), key=lambda i: reqs[i].arrival_time)
    pos = [0] * len(reqs)
    for rank, i in enumerate(order):
        pos[i] = rank
    return pos


work = sys.argv[1]
for name, raw, built in (
    ("Conversation", "conversation_trace.jsonl", "conv_5000.jsonl"),
    ("Tool&Agent", "toolagent_trace.jsonl", "toolagent_5000.jsonl"),
):
    p = pairs(os.path.join(work, raw))
    same = sum(1 for _, _, s in p if s)
    cells = []
    for seed, jitter in [(0, 0.0)] + [(k, 0.5) for k in range(1, 6)]:
        pos = replay_positions(os.path.join(work, built), jitter, seed)
        before = sum(1 for i, j, _ in p if pos[j] < pos[i])
        cells.append(f"seed {seed}: {before} ({100 * before / len(p):.1f}%)")
    print(f"{name}: pairs {len(p)}, same burst {same} ({100 * same / len(p):.1f}%); "
          "continuation before predecessor -> " + "; ".join(cells))
