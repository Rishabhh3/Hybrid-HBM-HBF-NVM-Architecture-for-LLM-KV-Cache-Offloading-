#!/usr/bin/env python3
"""Rebuild the TokenSim Mooncake workload files from the FAST25 release traces.

Reconstructed from the pipeline description recorded in findings.md:2276-2281
(the original build_trace.py is not on this machine):

    first N requests, timestamps ms->s, each 512-token hash id expanded to 32
    sub-ids and truncated to ceil(input_length/16), cache_salt unchanged

The raw FAST25 traces carry no cache_salt field, so none is emitted. cache_salt
reaches the simulator only through make_extra_hash (prefix_cache.py:16-27), which
salts every key in the run identically, so a constant salt and an absent one give
identical block/byte/hit counts -- only the literal key strings differ.

Sub-id expansion is h -> [32h .. 32h+31]: a 512-token chunk covers exactly 32
blocks at block_size 16, and the mapping preserves the prefix property, so a
shared leading raw id stays a shared leading sub-id run.

--share_partial_tail (default OFF) is a WORKLOAD VARIANT, not a correction; see
PHASE 13 in findings.md. Default OFF reproduces the original files byte for byte.
"""
import argparse
import json
import math
from collections import defaultdict

BLOCK = 16
CHUNK = 512
SUBS_PER_CHUNK = CHUNK // BLOCK  # 32


def read_rows(src, count):
    rows = []
    with open(src) as handle:
        for line in handle:
            if count is not None and len(rows) >= count:
                break
            rows.append(json.loads(line))
    return rows


def expand(row):
    """The raw 512-token chunk ids as 16-token sub-ids, truncated to the prompt."""
    n_blocks = math.ceil(row["input_length"] / BLOCK)
    return [
        SUBS_PER_CHUNK * h + j
        for h in row["hash_ids"]
        for j in range(SUBS_PER_CHUNK)
    ][:n_blocks]


def share_partial_tail(rows, subs):
    """Give each R' its predecessor R's sub-ids across R's last 512-token chunk.

    Matching rule, as in PHASE 12: R' continues R when R' starts with all of R's
    raw hash ids except the last, and R has at least two raw ids (a single-id R
    has an empty stem, which every request trivially matches). Among the
    qualifying predecessors the longest match wins; ties go to the most recent.

    Only pairs with R'.input_length >= R.input_length are touched, so a
    continuation whose prompt shrank -- 2.15% of pairs on Conversation, 14.41%
    on Tool&Agent -- is never rewritten.

    Copying stops at floor(R.input_length / 16): R's own trailing partial block
    is not shared, because only the tokens R actually holds can be claimed to
    match. Positions beyond that keep R's own sub-ids.

    This is an ASSUMPTION about token content the trace cannot show, not a fix.
    """
    by_stem = defaultdict(list)
    for index, row in enumerate(rows):
        ids = row["hash_ids"]
        if len(ids) >= 2:
            by_stem[tuple(ids[:-1])].append(index)

    changed_requests = 0
    changed_subids = 0
    for j, later in enumerate(rows):
        ids = later["hash_ids"]
        chosen = None
        for stem_len in range(len(ids), 0, -1):
            candidates = [
                i
                for i in by_stem.get(tuple(ids[:stem_len]), ())
                if i < j and rows[i]["input_length"] <= later["input_length"]
            ]
            if candidates:
                chosen = max(candidates)  # ties -> most recent
                break
        if chosen is None:
            continue
        earlier = rows[chosen]
        start = SUBS_PER_CHUNK * (len(earlier["hash_ids"]) - 1)
        stop = min(earlier["input_length"] // BLOCK, len(subs[j]), len(subs[chosen]))
        moved = 0
        for k in range(start, stop):
            if subs[j][k] != subs[chosen][k]:
                subs[j][k] = subs[chosen][k]
                moved += 1
        if moved:
            changed_requests += 1
            changed_subids += moved
    return changed_requests, changed_subids


def build(src, dst, count, share_tail=False):
    rows = read_rows(src, count)
    subs = [expand(row) for row in rows]
    changed_requests = changed_subids = 0
    if share_tail:
        changed_requests, changed_subids = share_partial_tail(rows, subs)
    with open(dst, "w") as out:
        for row, sub in zip(rows, subs):
            out.write(
                json.dumps(
                    {
                        "timestamp": row["timestamp"] / 1000.0,
                        "input_length": row["input_length"],
                        "output_length": row["output_length"],
                        "hash_ids": sub,
                    }
                )
                + "\n"
            )
    return len(rows), changed_requests, changed_subids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("count", type=int)
    parser.add_argument(
        "--share_partial_tail",
        action="store_true",
        help=(
            "Workload variant: give each continuation its predecessor's sub-ids "
            "across the predecessor's last 512-token chunk. Default off, which "
            "reproduces the original files byte for byte."
        ),
    )
    args = parser.parse_args()
    n, reqs, ids = build(args.src, args.dst, args.count, args.share_partial_tail)
    print(f"{args.dst}: {n} requests")
    if args.share_partial_tail:
        print(f"  tail-sharing: {reqs:,} requests changed, {ids:,} sub-ids rewritten")
