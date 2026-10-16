#!/usr/bin/env python3
"""PHASE 14 STEP 3a control table: pools sized for the 4 store-writing ranks.

    step3a_ctrl_table.py WORK

Arms, cap 48: global @ 142,213 (PHASE 9 pool; s0_* / s3a_def_c_* / p3_js*),
global @ 71,104 (c48g_*, store_partition=count, the 43.4 GiB/GPU design point)
and per_rank 4 x 17,776 (c48p_*). (d) sets are the unjittered run plus burst
jitter j = 0.05 x gap seeds 1..3. Batch is per ACTIVE scheduler (TP group):
batch_mean_cluster / 4.
Reading: sharing effect = per_rank vs global @ 71,104; capacity effect =
global @ 71,104 vs global @ 142,213. For (d) HBF, an effect only if the two
ranges do not overlap.
"""
import json
import os
import statistics
import sys

ACTIVE = 4
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def tags(arm, cfg, trace):
    if arm == "g142":
        if cfg == "b":
            return [f"s0_b_{trace}"]
        if cfg == "c":
            return [f"s3a_def_c_{trace}"]
        return [f"s0_d_{trace}"] + [f"p3_js{k}_d_{trace}" for k in (1, 2, 3)]
    prefix = "c48g" if arm == "g71" else "c48p"
    if cfg == "d":
        return [f"{prefix}_d_{trace}"] + [f"{prefix}_d_s{k}_{trace}" for k in (1, 2, 3)]
    return [f"{prefix}_{cfg}_{trace}"]


def metrics(d):
    n = d["request_count"]
    cross, same = d.get("partition_evictions_cross_rank"), d.get("partition_evictions_same_rank")
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "share": None if cross is None or arm_is_per_rank(d) else 100 * cross / (cross + same),
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "resid": d["mooncake_mean_residency_s"],
        "ssd_ev": d["mooncake_ssd_eviction_count"],
        "ranks": d.get("partition_ranks_seen"),
        "notdone": len(d["notdone"]),
    }


def arm_is_per_rank(d):
    return any(p.get("store_partition") == "per_rank" for p in d["mooncake_offload_profiles"])


def fmt(values, f="{:,.0f}"):
    if len(values) == 1:
        return f.format(values[0])
    return f"{f.format(statistics.mean(values))} [{f.format(min(values))}, {f.format(max(values))}]"


def main(work):
    ARMS = (("g142", "global @ 142,213 (PHASE 9 pool)"), ("g71", "global @ 71,104 — 43.4 GiB/GPU design point"),
            ("p71", "per_rank 4 × 17,776"))
    summary = {}
    for trace in ("conv", "ta"):
        print(f"\n### {TRACE[trace]}\n")
        print("| config | arm | N | HBF B/req (per TP shard) | PCM B/req (per TP shard) | hit rate | reuse hit blk "
              "| cross-rank share | mean PCM residency s | batch / active sched | HBF evictions | ranks seen |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            for arm, label in ARMS:
                rows = [metrics(load(work, t)) for t in tags(arm, cfg, trace)]
                assert all(r["notdone"] == 0 for r in rows)
                summary[(trace, cfg, arm)] = rows
                share = [r["share"] for r in rows if r["share"] is not None]
                ranks = {r["ranks"] for r in rows}
                print(f"| ({cfg}) | {label} | {len(rows)} | {fmt([r['hbf'] for r in rows])} | {fmt([r['pcm'] for r in rows])} "
                      f"| {fmt([r['hit'] for r in rows], '{:.4f}')} | {fmt([r['reuse'] for r in rows])} "
                      f"| {fmt(share, '{:.2f}%') if share else '—'} | {fmt([r['resid'] for r in rows], '{:.2f}')} "
                      f"| {fmt([r['batch'] for r in rows], '{:.3f}')} | {sum(r['ssd_ev'] for r in rows)} "
                      f"| {', '.join(str(x) for x in sorted(ranks, key=str))} |")
        print()
        for cfg in ("b", "c", "d"):
            g142, g71, p71 = (summary[(trace, cfg, a)] for a in ("g142", "g71", "p71"))
            m = lambda rows, k: statistics.mean(r[k] for r in rows)
            pct = lambda a, b: 100 * (a / b - 1)
            line = (f"({cfg}) {TRACE[trace]}: sharing (per_rank vs global@71k) HBF {pct(m(p71,'hbf'), m(g71,'hbf')):+.2f}%, "
                    f"PCM {pct(m(p71,'pcm'), m(g71,'pcm')):+.2f}%, hit {pct(m(p71,'hit'), m(g71,'hit')):+.2f}%, "
                    f"batch {pct(m(p71,'batch'), m(g71,'batch')):+.2f}% | capacity (global@71k vs global@142k) "
                    f"HBF {pct(m(g71,'hbf'), m(g142,'hbf')):+.2f}%, PCM {pct(m(g71,'pcm'), m(g142,'pcm')):+.2f}%, "
                    f"hit {pct(m(g71,'hit'), m(g142,'hit')):+.2f}%, batch {pct(m(g71,'batch'), m(g142,'batch')):+.2f}%")
            if cfg == "d":
                rng = lambda rows: (min(r["hbf"] for r in rows), max(r["hbf"] for r in rows))
                ov = lambda a, b: max(a[0], b[0]) <= min(a[1], b[1])
                line += (f" | (d) HBF ranges: g142 {rng(g142)[0]:,.0f}-{rng(g142)[1]:,.0f}, g71 {rng(g71)[0]:,.0f}-{rng(g71)[1]:,.0f}, "
                         f"p71 {rng(p71)[0]:,.0f}-{rng(p71)[1]:,.0f}; sharing ranges overlap: {'yes' if ov(rng(p71), rng(g71)) else 'NO'}; "
                         f"capacity ranges overlap: {'yes' if ov(rng(g71), rng(g142)) else 'NO'}")
            print(line)
        for arm, label in ARMS:
            b = statistics.mean(r["hbf"] for r in summary[(trace, "b", arm)])
            c = statistics.mean(r["hbf"] for r in summary[(trace, "c", arm)])
            ds = [r["hbf"] for r in summary[(trace, "d", arm)]]
            bb = statistics.mean(r["batch"] for r in summary[(trace, "b", arm)])
            db = statistics.mean(r["batch"] for r in summary[(trace, "d", arm)])
            print(f"{label}: (b)->(c) {100 * (1 - c / b):.2f}%; (b)->(d) {100 * (1 - max(ds) / b):.2f}–{100 * (1 - min(ds) / b):.2f}% "
                  f"(b ÷ d ×{b / max(ds):.1f}–×{b / min(ds):.1f}); batch gap (d) vs (b) {100 * (db / bb - 1):+.2f}%, "
                  f"(c) vs (b) {100 * (statistics.mean(r['batch'] for r in summary[(trace, 'c', arm)]) / bb - 1):+.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
