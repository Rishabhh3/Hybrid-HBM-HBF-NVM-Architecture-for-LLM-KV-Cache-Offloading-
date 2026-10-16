#!/usr/bin/env python3
"""PHASE 14 headline set table: design capacity (71,104, per_rank), cap 48.

    headline_table.py WORK

(b), (c): hl_{b,c}_<trace>. (d): hl_d_<trace> plus hl_d_j{005,05,45}_s{1..5}_<trace>
(16 runs). Reductions for (d) are against the single (b) run. Comparison pool:
rb142_{b,c,d}_<trace> (142,213, global, unjittered, the 86.8 GiB/GPU point).
Batch is per active scheduler (TP group) = batch_mean_cluster / 4. B/req are
per TP shard.
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


def d_tags(trace):
    tags = [f"hl_d_{trace}"]
    for j in ("005", "05", "45"):
        tags += [f"hl_d_j{j}_s{k}_{trace}" for k in range(1, 6)]
    return tags


def m(d):
    n = d["request_count"]
    written = d.get("hbf_readback_written_blocks")
    read = d.get("hbf_readback_read_blocks")
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "hbf_bytes": d["mooncake_ssd_write_bytes"],
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "per_reuse": d["mooncake_ssd_write_bytes"] / d["reuse_hit_blocks"],
        "written": written,
        "read": read,
        "share": None if not written else 100 * read / written,
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "ssd_ev": d["mooncake_ssd_eviction_count"],
        "cross": d.get("partition_evictions_cross_rank"),
        "notdone": len(d["notdone"]),
    }


def rng(values, fmt):
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(min(values))} – {fmt.format(max(values))}"


def table(rows_by_cfg, b_hit, b_hbf, b_batch, title):
    print(f"\n{title}\n")
    print("| config | N | HBF B/req | (b)→ reduction | PCM B/req | hit rate | hit rate vs (b) | reuse-hit blk "
          "| HBF bytes / reuse-hit blk | HBF-written blk | read back from HBF | read-back share "
          "| batch / active sched | batch gap vs (b) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, rows in rows_by_cfg:
        red = [100 * (1 - r["hbf"] / b_hbf) for r in rows]
        rel = [100 * (r["hit"] / b_hit - 1) for r in rows]
        gap = [100 * (r["batch"] / b_batch - 1) for r in rows]
        share = [r["share"] for r in rows if r["share"] is not None]
        print(f"| ({cfg}) | {len(rows)} | {rng([r['hbf'] for r in rows], '{:,.0f}')} "
              f"| {'—' if cfg == 'b' else rng(sorted(red), '{:.2f}%')} | {rng([r['pcm'] for r in rows], '{:,.0f}')} "
              f"| {rng([r['hit'] for r in rows], '{:.4f}')} | {'—' if cfg == 'b' else rng(sorted(rel), '{:+.2f}%')} "
              f"| {rng([r['reuse'] for r in rows], '{:,}')} | {rng([r['per_reuse'] for r in rows], '{:,.0f}')} "
              f"| {rng([r['written'] for r in rows], '{:,}')} | {rng([r['read'] for r in rows], '{:,}')} "
              f"| {rng(sorted(share), '{:.2f}%') if share else '—'} | {rng([r['batch'] for r in rows], '{:.3f}')} "
              f"| {'—' if cfg == 'b' else rng(sorted(gap), '{:+.2f}%')} |")


def main(work):
    for trace in ("conv", "ta"):
        b = m(load(work, f"hl_b_{trace}"))
        c = m(load(work, f"hl_c_{trace}"))
        ds = [m(load(work, t)) for t in d_tags(trace)]
        assert all(r["notdone"] == 0 and r["ssd_ev"] == 0 and r["cross"] == 0 for r in [b, c] + ds)
        print(f"\n### {TRACE[trace]}")
        table([("b", [b]), ("c", [c]), ("d", ds)], b["hit"], b["hbf"], b["batch"],
              f"**Headline — 71,104 blocks, per_rank 4 × 17,776, 43.4 GiB/GPU** ((d): N = {len(ds)})")
        worst = min(100 * (1 - r["hbf"] / b["hbf"]) for r in ds)
        print(f"\n(d) pooled: N = {len(ds)}, HBF B/req {min(r['hbf'] for r in ds):,.0f} – {max(r['hbf'] for r in ds):,.0f}, "
              f"worst-case (b)→(d) reduction {worst:.2f}%, best {max(100 * (1 - r['hbf'] / b['hbf']) for r in ds):.2f}%, "
              f"b ÷ d ×{b['hbf'] / max(r['hbf'] for r in ds):.1f} – ×{b['hbf'] / min(r['hbf'] for r in ds):.1f}; "
              f"(b)→(c) {100 * (1 - c['hbf'] / b['hbf']):.2f}%")
        for j in ("005", "05", "45"):
            sub = [m(load(work, f"hl_d_j{j}_s{k}_{trace}")) for k in range(1, 6)]
            v = [r["hbf"] for r in sub]
            print(f"  j = {dict(zip(('005', '05', '45'), ('0.005', '0.05', '0.45')))[j]} x gap: {min(v):,.0f} – {max(v):,.0f}, "
                  f"spread {100 * (max(v) - min(v)) / statistics.mean(v):.2f}%; values {[round(x) for x in v]}")
        print(f"  unjittered: {ds[0]['hbf']:,.0f}")
        rb = [m(load(work, f"rb142_{cfg}_{trace}")) for cfg in "bcd"]
        table([("b", [rb[0]]), ("c", [rb[1]]), ("d", [rb[2]])], rb[0]["hit"], rb[0]["hbf"], rb[0]["batch"],
              "**Comparison — 142,213 blocks, global, the 86.8 GiB/GPU point** (unjittered, single runs)")


if __name__ == "__main__":
    main(sys.argv[1])
