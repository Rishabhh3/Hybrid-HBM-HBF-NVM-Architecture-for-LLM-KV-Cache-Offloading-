#!/usr/bin/env python3
"""PHASE 14 STEP 3a table: store_partition count / per_rank against stock.

    step3a_table.py WORK

Stock references: (b), (d) = s0_<cfg>_<trace> (ced3806); (c) = s3a_def_c_<trace>.
Reading rule: for (b), (c), and PCM B/req and hit rate in every config, single
runs are readable (replicate spread < 1%). For (d) HBF B/req under per_rank, a
shift is an effect only if it falls outside the pooled cap-48 (d) range.
"""
import json
import os
import sys

TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}
POOLED_D = {"conv": (32048677, 43025695), "ta": (17168859, 26329219)}
IGNORED = {"simulator_wall_time", "mooncake_pool_keys", "mooncake_offload_profiles"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def stock_tag(cfg, trace):
    return f"s3a_def_c_{trace}" if cfg == "c" else f"s0_{cfg}_{trace}"


def non_counter_diff(a, b):
    keys = (set(a) | set(b)) - IGNORED
    return sorted(k for k in keys if not k.startswith("partition_") and a.get(k) != b.get(k))


def main(work):
    for trace in ("conv", "ta"):
        print(f"\n### {TRACE[trace]}\n")
        print("| config | mode | HBF B/req | Δ vs stock | PCM B/req | Δ | hit rate | Δ | reuse hit blk "
              "| memory evictions | cross-rank / same-rank | cross-rank share | per-rank peak blk min / max "
              "| ssd evictions | batch/sched | batch gap vs stock |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            stock = load(work, stock_tag(cfg, trace))
            n = stock["request_count"]
            for mode, tag in (("stock", stock_tag(cfg, trace)), ("count", f"s3a_cnt_{cfg}_{trace}"),
                              ("per_rank", f"s3a_pr_{cfg}_{trace}")):
                d = load(work, tag)
                hbf, pcm = d["mooncake_ssd_write_bytes"] / n, d["mooncake_memory_write_bytes"] / n
                s_hbf, s_pcm = stock["mooncake_ssd_write_bytes"] / n, stock["mooncake_memory_write_bytes"] / n
                cross = d.get("partition_evictions_cross_rank")
                same = d.get("partition_evictions_same_rank")
                share = f"{100 * cross / (cross + same):.2f}%" if cross is not None and cross + same else "—"
                peaks = (f"{d['partition_rank_peak_blocks_min']:,} / {d['partition_rank_peak_blocks_max']:,}"
                         if "partition_rank_peak_blocks_min" in d else "—")
                pct = lambda x, y: f"{100 * (x / y - 1):+.2f}%" if y else "—"
                flag = ""
                if cfg == "d" and mode == "per_rank":
                    lo, hi = POOLED_D[trace]
                    flag = " (inside pooled range)" if lo <= hbf <= hi else " (**outside pooled range**)"
                print(f"| ({cfg}) | {mode} | {hbf:,.0f}{flag} | {pct(hbf, s_hbf)} | {pcm:,.0f} | {pct(pcm, s_pcm)} "
                      f"| {d['prefix_cache_hit_rate']:.4f} | {pct(d['prefix_cache_hit_rate'], stock['prefix_cache_hit_rate'])} "
                      f"| {d['reuse_hit_blocks']:,} | {d['mooncake_memory_eviction_count']:,} "
                      f"| {'—' if cross is None else f'{cross:,} / {same:,}'} | {share} | {peaks} "
                      f"| {d['mooncake_ssd_eviction_count']:,} | {d['batch_mean_per_scheduler']:.4f} "
                      f"| {pct(d['batch_mean_per_scheduler'], stock['batch_mean_per_scheduler'])} |")
        print()
        for cfg in ("b", "c", "d"):
            diff = non_counter_diff(load(work, stock_tag(cfg, trace)), load(work, f"s3a_cnt_{cfg}_{trace}"))
            print(f"count vs stock, ({cfg}) {TRACE[trace]}: non-counter fields differing: {diff if diff else 'none'}")
        pr = load(work, f"s3a_pr_b_{trace}")
        print(f"per_rank capacities: memory {pr['partition_rank_memory_capacity_blocks']:,} blk/rank, "
              f"ssd {pr['partition_rank_ssd_capacity_blocks']:,} blk/rank, ranks seen {pr['partition_ranks_seen']}")


if __name__ == "__main__":
    main(sys.argv[1])
