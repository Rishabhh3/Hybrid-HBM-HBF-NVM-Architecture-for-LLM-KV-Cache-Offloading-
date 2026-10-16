#!/usr/bin/env python3
"""PHASE 14 STEP 3 table: (d) concurrency sweep with burst-jitter replicates.

    step3_table.py WORK

(d): WORK/runs/s3_c<cap>_s<k>_d_<trace>, k = 1..3, burst jitter j = 0.05 x min
inter-burst gap. (b): the stock PHASE 9 STEP 4 rows (findings.md ~3644-3668),
copied below, except Tool&Agent caps 192 and 256, which that sweep did not run:
cap 256 is s1_b256_ta (reproduces PHASE 9 STEP 3 revised) and cap 192 is
s3_b192_ta, both stock and unjittered.

Pre-registered reading: "batch drives (d) HBF" only if the per-cap means move
in one direction across caps AND the lowest-to-highest-cap change in mean
exceeds the largest per-cap spread. Otherwise "not shown".
"""
import json
import os
import statistics
import sys

CAPS = [8, 16, 32, 48, 64, 96, 192, 256]
SEEDS = [1, 2, 3]
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}
# (b) HBF B/req and achieved batch/sched from PHASE 9 STEP 4 (batch to 2 dp as
# recorded there; cap 48 exact from s0_b_*).
B_STOCK = {
    "conv": {
        8: (3377115693, 3.94), 16: (3370987815, 7.83), 32: (3383093625, 15.46),
        48: (3370675339, 23.167889), 64: (3376704651, 30.81), 96: (3385003082, 42.16),
        192: (3387654930, 43.61), 256: (3387654930, 43.61),
    },
    "ta": {
        8: (1696587055, 3.90), 16: (1697241367, 7.75), 32: (1695311987, 15.29),
        48: (1696787333, 22.885048), 64: (1697208861, 30.38), 96: (1678879752, 43.03),
    },
}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def b_row(work, trace, cap):
    if cap in B_STOCK[trace]:
        return B_STOCK[trace][cap] + ("PHASE 9 STEP 4",)
    tag = {192: "s3_b192_ta", 256: "s1_b256_ta"}[cap]
    d = load(work, tag)
    return d["mooncake_ssd_write_bytes"] / d["request_count"], d["batch_mean_per_scheduler"], tag


def main(work):
    for trace in ("conv", "ta"):
        print(f"\n### {TRACE[trace]} — (d), j = 0.05 x min gap, 3 seeds per cap\n")
        print("| cap | (d) HBF B/req mean [min, max] | spread | (d) PCM B/req mean [min, max] | hit rate [min, max] "
              "| (d) batch/sched mean [min, max] | (b) HBF B/req | (b) batch | (b)->(d) reduction | batch gap (d) vs (b) | (b) source |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        means, spreads = [], []
        for cap in CAPS:
            runs = [load(work, f"s3_c{cap}_s{k}_d_{trace}") for k in SEEDS]
            h = [r["mooncake_ssd_write_bytes"] / r["request_count"] for r in runs]
            p = [r["mooncake_memory_write_bytes"] / r["request_count"] for r in runs]
            hit = [r["prefix_cache_hit_rate"] for r in runs]
            bat = [r["batch_mean_per_scheduler"] for r in runs]
            assert all(len(r["notdone"]) == 0 for r in runs)
            m = statistics.mean(h)
            spread = 100 * (max(h) - min(h)) / m
            means.append(m)
            spreads.append(spread)
            bh, bb, src = b_row(work, trace, cap)
            red = [100 * (1 - x / bh) for x in h]
            print(f"| {cap} | {m:,.0f} [{min(h):,.0f}, {max(h):,.0f}] | {spread:.2f}% "
                  f"| {statistics.mean(p):,.0f} [{min(p):,.0f}, {max(p):,.0f}] | [{min(hit):.4f}, {max(hit):.4f}] "
                  f"| {statistics.mean(bat):.3f} [{min(bat):.3f}, {max(bat):.3f}] | {bh:,.0f} | {bb:.2f} "
                  f"| {min(red):.2f}–{max(red):.2f}% | {100 * (statistics.mean(bat) / bb - 1):+.2f}% | {src} |")
        steps = [b - a for a, b in zip(means, means[1:])]
        monotone = all(s > 0 for s in steps) or all(s < 0 for s in steps)
        change = 100 * (means[-1] / means[0] - 1)
        verdict = "batch drives (d) HBF" if monotone and abs(change) > max(spreads) else "not shown"
        print(f"\nmeans move in one direction: {'yes' if monotone else 'no'} "
              f"(signs of cap-to-cap steps: {''.join('+' if s > 0 else '-' for s in steps)}); "
              f"cap 8 -> cap 256 change in mean: {change:+.2f}%; largest per-cap spread: {max(spreads):.2f}% "
              f"(cap {CAPS[spreads.index(max(spreads))]}); extreme means: min {min(means):,.0f} (cap {CAPS[means.index(min(means))]}), "
              f"max {max(means):,.0f} (cap {CAPS[means.index(max(means))]}) -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
