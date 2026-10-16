#!/usr/bin/env python3
"""PHASE 14 STEP 2b table: arrival-jitter replicates.

    step2b_table.py WORK [PREFIX]

Reads WORK/runs/<PREFIX>_j<k>_c<cap>_<cfg>_<trace>/result_30.0.json (k = 1..5;
PREFIX s2b = request-mode jitter, s2c = burst-mode jitter; default s2b) and,
for the comparison questions, the STEP 2 arms s2_<arm>_d_<trace> and the
unjittered cap-48 / cap-256 runs s0_d_<trace> / s1_d256_<trace>.
"""
import json
import os
import statistics
import sys

SEEDS = range(1, 6)
STEP2_ARMS = ["ref", "free", "ssd", "plat025", "plat4", "pbw025", "pbw4"]
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def metrics(d):
    n = d["request_count"]
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "hbf_blk": d["mooncake_ssd_write_blocks"],
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "batch": d["batch_mean_per_scheduler"],
        "notdone": len(d["notdone"]),
    }


PREFIX = "s2b"


def replicates(work, cap, cfg, trace):
    return [metrics(load(work, f"{PREFIX}_j{k}_c{cap}_{cfg}_{trace}")) for k in SEEDS]


def summary(values):
    return statistics.mean(values), min(values), max(values)


def spread_pct(values):
    mean = statistics.mean(values)
    return 100.0 * (max(values) - min(values)) / mean


def main(work):
    print("| cap | config | trace | HBF B/req mean [min, max] | spread | PCM B/req mean [min, max] | spread "
          "| hit rate mean [min, max] | batch/sched mean [min, max] | notdone |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    reps = {}
    for cap in (48, 256):
        for cfg in ("b", "d"):
            for trace in ("conv", "ta"):
                r = replicates(work, cap, cfg, trace)
                reps[(cap, cfg, trace)] = r
                cells = []
                for key, fmt in (("hbf", "{:,.0f}"), ("pcm", "{:,.0f}")):
                    m, lo, hi = summary([x[key] for x in r])
                    cells.append(f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]")
                    cells.append(f"{spread_pct([x[key] for x in r]):.2f}%")
                m, lo, hi = summary([x["hit"] for x in r])
                cells.append(f"{m:.4f} [{lo:.4f}, {hi:.4f}]")
                m, lo, hi = summary([x["batch"] for x in r])
                cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
                cells.append(str(sum(x["notdone"] for x in r)))
                print(f"| {cap} | ({cfg}) | {TRACE[trace]} | " + " | ".join(cells) + " |")

    print("\nQ1. (d) HBF B/req: cap-48 replicate range vs the STEP 2 arm range")
    print("| trace | STEP 2 arms [min, max] | replicates [min, max] | replicates + unjittered ref [min, max] | replicates + unjittered ref cover STEP 2? |")
    print("|---|---|---|---|---|")
    for trace in ("conv", "ta"):
        arms = [metrics(load(work, f"s2_{a}_d_{trace}"))["hbf"] for a in STEP2_ARMS]
        rep = [x["hbf"] for x in reps[(48, "d", trace)]]
        ref = metrics(load(work, f"s0_d_{trace}"))["hbf"]
        both = rep + [ref]
        covers = min(both) <= min(arms) and max(both) >= max(arms)
        print(f"| {TRACE[trace]} | {min(arms):,.0f}, {max(arms):,.0f} | {min(rep):,.0f}, {max(rep):,.0f} "
              f"| {min(both):,.0f}, {max(both):,.0f} | {'yes' if covers else 'no'} |")

    print("\nQ2. (d) cap-256 vs cap-48 gap against the replicate spread at each cap")
    print("| trace | single runs (s1 vs s0) | replicate means, cap 256 vs cap 48 | cap-48 replicate range | cap-256 replicate range | ranges overlap? | spread cap 48 / cap 256 |")
    print("|---|---|---|---|---|---|---|")
    for trace in ("conv", "ta"):
        s0 = metrics(load(work, f"s0_d_{trace}"))["hbf"]
        s1 = metrics(load(work, f"s1_d256_{trace}"))["hbf"]
        r48 = [x["hbf"] for x in reps[(48, "d", trace)]]
        r256 = [x["hbf"] for x in reps[(256, "d", trace)]]
        gap = 100.0 * (statistics.mean(r256) / statistics.mean(r48) - 1.0)
        overlap = max(min(r48), min(r256)) <= min(max(r48), max(r256))
        print(f"| {TRACE[trace]} | {100.0 * (s1 / s0 - 1.0):+.2f}% | {gap:+.2f}% "
              f"| {min(r48):,.0f} – {max(r48):,.0f} | {min(r256):,.0f} – {max(r256):,.0f} "
              f"| {'yes' if overlap else 'no'} | {spread_pct(r48):.2f}% / {spread_pct(r256):.2f}% |")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        PREFIX = sys.argv[2]
    main(sys.argv[1])
