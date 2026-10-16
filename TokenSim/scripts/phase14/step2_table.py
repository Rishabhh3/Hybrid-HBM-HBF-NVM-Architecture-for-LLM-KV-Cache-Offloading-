#!/usr/bin/env python3
"""PHASE 14 STEP 2 table: write volume per arm against the charged reference.

    step2_table.py WORK

Reads WORK/runs/s2_<arm>_<cfg>_<trace>/result_30.0.json. Every arm is compared
with s2_ref_<cfg>_<trace> (charged timing, counters on). Pre-registered rule:
HBF and memory-tier bytes/request within 5% of the reference in every arm, for
both configs and both traces, means timing does not contaminate write volume.
"""
import json
import os
import sys

ARMS = ["ref", "free", "ssd", "plat025", "plat4", "pbw025", "pbw4"]
LABEL = {
    "ref": "charged (reference)",
    "free": "store_timing_mode=free",
    "ssd": "HBF read 100 us / 7.0 GiB/s (stock SSD default, not HBF)",
    "plat025": "pcm_write_latency_us 0.25 (0.25x, sensitivity)",
    "plat4": "pcm_write_latency_us 4.0 (4x, sensitivity)",
    "pbw025": "pcm_write_bw_gbps 5.0 (0.25x, sensitivity)",
    "pbw4": "pcm_write_bw_gbps 80.0 (4x, sensitivity)",
}
THRESHOLD = 5.0


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def row(d):
    n = d["request_count"]
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "batch": d["batch_mean_per_scheduler"],
        "skip_req": d.get("save_skip_requests"),
        "skip_blk": d.get("save_skip_missing_blocks"),
        "save_wait": d["mooncake_save_wait_time"],
        "notdone": len(d["notdone"]),
    }


def pct(new, old):
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return 100.0 * (new / old - 1.0)


def main(work):
    worst = 0.0
    for cfg in ("b", "d"):
        for trace in ("conv", "ta"):
            ref = row(load(work, f"s2_ref_{cfg}_{trace}"))
            print(f"\n### ({cfg}) {'Conversation' if trace == 'conv' else 'Tool&Agent'}\n")
            print(
                "| arm | HBF B/req | Δ | mem-tier B/req | Δ | hit rate | reuse hit blk "
                "| batch/sched | Δ batch | saves skipped (req / missing blk) | save_wait s |"
            )
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for arm in ARMS:
                r = row(load(work, f"s2_{arm}_{cfg}_{trace}"))
                dh, dp, db = pct(r["hbf"], ref["hbf"]), pct(r["pcm"], ref["pcm"]), pct(r["batch"], ref["batch"])
                if arm != "ref":
                    worst = max(worst, abs(dh), abs(dp))
                print(
                    f"| {LABEL[arm]} | {r['hbf']:,.0f} | {dh:+.2f}% | {r['pcm']:,.0f} | {dp:+.2f}% "
                    f"| {r['hit']:.4f} | {r['reuse']:,} | {r['batch']:.4f} | {db:+.2f}% "
                    f"| {r['skip_req']:,} / {r['skip_blk']:,} | {r['save_wait']:.1f} |"
                    + ("" if r["notdone"] == 0 else f" notdone={r['notdone']}")
                )
    verdict = "PASS (<5%)" if worst < THRESHOLD else "FAIL (>=5%)"
    print(f"\nLargest |Δ| in HBF or memory-tier B/req over all non-reference arms: {worst:.3f}% -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
