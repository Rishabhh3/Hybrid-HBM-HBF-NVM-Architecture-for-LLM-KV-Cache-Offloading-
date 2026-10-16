#!/usr/bin/env python3
"""PHASE 14 set A: headline sensitivity to store_lookup_start = after_hbm.

    setA_table.py WORK

Copy-at-admission (stock prefill save), 71,104 per_rank, cap 48. Each la_* run
is paired with the headline run of the same config/jitter/seed (hl_*).
Rule: the headline holds under the vLLM-style lookup if (b), (c) HBF and all
PCM B/req are within 5% of the headline, and every (d) run lies in the
per-trace band (Conv 5.65-12.29 M, T&A 3.88-6.67 M).
"""
import json
import os
import sys

ACTIVE = 4
BAND = {"conv": (5.65e6, 12.29e6), "ta": (3.88e6, 6.67e6)}
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def m(d):
    n = d["request_count"]
    w = d.get("hbf_readback_written_blocks")
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "per_reuse": d["mooncake_ssd_write_bytes"] / d["reuse_hit_blocks"],
        "lookups_hit": d["mooncake_store_hit_count"],
        "lookups": d["mooncake_store_hit_count"] + d["mooncake_store_miss_count"],
        "disk_tok": d["mooncake_disk_hit_tokens"],
        "rb": None if not w else 100 * d["hbf_readback_read_blocks"] / w,
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "notdone": len(d["notdone"]),
    }


def rng(v, f):
    return f.format(v[0]) if min(v) == max(v) else f"{f.format(min(v))} – {f.format(max(v))}"


def pct(a, b):
    return 100 * (a / b - 1)


def main(work):
    for tr in ("conv", "ta"):
        print(f"\n### {TRACE[tr]}\n")
        rows = {}
        for cfg in ("b", "c", "d"):
            sufs = ["", "_j05_s1", "_j05_s2", "_j05_s3"] if cfg == "d" else [""]
            rows[cfg] = [(m(load(work, f"la_{cfg}{s}_{tr}")), m(load(work, f"hl_{cfg}{s}_{tr}"))) for s in sufs]
            assert all(a["notdone"] == 0 for a, _ in rows[cfg])
        print("| config | N | store lookups hit / lookups | headline | store-served tokens | headline | reuse-hit blk | headline "
              "| HBF B/req | Δ vs headline | PCM B/req | Δ vs headline | hit rate | Δ | batch / active sched | Δ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            r = rows[cfg]
            A = [a for a, _ in r]
            H = [b for _, b in r]
            print(f"| ({cfg}) | {len(r)} | {rng([x['lookups_hit'] for x in A], '{:,}')} / {rng([x['lookups'] for x in A], '{:,}')} "
                  f"| {rng([x['lookups_hit'] for x in H], '{:,}')} / {rng([x['lookups'] for x in H], '{:,}')} "
                  f"| {rng([x['disk_tok'] for x in A], '{:,}')} | {rng([x['disk_tok'] for x in H], '{:,}')} "
                  f"| {rng([x['reuse'] for x in A], '{:,}')} | {rng([x['reuse'] for x in H], '{:,}')} "
                  f"| {rng([x['hbf'] for x in A], '{:,.0f}')} | {rng(sorted(pct(a['hbf'], b['hbf']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['pcm'] for x in A], '{:,.0f}')} | {rng(sorted(pct(a['pcm'], b['pcm']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['hit'] for x in A], '{:.4f}')} | {rng(sorted(pct(a['hit'], b['hit']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['batch'] for x in A], '{:.3f}')} | {rng(sorted(pct(a['batch'], b['batch']) for a, b in r), '{:+.2f}%')} |")
        b, c = rows["b"][0][0], rows["c"][0][0]
        D = [a for a, _ in rows["d"]]
        print(f"\n(b)->(c) {100 * (1 - c['hbf'] / b['hbf']):.2f}%; (b)->(d) {min(100 * (1 - x['hbf'] / b['hbf']) for x in D):.2f}–"
              f"{max(100 * (1 - x['hbf'] / b['hbf']) for x in D):.2f}%; (d) hit vs (b) {rng(sorted(pct(x['hit'], b['hit']) for x in D), '{:+.2f}%')}; "
              f"HBF bytes per reuse-hit block (b) {b['per_reuse']:,.0f} / (c) {c['per_reuse']:,.0f} / (d) {rng(sorted(x['per_reuse'] for x in D), '{:,.0f}')}; "
              f"read-back share (b) {b['rb']:.2f}% / (c) {c['rb']:.2f}% / (d) "
              + (rng(sorted(x['rb'] for x in D if x['rb'] is not None), '{:.2f}%') if any(x['rb'] is not None for x in D) else "— (nothing written to HBF)"))
        ok_ii = all(abs(pct(a["hbf"], h["hbf"])) < 5 for cfg in "bc" for a, h in rows[cfg]) and \
            all(abs(pct(a["pcm"], h["pcm"])) < 5 for cfg in "bcd" for a, h in rows[cfg])
        worst = max([abs(pct(a["hbf"], h["hbf"])) for cfg in "bc" for a, h in rows[cfg]] +
                    [abs(pct(a["pcm"], h["pcm"])) for cfg in "bcd" for a, h in rows[cfg]])
        lo, hi = BAND[tr]
        inside = sum(1 for x in D if lo <= x["hbf"] <= hi)
        print(f"RULE A1 (b),(c) HBF and all PCM within 5%: {'PASS' if ok_ii else 'FAIL'} (largest |Δ| {worst:.2f}%)")
        print(f"RULE A2 every (d) run in band {lo / 1e6:.2f}–{hi / 1e6:.2f} M: {inside}/{len(D)} -> {'PASS' if inside == len(D) else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1])
