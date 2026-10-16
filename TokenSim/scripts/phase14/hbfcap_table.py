#!/usr/bin/env python3
"""PHASE 14 HBF capacity check tables.

    hbfcap_table.py WORK

Finite HBF = hc_f_* (headline, admission counter on; must equal s4_* on every
other key). Unlimited HBF = hc_u_* (SENSITIVITY, not a hardware value).
Pre-registered rule: (b)->(c) under unlimited within 5 points of finite ->
"(c)'s halving is not driven by HBF capacity". B/req per TP shard; batch per
active scheduler.
"""
import json
import os
import sys

ACTIVE = 4
BLOCK = 2621440
RANK_HBF = 1258291
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}
SRC = (
    ("first", "pcm_admit_first_blocks"),
    ("after if_read drop", "pcm_admit_after_if_read_drop_blocks"),
    ("after HBF eviction", "pcm_admit_after_hbf_evict_blocks"),
    ("other", "pcm_admit_after_other_blocks"),
)


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def m(d):
    n = d["request_count"]
    absent = d.get("hbm_evict_residue_absent_blocks", 0)
    mem, disk = d["mooncake_memory_hit_tokens"], d["mooncake_disk_hit_tokens"]
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "hbf_adj": (d["mooncake_ssd_write_bytes"] + absent * BLOCK) / n,
        "hbf_blk": d["mooncake_ssd_write_blocks"],
        "adm": d["mooncake_memory_write_blocks"],
        "dem": d["mooncake_blocks_demoted"],
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "served": mem + disk,
        "pcm_share": 100 * mem / (mem + disk) if mem + disk else 0.0,
        "evict": d["mooncake_ssd_eviction_count"],
        "peak": d.get("hbf_capacity_peak_used_blocks"),
        "rank_peak": d.get("hbf_capacity_rank_peak_used_blocks_max"),
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "notdone": len(d["notdone"]),
        "src": [d.get(key) for _, key in SRC],
    }


def red(c, b, key):
    return 100 * (1 - c[key] / b[key])


def main(work):
    for tr in ("conv", "ta"):
        f = {cfg: m(load(work, f"hc_f_{cfg}_{tr}")) for cfg in "bcd"}
        u = {cfg: m(load(work, f"hc_u_{cfg}_{tr}")) for cfg in "bc"}
        assert all(r["notdone"] == 0 for r in [*f.values(), *u.values()])
        print(f"\n### {TRACE[tr]} — finite vs unlimited HBF (full trace, headline configuration)\n")
        print("| metric | finite (b) | finite (c) | unlimited (b) | unlimited (c) |")
        print("|---|---|---|---|---|")
        rows = [
            ("HBF B/req", "hbf", "{:,.0f}"),
            ("HBF write blocks", "hbf_blk", "{:,}"),
            ("PCM admissions (blocks)", "adm", "{:,}"),
            ("PCM demotions (blocks)", "dem", "{:,}"),
            ("PCM B/req", "pcm", "{:,.0f}"),
            ("hit rate", "hit", "{:.4f}"),
            ("store-served tokens", "served", "{:,}"),
            ("PCM-served share of store-served tokens", "pcm_share", "{:.2f}%"),
            ("HBF evictions", "evict", "{:,}"),
            ("batch / active sched", "batch", "{:.3f}"),
        ]
        for name, key, fmt in rows:
            vals = [f["b"][key], f["c"][key], u["b"][key], u["c"][key]]
            print(f"| {name} | " + " | ".join(fmt.format(v) for v in vals) + " |")
        print("| HBF peak held, pool / largest rank (unlimited only) | — | — | "
              + " | ".join(f"{u[c]['peak']:,} ({u[c]['peak'] / (ACTIVE * RANK_HBF):.2f}× pool) / "
                           f"{u[c]['rank_peak']:,} ({u[c]['rank_peak'] / RANK_HBF:.2f}× rank)" for c in "bc") + " |")
        for c in (f, u):
            assert c["b"]["hbf_blk"] == c["b"]["adm"] + c["b"]["dem"], "identity (b)"
            assert c["c"]["hbf_blk"] == c["c"]["dem"], "identity (c)"
        fr, ur = red(f["c"], f["b"], "hbf"), red(u["c"], u["b"], "hbf")
        fa, ua = red(f["c"], f["b"], "hbf_adj"), red(u["c"], u["b"], "hbf_adj")
        print(f"\n(b)->(c) raw: finite {fr:.2f}%, unlimited {ur:.2f}% (diff {ur - fr:+.2f} pt); "
              f"residue-adjusted: finite {fa:.2f}%, unlimited {ua:.2f}% (diff {ua - fa:+.2f} pt). "
              f"Rule (|diff| <= 5 pt): {'PASS' if max(abs(ur - fr), abs(ua - fa)) <= 5 else 'FAIL'}")
        print(f"unlimited vs finite: (b) HBF B/req {100 * (u['b']['hbf'] / f['b']['hbf'] - 1):+.2f}%, "
              f"(c) {100 * (u['c']['hbf'] / f['c']['hbf'] - 1):+.2f}%; hit (b) {100 * (u['b']['hit'] / f['b']['hit'] - 1):+.2f}%, "
              f"(c) {100 * (u['c']['hit'] / f['c']['hit'] - 1):+.2f}%; identities (b) HBF = adm + dem, (c) HBF = dem: hold in all 4")
        print(f"\n#### {TRACE[tr]} — PCM admissions by source (count_pcm_admit_source)\n")
        print("| run | PCM admissions | " + " | ".join(name for name, _ in SRC) + " | sum check |")
        print("|---|---|" + "---|" * len(SRC) + "---|")
        for label, r in [*((f"finite ({c})", f[c]) for c in "bcd"), *((f"unlimited ({c})", u[c]) for c in "bc")]:
            shares = " | ".join(f"{v:,} ({100 * v / r['adm']:.2f}%)" for v in r["src"])
            print(f"| {label} | {r['adm']:,} | {shares} | {'ok' if sum(r['src']) == r['adm'] else 'MISMATCH'} |")


if __name__ == "__main__":
    main(sys.argv[1])
