#!/usr/bin/env python3
"""PHASE 14 verification table: read_credit = served + load_span = missing.

    verify_table.py WORK

V1: v1_d_* (copy-at-admission, from-zero lookup), paired with the headline hl_d_*.
V2: v2_* (spill_only, after_hbm lookup), paired with set B lb_* (same model
without the fixes). Batch is per active scheduler; B/req per TP shard.
"""
import json
import os
import sys

ACTIVE = 4
BLOCK = 2621440
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def m(d):
    n = d["request_count"]
    mem, disk = d["mooncake_memory_hit_tokens"], d["mooncake_disk_hit_tokens"]
    absent = d.get("hbm_evict_residue_absent_blocks", 0)
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "hbf_blk": d["mooncake_ssd_write_blocks"],
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "per_reuse": d["mooncake_ssd_write_bytes"] / d["reuse_hit_blocks"],
        "pcm_share": None if mem + disk == 0 else 100 * mem / (mem + disk),
        "served": mem + disk,
        "passes": d.get("if_read_gate_passes"),
        "passes_pcm": d.get("if_read_gate_passes_pcm_served"),
        "residue_share": 100 * absent / (absent + d["mooncake_admission_count"]) if absent else 0.0,
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "notdone": len(d["notdone"]),
    }


def rng(v, f):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    return f.format(v[0]) if min(v) == max(v) else f"{f.format(min(v))} – {f.format(max(v))}"


def pct(a, b):
    return 100 * (a / b - 1)


D_ALL = [""] + [f"_j{j}_s{k}" for j in ("005", "05", "45") for k in range(1, 6)]


def main(work):
    print("## V1 — copy-at-admission + from-zero lookup, fixes ON (expected (d) HBF = 0)\n")
    print("| trace | run | (d) HBF B/req | headline (fixes off) | gate passes | passes with a PCM-served read | difference | store-served tokens (PCM / HBF) |")
    print("|---|---|---|---|---|---|---|---|")
    for tr in ("conv", "ta"):
        for s in ("", "_j05_s1", "_j05_s2", "_j05_s3"):
            v, h = m(load(work, f"v1_d{s}_{tr}")), m(load(work, f"hl_d{s}_{tr}"))
            d = load(work, f"v1_d{s}_{tr}")
            print(f"| {TRACE[tr]} | {s or 'unjittered'} | {v['hbf']:,.0f} | {h['hbf']:,.0f} | {v['passes']:,} | {v['passes_pcm']:,} "
                  f"| {v['passes'] - v['passes_pcm']:,} | {d['mooncake_memory_hit_tokens']:,} / {d['mooncake_disk_hit_tokens']:,} |")
    print("\n## V2 — spill_only + vLLM-style lookup, fixes ON\n")
    for tr in ("conv", "ta"):
        rows = {cfg: [m(load(work, f"v2_{cfg}{s}_{tr}")) for s in (D_ALL if cfg == "d" else [""])] for cfg in "bcd"}
        prev = {cfg: [m(load(work, f"lb_{cfg}{s}_{tr}")) for s in (D_ALL if cfg == "d" else [""])] for cfg in "bcd"}
        assert all(r["notdone"] == 0 for cfg in "bcd" for r in rows[cfg])
        b, c, D = rows["b"][0], rows["c"][0], rows["d"]
        print(f"\n### {TRACE[tr]}\n")
        print("| config | N | HBF B/req | set B (fixes off) | PCM B/req | hit rate | store-served tokens | from PCM | HBF bytes / reuse-hit blk "
              "| gate passes | with PCM-served read | difference | end-of-run residue share | batch / active sched |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in "bcd":
            R = rows[cfg]
            diff = [r["passes"] - r["passes_pcm"] for r in R] if cfg == "d" else [None]
            print(f"| ({cfg}) | {len(R)} | {rng([r['hbf'] for r in R], '{:,.0f}')} | {rng([r['hbf'] for r in prev[cfg]], '{:,.0f}')} "
                  f"| {rng([r['pcm'] for r in R], '{:,.0f}')} | {rng([r['hit'] for r in R], '{:.4f}')} "
                  f"| {rng([r['served'] for r in R], '{:,}')} | {rng([r['pcm_share'] for r in R], '{:.2f}%')} "
                  f"| {rng([r['per_reuse'] for r in R], '{:,.0f}')} "
                  f"| {rng([r['passes'] for r in R] if cfg == 'd' else [None], '{:,}')} "
                  f"| {rng([r['passes_pcm'] for r in R] if cfg == 'd' else [None], '{:,}')} | {rng(diff, '{:,}')} "
                  f"| {rng([r['residue_share'] for r in R], '{:.2f}%')} | {rng([r['batch'] for r in R], '{:.3f}')} |")
        red_c = 100 * (1 - c["hbf"] / b["hbf"])
        red_d = [100 * (1 - r["hbf"] / b["hbf"]) for r in D]
        print(f"\n(b)->(c) {red_c:.2f}%; (b)->(d) worst {min(red_d):.2f}%, best {max(red_d):.2f}% (N = {len(D)}); "
              f"(d) HBF {min(r['hbf'] for r in D):,.0f} – {max(r['hbf'] for r in D):,.0f}; "
              f"hit rate (b) {b['hit']:.4f} / (c) {c['hit']:.4f} / (d) {rng([r['hit'] for r in D], '{:.4f}')}; "
              f"(d) vs (b) {rng(sorted(pct(r['hit'], b['hit']) for r in D), '{:+.2f}%')}; "
              f"batch gap (c) vs (b) {pct(c['batch'], b['batch']):+.2f}%, (d) vs (b) {rng(sorted(pct(r['batch'], b['batch']) for r in D), '{:+.2f}%')}; "
              f"vs set B: (b) HBF {pct(b['hbf'], prev['b'][0]['hbf']):+.2f}%, (c) HBF {pct(c['hbf'], prev['c'][0]['hbf']):+.2f}%, "
              f"(b) hit {pct(b['hit'], prev['b'][0]['hit']):+.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
