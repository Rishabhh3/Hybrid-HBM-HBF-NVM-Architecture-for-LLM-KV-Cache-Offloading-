#!/usr/bin/env python3
"""PHASE 14 STEP 3b table: spill_only robustness check against the headline set.

    step3b_table.py WORK [PREFIX]

PREFIX is the spill run prefix: sp (from-zero lookup, default) or lb
(store_lookup_start = after_hbm).

Pairs every spill_only run (sp_*) with the headline run of the same config,
jitter and seed (hl_*). Rule (ii) is judged on RESIDUE-ADJUSTED values:
measured writes + resident keyed blocks absent from the store, each counted
once (x block bytes), per request. Rule (iii) is judged on raw (d) HBF B/req
against per-trace bands.
"""
import json
import os
import statistics
import sys

ACTIVE = 4
BLOCK = 2621440
BAND = {"conv": (5.65e6, 12.29e6), "ta": (3.88e6, 6.67e6)}
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def d_suffixes():
    return [""] + [f"_j{j}_s{k}" for j in ("005", "05", "45") for k in range(1, 6)]


def m(d):
    n = d["request_count"]
    absent = d.get("hbm_evict_residue_absent_blocks", 0)
    return {
        "hbf": d["mooncake_ssd_write_bytes"] / n,
        "hbf_adj": (d["mooncake_ssd_write_bytes"] + absent * BLOCK) / n,
        "pcm": d["mooncake_memory_write_bytes"] / n,
        "pcm_adj": (d["mooncake_memory_write_bytes"] + absent * BLOCK) / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "per_reuse": d["mooncake_ssd_write_bytes"] / d["reuse_hit_blocks"],
        "resid": d.get("hbm_evict_residue_blocks"),
        "absent": absent,
        "admissions": d["mooncake_admission_count"],
        "present": d.get("hbm_evict_present_blocks"),
        "evict_absent": d.get("hbm_evict_absent_blocks"),
        "spills": d.get("hbm_evict_blocks"),
        "p50": d.get("hbm_evict_first_spill_p50_s"),
        "p90": d.get("hbm_evict_first_spill_p90_s"),
        "rb": (100 * d["hbf_readback_read_blocks"] / d["hbf_readback_written_blocks"]
               if d.get("hbf_readback_written_blocks") else None),
        "evictions": d["mooncake_memory_eviction_count"],
        "lookups_hit": d["mooncake_store_hit_count"],
        "lookups": d["mooncake_store_hit_count"] + d["mooncake_store_miss_count"],
        "hbf_read": d["mooncake_ssd_read_blocks"],
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
            sufs = d_suffixes() if cfg == "d" else [""]
            rows[cfg] = [(m(load(work, f"{PREFIX}_{cfg}{s}_{tr}")), m(load(work, f"hl_{cfg}{s}_{tr}"))) for s in sufs]
            assert all(sp["notdone"] == 0 for sp, _ in rows[cfg])
        print("| config | store lookups that hit / lookups | headline: hit / lookups | HBF read blocks | headline HBF read blocks |")
        print("|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            r = rows[cfg]
            print(f"| ({cfg}) | {rng([a['lookups_hit'] for a, _ in r], '{:,}')} / {rng([a['lookups'] for a, _ in r], '{:,}')} "
                  f"| {rng([b['lookups_hit'] for _, b in r], '{:,}')} / {rng([b['lookups'] for _, b in r], '{:,}')} "
                  f"| {rng([a['hbf_read'] for a, _ in r], '{:,}')} | {rng([b['hbf_read'] for _, b in r], '{:,}')} |")
        print()
        print("| config | N | HBF B/req raw | HBF B/req residue-adjusted | headline HBF B/req | Δ adjusted vs headline "
              "| PCM B/req raw | PCM adjusted | Δ adjusted vs headline | hit rate | Δ hit vs headline | reuse-hit blk "
              "| memory evictions | batch / active sched | batch gap vs headline |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            r = rows[cfg]
            sp = [a for a, _ in r]
            hl = [b for _, b in r]
            print(f"| ({cfg}) | {len(r)} | {rng([x['hbf'] for x in sp], '{:,.0f}')} | {rng([x['hbf_adj'] for x in sp], '{:,.0f}')} "
                  f"| {rng([x['hbf'] for x in hl], '{:,.0f}')} | {rng(sorted(pct(a['hbf_adj'], b['hbf']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['pcm'] for x in sp], '{:,.0f}')} | {rng([x['pcm_adj'] for x in sp], '{:,.0f}')} "
                  f"| {rng(sorted(pct(a['pcm_adj'], b['pcm']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['hit'] for x in sp], '{:.4f}')} | {rng(sorted(pct(a['hit'], b['hit']) for a, b in r), '{:+.2f}%')} "
                  f"| {rng([x['reuse'] for x in sp], '{:,}')} | {rng([x['evictions'] for x in sp], '{:,}')} "
                  f"| {rng([x['batch'] for x in sp], '{:.3f}')} | {rng(sorted(pct(a['batch'], b['batch']) for a, b in r), '{:+.2f}%')} |")
        print("\n| config | HBM spills (capacity evictions) | spill key already in store / absent | end-of-run HBM residue (keyed) | of which absent from store | residue share of would-be store admissions | admission→first spill p50 / p90 s | HBF read-back share |")
        print("|---|---|---|---|---|---|---|---|")
        for cfg in ("b", "c", "d"):
            sp = [a for a, _ in rows[cfg]]
            share = [100 * x["absent"] / (x["absent"] + x["admissions"]) for x in sp]
            rb = [x["rb"] for x in sp if x["rb"] is not None]
            print(f"| ({cfg}) | {rng([x['spills'] for x in sp], '{:,}')} | {rng([x['present'] for x in sp], '{:,}')} / {rng([x['evict_absent'] for x in sp], '{:,}')} "
                  f"| {rng([x['resid'] for x in sp], '{:,}')} | {rng([x['absent'] for x in sp], '{:,}')} | {rng(share, '{:.2f}%')} "
                  f"| {rng([x['p50'] for x in sp], '{:.1f}')} / {rng([x['p90'] for x in sp], '{:.1f}')} | {rng(rb, '{:.2f}%') if rb else '—'} |")
        b_sp, b_hl = rows["b"][0]
        c_sp, c_hl = rows["c"][0]
        d_sp = [a for a, _ in rows["d"]]
        lo, hi = BAND[tr]
        inside = sum(1 for x in d_sp if lo <= x["hbf"] <= hi)
        print(f"\nreductions (spill_only, raw): (b)->(c) {100 * (1 - c_sp['hbf'] / b_sp['hbf']):.2f}% "
              f"(headline {100 * (1 - c_hl['hbf'] / b_hl['hbf']):.2f}%); (b)->(d) worst {min(100 * (1 - x['hbf'] / b_sp['hbf']) for x in d_sp):.2f}%, "
              f"best {max(100 * (1 - x['hbf'] / b_sp['hbf']) for x in d_sp):.2f}%; residue-adjusted (b)->(c) {100 * (1 - c_sp['hbf_adj'] / b_sp['hbf_adj']):.2f}%, "
              f"(b)->(d) worst {min(100 * (1 - x['hbf'] / b_sp['hbf_adj']) for x in d_sp):.2f}% ((d) raw vs (b) adjusted)")
        print(f"(d) hit rate vs (b), same model: {rng(sorted(pct(x['hit'], b_sp['hit']) for x in d_sp), '{:+.2f}%')}; "
              f"HBF bytes per reuse-hit block (b) {b_sp['per_reuse']:,.0f} / (c) {c_sp['per_reuse']:,.0f} / (d) {rng(sorted(x['per_reuse'] for x in d_sp), '{:,.0f}')}")
        adj_ok = all(abs(pct(a['hbf_adj'], b['hbf'])) < 5 for cfg in 'bc' for a, b in rows[cfg]) and \
                 all(abs(pct(a['pcm_adj'], b['pcm'])) < 5 for cfg in 'bcd' for a, b in rows[cfg])
        worst_ii = max([abs(pct(a['hbf_adj'], b['hbf'])) for cfg in 'bc' for a, b in rows[cfg]] +
                       [abs(pct(a['pcm_adj'], b['pcm'])) for cfg in 'bcd' for a, b in rows[cfg]])
        print(f"RULE (ii) residue-adjusted (b),(c) HBF and all PCM within 5% of headline: {'PASS' if adj_ok else 'FAIL'} (largest |Δ| {worst_ii:.2f}%)")
        hl_red = [100 * (1 - b["hbf"] / rows["b"][0][1]["hbf"]) for _, b in rows["d"]]
        sp_red = [100 * (1 - x["hbf"] / b_sp["hbf"]) for x in d_sp]
        ok_i = min(sp_red) >= min(hl_red) - 5 and max(sp_red) <= max(hl_red) + 5
        print(f"RULE (i) (b)->(d) reduction {min(sp_red):.2f}–{max(sp_red):.2f}% within 5 points of headline "
              f"{min(hl_red):.2f}–{max(hl_red):.2f}%: {'PASS' if ok_i else 'FAIL'}")
        print(f"RULE (iii) (d) HBF raw inside band {lo / 1e6:.2f}–{hi / 1e6:.2f} M: {inside}/{len(d_sp)} runs -> {'PASS' if inside == len(d_sp) else 'FAIL'}")


PREFIX = "sp"

if __name__ == "__main__":
    if len(sys.argv) > 2:
        PREFIX = sys.argv[2]
    main(sys.argv[1])
