#!/usr/bin/env python3
"""PHASE 14: break-even PCM endurance, per TP shard, full traces, headline.

    breakeven_pcm.py WORK

E_PCM = (HBF_cap_bytes x 3000 / HBF_B_per_req) x PCM_B_per_req / PCM_cap_bytes

Capacities per writing rank (one TP-0 shard on one GPU): HBF 1,258,291 blocks
(5,033,164 // 4, mooncake/ssd.py:157-158), PCM 17,776 blocks (71,104 // 4,
mooncake/store.py:446), 2,621,440 B per block. HBF 3000 P/E (TLC, CLAUDE.md).
B/req are pool-wide (all requests, all 4 ranks), so requests-to-HBF-exhaustion
uses the pool capacity: pool / B_req = 4 x rank / B_req. E_PCM itself depends
only on the capacity RATIO and is the same on either basis.
Assumes perfect wear levelling (both tiers), HBF WAF = 1, full-block writes,
and per-rank write ratios equal to the pooled ratio (not measured).
"""
import json
import os
import sys

BLOCK = 2621440
HBF_RANK, PCM_RANK, RANKS, PE_HBF = 1258291, 17776, 4, 3000
HBF_CAP, PCM_CAP = HBF_RANK * BLOCK, PCM_RANK * BLOCK
RUNS = {"b": [""], "c": [""], "d": ["", "_j05_s1", "_j05_s2", "_j05_s3"]}
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def e_pcm(hbf, pcm):
    return HBF_CAP * PE_HBF / hbf * pcm / PCM_CAP


def main(work):
    print(f"HBF cap/rank {HBF_CAP:,} B; PCM cap/rank {PCM_CAP:,} B; ratio {HBF_CAP / PCM_CAP:.4f}")
    for tr in ("conv", "ta"):
        print(f"\n### {TRACE[tr]}\n")
        print("| config | run | HBF B/req | PCM B/req | requests to HBF exhaustion (pool) | **E_PCM (cycles)** "
              "| E_PCM, residue spilled, never demoted | E_PCM, residue spilled, demoted at the config's own rate | demotion rate |")
        print("|---|---|---|---|---|---|---|---|---|")
        out = {}
        for cfg, seeds in RUNS.items():
            for s in seeds:
                d = load(work, f"s4_{cfg}{s}_{tr}")
                n = d["request_count"]
                hbf, pcm = d["mooncake_ssd_write_bytes"] / n, d["mooncake_memory_write_bytes"] / n
                r = d.get("hbm_evict_residue_absent_blocks", 0) * BLOCK / n
                e = e_pcm(hbf, pcm)
                # Residue blocks, if spilled: one PCM write each. HBF writes per residue block =
                # (write-through copy, (b) only) + (the config's own measured demotion rate,
                # blocks_demoted / memory evictions: 1 under always, small under if_read).
                dem_rate = d["mooncake_blocks_demoted"] / d["mooncake_memory_eviction_count"]
                per_hbf = (1 if cfg == "b" else 0) + dem_rate
                e_up = e_pcm(hbf, pcm + r)
                e_dem = e_pcm(hbf + per_hbf * r, pcm + r)
                out.setdefault(cfg, []).append((e, hbf, pcm))
                print(f"| ({cfg}) | {s or 'unjittered'} | {hbf:,.0f} | {pcm:,.0f} | {RANKS * HBF_CAP * PE_HBF / hbf:,.0f} "
                      f"| **{e:,.1f}** | {e_up:,.1f} | {e_dem:,.1f} | {dem_rate:.4f} |")
        ec = out["c"][0]
        for e, hbf, pcm in out["d"]:
            lhs = e / ec[0]
            rhs = (ec[1] / hbf) * (pcm / ec[2])
            assert abs(lhs / rhs - 1) < 1e-12, (lhs, rhs)
        lo, hi = min(x[0] for x in out["d"]), max(x[0] for x in out["d"])
        print(f"\n(d) E_PCM range N = 4: {lo:,.0f} – {hi:,.0f}; (d)/(c) threshold ratio "
              f"{lo / ec[0]:.2f} – {hi / ec[0]:.2f}; sanity check (E_d/E_c == HBF_c/HBF_d x PCM_d/PCM_c): holds in all 4 to 1e-12")


if __name__ == "__main__":
    main(sys.argv[1])
