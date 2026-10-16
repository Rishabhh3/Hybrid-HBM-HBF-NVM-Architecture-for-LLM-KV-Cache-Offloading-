#!/usr/bin/env python3
"""PHASE 14 STEP 4 table: full traces, headline configuration.

    step4_table.py WORK

Full: s4_{b,c}_<tr>, s4_d_<tr> + s4_d_j05_s{1,2,3}_<tr>, s4_ns_<tr>.
W = 5,000 comparison (by metric, not by digits -- jitter values and replay
pace differ): v2_{b,c}_<tr>, v2_d_<tr> + v2_d_j05_s{1,2,3}_<tr>, ns_<tr>.
Pre-registered expectation: residue share falls roughly in proportion to
request count (W = 5,000 share x 5,000 / N), to ~3.5% on both traces.
"""
import json
import os
import sys

ACTIVE = 4
BLOCK = 2621440
N_FULL = {"conv": 12031, "ta": 23608}
TRACE = {"conv": "Conversation", "ta": "Tool&Agent"}


def load(work, tag):
    with open(os.path.join(work, "runs", tag, "result_30.0.json")) as handle:
        return json.load(handle)


def m(d):
    n = d["request_count"]
    absent = d.get("hbm_evict_residue_absent_blocks", 0)
    mem, disk = d.get("mooncake_memory_hit_tokens", 0), d.get("mooncake_disk_hit_tokens", 0)
    w = d.get("hbf_readback_written_blocks")
    return {
        "n": n,
        "hbf": d.get("mooncake_ssd_write_bytes", 0) / n,
        "hbf_adj": (d.get("mooncake_ssd_write_bytes", 0) + absent * BLOCK) / n,
        "pcm": d.get("mooncake_memory_write_bytes", 0) / n,
        "hit": d["prefix_cache_hit_rate"],
        "reuse": d["reuse_hit_blocks"],
        "per_reuse": d.get("mooncake_ssd_write_bytes", 0) / d["reuse_hit_blocks"],
        "pcm_share": None if mem + disk == 0 else 100 * mem / (mem + disk),
        "residue": absent,
        "residue_share": 100 * absent / (absent + d["mooncake_admission_count"]) if "mooncake_admission_count" in d and absent else None,
        "rb": None if not w else 100 * d["hbf_readback_read_blocks"] / w,
        "gate": d.get("if_read_gate_passes"),
        "gate_pcm": d.get("if_read_gate_passes_pcm_served"),
        "batch": d["batch_mean_cluster"] / ACTIVE,
        "notdone": len(d["notdone"]),
    }


def rng(v, f):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    return f.format(v[0]) if min(v) == max(v) else f"{f.format(min(v))} – {f.format(max(v))}"


def summarise(work, prefix_full, tr, ns_tag):
    rows = {
        "b": [m(load(work, f"{prefix_full}_b_{tr}"))],
        "c": [m(load(work, f"{prefix_full}_c_{tr}"))],
        "d": [m(load(work, f"{prefix_full}_d{s}_{tr}")) for s in ("", "_j05_s1", "_j05_s2", "_j05_s3")],
    }
    ns = m(load(work, ns_tag))
    return rows, ns


def kept(h, b, ns):
    return 100 * (h - ns) / (b - ns)


def main(work):
    for tr in ("conv", "ta"):
        full, nsf = summarise(work, "s4", tr, f"s4_ns_{tr}")
        w5, ns5 = summarise(work, "v2", tr, f"ns_{tr}")
        assert all(r["notdone"] == 0 for cfg in "bcd" for r in full[cfg]) and nsf["notdone"] == 0
        print(f"\n### {TRACE[tr]} — full trace ({N_FULL[tr]:,} requests) vs W = 5,000 (by metric)\n")
        print("| metric | full trace | W = 5,000 (same 4-run (d) composition) |")
        print("|---|---|---|")

        def line(name, fn, fmt):
            print(f"| {name} | {fn(full, nsf)} | {fn(w5, ns5)} |")

        b = lambda R: R["b"][0]
        c = lambda R: R["c"][0]
        line("(b) → (c), raw", lambda R, ns: f"{100 * (1 - c(R)['hbf'] / b(R)['hbf']):.2f}%", None)
        line("(b) → (c), residue-adjusted", lambda R, ns: f"{100 * (1 - c(R)['hbf_adj'] / b(R)['hbf_adj']):.2f}%", None)
        line("(b) → (d), worst / best (N = 4)", lambda R, ns: f"{min(100 * (1 - x['hbf'] / b(R)['hbf']) for x in R['d']):.2f}% / "
             f"{max(100 * (1 - x['hbf'] / b(R)['hbf']) for x in R['d']):.2f}%", None)
        line("(b) HBF B/req", lambda R, ns: f"{b(R)['hbf']:,.0f}", None)
        line("(c) HBF B/req", lambda R, ns: f"{c(R)['hbf']:,.0f}", None)
        line("(d) HBF B/req range", lambda R, ns: rng([x['hbf'] for x in R['d']], '{:,.0f}'), None)
        line("hit rate (b) / (c) / (d)", lambda R, ns: f"{b(R)['hit']:.4f} / {c(R)['hit']:.4f} / {rng([x['hit'] for x in R['d']], '{:.4f}')}", None)
        line("no-store hit rate", lambda R, ns: f"{ns['hit']:.4f}", None)
        line("(d) hit rate vs (b)", lambda R, ns: rng(sorted(100 * (x['hit'] / b(R)['hit'] - 1) for x in R['d']), '{:+.2f}%'), None)
        line("store benefit kept (c) / (d)", lambda R, ns: f"{kept(c(R)['hit'], b(R)['hit'], ns['hit']):.1f}% / "
             f"{rng(sorted(kept(x['hit'], b(R)['hit'], ns['hit']) for x in R['d']), '{:.1f}%')}", None)
        line("PCM-served share of store-served tokens (b) / (c) / (d)", lambda R, ns: f"{b(R)['pcm_share']:.2f}% / {c(R)['pcm_share']:.2f}% / "
             f"{rng([x['pcm_share'] for x in R['d']], '{:.2f}%')}", None)
        line("HBF bytes per reuse-hit block (b) / (c) / (d)", lambda R, ns: f"{b(R)['per_reuse']:,.0f} / {c(R)['per_reuse']:,.0f} / "
             f"{rng([x['per_reuse'] for x in R['d']], '{:,.0f}')}", None)
        line("PCM B/req (b) / (c) / (d)", lambda R, ns: f"{b(R)['pcm']:,.0f} / {c(R)['pcm']:,.0f} / {rng([x['pcm'] for x in R['d']], '{:,.0f}')}", None)
        line("end-of-run residue share (b) / (c) / (d)", lambda R, ns: f"{b(R)['residue_share']:.2f}% / {c(R)['residue_share']:.2f}% / "
             f"{rng([x['residue_share'] for x in R['d']], '{:.2f}%')}", None)
        line("HBF read-back share (b) / (c) / (d)", lambda R, ns: f"{b(R)['rb']:.2f}% / {c(R)['rb']:.2f}% / {rng([x['rb'] for x in R['d']], '{:.2f}%')}", None)
        line("(d) gate passes / with PCM-served read", lambda R, ns: f"{rng([x['gate'] for x in R['d']], '{:,}')} / "
             f"{rng([x['gate_pcm'] for x in R['d']], '{:,}')} (diff {max(x['gate'] - x['gate_pcm'] for x in R['d'])})", None)
        line("batch / active sched (b) / (c) / (d) / no-store", lambda R, ns: f"{b(R)['batch']:.3f} / {c(R)['batch']:.3f} / "
             f"{rng([x['batch'] for x in R['d']], '{:.3f}')} / {ns['batch']:.3f}", None)
        line("batch gap (c) vs (b) / (d) vs (b)", lambda R, ns: f"{100 * (c(R)['batch'] / b(R)['batch'] - 1):+.2f}% / "
             f"{rng(sorted(100 * (x['batch'] / b(R)['batch'] - 1) for x in R['d']), '{:+.2f}%')}", None)
        exp = [x * 5000 / N_FULL[tr] for x in (w5['b'][0]['residue_share'], w5['c'][0]['residue_share'])]
        print(f"\nresidue-share expectation (W = 5,000 share x 5,000 / {N_FULL[tr]:,}): (b) {exp[0]:.2f}%, (c) {exp[1]:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])
