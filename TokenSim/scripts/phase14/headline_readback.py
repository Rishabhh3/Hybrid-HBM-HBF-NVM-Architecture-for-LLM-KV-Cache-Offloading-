#!/usr/bin/env python3
"""PHASE 14 headline: decompose (d)'s HBF-write drop, 142,213 global -> 71,104
per_rank, into blocks read back from HBF vs never read back.

    headline_readback.py WORK
"""
import json
import os
import sys

work = sys.argv[1]
L = lambda t: json.load(open(os.path.join(work, "runs", t, "result_30.0.json")))
tags = lambda tr: [f"hl_d_{tr}"] + [f"hl_d_j{j}_s{k}_{tr}" for j in ("005", "05", "45") for k in range(1, 6)]
for tr in ("conv", "ta"):
    big = L(f"rb142_d_{tr}")
    bw, br = big["hbf_readback_written_blocks"], big["hbf_readback_read_blocks"]
    rows = [(L(t)["hbf_readback_written_blocks"], L(t)["hbf_readback_read_blocks"]) for t in tags(tr)]
    shares = [100 * ((bw - br) - (w - r)) / (bw - w) for w, r in rows]
    print(f"{tr}: 142k written {bw:,} read {br:,}; 71k written {min(w for w, _ in rows):,}-{max(w for w, _ in rows):,}; "
          f"never-read-back share of the drop {min(shares):.1f}%-{max(shares):.1f}% (unjittered {shares[0]:.1f}%)")
