#!/usr/bin/env python3
"""Regenerate hardware_models_h3b200.json from the tracked catalogue.

Adds the H3-B200-KVONLY entry WITHOUT mutating the tracked catalogue
(CLAUDE.md, Working rules). Run from the TokenSim root:

    .venv/bin/python data/hardware/make_h3b200.py

H3 (SK hynix, IEEE CAL Jan-Jun 2026), per GPU, B200-class:
  HBM3e 192 GB @ 8 TB/s  (8 cubes x 24 GB x 1 TB/s)   -- all of it KV cache
  HBF   3 TB   @ 8 TB/s  (8 stacks x 384 GB)          -- daisy-chained behind HBM

cache_config.py:70-74 subtracts model_param_size from HBM unconditionally, but
H3 puts the weights in HBF. Capacity is pre-compensated so that exactly 192 GiB
of KV survives the subtraction. This is a workaround, not a model change.
"""
import json, os

GiB = 1 << 30
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "TransformerRoofline", "hardware_models.json")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "hardware_models_h3b200.json")

# LLaMa2-70B-GQA, the exact expression cache_config.py uses, at TP2/PP1.
Nlayer, Dmodel, TP, PP = 80, 8192, 2, 1
param_bytes = (12 * Nlayer * Dmodel**2 + 50000 * Dmodel) * 2 / (TP * PP)
CAPACITY = 192.0 + param_bytes / GiB          # 252.3814697265625

cat = json.load(open(SRC))
assert not any(h["Name"] == "H3-B200-KVONLY" for h in cat["hardware"])
b200 = next(h for h in cat["hardware"] if h["Name"] == "B200")
cat["hardware"].append({**b200, "Name": "H3-B200-KVONLY", "Capacity": CAPACITY})
json.dump(cat, open(DST, "w"), indent=2)

kv = CAPACITY * GiB - param_bytes
print(f"model_param_size (TP{TP}/PP{PP}) = {param_bytes:,.0f} B = {param_bytes/GiB:.10f} GiB")
print(f"Capacity = 192 + {param_bytes/GiB:.10f} = {CAPACITY!r}")
print(f"check: KV after subtraction = {kv:,.0f} B = {kv/GiB:.10f} GiB")
print(f"wrote {DST}")
