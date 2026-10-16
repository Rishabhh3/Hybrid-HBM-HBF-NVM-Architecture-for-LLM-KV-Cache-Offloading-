#!/usr/bin/env python3
"""Inter-burst gaps in replay time, measured with the simulator's own loader.

    burst_gaps.py TRACE.jsonl W TARGET_QPS [FRACTION]

Loads the trace exactly as benchmark.py does (qwen_jsonl, first W requests,
--trace_target_qps rescaling, no jitter), groups requests sharing an arrival
time into bursts, and prints min / p50 / max of the gaps between consecutive
bursts in replay seconds, plus FRACTION (default 0.45) x the minimum gap in ms:
the PHASE 14 burst-jitter half-width, a methodology choice, not a hardware
number.
"""
import os
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TokenSim.workload.loaders import burst_gaps_s, load_workload  # noqa: E402

path, w, qps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
args = SimpleNamespace(
    workload_type="qwen_jsonl", dataset_path=path, request_count=w,
    dataset_skip_count=0, trace_timestamp_scale=None, trace_target_qps=qps,
    random_seed=0,
)
model = SimpleNamespace(moe_config=SimpleNamespace(num_experts=0))
reqs = load_workload(args, model)
gaps = burst_gaps_s(reqs)
print(f"{os.path.basename(path)}: requests {len(reqs)}, bursts {len(gaps) + 1}, "
      f"replay gap min {min(gaps):.6f} s, p50 {statistics.median(gaps):.6f} s, "
      f"max {max(gaps):.6f} s; j = {fraction} x min = {fraction * min(gaps) * 1000:.3f} ms")
