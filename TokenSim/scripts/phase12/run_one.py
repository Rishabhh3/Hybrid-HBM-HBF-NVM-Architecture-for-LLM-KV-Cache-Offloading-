#!/usr/bin/env python3
"""One PHASE 12 run.

    WORK=<dir> run_one.py TAG '<json overrides>' TRACE W CAP QPS

WORK holds the built trace files and receives runs/<TAG>/result_<QPS>.json.
BASE below is config (b) of PHASE 9/10 -- the HBF-only baseline -- and the
overrides are merged over it, so a run differs from the recorded baseline only
in what it names.

Capacities are cluster pool totals (8 workers), per CLAUDE.md:
  memory (PCM) 43.4 GiB/GPU x 8 = 347.2 GiB = 142,213 blocks at 2,621,440 B
  offload (HBF)   3 TiB/GPU x 8 =    24 TiB = 10,066,329 blocks
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("WORK") or os.path.join(ROOT, "phase12-work")

BASE = {
    "mode": "standalone-store",
    "protocol": "rdma",
    "global_segment_size_gb": 1,
    "local_buffer_size_gb": 1,
    "enable_offload": True,
    "memory_capacity_blocks": 142213,
    "ssd_capacity_blocks": 10066329,
    "ssd_read_bw_gbps": 7451,          # 8 TB/s HBF
    "ssd_write_bw_gbps": 3.0,
    "ssd_read_latency_us": 0.1,        # LHB perfect-prefetch upper bound
    "ssd_write_latency_us": 200,
    "replica_num": 1,
    "admission_policy": "always",
    "eviction_policy": "lru",
    "charge_eviction_writes": True,
    "load_async": False,
    "transfer_overlap": False,
}


def command(tag, overrides, trace, w, cap, qps):
    extra = {**BASE, **json.loads(overrides)}
    config_path = os.path.join(WORK, f"kv_{tag}.json")
    os.makedirs(WORK, exist_ok=True)
    with open(config_path, "w") as handle:
        json.dump(
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_connector_extra_config": extra,
            },
            handle,
            indent=2,
        )
    out = os.path.join(WORK, "runs", tag)
    os.makedirs(out, exist_ok=True)
    return [
        f"{ROOT}/.venv/bin/python", "benchmark.py",
        "--qps", qps, "--batching", "paged-attn", "--request_count", w,
        "--block_size", "16",
        "--model", os.path.join(HERE, "llama-70b-gqa.json"),
        "--cluster", "data/clusters/8_b200_h3/h8_tp2dp4.json",
        "--hardware_models", "data/hardware/hardware_models_h3b200.json",
        "--kv_transfer_config", config_path,
        "--tensor_parallel_size", "2", "--data_parallel_size", "4",
        "--max_parallem_sum", cap,
        "--workload_type", "qwen_jsonl",
        "--dataset_path", os.path.join(WORK, trace),
        "--trace_target_qps", qps,
        "--results_path", out,
        "--verbose", "none", "--random_seed", "0",
    ]


if __name__ == "__main__":
    cmd = command(*sys.argv[1:7])
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)
