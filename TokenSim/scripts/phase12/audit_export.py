#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys, runpy
import util.results as R
from TokenSim.config.psla_config import LLMResult

WORK, KV = sys.argv[1], sys.argv[2]
seen = {}
real = R.LLMResult
def spy(**kwargs):
    seen.update(kwargs)
    return real(**{k: v for k, v in kwargs.items()
                   if k in LLMResult.__dataclass_fields__})
R.LLMResult = spy

sys.argv = ["benchmark.py", "--qps", "30", "--batching", "paged-attn",
    "--request_count", "60", "--block_size", "16",
    "--model", "scripts/phase12/llama-70b-gqa.json",
    "--cluster", "data/clusters/8_b200_h3/h8_tp2dp4.json",
    "--hardware_models", "data/hardware/hardware_models_h3b200.json",
    "--kv_transfer_config", KV,
    "--tensor_parallel_size", "2", "--data_parallel_size", "4",
    "--max_parallem_sum", "48", "--workload_type", "qwen_jsonl",
    "--dataset_path", WORK + "/conv_5000.jsonl",
    "--trace_target_qps", "30", "--results_path", WORK + "/runs/audit",
    "--verbose", "none", "--random_seed", "0"]
try:
    runpy.run_path("benchmark.py", run_name="__main__")
except SystemExit:
    pass

declared = set(LLMResult.__dataclass_fields__)
dropped = sorted(set(seen) - declared)
print("\n=== keys handed to LLMResult:", len(seen))
print("=== declared by LLMResult   :", len(declared))
print("=== DROPPED                 :", len(dropped))
for k in dropped:
    print("   ", k, "=", seen[k])
