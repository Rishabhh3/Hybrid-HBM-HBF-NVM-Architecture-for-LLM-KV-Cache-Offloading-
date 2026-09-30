#!/bin/bash
# PHASE 12 driver. Config (b) only: dram memory tier, write_through admission,
# always demote -- the HBF-only baseline, identical to PHASE 9/10.
#
#   p12_def_*   cap 48, stock policy. The byte-identity pair: these must match
#               the same run on the pre-PHASE-12 commit exactly, excluding
#               simulator_wall_time and mooncake_pool_keys.
#   s*          "count": instrumentation only, no behaviour change. Supplies
#               the stock arm of the falsifier table.
#   e*          "on_evict": prefill save PLUS an eviction-driven save.
#
# Batch points are the lowest- and highest-occupancy rows of the PHASE 9
# STEP 4 sweep: Conversation caps 8 and 256, Tool&Agent caps 8 and 96.
#
# Usage: WORK=<dir holding conv_5000.jsonl> scripts/phase12/step3.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"

B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
launch(){ nohup "$HERE/run_one.py" "$1" "$2" "$3" 5000 "$4" 30 > "$WORK/runs/$1.log" 2>&1 & }
# Four concurrent runs is what 62 GB of RAM holds; each needs ~1.7 GB.
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 4 ]; do sleep 10; done; launch "$@"; sleep 5; }

q p12_def_conv "{$B}"                                     conv_5000.jsonl      48
q p12_def_ta   "{$B}"                                     toolagent_5000.jsonl 48
q s8_conv      "{$B,\"hbm_evict_save_policy\":\"count\"}" conv_5000.jsonl      8
q s8_ta        "{$B,\"hbm_evict_save_policy\":\"count\"}" toolagent_5000.jsonl 8
q s256_conv    "{$B,\"hbm_evict_save_policy\":\"count\"}" conv_5000.jsonl      256
q s96_ta       "{$B,\"hbm_evict_save_policy\":\"count\"}" toolagent_5000.jsonl 96
q e8_conv   "{$B,\"hbm_evict_save_policy\":\"on_evict\"}" conv_5000.jsonl      8
q e8_ta     "{$B,\"hbm_evict_save_policy\":\"on_evict\"}" toolagent_5000.jsonl 8
q e256_conv "{$B,\"hbm_evict_save_policy\":\"on_evict\"}" conv_5000.jsonl      256
q e96_ta    "{$B,\"hbm_evict_save_policy\":\"on_evict\"}" toolagent_5000.jsonl 96
wait
echo ALLDONE
