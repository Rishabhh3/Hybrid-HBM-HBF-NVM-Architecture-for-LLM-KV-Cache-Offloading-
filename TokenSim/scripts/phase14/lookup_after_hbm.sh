#!/bin/bash
# PHASE 14: store_lookup_start = after_hbm (the vLLM KV-connector contract:
# the external store is asked for tokens beyond the locally computed ones).
# 71,104 per_rank, HBF 5,033,164, cap 48, count_hbf_readback on.
#   A) la_*   copy-at-admission (stock prefill save) + after_hbm:
#             (b), (c) single; (d) unjittered + j = 0.05 x gap seeds 1..3.
#   B) lb_*   spill_only + after_hbm: (b), (c) single; (d) 16-run pooled set.
#   la_id_*   new code, flags at default: byte identity.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK}"
export WORK
cd "$ROOT"
RUN="$HERE/run_one.py"
POOL='"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"per_rank","store_partition_ranks":4,"count_hbf_readback":true,"store_lookup_start":"after_hbm"'
SPILL='"hbm_evict_save_policy":"spill_only"'
B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }
for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; J=(1.569 15.685 141.168); else T=toolagent_5000.jsonl; J=(3.029 30.287 272.582); fi
  q la_id_b_$tr "{$B}" $T 5000 48 30
  q la_id_d_$tr "{$D}" $T 5000 48 30
  # A
  q la_b_$tr "{$B,$POOL}" $T 5000 48 30
  q la_c_$tr "{$C,$POOL}" $T 5000 48 30
  q la_d_$tr "{$D,$POOL}" $T 5000 48 30
  for k in 1 2 3; do
    q la_d_j05_s${k}_$tr "{$D,$POOL}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[1]} --jitter_seed $k
  done
  # B
  q lb_b_$tr "{$B,$POOL,$SPILL}" $T 5000 48 30
  q lb_c_$tr "{$C,$POOL,$SPILL}" $T 5000 48 30
  q lb_d_$tr "{$D,$POOL,$SPILL}" $T 5000 48 30
  i=0
  for jn in 005 05 45; do
    for k in 1 2 3 4 5; do
      q lb_d_j${jn}_s${k}_$tr "{$D,$POOL,$SPILL}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[$i]} --jitter_seed $k
    done
    i=$((i+1))
  done
done
wait
echo ALLDONE
