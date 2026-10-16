#!/bin/bash
# PHASE 14 verification of the bookkeeping fixes: read_credit = served,
# load_span = missing (both ON), count_if_read_gate + count_hbf_readback on.
# 71,104 per_rank, HBF 5,033,164, cap 48, both traces.
#   v1_*   copy-at-admission + store_lookup_start = zero:
#          (d) unjittered + j = 0.05 x gap seeds 1..3. Expected (d) HBF = 0.
#   v2_*   spill_only + store_lookup_start = after_hbm (set B):
#          (b), (c) single; (d) 16-run pooled set.
#   v_id_* new code, flags at default: byte identity.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK}"
export WORK
cd "$ROOT"
RUN="$HERE/run_one.py"
POOL='"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"per_rank","store_partition_ranks":4,"count_hbf_readback":true,"read_credit":"served","load_span":"missing","count_if_read_gate":true'
V1='"store_lookup_start":"zero"'
V2='"store_lookup_start":"after_hbm","hbm_evict_save_policy":"spill_only"'
B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }
for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; J=(1.569 15.685 141.168); else T=toolagent_5000.jsonl; J=(3.029 30.287 272.582); fi
  q v_id_b_$tr "{$B}" $T 5000 48 30
  q v_id_d_$tr "{$D}" $T 5000 48 30
  q v1_d_$tr "{$D,$POOL,$V1}" $T 5000 48 30
  for k in 1 2 3; do
    q v1_d_j05_s${k}_$tr "{$D,$POOL,$V1}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[1]} --jitter_seed $k
  done
  q v2_b_$tr "{$B,$POOL,$V2}" $T 5000 48 30
  q v2_c_$tr "{$C,$POOL,$V2}" $T 5000 48 30
  q v2_d_$tr "{$D,$POOL,$V2}" $T 5000 48 30
  i=0
  for jn in 005 05 45; do
    for k in 1 2 3 4 5; do
      q v2_d_j${jn}_s${k}_$tr "{$D,$POOL,$V2}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[$i]} --jitter_seed $k
    done
    i=$((i+1))
  done
done
wait
echo ALLDONE
