#!/bin/bash
# PHASE 14 STEP 3b driver: robustness check, HBM evictions spill to PCM.
# NOT the headline. hbm_evict_save_policy = spill_only: the prefill/admission
# save is off; a block enters the store only when HBM evicts it.
# Same configuration as the headline set otherwise: 71,104 per_rank (4 x 17,776),
# HBF 5,033,164, cap 48, count_hbf_readback on.
#   sp_{b,c}_*                  single unjittered runs
#   sp_d_*, sp_d_j{005,05,45}_s<k>_*  (d) pooled set, N = 16 per trace
#   sp_id_{b,d}_*               new code, flags at default: byte identity
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK}"
export WORK
cd "$ROOT"
RUN="$HERE/run_one.py"
HL='"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"per_rank","store_partition_ranks":4,"count_hbf_readback":true,"hbm_evict_save_policy":"spill_only"'
B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }
for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; J=(1.569 15.685 141.168); else T=toolagent_5000.jsonl; J=(3.029 30.287 272.582); fi
  q sp_b_$tr "{$B,$HL}" $T 5000 48 30
  q sp_c_$tr "{$C,$HL}" $T 5000 48 30
  q sp_d_$tr "{$D,$HL}" $T 5000 48 30
  i=0
  for jn in 005 05 45; do
    for k in 1 2 3 4 5; do
      q sp_d_j${jn}_s${k}_$tr "{$D,$HL}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[$i]} --jitter_seed $k
    done
    i=$((i+1))
  done
  q sp_id_b_$tr "{$B}" $T 5000 48 30
  q sp_id_d_$tr "{$D}" $T 5000 48 30
done
wait
echo ALLDONE
