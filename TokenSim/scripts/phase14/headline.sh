#!/bin/bash
# PHASE 14 headline set: design capacity (43.4 GiB/GPU, binary Gb confirmed),
# per_rank, cap 48, both traces, count_hbf_readback on.
#
#   PCM  per_rank 4 x 17,776    = 71,104 blocks   (4 writing TP-0 ranks)
#   HBF  per_rank 4 x 1,258,291 = 5,033,164 blocks
#
#   hl_{b,c}_*                (b), (c): single unjittered runs.
#   hl_d_*                    (d) unjittered (stock arrival times).
#   hl_d_j{005,05,45}_s<k>_*  (d) burst jitter, seeds 1..5, at j = 0.005 / 0.05 /
#                             0.45 x min inter-burst gap
#                             (Conv 1.569 / 15.685 / 141.168 ms;
#                              T&A  3.029 / 30.287 / 272.582 ms).
#   rb142_{b,c,d}_*           the 86.8 GiB/GPU point (142,213, global, PHASE 9
#                             stock), unjittered, count_hbf_readback on.
#   hl_id_{b,d}_*             new code, every flag at default: byte identity.
# j values are methodology choices, not hardware numbers.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

HL='"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"per_rank","store_partition_ranks":4,"count_hbf_readback":true'
RB='"count_hbf_readback":true'
B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; J=(1.569 15.685 141.168); else T=toolagent_5000.jsonl; J=(3.029 30.287 272.582); fi
  q hl_b_$tr "{$B,$HL}" $T 5000 48 30
  q hl_c_$tr "{$C,$HL}" $T 5000 48 30
  q hl_d_$tr "{$D,$HL}" $T 5000 48 30
  i=0
  for jn in 005 05 45; do
    for k in 1 2 3 4 5; do
      q hl_d_j${jn}_s${k}_$tr "{$D,$HL}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms ${J[$i]} --jitter_seed $k
    done
    i=$((i+1))
  done
  q rb142_b_$tr "{$B,$RB}" $T 5000 48 30
  q rb142_c_$tr "{$C,$RB}" $T 5000 48 30
  q rb142_d_$tr "{$D,$RB}" $T 5000 48 30
  q hl_id_b_$tr "{$B}" $T 5000 48 30
  q hl_id_d_$tr "{$D}" $T 5000 48 30
done
wait
echo ALLDONE
