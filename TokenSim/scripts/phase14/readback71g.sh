#!/bin/bash
# PHASE 14: de-confound the read-back test. (d) at global @ 71,104 (the
# STEP 3a c48g_* controls: store_partition = count, behaviourally global),
# now with count_hbf_readback on, so global 142,213 vs global 71,104 differ
# only in capacity. Unjittered + burst jitter j = 0.05 x gap seeds 1..3.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK}"
export WORK
cd "$ROOT"
RUN="$HERE/run_one.py"
P='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read","memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"count","count_hbf_readback":true'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; J=15.685; else T=toolagent_5000.jsonl; J=30.287; fi
  launch rb71g_d_$tr "{$P}" $T 5000 48 30
  for k in 1 2 3; do
    launch rb71g_d_s${k}_$tr "{$P}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms $J --jitter_seed $k
  done
done
wait
echo ALLDONE
