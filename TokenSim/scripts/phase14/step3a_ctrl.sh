#!/bin/bash
# PHASE 14 STEP 3a control runs: pools sized for the 4 store-writing ranks.
#
# Only TP-0/PP-0 workers are scheduled (llm_engine.py:577-579), so 4 of 8 ranks
# write to the store. The TP-1 stream is an exact mirror of TP-0 and is not
# simulated. Every tier is therefore sized for the 4 writing ranks:
#   PCM  4 x 17,776    = 71,104 blocks    = 43.4 GiB per writing GPU
#   HBF  4 x 1,258,291 = 5,033,164 blocks = 3 TiB per writing GPU
#
#   c48g_*   store_partition = count (behaviourally global, counters on):
#            the "43.4 GiB/GPU design point".
#   c48p_*   store_partition = per_rank, 4 partitions of 17,776 / 1,258,291.
# (b), (c): one unjittered run each. (d): unjittered + burst jitter
# j = 0.05 x min gap (15.685 / 30.287 ms) seeds 1..3.
# Cap 48, 30 QPS, W = 5,000, charged timing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

CAP='"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164'
G='"store_partition":"count"'
PR='"store_partition":"per_rank","store_partition_ranks":4'
B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for arm in g p; do
  if [ $arm = g ]; then M="$G"; else M="$PR"; fi
  for tr in conv ta; do
    if [ $tr = conv ]; then T=conv_5000.jsonl; J=15.685; else T=toolagent_5000.jsonl; J=30.287; fi
    q c48${arm}_b_$tr "{$B,$CAP,$M}" $T 5000 48 30
    q c48${arm}_c_$tr "{$C,$CAP,$M}" $T 5000 48 30
    q c48${arm}_d_$tr "{$D,$CAP,$M}" $T 5000 48 30
    for k in 1 2 3; do
      q c48${arm}_d_s${k}_$tr "{$D,$CAP,$M}" $T 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms $J --jitter_seed $k
    done
  done
done
wait
echo ALLDONE
