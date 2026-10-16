#!/bin/bash
# PHASE 14 STEP 3a driver: cross-rank eviction in the shared store.
#
#   s3a_id_{b,d}_*  new code, every flag at default: byte-identity against
#                   ced3806 (s0_*).
#   s3a_def_c_*     config (c) at default: the stock (c) reference (no cap-48 (c)
#                   result survives from PHASE 9; checked against its recorded
#                   values instead).
#   s3a_cnt_*       store_partition = count: counters only.
#   s3a_pr_*        store_partition = per_rank, store_partition_ranks = 8:
#                   per-rank LRU and capacity on both tiers, floored:
#                   PCM  floor(142,213 / 8)    = 17,776 blocks/rank (5 blocks unused)
#                   HBF  floor(10,066,329 / 8) = 1,258,291 blocks/rank (1 unused)
#
# Configs (b), (c), (d), cap 48, 30 QPS, W = 5,000, unjittered, charged timing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for tr in conv ta; do
  if [ $tr = conv ]; then T=conv_5000.jsonl; else T=toolagent_5000.jsonl; fi
  q s3a_id_b_$tr  "{$B}" $T 5000 48 30
  q s3a_id_d_$tr  "{$D}" $T 5000 48 30
  q s3a_def_c_$tr "{$C}" $T 5000 48 30
  for cfg in b c d; do
    case $cfg in b) P="$B";; c) P="$C";; d) P="$D";; esac
    q s3a_cnt_${cfg}_$tr "{$P,\"store_partition\":\"count\"}" $T 5000 48 30
    q s3a_pr_${cfg}_$tr  "{$P,\"store_partition\":\"per_rank\",\"store_partition_ranks\":8}" $T 5000 48 30
  done
done
wait
echo ALLDONE
