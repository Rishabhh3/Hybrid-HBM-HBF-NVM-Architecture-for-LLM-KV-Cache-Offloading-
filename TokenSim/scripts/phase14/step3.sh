#!/bin/bash
# PHASE 14 STEP 3 driver: (d) concurrency sweep with burst-jitter replicates,
# plus the j = 0.005 check.
#
#   s3_c<cap>_s<k>_d_*  (d), caps {8,16,32,48,64,96,192,256}, seeds 1..3,
#                       burst jitter j = 0.05 x min inter-burst gap
#                       (Conversation 15.685 ms, Tool&Agent 30.287 ms).
#                       The cap-48 runs must reproduce p3_js<k>_d_* byte for byte.
#   s3_jt<k>_d_*        (d), cap 48, j = 0.005 x min gap
#                       (Conversation 1.569 ms, Tool&Agent 3.029 ms), seeds 1..5.
# j values are methodology choices, not hardware numbers. (b) is not run: the
# stock PHASE 9 STEP 4 rows are reused.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for k in 1 2 3 4 5; do
  q s3_jt${k}_d_conv "{$D}" conv_5000.jsonl      5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 1.569 --jitter_seed $k
  q s3_jt${k}_d_ta   "{$D}" toolagent_5000.jsonl 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 3.029 --jitter_seed $k
done
for cap in 8 16 32 48 64 96 192 256; do
  for k in 1 2 3; do
    q s3_c${cap}_s${k}_d_conv "{$D}" conv_5000.jsonl      5000 $cap 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 15.685 --jitter_seed $k
    q s3_c${cap}_s${k}_d_ta   "{$D}" toolagent_5000.jsonl 5000 $cap 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 30.287 --jitter_seed $k
  done
done
wait
echo ALLDONE
