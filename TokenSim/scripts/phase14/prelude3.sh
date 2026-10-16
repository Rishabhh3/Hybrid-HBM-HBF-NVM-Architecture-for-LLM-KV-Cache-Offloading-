#!/bin/bash
# PHASE 14 STEP 3 prelude driver.
#
#   p3_js<k>_d_*   (d), cap 48, burst jitter j = 0.05 x min inter-burst gap
#                  (Conversation 0.05 x 313.706 ms = 15.685 ms,
#                   Tool&Agent   0.05 x 605.737 ms = 30.287 ms), seeds 1..5.
#                  Jitter-size check against the j = 0.45 replicates (s2c_*).
#   p3_c<k>_c_*    (c) = +PCM, write_back + always, cap 48, burst jitter
#                  j = 0.45 x min gap (141.168 / 272.582 ms), seeds 1..5.
# j values are methodology choices, not hardware numbers.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

C='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for k in 1 2 3 4 5; do
  q p3_js${k}_d_conv "{$D}" conv_5000.jsonl      5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 15.685  --jitter_seed $k
  q p3_js${k}_d_ta   "{$D}" toolagent_5000.jsonl 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 30.287  --jitter_seed $k
  q p3_c${k}_c_conv  "{$C}" conv_5000.jsonl      5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 141.168 --jitter_seed $k
  q p3_c${k}_c_ta    "{$C}" toolagent_5000.jsonl 5000 48 30 0 --arrival_jitter_mode burst --arrival_jitter_ms 272.582 --jitter_seed $k
done
wait
echo ALLDONE
