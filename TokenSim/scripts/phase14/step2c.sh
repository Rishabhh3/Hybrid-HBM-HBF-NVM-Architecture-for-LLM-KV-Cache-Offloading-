#!/bin/bash
# PHASE 14 STEP 2b(c) driver: clean replicates with burst-level jitter.
#
#   s2c_id_*        new code, no jitter flag: byte-identity against ced3806 (s0_*).
#   s2c_req_d_ta    --arrival_jitter_mode request, j 0.5, seed 1, cap 48: must
#                   reproduce s2b_j1_c48_d_ta (the run made before the mode flag
#                   existed) byte for byte.
#   s2c_j<k>_*      --arrival_jitter_mode burst --jitter_seed k, k = 1..5.
#                   j = 0.45 x the minimum inter-burst gap in replay time
#                   (scripts/phase14/burst_gaps.py, W = 5,000, 30 QPS):
#                   Conversation 0.45 x 313.706 ms = 141.168 ms,
#                   Tool&Agent   0.45 x 605.737 ms = 272.582 ms.
#                   A methodology choice, not a hardware number.
#
# Configs (b) and (d), both traces, caps 48 and 256, 30 QPS, W = 5,000, charged
# timing, stock values.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$HERE/run_one.py"

B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
launch(){ local tag=$1; shift; nohup "$RUN" "$tag" "$@" > "$WORK/runs/$tag.log" 2>&1 & }
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for cfg in b d; do
  if [ $cfg = b ]; then P="$B"; else P="$D"; fi
  for tr in conv ta; do
    if [ $tr = conv ]; then T=conv_5000.jsonl; else T=toolagent_5000.jsonl; fi
    q s2c_id_${cfg}_${tr} "{$P}" $T 5000 48 30
  done
done
q s2c_req_d_ta "{$D}" toolagent_5000.jsonl 5000 48 30 0 --arrival_jitter_ms 0.5 --jitter_seed 1 --arrival_jitter_mode request
for cap in 48 256; do
  for k in 1 2 3 4 5; do
    for cfg in b d; do
      if [ $cfg = b ]; then P="$B"; else P="$D"; fi
      for tr in conv ta; do
        if [ $tr = conv ]; then T=conv_5000.jsonl; J=141.168; else T=toolagent_5000.jsonl; J=272.582; fi
        q s2c_j${k}_c${cap}_${cfg}_${tr} "{$P}" $T 5000 $cap 30 0 --arrival_jitter_mode burst --arrival_jitter_ms $J --jitter_seed $k
      done
    done
  done
done
wait
echo ALLDONE
