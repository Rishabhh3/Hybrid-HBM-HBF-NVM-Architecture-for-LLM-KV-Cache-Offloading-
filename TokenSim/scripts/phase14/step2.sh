#!/bin/bash
# PHASE 14 STEP 2 driver: do the endurance numbers depend on timing at all?
# Configs (b) and (d), cap 48, 30 QPS, W = 5,000, both traces.
#
#   id_*      every new flag at default: the byte-identity pair against main.
#   ref_*     count_aligned_save_skips only (instrumentation, no behaviour
#             change): the charged reference every other arm is compared to.
#   free_*    store_timing_mode = free.
#   ssd_*     HBF read at the in-tree stock SSD default (100 us, 7.0 GiB/s)
#             instead of the LHB perfect-prefetch setting.
#             LABEL: stock SSD default, not HBF.
#   plat*_*   pcm_write_latency_us at 0.25x / 4x of the assumed 1.0 us.
#   pbw*_*    pcm_write_bw_gbps at 0.25x / 4x of the assumed 20.0 GiB/s.
#             LABEL: sensitivity around an assumption, not values.
#             Varied one at a time. Inert under config (b) (memory_media=dram),
#             run anyway so every arm has both configs.
#
# Usage: WORK=<dir holding conv_5000.jsonl> scripts/phase14/step2.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
: "${WORK:?set WORK to the directory holding conv_5000.jsonl}"
export WORK
cd "$ROOT"
mkdir -p "$WORK/runs"
RUN="$ROOT/scripts/phase12/run_one.py"

B='"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"'
D='"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"'
C='"count_aligned_save_skips":true'
launch(){ nohup "$RUN" "$1" "$2" "$3" 5000 48 30 > "$WORK/runs/$1.log" 2>&1 & }
# ~1.7 GB per run; eight fit in this machine's free memory.
q(){ while [ "$(pgrep -cf 'benchmark.py' || echo 0)" -ge 8 ]; do sleep 10; done; launch "$@"; sleep 3; }

for cfg in b d; do
  if [ $cfg = b ]; then P="$B"; else P="$D"; fi
  for tr in conv ta; do
    if [ $tr = conv ]; then T=conv_5000.jsonl; else T=toolagent_5000.jsonl; fi
    q s2_id_${cfg}_${tr}     "{$P}"                                               $T
    q s2_ref_${cfg}_${tr}    "{$P,$C}"                                            $T
    q s2_free_${cfg}_${tr}   "{$P,$C,\"store_timing_mode\":\"free\"}"             $T
    q s2_ssd_${cfg}_${tr}    "{$P,$C,\"ssd_read_latency_us\":100,\"ssd_read_bw_gbps\":7.0}" $T
    q s2_plat025_${cfg}_${tr} "{$P,$C,\"pcm_write_latency_us\":0.25}"             $T
    q s2_plat4_${cfg}_${tr}  "{$P,$C,\"pcm_write_latency_us\":4.0}"               $T
    q s2_pbw025_${cfg}_${tr} "{$P,$C,\"pcm_write_bw_gbps\":5.0}"                  $T
    q s2_pbw4_${cfg}_${tr}   "{$P,$C,\"pcm_write_bw_gbps\":80.0}"                 $T
  done
done
wait
echo ALLDONE
