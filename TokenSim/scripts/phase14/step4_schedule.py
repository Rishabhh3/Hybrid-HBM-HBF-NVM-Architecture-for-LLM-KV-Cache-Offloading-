#!/usr/bin/env python3
"""PHASE 14 STEP 4 scheduler: full traces, headline configuration.

    WORK=<dir> step4_schedule.py

Runs the 13 runs that follow the T&A (b) calibration (s4_b_ta): T&A (c),
T&A (d) x 4, Conv (b), (c), (d) x 4, and one no-store run per trace. At most
2 at a time, never two T&A (b)/(c) together, and a run starts only when
MemAvailable exceeds its estimated peak RSS plus a margin. Each run is wrapped
in /usr/bin/time -v so its peak RSS is recorded in its log.

j = 0.05 x min inter-burst gap on the full traces: 16.989 ms (Conv),
33.338 ms (T&A). A methodology choice, not a hardware number.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ["WORK"]
RUNS = os.path.join(WORK, "runs")
HL = ('"memory_capacity_blocks":71104,"ssd_capacity_blocks":5033164,"store_partition":"per_rank",'
      '"store_partition_ranks":4,"store_lookup_start":"after_hbm","hbm_evict_save_policy":"spill_only",'
      '"read_credit":"served","load_span":"missing","count_hbf_readback":true,"count_if_read_gate":true')
CFG = {
    "b": '"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"',
    "c": '"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"',
    "d": '"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"',
}
TRACE = {"conv": ("conv_full.jsonl", "12031", "16.989"), "ta": ("toolagent_full.jsonl", "23608", "33.338")}
MAX_PARALLEL = 2
MARGIN_GIB = 3.0


def est_gib(tr, cfg):
    calib = float(os.environ.get("TA_B_PEAK_GIB", "14.1"))
    if cfg == "ns":
        return 4.0
    if tr == "ta":
        return calib if cfg in "bc" else calib * 2.44 / 2.98
    conv_b = calib * (4.83 / 2.98) * (12031 / 23608)
    return conv_b if cfg in "bc" else conv_b * 2.44 / 2.98


def jobs():
    out = []
    for tr in ("ta", "conv"):
        trace, w, j = TRACE[tr]
        for cfg in ("b", "c", "d") if tr == "conv" else ("c", "d"):
            out.append((f"s4_{cfg}_{tr}", tr, cfg, [f"{{{CFG[cfg]},{HL}}}", trace, w, "48", "30"]))
        for k in (1, 2, 3):
            out.append((f"s4_d_j05_s{k}_{tr}", tr, "d",
                        [f"{{{CFG['d']},{HL}}}", trace, w, "48", "30", "0",
                         "--arrival_jitter_mode", "burst", "--arrival_jitter_ms", j, "--jitter_seed", str(k)]))
        out.append((f"s4_ns_{tr}", tr, "ns", [trace, w, "48", "30"]))
    return out


def mem_available_gib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def main():
    pending = jobs()
    running = {}
    while pending or running:
        for tag, (proc, *_rest) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "done", tag, "rc", proc.returncode, flush=True)
                del running[tag]
        launched = False
        if len(running) < MAX_PARALLEL:
            for i, (tag, tr, cfg, args) in enumerate(pending):
                big = tr == "ta" and cfg in "bc"
                if big and any(t == "ta" and c in "bc" for (_p, t, c) in running.values()):
                    continue
                need = est_gib(tr, cfg) + MARGIN_GIB
                if mem_available_gib() < need:
                    continue
                script = "run_nostore.py" if cfg == "ns" else "run_one.py"
                cmd = ["/usr/bin/time", "-v", os.path.join(HERE, script), tag, *args]
                log = open(os.path.join(RUNS, f"{tag}.log"), "w")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                running[tag] = (proc, tr, cfg)
                print(time.strftime("%H:%M:%S"), "start", tag, f"est {est_gib(tr, cfg):.1f} GiB, "
                      f"avail {mem_available_gib():.1f} GiB", flush=True)
                pending.pop(i)
                launched = True
                break
        time.sleep(5 if launched else 20)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
