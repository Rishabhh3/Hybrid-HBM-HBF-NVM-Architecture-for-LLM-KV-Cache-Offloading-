#!/usr/bin/env python3
"""PHASE 14 HBF capacity check (step 3 of the post-STEP-4 list).

    WORK=<dir> hbfcap_schedule.py [jit]

hc_id_{b,d}_<tr>  new code, every flag at default, W = 5,000: byte identity
                  against ced3806 (s0_*).
hc_u_{b,c}_<tr>   full trace, headline configuration, hbf_capacity_mode =
                  unlimited (SENSITIVITY, not a hardware value), with
                  count_pcm_admit_source on.
hc_f_{b,c,d}_<tr> full trace, headline configuration (finite HBF), with
                  count_pcm_admit_source on: must reproduce s4_{b,c,d}_<tr> on
                  every other key, and gives the PCM admission shares.

hc_f_d_j05_s{1,2,3}_<tr>  (argument "jit") the STEP 4 jittered (d) runs (burst
                  jitter j = 0.05 x min gap) with count_pcm_admit_source on, so
                  (d)'s PCM admission shares are N = 4 like every other (d)
                  figure. Must reproduce s4_d_j05_s{1,2,3}_<tr> on every other key.

(d) under unlimited HBF is not run: (d) evicts nothing from HBF on either full
trace (STEP 4), so its capacity never binds and the mode cannot change it
(tests/test_hbf_capacity_mode.py, test_identical_when_capacity_never_binds).

Same scheduling rules as STEP 4: at most 2 at a time, never two T&A (b)/(c)
together, a run starts only when MemAvailable exceeds its estimate plus a
margin, and each run is wrapped in /usr/bin/time -v.
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
      '"read_credit":"served","load_span":"missing","count_hbf_readback":true,"count_if_read_gate":true,'
      '"count_pcm_admit_source":true')
UNLIMITED = '"hbf_capacity_mode":"unlimited"'
CFG = {
    "b": '"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"',
    "c": '"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"always"',
    "d": '"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"',
}
FULL = {"conv": ("conv_full.jsonl", "12031"), "ta": ("toolagent_full.jsonl", "23608")}
JITTER_MS = {"conv": "16.989", "ta": "33.338"}
W5 = {"conv": "conv_5000.jsonl", "ta": "toolagent_5000.jsonl"}
MAX_PARALLEL = 2
MARGIN_GIB = 3.0
# Peak RSS from STEP 4 (finite); unlimited HBF keeps every written key, so its
# estimate is padded.
EST = {("ta", "b"): 11.2, ("ta", "c"): 11.2, ("ta", "d"): 9.2,
       ("conv", "b"): 9.3, ("conv", "c"): 9.3, ("conv", "d"): 7.6}


def jobs():
    out = []
    for tr in ("ta", "conv"):
        trace, w = FULL[tr]
        for cfg in ("b", "c"):
            out.append((f"hc_u_{cfg}_{tr}", tr, cfg, EST[tr, cfg] * 1.5,
                        [f"{{{CFG[cfg]},{HL},{UNLIMITED}}}", trace, w, "48", "30"]))
        for cfg in ("b", "c", "d"):
            out.append((f"hc_f_{cfg}_{tr}", tr, cfg, EST[tr, cfg] * 1.1,
                        [f"{{{CFG[cfg]},{HL}}}", trace, w, "48", "30"]))
        for cfg in ("b", "d"):
            out.append((f"hc_id_{cfg}_{tr}", tr, "id", 4.0, [f"{{{CFG[cfg]}}}", W5[tr], "5000", "48", "30"]))
    return out


def jit_jobs():
    out = []
    for tr in ("ta", "conv"):
        trace, w = FULL[tr]
        for k in (1, 2, 3):
            out.append((f"hc_f_d_j05_s{k}_{tr}", tr, "d", EST[tr, "d"] * 1.1,
                        [f"{{{CFG['d']},{HL}}}", trace, w, "48", "30", "0",
                         "--arrival_jitter_mode", "burst", "--arrival_jitter_ms", JITTER_MS[tr],
                         "--jitter_seed", str(k)]))
    return out


def mem_available_gib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def main():
    pending = jit_jobs() if sys.argv[1:] == ["jit"] else jobs()
    running = {}
    while pending or running:
        for tag, (proc, *_rest) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "done", tag, "rc", proc.returncode, flush=True)
                del running[tag]
        launched = False
        if len(running) < MAX_PARALLEL:
            for i, (tag, tr, cfg, est, args) in enumerate(pending):
                big = tr == "ta" and cfg in "bc"
                if big and any(t == "ta" and c in "bc" for (_p, t, c) in running.values()):
                    continue
                if mem_available_gib() < est + MARGIN_GIB:
                    continue
                cmd = ["/usr/bin/time", "-v", os.path.join(HERE, "run_one.py"), tag, *args]
                log = open(os.path.join(RUNS, f"{tag}.log"), "w")
                running[tag] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), tr, cfg)
                print(time.strftime("%H:%M:%S"), "start", tag, f"est {est:.1f} GiB, "
                      f"avail {mem_available_gib():.1f} GiB", flush=True)
                pending.pop(i)
                launched = True
                break
        time.sleep(5 if launched else 20)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
