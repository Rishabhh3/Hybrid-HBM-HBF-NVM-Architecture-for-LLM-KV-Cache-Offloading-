#!/usr/bin/env python3
"""One PHASE 14 run: scripts/phase12/run_one.py with an optional seed.

    WORK=<dir> run_one.py TAG '<json overrides>' TRACE W CAP QPS [SEED [EXTRA CLI ARGS...]]

Identical to the PHASE 12 harness (same BASE config, same command line) except
that --random_seed takes SEED instead of the fixed 0, and any further arguments
are appended to the benchmark.py command line (e.g. --arrival_jitter_ms 0.5
--jitter_seed 3). SEED defaults to 0, so a run without either is the PHASE 12
command exactly.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase12"))
import run_one as phase12  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:7]
    seed = sys.argv[7] if len(sys.argv) > 7 else "0"
    cmd = phase12.command(*args)
    cmd[cmd.index("--random_seed") + 1] = seed
    cmd += sys.argv[8:]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=phase12.ROOT).returncode)
