#!/usr/bin/env python3
"""PHASE 14 no-store control: the PHASE 12 command with the KV connector set to
NoopConnector, i.e. no Mooncake store at all. Everything else (model, cluster,
hardware, cap, QPS, trace, seed) is identical.

    WORK=<dir> run_nostore.py TAG TRACE W CAP QPS
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase12"))
import run_one as phase12  # noqa: E402

if __name__ == "__main__":
    tag, trace, w, cap, qps = sys.argv[1:6]
    cmd = phase12.command(tag, "{}", trace, w, cap, qps)
    config_path = cmd[cmd.index("--kv_transfer_config") + 1]
    with open(config_path, "w") as handle:
        json.dump({"kv_connector": "NoopConnector"}, handle, indent=2)
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, cwd=phase12.ROOT).returncode)
