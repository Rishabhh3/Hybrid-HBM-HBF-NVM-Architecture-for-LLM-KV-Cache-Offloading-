# PHASE 12 harness — HBM-eviction falsifier

Rebuilds the workload files and drives the runs recorded in
`docs/findings.md` PHASE 12. Everything here is reproducible from a clean
checkout; only the trace files themselves are not committed (33 MB / 21 MB).

## 1. Workload

The two FAST'25 release traces are the input:

```
https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl
https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
```

| file | md5 |
|---|---|
| `conversation_trace.jsonl` | `bc2f368504912f72f59caee90b5f2e46` |
| `toolagent_trace.jsonl` | `53cf2bc5eff72e9953348bda80028c93` |
| `conv_5000.jsonl` (built) | `5a19d814562e650fe6e44e4234a4793c` |
| `toolagent_5000.jsonl` (built) | `c6cd3b0c8022eebb88358719a2596f1c` |

```bash
WORK=/path/to/workdir
python scripts/phase12/build_trace.py $WORK/conversation_trace.jsonl $WORK/conv_5000.jsonl      5000
python scripts/phase12/build_trace.py $WORK/toolagent_trace.jsonl    $WORK/toolagent_5000.jsonl 5000
```

`build_trace.py` was reconstructed from the recipe in `docs/findings.md`
(~line 2277); the original is not in this tree. It is validated by
reproduction, not by inspection — see below.

## 2. Runs

```bash
WORK=$WORK scripts/phase12/step3.sh            # the ten PHASE 12 runs
WORK=$WORK scripts/phase12/run_one.py TAG '<json overrides>' TRACE W CAP QPS   # one run
```

`run_one.py` holds the config (b) knob set: the 347.2 GiB PCM pool
(142,213 blocks), the 24 TiB HBF pool (10,066,329 blocks), HBF read/write
timings, and `charge_eviction_writes`. Overrides are merged over that base.

## 3. Reproduction gate

Before trusting any PHASE 12 number, the rebuilt workload must reproduce the
PHASE 9 STEP 3 config (b) row at cap 48 to every digit:

```bash
WORK=$WORK scripts/phase12/run_one.py gate_conv \
  '{"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"}' \
  conv_5000.jsonl 5000 48 30
```

| | HBF write bytes | HBF write blocks | hit rate | reuse hit blocks | batch/sched |
|---|---|---|---|---|---|
| Conversation | 16,853,376,696,320 | 6,429,053 | 0.1643 | 670,585 | 23.167889 |
| Tool&Agent | 8,483,936,665,600 | 3,236,365 | 0.4037 | 1,174,042 | 22.885048 |

## 4. STEP C

`stepc.py` is trace analysis only — it never starts the simulator.

```bash
python scripts/phase12/stepc.py $WORK
```

## 5. STEP 1 (addendum) — continuations in tokens, and the export audit

```bash
python scripts/phase12/stepc_tokens.py $WORK    # trace analysis only
python scripts/phase12/audit_export.py $WORK <kv_config.json>
```

`audit_export.py` runs a 60-request simulation with `LLMResult` wrapped to
capture every keyword `export_result` hands it, then diffs against the declared
field set. It patches nothing in the tree. Pass a config with
`"hbm_evict_save_policy": "count"` to make the check meaningful.

## 6. PHASE 13 — corrections, limitation, tail-sharing sensitivity

```bash
python scripts/phase12/generated_share.py  $WORK   # generated KV as a share of all KV
python scripts/phase12/tail_divergence.py  $WORK   # the unreachable prompt tail
python scripts/phase12/build_trace.py $WORK/conversation_trace.jsonl \
       $WORK/conv_5000_tailshare.jsonl 5000 --share_partial_tail
```

`--share_partial_tail` is a **workload variant, not a correction** — see PHASE 13
in `findings.md`. Default off reproduces the original files byte for byte.

| built with `--share_partial_tail` | md5 |
|---|---|
| `conv_5000_tailshare.jsonl` | `13d05f75aaaae8669619ae88efbb4998` |
| `toolagent_5000_tailshare.jsonl` | `edcd7c1028738dff7c0228fd175ea399` |
