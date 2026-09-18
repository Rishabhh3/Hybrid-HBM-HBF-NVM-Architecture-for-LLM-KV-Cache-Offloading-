# Findings — PCM tier write-sensitivity extension

Autonomous run, 2026-09-11. Scope: test whether the write-media-latency
elasticity established for the standing Mooncake workload (8 MiB MHA block,
~1.31 off / ~1.34 ON) is block-size independent, and whether a bandwidth-driven
write-amplification proxy agrees with a fixed-latency-driven sweep at the same
block size. All runs use the standing workload: Mooncake FAST'25
`conversation_trace.jsonl`, first 500 requests, 3 QPS (`--trace_target_qps 3`),
H200x8, 512→16 token re-granulation (32 sub-blocks/trace hash),
`memory_capacity_blocks=4096`, `ssd_capacity_blocks=262144`,
`workload_type qwen_jsonl`, `--request_count 500`. `charge_eviction_writes`
swept False/True (ON) throughout. Elasticity = (%Δ mean e2e) / (%Δ write
media latency), endpoint method (first vs. last sweep point) — reproduces the
CLAUDE.md-quoted 1.31/1.34 and 0.06/0.03 figures from the original A_write/
B_read sweeps to 3 decimal places, so the same method is used here.
No run had unfinished requests or preemptions (`notdone=0`, `preempt=0`
everywhere below).

## Reference — 8 MiB write-latency sweep (recomputed this session)

Model `llama-7b.json` (MHA, block_size=16 tokens → 8,388,608 B/block),
cluster `8_h200/h8.json` (no TP/DP override), `ssd_write_bw_gbps=3.0` fixed,
`ssd_write_latency_us` swept 100/187/281/375. This reproduces the standing
result and serves as this session's same-block-size, same-mechanism baseline
for the Phase 2 comparison.

| wlat (us) | charge | media latency (us) | mean e2e | p99 e2e | tok/s | mem_evict | ssd_evict |
|---|---|---|---|---|---|---|---|
| 100 | off | 2704.17 | 64.5387 | 156.8301 | 25625.3 | 357069 | 99021 |
| 100 | ON | 2704.17 | 152.8005 | 308.3001 | 16413.4 | 359557 | 101509 |
| 187 | off | 2791.17 | 67.0859 | 159.2844 | 25251.5 | 357488 | 99440 |
| 187 | ON | 2791.17 | 160.2832 | 318.4820 | 16043.1 | 360161 | 102113 |
| 281 | off | 2885.17 | 70.3034 | 168.1346 | 24692.6 | 358860 | 100812 |
| 281 | ON | 2885.17 | 166.3373 | 323.6291 | 15811.7 | 358669 | 100621 |
| 375 | off | 2979.17 | 73.1182 | 173.7321 | 24235.2 | 359088 | 101040 |
| 375 | ON | 2979.17 | 173.6705 | 337.8157 | 15276.1 | 360870 | 102822 |

**Elasticity (8 MiB, latency sweep): off = 1.3072, ON = 1.3431.** Matches
CLAUDE.md's quoted ~1.31/1.34. Throughput (tok/s) and mean e2e both move
monotonically with wlat under both flags; eviction counts are flat within
noise (~357–361k mem_evict, ~99–103k ssd_evict) across all 8 points — no
scheduling-feedback anomaly.

## PHASE 1 — Block-size independence check

**Step 1:** ran the write-latency sweep (same wlat grid, same bw=3.0, same
tier capacities, same trace/QPS/cluster) at a GQA-style KV block instead of
8 MiB. Model `LLaMa2-70B-GQA` (stock entry in
`TransformerRoofline/hardware_models.json`: Nhead=64, Grouped_Num=8,
Dmodel=8192, Nlayer=80), `--tensor_parallel_size 2 --data_parallel_size 4` on
the 8×H200 cluster (required to fit the 70B model). Measured block size:
**2,621,440 B = 2.5 MiB** per replica (TP=2 halves the per-worker KV heads),
the closest stock GQA configuration to the requested "~2 MiB"; reported as
measured rather than forced to an arbitrary 2 MiB.

| wlat (us) | charge | media latency (us) | mean e2e | p99 e2e | tok/s | mem_evict | ssd_evict |
|---|---|---|---|---|---|---|---|
| 100 | off | 913.80 | 138.2125 | 265.4240 | 18225.7 | 406218 | 148170 |
| 100 | ON | 913.80 | 202.3318 | 370.9370 | 14585.5 | 404652 | 146604 |
| 187 | off | 1000.80 | 144.0633 | 274.5243 | 17778.6 | 405335 | 147287 |
| 187 | ON | 1000.80 | 215.1216 | 395.3949 | 13906.5 | 405260 | 147212 |
| 281 | off | 1094.80 | 150.6790 | 271.6963 | 17349.4 | 405382 | 147334 |
| 281 | ON | 1094.80 | 229.1384 | 427.3045 | 13460.5 | 405252 | 147204 |
| 375 | off | 1188.80 | 156.5825 | 290.1606 | 16744.1 | 405382 | 147334 |
| 375 | ON | 1188.80 | 243.0723 | 448.3979 | 13061.8 | 402589 | 144541 |

**Step 2:** elasticity at 2.5 MiB: **off = 0.4417, ON = 0.6691.**
Both mean e2e and tok/s move monotonically with wlat under both flags;
eviction counts are flat (~403–406k mem_evict, ~145–148k ssd_evict), no
scheduling anomaly. Distinct key count is identical (410,118) across all 8
runs, as expected — block-size changes don't change the hash-key space.

### BRANCH — block-size dependence check

| | off | ON |
|---|---|---|
| Elasticity @ 8 MiB (reference) | 1.3072 | 1.3431 |
| Elasticity @ 2.5 MiB (GQA) | 0.4417 | 0.6691 |
| % difference | **−66.2%** | **−50.2%** |

**FLAGGED: the result is block-size dependent.** Both flag settings differ
from the 8 MiB reference by far more than the ~30% tolerance (−66% off,
−50% ON) — proceeding to Phase 2 per the branch rule, but this elasticity
number cannot be quoted context-free; it must always be paired with the
block size it was measured at.

Mechanically: at the smaller block, the *absolute* fixed-latency effect on
mean e2e is nearly unchanged in relative terms (off: 13.29% dy at 2.5 MiB vs.
13.30% dy at 8 MiB reference — a striking near-match on this dataset,
reported as observed, not a mechanism claim), but the wlat sweep's *own*
percent-change in media latency roughly triples (30.09% at 2.5 MiB vs.
10.17% at 8 MiB) because the bandwidth term shrinks with block size,
shrinking the denominator of the elasticity ratio. So the elasticity metric
here isn't just measuring "how much writes matter" — it's also sensitive to
how much of the media-latency denominator the fixed term occupies at a given
block size, which is a block-size artifact of the metric, not only a
workload effect.

## PHASE 2 — Bandwidth sweep (write-amplification proxy) at 8 MiB

**Step 1:** at the 8 MiB block (same model/cluster/config as the Reference
sweep above), swept `ssd_write_bw_gbps` at 3.0/1.5/1.0 (WAF 1/2/3 proxy),
`ssd_write_latency_us` held at 200 (baseline), both charge flags.

| WAF | bw (GB/s) | charge | media latency (us) | mean e2e | p99 e2e | tok/s | mem_evict | ssd_evict |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.0 | off | 2804.17 | 67.2586 | 160.0286 | 25186.5 | 357488 | 99440 |
| 1 | 3.0 | ON | 2804.17 | 160.6309 | 319.9598 | 15989.4 | 360244 | 102196 |
| 2 | 1.5 | off | 5408.33 | 154.8641 | 304.3212 | 16561.4 | 359684 | 101636 |
| 2 | 1.5 | ON | 5408.33 | 334.5228 | 666.9342 | 9613.2 | 361851 | 103803 |
| 3 | 1.0 | off | 8012.50 | 247.8665 | 474.9138 | 12802.8 | 360215 | 102167 |
| 3 | 1.0 | ON | 8012.50 | 493.7646 | 913.3764 | 7002.1 | 362244 | 104196 |

Mean e2e and tok/s move monotonically across WAF 1→2→3 under both flags
(no non-monotonic series). Eviction counts drift up only slightly with WAF
(off: 357k→360k mem_evict, 99k→102k ssd_evict; similar for ON) — not a sharp
departure. Distinct key count is flat at 372,097 across all 6 runs. Note the
shape: mean e2e roughly doubles from WAF1→WAF2 (+130% off, +108% ON) but
grows more slowly from WAF2→WAF3 (+60% off, +48% ON) — in terms of absolute
media latency (which scales linearly with WAF here) the mean-e2e-vs-media_us
curve is close to linear with a very slight upward curvature (local slope
0.0337 e2e/us on the first segment vs 0.0357 on the second, off-flag); it is
not a step-function or a scheduling-feedback cliff.

**Step 2:** elasticity vs. total write media latency: **off = 1.4458, ON = 1.1166.**

### BRANCH — corroboration check vs. Reference (8 MiB latency sweep)

| | off | ON |
|---|---|---|
| Elasticity @ 8 MiB, bandwidth sweep (Phase 2) | 1.4458 | 1.1166 |
| Elasticity @ 8 MiB, latency sweep (Reference) | 1.3072 | 1.3431 |
| % difference | **+10.6%** | **−16.9%** |

**Both within the ~40% tolerance — this is a corroboration.** At fixed block
size, expressing cost as write media latency gives a comparable elasticity
whether the fixed term (latency knob) or the bandwidth term (WAF knob) is
what's varied. The off-flag numbers agree more tightly (+10.6%) than ON
(−16.9%); ON's bigger gap is plausibly because charging eviction writes adds
concurrent SSD-bound work that competes for the same shrinking bandwidth
budget as WAF rises, which the fixed-latency sweep never exercises (raising
wlat doesn't create bandwidth contention the way lowering bw does) — flagged
as a hypothesis, not confirmed further in this run.

## Go/no-go for MQSim

**Elasticity to carry forward (vs. write media latency):** the two 8 MiB
measurements bracket **~1.1–1.4** (Reference latency-sweep: 1.31 off / 1.34
ON; Phase 2 bandwidth-sweep: 1.45 off / 1.12 ON), all four values inside a
~30% band of each other at fixed block size. **Confidence: moderate at 8
MiB, low as a universal constant** — Phase 1 showed the same metric drops to
0.44–0.67 at a 2.5 MiB block, a >2x change from a block-size difference
alone. Any single number handed to MQSim needs its block size attached.

**WAF needed for a 2x change in mean e2e latency**, solved three ways from
the Phase 2 8 MiB grid (baseline WAF=1, media latency 2804.17 us,
`ssd_write_latency_us=200` fixed):

| Method | off | ON |
|---|---|---|
| Power-law extrapolation from Phase 2's own elasticity (1.4458/1.1166) | WAF ≈ 1.66 | WAF ≈ 1.93 |
| Power-law extrapolation from the Reference elasticity (1.3072/1.3431) | WAF ≈ 1.75 | WAF ≈ 1.73 |
| Direct log-log interpolation between the measured WAF=1 and WAF=2 grid points | WAF ≈ 1.78 | WAF ≈ 1.93 |

All three methods converge: **WAF ≈ 1.7–1.9** is enough to double mean e2e
latency on this workload at 8 MiB blocks, for both charge-flag settings.
Note the measured WAF=2 grid point already *exceeds* a 2x change outright
(2.30x off, 2.08x ON mean e2e vs. WAF=1) — so "WAF=2 roughly doubles mean e2e
latency" is a defensible one-line summary directly off the grid, without
extrapolation.

**What should change in MQSim's workload generator:**
- Phase 1's finding means MQSim's access-pattern generator must be built
  against the KV block size actually planned for deployment, not an
  arbitrary reference size — the write-latency sensitivity is not a
  block-size-independent property of this workload, it moved by more than
  2x between 2.5 MiB and 8 MiB blocks in this simulator.
- The two knobs (fixed write latency vs. write bandwidth/WAF) agree well
  enough at fixed block size (within ~40%, mostly within ~17%) that MQSim
  does not need to separately calibrate against both — a single write-media-
  latency-denominated model should transfer between them, at a given block
  size.
- The ON-flag corroboration gap (−16.9%, the largest of the four
  cross-checks) suggests WAF-driven bandwidth pressure and charged-eviction
  concurrency may interact in a way a pure media-latency abstraction doesn't
  fully capture; worth a sensitivity check once MQSim is running rather than
  assuming the media-latency model transfers perfectly under charging.

---

## MQSim environment setup (post-gate, setup only — no workload generator changes)

Cloned `CMU-SAFARI/MQSim` (commit `51f0f2d`) to `/data/rishabh/MTP/MQSim` —
kept separate from an unrelated existing checkout at
`/data/rishabh/WAFexp/repos/MQSim` (different, unrelated experiment; not
reused or modified). Built clean with the shipped `Makefile`
(`g++ -std=c++11 -O3 -g`, no warnings/errors). Ran the stock binary against
the shipped baseline `ssdconfig.xml` / `workload.xml` unmodified:

```
./MQSim -i ssdconfig.xml -w workload.xml
```

All 3 scenarios in `workload.xml` completed successfully (2 synthetic I/O
flows + 1 trace-replay scenario against `traces/tpcc-small.trace`, 6,999
requests, all serviced, 0 errors). Results were written in place to
`workload_scenario_1.xml` / `_2.xml` / `_3.xml` (MQSim's convention — it
overwrites the scenario file with a `<MQSim_Results>` block containing
per-flow device response time / min / max), confirming the full
config→simulate→report pipeline works before any workload-generator design
work starts.

No changes were made to MQSim's source, config schema, or workload format in
this step — pure feasibility/setup check per the go/no-go instruction.
Workload-generator design (matching the 8 MiB deployment block size and
WAF-driven write pressure from Phases 1–2 above) is the next step, not yet
started.

## MQSim: NAND write amplification characterization

Goal: a synthetic write-heavy, low-reuse access pattern for MQSim, built
directly from TokenSim's real per-block SSD-tier behavior (not a generic
fio-style benchmark), then run against MQSim's stock SSD config to get an
honest, uncalibrated baseline WAF and read-latency read on this workload.
MQSim's own default output has no percentile fields (see Step 3) — this
whole section reports what it does expose.

### STEP 1 — extracting the real access pattern from TokenSim

TokenSim does **not** log per-block write/read/eviction timing in its
exported metrics: `MooncakeStore`'s `now` parameter is threaded through
`put_with_timing` / `lookup` / `get` / `_evict_memory_lru` (store.py) but is
only used for LRU bookkeeping, never recorded — confirmed by reading
store.py end to end before writing any instrumentation. Rather than
fabricate an access pattern from the aggregate counters (`mooncake_ssd_*`
counts only), we monkeypatched `MooncakeStore` / `OffloadTier` at runtime
(no repo source changed) to log every event at the real sim time already
passed through the existing `now=` keyword:

- `OffloadTier.write` (wrapped): every write that reaches the SSD tier,
  both direct admission (memory full) and eviction-driven (memory LRU
  victim demoted) — both funnel through this one call.
- `OffloadTier._evict_lru` (wrapped): final removal of a block from the
  SSD tier under capacity pressure.
- `MooncakeStore.get` (wrapped): every read that touches a block resident
  on the SSD tier.
- `put_with_timing` / `lookup` / `_evict_memory_lru` (wrapped): capture the
  real `now` passed by the caller so the two hooks above can timestamp
  correctly.

Rerun: the Reference 8 MiB config (`llama-7b.json`, MHA, `block_size=16`,
`8_h200/h8.json` cluster, `memory_capacity_blocks=4096`,
`ssd_capacity_blocks=262144`, `ssd_write_latency_us=200`,
`ssd_write_bw_gbps=3.0`, `charge_eviction_writes=False`), same trace/QPS as
the standing workload. Command:

```
.venv/bin/python step1_instrument.py <scratch_dir>
```

719,072 accepted SSD-tier writes captured over a 261.6 s run (matches the
Phase 2 WAF=1/off baseline exactly — same config point). Findings:

| Metric | Value |
|---|---|
| Total accepted SSD writes | 719,072 |
| Evicted by run end | 99,440 (13.8%) |
| Still resident at run end (censored) | 619,632 (86.2%) |
| Ever read before eviction/censor | 94,410 (13.13%) |
| Time-to-eviction (s), evicted subset | mean 165.14, p50 164.06, p90 173.08, p99 207.07, max 217.68 |
| Time-to-first-read (s), reused subset | mean 91.31, p50 95.92, p90 133.33, max 165.82 |
| Consecutive-write inter-arrival gap = 0 (same sim instant) | 718,831 / 719,071 pairs (99.97%) |
| Inter-arrival, nonzero-gap pairs only (s) | mean 1.090, p50 0.804, p90 2.461, p99 5.395 (n=240 distinguishable bursts) |

This confirms the CLAUDE.md characterization: write-dominated, low
read-before-evict reuse (13.13%), and — new information not in the
aggregate counters — **extremely bursty arrival**: 99.97% of consecutive
accepted writes share the exact same simulated instant (a whole prompt's
KV blocks get admitted together in one scheduling step), with real
inter-burst gaps (mean ~1.09 s) only between ~241 distinguishable bursts
over the run. No non-monotonic anomalies in this extraction — it's a raw
event log, not a derived series.

### STEP 2 — translating into an MQSim workload

**Trace-replay was chosen over MQSim's synthetic-flow generator.** MQSim's
built-in synthetic types (`STREAMING` / `RANDOM_UNIFORM` / `RANDOM_HOTCOLD`
/ `MIXED_STREAMING_RANDOM`, one scalar `Read_Percentage`) have no way to
express "13.13% of writes get exactly one read back, at an empirically
observed delay, and 99.97% of writes cluster into simultaneous bursts with
a distinct between-burst gap distribution." Trace-replay can encode all of
that directly and losslessly from real data, so it was the only option
that doesn't throw away the Step 1 findings.

**Method** (`build_mqsim_trace.py`): systematic every-k-th sample (in real
chronological order, preserving real absolute write times and real
write→first-read latencies) of the 719,072 Step 1 write-intervals — not a
parametric refit, a direct empirical downsample. Each retained write is
assigned an LBA via a bounded circular buffer (`BLOCK_SLOTS`), and each
retained "ever read" write gets a paired read line at its real first-read
offset, at the same LBA.

**Scale note (had to be revised once):** the first attempt used
`TARGET_N=50000`, `BLOCK_SLOTS=1000` (50× overwrite of an 8 GiB range).
Wall-clock calibration showed this did not finish in a practical time this
session (>25 min and still <30% through one of three planned scenarios);
rescaled to `TARGET_N=15000`, `BLOCK_SLOTS=1500` (10× overwrite of an
11.72 GiB range), calibrated at ~44 s/scenario, and used for all Step 3
runs below. (Separately: an apparent multi-minute "hang" on two earlier,
even smaller feasibility traces turned out to be a pipe-buffering artifact
of piping MQSim's carriage-return progress output through `tail` in this
sandbox, not a real simulator stall — confirmed by rerunning the identical
command with output redirected straight to a file, which completed in 9–20
s. All Step 3 runs below redirect directly to a file, never through a
pipe.)

**Address range (`BLOCK_SLOTS=1500` → 11.72 GiB) is an explicitly open
parameter** — no settled HBF/far-tier capacity figure exists yet in this
file (TokenSim's own `ssd_capacity_blocks=262144` config abstraction
implies ~2 TiB, but that number is itself a simulation parameter, not a
deployment spec). 11.72 GiB was chosen only to comfortably fit inside the
device's free space at every occupancy point tested below (worst case, 95%
occupancy → ~24 GiB nominally free) while still forcing heavy (10×) reuse
of every address — this satisfies "small capacity relative to total
writes, forcing continuous overwrite pressure" but the absolute number
should be revisited once a real far-tier capacity is settled.

Result: `step1/kv_write_heavy.trace`, 17,009 lines (15,000 writes + 2,009
reads, reuse fraction 13.50% vs. the reference 13.13% — close, expected
sampling noise from the stride-47 downsample), spanning the real 0–259.64 s
window, `Time_Unit=MICROSECOND` (TokenSim's internal clock is confirmed
seconds — `duration`/`request_count` reproduces `output_qps` exactly in
every prior sweep — so real times are scaled ×1e6 directly, no distortion).

### STEP 3 — run against stock MQSim and report

Device config: `ssdconfig.xml`, **byte-identical to the stock clone**
(`diff` confirmed) — 512 GiB physical / ~478.5 GiB usable
(`OP=0.07`), page-level FTL, `GC_Exec_Threshold=0.05`,
`GC_Hard_Threshold=0.005`, `CMT_Capacity=2,097,152` entries. Not modified.

Three `workload.xml` scenarios, identical except `Initial_Occupancy_Percentage`
(40/70/95), same trace file:

```
./MQSim -i step3/ssdconfig_stock.xml -w wl_occ40.xml > step3/run_occ40.log 2>&1
./MQSim -i step3/ssdconfig_stock.xml -w wl_occ70.xml > step3/run_occ70.log 2>&1
./MQSim -i step3/ssdconfig_stock.xml -w wl_occ95.xml > step3/run_occ95.log 2>&1
```

**Finding A — the 40/70/95% sweep produced byte-identical results.** All
three `result_occ*.xml` files have the same md5
(`157690b6c3f3bd7ea9e97d4bf4677a27`); every counter below is identical
across all three runs.

Diagnosis (from `src/ssd/FTL.cpp`): for a trace-based flow, MQSim's
`Initial_Occupancy_Percentage` preconditioning warms the **CMT (the
logical→physical mapping cache)** using a histogram built from *the
trace's own observed addresses* (`Bring_to_CMT_for_preconditioning` over
`trace_lpas_sorted_histogram`) — it is not independent background data
occupying separate physical capacity from another tenant. Our trace only
touches 1,500 distinct addresses, three orders of magnitude below
`CMT_Capacity` (2,097,152), so every occupancy setting ends up with the
entire working set trivially CMT-resident either way. **This means the
40/70/95% axis, as executed, is not a valid device-fill/GC-pressure sweep
in this MQSim version for a narrow-working-set trace — it's the same run
three times.** Flagging this rather than silently picking a different knob:
a real utilization sweep here would need either a working set close to or
above `CMT_Capacity`, or independent filler I/O spanning the full device
LBA range to actually consume the physical spare-block pool GC cares about,
or `Enabled_Preconditioning` (device-level, currently `false`) instead of
the trace-flow's `Initial_Occupancy_Percentage`.

**Finding B — measured WAF = 1.00, exactly, in all three runs.**

| Quantity | Value |
|---|---|
| Host write pages (`Total_CMT_Queries_For_Writes`, = 15,000 reqs × 1024 pages) | 15,360,000 |
| NAND page programs (`Issued_Flash_Program_CMD` + `Issued_Flash_Multiplane_Program_CMD`×2) | 510,974 + 7,424,513×2 = 15,360,000 |
| **WAF (NAND pages programmed ÷ host pages written)** | **1.000000** |
| `Total_GC_Executions` | 0 |
| `Total_WL_Executions` | 0 |

Every NAND page programmed is directly attributable to a host write —
confirmed two independent ways (the exact page-for-page match above, and
`Total_GC_Executions=0` meaning no GC-driven page movement could have
occurred at all). No amplification was produced by this baseline.

**Read/write latency (mean, min, max only — see caveat below):**

| | Read | Write | Overall (Host.IO_Flow) |
|---|---|---|---|
| Request count | 2,009 | 15,000 | 17,009 |
| IOPS | 3.39 | 25.29 | 28.67 |
| Bandwidth (MB/s) | 27.1 | 202.3 | 229.4 |
| Avg transaction turnaround (device-side, µs) | 35,345 | 237,414 | — |
| ...of which avg wait (µs) | 35,244 | 236,638 | — |
| Avg / Min / Max end-to-end response (µs) | — | — | 311,959,480 / 41,901 / 592,957,846 |

**Caveat — no percentiles are available.** MQSim's default XML output
exposes only average/min/max per flow; there is no per-request histogram
or p50/p90/p99/p999 field. `Enable_ResponseTime_Logging` in
`ssdconfig.xml` is `false` (stock default) and was **not** enabled here —
flagging it as the config change needed for real tail-latency percentiles,
rather than making it myself.

**The huge average/max end-to-end response times (avg 312 s, max 593 s)
are, given `Total_GC_Executions=0`, queueing backlog from the trace's
inherited real burstiness — not GC stalls.** With every request marked
the same `Priority_Class=HIGH` and the real access pattern's ~99.97%
simultaneous-arrival bursts (Step 1) replayed faithfully, large clusters of
8 MiB (1024-page) requests pile up in the device's queue at the same
simulated instant; per-op media latency (750 ns page program, 75 ns page
read) is negligible next to that queueing wait (write wait 236.6 ms avg,
read wait 35.2 ms avg — both dwarf execution+transfer time). This is a
scheduling/queueing effect of replaying a bursty arrival trace against a
single-priority flow, not evidence about the device or GC.

### Go/no-go read for this baseline

- **Measured WAF sits at the floor (1.00), well below the 1.7–1.9 target**
  that Phase 3 (TokenSim side) computed as sufficient to double mean e2e
  latency. Confidence in this specific number as representative of a real
  deployment is **low** — Finding A shows the intended utilization sweep
  didn't vary anything, and even at 10× logical overwrite of an (open,
  arbitrarily-sized) 11.72 GiB range, the device's physical spare-block
  pool was never pressured enough to trigger even one GC execution in a
  15,000-write-request run.
- This is not evidence that the PCM-absorbs-writes argument is wrong — it's
  evidence that this particular baseline (stock GC/FTL thresholds, narrow
  working set relative to `CMT_Capacity`, no real background fill) hasn't
  yet been built to stress GC at all. Two config knobs stand out as
  needing domain-specific tuning before a meaningful WAF read is possible
  — flagged, not changed: `Overprovisioning_Ratio` (0.07, stock) /
  `GC_Exec_Threshold` (0.05, stock) relative to whatever far-tier capacity
  is eventually settled, and `Enabled_Preconditioning` / a real background-fill
  mechanism to replace the no-op `Initial_Occupancy_Percentage` sweep used here.
- Read-latency tails could not be characterized at all (no percentile
  output in stock config) — `Enable_ResponseTime_Logging` needs enabling
  for that, flagged rather than changed.
- The dominant effect actually observed — enormous average/max response
  times from arrival-burst queueing, not GC — is itself a real, useful
  result: it suggests MQSim's workload generator should preserve the
  bursty arrival structure found in Step 1 (already done here) *and* that
  burst-driven queueing, not GC-driven WAF, may be the first-order tail
  effect worth characterizing at this device scale, separate from the WAF
  question the sweep was designed to answer.

---

## STEP 4 — capacity-based occupancy fix + response-time logging (new complication found)

Addressing two issues raised on the Step 3 report, in the same order.

### Issue 1 fix — occupancy via device capacity, not `Initial_Occupancy_Percentage`

Confirmed diagnosis: for trace-based flows `Initial_Occupancy_Percentage`
only preconditions the CMT from the trace's own address histogram, so it
cannot create real device-fill pressure. Replaced it with shrinking the
device's own logical capacity so the fixed 11.72 GiB working set
(`BLOCK_SLOTS=1500`, unchanged) represents ~40/70/95% of the device.
Only `Block_No_Per_Plane` was changed (capacity knob) plus
`Enable_ResponseTime_Logging` (Issue 2); every GC/FTL parameter
(`Overprovisioning_Ratio=0.07`, `GC_Exec_Threshold=0.05`,
`GC_Hard_Threshold=0.005`, `CMT_Capacity`, block selection policy, etc.)
is untouched — confirmed by `diff` against the stock file (2 lines changed,
both listed below).

| Target occupancy | `Block_No_Per_Plane` | Resulting usable capacity | Working-set / capacity |
|---|---|---|---|
| 40% | 125 (was 2048) | 29.21 GiB | 40.12% |
| 70% | 72 | 16.82 GiB | 69.66% |
| 95% | 53 | 12.38 GiB | 94.63% |

`Initial_Occupancy_Percentage` set to `0` in all three `workload.xml`
(no longer meaningful for this purpose, so left inert rather than sweeping
a no-op knob). Same `kv_write_heavy.trace` (17,009 lines) replayed against
all three.

```
./MQSim -i step4/ssdconfig_occ40.xml -w step4/workload_occ40.xml > step4/run_occ40.log 2>&1
./MQSim -i step4/ssdconfig_occ70.xml -w step4/workload_occ70.xml > step4/run_occ70.log 2>&1
./MQSim -i step4/ssdconfig_occ95.xml -w step4/workload_occ95.xml > step4/run_occ95.log 2>&1
```

**MQSim does not report an occupancy/valid-page/free-block percentage
anywhere in its XML output** (checked `Address_Mapping_Unit`,
`Flash_Block_Manager`, `GC_and_WL_Unit` — no such field exists in this
version). The "working set ÷ capacity" ratios above are our own
engineered target, not something MQSim confirms back; `Total_GC_Executions`
below is the closest indirect evidence of real pressure, and it does not
behave as a monotonic confirmation (see below).

### New complication — most requests were never serviced, not delayed

| | occ40 | occ70 | occ95 |
|---|---|---|---|
| Requests generated (from trace) | 17,009 | 17,009 | 17,009 |
| **Requests serviced** | **3,701 (21.8%)** | **1,920 (11.3%)** | **1,312 (7.7%)** |
| Progress bar at which MQSim exited cleanly | 20% | 10% | 5% |
| `Total_GC_Executions` | 255 | 100 | 0 |
| `Average_Page_Movement_For_GC` | 0.878 | 0.000 | n/a (no GC ran) |
| WAF (of the serviced fraction only) | 1.0000 | 0.9999 | 0.9999 |
| Response time (µs): min / p50 / p90 / p99 / max | 41,901 / 74,992,634 / 133,167,548 / 144,929,501 / 146,374,894 | 41,901 / 40,066,204 / 70,274,777 / 77,037,398 / 77,784,968 | 41,901 / 27,568,050 / 48,716,678 / 53,359,131 / 53,870,626 |

MQSim exits cleanly ("MQSim finished", a normal wall-clock timing line, no
error/warning/exception in any of the three logs) well before consuming
the whole trace — at 20%/10%/5% of the file, using less than the full
259.6 s the trace spans (last logged completion at 146.4 s / 77.8 s /
53.9 s of simulated time respectively). The per-request response-time log
(`Enable_ResponseTime_Logging`, new this step) shows response time growing
**almost perfectly linearly** across every successive completion — not a
mostly-fast population with occasional tail spikes, which is what a
GC-stall signature would look like. That shape is the signature of a
persistently overloaded FIFO queue where arrivals outrun service, not of
intermittent GC pauses. GC execution counts are also small and don't scale
up with occupancy the way sustained back-pressure would predict — occ95
(smallest device, most pressure by design) shows **zero** GC executions,
the opposite of what the target sweep intended.

**Best-supported read (not confirmed further, flagging rather than
guessing past this point):** shrinking `Block_No_Per_Plane` shrinks the
*device* capacity as intended, but it also shrinks the per-plane physical
*resource pool* (from 2048 blocks/plane down to 53–125) that the same
128-way channel/chip/die/plane parallelism and the same burst-heavy
arrival pattern (Step 1: 99.97% simultaneous-instant writes) now contend
over. At this scale the workload's burst concurrency appears to overwhelm
the device — plausibly a genuine write-admission stall (the device
reports itself full and further writes are dropped rather than queued)
rather than every request eventually completing slowly. This has **not**
been confirmed at the source level (would mean reading
`Address_Mapping_Unit_Page_Level` / `GC_and_WL_Unit_Page_Level`'s handling
of an exhausted free-block pool) and no GC/FTL parameter was changed to
test it — reporting the symptom, not a fix.

### Issue 2 — queueing vs. device service time

Two different things, both now available:

1. **Already separated, no config change needed:** `SSDDevice.IO_Stream`
   has always reported `Average_..._Waiting_Time` separately from
   `Average_..._Execution_Time` + `..._Transfer_Time`, per read/write, in
   the Step 3 output too (we just hadn't surfaced it prominently). At
   occ40: write turnaround 824,965 µs, of which 824,284 µs (99.9%) is
   queueing wait and only 658+22 µs is actual device execution+transfer;
   read turnaround 7,902 µs, of which 7,801 µs (98.7%) is wait. This
   confirms queueing dominates end-to-end time here, consistent across
   all three occupancy points.
2. **`Enable_ResponseTime_Logging=true`** (new) gives a per-completion time
   series (`SimulationTime`, `ResponseTime`, `EndToEndDelay` columns) —
   this is what let us compute the percentiles above and see the
   linear-backlog shape directly, rather than only a final average.

**Neither output separates "GC-stalled" from "queue-backed-up" — both
present as elevated `Waiting_Time`/`ResponseTime`.** Given the shape found
here (linear, near-total-population growth, not sparse tail spikes) this
run's delay reads as queueing overload, not GC stalling, so the
GC-stall-tail question the thesis needs remains unanswered by this attempt.
To actually isolate GC-stalled requests, the proposed method (not yet
built, no source changes made): instrument `Total_GC_Executions` /
`Total_page_movements_for_gc` at a per-period granularity the same way
`Enable_ResponseTime_Logging` already timestamps per completion (or add a
lightweight GC start/end log analogous to the existing response-time
logger), then join the two time series to check whether response-time
spikes co-occur with logged GC activity — this requires a small, flagged
MQSim source change, not a config toggle, so it hasn't been done without
checking first.

**Not proceeding further — reporting this back per instruction.** Two
open questions before continuing: (1) how to fix the request-drop/stall
found at all three occupancy points (a larger capacity margin than
40/70/95% nominal, a paced/throttled replay of the same trace, or
confirming the write-rejection hypothesis first), and (2) whether adding
minimal GC-event logging to MQSim's source (the only way found so far to
actually separate GC stalls from arrival-burst queueing) is in scope, or
whether a different isolation method is preferred. No GC policy or FTL
parameter has been changed.

## STEP 5 — MQSim bug: `decision_dist_type` stale-default crash under `Enabled_Preconditioning`

### Bug report (for upstream MQSim, CMU-SAFARI/MQSim commit `51f0f2d`)

**Symptom:** any run with `Enabled_Preconditioning=true` that includes a
trace-based I/O flow terminates immediately with:
```
ERROR:Unhandled address distribution type in FTL's preconditioning function.
```

**Location:** `src/ssd/FTL.cpp`, `FTL::Perform_precondition`, the
`switch (decision_dist_type)` starting at line 435 (crash at the
`default:` case, line ~617-618).

**Root cause:** `decision_dist_type` is captured once, per flow, at line 85:
```cpp
Utils::Address_Distribution_Type decision_dist_type = stat->Address_distribution_type;
```
This runs *before* the trace-based classification branch (the `else` of
`if (stat->Type == Utils::Workload_Type::SYNTHETIC)`, further down the same
loop body) has a chance to set `stat->Address_distribution_type` from the
trace's actual content (`STREAMING` / `RANDOM_UNIFORM` / `RANDOM_HOTCOLD`,
decided starting around line 354). For a trace-based flow,
`stat->Address_distribution_type` is still at its pre-classification
default — `Address_Distribution_Type::MIXED_STREAMING_RANDOM` (enum value
0) — at the moment `decision_dist_type` is copied. `decision_dist_type` is
never refreshed afterward for trace-based flows (the only place that
reassigns it, line 121, is itself gated behind
`if (stat->Type == Utils::Workload_Type::SYNTHETIC)`), so it reaches the
line-435 switch still holding `MIXED_STREAMING_RANDOM`, which that switch
has no case for. Confirmed by instrumenting the crash site (temporarily,
then removed): for our trace flow, `stat->Type=TRACE_BASED (1)`,
`stat->Address_distribution_type=RANDOM_UNIFORM (2)` (i.e. classification
*did* run and produced a real, sane result), but `decision_dist_type=0`
(`MIXED_STREAMING_RANDOM`) at the crash — the stale pre-classification
snapshot, not the trace's real classification.

**Correction to our own earlier (wrong) diagnosis:** we first assumed
`RANDOM_HOTCOLD` was the unhandled case, mirroring the `RANDOM_HOTCOLD`→
`RANDOM_UNIFORM` fallback that already exists elsewhere in this same file
(line ~119-121, scoped to synthetic flows). That assumption was incorrect
— `RANDOM_HOTCOLD` already has full, dedicated handling in the line-435
switch (line 436 onward). `MIXED_STREAMING_RANDOM` is the only value
actually missing from that switch, and it is unreachable through normal
configuration (no XML string maps to it — `IO_Flow_Parameter_Set.cpp:344-354`
only parses `STREAMING`/`RANDOM_HOTCOLD`/`RANDOM_UNIFORM` and errors on
anything else); it only arises through this stale-variable path. This
means the crash reproduces for **any** trace-based flow combined with
`Enabled_Preconditioning=true`, independent of what the trace's real
content is or how it gets classified — we only discovered this because our
own workload happened to trigger it, not because our specific trace is
unusual.

**Applied fix** (`src/ssd/FTL.cpp`, purely additive, 1 line):
```diff
+			case Utils::Address_Distribution_Type::MIXED_STREAMING_RANDOM://Not modeled separately; MQSim's own trace-classification path (Step 1-3 above) can leave a trace-based flow's decision_dist_type at this pre-classification default before Address_distribution_type is set, so fall back to the RANDOM_UNIFORM estimation rather than crash (see docs/findings.md bug note).
 			case Utils::Address_Distribution_Type::RANDOM_UNIFORM:
 			{
 				switch (GC_and_WL_Unit->Get_gc_policy()) {
```
Adds a fall-through case label into the existing `RANDOM_UNIFORM` block
(reusing its math, per instruction); no existing line changed or removed.
Note this masks the crash for the stale-default case specifically — it
does not fix the underlying stale-snapshot timing bug in general (a
correct fix would recompute `decision_dist_type` from
`stat->Address_distribution_type` after trace classification runs, or move
the snapshot after that classification). Flagging this distinction for
anyone filing the upstream report: our patch is a safe, narrow workaround
for the crash, not a fix of the root timing issue.

**Verification performed before applying to real runs:** built both an
unpatched and patched binary from the same otherwise-identical source
(`git stash` / `git stash pop` around the one-line change), ran both
against an identical single-flow, `RANDOM_UNIFORM`-classified synthetic
workload (`Enabled_Preconditioning=true`, 60% initial occupancy, 3000
requests) that does **not** hit the missing branch:
```
./MQSim -i ssdconfig_verify.xml -w workload_verify.xml
```
Both runs completed (3000/3000 requests serviced, identical console
summary) and produced **byte-identical result XML** (`md5sum` match,
`diff` exit code 0) — confirming the patch is purely additive and the
already-working path is untouched.

### Step B status — blocked by two further problems, not yet fixed

Rerunning the intended 3-point pre-fill sweep (1 trace flow + 19
background synthetic flows per scenario, from the previous message) with
the patched binary got past the crash above, but failed differently at
all three points — **stopping here, not patching further, per
instruction**:

- **occ40 and occ70:** a *different* crash during preconditioning:
  `ERROR:It is not possible to assign PPA to all LPAs in
  Allocate_address_for_preconditioning! It is not safe to continue
  preconditioning.` (`Address_Mapping_Unit_Page_Level.cpp:738`). Not yet
  root-caused — plausibly the 19 background flows' combined preconditioned
  volume oversubscribes specific physical planes once CWDP striping is
  accounted for, even though each flow's own nominal occupancy target
  (39.5%/71.1%) looked achievable in isolation.
- **occ95:** preconditioning *completed* this time (3m27s), but the run
  then crashed once actual simulation started:
  `terminate called after throwing an instance of 'std::invalid_argument'
  what(): Unknown register is written!` — traced to
  `Host_Interface_NVMe.cpp:274`. MQSim's NVMe host interface hard-codes
  exactly 8 submission/completion queue register pairs
  (`SUBMISSION_QUEUE_REGISTER_1`...`_8`, `Host_Interface_NVMe.cpp:225-274`);
  anything beyond queue 8 falls through to `default:` and throws. **This
  confirms the README's "up to 8 different workloads" line (which we'd
  read as soft guidance) is a real, hard-coded limit** — our 20-flow design
  (1 trace + 19 background) structurally cannot run a full simulation
  in this MQSim version, only get through preconditioning.

Neither of these has been patched or worked around. Reverting to ≤8 total
flows (recomputed earlier: 7 background flows can reach ~42.9%/77.2%/~100%
per-flow fill for the 40/70/95% targets, capping actual achievable overall
occupancy at ~89.9% for the 95% case) is the obvious next step, but is not
applied yet, pending direction.

## STEP 6 — 8-flow pre-fill design: one of three occupancy points runs

### Why the 95% target is gone (simulator constraint, not a convenience)

The original 40/70/95% sweep cannot be built in this MQSim version. Two
independent hard limits cap it, both in stock MQSim, neither introduced by
our configuration:

1. **8 I/O flows maximum.** `Host_Interface_NVMe.cpp:225-274` enumerates
   exactly eight submission/completion queue register pairs
   (`SUBMISSION_QUEUE_REGISTER_1`…`_8`); a ninth flow's register write falls
   to `default:` and throws `std::invalid_argument("Unknown register is
   written!")` (`Host_Interface_NVMe.cpp:274`). The README's "up to 8
   different workloads" is therefore a hard code limit, not guidance. With
   1 trace flow + 7 background flows, and MQSim splitting logical capacity
   evenly across flows, the background flows can hold at most 7/8 of the
   device, so even at 100% background fill the whole-device occupancy tops
   out at **89.9611%**.
2. **Per-flow preconditioning clamp at 90.9091%.** `FTL.cpp:171-184`
   requires `(max_lpa - min_lpa) >= 1.1 * no_of_logical_pages_in_steadystate`
   for a `RANDOM_UNIFORM` flow; above `1/1.1 = 90.9091%` per-flow fill MQSim
   prints "MQSim made some adjustments!" and silently clamps the actual fill
   to 90% of the flow's range. So the 89.9611% figure is itself unreachable:
   requesting 100% background fill would deliver 90%.

**Reachable ceiling = 81.2111%** (background flows at 90.00% fill, safely
under the clamp — confirmed at runtime: the ceiling run printed no
"adjustments" warning). Reporting both numbers rather than the rounded
~89.9% quoted earlier: 89.9611% is the flow-count arithmetic ceiling,
81.2111% is what MQSim will actually precondition to.

### Per-flow sanity check (done before running, to avoid the Step A failure mode)

Capacity per MQSim's own formula
(`Logical_Address_Partitioning_Unit.cpp:136-147`): all 8 flows declare all
128 planes, so each plane is shared 8 ways and each flow gets
`128 * int(2048*256*16*0.93/8)` = **124,822,400 sectors = 59.5200 GiB =
7,801,400 pages**; device logical total 476.1597 GiB.

| point | bg fill/flow | bg pages/flow | vs 90.909% clamp | overall occ | physical blocks free |
|---|---|---|---|---|---|
| ~40% | 42.9016% | 3,346,925 | 0.472x | 40.0000% | 62.80% |
| ~70% | 77.1873% | 6,021,691 | 0.849x | 70.0000% | 34.90% |
| ceiling | 90.0000% | 7,021,260 | 0.990x | 81.2111% | 24.47% |

Step A failure mode (writes stalling on an exhausted per-plane free-block
pool) is not a risk here: `Block_No_Per_Plane` is back to stock 2048, so the
hardcoded `max_ongoing_gc_reqs_per_plane=10` GC threshold floor is 0.488% of
the pool (it was 8.0-18.9% at the 53-125 blocks/plane of Step A, 16-39x
worse). The trace flow uses only 1,536,000 of its own 7,801,400 pages
(19.69%, 5.08x headroom). Worst point still leaves 24.5% of physical blocks
free, ~50x above the GC trigger.

### Results — 1 of 3 points completed

| | occ40 (40.0000%) | occ70 (70.0000%) | ceiling (81.2111%) |
|---|---|---|---|
| Completed | **no** (exit 1) | **yes** (exit 0) | **no** (exit 1) |
| Failure | preconditioning: `It is not possible to assign PPA to all LPAs in Allocate_address_for_preconditioning` (`Address_Mapping_Unit_Page_Level.cpp:738`) | — | preconditioning OK (4m3s, no clamp warning), then crashed at 55% of the trace: `Illegal operation: Unlocking an LPA that has not been locked!` (`Address_Mapping_Unit_Page_Level.cpp:1801`) |
| Trace requests serviced | n/a | **17,009 / 17,009 = 100.00%** | n/a (died mid-run) |
| WAF | n/a | **1.05141** | n/a |
| `Total_GC_Executions` | n/a | **24,737** | n/a |
| `Average_Page_Movement_For_GC` | n/a | 31.9206 | n/a |
| `Total_WL_Executions` | n/a | 125 | n/a |

WAF arithmetic for occ70 (page programs, multiplane command = 2 pages —
independently confirmed here): `Issued_Flash_Program_CMD` 2,311,793 +
2 x `Issued_Flash_Multiplane_Program_CMD` 6,918,917 = **16,149,627** NAND
page programs against **15,360,000** host write pages (15,000 requests x
1024 pages; the 7 background flows issue 1 request each during the run).
The 789,627-page difference matches GC page movements
(24,737 x 31.9206 = 789,620) to within rounding, so the identity
`NAND programs = host pages + GC movements` holds exactly.

**This is the first configuration in which GC actually engages** (24,737
executions vs. 0/100/255 in the capacity-shrink attempts) **and the first
in which 100% of the trace is serviced.** Pre-fill is the right mechanism;
capacity-shrink was not.

### Response-time split, occ70 (trace flow, stream 0)

Per-transaction averages from `SSDDevice.IO_Stream`:

| | turnaround | waiting (queueing) | execution | transfer | wait share |
|---|---|---|---|---|---|
| Read | 35,525 us | 35,424 us | 75 us | 25 us | 99.72% |
| Write | 270,921 us | 270,145 us | 750 us | 25 us | 99.71% |

Per-request response-time percentiles from the flow log
(`Enable_ResponseTime_Logging`, n=16,830):

| min | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| 41,902 us | 313,427,781 us | 551,392,474 us | 592,848,802 us | 599,005,567 us | 599,649,433 us |

Even with GC now genuinely active, latency remains ~99.7% queueing wait and
the percentile curve still has the linear-backlog shape (min 42 ms rising
steadily to 600 s), not a mostly-fast population with GC spikes. GC adds
about 5% to write volume here; it is not what is setting the tail at this
arrival rate.

### Status

**WAF = 1.051 at 70% occupancy — far below the 1.7-1.9 target** from the
TokenSim side, and the single working point is not enough to say whether
WAF rises with occupancy (occ40 and the ceiling both died on separate
pre-existing MQSim defects, neither related to the `MIXED_STREAMING_RANDOM`
fix from Step 5, and neither patched). Notably the run aborts at low
occupancy (42.9% fill) and at high occupancy (90% fill) but succeeds in the
middle, so the two failures are not a single "too much data" effect.
The `Unlocking an LPA that has not been locked` failure is worth flagging
upstream alongside the Step 5 bug: HEAD (`51f0f2d`) is itself the merge of
PR #77 "1-lpa-locking-sanity-and-memory-leaks", i.e. LPA-lock accounting is
an area with known recent churn.

Stopping here per instruction; Step C (GC-event logging) not started.

---

# STEP 7 — Is WAF = 1.05 an artifact of unpaced replay?

Motivation (user): the occ70 run showed 99.71 % of write turnaround was queueing
wait, median request response 313 s, and linear growth across the request
population — i.e. the device looked like it was draining one huge backlog
rather than operating under sustained arrival pressure. GC under backlog-drain
is the easiest possible condition, so a low WAF might be an artifact of the
replay rather than a property of the access pattern.

## Step 7.1 — How is the trace actually being replayed?

**The trace *file* preserves the source timing.** Comparing the inter-burst gap
distribution of `step1/kv_write_heavy.trace` against the Step 1 source event log:

| quantity | source event log | MQSim trace file |
|---|---|---|
| nonzero inter-burst gaps | 240 | 234 |
| mean gap | 1.0900 s | 1.0215 s |
| p50 | 0.8039 s | 0.7871 s |
| p90 | 2.4608 s | 2.3911 s |
| p99 | 5.3946 s | 5.2749 s |

The every-*k*-th downsampling kept each retained event's real absolute write
time, so timestamps were neither collapsed nor renormalised. Step 7.1's literal
question — "were the ~241 gaps preserved?" — answers **yes**.

### But the replay was still effectively unpaced — `Time_Unit` is dead code

`build_mqsim_trace.py` emitted timestamps in microseconds and the workload XML
declared `<Time_Unit>MICROSECOND</Time_Unit>`. **MQSim ignores that tag.**

`time_unit` appears in exactly four places in the whole source tree:

- `src/host/IO_Flow_Trace_Based.h:17` — constructor parameter
- `src/host/IO_Flow_Trace_Based.h:31` — member declaration
- `src/host/IO_Flow_Trace_Based.cpp:10` — constructor parameter
- `src/host/IO_Flow_Trace_Based.cpp:13` — member initialiser

It is **never read**. Arrival events are registered with the raw, unscaled
integer from the trace's time column:

```cpp
// IO_Flow_Trace_Based.cpp:125  (first line of the trace)
Simulator->Register_sim_event(std::strtoll(current_trace_line[ASCIITraceTimeColumn].c_str(), &pEnd, 10), this);
// IO_Flow_Trace_Based.cpp:162  (every subsequent line)
Simulator->Register_sim_event(time_offset + std::strtoll(current_trace_line[ASCIITraceTimeColumn].c_str(), &pEnd, 10), this);
```

`Register_sim_event` takes `sim_time_type` in the engine's base unit, which is
nanoseconds (`SIM_TIME_TO_SECONDS_COEFF 1000000000`, `Sim_Defs.h:30`). So a
trace authored in microseconds is replayed **1000× too fast**: our intended
259.64 s of arrivals actually landed inside 259.64 **ms**. Every MQSim run in
this project prior to Step 7 had this defect. This is a third genuine MQSim
bug, independent of our configuration (see also Step 5's `decision_dist_type`
snapshot bug); `Time_Unit` is parsed from XML (`IO_Flow_Parameter_Set.cpp:454-462`),
serialised back out (`:423-431`), threaded through `Host_System.cpp:70` — and
then silently dropped.

## Step 7.2 — Regenerate with real pacing and rerun occ70

Two corrected traces were built (timestamps written directly in nanoseconds, so
no unit conversion is needed):

- `step7/kv_ns_true.trace` — source pacing ×1000 → **259.644 s** span (true TokenSim timing)
- `step7/kv_ns_5x.trace` — a further 5× stretch → **1298.221 s** span (deliberately de-saturated control)

Same device config as Step 6 (`step5/ssdconfig_stepB.xml`, stock + preconditioning
+ response-time logging), same 8-flow occ70 pre-fill design, same 17,009-request
trace flow. Both runs completed with exit 0 and **100 % of requests serviced**.

### Results

| | occ70 (pre-fix, 0.2596 s span) | **occ70 paced true (259.64 s)** | occ70 paced ×5 (1298.22 s) |
|---|---|---|---|
| requests serviced | 17009 / 17009 (100 %) | 17009 / 17009 (100 %) | 17009 / 17009 (100 %) |
| **WAF** | **1.05141** | **1.05141** | **1.05300** |
| NAND program pages | 16,149,627 | 16,149,627 | 16,174,133 |
| host write pages | 15,360,000 | 15,360,000 | 15,360,000 |
| **GC executions** | **24,737** | **24,737** | **24,826** |
| avg pages moved / GC | 31.9206 | 31.9206 | 32.7933 |
| erases (single + multiplane) | 17,916 + 3,473 | 17,916 + 3,473 | 17,143 + 3,904 |
| sim end time | 599.89 s | 599.89 s | 1298.32 s |
| achieved write BW | 209.75 MB/s | 209.75 MB/s | 96.92 MB/s |
| device duty cycle | 100 % | 100 % | 46.2 % |
| mean request response | 313.35 s | 203.07 s | **4.78 s** |
| p50 / p90 / p99 response | 313.4 / 551.4 / 592.8 s | 213.4 / 337.8 / 363.3 s | **2.98 / 12.00 / 23.42 s** |
| max response | 599.65 s | 365.72 s | 30.16 s |
| write txn turnaround | 270.921 µs | 167.742 µs | 1364.010 µs |
| ‑ of which queueing wait | 270.145 µs (**99.71 %**) | 166.966 µs (**99.54 %**) | 1363.234 µs (**99.94 %**) |
| ‑ device execution | 0.750 µs | 0.750 µs | 0.750 µs |
| ‑ data transfer | 0.025 µs | 0.025 µs | 0.025 µs |
| read txn turnaround | 35.525 µs (99.72 % wait) | 35.525 µs (99.72 % wait) | 287.203 µs (99.97 % wait) |
| completions after last arrival | ~100 % | 10,086 / 16,830 (**59.9 %**) | 22 / 16,968 (**0.1 %**) |
| post-arrival drain | 599.6 s | 340.24 s (56.7 % of run) | **0.10 s (0.0 % of run)** |
| inter-completion gaps > 50 ms | 0 | 0 (max gap 42 ms) | 185, totalling 669.18 s (**51.5 % of run**) |

### The key check — does the device now show idle periods between bursts?

**At true source pacing: no.** `ns_true` still ends 340.24 s after the last
arrival; 59.9 % of completions land after the trace is over; there is not a
single inter-completion gap above 50 ms. The device is 100 % busy end to end.

This is **not** a trace-construction artifact — it is the workload. The
downsampled trace offers 125.83 GB of writes across 259.64 s = **484.6 MB/s**,
against a device that sustains **209.75 MB/s**: a **2.31× oversubscription**.
(The *undownsampled* TokenSim SSD-tier stream is ~110× this device.) At the real
arrival rate this SSD is saturated by construction, so a backlog is the correct
behaviour, not an error in how we built the trace.

**With a further 5× stretch: yes, decisively.** `ns_5x` finishes 0.10 s after
the last arrival, has 185 idle gaps totalling 51.5 % of the run (largest 53.1 s),
and runs at a 46.2 % duty cycle. This is the genuinely non-saturated,
burst-and-idle regime the Step 7 question was designed to reach.

### Verdict: WAF = 1.05 is not an artifact of unpaced replay

Going from 100 % device busy with a 340 s backlog to 46 % busy with half the
run idle moves WAF by **+0.15 %** (1.05141 → 1.05300) and GC executions by
**+0.36 %** (24,737 → 24,826). Per-request response time collapses by 65× (313 s
→ 4.78 s) over the same range, so the runs are unambiguously in different
queueing regimes — the WAF simply does not care.

The low WAF is a property of the access pattern, not of the replay:
the KV block stream is a circular overwrite of a 1,500-slot (11.72 GiB) working
set inside a 478.5 GiB device, written in whole 8 MiB units. Blocks are
invalidated in bulk and in roughly the order they were written, so GC nearly
always finds near-empty victims. 24,737 GC executions × 31.92 pages = 789,559
pages moved against 15,360,000 host pages = 5.14 % overhead, which is exactly
the 1.0514 measured.

Note also that `ns_true` reproduces the pre-fix run's flash counters
**exactly** — the two result files differ only in the trace path and the four
response-time fields (26 diff lines total). The `Time_Unit` bug therefore
corrupted the *latency* numbers in every earlier run but left every
WAF/GC/endurance number intact, because the device was throughput-bound in both
cases and processed the identical LPA sequence in the identical order.

### A caveat on the "99.7 % queueing wait" figure

The `Average_Write_Transaction_Waiting_Time` share stays at 99.5-99.9 % in *all*
three runs, including the one that is idle half the time. That statistic is
therefore not a saturation indicator. It measures waiting inside the device's
per-chip transaction queues, where a single 8 MiB request fans out to 1,024
page programs that must serialise onto 128 planes — so even an isolated request
sees ~1,000 × 750 µs of self-inflicted queueing behind its own pages. The
saturation signal is at the *request* level (post-arrival drain, completions
after last arrival, duty cycle), not the transaction level. The earlier reading
of "99.71 % wait ⇒ backlog drain" conflated the two; the backlog drain was real,
but this statistic was not the evidence for it.

## Step 7.3 — not run

The user's gate was "only if the paced occ70 run differs materially." The paced
`ns_true` run is byte-identical to the pre-fix occ70 run on every WAF, GC,
erase and bandwidth counter. It does not differ materially, so occ60 and occ75
were not added.

Stopping here per instruction. The two crashes (PPA assignment at
`Address_Mapping_Unit_Page_Level.cpp:738`, LPA-lock at `:1801`) remain
unpatched, and Step C (GC-event logging) is not started.

---

# ENDURANCE/POWER RESCOPE — headline metric is NAND write volume

Autonomous run, 2026-09-13. The thesis claim is now scoped to **endurance and
power**, not serving latency: the headline metric is NAND write volume and the
device lifetime derived from it. Latency is reported alongside throughout but is
secondary.

Standing workload unchanged (Mooncake FAST'25 `conversation_trace.jsonl`, first
500 requests, 3 QPS, H200x8, `llama-7b.json` MHA, `block_size=16` →
**8,388,608 B = 8 MiB** per block, `memory_capacity_blocks=4096`,
`ssd_capacity_blocks=262144`, `ssd_write_latency_us=200`,
`ssd_read_latency_us=100`, `ssd_write_bw_gbps=3.0`, `ssd_read_bw_gbps=7.0`).
`charge_eviction_writes=ON` for every run in Phases 1-3 unless a row says
otherwise — writes are the whole thesis now and the stock (off) accounting
undercharges them. No run below had unfinished requests, preemptions or
recomputations (`notdone=0`, `preempt=0`, `recompute=0` everywhere).

## PHASE 1 — `demote_policy`

### STEP 1.1 — instrumentation

`StoreObject` (`TokenSim/mooncake/store.py`) gained two fields:

- `read_count: int = 0` — incremented **per object** by iterating `hit_keys` on
  the hit path of `MooncakeStore.lookup`, not a global counter. A missed lookup
  credits nothing.
- `write_time: float = 0.0` — the `now` of the put that admitted the object.

`_evict_memory_lru` now classifies every memory-tier victim through
`MooncakeStats.record_memory_eviction`: `read_count == 0` is a **wasted write**,
`read_count > 0` is **useful**, and residency (`now - write_time`) goes into a
running mean plus a fixed-edge histogram (edges 0.1/1/10/30/60/120/180/240/300 s,
final bucket open-ended). New exported fields alongside the existing `mooncake_*`
set: `mooncake_wasted_write_blocks`, `mooncake_useful_write_blocks`,
`mooncake_mean_residency_s`, `mooncake_residency_histogram`,
`mooncake_blocks_dropped`, `mooncake_blocks_demoted`.

The histogram is a per-bucket counter list, so `MooncakeStats.aggregate` sums it
element-wise (`_COUNTER_LIST_FIELDS`) rather than concatenating-and-truncating it
the way the pool-key sample list is handled.

### STEP 1.2 — the policy

`demote_policy` added to `kv_connector_extra_config`, default `"always"`
(stock behaviour), alternative `"if_read"`. Under `if_read` a memory-tier victim
with `read_count == 0` is dropped from the write stream instead of being demoted;
`read_count > 0` demotes exactly as before.

**Structural finding that governs how this policy behaves — the offload write is
already a duplicate.** `_put_one` write-*through*s every memory admission to the
offload tier (`store.py:221-232`), and `_evict_memory_lru` then writes the *same
key with the same bytes* again when that object is later evicted from memory. So
in stock code every block is written to the offload tier **twice**: once at
admission, once at demotion, byte-identical. The store's own
`_drop_offload_victims` docstring already describes the admission copy as a
"backup copy". Consequences, all confirmed in the runs below:

- `if_read`'s drop path declines only the *second*, redundant write. The block's
  admission copy stays on the tier, so **no data is lost** and no cache hit is
  given up (hits/misses are identical across policies below).
- The ceiling on writes this policy can remove is therefore ~50% — exactly the
  demote half of the doubled write stream — not the ~87% a
  "13.13%-read-before-eviction" argument would suggest if demotion were the only
  write.
- The one real behavioural difference beyond write count: a demote calls
  `OffloadTier.write`, which re-inserts the key at the MRU end of the offload
  LRU; a drop does not touch it. Offload eviction order therefore differs
  slightly between policies. This is a genuine policy effect, not an artifact.

Write accounting closes exactly under both policies:
`admission_count + blocks_demoted == ssd_write_blocks` (364,340 + 360,244 =
724,584 under `always`; 361,584 + 32 = 361,616 under `if_read`).

**Verification — `demote_policy="always"` is byte-identical to pre-change code.**
A reverse-patched copy of the pre-change working tree was built in the scratch
directory and verified against git: `diff HEAD baseline` contains only the
*previous* session's work (`charge_eviction_writes`, the memory/SSD eviction
counter split, offload-victim dropping) and no trace of this session's edits.
Running the standing workload on both trees at `charge_eviction_writes=ON`:

- **85 result-JSON fields byte-identical** after excluding `simulator_wall_time`
  and `mooncake_pool_keys` per instruction, plus the 6 new counters (absent from
  the baseline by construction).
- No field exists in the baseline that is missing from the patched result.
- The run also reproduces the findings.md Phase 2 WAF=1/ON reference row exactly:
  `mem_evict=360,244`, `ssd_evict=102,196`, `p99 e2e=319.9598`, `tok/s=15,989.4`.

Tests added (`tests/test_mooncake_simulator.MooncakeDemotePolicyTest`, 14 cases;
full suite 47 tests, all passing): per-object (not global) read counting,
`write_time` stamping, missed lookups crediting nothing,
`wasted + useful == memory evictions` and `dropped + demoted == memory evictions`
under **both** policies, byte conservation with the drop path active
(`ssd_write_bytes == ssd_write_blocks × block_bytes`, offload occupancy equals
the sum of its per-key block counts and never exceeds capacity), a dropped block
still being readable from its write-through copy, a read victim still being
demoted under `if_read`, config rejection of an unknown policy, the `"always"`
default, and element-wise histogram aggregation.

### STEP 1.3 — measurement

`charge_eviction_writes=ON`, separate `--results_path` per run.

| | `demote_policy=always` | `demote_policy=if_read` |
|---|---|---|
| **offload write blocks** | **724,584** | **361,616** |
| **offload write bytes** | **6,078,251,139,072** (5,660.81 GiB / 6.078 TB) | **3,033,454,870,528** (2,825.12 GiB / 3.033 TB) |
| offload read blocks / bytes | 48,485 / 406,721,658,880 | 52,172 / 437,650,456,576 |
| memory evictions | 360,244 | 357,488 |
| offload evictions | 102,196 | 99,440 |
| **wasted write blocks** | **359,917** | **357,456** |
| useful write blocks | 327 | 32 |
| **wasted-write fraction** | **99.9092%** | **99.9910%** |
| blocks dropped | 0 | 357,456 |
| blocks demoted | 360,244 | 32 |
| **mean residency (s)** | **4.7383** | **2.8837** |
| store hits / misses | 492 / 8 | 492 / 8 |
| memory-tier / disk-tier hits | 7 / 485 | 10 / 482 |
| admissions | 364,340 | 361,584 |
| save wait (s) | 2,088.84 | 1,070.57 |
| load wait (s) | 113.37 | 121.68 |
| **throughput (tok/s)** | **15,989.36** | **25,186.47** |
| output QPS | 1.0943 | 1.7237 |
| simulated duration (s) | 456.92 | 290.07 |
| e2e p50 / p99 / max (s) | 164.09 / 319.96 / 364.07 | 68.28 / 160.03 / 190.13 |
| TTFT p50 / p99 (s) | 66.00 / 250.17 | 14.95 / 98.00 |

Residency histogram (memory-tier evictions, counts per bucket):

| policy | 0-0.1s | 0.1-1s | 1-10s | 10-30s | 30-60s | ≥60s |
|---|---|---|---|---|---|---|
| always | 47,612 | 51,701 | 221,926 | 36,684 | 2,321 | 0 |
| if_read | 45,585 | 39,771 | 265,436 | 6,696 | 0 | 0 |

**Ratio of NAND writes eliminated.** Two figures, because the two policies do not
produce the same simulated timeline once eviction writes are charged:

- **Cross-policy, as run: −50.0933%** of write blocks and of write bytes
  (724,584 → 361,616; **2.0037× fewer**). This is the number a deployment would
  see, but it folds in a scheduling-feedback difference (see below).
- **Same-trajectory: −49.7107%** (719,072 → 361,616; **1.9885×**), comparing
  against `always` with `charge_eviction_writes=off`, which reproduces the
  identical simulated timeline (see the cross-check).

**Wasted-write fraction is 99.91%, not ~87%.** The 13.13% read-before-eviction
figure from the earlier MQSim Step 1 extraction measured reads over an object's
*whole* lifetime, most of which is spent on the offload tier: mean time-to-first-
read there was 91.31 s. Mean *memory* residency is **4.74 s**. Objects are pushed
out of the 4,096-block memory tier roughly 19× faster than they are read back, so
essentially nothing is ever read while still in memory — hence 99.91%, and hence
only 327 of 360,244 victims qualifying as "useful" under `always`. The two
numbers are consistent; they measure different windows.

**Flagged — scheduling feedback, and a non-monotonic eviction series.** With
eviction writes charged, halving the write stream shortens the run from 456.92 s
to 290.07 s and raises throughput 57.5%. The shorter, less congested run then
performs *fewer* evictions (360,244 → 357,488) and fewer offload evictions
(102,196 → 99,440) — so eviction counts move in the opposite direction to what a
"policy that writes less must be under less pressure" reading would predict only
because the whole timeline compressed. Any per-run counter compared across these
two configurations is measured over a different simulated duration; the
same-trajectory column below is the artifact-free comparison.

**Cross-check — the drop path is provably timing-neutral.** `if_read` +
`charge_eviction_writes=ON` was compared against `always` +
`charge_eviction_writes=off`. Every field matches to the last printed digit —
duration 290.0683 s, tok/s 25,186.4680, e2e p50/p99/max 68.2847/160.0286/190.1319,
mem_evict 357,488, ssd_evict 99,440, hits 492, misses 8, wasted 357,456, mean
residency 2.8837 s — with only the write counters differing (361,616 vs 719,072
blocks) and `save_wait` differing by 0.09 s, which is the 32 demote writes
`if_read` still performs. Two independent corroborations fall out of this: the
`always`/off row reproduces the earlier findings.md WAF=1/off reference exactly
(357,488 / 99,440 / 160.0286 / 25,186.5), and its 719,072 write blocks reproduce
the "719,072 accepted SSD-tier writes" from the earlier MQSim Step 1 extraction
exactly. Dropping a demote write and declining to charge a demote write are the
same thing for time, and differ only in NAND volume — which is what makes the
write saving cleanly separable from the latency effect.

### BRANCH — wasted-write fraction vs. the 50% gate

**Wasted-write fraction = 99.91% (`always`), far above the 50% gate. Proceeding
to Phase 2.** The absorption argument is stronger than the 13.13% figure
predicted, for the reason given above (memory residency is 19× shorter than
time-to-first-read). The caveat to carry forward is not the fraction but the
*ceiling*: because the store write-throughs on admission, at most ~50% of NAND
writes are removable by any eviction-time demote policy, however good its
prediction is. A policy that also declines the admission write-through would be
needed to go past that, and that is a different mechanism with a real hit-rate
cost — not attempted here.

## PHASE 2 — baseline reconstruction

### STEP 2.1 — HBF-1 and HBF-2 hardware entries

Derived from the stock `H200` entry in `TransformerRoofline/hardware_models.json`
(`TFLOPS 1979, BW_TBs 4.8, Capacity 141, TDP 700, Card_Num 1`) and the H3
organization: some HBM sites on the package are given over to package-local
flash, and the HBM that remains gives up per-stack bandwidth to pay for the
co-packaged flash.

Written to an **untracked copy** in the scratch directory and selected with the
new `--hardware_models` flag (added this session to `benchmark.py`; it did not
exist — the catalogue path was hard-coded. It defaults to the tracked file, so
stock behaviour is unchanged). `git diff TransformerRoofline/hardware_models.json`
is empty: **the tracked file was never mutated.**

**Assumptions, stated so they can be checked or replaced:**

| # | assumption | value |
|---|---|---|
| A1 | HBM3e stacks on the stock H200 package | 6 sites |
| A2 | per-stack capacity = `141 / 6` | 23.5 GiB |
| A3 | per-stack bandwidth = `4.8 / 6` | 0.8 TB/s |
| A4 | an HBF stack holds this multiple of one HBM3e stack | 16× |
| A5 | per-stack HBM bandwidth derate per co-packaged HBF stack (interposer beachfront and routing given up to the flash stack) | −10% each |
| A6 | logic die, TFLOPS, TDP and links unchanged | as H200 |

**The arithmetic:**

| entry | HBF sites | HBM sites | derate (A5) | per-stack BW | **`BW_TBs`** | **`Capacity`** | package flash (A4) | flash in 8 MiB blocks |
|---|---|---|---|---|---|---|---|---|
| H200 (stock) | 0 | 6 | 1.00 | 0.800 TB/s | **4.80** | **141.0** | — | — |
| **HBF-1** | 1 | 5 | 0.90 | `0.8 × 0.90` = 0.720 | `5 × 0.720` = **3.60** | `5 × 23.5` = **117.5** | `1 × 16 × 23.5` = 376.0 GiB | **48,128** |
| **HBF-2** | 2 | 4 | 0.80 | `0.8 × 0.80` = 0.640 | `4 × 0.640` = **2.56** | `4 × 23.5` = **94.0** | `2 × 16 × 23.5` = 752.0 GiB | **96,256** |

A5 is the one free parameter and is the number to argue about: at a 0% derate
the entries would read 4.00 / 3.20 TB/s instead of 3.60 / 2.56, i.e. A5 moves
`BW_TBs` by −10%/−20%. A1–A4 are either read directly off the stock entry or are
simple ratios of it. `Capacity` here is HBM only — the package flash is the
*offload tier*, modelled by `ssd_capacity_blocks`, not by this field.

### STEP 2.2 — four configurations

Same trace, same QPS, `charge_eviction_writes=ON` throughout. HBF = offload
tier; its capacity is the derived package flash, not the reference 2 TiB SSD.
`(b2)` is a supplementary run, not one of the four.

| | (a) SSD-Mooncake | (b) HBF, always | (c) HBF+PCM, always | (d) HBF+PCM, if_read | (b2) HBF-2, always |
|---|---|---|---|---|---|
| GPU entry | H200 | HBF-1 | HBF-1 | HBF-1 | HBF-2 |
| far tier capacity | 262,144 blk (2 TiB) | 48,128 blk (376 GiB) | 48,128 blk | 48,128 blk | 96,256 blk (752 GiB) |
| memory tier media | dram | dram | **pcm** | **pcm** | dram |
| **far-tier write blocks** | **724,584** | **839,348** | **844,572** | **426,574** | 820,970 |
| **far-tier write bytes** | 6,078,251,139,072 | 7,040,961,347,584 | 7,084,783,435,776 | 3,578,362,068,992 | 6,886,953,451,520 |
| far-tier read blocks | 48,485 | 6,565 | 3,872 | 2,106 | 12,060 |
| **memory-tier write blocks** | **364,340** | **421,722** | **424,334** | **426,382** | 411,206 |
| memory-tier read blocks | 263 | 98 | 0 | 96 | 205 |
| PCM write media time (s) | — | — | 166.180 | 166.982 | — |
| PCM read media time (s) | — | — | 0.000 | 0.008 | — |
| **blocks dropped** | 0 | 0 | 0 | **422,094** | 0 |
| **blocks demoted** | 360,244 | 417,626 | 420,238 | **192** | 411,206 |
| memory evictions | 360,244 | 417,626 | 420,238 | 422,286 | 411,206 |
| far-tier evictions | 102,196 | 373,594 | 376,206 | 378,254 | 344,502 |
| wasted / useful blocks | 359,917 / 327 | 417,336 / 290 | 420,109 / 129 | 422,094 / 192 | 410,918 / 288 |
| wasted-write fraction | 99.9092% | 99.9306% | 99.9693% | 99.9545% | 99.9299% |
| mean residency (s) | 4.7383 | 4.4578 | 4.5580 | 3.0522 | 4.3925 |
| **hits / misses** | 492 / 8 | 461 / 39 | 479 / 21 | 485 / 16 | 471 / 29 |
| memory / disk hits | 7 / 485 | 40 / 421 | 21 / 458 | 46 / 439 | 26 / 445 |
| admissions | 364,340 | 421,722 | 424,334 | 426,382 | 411,206 |
| save / load wait (s) | 2,088.84 / 113.37 | 2,419.74 / 15.42 | 2,600.89 / 9.03 | 1,429.84 / 5.03 | 2,290.48 / 30.38 |
| **mean e2e (p50, s)** | 164.09 | 173.80 | 184.99 | **96.09** | 170.77 |
| e2e p99 / max (s) | 319.96 / 364.07 | 350.74 / 381.26 | 382.68 / 409.45 | 207.97 / 245.31 | 341.07 / 371.34 |
| TTFT p50 / p99 (s) | 66.00 / 250.17 | 83.40 / 297.94 | 93.42 / 303.11 | 34.06 / 154.22 | 79.26 / 289.52 |
| **throughput (tok/s)** | 15,989.36 | 14,599.82 | 14,010.86 | **20,853.77** | 15,109.55 |
| output QPS | 1.0943 | 0.9992 | 0.9589 | 1.4272 | 1.0341 |
| duration (s) | 456.92 | 500.40 | 521.44 | 350.33 | 455.90 |
| notdone / preempt / recompute | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / **1** / **1** | 0 / 0 / 0 |

**The PCM memory tier had to be added.** Before this session the memory tier took
a single protocol latency in both directions (`TransferEngineSimulator._fixed_latency`
and `._bandwidth` depend only on protocol and topology, never on direction), so
it could not express PCM's write asymmetry at all. Added behind
`memory_media` (`"dram"` default = stock, zero memory-tier media time; `"pcm"`
charges asymmetric media time), with `pcm_read_latency_us=0.1`,
`pcm_write_latency_us=1.0`, `pcm_read_bw_gbps=100.0`, `pcm_write_bw_gbps=20.0`
(a 5× write/read bandwidth asymmetry). Re-verified after this change that the
stock path is still byte-identical to pre-change code: 84 result fields match,
with only the new counters and new `offload_profile` keys added. 7 more tests
(`MooncakePCMMemoryTierTest`); full suite now **54 tests, all passing**.

**Modelling choice, stated because it is not neutral:** a demote reads the block
out of the memory tier before writing it to flash, and that PCM read is *not*
charged. Charging it would penalise `always` (which demotes ~420k blocks) and
barely touch `if_read` (192 blocks), so leaving it out **understates** the
advantage the `if_read` results below show. It is the conservative direction.

#### (b) → (c): the tier's effect, isolated from the policy

| quantity | (b) | (c) | Δ |
|---|---|---|---|
| far-tier write blocks | 839,348 | 844,572 | **+0.62%** |
| far-tier write bytes | 7,040,961,347,584 | 7,084,783,435,776 | **+0.62%** |
| memory-tier write blocks | 421,722 | 424,334 | +0.62% |
| throughput (tok/s) | 14,599.82 | 14,010.86 | **−4.03%** |
| e2e p50 (s) | 173.80 | 184.99 | +6.44% |
| duration (s) | 500.40 | 521.44 | +4.20% |

Swapping DRAM for PCM on the memory tier **does not reduce NAND writes at all**
— it raises them slightly (+0.62%) and costs 4% throughput. On its own the PCM
tier buys no endurance. It slows the pipeline (166.18 s of PCM write media time
across the run), the run stretches 4.2%, and marginally more blocks churn
through the memory tier in that longer window.

#### (c) → (d): the policy's effect, on the same tier

| quantity | (c) | (d) | Δ |
|---|---|---|---|
| far-tier write blocks | 844,572 | 426,574 | **−49.49%** |
| far-tier write bytes | 7,084,783,435,776 | 3,578,362,068,992 | **−49.49%** |
| memory-tier write blocks | 424,334 | 426,382 | +0.48% |
| throughput (tok/s) | 14,010.86 | 20,853.77 | **+48.84%** |
| e2e p50 (s) | 184.99 | 96.09 | −48.06% |
| duration (s) | 521.44 | 350.33 | −32.81% |

**The entire write saving comes from the policy, not the tier.** The policy also
pays back the PCM tier's latency cost several times over, because the demote
writes it removes were being charged.

#### (a) → (b): the SSD → HBF organization step, for context

Far-tier writes rise **+15.84%** (724,584 → 839,348), throughput falls 8.69%,
hits fall 492 → 461. This step is **confounded by construction** and should not
be read as a clean media comparison: it simultaneously changes GPU HBM
(141 → 117.5 GiB, 4.8 → 3.6 TB/s, so fewer GPU KV blocks) *and* shrinks the far
tier 5.4× (2 TiB → 376 GiB, so far-tier evictions rise 102,196 → 373,594 and
evicted blocks must be re-admitted and re-written). Both effects push writes up.
`(b2)` isolates part of it: doubling the package flash to 752 GiB while dropping
HBM further (94 GiB, 2.56 TB/s) *lowers* writes to 820,970 and raises throughput
to 15,109.55, i.e. far-tier capacity matters more than the HBM give-up here.

**Flagged — non-monotonic series and scheduling feedback.**
1. Hits are non-monotonic across (b) → (c) → (d): 461 → 479 → 485, while misses
   fall 39 → 21 → 16. A configuration that is *slower* (c) should not hit more
   often than (b) on a fixed trace. It does because arrivals are paced against
   simulated time: a slower run spreads the same 500 requests over a longer
   window, so more of each request's prefix is still resident when the next
   arrives. This is arrival-pacing feedback, not a cache-quality result, and it
   propagates into the far-tier read counts (6,565 → 3,872 → 2,106).
2. Every per-run counter in the table is accumulated over a *different*
   simulated duration (350–521 s). Write *rates* and the per-config lifetime
   figures in Phase 3 normalise by duration; raw counts do not.
3. Config (d) is the only run with a preemption and a recomputation (1 each).
   Both are single events and neither is large enough to move the totals, but
   (d) is no longer strictly in the "no preemption anywhere" regime the earlier
   phases held to.

**Flagged — the far tier has no shared-bandwidth model.** All 8 workers share one
`MooncakeStore` and one `OffloadTier` (`get_mooncake_service` keys on config
fingerprint, so the 8 workers resolve to the same service), but the tier charges
per-operation latency to whichever worker called it, with no arbitration for a
shared write bandwidth. Achieved far-tier write bandwidth is therefore **3.2–4.4×
the nominal 3.0 GiB/s**:

| | (a) | (b) | (c) | (d) | (b2) |
|---|---|---|---|---|---|
| achieved (GiB/s) | 12.389 | 13.104 | 12.654 | 9.513 | 11.703 |
| over-delivery vs 3.0 nominal | 4.13× | 4.37× | 4.22× | 3.17× | 3.90× |

The ceiling is ~8 × 3.0 = 24 GiB/s (one stream per worker). This inflates
absolute write *rate* and so deflates the absolute lifetimes in Phase 3 by a
similar factor. It affects all configurations the same way, so the **ratios**
between them — which is what Phase 3's multiplier needs — are not affected.

### STEP 2.3 — faithfulness check

Config (b) qualitatively reproduces the HBF paper's central finding. Its far tier
is overwhelmingly write-dominated: **6,565 read blocks against 839,348 write
blocks, a read:write ratio of 1 : 127.9** (identical in bytes, 55.07 GB read vs
7.04 TB written; writes are 99.22% of far-tier block traffic). Writes do not
merely exceed reads, they exceed them by more than two orders of magnitude, which
is the qualitative shape the HBF work reports for a KV far tier. The reference
SSD configuration (a) shows the same sign but far weaker — 1 : 14.9 — precisely
because its 2 TiB tier retains blocks long enough to be read back, while the
376 GiB package-local tier evicts them first (far-tier evictions 102,196 → 373,594).
So the write-domination is not a property of the media; it is a property of a far
tier that is small relative to the write stream, which is what the package-local
organization forces. No attempt was made to match absolute numbers — different
trace, reconstructed tier — and their five-model / two-GPU / four-trace sweep was
not run.

## PHASE 3 — lifetime table

TLC endurance taken as **3,000 P/E cycles**. Writes per cell = total bytes
written ÷ tier capacity (full-capacity overwrites), which assumes WAF = 1 and
ideal wear levelling; the WAF = 1.05141 column applies the write amplification
measured on this same access pattern in the earlier MQSim occ70 run. Cycles
consumed over the trace = writes per cell; the rate is that divided by the run's
own simulated duration.

| config | tier capacity | **bytes written to NAND** | **writes/cell** | duration (s) | **cycles/s** | **lifetime @ WAF 1** | lifetime @ WAF 1.051 |
|---|---|---|---|---|---|---|---|
| (a) SSD-Mooncake | 2,199,023,255,552 B (2 TiB) | 6,078,251,139,072 (6.078 TB) | 2.764 | 456.92 | 0.006049 | **5.740 d** | 5.459 d |
| (b) HBF, always | 403,726,925,824 B (376 GiB) | 7,040,961,347,584 (7.041 TB) | 17.441 | 500.40 | 0.034854 | **0.996 d** | 0.948 d |
| (c) HBF+PCM, always | 403,726,925,824 B | 7,084,783,435,776 (7.085 TB) | 17.549 | 521.44 | 0.033654 | **1.032 d** | 0.981 d |
| (d) HBF+PCM, if_read | 403,726,925,824 B | 3,578,362,068,992 (3.578 TB) | 8.863 | 350.33 | 0.025300 | **1.372 d** | 1.305 d |
| (b2) HBF-2, always | 807,453,851,648 B (752 GiB) | 6,886,953,451,520 (6.887 TB) | 8.529 | 455.90 | 0.018709 | **2.231 d** | 2.122 d |

Worked example, config (b): `7,040,961,347,584 B ÷ 403,726,925,824 B = 17.441`
writes per cell over `500.40 s` → `17.441 / 500.40 = 0.034854` cycles/s →
`3000 / 0.034854 = 86,073 s = 0.996 days`.

**Lifetime multipliers on the same 376 GiB device:**

| | multiplier | lifetime |
|---|---|---|
| **(b) → (d)** | **1.3776×** | 0.996 d → 1.372 d |
| (c) → (d) | 1.3302× | 1.032 d → 1.372 d |
| (b) → (c) | 1.0356× | 0.996 d → 1.032 d |

The (b)→(d) multiplier is **1.38×, not the ~2× the 49.5% write reduction alone
would give**. The gap is the duration term: (d) is so much faster that it
compresses the same 500 requests into 350 s instead of 500 s, so its *rate* of
cycle consumption falls by less than its write count does. Per unit of work
delivered the saving is the full ~2×; per unit of wall-clock time under a
saturating arrival stream it is 1.38×. Both are true and they answer different
questions — for a device that will be driven at saturation regardless, 1.38× is
the honest figure.

**These absolute lifetimes are implausibly short and should not be quoted
on their own.** Roughly one day for the HBF configurations follows directly from
driving a 376 GiB tier with a 7 TB write stream in 500 s. Two things inflate the
rate: the missing shared-bandwidth model flagged in Phase 2 (3.2–4.4× the nominal
tier bandwidth), and the fact that the standing workload saturates the far tier
by construction (the earlier MQSim Step 7 analysis measured the undownsampled
TokenSim SSD stream at ~110× a real device). The *ratios* survive both; the
absolute day counts do not.

## Go/no-go for NVSim

**Lifetime multiplier from (b) to (d): 1.3776×** (0.996 → 1.372 days at TLC 3000,
WAF 1; 0.948 → 1.305 days at the measured WAF 1.051). On a per-request rather
than per-second basis the write reduction is 49.49%, i.e. 1.98×. Both figures
should travel together — the first is what a saturated device sees, the second is
what a fixed amount of served work costs in NAND.

**What the data says should change in how the PCM device model is parameterized:**

- **Parameterize PCM writes carefully and PCM reads barely at all.** Across the
  whole of config (c) the memory tier absorbed **166.180 s of write media time
  and 0.000 s of read media time** — memory-tier read blocks were literally zero
  (98 in (b), 96 in (d), against 420k+ writes). Reads come from the far tier, not
  the memory tier, because mean memory residency is 3.0–4.6 s while
  time-to-first-read is ~91 s. NVSim effort should go into PCM *program* energy,
  latency and endurance; PCM read latency is very nearly a free parameter on this
  workload, and `pcm_read_bw_gbps` / `pcm_read_latency_us` can be left at
  defaults without materially changing any result above.
- **The PCM tier must be sized against a ~3 s residency, not a cache lifetime.**
  The residency histogram is concentrated in 1–10 s (242,579 of 420,238 evictions
  in (c); 293,229 of 422,286 in (d)) with nothing at all beyond 30 s. A PCM tier
  at this capacity is a write buffer with a few-second dwell, so its own
  endurance matters: 424,334 memory-tier write blocks over 521 s against a
  32 GiB tier is ~103 full-capacity overwrites — three orders of magnitude more
  cycling per cell than the NAND tier sees, which PCM's ~10^8 endurance absorbs
  but which should be computed explicitly rather than assumed safe.
- **The write asymmetry ratio is the sensitive parameter, and 5× was assumed,
  not derived.** `pcm_write_bw_gbps=20.0` vs `pcm_read_bw_gbps=100.0` is the
  assumption that produced the 4.03% throughput cost in (b)→(c). That cost is
  what the PCM tier has to earn back, so NVSim should pin this ratio first.
- **A demote's PCM read is currently uncharged** (see the modelling note in Step
  2.2). Once NVSim gives a real PCM read cost, charging it will widen the
  (c)→(d) gap, since `always` demotes ~420k blocks and `if_read` demotes 192.
- **Do not carry the absolute lifetimes into NVSim.** Fix the far tier's missing
  shared-bandwidth model first, or normalise per request served; the 1.38× and
  1.98× ratios are the transferable results.

**Stopping here per instruction.** NVSim, energy and thermal work not started.

---

# PHASE 4 — moving the policy to admission (write-back)

Follow-on from Phase 1's structural finding: because the store write-throughs on
admission, the eviction-time demote is a byte-identical duplicate and any
eviction-time policy caps at ~50%. This phase moves the decision to admission.

## STEP 1 — write-back admission policy

**Naming collision, resolved rather than papered over.** `admission_policy`
already exists in `kv_connector_extra_config` and means something else — whether
a put is admitted at all, values `{"always", "never"}`, checked at
`store.py:176` and covered by an existing test. Overloading it with
`write_through`/`write_back` would have destroyed the ability to express
"never admit". The new knob is therefore **`admission_write_policy`**
(`"write_through"` default = stock, `"write_back"`). Because the two value sets
are disjoint, `parse_mooncake_config` also **accepts `admission_policy:
"write_through"|"write_back"` and routes it** to the new field, so configs
written either way work; supplying both names with conflicting values raises.

Semantics, kept orthogonal to `demote_policy` rather than folded into it:

| | `demote_policy=always` | `demote_policy=if_read` |
|---|---|---|
| `write_through` (stock) | write at admission **and** at every eviction (duplicate) | write at admission; skip the duplicate demote for unread victims |
| `write_back` | no admission write; write at every eviction | **no admission write; write only when an evicted block has `read_count > 0`** |

The bottom-right cell is the policy requested. Composing the two knobs this way
keeps `(c)` vs `(d)` a clean isolation of the demote decision under write-back.

Under `write_back` an unread victim has no copy anywhere, so the existing drop
path removes it from the object table entirely — the block is genuinely gone and
a later lookup misses. (Under `write_through` the same path kept the object,
because the admission copy was still on the tier.) One case is deliberately left
writing at admission: a block the memory tier cannot hold at all goes straight to
the offload tier under either policy, since it was never a memory admission.
That path is now counted as `mooncake_memory_bypass_blocks` (zero in every run
below) so the write identity stays exact:

- `write_through`: `ssd_write_blocks == admission_count + blocks_demoted`
- `write_back`: `ssd_write_blocks == memory_bypass_blocks + blocks_demoted`

**Verification — `write_through` is byte-identical.** Against the pre-session
baseline tree: 84 fields identical. Against Phase 2 config (a): **96 fields
identical**, the only new key being `mooncake_memory_bypass_blocks`.

Tests added (`MooncakeWriteBackAdmissionTest`, 13 cases; suite now **67 tests,
all passing**): the `write_through` default, rejection of an unknown value,
the `admission_policy` alias, `admission_policy: "never"` still working,
rejection of conflicting spellings, no offload write at admission under
write-back, an unread victim being dropped and never reaching the far tier (and
the key then missing from the store), a read victim demoting on eviction,
**`ssd_write_blocks == useful_write_blocks == blocks_demoted` under
write_back + if_read**, the write identity under both policies, and byte
conservation (`ssd_write_bytes == ssd_write_blocks × block_bytes`, same for
reads and memory writes, tier occupancy consistent and within capacity, and both
eviction partitions summing to `memory_eviction_count`).

## STEP 2 — the four configurations

`charge_eviction_writes=ON`. (a) and (b) keep `write_through`; (c) and (d) use
`write_back`. (a)/(b) reproduce their Phase 2 rows exactly.

| | (a) SSD, wt/always | (b) HBF, wt/always | (c) HBF+PCM, **wb**/always | (d) HBF+PCM, **wb**/if_read |
|---|---|---|---|---|
| **far-tier write blocks** | 724,584 | **839,348** | **416,748** | **1,536** |
| **far-tier write bytes** | 6,078,251,139,072 | 7,040,961,347,584 | 3,495,935,606,784 | **12,884,901,888** (12 GiB) |
| far-tier read blocks | 48,485 | 6,565 | 6,963 | 32 |
| **read:write at the far tier** | 1 : 14.9 | 1 : 127.9 | 1 : 59.9 | **1 : 48.0** |
| far-tier write share | 93.73% | 99.22% | 98.36% | 97.96% |
| **blocks dropped** | 0 | 0 | 0 | **421,965** |
| **blocks demoted** | 360,244 | 417,626 | 416,748 | **1,536** |
| memory-bypass blocks | 0 | 0 | 0 | 0 |
| memory evictions | 360,244 | 417,626 | 416,748 | 423,501 |
| far-tier evictions | 102,196 | 373,594 | 368,620 | **0** |
| wasted / useful blocks | 359,917 / 327 | 417,336 / 290 | 416,556 / 192 | 421,965 / **1,536** |
| memory-tier write blocks | 364,340 | 421,722 | 420,844 | 427,597 |
| memory-tier read blocks | 263 | 98 | 96 | **1,472** |
| PCM write / read media (s) | — | — | 164.813 / 0.008 | 167.458 / 0.115 |
| mean residency (s) | 4.7383 | 4.4578 | 3.0903 | 1.7576 |
| **store hits / misses** | 492 / 8 | 461 / 39 | 478 / 23 | **492 / 8** |
| **prefix cache hit rate** | 0.1483 | **0.0530** | 0.0534 | **0.0390** |
| **reuse hit blocks** | 65,996 | **23,591** | 23,763 | **17,344** |
| **disk hit tokens** | 779,968 | **106,608** | 112,944 | **24,064** |
| local GPU hit tokens | 306,176 | 275,456 | 272,896 | 253,952 |
| store get count | 59 | 9 | 10 | 1 |
| save / load wait (s) | 2,088.84 / 113.37 | 2,419.74 / 15.42 | 1,399.34 / 16.35 | **238.62 / 1.83** |
| **throughput (tok/s)** | 15,989.36 | 14,599.82 | 20,853.77 | **36,288.28** |
| output QPS | 1.0943 | 0.9992 | 1.4272 | 2.4835 |
| e2e p50 / p99 (s) | 164.09 / 319.96 | 173.80 / 350.74 | 94.73 / 205.54 | **11.79 / 57.95** |
| duration (s) | 456.92 | 500.40 | 350.33 | 201.33 |
| notdone / preempt / recompute | 0/0/0 | 0/0/0 | 0/**1**/**1** | 0/0/0 |

Write reductions relative to (b):

| | far-tier write blocks | reduction vs (b) |
|---|---|---|
| (b) write_through + always | 839,348 | — |
| **(c) write_back + always** | 416,748 | **50.349%** |
| **(d) write_back + if_read** | **1,536** | **99.817%** |

(c) confirms the Phase 1 arithmetic exactly: removing the admission write alone
halves the stream (50.35%), because admission and demote writes were a
one-for-one duplicate pair. (d) then removes almost all of what remains.

### BRANCH — 70% gate

**Far-tier write reduction under write-back is 99.817%, far above the 70% gate,
so this did not stop.** But the gate's stated rationale needs correcting, because
the reduction is this large for the *opposite* reason to the one the gate was
guarding against:

**Reuse is not behaving as the 13.13% read-before-eviction figure suggests.**
Only **1,536 of 423,501 memory evictions (0.363%)** had `read_count > 0`. The
13.13% figure measured reads over an object's whole lifetime, the bulk of which
was spent on the far tier, where time-to-first-read averaged 91.31 s. The
write-back decision is made at *memory* eviction, after a mean residency of
**1.76 s** — roughly 50× shorter than time-to-first-read. So the 99.8% reduction
is not evidence of good reuse prediction; it is the consequence of a predictor
that, at this residency, says "no" almost always and is almost always right about
*that window* while being wrong about the object's eventual reuse.

### What the write reduction costs — `store_hit_count` is misleading

`mooncake_store_hit_count` recovers to 492/500 in (d), equal to (a). **That
counter records lookups with at least one hit block, not how much was hit**, and
it hides a real regression. The block-level measures all fall:

| vs (b) | (b) | (d) | Δ |
|---|---|---|---|
| prefix cache hit rate | 0.0530 | 0.0390 | **−26.4%** |
| reuse hit blocks | 23,591 | 17,344 | **−26.5%** |
| disk hit tokens | 106,608 | 24,064 | **−77.4%** |
| store get count | 9 | 1 | −88.9% |

So write-back + if_read buys its 99.8% write reduction with roughly **a quarter
of the store's block-level reuse**, and it nearly stops using the far tier as a
cache at all (one load over the whole run; far-tier evictions fall to zero
because only 12 GiB of a 376 GiB tier is ever occupied). Throughput still rises
2.49× because on this workload the write traffic, not the cache miss rate, was
the binding constraint — the recompute cost of the lost hits is far cheaper than
the writes avoided.

**Flagged — (d) leaves the saturated regime, so it is not a like-for-like
comparison.** With 500 requests at a 3 QPS target the arrival-limited floor is
166.67 s. Durations sit at 2.74× (a), 3.00× (b), 2.10× (c) and **1.21× (d)** that
floor. (b) is fully backlogged; (d) is close to arrival-limited. Much of (d)'s
latency and hit-count improvement is that regime change, which also feeds back
into reuse (a faster run re-touches a prefix sooner, so more of it is still
memory-resident). The block-level reuse numbers above are the ones that survive
this; `store_hit_count` and e2e latency do not.

## STEP 3 — work-normalized lifetime

**Total work is identical in every configuration** — the same 500 requests and
the same 7,305,797 tokens (7,124,855 prefill + 180,942 decode), fixed by the
trace. Per-request and per-token normalization therefore give the *same*
multiplier, and both reduce to the ratio of bytes written. This is the point of
normalizing: it removes the duration term that made the Phase 3 wall-clock
multiplier (1.38×) disagree with the write-volume ratio.

TLC endurance 3,000 P/E cycles; writes per cell = bytes written ÷ tier capacity.

| config | tier capacity | bytes written | writes/cell | **cycles/request** | **cycles/Mtoken** | **requests until EOL** | **tokens until EOL** |
|---|---|---|---|---|---|---|---|
| (a) SSD wt/always | 2 TiB | 6,078,251,139,072 | 2.764 | 0.005528 | 0.3783 | 542,678 | 7.93 × 10⁹ |
| (b) HBF wt/always | 376 GiB | 7,040,961,347,584 | 17.440 | 0.034880 | 2.3871 | **86,010** | 1.257 × 10⁹ |
| (c) HBF+PCM wb/always | 376 GiB | 3,495,935,606,784 | 8.659 | 0.017318 | 1.1852 | 173,227 | 2.531 × 10⁹ |
| (d) HBF+PCM wb/if_read | 376 GiB | 12,884,901,888 | 0.032 | 0.000064 | 0.0044 | **47,000,000** | 6.867 × 10¹¹ |

Worked example, config (b): `7,040,961,347,584 ÷ 403,726,925,824 = 17.440`
writes per cell for 500 requests → `17.440 / 500 = 0.034880` cycles/request →
`3000 / 0.034880 = 86,010` requests before the tier reaches 3,000 P/E cycles.

**(b) → (d) multiplier, both bases:**

| basis | (b) | (d) | multiplier |
|---|---|---|---|
| **per request served** | 86,010 requests | 47,000,000 requests | **546.45×** |
| **per token generated** | 1.257 × 10⁹ tokens | 6.867 × 10¹¹ tokens | **546.45×** |
| (b) → (c), per request | 86,010 | 173,227 | 2.014× |
| (c) → (d), per request | 173,227 | 47,000,000 | 271.32× |

**Absolute day counts are not quotable, and are deliberately omitted from this
table.** The far tier has no shared-bandwidth model: all 8 workers resolve to one
`MooncakeStore` and one `OffloadTier` (`get_mooncake_service` keys on the config
fingerprint), but the tier charges per-operation latency to whichever worker
called it, with no arbitration for a shared write bandwidth. Achieved far-tier
write bandwidth is therefore **3.2–4.4× the nominal 3.0 GiB/s** (measured:
4.13× / 4.37× / 4.22× / 3.17× for the Phase 2 configs), with a ceiling near
8 × 3.0 = 24 GiB/s. Any lifetime expressed per wall-clock second inherits that
error directly. The per-request and per-token figures above do not, because they
divide by work the trace fixes rather than by simulated time.

**The 546× is arithmetically correct and operationally misleading on its own.**
It says the far tier receives 12 GiB over a run in which (b) writes 6.56 TiB —
which is true, and is exactly why it should be read as *the far tier has stopped
being used*, not as *the far tier now lasts 546× longer while doing the same
job*. Under (d) the 376 GiB HBF tier is 97% empty, serves one load in the entire
run, and never evicts. The honest pairing for any quotation of 546× is the
−26.5% block-level reuse and −77.4% disk hit tokens it costs. Config (c) is the
more defensible operating point to carry forward: **2.01× work-normalized
lifetime for a +0.8% change in prefix cache hit rate** (0.0530 → 0.0534, i.e.
none), obtained purely by not writing the same bytes twice.

**Stopping here per instruction. No NVSim work started.**

---

# PHASE 5 — is the reuse distance real, and does config (c) hold up?

## STEP 1 — is the ~91 s reuse distance trace-specific?

**No local coding/agentic trace exists.** A filesystem search found only the
Mooncake conversation trace (`mooncake_real_500.jsonl` and the fuller
`mooncake_conv_trace.jsonl`); the other `dataset/*.jsonl` files are 300–600 byte
examples. So the comparison was constructed from the existing trace, as
instructed — but the intended segmentation turned out to be degenerate, and that
degeneracy is itself the answer.

### The trace's own turn structure

All 500 requests share one root hash, so "conversation" cannot be read off the
root. Turn structure was instead derived by longest-shared-prefix: each request's
predecessor is the earlier request with the longest common `hash_ids` prefix
(ties → most recent), and the inter-turn gap is the arrival-time difference.
499/500 requests have such a predecessor; median shared prefix is 32 blocks
(exactly one 512-token trace hash) out of a mean 891 blocks per request.

| inter-turn gap (s) | mean | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| whole population (n=499) | 10.082 | 0.000 | **0.000** | **0.000** | **0.000** | 53.999 | 147.000 |

**The tightest-gap quartile is not separable: p25 = p50 = p75 = 0.000 s.**
376 of 499 requests (75.4%) arrive at the *same* timestamp as their
shared-prefix predecessor. A quartile split would compare 0 s against 0 s, so
the segmentation below is reported as zero-gap vs nonzero-gap instead, which is
the finest split the data supports.

### Measured reuse distance, in simulation

Runtime monkeypatch probe (`reuse_probe.py`, no repo source changed) recording
per block: admission time, first-read time, whether that read happened while the
block was still memory-resident, residency at eviction, and the reading request.
One correction made during this step: the first version carried a first-read
timestamp across re-admissions of the same key, producing negative
time-to-first-read; it now resets per incarnation.

| config (floor ratio) | blocks written | ever read | read **while memory-resident** | TTFR p50 | TTFR p90 | TTFR p99 | residency mean | residency p50 | evictions with `read_count>0` |
|---|---|---|---|---|---|---|---|---|---|
| (a) SSD 2 TiB (2.74×) | 359,764 | 49,029 (**13.628%**) | 327 (0.091%) | **156.147** | 215.981 | 285.816 | 4.738 | 3.460 | 0.0908% |
| (b) HBF wt/always (3.00×) | 371,477 | 5,632 (1.516%) | 32 (0.009%) | **20.922** | 46.568 | 47.874 | 4.458 | 2.804 | 0.0694% |
| (c) HBF+PCM wb/always (2.10×) | 370,855 | 6,963 (1.878%) | 32 (0.009%) | **38.285** | 38.285 | 53.740 | 3.090 | 2.364 | 0.0461% |
| (d) HBF+PCM wb/if_read (1.21×) | 372,097 | 1,536 (0.413%) | 1,536 (0.413%) | **0.030** | 0.030 | 0.270 | 1.758 | 1.719 | 0.3627% |

Config (a) independently reproduces the historical figure: **13.628% ever read**
against the 13.13% from the earlier MQSim Step 1 extraction.

Segmented by the reading request's trace-level gap (config (a), the least
censored):

| segment | n | mean TTFR | p50 | p90 |
|---|---|---|---|---|
| zero-gap readers (trace gap = 0 s) | 2,144 | **53.375** | 66.892 | 66.892 |
| nonzero-gap readers | 46,885 | **139.546** | 161.159 | 215.981 |

### The finding: the reuse distance is mostly *not* a trace property

Three things say so:

1. **The trace's own gaps are ~0.** Median inter-turn gap is 0.000 s, not 91 s.
   The human-scale multi-turn gap hypothesis is not what the timestamps show.
2. **TTFR moves by four orders of magnitude across configs on the identical
   trace** — 156.1 s (a) → 38.3 s (c) → 0.030 s (d) — tracking the backlog
   ratio (2.74× → 2.10× → 1.21× the 166.67 s arrival-limited floor). A trace
   property cannot do that.
3. **Trace structure does matter, but by ~2.6×, not by the 11–29× needed.**
   Zero-gap readers see 53.4 s vs 139.5 s for nonzero-gap — a real effect in the
   right direction, but even 53.4 s exceeds config (a)'s 4.74 s mean residency
   by 11×.

So the ~91–156 s reuse distance is **dominated by service backlog, not by turn
structure**. A coding or agentic trace with tighter turns would shorten it, but
on this evidence not by enough to close an 11–29× gap on its own. That also
means the earlier "~91 s is a human-scale multi-turn gap" reading in Phase 4 was
wrong, and is corrected here.

**Censoring caveat.** Config (d)'s 0.030 s p50 is survivorship-biased: `if_read`
drops unread blocks at ~1.76 s, so blocks whose reuse distance exceeds that
never get a TTFR at all. Its distribution is censored by its own policy. The
unbiased estimate comes from (b)/(c), which retain everything: **p50 38.285 s,
p99 53.740 s**.

### Is there a reuse profile where `if_read` retains rather than discards?

Yes — and it is a capacity condition, measured rather than extrapolated. The
requirement is **memory-tier residency ≥ reuse distance**. Under LRU, residency
≈ capacity ÷ admission rate, so capacity is the lever. Sweeping the PCM tier
under `write_back + if_read`, everything else as config (d):

| PCM capacity | GiB | mean residency (s) | evictions `read_count>0` | far-tier write blocks | prefix hit rate | reuse hit blocks | vs (b) |
|---|---|---|---|---|---|---|---|
| 4,096 | 32 | 1.758 | 0.363% | 1,536 | 0.0390 | 17,344 | 73.5% |
| 16,384 | 128 | 6.881 | 0.374% | 1,536 | 0.0390 | 17,344 | 73.5% |
| 46,000 | 359 | 19.352 | 0.562% | 2,133 | 0.0402 | 17,877 | 75.8% |
| **65,536** | **512** | **27.456** | **0.828%** | **2,940** | **0.0529** | **23,523** | **99.7%** |
| 131,072 | 1024 | 57.552 | 2.238% | 6,142 | 0.0649 | 28,867 | 122.4% |

Reference (b): prefix hit rate 0.0530, reuse hit blocks 23,591, far-tier writes
839,348.

**The crossover is ~512 GiB of PCM — 16× the current 32 GiB tier.** There,
residency (27.46 s) reaches the same order as the unbiased median reuse distance
(38.29 s), `reuse_hit_blocks` recovers to **99.7% of (b)** and prefix hit rate to
0.0529 vs 0.0530 — i.e. **no hit-rate loss at all** — while far-tier writes are
2,940 against (b)'s 839,348, a **99.65% reduction**.

The mechanism is not that the predictor gets smarter. It is that a 512 GiB PCM
tier **absorbs the reuse itself**, so the blocks `if_read` discards are ones
nothing would have asked the far tier for anyway; the NAND tier degenerates into
a small archive of genuinely twice-read blocks.

Two caveats on that number. A first-order estimate (capacity = admission rate ×
target residency) predicted 359 GiB, about 1.4× optimistic, because raising
capacity also changes the admission rate it assumed constant — the measured
65,536 is the number to use. And **512 GiB of PCM exceeds the 376 GiB HBF far
tier it is feeding**, which is architecturally odd and needs a decision rather
than an assumption: at that ratio the far tier's role is no longer "the cache"
but "the overflow archive". The Mooncake memory tier is host/CXL-attached in this
model, not in-package, so 512 GiB does not compete for package beachfront with
the HBF stacks — but it is a substantial and separate cost.

## STEP 2 — config (c), solidified

(b) and (c) were rerun from scratch. Both reproduce their Phase 4 results exactly:
the only differing fields are `simulator_wall_time` and `mooncake_pool_keys` (a
64-entry sample whose order depends on worker aggregation) — the same two fields
the verification protocol excludes. **Every numeric result is deterministic.**

| metric | (b) HBF, wt/always | (c) HBF+PCM, wb/always | Δ |
|---|---|---|---|
| **far-tier write blocks** | 839,348 | **416,748** | **−50.35%** |
| **far-tier write bytes** | 7,040,961,347,584 | **3,495,935,606,784** | **−50.35%** |
| far-tier read blocks | 6,565 | 6,963 | +6.06% |
| blocks dropped / demoted | 0 / 417,626 | 0 / 416,748 | −0.21% |
| memory evictions | 417,626 | 416,748 | −0.21% |
| far-tier evictions | 373,594 | 368,620 | −1.33% |
| **prefix cache hit rate** | 0.0530 | **0.0534** | **+0.73%** |
| **reuse hit blocks** | 23,591 | **23,763** | **+0.73%** |
| **disk hit tokens** | 106,608 | **112,944** | **+5.94%** |
| reuse hit tokens | 377,456 | 380,208 | +0.73% |
| local GPU hit tokens | 275,456 | 272,896 | −0.93% |
| store hit count (lookup-level) | 461 | 478 | +3.69% |
| store miss count | 39 | 23 | −41.03% |
| throughput (tok/s) | 14,599.82 | 20,853.77 | +42.84% |
| e2e p50 / p99 (s) | 173.80 / 350.74 | 94.73 / 205.54 | −45.50% / −41.40% |
| duration (s) | 500.40 | 350.33 | −29.99% |
| preemptions / recomputed tokens | 0 / 0 | 1 / 282 | — |

**Block-level hit metrics are confirmed unchanged — not merely
`store_hit_count`.** All three block-level measures move *up* slightly:
prefix cache hit rate +0.73%, reuse hit blocks +0.73% (23,591 → 23,763), disk hit
tokens +5.94%. This is the check Phase 4 showed config (d) failing (−26.5% reuse
hit blocks, −77.4% disk hit tokens); (c) passes it. Nothing is discarded under
(c) — `blocks_dropped` is 0 — so this is expected, and the numbers confirm it.

**Work-normalized lifetime** (TLC 3,000 P/E; 376 GiB tier; 500 requests,
7,305,797 tokens, fixed by the trace):

| config | writes/cell | cycles/request | cycles/Mtoken | **requests until EOL** | **tokens until EOL** |
|---|---|---|---|---|---|
| (b) | 17.440 | 0.034880 | 2.3871 | 86,010 | 1.257 × 10⁹ |
| (c) | 8.659 | 0.017318 | 1.1852 | **173,227** | **2.531 × 10⁹** |

**(b) → (c) multiplier: 2.014× per request served and 2.014× per token
generated** — identical, because total tokens is fixed by the trace, so both
reduce to the ratio of bytes written.

**Regime check — (c) stays saturated; it has not left the regime as (d) did.**
Against the 166.67 s arrival-limited floor: **(b) 3.002×, (c) 2.102×,** versus
(d) at 1.208×. (c) is unambiguously still backlogged, so its comparison with (b)
is like-for-like in a way (d)'s was not. It is, however, meaningfully *less*
saturated than (b) (2.10× vs 3.00×), so (c)'s latency and throughput gains are
partly a congestion effect and should not be quoted as a pure media result; the
write-volume and lifetime figures are unaffected by this, since they are
normalized by work rather than time.

One flag: (c) records 1 preemption and 282 recomputed tokens against (b)'s zero.
Both are negligible against 7.3 M tokens but mean (c) is not strictly in the
"no preemption anywhere" regime the earlier phases held to.

### Config (d), retained as a negative result

`write_back + if_read` at the stock 32 GiB PCM tier removes 99.817% of far-tier
writes but is **not** the headline result. Only **1,536 of 423,501 memory
evictions (0.363%)** had `read_count > 0`, because the policy decides at a mean
residency of 1.76 s while the unbiased median reuse distance is 38.29 s — a 22×
gap. It therefore discards blocks that would have been reused rather than dead
ones, costing −26.5% reuse hit blocks and −77.4% disk hit tokens, and it leaves
the saturated regime (1.21× floor), making its latency and throughput numbers
non-comparable. Step 1 above shows this is a capacity failure, not a policy
failure: at 512 GiB of PCM the same policy costs nothing.

## Go/no-go for NVSim

**Lifetime multiplier to carry forward: 2.014×**, work-normalized, identical per
request served and per token generated, from config (b) → (c) — i.e. from
write-through admission to **write-back admission with unconditional demotion**.
It is bought with no hit-rate cost (prefix cache hit rate 0.0530 → 0.0534, reuse
hit blocks +0.73%), in the same saturated regime (3.00× → 2.10× the arrival
floor), and it comes entirely from not writing the same bytes to NAND twice.
Absolute day counts remain not quotable: the far tier has no shared-bandwidth
model (8 workers each charge per-operation latency against one `OffloadTier`,
delivering 3.2–4.4× the nominal 3.0 GiB/s), so any per-wall-clock-second lifetime
inherits that error while per-request and per-token figures do not.

**`if_read` is conditional, not dead.** It fails at the stock 32 GiB PCM tier
(1.76 s residency vs 38.29 s median reuse distance → 0.363% retention, −26.5%
block-level reuse). It succeeds at **65,536 blocks / 512 GiB of PCM**
(27.46 s residency), where it recovers 99.7% of config (b)'s reuse hit blocks and
an equal prefix hit rate while cutting far-tier writes 99.65%. The open question
is not the policy but whether 512 GiB of PCM — larger than the 376 GiB HBF tier
it feeds — is a defensible configuration. That is a cost and packaging question,
and it should be settled before any `if_read` result is promoted.

**What NVSim now needs to produce, and what it does not.**

- **Irrelevant here: PCM read paths.** Across config (c) the memory tier absorbed
  **164.813 s of write media time against 0.008 s of read media time**, with
  memory-tier read blocks of 96 against 420,844 writes. Reads are served from the
  far tier or from the GPU-local prefix cache, essentially never from PCM.
  `pcm_read_latency_us` and `pcm_read_bw_gbps` can stay at defaults; NVSim effort
  spent on PCM read paths will not change any number in this study.
- **Needed, in priority order: (1) PCM program energy per bit**, since the power
  half of the thesis rests on it and the tier absorbs 420,844 block writes
  (3.2 TiB) per 350 s run; **(2) PCM write latency and write bandwidth**, which
  set the 5× write/read asymmetry currently *assumed* (`pcm_write_bw_gbps=20.0`
  vs `pcm_read_bw_gbps=100.0`) — that assumption produced the 4.03% throughput
  cost the tier has to earn back, and it is the most sensitive unvalidated
  parameter in the model; **(3) PCM area/density at 32 GiB and at 512 GiB**,
  because the `if_read` question above is decided entirely by whether the larger
  tier is buildable.
- **Also needed: PCM cell endurance at this cycling rate.** The memory tier takes
  420,844 block writes over 521 s against a 32 GiB capacity — roughly 103
  full-capacity overwrites per run, three orders of magnitude more cycling per
  cell than the NAND tier sees. PCM's ~10⁸ endurance should absorb it, but the
  number should be computed from NVSim output rather than assumed.
- **One model gap to close first:** a demote reads the block out of PCM before
  writing it to NAND, and that read is currently uncharged. It is the
  conservative direction (it understates `if_read`), but once NVSim supplies a
  real PCM read cost it should be charged, since (c) demotes 416,748 blocks.

**Stopping here per instruction. No NVSim work started.**

---

# PHASE 5 — Policy comparison at the corrected PCM capacity

Follows the design parameters now pinned in `/data/rishabh/MTP/CLAUDE.md`
("Design parameters (authoritative)"). All runs in this phase are **GQA only**.

**Workload caveat:** these runs use the Mooncake FAST'25 conversational trace as
a **stand-in for the target CAG workload**. CAG corpus size, sharing factor and
transient KV per request are OPEN. The reuse structure of the stand-in turns out
to be the binding constraint on this phase's headline metric — see the branch
resolution below.

## Model and block size (recorded per CLAUDE.md)

| | value |
|---|---|
| model | `LLaMa2-70B-GQA` |
| head count `Nhead` | 64 |
| KV head count (`Nhead / Grouped_Num`) | 8 |
| layer count `Nlayer` | 80 |
| hidden dim `Dmodel` | 8192 |
| head dim | 128 |
| **KV precision** | **FP16** — hardcoded in `TokenSim/config/cache_config.py:44-50` (the trailing `× 2`); FP8 is not expressible without a code change |
| parallelism | TP=2, PP=1, DP=4 on 8 workers |
| local KV heads per rank (TP=2) | 4 |
| **KV bytes per token (per rank)** | `4 × 128 × 2 × 2 × 80` = **163,840 B** |
| **block size** | 16 tokens → **2,621,440 B (2.5 MiB)** |

## STEP 1 — audit of what was actually configured

**PCM / memory tier, in bytes.** Configs (b)/(c)/(d) and the capacity sweep ran
`memory_capacity_blocks = 4096` against the MHA `llama-7b` block of
**8,388,608 B**:

```
4096 × 8,388,608 B = 34,359,738,368 B = 32.000 GiB   (CLUSTER pool, 8 workers)
                                      =  4.000 GiB   per GPU
```

**Block size, in bytes.** 8,388,608 B in those runs (MHA). Under the GQA model
this phase requires, it is **2,621,440 B**.

**Discrepancy against the design.** The design is **43.4 GiB per GPU**
(= 347.2 GiB as a cluster pool).

| | per GPU | cluster pool | shortfall |
|---|---|---|---|
| design | 43.4 GiB | 347.2 GiB | — |
| as configured (4096 blk @ 8 MiB) | 4.000 GiB | 32.000 GiB | **10.85×** |
| 4096 blk at the GQA block size | 1.250 GiB | 10.000 GiB | **34.72×** |

So the previously recorded "10× error" is confirmed at 10.85×, and would have
been 34.7× had the same block count been reused for the GQA model.

**Does the hardware entry reserve HBM for model weights? Yes.**
`TokenSim/config/cache_config.py:70-74`:

```python
gpu_memory_bytes = hardware_conf.MM_Card_Num * hardware_conf.Capacity * _GB
...
self.num_gpu_blocks = (gpu_memory_bytes - self.model_param_size) / self.size_per_token // self.block_size
```

`model_param_size` is subtracted unconditionally. With `LLaMa2-70B-GQA` the
unsharded weight estimate is 129,668,218,880 B = 120.763 GiB, i.e. **60.381 GiB
per rank at TP=2**. Against the HBF-1 entry actually used in Phase 2/4:

| entry | `Capacity` | HBM | minus weights | **KV actually available** | vs design (192 GB) |
|---|---|---|---|---|---|
| H200 | 141.0 | 141.00 GiB | 60.381 GiB | 80.619 GiB (57.2% of HBM) | 42.0% |
| **HBF-1 (used in Phase 2/4)** | **117.5** | **117.50 GiB** | **60.381 GiB** | **57.119 GiB (48.6% of HBM)** | **29.7%** |

Under the H3 design weights live in HBF and HBM reserves nothing for them, so
every prior run gave KV **less than a third** of the HBM the design specifies.

## STEP 2 — corrections applied

Tracked files untouched; a scratch catalogue is selected with `--hardware_models`
(`git diff TransformerRoofline/hardware_models.json` is empty).

**HBM.** New entry `H3-KVONLY`, a copy of `H200` with `Capacity` raised so that
KV capacity *after* the unavoidable weight subtraction equals 192 GiB:

```
Capacity = 192 + 60.3815 = 252.3814697265625
check:  252.3814697265625 GiB − 60.3815 GiB = 192.0000 GiB of KV per rank
num_gpu_blocks = 78,643  (vs 23,395 under HBF-1)
```

This is a workaround for the hardcoded subtraction, not a model change: the
subtraction still happens, it is just pre-compensated. Flagged as a deviation
from a literal reading of the design.

**PCM tier.** 43.4 GiB is a **per-GPU** figure; the Mooncake tier is a
**cluster pool** (CLAUDE.md). Both readings were run rather than guessed:

```
per-GPU literal : 43.4 GiB × 2^30 = 46,600,395,162 B / 2,621,440 B = 17,776.64 → 17,776 blocks (43.398 GiB)
cluster-correct : 347.2 GiB       = 372,803,161,293 B / 2,621,440 B = 142,213.12 → 142,213 blocks (347.200 GiB)
```

A third point at the old `4096` blocks (10.0 GiB pool, 1.25 GiB/GPU) is kept as
a control, to separate the effect of the capacity fix from the MHA→GQA switch.

**HBF far tier.** Dies-per-stack is **OPEN**, so no sourced capacity exists. Set
to the lower bound the design already fixes — 8 stacks/GPU × 1 die × 128 GiB
= 1 TiB/GPU = 8 TiB cluster = **3,355,443 blocks**. **`mooncake_ssd_eviction_count = 0`
in all 12 runs**, so the far tier never bound and no result in this phase depends
on the OPEN parameter.

## STEP 3 — results

GQA, `charge_eviction_writes=ON`, 500 requests at 3 QPS, `notdone=0` and
`preempt=0` everywhere. Arrival-limited floor = 500 / 3 QPS = **166.67 s**;
floor ratio = duration ÷ floor (1.00× = arrival-limited, higher = saturated).
Conservation checked per run: `demoted + dropped == memory evictions ==
useful + wasted`.

| config | PCM pool | per GPU | far-tier wblk | far-tier bytes | demoted | dropped | **evict rc>0** | **mean resid** | prefix hit | reuse hit blk | disk hit tok | tok/s | e2e p50 / p99 | floor ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) HBF-NAND, wt/always | **347.2 GiB** | 43.40 | 677,959 | 1,777,228,840,960 | 267,873 | 0 | **3.861%** | **123.97 s** | 0.0785 | 34,919 | 512 | 15,601 | 258.09 / 409.20 s | 2.81× |
| (c) +PCM, wb/always | **347.2 GiB** | 43.40 | 267,880 | 702,231,347,200 | 267,880 | 0 | **3.120%** | **97.45 s** | 0.0785 | 34,919 | 512 | 19,855 | 166.81 / 290.36 s | 2.21× |
| **(d) +PCM, wb/if_read** | **347.2 GiB** | 43.40 | **7,232** | **18,958,254,080** | 7,232 | 267,713 | **2.630%** | **72.45 s** | 0.0778 | 34,631 | 0 | **24,618** | **98.68 / 221.88 s** | **1.78×** |
| (b) HBF-NAND, wt/always | 43.4 GiB | 5.42 | 800,632 | 2,098,808,750,080 | 391,428 | 0 | 0.033% | 18.41 s | 0.0783 | 34,855 | 6,768 | 14,493 | 286.81 / 436.45 s | 3.02× |
| (c) +PCM, wb/always | 43.4 GiB | 5.42 | 392,310 | 1,028,417,126,400 | 392,310 | 0 | 0.033% | 14.18 s | 0.0785 | 34,919 | 512 | 18,023 | 198.54 / 341.09 s | 2.43× |
| (d) +PCM, wb/if_read | 43.4 GiB | 5.42 | 128 | 335,544,320 | 128 | 411,293 | 0.031% | 9.56 s | 0.0778 | 34,631 | 0 | 24,626 | 97.38 / 218.57 s | 1.78× |
| (b) HBF-NAND, wt/always | 10.0 GiB (control) | 1.25 | 816,076 | 2,139,294,269,440 | 405,990 | 0 | 0.213% | 4.27 s | 0.0785 | 34,919 | 512 | 14,492 | 291.67 / 440.28 s | 3.02× |
| (c) +PCM, wb/always | 10.0 GiB (control) | 1.25 | 405,990 | 1,064,278,425,600 | 405,990 | 0 | 0.032% | 3.32 s | 0.0785 | 34,919 | 512 | 17,982 | 203.11 / 342.90 s | 2.44× |
| (d) +PCM, wb/if_read | 10.0 GiB (control) | 1.25 | 128 | 335,544,320 | 128 | 424,973 | 0.030% | 2.20 s | 0.0778 | 34,631 | 0 | 24,624 | 97.38 / 218.58 s | 1.78× |

**Non-monotonicity, flagged:** config (b)'s `evict rc>0` runs 0.213% → 0.033% →
3.861% across increasing capacity. The dip at 43.4 GiB is real, not noise; the
residency histograms show why — at 43.4 GiB the (b) distribution straddles the
10-30 s bucket (199,062 of 391,428), where a small capacity change moves a large
mass of blocks across the eviction boundary.

**Residency distributions at 347.2 GiB** (no evictions at all below 30 s):

| run | 30-60 s | 60-120 s | 120-180 s | 180-240 s |
|---|---|---|---|---|
| (b) | 0 | 144,126 | 87,268 | 36,479 |
| (c) | 0 | 228,922 | 38,958 | 0 |
| (d) | 32,726 | 242,219 | 0 | 0 |

## KEY QUESTION — answered

**Does mean PCM residency now exceed the ~91 s time-to-first-read?**
At 43.4 GiB **per GPU** (347.2 GiB pool): **(b) 123.97 s yes, (c) 97.45 s yes,
(d) 72.45 s no.** At 43.4 GiB read as a *pool* (5.42 GiB/GPU) nothing comes
close (9.6-18.4 s). Caveat: the ~91 s TTFR was measured under the MHA model and
the old HBM sizing; it has not been re-measured under this configuration and may
not transfer.

**Does the read_count>0 eviction fraction rise substantially above 0.363%?**
It rises to **2.630% (d) / 3.120% (c) / 3.861% (b)** — roughly 7-10×, but far
below 30%.

### Branch resolution: the metric is below 10% — but the branch's inference does not hold

By the stated rule, 2.630% < 10% selects *"if_read is dead regardless of
capacity — something other than residency is driving it."* The first half of that
is **not supported by the data**; the second half is, and the driver is
identifiable.

**What is driving it: the stand-in workload's reuse rate caps the metric at
~13%.** Only **8.5%** of admitted blocks are ever reused at all
(`reuse_hit_blocks / admissions` = 34,919 / 410,086). A block cannot be evicted
with `read_count > 0` if nothing ever reads it, so the ceiling on this metric is
`reuse_hit_blocks / evictions` ≈ **13.0%**. **The 30% branch threshold is above
the workload's structural ceiling and is unreachable at any PCM capacity.**

Measured against that ceiling, capacity mattered enormously:

| PCM pool | (b) rc>0 | ceiling | **capture of available reuse** |
|---|---|---|---|
| 10.0 GiB | 0.213% | 8.60% | **2.5%** |
| 43.4 GiB | 0.033% | 8.90% | **0.4%** |
| **347.2 GiB** | **3.861%** | **13.04%** | **29.6%** |

(c) captures 23.9% and (d) 20.9% of their ceilings at 347.2 GiB, against 0.4%
at the void capacity. So residency *is* doing work — the raw percentage just
cannot go past ~13% with this trace.

**Ruled out: HBM sizing.** A control at the old HBM (`Capacity = 117.5`,
57.12 GiB KV/rank) with the corrected 347.2 GiB PCM tier gives **(b) 3.079%,
(c) 3.021%, (d) 1.793%** — the same order, slightly *lower*, despite the store
being read far more (`disk_hit_tokens` 178,800 vs 512). The 192 GiB KV-only HBM
is not what suppresses the metric.

### What the data does say about `if_read`

On outcome rather than on the rc>0 proxy, `if_read` at the corrected capacity is
the strongest configuration measured in this project:

| vs (b) at 347.2 GiB | (d) |
|---|---|
| far-tier write blocks | 7,232 vs 677,959 — **−98.93%** |
| far-tier write bytes | 18.96 GB vs 1,777.23 GB — **−98.93%** |
| reuse hit blocks | 34,631 vs 34,919 — **99.18% retained** |
| prefix hit rate | 0.0778 vs 0.0785 — **−0.89%** |
| throughput | 24,618 vs 15,601 tok/s — **+57.8%** |
| mean e2e (p50) | 98.68 s vs 258.09 s — **−61.8%** |
| floor ratio | 1.78× vs 2.81× — closest to arrival-limited |

The earlier finding that `if_read` "fails at 32 GiB and needs 512 GiB" is
superseded: at the design capacity it removes 98.9% of NAND writes for a 0.9%
prefix-hit-rate cost. The VOID ~4 GiB results should not be cited.

**Saturation:** every configuration is above the arrival-limited floor, so all
e2e and throughput numbers are queueing-influenced; (b) at 2.81× most of all.
Only (d) approaches arrival-limited operation at 1.78×.

Stopping here per instruction. NVSim, thermal and batch-size work not started.

---

# PHASE 6 — Headline verification, reuse structure, and the Tool&Agent contrast

**Configuration status fixed.** `memory_capacity_blocks = 142213`
(347.2 GiB pool = **43.4 GiB/GPU × 8 GPUs**) is the design configuration and the
only one reported as a result. The 43.4 GiB-*pool* (17,776 blk) and ~4 GiB
(4,096 blk) runs from Phase 5 are **controls**, not results. Recorded
permanently in `CLAUDE.md` → "Capacity bookkeeping: per-GPU vs cluster pool".

**Precision limitation.** All numbers here are **FP16**, hardcoded in
`TokenSim/config/cache_config.py` (the trailing `× 2` of `size_per_token`); FP8
is not expressible without a code change. FP8 would halve KV bytes per token,
halve block size, double effective tier capacity in blocks and improve every
write-volume and endurance figure. **FP16 is therefore the conservative
reporting choice and understates the architecture's benefit.** Not changed.

**Warm-up exclusion: there is none**, in this phase or any earlier one. No run
in this project has ever excluded a warm-up prefix; all use the full 500
requests. The two traces are treated identically in this respect.

## STEP 1 — is the headline a measurement artifact? No.

Three independent checks on the 347.2 GiB runs:

**1. Byte conservation is exact.** Every tier's bytes equal blocks × 2,621,440
with no remainder:

| run | far-tier bytes | ÷ blocks | exact? |
|---|---|---|---|
| (b) | 1,777,228,840,960 | 677,959 | **2,621,440 B — yes** |
| (c) | 702,231,347,200 | 267,880 | **2,621,440 B — yes** |
| (d) | 18,958,254,080 | 7,232 | **2,621,440 B — yes** |

Memory-tier and far-tier read bytes are exact multiples too.

**2. `blocks_dropped + blocks_demoted == memory evictions` holds in all three**,
and `useful + wasted` independently equals the same total:

| run | demoted + dropped | evictions | useful + wasted |
|---|---|---|---|
| (b) | 267,873 + 0 = 267,873 | 267,873 ✓ | 10,343 + 257,530 = 267,873 ✓ |
| (c) | 267,880 + 0 = 267,880 | 267,880 ✓ | 8,359 + 259,521 = 267,880 ✓ |
| (d) | 7,232 + 267,713 = 274,945 | 274,945 ✓ | 7,232 + 267,713 = 274,945 ✓ |

The write-path accounting also reconciles against policy: under `write_through`
(b) far-tier writes should be `admissions + demoted` = 410,086 + 267,873 =
**677,959** — exactly the measured value; under `write_back` (c)/(d) they should
be `demoted` only — 267,880 and 7,232, both exact.

*Caveat worth stating:* in (d) the `useful/wasted` split is numerically
identical to the `demoted/dropped` split. That is tautological, not
corroborating — `if_read` demotes exactly the blocks with `read_count > 0`. For
(d) the rc>0 figure is a restatement of the policy, not an independent
measurement. (b) and (c) are the honest measurements of that quantity.

**3. The retained reuse hits are genuinely block-level.** `reuse_hit_blocks`
comes from `sum(req.reuse_hit_blocks)` (`util/results.py:19`), set from
`plan.hit_block_count` in `TokenSim/block/kv_cache_manager.py:87` — a per-request
count of prefix **blocks**. It is a different quantity from
`mooncake_store_hit_count` (**496**, lookup-level, and identical across all runs
— that is the metric that would have been misleading). Independent cross-check:
`reuse_hit_blocks × 16 tokens` reproduces the token counter exactly —
(b) 34,919 × 16 = **558,704** = `local_gpu_hit_tokens`; (d) 34,631 × 16 =
**554,096** = `local_gpu_hit_tokens`.

**Conclusion: the 98.93% / 99.18% headline is not an artifact.**

## STEP 2 — the clean claim, separated from the contaminated one

### Headline (work-normalized, regime-independent)

Total NAND write volume is fixed by the workload and the policy, not by
queueing: the same 500 requests and the same 7,305,797 generated tokens in both
runs (ratio 1.000000).

| **Conversation trace, 347.2 GiB pool** | (b) HBF-NAND | (d) +PCM if_read | change |
|---|---|---|---|
| **far-tier write blocks** | 677,959 | **7,232** | **−98.93%** |
| **far-tier write bytes** | 1,777,228,840,960 | **18,958,254,080** | **−98.93%** |
| **per request** | 3,554,457,682 B | **37,916,508 B** | **−98.93%** |
| **per generated token** | 243,262.8 B | **2,595.0 B** | **−98.93%** |
| reuse hit blocks (block-level) | 34,919 | 34,631 | **99.18% retained** |
| prefix cache hit rate | 0.0785 | 0.0778 | −0.89% |

### Queueing-influenced, reported separately and NOT as a primary result

Every configuration runs above the arrival-limited floor of 500 / 3 QPS =
**166.67 s**, and they do not run at the same distance above it — **(d) at
1.78×, (c) at 2.21×, (b) at 2.81×**. The configurations are therefore in
different queueing regimes, and the latency and throughput gaps below are
**contaminated by that difference**: part of the improvement is the policy, part
is simply that (d) is closer to arrival-limited operation. These numbers are
**not** a primary result and should not be quoted as the benefit of `if_read`.

| queueing-influenced | (b) | (c) | (d) |
|---|---|---|---|
| floor ratio | 2.81× | 2.21× | **1.78×** |
| throughput (tok/s) | 15,601 | 19,855 | 24,618 |
| mean e2e p50 / p99 | 258.09 / 409.20 s | 166.81 / 290.36 s | 98.68 / 221.88 s |

Separating them cleanly would need a run at or below the floor, which this
device configuration cannot reach for this workload.

## STEP 3 — what the never-reused blocks are

Measured on the trace itself (block instance = one (request, hash_id) pair =
one 16-token KV block written). A block is "reused" if the same id appears in a
**later** request of the window.

| conversation trace, first 500 requests | |
|---|---|
| block instances written | 445,540 |
| reused later **in the 500-request window** | 72,977 (**16.38%**) |
| never reused in window | 372,563 (**83.62%**) |

**Fraction of never-reused blocks belonging to requests with no subsequent turn
at all: 100.00%** (372,563 of 372,563; 83.62% of all blocks written). 495 of the
500 requests have no subsequent turn in the window.

**This is structural, not a coincidence.** In these traces a follow-up turn
repeats its predecessor's entire prefix and appends new blocks. So if a request
has *any* subsequent turn, *every one* of its blocks is reused; if it has none,
*none* are. The never-reused set and the no-subsequent-turn set are the same
set by construction. Classifying by chain position confirms it — 494 of 500
requests are the final turn of a chain *within the window*, owning 99.90% of
never-reused blocks; genuine single-turn conversations are 1 request and 0.10%.

### The dominant cause is window truncation, not absence of reuse

The 500-request window is **165 s of a 3,537 s trace — 4.7%**. Extending the
horizon while keeping the same 500 written blocks:

| reuse horizon | span | blocks of the window reused | |
|---|---|---|---|
| first 500 requests (the simulated window) | 165 s | 72,977 | **16.38%** |
| first 1,000 | 330 s | 141,929 | 31.86% (+15.48 pts) |
| first 2,000 | 669 s | 180,553 | 40.52% (+8.67) |
| first 4,000 | 1,302 s | 186,336 | 41.82% (+1.30) |
| full 12,031 | 3,537 s | 188,544 | **42.32%** (+0.50) |

**31.02% of the "never reused" blocks are reused later in the full trace** —
their reuse simply falls outside the simulated window. The true reuse rate of
this workload is **42.32%**, and the 500-request window sees **16.38%**, about
**39% of it**. So:

- ~0.1% of never-reused blocks are genuine single-turn traffic,
- ~31% have their reuse censored by the window,
- the remainder are final-turn-so-far blocks whose conversations do not continue
  within the hour.

**Implication for the headline:** the window under-measures reuse by ~2.6×, so
it under-measures what `if_read` throws away. A longer window would raise the
cost side of `if_read`. It would also raise the benefit side of the PCM tier.
This bounds the confidence in the 99.18% retention figure and is the single
largest methodological caveat on Phase 5/6.

## STEP 4 — Tool&Agent contrast

Same three configs, same 347.2 GiB pool, same GQA model, same (absent) warm-up
exclusion. The Tool&Agent trace was built from
`FAST25-release/traces/toolagent_trace.jsonl` by the **exact** pipeline that
produced the conversation workload file — recovered and verified by
reproducing `mooncake_real_500.jsonl` byte-identically (`build_trace.py`):
first 500 requests, timestamps ms→s, each 512-token hash id expanded to 32
sub-ids and truncated to `ceil(input_length/16)`, `cache_salt` unchanged.

Both traces are rescaled to 3 QPS by `--trace_target_qps`
(`loaders.py:279-292`, linear), giving both a ~166.3 s arrival span:
conversation ×1.008, Tool&Agent ×1.848.

| | Conversation | Tool&Agent |
|---|---|---|
| requests / window span (native) | 500 / 165 s | 500 / 90 s |
| mean input / output length | 14,250 / 362 tok | 9,713 / 206 tok |
| block instances written | 445,540 | 303,755 |
| **mean PCM residency** (c) | **97.45 s** | **127.34 s** |
| mean PCM residency (b) / (d) | 123.97 s / 72.45 s | 134.20 s / 123.34 s |
| **median time-to-first-read** (in-simulator, uncensored (c)) | **80.60 s** | **80.84 s** |
| TTFR mean / p90 / p99 | 94.54 / 151.85 / 276.81 s | 63.99 / 134.21 / 134.21 s |
| **residency > median TTFR?** | (b) ✓ (c) ✓ **(d) ✗** | (b) ✓ (c) ✓ **(d) ✓** |
| **rc>0 fraction** (b) / (c) / (d) | 3.861% / 3.120% / 2.630% | **6.590% / 6.377% / 6.342%** |
| **reuse ceiling** `min(1, reuse_hit/evictions)` | **13.04%** | **100% (not binding)** |
| in-window reuse rate | 16.38% | **31.23%** |
| full-trace reuse rate | 42.32% | **54.19%** |
| prefix cache hit rate | 0.0785 | **0.2927** |
| admissions / evictions | 410,086 / 267,873 (65.3%) | 213,970 / 71,757 (**33.5%**) |
| **far-tier write reduction (b)→(d)** | **−98.93%** | **−98.39%** |
| reuse hits retained by (d) | 34,631 / 34,919 = **99.18%** | 88,768 / 88,768 = **100.00%** |
| floor ratio (b) / (c) / (d) | 2.81× / 2.21× / 1.78× | 1.61× / 1.47× / 1.39× |

### What the contrast settles

**`if_read` is stronger on Tool&Agent, not weaker.** It removes 98.39% of NAND
writes at **zero** hit-rate cost — `reuse_hit_blocks` is 88,768 in all three
configs, identical to the digit. On the trace with 3.7× the prefix reuse
(0.2927 vs 0.0785), the policy costs nothing at all.

**The rc>0 ceiling explanation from Phase 5 applies only to the conversation
trace.** On Tool&Agent the ceiling is not binding (reuse hits 88,768 exceed
evictions 71,757), yet rc>0 is still only 6.59%. So a second mechanism is
doing the work, and it is visible in the admissions/evictions row:

> **Under LRU, the blocks that get evicted are by construction the
> least-recently-used — that is, precisely the ones nothing has touched.**
> A low "evictions with read_count > 0" fraction is what a correctly-working LRU
> tier *produces*; it is not evidence that reuse is absent or that residency is
> too short.

At 347.2 GiB the tier ends both runs exactly full (142,213 blocks resident) and
evicts only 65.3% (conversation) / 33.5% (Tool&Agent) of what it admitted. The
reused blocks are still resident; the evicted ones are the cold tail. This —
not residency, and not HBM sizing (ruled out by the Phase 5 control) — is why
the rc>0 metric stays low. **The metric is a poor proxy for `if_read` viability,
and the 30% / 10% thresholds should not be used to judge it.** Judge it on
write reduction versus retained reuse hits, which is what Step 2 reports.

## STEP 5 — work-normalized NAND lifetime multiplier, (b) → (d)

Work is identical between the configs being compared (same 500 requests, same
generated-token count to 6 decimal places), so per-request and per-token
normalization give the same multiplier — which is the point: the figure is a
property of the policy, not of the regime.

| | Conversation | Tool&Agent |
|---|---|---|
| generated tokens, (b) and (d) | 7,305,797 | 4,959,259 |
| NAND write bytes (b) | 1,777,228,840,960 | 749,016,186,880 |
| NAND write bytes (d) | 18,958,254,080 | 12,079,595,520 |
| per request (b) → (d) | 3,554,457,682 → 37,916,508 B | 1,498,032,374 → 24,159,191 B |
| per token (b) → (d) | 243,262.8 → 2,595.0 B | 151,033.9 → 2,435.8 B |
| **lifetime multiplier, per request** | **×93.74** | **×62.01** |
| **lifetime multiplier, per token** | **×93.74** | **×62.01** |

**Absolute day counts are not quotable from this.** The model has no
shared-bandwidth model for the HBF tier: far-tier service time is a fixed
per-request latency plus a per-flow bandwidth term, with no contention between
the 8 workers sharing the device and no NAND-level queueing, GC or WAF coupling
back into it. The multiplier is a ratio of write *volumes* under identical
workloads and survives that gap; a day count would not. NAND endurance in days
must come from the MQSim side (which supplies WAF) combined with a bandwidth
model that does not yet exist. The earlier Phase-2 day figures in this document
inherit the same gap and should be read as ratios only.

Two further constraints on these multipliers:
- **TLC at 3000 P/E cycles and PCM endurance are OPEN** (`CLAUDE.md`), so the
  multiplier cannot yet be turned into a cell-level lifetime on either tier.
- The PCM tier absorbs the writes the NAND tier no longer takes: (d) writes
  1,093,554,667,520 B to PCM (conversation). Whether PCM endurance absorbs that
  cycling rate is exactly the OPEN parameter.

Stopping here per instruction.

---

# METHODOLOGICAL NOTE — `rc>0`-at-eviction is not a valid proxy for `if_read` viability

Recorded here because two rounds of this project's branch decisions were keyed
on it, and it cannot support that weight.

**The metric.** "Fraction of memory-tier evictions whose block had
`read_count > 0`", computed as `useful_write_blocks / (useful + wasted)`
(`TokenSim/mooncake/metrics.py:135-137`, fed by `record_memory_eviction` in
`store.py:290-294`).

**Why it is not a viability signal.** The memory tier evicts by LRU
(`_evict_memory_lru`, `store.py:283`). LRU by construction selects the
*least recently used* block — that is, preferentially a block nothing has
touched. Blocks that *were* read get moved to the tail of the LRU order by
`_touch` (`store.py:329-333`) and are therefore the last things evicted. **A low
`rc>0`-at-eviction fraction is what a correctly functioning LRU tier produces.**
It says the tier is keeping the right blocks, not that reuse is absent or that
residency is too short.

This was confirmed on both traces at the design capacity. The tier ends every
run exactly full (142,213 blocks resident) and evicts only the cold tail:

| | admissions | evictions | % of admits evicted | reuse hits | `rc>0` |
|---|---|---|---|---|---|
| Conversation (b) | 410,086 | 267,873 | 65.3% | 34,919 | 3.861% |
| Tool&Agent (b) | 213,970 | 71,757 | **33.5%** | 88,768 | 6.590% |

On Tool&Agent the reuse ceiling is *not* binding — 88,768 reuse hits exceed
71,757 evictions, so in principle every eviction could have carried a read — yet
`rc>0` is still only 6.59%, because the read blocks are still resident. The
metric is measuring LRU's ordering, not the workload's reuse.

**The `(d)` tautology.** Under `demote_policy=if_read` the `useful/wasted` split
is *numerically identical* to the `demoted/dropped` split, because the policy
demotes exactly the blocks with `read_count > 0` (`_drops_on_eviction`,
`store.py:326-329`). For config (d), `rc>0` is a restatement of the policy, not
an independent measurement of it. **Only (b) and (c) measure this quantity
honestly.** Any (d) `rc>0` figure previously reported should be read as a
description of what `if_read` did, never as evidence about whether it was right
to do it.

**Use outcome measures instead.** `if_read` should be judged on:

1. **NAND writes removed** — `mooncake_ssd_write_bytes` (b) vs (d),
   work-normalized per request and per token. Fixed by workload and policy, not
   by queueing regime.
2. **Reuse hits retained** — `reuse_hit_blocks` (block-level, from
   `plan.hit_block_count`; *not* `mooncake_store_hit_count`, which is
   lookup-level and was identical across every run in Phases 5-6), plus the
   prefix cache hit rate, and stated against the window's coverage of
   full-trace reuse.

Those two are what Phases 6 and 7 report as the headline. Throughput and
latency remain secondary and queueing-contaminated wherever the configurations
sit at different multiples of the arrival-limited floor.

---

# PHASE 7 — Closing the truncation gap: rerun at W = 5,000 requests

Motivation: at W = 500 the simulated window saw only 16.38% in-window reuse
against a 42.32% full-trace rate, so 31.02% of blocks counted as "never reused"
were in fact reused later. That made the 99.18% retention figure for `if_read`
the most attackable number in Phase 6.

## STEP 1 — where in-window reuse converges

In-window reuse rate at window W = (block instances in the first W requests that
recur in a **later request of the same window**) / (block instances in those W
requests) — i.e. what a simulation of exactly those W requests can observe.

| window W | span (conv / T&A) | Conversation rate | % of full | Tool&Agent rate | % of full |
|---|---|---|---|---|---|
| 500 | 165 / 90 s | 16.38% | 43.9% | 31.23% | 54.8% |
| 1,000 | 330 / 174 s | 21.56% | 57.7% | 39.02% | 68.4% |
| 2,000 | 669 / 336 s | 29.40% | 78.7% | 44.84% | 78.6% |
| 4,000 | 1,302 / 669 s | 33.12% | 88.7% | 50.85% | 89.2% |
| **5,000** | **1,593 / 825 s** | **33.92%** | **90.8%** | **51.87%** | **91.0%** |
| 6,000 | 1,872 / 993 s | 35.25% | 94.4% | 52.73% | 92.5% |
| 8,000 | 2,436 / 1,305 s | 35.73% | 95.7% | 53.68% | 94.2% |
| full | 3,537 s | **37.34%** | 100% | **57.01%** | 100% |

**Chosen window: W = 5,000** — the smallest window within ~10% of the
full-trace rate **for both traces**. W = 4,000 just misses on both (88.7% /
89.2%); a single window is used for both traces so they stay comparable.

Measured on the windows' own blocks, truncation falls sharply:

| | reuse visible in-window | reuse in full trace | **coverage** | censored |
|---|---|---|---|---|
| Conversation W=500 | 72,977 (16.38%) | 188,544 (42.32%) | **38.7%** | 61.3% |
| **Conversation W=5,000** | 1,385,700 (33.92%) | 1,626,388 (39.81%) | **85.2%** | **14.8%** |
| Tool&Agent W=500 | 94,871 (31.23%) | 164,595 (54.19%) | **57.6%** | 42.4% |
| **Tool&Agent W=5,000** | 1,511,109 (51.87%) | 1,669,867 (57.32%) | **90.5%** | **9.5%** |

## STEP 2 — the three configs at W = 5,000

347.2 GiB pool (142,213 blk), GQA, `charge_eviction_writes=ON`, `H3-KVONLY`
hardware, far tier 3,355,443 blk. No warm-up exclusion (there has never been
one). `notdone = 0` everywhere; `demoted + dropped == evictions` and byte
exactness verified in all six runs; `ssd_eviction_count = 0` in all six.

> **Far-tier headroom warning.** At W = 5,000 config (b) on the conversation
> trace holds 3,281,449 distinct far-tier keys against the 3,355,443-block
> capacity — **97.8% full**. It did not evict, so the far tier is still
> non-binding and the results stand, but a modestly longer window would make the
> OPEN dies-per-stack parameter start to matter. (c) is at 93.7%, (d) at 2.1%;
> Tool&Agent is at 50.1% / 46.1% / 1.1%.

### Conversation, W = 5,000

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| far-tier write blocks | 6,420,685 | 3,142,548 | **68,911** |
| far-tier write bytes | 16,831,440,486,400 | 8,238,001,029,120 | **180,646,051,840** |
| blocks demoted / dropped | 3,139,236 / 0 | 3,142,548 / 0 | 68,911 / **3,587,843** |
| memory evictions | 3,139,236 | 3,142,548 | 3,656,754 |
| rc>0 at eviction | 2.486% | 2.803% | *1.884% (tautological)* |
| mean PCM residency | 147.60 s | 116.12 s | 73.32 s |
| **prefix cache hit rate** | **0.1640** | 0.1640 | **0.0931** |
| **reuse hit blocks** | **669,192** | 669,320 | **379,786** |
| disk hit tokens | 5,503,744 | 5,463,808 | **661,360** |
| admissions | 3,281,449 | 3,284,761 | **3,798,967** |
| throughput (tok/s) | 18,957 | 23,832 | 32,214 |
| e2e p50 / p99 | 1159.9 / 1969.1 s | 722.4 / 1265.8 s | 283.9 / 593.4 s |
| floor ratio | 2.12× | 1.69× | 1.25× |

### Tool&Agent, W = 5,000

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| far-tier write blocks | 3,221,751 | 1,546,232 | **37,899** |
| far-tier write bytes | 8,445,626,941,440 | 4,053,354,414,080 | **99,349,954,560** |
| blocks demoted / dropped | 1,539,769 / 0 | 1,546,232 / 0 | 37,899 / **1,733,862** |
| memory evictions | 1,539,769 | 1,546,232 | 1,771,761 |
| rc>0 at eviction | 3.004% | 3.355% | *2.139% (tautological)* |
| mean PCM residency | 148.65 s | 142.18 s | 124.29 s |
| **prefix cache hit rate** | **0.4039** | 0.4042 | **0.3676** |
| **reuse hit blocks** | **1,174,745** | 1,175,585 | **1,069,215** |
| disk hit tokens | 1,948,480 | 1,903,008 | **143,360** |
| admissions | 1,681,982 | 1,688,445 | **1,913,974** |
| throughput (tok/s) | 25,341 | 27,298 | 27,552 |
| e2e p50 / p99 | 113.5 / 620.2 s | 21.6 / 264.0 s | 7.8 / 105.3 s |
| floor ratio | 1.12× | 1.04× | **1.03×** |

### Retention measured against full-trace reuse, not just in-window

`if_read` under `write_back` **destroys** a dropped block — there is no
write-through copy to fall back on (`_drops_on_eviction` pops the object when
the offload tier does not contain the key, `store.py:300-307`). So any reuse
falling outside the window that targets a dropped block is an *additional*
loss the simulation cannot see. That bounds the retention figure:

| | in-window retention (d)/(b) | window coverage | **upper bound** | **lower bound** |
|---|---|---|---|---|
| **Conversation** | **56.75%** (379,786 / 669,192) | 85.2% | **56.75%** | **48.35%** |
| **Tool&Agent** | **91.02%** (1,069,215 / 1,174,745) | 90.5% | **91.02%** | **82.37%** |

Upper bound = censored reuse splits like observed reuse. Lower bound = every
censored reuse is of a block (d) dropped.

## The headline changes: `if_read` is not free

| | W = 500 | **W = 5,000** |
|---|---|---|
| **Conversation** — far-tier write reduction | 98.93% | **98.93%** |
| **Conversation** — reuse hits retained | **99.18%** | **56.75%** (48.35-56.75%) |
| **Conversation** — prefix hit rate (b) → (d) | 0.0785 → 0.0778 (−0.89%) | 0.1640 → 0.0931 (**−43.2%**) |
| **Tool&Agent** — far-tier write reduction | 98.39% | **98.82%** |
| **Tool&Agent** — reuse hits retained | **100.00%** | **91.02%** (82.37-91.02%) |
| **Tool&Agent** — prefix hit rate (b) → (d) | 0.2927 → 0.2927 (0.0%) | 0.4039 → 0.3676 (**−9.0%**) |

**The Phase 6 retention figures do not survive the longer window, exactly as
suspected.** On the conversation trace `if_read` discards **43% of the reuse**,
not 0.8%. The 500-request window was hiding almost all of that cost: it could
only observe 38.7% of the reuse those blocks actually receive.

**The write-reduction side is unchanged** — 98.93% and 98.82%, essentially
identical to W = 500. That claim was work-normalized and regime-independent, and
it held.

So the honest statement of the result is a **trade**, not a free win:

- **Conversation:** −98.93% NAND writes for **−43.2% prefix hit rate**.
- **Tool&Agent:** −98.82% NAND writes for **−9.0% prefix hit rate**.

`if_read` remains clearly attractive on Tool&Agent. On the conversation trace it
is now a genuine design trade-off that has to be argued, not assumed — and
config **(c)** (`write_back` + `always`) becomes interesting in its own right: it
halves far-tier writes (16.83 → 8.24 TB, **−51.1%**) at **zero** hit-rate cost
(0.1640 vs 0.1640, reuse hits 669,320 vs 669,192).

### A second cost of `if_read`: re-admission churn shifts wear onto PCM

Dropped blocks must be re-admitted when they are next requested, so (d) admits
**more** than (b) and writes **more** to the PCM tier:

| | admissions (b) → (d) | PCM write bytes (b) → (d) |
|---|---|---|
| Conversation | 3,281,449 → 3,798,967 (**+15.8%**) | 8.60 TB → 9.96 TB (**+15.8%**) |
| Tool&Agent | 1,681,982 → 1,913,974 (**+13.8%**) | 4.41 TB → 5.02 TB (**+13.8%**) |

`if_read` does not only move wear off NAND — it *adds* 14-16% of wear onto PCM.
With **PCM endurance still OPEN** (`CLAUDE.md`), this is an unpriced cost.

## STEP 3 — lifetime multiplier at W = 5,000

Work is identical within each comparison (same requests, same generated tokens
to 6 dp), so per-request and per-token multipliers coincide.

| | Conversation | Tool&Agent |
|---|---|---|
| generated tokens (b) = (d) | 67,061,296 | 47,497,243 |
| NAND write bytes (b) | 16,831,440,486,400 | 8,445,626,941,440 |
| NAND write bytes (d) | 180,646,051,840 | 99,349,954,560 |
| per request (b) → (d) | 3,366,288,097 → 36,129,210 B | 1,689,125,388 → 19,869,991 B |
| per token (b) → (d) | 250,985.9 → 2,693.7 B | 177,813.0 → 2,091.7 B |
| **multiplier, per request** | **×93.17** | **×85.01** |
| **multiplier, per token** | **×93.17** | **×85.01** |
| vs W = 500 | ×93.74 → **×93.17 (holds, −0.6%)** | ×62.01 → **×85.01 (rises, +37.1%)** |

**Why the conversation multiplier holds and the Tool&Agent one rises.** Two
effects move in opposite directions as the window lengthens:

1. **Higher prefix hit rate reduces admissions per request.** More reuse is
   visible, so fewer blocks need admitting: conversation 820 → 656 blocks/req,
   Tool&Agent 428 → 336 blocks/req. This *lowers* (b)'s write-through cost.
2. **The fixed 347.2 GiB pool is relatively smaller against a 10× larger
   workload, so a larger share of admissions is eventually evicted and
   demoted.** Conversation 536 → 628 demotes/req; Tool&Agent 143 → 308
   demotes/req. This *raises* (b)'s demote cost. (d) is nearly unaffected —
   it demotes only the small read subset.

On the conversation trace the two roughly cancel ((b) 3.554 → 3.366 GB/req), so
the multiplier is flat. On Tool&Agent the demote effect dominates — its demote
rate more than doubles — so (b) writes *more* per request (1.498 → 1.689 GB/req)
while (d) stays small, and the multiplier rises 37%.

This means the multiplier is **not** a window artifact in either direction: it is
stable where the effects cancel and moves in the explainable direction where
they do not. It is, however, a function of the pool-to-workload ratio, so it
should always be quoted with the capacity and the window.

**Absolute day counts remain unquotable.** Unchanged from Phase 6: the far-tier
model is a pure per-operation latency function
(`TokenSim/mooncake/ssd.py:107-121`) with no SimPy resource, no contention among
the 8 workers, and no NAND queueing/GC/WAF coupling. The multipliers are ratios
of write volumes under identical workloads and survive that gap; day counts do
not. TLC 3000 P/E and PCM endurance are OPEN.

## Queueing status at W = 5,000

Reported separately, not as a primary result. Tool&Agent is now close to
arrival-limited (**1.12× / 1.04× / 1.03×** of the 1,666.7 s floor), so its
latency and throughput figures are only lightly contaminated. The conversation
trace remains saturated and spread (**2.12× / 1.69× / 1.25×**), so its latency
and throughput gaps are still substantially regime-driven and should not be
quoted as policy benefit.

Stopping here per instruction.

---

# PHASE 8 — Hardware baseline rebuilt on the correct H3 topology

The `HBF-1` / `HBF-2` entries modelled the wrong architecture. This phase
supersedes them, records the real H3 platform in `CLAUDE.md`, and reruns the
three configurations on it.

## STEP 0 — why the old baseline is wrong, and what it invalidates

`HBF-1` / `HBF-2` (Phase 2, STEP 2.1) were built as **reduced-HBM**
configurations: six HBM sites of which one or two are given over to flash
(`Capacity` 117.5 / 94.0 GiB against the stock 141), plus an **invented −10%
per-stack bandwidth derate** (assumption A5; `BW_TBs` 3.60 / 2.56 against the
stock 4.80). That is a **side-by-side** package in which HBM is surrendered to
make room for flash.

**H3 (SK hynix, IEEE Computer Architecture Letters, Jan–Jun 2026) does not do
this.** It keeps **all** the HBM and **daisy-chains** HBF behind it: the HBM
cubes stay on the GPU shoreline, an address decoder and router in the HBM base
die splits traffic between the two paths, and the GPU reaches HBF *through* the
HBM base die. No HBM capacity is given up and **no bandwidth derate exists** —
A5 has no basis in the paper.

Marked SUPERSEDED in `CLAUDE.md` with this reason. The findings that used them
are kept, not deleted. **Flagged as measured against the wrong topology, and
not to be quoted as a baseline:**

| section | what it contains | why flagged |
|---|---|---|
| **Phase 2, STEP 2.1** | the `HBF-1`/`HBF-2` derivation and assumptions A1–A6 | the entries themselves |
| **Phase 2, STEP 2.2** | the four-configuration table, incl. the `(b2)` HBF-2 run | every column ran on `HBF-1`/`HBF-2` |
| **Phase 2, STEP 2.3** | faithfulness check | same runs |
| **Phase 3** | the lifetime table and its multipliers | derived from the Phase 2 write volumes |
| **Phase 3, "Go/no-go for NVSim"** | the go/no-go read | derived from the Phase 3 table |
| **Phase 4, STEPS 1–3 + the 70% BRANCH** | write-back admission policy results | `(a)`/`(b)` explicitly "reproduce their Phase 2 rows exactly" |

Two compounding defects in those sections, for the record: HBM was **48.6% of
the design value** (57.1 GiB of KV per rank after the weight subtraction, vs
192 GiB), and the far tier rested on the unsourced "16 dies per stack"
assumption. Phase 5 already diagnosed the first; this phase removes both.

**Phases 5–7 are not flagged for HBM topology** — they used `H3-KVONLY`, which
already gives the full 192 GiB to KV with no derate. They *are* superseded on
far-tier capacity (they used the explicitly-labelled 8 TiB lower bound in place
of the then-OPEN dies-per-stack figure) and on GPU class (H200, not B200).

## STEP 1 — the H3 hardware entry

### Architecture recorded in `CLAUDE.md`

Per GPU, B200-class, all values from the H3 paper:

| | |
|---|---|
| **HBM3e** | 192 GB, 8 TB/s total — 8 cubes × 24 GB × 1 TB/s, TDP 40 W/cube |
| **HBF** | 3 TB, 8 TB/s total — 8 stacks × 384 GB, TDP 160 W/cube |
| **Topology** | HBM cubes on the GPU shoreline; HBM and HBF **daisy-chained**; address decoder + router in the HBM base die splits traffic; unified address space, divided regions |
| **LHB** | 40 MB SRAM in the HBM base die, 8.06 mm² of a 121 mm² base die (~6.7%); prefetches from HBF so its µs latency is hidden. `LHB capacity = 2 × BW_HBF × Latency_HBF`, with BW 1 TB/s per cube and latency **20 µs assumed from SLC NAND** |
| **HBF vs HBM** | up to 16× capacity, comparable bandwidth, slower access (µs vs ns), lower write endurance, up to 4× higher power per bit |

**OPEN markers resolved:** HBF **capacity per stack = 384 GB** (was OPEN).
HBF **dies per stack remains OPEN** — the paper states capacity per stack, not
a die count — but capacity no longer depends on it, so no result in this phase
rests on an unsourced number. **PCM endurance cycles remain OPEN.**

**Scope — topology, not workload.** H3 targets read-only data (weights and
shared pre-computed KV in HBF) and argues that low write endurance is therefore
not a disadvantage. We take the platform and change the workload: general
write-heavy serving, where HBF *does* receive a transient write stream.
So model weights and shared cache are not the subject; **generated/transient KV
is**. HBM is the memory tier holding transient KV, HBF the offload tier
receiving spill, PCM the write absorber between them.

### The entry

Written to a scratch catalogue and selected with `--hardware_models`. **The
tracked catalogue was not mutated** — `git diff
TransformerRoofline/hardware_models.json` is empty.

```json
{"Name": "H3-B200-KVONLY", "Type": "Homo", "TFLOPS": 4500, "BW_TBs": 8.0,
 "Card_Num": 1, "Capacity": 252.3814697265625, "Static_Power": 0,
 "TDP": 1000, "Price": 500000, "pcie": "pcie5.0x16", "nvlink": "nvlink5x18"}
```

Kept at `TokenSim/data/hardware/hardware_models_h3b200.json` (untracked) and
regenerable with `data/hardware/make_h3b200.py`, which rebuilds it from the
tracked catalogue and prints the arithmetic below; the regenerated file is
byte-identical to the one these runs used. Cluster:
`TokenSim/data/clusters/8_b200_h3/h8_tp2dp4.json`.

`BW_TBs = 8.0` and `TFLOPS = 4500` are the stock `B200` values, matching the
paper's 8 TB/s HBM. **`Capacity` carries the same arithmetic workaround as the
existing `H3-KVONLY` entry**, because `cache_config.py:70-74` subtracts
`model_param_size` from HBM unconditionally and H3 puts the weights in HBF:

```
model_param_size_unsharded = (12 × Nlayer × Dmodel² + 50000 × Dmodel) × 2
                           = (12 × 80 × 8192² + 50000 × 8192) × 2
                           = 129,668,218,880 B = 120.762939 GiB
model_param_size (TP2/PP1) = 129,668,218,880 / 2
                           = 64,834,109,440 B = 60.3814697266 GiB

Capacity = 192 + 60.3814697266 = 252.3814697265625 GiB

check (through the real code path, CacheConfig on this entry):
  252.3814697265625 × 2^30 − 64,834,109,440 = 206,158,430,208 B = 192.0000000000 GiB
  num_gpu_blocks = 78,643   (78,643 × 2,621,440 B = 191.9995 GiB)
```

The subtraction still happens; it is pre-compensated. **This is a deviation from
a literal reading of the design and is flagged as such** — stated here because
the alternative is a code change to `cache_config.py`.

`TDP` is left at the stock B200 1000 W. The paper's per-cube TDPs (HBM 40 W,
HBF 160 W) are recorded in `CLAUDE.md` but the field feeds no result in this
phase; power is out of scope.

### Model and block size (recorded per `CLAUDE.md`)

| | |
|---|---|
| model | **LLaMa2-70B-GQA** |
| `Nhead` / KV heads (`Nhead / Grouped_Num`) | **64** / **8** |
| `Nlayer` / `Dmodel` / head_dim | **80** / **8192** / **128** |
| parallelism | **TP 2, PP 1, DP 4** across 8 workers |
| local KV heads per rank | 8 / 2 = **4** |
| **KV precision** | **FP16** — hardcoded in `cache_config.py`; the conservative choice, understating the architecture's benefit |
| **KV bytes per token** | `4 × 128 × 2 × 2 × 80` = **163,840 B** |
| **block size** | 16 tokens × 163,840 = **2,621,440 B (2.5 MiB)** |

### Block counts

| tier | bytes | arithmetic | **blocks** |
|---|---|---|---|
| **HBM** (per rank, GPU-local) | 192 GiB | `192 × 2^30 / 2,621,440 = 78,643.2` | **78,643** |
| **PCM** (cluster pool) | 347.2 GiB = 43.4 GiB/GPU × 8 | `347.2 × 2^30 / 2,621,440 = 142,213.12` | **142,213** |
| **HBF** (cluster pool) | 24 TiB = 8 stacks × 384 GiB × 8 GPUs | `24 × 2^40 / 2,621,440 = 10,066,329.6` | **10,066,329** |

`GB` is read as 2³⁰ throughout, per the unit note in `CLAUDE.md`.

### HBF timing — the LHB modelling assumption

The Mooncake offload tier is a per-operation latency function
(`TokenSim/mooncake/ssd.py:107-121`): `latency = fixed_us/1e6 + bytes/2^30 / bw`.
The LHB is not simulated as a buffer; its effect is folded into the read knobs.

| knob | prior phases | **this phase** | basis |
|---|---|---|---|
| `ssd_read_latency_us` | 100 | **0.1** | LHB prefetch hit, SRAM-class. **Modelling assumption sourced from the paper's argument, not a measured value** — a *perfect-prefetch upper bound*. |
| `ssd_read_bw_gbps` | 7.0 | **7451** | the paper's 8 TB/s HBF bandwidth. Knob is GiB/s: `7451 × 2^30 = 8.0005e12 B/s`. |
| `ssd_write_latency_us` | 200 | **200** (unchanged) | NAND-realistic. **Not** hidden — the LHB is a read-side prefetch buffer. |
| `ssd_write_bw_gbps` | 3.0 | **3.0** (unchanged) | NAND-realistic. Unchanged so the configs stay comparable across the topology change. |

#### CORRECTION (recorded after the fact) — what the 0.1 µs actually assumes

The reasoning above was stated too loosely. Precisely:

**The H3 paper does not claim HBF is low-latency.** It claims HBF latency is
**hidden by prefetching**. The LHB is a prefetch buffer, and it works because
**LLM inference has a deterministic, sequential access pattern** — layer N+1's
weights and shared KV cache are known while layer N is still computing, so the
LHB can have them in SRAM before they are asked for. The mechanism is
predictability, not speed.

Two consequences follow, and they are not symmetric:

1. **The LHB does nothing for writes.** A prefetch buffer works ahead of demand
   on the read path; there is no equivalent for a write that must reach the NAND
   array. So **reads at 0.1 µs against writes at 200 µs / 3.0 GB/s is
   structurally correct** and is retained.

2. **Prefetching hides latency only for *predictable* reads.** H3's read-only
   data — weights, shared pre-computed KV — qualifies: the access order is known
   in advance. **Our HBF reads do not.** They are demand-driven cache hits on
   *transient* KV, looked up when a request arrives and a prefix happens to
   match. There is no layer-ahead schedule from which to prefetch them. So
   **0.1 µs is more generous for our workload than the paper justifies for its
   own.**

**Status of the assumption:** paper-sourced, **applied more generously to our
access pattern than the paper justifies**, and **retained** — because it makes
HBF reads cheap, which is exactly what makes a write-through,
everything-on-NAND design (b) tolerable. It therefore **flatters baseline (b)
and is conservative for the PCM argument**: a higher, more realistic read
latency would penalise (b) and (c) (which serve reuse from HBF) far more than
(d) (which barely reads HBF at all — 21,728 blocks vs 331,177). **A sensitivity
point at higher HBF read latency is deferred, not overlooked.**

The same applies to `ssd_read_bw_gbps = 7451`: treating the LHB as also
delivering the paper's full 8 TB/s on demand-driven reads is the most generous
reading of the paper available to (b).

#### Further unmodelled elements of the H3 path

Recorded alongside the existing `ssd.py:107-121` caveat so the gaps are in one
place. The far tier is a pure per-operation latency function
`latency = fixed_us/1e6 + bytes/2^30 / bw`, and therefore:

- **No shared-bandwidth model.** All 8 workers resolve to one tier object with
  no SimPy resource and no contention between them; concurrent HBF traffic is
  free.
- **The D2D hop is not modelled at all.** H3 reaches HBF *through* the HBM base
  die over a die-to-die link. That hop has its own latency, bandwidth ceiling
  and energy; none of it appears in the model.
- **The HBM base-die address decoder and router are not modelled at all.** The
  component that splits traffic between the HBM and HBF paths costs nothing
  here, contends for nothing, and has no throughput limit.
- **The LHB itself is not modelled as a buffer.** It has no capacity (40 MB), no
  hit/miss behaviour and no eviction; its effect is folded wholesale into the
  read knobs, which is equivalent to assuming a 100% prefetch hit rate.

All four omissions make the HBF path *cheaper* than it would be, so all four
push in the same direction as the read-latency assumption: they flatter (b) and
(c), and are conservative for the PCM argument.

**None of them affects the endurance results.** The lifetime multipliers are
ratios of **work-normalized byte counts** — bytes written to HBF per request and
per token, under identical workloads. Interconnect timing changes *when* those
bytes are written and how long the run takes; it does not change *how many*
bytes are written, because the write stream is determined by the policy and the
pool capacity, not by the speed of the path. So the D2D hop, the base-die
router, the LHB's own capacity and the missing shared-bandwidth model are
caveats on the **latency and throughput** numbers only. The endurance
conclusions are independent of all of them.

(Phase 9 STEP 4 demonstrates this empirically from the other direction: driving
the system from 25% to 99% HBM occupancy — a 12× change in concurrency and a
large change in every timing figure — moves HBF write volume per request by
less than 0.5%.)

### Harness change

`MetricData` (`TokenSim/config/psla_config.py:12-32`) recorded only p50/p99/max.
A `mean` field was added (defaulted, so `MetricData.inf()` and the PSLA config
files are unaffected), because TTFT and TBT are quoted as means below and a
p50-only record cannot be compared against the HBF-endurance literature.

## STEP 2 — the three configurations

`H3-B200-KVONLY` × 8 workers, TP2/DP4, GQA, block_size 16,
`charge_eviction_writes=ON`, `memory_capacity_blocks=142213`,
`ssd_capacity_blocks=10066329`, **W = 5,000 requests** at 3 QPS
(`--trace_target_qps 3`), both Mooncake FAST'25 traces, separate
`--results_path` per run. Arrival-limited floor = 5,000 / 3 = **1,666.7 s**.

**Window justification carries over from Phase 7 STEP 1** — W = 5,000 is the
smallest window whose in-window reuse is within ~10% of the full-trace rate for
both traces (conversation 33.92% of 37.34% = 90.8%; Tool&Agent 51.87% of 57.01%
= 91.0%). Residual censoring 14.8% / 9.5% respectively.

**Conservation verified in all six runs:** `demoted + dropped == memory
evictions`, `ssd_write_bytes == ssd_write_blocks × 2,621,440`, `notdone = 0`,
`preemption_count = 0`.

**The far tier is no longer near-binding.** Phase 7's 97.8%-full warning is
gone: at 24 TiB the conversation trace peaks at **32.6%** occupancy and
Tool&Agent at **16.8%**; `mooncake_ssd_eviction_count = 0` in all six runs.

| | (b) H3 as-is | (c) H3 + PCM | (d) H3 + PCM |
|---|---|---|---|
| memory tier media | dram | **pcm** | **pcm** |
| admission write policy | `write_through` | **`write_back`** | **`write_back`** |
| demote policy | `always` | `always` | **`if_read`** |

### Conversation, W = 5,000

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **HBF write blocks** | 6,422,625 | 3,138,753 | **71,119** |
| **HBF write bytes** | 16,836,526,080,000 | 8,228,052,664,320 | **186,434,191,360** |
| HBF read blocks / bytes | 331,177 / 868,160,634,880 | 335,241 / 878,814,167,040 | 21,728 / 56,958,648,320 |
| **blocks demoted / dropped** | 3,140,206 / **0** | 3,138,753 / **0** | **71,119 / 3,588,022** |
| memory evictions | 3,140,206 | 3,138,753 | 3,659,141 |
| **evictions with read_count>0** | **2.389%** (75,006) | **2.288%** (71,807) | *1.944% (71,119) — tautological under `if_read`* |
| **mean PCM residency** | 109.62 s | 78.99 s | 62.31 s |
| **prefix cache hit rate** | **0.1640** | **0.1640** | **0.0965** |
| **reuse hit blocks** | **669,064** | **669,320** | **393,987** |
| **disk (HBF) hit tokens** | **5,571,728** | 5,660,304 | **347,648** |
| local GPU hit tokens | 5,372,160 | 5,286,144 | 5,966,384 |
| admissions | 3,282,419 | 3,280,966 | 3,801,354 |
| PCM-tier write blocks / bytes | 3,282,419 / 8,604,664,463,360 | 3,280,966 / 8,600,855,511,040 | 3,801,354 / 9,965,021,429,760 |
| **throughput** | 25,625 tok/s | 35,253 tok/s | **39,704 tok/s** |
| **e2e mean / p99** | 618.33 / 1112.65 s | 205.53 / 474.06 s | **17.43 / 66.77 s** |
| **TTFT mean** | 443.73 s | 91.62 s | **1.75 s** |
| **TBT mean / p99** | 685.47 / 4733.47 ms | 460.00 / 2926.30 ms | **65.75 / 454.32 ms** |
| **recomputation count** | **0** | **0** | **0** |
| duration | 2617.0 s | 1902.3 s | 1689.0 s |
| **floor ratio** | **1.57×** | **1.14×** | **1.01×** |
| far-tier occupancy | 32.6% | 32.6% | 37.8% |

### Tool&Agent, W = 5,000

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **HBF write blocks** | 3,235,553 | 1,546,087 | **37,451** |
| **HBF write bytes** | 8,481,808,056,320 | 4,052,974,305,280 | **98,175,549,440** |
| HBF read blocks / bytes | 117,754 / 308,685,045,760 | 118,234 / 309,943,336,960 | 10,464 / 27,430,748,160 |
| **blocks demoted / dropped** | 1,546,670 / **0** | 1,546,087 / **0** | **37,451 / 1,735,515** |
| memory evictions | 1,546,670 | 1,546,087 | 1,772,966 |
| **evictions with read_count>0** | **2.735%** (42,303) | **2.897%** (44,784) | *2.112% (37,451) — tautological* |
| **mean PCM residency** | 142.04 s | 141.80 s | 124.27 s |
| **prefix cache hit rate** | **0.4042** | **0.4036** | **0.3687** |
| **reuse hit blocks** | **1,175,585** | **1,174,017** | **1,072,511** |
| **disk (HBF) hit tokens** | **1,885,088** | 1,891,744 | **167,424** |
| local GPU hit tokens | 17,002,992 | 16,971,248 | 16,995,312 |
| admissions | 1,688,883 | 1,688,300 | 1,915,179 |
| PCM-tier write blocks / bytes | 1,688,883 / 4,427,305,451,520 | 1,688,300 / 4,425,777,152,000 | 1,915,179 / 5,020,526,837,760 |
| **throughput** | 27,824 tok/s | 28,051 tok/s | **28,100 tok/s** |
| **e2e mean / p99** | 32.44 / 232.99 s | 12.55 / 82.50 s | **6.21 / 33.67 s** |
| **TTFT mean** | 5.66 s | 2.23 s | **1.21 s** |
| **TBT mean / p99** | 1338.74 / 26059.16 ms | 412.37 / 6214.42 ms | **69.97 / 643.39 ms** |
| **recomputation count** | **0** | **0** | **0** |
| duration | 1707.1 s | 1693.3 s | 1690.3 s |
| **floor ratio** | **1.02×** | **1.02×** | **1.01×** |
| far-tier occupancy | 16.8% | 16.8% | 19.0% |

### Reading the two new metrics

**TBT.** Reported from `decode_time`, which `llm_request.py:86-90` defines as
`decode_time_sum / (steps − 1)` per request — i.e. it *is* mean time between
tokens, already per-request; the table gives the mean and p99 across requests.
TBT tracks the queueing regime almost exactly (conversation 685 → 460 → 66 ms
against floor ratios 1.57 → 1.14 → 1.01), so **it is a congestion readout here,
not a policy benefit**, and is subject to the same contamination caveat as e2e
and throughput.

**Recomputation is 0 in all six runs — and that is the wrong metric for what
it was added to measure.** `recomputation_count` counts *scheduler-preemption*
recomputation (`preemption_count = 0` everywhere, so it is structurally 0: at
192 GiB of KV per rank the HBM tier never forces a running request to give up
its blocks). The cost of `if_read` discarding a block whose reuse had not yet
arrived does not surface there — it surfaces as prompt tokens that must be
prefilled again instead of being loaded from cache. The metric that actually
measures it is **`effective_prefill_tokens`**:

| | (b) | (c) | **(d)** | **(d) vs (b)** |
|---|---|---|---|---|
| **Conversation** `effective_prefill_tokens` | 54,625,935 | 54,621,839 | **59,027,167** | **+4,401,232 (+8.06%)** |
| Conversation `reuse_hit_tokens` | 10,705,024 | 10,709,120 | 6,303,792 | −4,401,232 |
| **Tool&Agent** `effective_prefill_tokens` | 27,765,339 | 27,790,427 | **29,414,523** | **+1,649,184 (+5.94%)** |
| Tool&Agent `reuse_hit_tokens` | 18,809,360 | 18,784,272 | 17,160,176 | −1,649,184 |

Total prompt tokens are identical within each trace (65,330,959 / 46,574,699),
so the two rows trade exactly. **`if_read` costs 8.06% more prefill work on the
conversation trace and 5.94% more on Tool&Agent.** That is the direct price of
the 98.9% / 98.8% NAND write reduction, and it is a *real* cost, not a
queueing artifact — it is a token count, invariant to the regime.

### What the rebuild changes versus Phase 7

The write-side results are essentially unchanged; the latency-side results move
a lot, and in the direction the LHB predicts.

| | Phase 7 (H200, 8 TiB far tier, 100 µs / 7 GB/s reads) | **Phase 8 (H3-B200, 24 TiB, LHB reads)** |
|---|---|---|
| Conversation, HBF write bytes (b) | 16,831,440,486,400 | 16,836,526,080,000 (+0.03%) |
| Conversation, (b)→(d) write reduction | 98.93% | **98.89%** |
| Conversation, prefix hit (b) → (d) | 0.1640 → 0.0931 | **0.1640 → 0.0965** |
| Conversation, (b) floor ratio | 2.12× | **1.57×** |
| Conversation, (b) e2e p99 | 1969.1 s | **1112.65 s** |
| Tool&Agent, (b)→(d) write reduction | 98.82% | **98.84%** |
| Tool&Agent, (b) floor ratio | 1.12× | **1.02×** |
| far-tier occupancy, worst case | **97.8%** (near-binding) | **32.6%** |

Two things to note. First, **the write-volume claim is topology-independent** —
it moved by 0.03-0.04%, because it is a function of the policy and the PCM pool
size, neither of which changed. Second, **cheap HBF reads relieve most of the
queueing that contaminated Phase 7's latency numbers**: Tool&Agent is now
arrival-limited in all three configs (1.02× / 1.02× / 1.01×) and the
conversation trace is much closer (1.57× / 1.14× / 1.01×). The conversation
(b) run is still saturated, so its latency and throughput gaps are still partly
regime-driven and should not be quoted as pure policy benefit.

### The trade, restated on the correct hardware

| | write reduction (b)→(d) | prefix hit rate (b)→(d) | extra prefill work |
|---|---|---|---|
| **Conversation** | **−98.89%** | 0.1640 → 0.0965 (**−41.2%**) | **+8.06%** |
| **Tool&Agent** | **−98.84%** | 0.4042 → 0.3687 (**−8.8%**) | **+5.94%** |

Phase 7's conclusion survives the rebuild intact: **`if_read` is clearly
attractive on Tool&Agent and a genuine design trade-off on the conversation
trace**, and **(c) remains interesting in its own right** — it halves HBF writes
(16.84 → 8.23 TB, **−51.13%**; Tool&Agent 8.48 → 4.05 TB, **−52.22%**) at
**zero** hit-rate cost (conversation 0.1640 vs 0.1640, reuse hits 669,320 vs
669,064; Tool&Agent 0.4036 vs 0.4042, 1,174,017 vs 1,175,585 — a −0.13%
difference). The simulator is deterministic (`--random_seed 0`), so that −0.13%
is not noise: `write_back` removes the admission-time offload write, which
shortens save waits, which shifts scheduling order and hence which blocks are
resident at each lookup. It is a second-order timing effect, not a property of
the policy — it appears with the opposite sign on the conversation trace
(669,320 vs 669,064, **+0.04%**).

**Re-admission churn still shifts wear onto PCM**, unchanged in character from
Phase 7: (d) admits more than (b) and so writes more to the PCM tier —
conversation 8.60 → 9.97 TB (**+15.8%**), Tool&Agent 4.43 → 5.02 TB
(**+13.4%**). **PCM endurance is OPEN**, so this remains an unpriced cost.

## STEP 3 — work-normalized HBF lifetime, (b) → (d)

Generated tokens are **identical** within each trace across all three configs
(conversation 67,061,296; Tool&Agent 47,497,243; ratio 1.00000000), so the
per-request and per-token multipliers coincide exactly.

| | Conversation | Tool&Agent |
|---|---|---|
| requests / generated tokens | 5,000 / 67,061,296 | 5,000 / 47,497,243 |
| HBF write bytes (b) | 16,836,526,080,000 | 8,481,808,056,320 |
| HBF write bytes (d) | 186,434,191,360 | 98,175,549,440 |
| per request (b) → (d) | 3,367,305,216 → 37,286,838 B | 1,696,361,611 → 19,635,110 B |
| per token (b) → (d) | 251,061.7 → 2,780.1 B | 178,574.7 → 2,067.0 B |
| **multiplier, per request** | **×90.31** | **×86.39** |
| **multiplier, per token** | **×90.31** | **×86.39** |
| (c) as an intermediate point | ×2.05 | ×2.09 |
| vs Phase 7 (8 TiB far tier, H200) | ×93.17 → **×90.31** (−3.1%) | ×85.01 → **×86.39** (+1.6%) |

Both multipliers hold to within a few percent of Phase 7. They are ratios of
write volumes under identical workloads, so the topology change — which altered
read cost, not write cost — barely touches them.

Endurance budget, stated as a ratio only: the 24 TiB cluster HBF at TLC **3,000
P/E cycles** absorbs `24 × 2^40 × 3000 = 79,164,837,199,872,000 B` of writes.
At the measured per-request rates that is **23.5 M requests under (b)** versus
**2.12 B requests under (d)** on the conversation trace (**46.7 M** vs
**4.03 B** on Tool&Agent) — the same ×90.31 / ×86.39, expressed as work.
This assumes WAF = 1 and ideal wear levelling.

> **Absolute day counts are still not quotable.** Unchanged from Phases 6 and 7:
> the far tier is a pure per-operation latency function
> (`TokenSim/mooncake/ssd.py:107-121`) — **no shared-bandwidth model**, no SimPy
> resource, no contention among the 8 workers, and no NAND queueing/GC/WAF
> coupling. The multipliers are ratios of write volumes under identical
> workloads and survive that gap. Day counts do not. PCM endurance is OPEN.

Stopping here per instruction: no NVSim, no thermal, no batch-size sweep, no
three-tier work.

---

# PHASE 9 — Batch size as a confound, and the lifetime-vs-batch curve

Phase 8 left the three configs running at **different concurrent batch sizes**,
which contaminates every comparison between them. This phase identifies the
control, measures the confound, reruns at matched batch, and sweeps batch to
produce a deployment rule.

## STEP 1 — how TokenSim determines concurrent batch size

**There is a direct cap, and it is already on the CLI. No code change is needed
to pin batch size.**

All runs use `--batching paged-attn` → `LLMPagedAttnScheduler`
(`TokenSim/llm/llm_engine.py:244-250`). Its admission loop
(`TokenSim/llm/llm_scheduler.py:194-214`) breaks on four conditions, in order:

| # | gate | line | what it limits |
|---|---|---|---|
| 1 | `not self._is_occupy_below_usage()` | `llm_scheduler.py:195-196` | **HBM occupancy watermark.** `_is_occupy_below_usage` (`llm_scheduler.py:315-317`) is `used_blocks / all_blocks < max_occupy_ratio`. Knob: **`--max_occupy_ratio`**, default **1.0** (no watermark). |
| 2 | `len(self.running) >= self.max_parallem_sum` | `llm_scheduler.py:198-201` | **A hard cap on concurrent requests** — the direct equivalent of vLLM's `max_num_seqs`. Knob: **`--max_parallem_sum`**, default **99999** (`benchmark.py:291`), i.e. effectively unlimited. |
| 3 | `req_phase != admission_phase` | `llm_scheduler.py:210-211` | phase purity — a batch is all-prefill, all-recompute or all-decode; not a size limit. |
| 4 | `not self.block_manager.can_allocate(req)` | `llm_scheduler.py:213-214` | **capacity-derived**: the request's blocks must fit in the rank's HBM. |

Below that, the decode path (`llm_scheduler.py:250-270`) admits token slots via
`can_append_slot` and **preempts** the lowest-priority request when HBM cannot
grow, which is the other capacity-derived limit.

**So batch size is *not* purely capacity-derived.** With the shipped defaults
(`max_parallem_sum=99999`, `max_occupy_ratio=1.0`) gates 1 and 2 never fire and
batch size *is* purely capacity-derived — which is what every run up to and
including Phase 8 did. But gate 2 is a real, plumbed, per-scheduler
`max_num_seqs`.

**Three traps worth recording:**

1. **`max_parallem_sum` means two different things.** In `LLMPagedAttnScheduler`
   it is compared against `len(self.running)` — a **request count**. In
   `LLMDynamicScheduler` (`:94`), `LLMStaticScheduler` (`:119`, `:126`) and
   `LLMPrefillScheduler` (`:142`) the same CLI flag is compared against
   `sum(req.prefill_len for req in running)` — a **token sum**. The flag's
   meaning depends on `--batching`.
2. **A dead token-sum break survives in the paged-attn scheduler**
   (`llm_scheduler.py:238-239`, commented out). It is *not* the active gate.
3. **The cap is per scheduler, not per cluster.** Each data-parallel group has
   its own `LLMPagedAttnScheduler`; this configuration has **8 schedulers**, so
   cluster concurrency is `8 × max_parallem_sum`.

**Cleanest way to pin batch: `--max_parallem_sum N`.** It is exact, per-request,
needs no code change, and is checked before allocation so it binds regardless of
capacity. `--max_occupy_ratio` is the weaker alternative (it bounds HBM bytes,
not request count, so achieved batch still floats with prompt length). Capping
QPS below saturation is the *worst* option for this purpose — it controls
arrivals, not concurrency, and by Little's law the achieved batch then depends
on the service time, which is exactly the thing that differs between configs.

**One limitation, which shapes STEP 3.** `max_parallem_sum` is an **upper
bound**: it can lower a config's batch but cannot raise it. At the Phase 8
offered load of 3 QPS, config (d) is arrival-limited and runs a batch of ~6 per
scheduler, so no cap can make (b) and (d) meet at a batch that keeps HBM
meaningfully occupied. Matching them therefore also requires raising the offered
load until all three configs are cap-bound rather than arrival-bound.

## STEP 2 — the confound, confirmed (and running the opposite way)

Measured on the Phase 8 W = 5,000 runs, re-executed with new instrumentation
(see *Harness change* below). **The re-runs reproduce Phase 8 exactly** — HBF
write blocks identical to the digit in all six runs — so these are the same
runs, now observed.

Batch and occupancy are **time-weighted** integrals over simulated time, not
per-step averages. 8 schedulers, 78,643 HBM blocks per rank.

### Conversation

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **floor ratio** | 1.57× | 1.14× | 1.01× |
| **mean batch / scheduler** | **42.2** | **38.1** | **6.3** |
| peak batch / scheduler | 139 | 117 | 49 |
| mean batch, cluster | 337.5 | 304.6 | 50.3 |
| **mean HBM occupancy** | **45.35%** | **41.67%** | **7.98%** |
| **peak HBM occupancy** | **99.21%** | **99.30%** | **68.32%** |
| peak used blocks / rank | 78,023 | 78,090 | 53,732 |

### Tool&Agent

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **floor ratio** | 1.02× | 1.02× | 1.01× |
| **mean batch / scheduler** | **10.4** | **4.3** | **2.3** |
| peak batch / scheduler | 110 | 94 | 27 |
| mean batch, cluster | 83.1 | 34.2 | 18.1 |
| **mean HBM occupancy** | **11.20%** | **5.05%** | **2.82%** |
| **peak HBM occupancy** | **99.02%** | **66.52%** | **40.65%** |

**The confound is confirmed, and it is large** — on the conversation trace mean
batch differs by **6.7×** between (b) and (d), and mean HBM occupancy by
**5.7×**. The floor ratios 1.57 / 1.14 / 1.01 track it exactly. No comparison
between these three columns is a clean policy comparison.

**But the hypothesised direction is refuted.** The suspicion was that the
scheduler consumed the added PCM capacity with a larger batch rather than using
it as a write absorber. The data says the reverse: **the configs with PCM run
the *smaller* batches.** (d) — the most write-relieved config — has the smallest
batch of the three, on both traces.

The mechanism is the other way round, and it is a feedback loop:

```
(b) stalls on far-tier writes  (save_wait 6,672 s vs 726 s in (d))
  -> each request occupies HBM for longer
  -> more requests are concurrently resident      [batch 42.2 vs 6.3]
  -> HBM fills                                    [peak 99.21% vs 68.32%]
  -> more pressure on the Mooncake pool, more evictions
  -> more far-tier writes
  -> (b) stalls harder
```

So batch size is **downstream of the write stalls, and then feeds back into
them**. This matters for how Phase 8 should be read: (b)'s write volume is
inflated not only by its write-through policy but also by the larger batch that
policy induces. The matched-batch rerun below separates the two.

It also means peak HBM occupancy of **99.2% / 99.3% / 99.0%** in (b)/(c)
conversation and (b) Tool&Agent is a real finding in its own right: **with the
shipped defaults the scheduler runs HBM to saturation**, and gate 4
(`can_allocate`) is what stops it. `preemption_count = 0` throughout, so it
stops just short of thrashing — but there is no headroom.

## STEP 3 — matched-batch rerun

### Choosing the operating point

Pinned with **`--max_parallem_sum 48`** (per scheduler; 8 schedulers → 384
cluster-wide), at an **offered load of 30 QPS**.

Why 48, from a calibration sweep at W = 1,000:

| cap / scheduler | achieved mean batch (b / d) | HBM mean | **HBM peak** |
|---|---|---|---|
| 32 | 14.61 / 13.54 | 17.4% / 16.1% | 56.8% / 59.0% |
| **48** | **21.67 / 19.40** | **24.8% / 22.9%** | **77.6% / 82.6%** |
| 64 | 28.48 / 26.00 | 31.7% / 29.9% | **96.0% / 98.8%** — saturated |

48 is the largest cap that keeps peak HBM occupancy clear of the ceiling. At 64
HBM peaks at 96–99%, which is the Phase 8 failure mode reproduced on purpose;
at 32 HBM barely exceeds half. 48 keeps HBM **meaningfully occupied** (peak
75–98% in the real W = 5,000 runs, mean ~25%) **without saturating**.

**Why the offered load had to rise from 3 to 30 QPS.** Per STEP 1,
`max_parallem_sum` is an upper bound. At 3 QPS config (d) is arrival-limited at
a batch of ~6 per scheduler (Little's law: 2.96 req/s × 17.4 s ≈ 51 cluster-wide
≈ the measured 50.3), so no cap can lift it to 48. 30 QPS is the smallest
offered load at which **all three configs reach the cap**; a check at 60 QPS
gave no tighter match (21.12 vs 19.04, versus 21.67 vs 19.40 at 30 QPS), so
the residual gap is structural, not arrival-driven.

**This is a deviation from the Phase 8 protocol and from the trace's own
pacing** — 3 QPS is the trace's natural rate. It is required by the experiment:
matched batch at a meaningful HBM occupancy is unreachable at 3 QPS. Everything
in this STEP is therefore a *saturated* measurement and its latency numbers are
not comparable to Phase 8's.

### Achieved batch — confirmed equal

| | (b) | (c) | (d) | spread |
|---|---|---|---|---|
| **Conversation**, mean batch / scheduler | **23.17** | **23.08** | **22.38** | **3.4%** |
| Conversation, peak batch / scheduler | 48 | 48 | 48 | cap reached |
| Conversation, HBM mean / peak | 26.43% / 85.66% | 26.38% / 97.90% | 25.39% / 90.54% | |
| **Tool&Agent**, mean batch / scheduler | **22.89** | **22.64** | **22.43** | **2.0%** |
| Tool&Agent, peak batch / scheduler | 48 | 48 | 48 | cap reached |
| Tool&Agent, HBM mean / peak | 25.41% / 75.12% | 25.38% / 82.28% | 25.12% / 79.49% | |

**Matched.** Against Phase 8's 6.7× spread on the conversation trace, the
configs now sit within **3.4%** (conversation) and **2.0%** (Tool&Agent) of each
other, with all three reaching the cap exactly. The small residual is
structural — the faster configs drain the running set slightly more often
between admissions — and cannot be removed by an upper-bound cap.

Conservation verified in all six runs: `demoted + dropped == memory evictions`,
byte exactness, `ssd_eviction_count = 0`, `notdone = 0`, `preemption_count = 0`,
`recomputation_count = 0`.

### Conversation, matched batch

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **achieved batch / scheduler** | **23.17** | **23.08** | **22.38** |
| **HBF write blocks** | 6,429,053 | 3,143,664 | **79,921** |
| **HBF write bytes** | 16,853,376,696,320 | 8,240,926,556,160 | **209,508,106,240** |
| **blocks demoted / dropped** | 3,143,420 / 0 | 3,143,664 / 0 | **79,921 / 3,568,008** |
| memory evictions | 3,143,420 | 3,143,664 | 3,647,929 |
| **mean PCM residency** | 113.34 s | 81.22 s | 42.16 s |
| **prefix cache hit rate** | **0.1643** | **0.1645** | **0.0974** |
| **reuse hit blocks** | **670,585** | **671,266** | **397,596** |
| **effective_prefill_tokens** | **54,601,599** | 54,590,703 | **58,969,423** (+8.00%) |
| **throughput** | 25,049 tok/s | 34,796 tok/s | **56,089 tok/s** |
| **e2e mean / p99** | 1356.97 / 2473.98 s | 937.37 / 1725.77 s | **526.95 / 992.41 s** |
| **TTFT mean** | 1258.89 s | 867.10 s | **485.04 s** |
| **TBT mean / p99** | 344.95 / 1623.65 ms | 227.62 / 829.12 ms | **129.24 / 356.62 ms** |
| duration | 2677.2 s | 1927.3 s | 1195.6 s |
| **floor ratio** | **16.06×** | **11.56×** | **7.17×** |
| far-tier save_wait | 6678.3 s | 3751.9 s | 732.6 s |

### Tool&Agent, matched batch

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **achieved batch / scheduler** | **22.89** | **22.64** | **22.43** |
| **HBF write blocks** | 3,236,365 | 1,547,651 | **32,747** |
| **HBF write bytes** | 8,483,936,665,600 | 4,057,074,237,440 | **85,844,295,680** |
| **blocks demoted / dropped** | 1,547,076 / 0 | 1,547,651 / 0 | **32,747 / 1,736,074** |
| memory evictions | 1,547,076 | 1,547,651 | 1,768,821 |
| **mean PCM residency** | 113.02 s | 80.80 s | 42.71 s |
| **prefix cache hit rate** | **0.4037** | **0.4040** | **0.3650** |
| **reuse hit blocks** | **1,174,042** | **1,174,945** | **1,061,727** |
| **effective_prefill_tokens** | **27,790,027** | 27,775,579 | **29,587,067** (+6.47%) |
| **throughput** | 34,681 tok/s | 47,973 tok/s | **78,029 tok/s** |
| **e2e mean / p99** | 627.55 / 1190.53 s | 412.45 / 804.45 s | **220.58 / 432.29 s** |
| **TTFT mean** | 577.96 s | 377.06 s | **199.21 s** |
| **TBT mean / p99** | 351.99 / 3013.95 ms | 231.01 / 1508.54 ms | **146.45 / 755.04 ms** |
| duration | 1369.6 s | 990.1 s | 608.7 s |
| **floor ratio** | **8.22×** | **5.94×** | **3.65×** |
| far-tier save_wait | 3363.7 s | 1859.6 s | 361.8 s |

### The headline survives batch matching almost unchanged

This is the point of the exercise, so it is worth stating plainly.

| | Phase 8 (batch unmatched, 6.7× spread) | **Phase 9 (batch matched, 3.4% spread)** |
|---|---|---|
| Conversation HBF bytes (b) | 16,836,526,080,000 | 16,853,376,696,320 (**+0.10%**) |
| Conversation HBF bytes (c) | 8,228,052,664,320 | 8,240,926,556,160 (**+0.16%**) |
| Conversation HBF bytes (d) | 186,434,191,360 | 209,508,106,240 (**+12.4%**) |
| Conversation (b)→(d) reduction | 98.89% | **98.76%** |
| Conversation (b)→(c) reduction | 51.13% | **51.10%** |
| Conversation prefix hit (b) → (d) | 0.1640 → 0.0965 | **0.1643 → 0.0974** |
| Conversation extra prefill under (d) | +8.06% | **+8.00%** |
| Tool&Agent (b)→(d) reduction | 98.84% | **98.99%** |
| Tool&Agent prefix hit (b) → (d) | 0.4042 → 0.3687 | **0.4037 → 0.3650** |
| Tool&Agent extra prefill under (d) | +5.94% | **+6.47%** |

**The confound was real but not distorting.** Every write-volume and hit-rate
conclusion in Phase 8 holds at matched batch, most of them to within a few
tenths of a percent. The one visibly-moved number is (d)'s absolute HBF write
volume (+12.4% on the conversation trace), which is the small demote-only
stream and therefore the most sensitive to scheduling order; it moves the
reduction figure from 98.89% to 98.76%. **Phase 8's headline stands.**

### Floor ratios did NOT converge — and the reason is not batch

| | (b) | (c) | (d) | (b)/(d) |
|---|---|---|---|---|
| Phase 8, 3 QPS, unmatched | 1.57× | 1.14× | 1.01× | 1.55× |
| **Phase 9, 30 QPS, matched** | **16.06×** | **11.56×** | **7.17×** | **2.24×** |

They **diverged**. The expectation that matching batch would bring them together
was wrong, and the reason is that the Phase 8 floor ratios were compressed by an
artifact, not by similarity: at 3 QPS, (d) was **arrival-limited** (1.01×), so
its duration was pinned at the arrival floor no matter how fast it could have
gone. That floor masked the real throughput gap. Raising the offered load to
30 QPS removes the mask and all three configs become throughput-limited, so the
true gap is exposed.

**What else is varying, now that batch is not:** the time each config spends
stalled on far-tier writes. `save_wait` tracks duration closely across all six
runs — conversation **6678 / 3752 / 733 s** against durations **2677 / 1927 /
1196 s**, Tool&Agent **3364 / 1860 / 362 s** against **1370 / 990 / 609 s**.
Ranking, ordering and rough proportionality all match. This is the genuine,
un-confounded throughput cost of the NAND write stream: with concurrency held
equal, the config that writes 80× less to HBF finishes the same work **2.24×**
faster on the conversation trace and **2.25×** faster on Tool&Agent.

### PCM residency against time-to-first-read — the capacity is still short

Time-to-first-read measured in-simulator on the same clock as
`mooncake_mean_residency_s`, by monkeypatching `MooncakeStore.lookup` to record
`now − write_time` at each object's *first* read. Measured under config (c),
which retains every block and so gives an **uncensored** distribution — (d)
drops unread blocks and would censor its own measurement.

| trace | first reads observed | **TTFR median** | TTFR mean | TTFR p90 |
|---|---|---|---|---|
| Conversation | 379,552 | **202.51 s** | 316.60 s | 748.59 s |
| Tool&Agent | 198,911 | **277.56 s** | 366.05 s | 767.77 s |

| | mean PCM residency | **vs TTFR median** |
|---|---|---|
| **Conversation** (b) | 113.34 s | **0.56×** |
| Conversation (c) | 81.22 s | **0.40×** |
| Conversation (d) | 42.16 s | **0.21×** |
| **Tool&Agent** (b) | 113.02 s | **0.41×** |
| Tool&Agent (c) | 80.80 s | **0.29×** |
| Tool&Agent (d) | 42.71 s | **0.15×** |

**Residency did not rise with batch pinned — and the expectation that it would
was based on the wrong model.** Residency in an LRU pool is set by capacity
divided by admission rate, not by concurrency:

```
conversation (b): 142,213 blocks / (3,282,419 admissions / 2,677.2 s)
                = 142,213 / 1,226 blk·s⁻¹ = 116.0 s      (measured 113.34 s)
```

Pinning batch does not change the pool size and barely changes total admissions,
so it cannot raise residency. What *does* move residency is **duration**: the
Tool&Agent (b) run is 1,369.6 s here against 1,707.1 s in Phase 8, a ratio of
0.80, and its residency moves 142.04 → 113.02 s, a ratio of **0.80** — exactly
proportional. The faster a config runs, the faster it churns the pool, so **the
most write-relieved config has the shortest residency**, not the longest.

**The substantive result: at 43.4 GiB per GPU the PCM tier is still too small to
hold a typical block until its first read.** Mean residency is below the *median*
TTFR in all six matched runs — at best 0.56×, at worst 0.15×. The tier is doing
its job as a **write absorber** (that is what the 51% and 99% write reductions
are), but it is **not** functioning as a reuse cache: the majority of blocks
leave PCM before anyone reads them. This is a capacity statement about the 2
dies × 35 mm² × 0.62 Gb/mm² PCM budget, and it is independent of the batch
confound.

---

## STEP 3/4 PREREQUISITE — HBM eviction does not feed the Mooncake store

Both STEP 3 ("HBM must saturate, or nothing evicts, nothing reaches PCM or HBF,
and there is no write stream to absorb") and STEP 4 ("points below the ceiling
should show near-zero HBF writes — that is the read-only regime") rest on a
model of the simulator in which **HBM is the top of the Mooncake hierarchy and
spills into PCM, which spills into HBF**. That is the architecture
`CLAUDE.md` describes. **It is not what TokenSim implements.**

### What TokenSim actually implements

HBM (the `BlockManager`'s GPU blocks) and the Mooncake store are **two parallel
structures, not two levels of one hierarchy.**

**The only writer into the Mooncake store is the save path:**

```
LLMPagedAttnScheduler.schedule()            admits a request for prefill
  -> schedule_output.scheduled
  -> MooncakeStoreConnector.build_connector_meta()   mooncake_store.py:119-153
       builds `saves` from `scheduler_output.scheduled`, once per request
       (`self._saved_request_ids.add(req.id)`, :132-135)
  -> MooncakeStoreConnector.wait_for_save()          mooncake_store.py:186-193
  -> MooncakeStore.put_with_timing()                 store.py:168
```

**HBM eviction writes nothing.** The two paths that release GPU blocks —
`_preempt` (`llm_scheduler.py:319-325`) and `_release_delayed_blocks`
(`llm_scheduler.py:365-367`) — both call only
`block_manager.release_request_blocks(req)`. Neither touches the store.
`handle_preemptions` (`mooncake_store.py:270-274`) only *forgets* connector
state. `grep` for store writes finds exactly one call site
(`mooncake_store.py:191`), and it is the save path above.

### The consequences

1. **Blocks enter the Mooncake pool once per request at prefill admission**,
   regardless of HBM pressure. Total admissions are therefore a **workload**
   property — the distinct prompt-block footprint of the 5,000 requests —
   ~3.29 M blocks (conversation) and ~1.69 M (Tool&Agent), essentially constant
   across every batch size tested.
2. **HBF writes are Mooncake *memory-tier* evictions**, driven by the 142,213-block
   PCM pool overflowing against that ~3.29 M-block footprint — a 23× oversubscription
   that exists at *any* batch size.
3. **HBM occupancy therefore has almost no effect on the HBF write stream.**

### Measured confirmation

Across a **12× range of batch size**, spanning HBM peak occupancy from 25% to
99.1% (i.e. from far below the ceiling to fully saturated):

| | HBF bytes per request, (b) | spread |
|---|---|---|
| **Conversation**, caps 8 → 96 | 3,377,115,693 → 3,385,003,082 | **0.44%** |
| **Tool&Agent**, caps 8 → 96 | 1,696,587,055 → 1,678,879,752 | **1.05%** |

So the prerequisite does not hold, in either direction:

- **STEP 3's premise is unnecessary.** The spill regime is *already* active at
  every batch size tested — there is a full write stream to absorb at cap 8 with
  HBM at 25%. The runs are not measuring nothing. (The above-ceiling runs were
  done anyway and are reported below.)
- **STEP 4's premise is false.** There is **no read-only regime below the HBM
  ceiling**, so there is no threshold N to anchor the curve on, and no X → Y for
  PCM to move. This is reported as measured, not worked around.

### Is this a simulator gap or the right model?

It is a **simulator gap relative to `CLAUDE.md`'s architecture**, and it should
be recorded as one. In the intended design, transient KV evicted from HBM is
what lands in PCM and then HBF; in TokenSim, what lands in the Mooncake store is
a *copy* of each request's prompt blocks taken at prefill, and HBM eviction
discards its blocks silently. The two coincide in write *volume* only because
every request's prompt blocks get saved exactly once either way.

**What this does and does not invalidate:**

- **It does not invalidate the endurance results.** (b)/(c)/(d) differ only in
  what the *Mooncake store* does with a block after admission —
  `write_through` vs `write_back`, `always` vs `if_read`. That comparison is
  internal to the store and is unaffected by how blocks got there. The 51% and
  99% write reductions, and the ×90/×86 lifetime multipliers, stand.
- **It does invalidate any claim that links HBF wear to batch size, HBM
  capacity, or concurrency.** In this simulator there is no such link. A
  deployment rule of the form "below batch N, HBF stays read-only" **cannot be
  derived from TokenSim as it stands.**
- **Making that rule derivable requires a code change**: HBM eviction would have
  to write the victim into the Mooncake store (a demote path from
  `_release_delayed_blocks` / `_preempt` into `MooncakeStore.put`), and the save
  path would have to stop writing the full prompt at prefill. That is a
  substantive change to the KV-transfer model, not a knob, and it is **not made
  here** — flagged per the working rule on simulator constraints.

## STEP 3 (revised) — matched batch ABOVE the HBM ceiling, HBM saturated

The cap-48 runs above kept HBM below saturation. This section reruns with the
batch pinned **well above** the HBM ceiling so HBM fills, per instruction.

### The HBM ceiling

192 GiB per rank = **78,643 blocks** at the GQA block size (2,621,440 B, 16
tokens). Per-request footprint from the two traces:

| | prompt tokens (mean / median) | decode tokens (mean / median) | blocks/req at end of decode | **analytic ceiling** |
|---|---|---|---|---|
| **Conversation** | 13,066 / 7,748 | 346 / 356 | 838.7 mean, 504 median | **93.8 concurrent req/rank** |
| **Tool&Agent** | 9,315 / 6,397 | 185 / 30 | 594.2 mean, 404 median | **132.4 concurrent req/rank** |

The analytic ceiling divides HBM by the *mean* footprint. The **empirical**
ceiling is lower in mean-batch terms, because request lengths are heavily skewed
(median footprint is ~60% of the mean) and HBM saturates on the peaks: the
system reaches 99% peak HBM occupancy at a *mean* batch of ~43.6 (conversation)
and ~48.4 (Tool&Agent), with instantaneous peaks of 127-256.

### The chosen batch: `--max_parallem_sum 256`

**256 per scheduler is 2.7× the conversation analytic ceiling and 1.9× the
Tool&Agent one** — far enough above that the cap stops binding and HBM capacity
becomes the sole limit. Confirmed: caps of **192 and 256 give byte-identical
results** (conversation (b): 16,938,274,652,160 B at both), so the cap is
inactive and every config sits at its own capacity-determined equilibrium.

This is the correct way to equalise eviction pressure: rather than imposing the
same *cap*, it removes the cap and lets all three configs run against the same
*capacity*, which is the shared constraint.

### Achieved batch and HBM saturation — confirmed

| | (b) | (c) | (d) | spread |
|---|---|---|---|---|
| **Conversation** mean batch / scheduler | **43.61** | **42.90** | **41.55** | **4.7%** |
| Conversation peak batch / scheduler | 127 | 129 | 129 | |
| Conversation mean HBM occupancy | 46.90% | 46.13% | 44.86% | |
| **Conversation peak HBM occupancy** | **99.19%** | **99.17%** | **99.15%** | **saturated** |
| **Tool&Agent** mean batch / scheduler | **48.44** | **47.32** | **44.72** | **7.7%** |
| Tool&Agent peak batch / scheduler | 243 | 256 | 238 | |
| Tool&Agent mean HBM occupancy | 45.68% | 45.20% | 43.01% | |
| **Tool&Agent peak HBM occupancy** | **99.17%** | **99.09%** | **99.22%** | **saturated** |

**HBM is saturated in all six runs** (99.09-99.22% peak) and the achieved batch
is matched within **4.7%** / **7.7%**. `preemption_count = 0` and
`recomputation_count = 0` throughout — the scheduler runs HBM right up to the
ceiling without thrashing.

### Conversation, above ceiling

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **achieved batch / scheduler** | **43.61** | **42.90** | **41.55** |
| **peak HBM occupancy** | 99.19% | 99.17% | 99.15% |
| **HBF write blocks** | 6,461,439 | 3,150,432 | **80,968** |
| **HBF write bytes** | 16,938,274,652,160 | 8,258,668,462,080 | **212,252,753,920** |
| **blocks demoted / dropped** | 3,159,613 / 0 | 3,150,432 / 0 | **80,968 / 3,563,313** |
| memory evictions | 3,159,613 | 3,150,432 | 3,644,281 |
| **mean PCM residency** | 109.17 s | 77.36 s | 38.85 s |
| **prefix cache hit rate** | **0.1631** | **0.1641** | **0.0942** |
| **reuse hit blocks** | **665,419** | **669,864** | **384,393** |
| **effective_prefill_tokens** | 54,684,255 | 54,613,135 | **59,180,671** (+8.22%) |
| **throughput** | 25,698 tok/s | 35,668 tok/s | **59,766 tok/s** |
| **e2e mean / p99** | 1379.04 / 2510.14 s | 938.95 / 1687.41 s | **511.66 / 925.39 s** |
| **TTFT mean** | 1200.39 s | 813.04 s | **440.24 s** |
| **TBT mean / p99** | 766.28 / 5053.96 ms | 498.82 / 3267.20 ms | **237.72 / 700.08 ms** |
| duration | 2609.6 s | 1880.2 s | 1122.1 s |
| **floor ratio** | **15.66×** | **11.28×** | **6.73×** |
| far-tier save_wait | 6712.2 s | 3760.0 s | 733.1 s |

### Tool&Agent, above ceiling

| | (b) wt/always | (c) +PCM wb/always | (d) +PCM wb/if_read |
|---|---|---|---|
| **achieved batch / scheduler** | **48.44** | **47.32** | **44.72** |
| **peak HBM occupancy** | 99.17% | 99.09% | 99.22% |
| **HBF write blocks** | 3,234,691 | 1,534,945 | **55,703** |
| **HBF write bytes** | 8,479,548,375,040 | 4,023,766,220,800 | **146,022,072,320** |
| **blocks demoted / dropped** | 1,546,239 / 0 | 1,534,945 / 0 | **55,703 / 1,694,505** |
| memory evictions | 1,546,239 | 1,534,945 | 1,750,208 |
| **mean PCM residency** | 107.48 s | 76.27 s | 39.09 s |
| **prefix cache hit rate** | **0.4014** | **0.4033** | **0.3733** |
| **reuse hit blocks** | **1,167,351** | **1,173,139** | **1,085,863** |
| **effective_prefill_tokens** | 27,897,083 | 27,804,475 | **29,200,891** (+4.67%) |
| **throughput** | 35,573 tok/s | 50,515 tok/s | **82,693 tok/s** |
| **e2e mean / p99** | 656.06 / 1273.99 s | 414.32 / 769.70 s | **208.44 / 404.20 s** |
| **TTFT mean** | 557.61 s | 346.36 s | **170.52 s** |
| **TBT mean / p99** | 2871.65 / 60067.35 ms | 1855.74 / 42193.99 ms | **749.21 / 21567.68 ms** |
| duration | 1335.2 s | 940.3 s | 574.4 s |
| **floor ratio** | **8.01×** | **5.64×** | **3.45×** |
| far-tier save_wait | 3362.2 s | 1844.6 s | 381.9 s |

**Write reductions at saturation:** conversation **(b)→(c) 51.24%**, **(b)→(d)
98.75%**; Tool&Agent **(b)→(c) 52.55%**, **(b)→(d) 98.28%**. These are within
half a point of both the Phase 8 (unmatched) and the cap-48 (matched,
unsaturated) figures. **Saturating HBM changes nothing about the write-side
result**, for the reason established in the prerequisite section.

### Floor ratios still do not converge

| | (b) | (c) | (d) | (b)/(d) |
|---|---|---|---|---|
| Phase 8, 3 QPS, batch unmatched | 1.57× | 1.14× | 1.01× | 1.55× |
| Phase 9, cap 48, matched, unsaturated | 16.06× | 11.56× | 7.17× | 2.24× |
| **Phase 9, cap 256, matched, HBM saturated** | **15.66×** | **11.28×** | **6.73×** | **2.33×** |

Matching batch and saturating HBM did not bring them together; the gap widened
slightly. **What is varying is not concurrency — it is time spent stalled on
far-tier writes.** `save_wait` is **6712 / 3760 / 733 s** against durations
**2610 / 1880 / 1122 s** (conversation) and **3362 / 1845 / 382 s** against
**1335 / 940 / 574 s** (Tool&Agent): same ranking, same rough proportionality,
across both traces and all three configs.

With concurrency held equal *and* HBM saturated, the config that writes **80×
less** to HBF completes identical work **2.33×** faster (conversation) and
**2.32×** faster (Tool&Agent). That is the un-confounded throughput cost of the
NAND write stream, and it is now the cleanest version of that number this
project has.

### PCM residency vs time-to-first-read — unchanged by saturation

| | mean PCM residency | **vs TTFR median** |
|---|---|---|
| **Conversation** (TTFR median 202.51 s) (b) | 109.17 s | **0.54×** |
| Conversation (c) | 77.36 s | **0.38×** |
| Conversation (d) | 38.85 s | **0.19×** |
| **Tool&Agent** (TTFR median 277.56 s) (b) | 107.48 s | **0.39×** |
| Tool&Agent (c) | 76.27 s | **0.27×** |
| Tool&Agent (d) | 39.09 s | **0.14×** |

**Residency did not rise.** It is essentially identical to the unsaturated
cap-48 runs (conversation (b) 113.34 → 109.17 s; (c) 81.22 → 77.36 s), and the
expectation that saturating HBM would raise it does not hold — for the same
reason given under the cap-48 results, now reinforced by the prerequisite
section: **residency is pool capacity ÷ admission rate**, and neither term
depends on HBM occupancy. Admissions come from the save path at prefill, not
from HBM eviction.

**The capacity conclusion is therefore robust across every regime tested.** In
all twelve matched runs — unsaturated and saturated, both traces, all three
policies — **mean PCM residency is below the median time-to-first-read**, at
between 0.54× and 0.14×. At 43.4 GiB per GPU the PCM tier absorbs the write
stream but does not retain blocks long enough to serve their reuse.

## STEP 4 — HBF lifetime vs batch size: the curve is flat

Eight batch points, `--max_parallem_sum` ∈ {8, 16, 32, 48, 64, 96, 192, 256},
W = 5,000, 30 QPS offered, both traces, configs (b) and (c). Lifetime is
work-normalized: bytes written to HBF per request against the cluster endurance
budget `24 TiB × 3000 P/E = 79,164,837,199,872,000 B`, assuming WAF = 1 and
ideal wear levelling.

### Conversation

| cap | achieved batch | **peak HBM** | (b) B/req | (b) B/tok | **(b) requests to wear out** | (c) B/req | **(c) requests to wear out** | **(b)→(c)** |
|---|---|---|---|---|---|---|---|---|
| 8 | 3.94 / 3.92 | 25.0% / 29.1% | 3,377,115,693 | 251,793 | 23,441,553 | 1,650,193,859 | 47,973,053 | **×2.046** |
| 16 | 7.83 / 7.84 | 39.8% / 37.6% | 3,370,987,815 | 251,336 | 23,484,166 | 1,651,107,693 | 47,946,501 | **×2.042** |
| 32 | 15.46 / 15.48 | 62.1% / 70.3% | 3,383,093,625 | 252,239 | 23,400,132 | 1,650,764,284 | 47,956,476 | **×2.049** |
| 48 | 23.17 / 23.08 | 85.7% / 97.9% | 3,370,675,339 | 251,313 | 23,486,343 | 1,648,185,311 | 48,031,515 | **×2.045** |
| 64 | 30.81 / 30.35 | 99.0% / 99.0% | 3,376,704,651 | 251,763 | 23,444,407 | 1,648,492,544 | 48,022,563 | **×2.048** |
| 96 | 42.16 / 41.35 | 99.1% / 99.1% | 3,385,003,082 | 252,381 | 23,386,932 | 1,652,367,557 | 47,909,944 | **×2.049** |
| 192 | 43.61 / 42.90 | 99.2% / 99.2% | 3,387,654,930 | 252,579 | 23,368,625 | 1,651,733,692 | 47,928,330 | **×2.051** |
| 256 | 43.61 / 42.90 | 99.2% / 99.2% | 3,387,654,930 | 252,579 | 23,368,625 | 1,651,733,692 | 47,928,330 | **×2.051** |
| | | | **spread 0.50%** | | **spread 0.50%** | **spread 0.25%** | **spread 0.25%** | **spread 0.44%** |

### Tool&Agent

| cap | achieved batch | **peak HBM** | (b) B/req | (b) B/tok | **(b) requests to wear out** | (c) B/req | **(c) requests to wear out** | **(b)→(c)** |
|---|---|---|---|---|---|---|---|---|
| 8 | 3.90 / 3.89 | 29.5% / 33.6% | 1,696,587,055 | 178,599 | 46,661,229 | 809,698,329 | 97,770,780 | **×2.095** |
| 16 | 7.75 / 7.72 | 41.5% / 39.9% | 1,697,241,367 | 178,667 | 46,643,240 | 810,559,734 | 97,666,876 | **×2.094** |
| 32 | 15.29 / 15.27 | 58.5% / 59.0% | 1,695,311,987 | 178,464 | 46,696,324 | 809,890,742 | 97,747,552 | **×2.093** |
| 48 | 22.89 / 22.64 | 75.1% / 82.3% | 1,696,787,333 | 178,620 | 46,655,721 | 811,414,847 | 97,563,949 | **×2.091** |
| 64 | 30.38 / 30.22 | 96.9% / 98.8% | 1,697,208,861 | 178,664 | 46,644,134 | 810,043,834 | 97,729,078 | **×2.095** |
| 96 | 43.03 / 42.40 | 99.1% / 99.1% | 1,678,879,752 | 176,734 | 47,153,369 | 807,156,056 | 98,078,725 | **×2.080** |
| | | | **spread 1.10%** | | **spread 1.09%** | **spread 0.53%** | **spread 0.53%** | **spread 0.72%** |

### The result: there is no N, and therefore no X → Y

**HBF write volume per request and per token is flat in batch size.** Over a
**12× range of concurrency**, driving peak HBM occupancy from **25% to 99.2%**,
HBF bytes per request move by **0.50%** (conversation) and **1.10%**
(Tool&Agent) — noise, and not even monotone. The PCM multiplier is flat at
**×2.05** (conversation) and **×2.09** (Tool&Agent) at every single point.

So the deployment rule the sweep was meant to produce **does not exist in this
model**:

- **X (the batch below which HBF stays effectively read-only, without PCM): there
  is none.** At the smallest batch tested — 3.94 concurrent requests per
  scheduler, HBM at 25% peak — HBF still absorbs **16.89 TB** over 5,000
  requests (conversation). The read-only regime is never entered.
- **Y (where PCM moves that threshold to): undefined**, because X is undefined.
  PCM halves the write volume *uniformly* at every batch size; it does not shift
  a threshold, because there is no threshold to shift.

**The reason is structural, not a measurement problem** — see the prerequisite
section above. HBF writes are Mooncake *memory-tier* evictions, and blocks enter
that tier from the **save path at prefill admission**, once per request
(`mooncake_store.py:119-153, 186-193`), never from HBM eviction
(`llm_scheduler.py:319-325, 365-367` release GPU blocks and write nothing). The
pool is 142,213 blocks against a ~3.29 M-block workload footprint — a **23×
oversubscription that exists at every batch size** — so it overflows into HBF at
the same rate whatever HBM is doing.

**What the flat curve does establish, and it is worth keeping:**

1. **The endurance results are concurrency-invariant.** Every lifetime number in
   Phases 8 and 9 — ×90.31, ×86.39, ×2.05, ×2.09 — holds unchanged from 25% to
   99% HBM occupancy. They are properties of the policy and the pool size, not
   of the operating point, which makes them considerably more robust than a
   figure measured at one batch would be.
2. **It independently confirms the correction recorded above**: interconnect and
   scheduling effects change timing, not byte counts. Throughput varies 19,439 →
   25,698 tok/s across the conversation sweep (+32%) while HBF bytes per request
   vary 0.50%.
3. **The write stream is set by pool capacity against workload footprint**, and
   that is the knob that would move it. A sweep of `memory_capacity_blocks` —
   not of batch size — is what would produce a deployment rule of this shape.

**To make a batch-dependent rule derivable at all, TokenSim needs a code
change**: HBM eviction would have to demote its victim into the Mooncake store,
and the prefill-time whole-prompt save would have to be removed. That is a
change to the KV-transfer model rather than a configuration, so per the working
rule on simulator constraints it is reported here and **not made**.

Stopping here per instruction.

---

# SCOPE NOTE — making HBF wear depend on HBM pressure

Phase 9 STEP 4 concluded that the deployment rule "below batch N, HBF stays
read-only" is not derivable from TokenSim, and that fixing it needs a code
change rather than a knob. That change is scoped in
**`TokenSim/docs/hbm-demote-scope.md`** (scope only — nothing implemented).

Three points from it worth recording here, because they change what a future
session should do:

1. **The obvious hook is the wrong one.** Routing `release_request_blocks`
   (`llm_scheduler.py:319-325`, `:365-367`) into `store.put` would *not* produce
   batch dependence — every request finishes eventually, so the write stream
   would stay flat. The capacity-driven event is **`KVCacheManager.evict`**
   (`kv_cache_manager.py:158-161`), called only from `BlockAllocator.allocate`
   (`block_manager.py:38-42`) when a cached free block is repurposed because HBM
   is full.

2. **Generated KV cannot currently be stored at all.** `pool_keys_for_request`
   truncates to `prefill_len // block_size` (`pool_key.py:59-62`), and the
   optional `output_hash_ids` field is absent from both Mooncake trace files, so
   `register_output_blocks` short-circuits (`kv_cache_manager.py:105`).
   Representing the transient KV write stream — the stated subject of this
   thesis — requires adding a key space for decode blocks.

3. **A cheaper experiment answers most of the same question.** Phase 9 showed
   the write stream is set by **pool capacity against workload footprint**, not
   by concurrency. Sweeping `memory_capacity_blocks` needs **no code change**
   and yields HBF lifetime versus PCM capacity — "how much PCM before HBF wear
   is acceptable" — which is arguably the more useful rule for this thesis.
   Recommended before committing to the demote-path rewrite.

---

# SCOPE STATEMENT — the store has only ever held prompt blocks

**This applies retroactively to every result in this document.** It is a
statement about what the measurements mean, **not a defect report**.

## The mechanism

`pool_keys_for_request` truncates the key chain to the prompt:

```python
# TokenSim/mooncake/pool_key.py:59-62
full_input_blocks = req.prefill_len // req.block_size
prefix_keys = build_prefix_keys(
    req.hash_ids[:full_input_blocks], ...
)
```

Decode-generated blocks therefore have **no keys at all**, and a block with no
key cannot be put into the store. `_build_save_plan` applies the same truncation
independently (`mooncake_store.py:329`, `full_blocks = min(len(keys),
req.prefill_len // self.block_size)`).

There is an optional path for generated blocks — `register_output_blocks`
(`kv_cache_manager.py:104-135`) — but it is driven by a trace field,
`output_hash_ids` (`workload/loaders.py:128`, `:215`), and **neither Mooncake
trace file carries it.** Both `conv_5000.jsonl` and `toolagent_5000.jsonl` have
exactly `timestamp, input_length, output_length, hash_ids, cache_salt`, so the
function short-circuits on its first line (`kv_cache_manager.py:105`,
`if not req.output_hash_ids: return`). And in any case that path registers
blocks in the **GPU prefix cache**, not in the Mooncake store.

## What this means for every result reported here

**Every write volume, residency, hit rate and lifetime multiplier in Phases 1-9
measures prefix-cache offload traffic — the movement of *prompt* KV between the
memory tier and HBF. None of it measures transient generated KV.**

Concretely, for the two traces: prompt-block instances 4,080,885
(conversation) and 2,908,548 (Tool&Agent), of which distinct blocks are
**2,695,220** (6,580 GiB) and **1,397,479** (3,412 GiB) — a 1.51× and 2.08×
dedup factor. Generated tokens — 67,061,296 and 47,497,243 across the two
workloads, roughly 4.19 M and 2.97 M blocks' worth — **never enter the store at
all.**

## The constraint this places on claims

`CLAUDE.md` states the thesis subject as: *"Model weights and any shared cache
are NOT the subject here. Generated/transient KV is."* The measurements do not
currently cover that subject.

> **No result in this document may be described as absorbing the transient KV
> write stream until the key-space work is done.** The defensible claim is
> narrower and should be stated in these terms: *PCM as a write-back buffer in
> front of HBF reduces prefix-cache offload writes by 51% (`always`) or 99%
> (`if_read`).* That claim is sound, it is what was measured, and it is
> independent of the gap above.

The key-space work is scoped in `TokenSim/docs/hbm-demote-scope.md` §3 workstream
C. It is **not** started here.

---

# PHASE 10 — PCM capacity sweep: how much PCM before HBF wear is acceptable

Phase 9 showed HBF write volume is set by **pool capacity against workload
footprint**, not by batch size (0.50% / 1.10% across a 12× concurrency range).
This phase sweeps the knob that does connect. **No code change.**

## Setup

Identical to the Phase 9 saturated runs: `H3-B200-KVONLY` (192 GiB KV per rank),
LLaMa2-70B-GQA TP2/DP4, block 2,621,440 B, HBF 10,066,329 blocks (24 TiB),
HBF reads 0.1 µs / 7451 GiB/s, writes 200 µs / 3.0 GB/s,
`charge_eviction_writes=ON`, W = 5,000, `--max_parallem_sum 256` (HBM
saturated), 30 QPS offered, both traces, separate `--results_path` per run.
Configs **(b)** `write_through`+`always` and **(c)** `write_back`+`always`;
(d) skipped as instructed. The 142,213 points are the Phase 9 `sat_*` runs
reused unchanged.

### The workload footprint — measured, not estimated

Counted directly off the traces using the same truncation the store applies
(`pool_key.py:59`, `input_length // 16`):

| trace | prompt-block instances | **distinct blocks** | **as bytes** | dedup |
|---|---|---|---|---|
| **Conversation** | 4,080,885 | **2,695,220** | **6,580 GiB** | 1.51× |
| **Tool&Agent** | 2,908,548 | **1,397,479** | **3,412 GiB** | 2.08× |

Eight pool sizes were run as **absolute** capacities (the pool is hardware) and
are reported as a percentage of each trace's own footprint. Conservation checks
(`demoted + dropped == memory evictions`, byte exactness, `notdone = 0`,
`ssd_eviction_count = 0`) pass in all 28 runs.

## The curve — HBF lifetime vs PCM capacity

Requests to wear out the 24 TiB cluster HBF at TLC 3000 P/E, WAF = 1.

### Conversation (footprint 2,695,220 blk = 6,580 GiB)

| pool blocks | GiB | **GiB/GPU** | % footprint | (b) B/req | **(b) requests** | (c) B/req | **(c) requests** | (b)→(c) |
|---|---|---|---|---|---|---|---|---|
| 26,952 | 66 | 8.2 | 1.0% | 3,443,815,088 | 22,987,540 | 1,709,646,021 | 46,304,812 | ×2.01 |
| 107,809 | 263 | 32.9 | 4.0% | 3,407,126,987 | 23,235,071 | 1,670,769,541 | 47,382,260 | ×2.04 |
| **142,213** | **347** | **43.4** | **5.3%** | **3,387,654,930** | **23,368,625** | **1,651,733,692** | **47,928,330** | **×2.05** |
| 269,522 | 658 | 82.3 | 10.0% | 3,312,823,828 | 23,896,483 | 1,582,069,973 | 50,038,771 | ×2.09 |
| 673,805 | 1,645 | 205.6 | 25.0% | 3,097,571,623 | 25,557,064 | 1,369,868,075 | 57,790,118 | ×2.26 |
| 1,347,610 | 3,290 | 411.3 | 50.0% | 2,744,303,747 | 28,846,966 | 1,016,600,199 | 77,872,144 | ×2.70 |
| 2,695,220 | 6,580 | 822.5 | 100.0% | 2,037,767,995 | 38,848,798 | 310,064,447 | **255,317,363** | ×6.57 |
| 4,042,830 | 9,870 | 1,233.8 | 150.0% | 1,725,419,749 | 45,881,495 | **0** | **NEVER WEARS** | **∞** |

### Tool&Agent (footprint 1,397,479 blk = 3,412 GiB)

| pool blocks | GiB | **GiB/GPU** | % footprint | (b) B/req | **(b) requests** | (c) B/req | **(c) requests** | (b)→(c) |
|---|---|---|---|---|---|---|---|---|
| 26,952 | 66 | 8.2 | 1.9% | 1,753,694,077 | 45,141,760 | 865,640,907 | 91,452,283 | ×2.03 |
| 107,809 | 263 | 32.9 | 7.7% | 1,703,842,152 | 46,462,542 | 823,357,604 | 96,148,790 | ×2.07 |
| **142,213** | **347** | **43.4** | **10.2%** | **1,695,909,675** | **46,679,866** | **804,753,244** | **98,371,566** | **×2.11** |
| 269,522 | 658 | 82.3 | 19.3% | 1,618,618,614 | 48,908,888 | 740,804,788 | 106,863,290 | ×2.18 |
| 673,805 | 1,645 | 205.6 | 48.2% | 1,408,679,543 | 56,197,904 | 528,720,331 | 149,729,134 | ×2.66 |
| 1,347,610 | 3,290 | 411.3 | 96.4% | 1,055,411,667 | 75,008,492 | 175,452,455 | 451,203,930 | ×6.02 |
| 2,695,220 | 6,580 | 822.5 | 192.9% | 880,973,709 | 89,860,613 | **0** | **NEVER WEARS** | **∞** |
| 4,042,830 | 9,870 | 1,233.8 | 289.3% | 880,973,709 | 89,860,613 | **0** | **NEVER WEARS** | **∞** |

## 1. Why the curve has this shape

The two configs obey different identities, and they are exact:

```
(c) write_back : HBF writes == memory-tier evictions          (exactly, all 14 points)
(b) write_through: HBF writes == admissions + evictions       (e.g. T&A @26,952:
                                                               1,685,929 + 1,658,977 = 3,344,906)
```

As the pool grows toward the footprint, evictions → 0. So:

- **(c) collapses to zero.** With no evictions there are no demotes, and
  write-back spends nothing at admission. HBF becomes genuinely read-only.
- **(b) cannot.** Write-through spends one HBF write per admission no matter
  how large the pool. Its floor is the distinct footprint: 3,290,977 blocks
  (conversation) and 1,680,324 (Tool&Agent) — reached exactly, and unchanged
  between the 192.9% and 289.3% Tool&Agent points.

**This is the sharpest statement of the PCM argument the project has produced.**
The benefit of write-back + PCM is not a constant factor; it is the difference
between a stream with a floor and one that can reach zero.

## 2. Where the buildable design point sits

**43.4 GiB/GPU = 347.2 GiB pool = 142,213 blocks = 5.3% (conversation) /
10.2% (Tool&Agent) of the workload footprint.** Marked in both tables.

It sits **far below the knee, on the flat foot of the curve.**

## 3. Linear or flattening through that point? — Neither

The curve through the design point is **flat, not linear**, and it is the flat
*foot* of a hockey stick, not a plateau after a rise. Local returns:

| change | PCM per GPU | (c) lifetime gain |
|---|---|---|
| **Conversation** 5.3% → 10.0% footprint | 43.4 → 82.3 GiB (**2× the silicon**) | 47.9 M → 50.0 M requests (**+4.4%**) |
| Conversation 1.0% → 5.3% | 8.2 → 43.4 GiB (5.3× the silicon) | 46.3 M → 47.9 M (**+3.5%**) |
| **Tool&Agent** 10.2% → 19.3% | 43.4 → 82.3 GiB (**2× the silicon**) | 98.4 M → 106.9 M (**+8.6%**) |

**Doubling PCM area at the design point buys 4-9% more HBF life.** So the
answer to "is size purely an area decision" is **no — at this scale more PCM
does not meaningfully pay.** Neither does less: dropping from 43.4 to 8.2
GiB/GPU costs only 3.4% (conversation). Over the whole 1%→10% region the
lifetime multiplier barely moves (×2.01 → ×2.09).

**43.4 GiB/GPU is therefore not "near the right provisioning point" in the sense
of sitting at a knee.** It is in a region where PCM capacity is close to
irrelevant to HBF lifetime, and the ×2.05 / ×2.11 benefit measured there comes
almost entirely from the **write-back policy**, not from the capacity. That is
the honest reading, and it is a stronger claim for the policy than for the
capacity: **the policy is what buys the 2×; the capacity is not what buys it.**

## 4. Does a knee exist, and where?

**Yes — a very sharp one, at ~100% of the workload footprint**, and it is a true
collapse, not an interpolation:

| trace | last non-zero point | first zero point | **knee** |
|---|---|---|---|
| **Conversation** | 100.0% footprint — 591,401 blk written, ×6.57 | 150.0% — **0 written** | between **100% and 150%** of footprint = **822-1,234 GiB/GPU** |
| **Tool&Agent** | 96.4% — 334,649 blk, ×6.02 | 192.9% — **0 written** | between **96% and 193%** = **411-823 GiB/GPU** |

The approach is superlinear: (c) lifetime runs ×2.05 → ×2.09 → ×2.26 → ×2.70 →
×6.57 → ∞ as capacity goes 5.3% → 10% → 25% → 50% → 100% → 150%.

**The knee is real but unbuildable.** It sits at roughly **822 GiB per GPU** —
**19× the 43.4 GiB/GPU the PCM die budget allows** (2 dies × 35 mm² × 0.62
Gb/mm² × 8 stacks). Reaching it at the stated density would need ~38 dies per
stack rather than 2, or a ~19× density improvement. **This should be reported as
a finding about the gap between the buildable point and the effective point, not
as a provisioning recommendation.**

Note also that the knee is a property of *this workload's footprint*, not of the
hardware. A workload with a 6,580 GiB working set needs ~6,580 GiB of PCM to go
read-only. The rule generalises as **"PCM ≈ working-set size"**, which is the
same statement as "the cache has to hold the working set" — the sweep locates
it, it does not soften it.

## 5. Does residency cross the median TTFR? — Yes, at 1.8-2.8× the design point

**Method correction, made before reporting.** A first pass compared each point's
residency against the single TTFR median measured at 142,213 blocks (202.51 s /
277.56 s). That is wrong, and the direction of the error was not the one
anticipated. TTFR is measured in *simulated seconds* on the same clock as
residency, and larger pools make the run finish faster, so **TTFR shrinks with
capacity too**. Measured directly at 150% of footprint:

| | TTFR median @ 142,213 | **TTFR median @ 4,042,830** | first reads observed |
|---|---|---|---|
| **Conversation** | 202.51 s (duration 1,902.3 s) | **129.02 s** (duration 1,021.2 s) | 379,552 → 379,008 |
| **Tool&Agent** | 277.56 s (duration 1,693.3 s) | **70.12 s** (duration 1,113.4 s) | 198,911 → 198,684 |

The first-read count is essentially unchanged (−0.14% / −0.11%), so **this is not
censoring — it is the clock.** `TTFR_median / duration` is near-constant:
0.1065 → 0.1263 (conversation, mean **0.1164**) and 0.1639 → 0.1338
(Tool&Agent, mean **0.1489**).

The meaningful comparison is therefore against a TTFR scaled to each point's own
duration. Recomputed that way:

| pool | % footprint | (b) resid / TTFR | (c) resid / TTFR |
|---|---|---|---|
| **Conversation** 26,952 | 1.0% | 0.07× | 0.07× |
| 107,809 | 4.0% | 0.27× | 0.27× |
| **142,213 (DESIGN)** | **5.3%** | **0.36×** | **0.35×** |
| 269,522 | 10.0% | 0.69× | 0.67× |
| 673,805 | 25.0% | **1.74×** | **1.73×** |
| 1,347,610 | 50.0% | 3.42× | 3.40× |
| 2,695,220 | 100.0% | 6.39× | 6.08× |
| **Tool&Agent** 26,952 | 1.9% | 0.10× | 0.10× |
| 107,809 | 7.7% | 0.41× | 0.41× |
| **142,213 (DESIGN)** | **10.2%** | **0.54×** | **0.54×** |
| 269,522 | 19.3% | **1.07×** | **1.05×** |
| 673,805 | 48.2% | 2.62× | 2.57× |
| 1,347,610 | 96.4% | 4.95× | 4.59× |

**Interpolated crossings:**

| | crossing capacity | **GiB/GPU** | % footprint | **× design point** |
|---|---|---|---|---|
| **Conversation** (b) | ~389,701 blk | **119 GiB/GPU** | 14.5% | **2.7×** |
| **Conversation** (c) | ~394,417 blk | **120 GiB/GPU** | 14.6% | **2.8×** |
| **Tool&Agent** (b) | ~253,291 blk | **77 GiB/GPU** | 18.1% | **1.8×** |
| **Tool&Agent** (c) | ~257,711 blk | **79 GiB/GPU** | 18.4% | **1.8×** |

Two things are worth noting. First, **(b) and (c) now cross at almost the same
capacity** (within 1.9%), which is the coherent result: both residency and TTFR
scale with the run's duration, so the policy's throughput advantage cancels and
what remains is the capacity effect alone. The raw-TTFR pass had them 44% apart,
which was an artifact of comparing a fast run's residency against a slow run's
TTFR.

Second, **the crossing is much closer to the design point than Phase 9
suggested** — **1.8-2.8×**, i.e. **77-120 GiB/GPU** against the buildable 43.4.
Phase 9's 0.14×-0.54× was a correct measurement at one capacity, and the
duration-normalised value at that capacity is 0.35×-0.54×, consistent with it.

**So on residency the design point is close to viable — within a factor of two
to three — while on the write-volume knee it is 19× short.** Those are different
questions and they give different answers: PCM at 43.4 GiB/GPU is nearly large
enough to hold a typical block until its first read, but nowhere near large
enough to stop the block reaching HBF at all.

## 6. What PCM capacity does NOT buy: hit rate

**Prefix cache hit rate is flat across the entire sweep** — 0.1624-0.1646
(conversation), 0.4007-0.4036 (Tool&Agent), a range of ±0.7% and ±0.4% against
a 150× change in capacity. Reuse hit blocks likewise: 662,827-671,624 and
1,165,407-1,173,985.

The reason is visible in the tier split: total store hits are constant, and
capacity only moves them **between tiers**.

| Conversation | memory-tier hit tokens | disk-tier (HBF) hit tokens | **total** |
|---|---|---|---|
| 142,213 (design) | 0 | 5,613,600 | 5,613,600 |
| 673,805 | 1,571,120 | 4,005,104 | 5,576,224 |
| 2,695,220 | 5,474,336 | 101,888 | 5,576,224 |
| 4,042,830 | 5,576,224 | 0 | 5,576,224 |

A block demoted to HBF stays in the store's index and is still found by lookup —
it is just served from the slower tier. **So PCM capacity buys write reduction
and latency, never hit rate.** Throughput reflects that: conversation (c) runs
35,393 → 65,668 tok/s across the sweep (+86%) with `save_wait` falling
3,871 → 565 s, entirely from spending fewer HBF writes.

This matters for how the thesis frames PCM: it is a **write absorber**, and this
sweep shows it is *only* a write absorber. It is not a hit-rate device at any
capacity.

## Summary — the deployment rule this sweep does support

1. **HBF wear is governed by pool capacity ÷ workload footprint, not by batch
   size or concurrency** (this phase plus Phase 9 STEP 4).
2. **Below ~50% of footprint the curve is flat**: PCM capacity is nearly
   irrelevant to HBF lifetime, and the ×2.05 / ×2.11 benefit at the design point
   is bought by the **write-back policy**, not by the PCM capacity.
3. **The knee is at ~100% of footprint** — HBF goes fully read-only under
   write-back — but that is **~822 GiB/GPU, 19× the buildable budget**.
4. **43.4 GiB/GPU sits at 5.3% / 10.2% of footprint**, on the flat foot. It is
   defensible as *the buildable point*, and the ×2.05 / ×2.11 it delivers is
   real, but it must not be described as a capacity optimum.
5. **Only write-back can reach zero.** Write-through has a hard floor at one HBF
   write per distinct block, at any capacity.
6. **Residency and write volume give different answers.** Mean PCM residency
   crosses the median time-to-first-read at **1.8-2.8× the design point**
   (77-120 GiB/GPU), but HBF writes do not collapse until **19×** it. The tier
   is nearly big enough to *hold* a block until its reuse; it is nowhere near
   big enough to *keep it off HBF*.
7. **PCM capacity buys write reduction and latency, never hit rate** — flat at
   0.163 / 0.402 across a 150× capacity range.

**Caveat carried forward.** Per the scope statement above, all of this measures
**prefix-cache offload traffic on prompt blocks**. The workload footprints that
set the knee (6,580 GiB / 3,412 GiB) are *prompt* working sets. Generated KV is
absent from the store entirely, so the knee's position would move if the
key-space work were done — almost certainly to a larger capacity, since the
generated stream (~4.19 M / ~2.97 M blocks' worth) is comparable in size to the
prompt footprint and is not deduplicated.

---

# PHASE 11 — PCM re-admission wear at matched batch (2026-09-18)

Single-purpose entry: obtain the **PCM/local-tier re-admission wear cost**,
config (b) vs config (d), at the **matched-batch** configuration of PHASE 9
STEP 3. This number had been recorded only for *pre-matched-batch* runs, and
those figures must not be assumed to carry over.

## Which run was reproduced, and how it was obtained

The configuration is **PHASE 9 → STEP 3 — matched-batch rerun**:
W = 5,000, `ssd_capacity_blocks = 10066329` (24 TiB HBF pool),
`memory_capacity_blocks = 142213` (347.2 GiB PCM pool), `H3-B200-KVONLY`
hardware, GQA `LLaMa2-70B-GQA` TP2/DP4, block 2,621,440 B, FP16,
`charge_eviction_writes = ON`, offered load **30 QPS**, concurrency cap
**`--max_parallem_sum 48`** per scheduler (read from findings.md, not assumed;
the "peak batch 48" in the source text is the cap being reached, and the cap
itself is 48), achieved batch ~23/scheduler.

**The invocation was found, not reconstructed.** `findings.md` does not record
command lines verbatim, but the PHASE 9 harness survives in the scratchpad of
the session that produced it and was reused **unmodified**:

- runner: `1ed68e7f-.../scratchpad/run_one.py` (holds the HBF/PCM knobs listed above)
- driver: `1ed68e7f-.../scratchpad/step3.sh`, whose four relevant lines are
  `run_one.py m48_{b,d}_{conv,ta} "<overrides>" {conv,toolagent}_5000.jsonl 5000 48 30`
  with
  `(b) = {"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"}`
  and
  `(d) = {"memory_media":"pcm","admission_write_policy":"write_back","demote_policy":"if_read"}`

Reruns were issued through that same `run_one.py` with only the **tag** changed
(`m48_*` → `p11_*`), which changes only `--results_path`. Every other flag,
input file and config byte is identical. The scratchpad hardware catalogue is
byte-identical (md5 `9b4cba823dd4158d9bd3efe16bf74491`) to the repo copy named
in `CLAUDE.md`, `data/hardware/hardware_models_h3b200.json`.

## Reproduction check (this is the proof it is the same configuration)

All four runs reproduce the already-recorded PHASE 9 STEP 3 values **exactly, to
every digit**:

| run | HBF write bytes | HBF write blocks | prefix hit rate | reuse hit blocks | batch/sched |
|---|---|---|---|---|---|
| (b) conv | 16,853,376,696,320 ✓ | 6,429,053 ✓ | 0.1643 ✓ | 670,585 ✓ | 23.167889 ✓ |
| (d) conv | 209,508,106,240 ✓ | 79,921 ✓ | 0.0974 ✓ | 397,596 ✓ | 22.378200 ✓ |
| (b) T&A | 8,483,936,665,600 ✓ | 3,236,365 ✓ | 0.4037 ✓ | 1,174,042 ✓ | 22.885048 ✓ |
| (d) T&A | 85,844,295,680 ✓ | 32,747 ✓ | 0.3650 ✓ | 1,061,727 ✓ | 22.429517 ✓ |

Peak batch = 48 per scheduler (cap reached) and `batch_num_gpu_blocks_per_rank`
= 78,643 in all four, as recorded. `notdone = 0`, `preempt = 0`,
`recomputed_tokens = 0`, `demoted + dropped == memory evictions`, and
`ssd_eviction_count = 0` in all four — so the STEP 3 caveat about this phase
being a *saturated* regime does not manifest as unfinished or preempted
requests here.

## The number

Field names: `mooncake_admission_count` and `mooncake_memory_write_bytes`. The
identity **`admissions × 2,621,440 B == memory write bytes`** was verified and
**holds exactly in all four runs** (`mooncake_memory_write_blocks` equals
`mooncake_admission_count` in every case) — every admission writes exactly one
block into the local/PCM tier, so the two rows below carry the same information
in different units.

### Conversation trace

| | (b) wt/always | (d) wb/if_read | change |
|---|---|---|---|
| **admissions** | **3,285,633** | **3,790,142** | **+504,509 = +15.36%** |
| **PCM/local-tier write bytes** | **8,613,089,771,520** (8.61 TB) | **9,935,629,844,480** (9.94 TB) | **+1,322,540,072,960 = +15.36%** |

### Tool&Agent trace

| | (b) wt/always | (d) wb/if_read | change |
|---|---|---|---|
| **admissions** | **1,689,289** | **1,911,034** | **+221,745 = +13.13%** |
| **PCM/local-tier write bytes** | **4,428,369,756,160** (4.43 TB) | **5,009,660,968,960** (5.01 TB) | **+581,291,212,800 = +13.13%** |

## Supersession

This **supersedes** the earlier PCM re-admission wear figures, which were
measured **before matched batch was introduced** (3 QPS, uncapped concurrency,
so the three configs ran at very different achieved batches):

| entry | location | figures | status |
|---|---|---|---|
| PHASE 8, "Re-admission churn still shifts wear onto PCM" | findings.md ~L3027-3030 | conversation **+15.8%**, Tool&Agent **+13.4%** | **SUPERSEDED** by this entry |
| PHASE 7, "A second cost of `if_read`: re-admission churn shifts wear onto PCM" | findings.md ~L2563-2571 | conversation **+15.8%**, Tool&Agent **+13.8%** (and on the superseded 8 TiB far tier) | **SUPERSEDED** by this entry |

Per this project's append-never-overwrite rule those sections are **kept, not
deleted**; they are marked superseded here rather than edited in place. Do not
quote them as the re-admission cost.

## Does the new number confirm, weaken, or strengthen the earlier finding?

**It confirms it, and very slightly weakens the magnitude.**

| trace | pre-matched-batch | **matched batch** | shift |
|---|---|---|---|
| Conversation | +15.8% | **+15.36%** | −0.44 pts |
| Tool&Agent | +13.4% | **+13.13%** | −0.27 pts |

The effect is **robust to batch matching**: forcing all configs to the same
achieved batch (~23/scheduler) at a 10× higher offered load moves the
re-admission penalty by less than half a percentage point on either trace, and
not at all in character. The mechanism is unchanged — `if_read` destroys dropped
blocks (`write_back` leaves no write-through copy, `store.py:300-307`), so they
must be re-admitted on their next request, and every re-admission is one more
PCM block write.

The direction of the small shift is consistent with the rest of PHASE 9: batch
matching slightly *reduces* (d)'s advantage-driven churn because (d) is no
longer allowed to run at a larger effective batch than (b). It does not change
the conclusion.

**The cost remains unpriced.** `if_read` buys a ~×80-95 HBF lifetime multiplier
by moving wear off NAND, but it *adds* **13-15% of write volume onto PCM**, and
**PCM endurance is still OPEN** (`CLAUDE.md`). Until that parameter is sourced,
the net endurance argument for `if_read` cannot be closed — only the HBF half
of it is measured.

Scope unchanged: Mooncake conversational and Tool&Agent traces as **stand-ins
for CAG**; FP16 KV (hardcoded, conservative); saturated regime (this phase runs
at 30 QPS against the traces' natural 3 QPS, so its latency numbers are not
comparable to PHASE 8's).
