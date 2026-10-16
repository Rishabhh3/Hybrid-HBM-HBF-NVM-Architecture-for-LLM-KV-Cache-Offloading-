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

---

# PHASE 12 — Falsifier: does HBM eviction pressure drive far-tier writes?

**Hypothesis under test.** PHASE 9 STEP 4 found HBF bytes/request flat in batch
size and attributed that to HBM eviction never reaching the Mooncake store. If
an HBM eviction *did* write its victim to the store, HBF bytes/request would
depend on HBM occupancy.

**Decision rule, fixed before any run.** With the eviction hook ON, config (b),
if HBF bytes/request at the high-batch point differs from the low-batch point by
**< 5% on both traces**, the hypothesis is dead.

**Result: it is dead. +0.00% (Conversation), −0.07% (Tool&Agent).** Details
below, numbers first.

**Scope of the verdict.** It covers **prompt-block eviction only.** No decode
block reaches the hook — see STEP 0(b). Generated KV is untouched by this phase,
and the SCOPE STATEMENT above still stands in full.

## Provenance — the workload and harness were rebuilt, then proved identical

Neither trace file, nor `build_trace.py`, nor the PHASE 9/11 `run_one.py`
survived on this machine; every prior session scratchpad was empty. All were
rebuilt from the recipes in this document and are now committed under
`scripts/phase12/` so this cannot recur.

Source: the FAST'25 release traces, fetched from `kvcache-ai/Mooncake`
(`FAST25-release/traces/`).

| file | md5 |
|---|---|
| `conversation_trace.jsonl` (raw) | `bc2f368504912f72f59caee90b5f2e46` |
| `toolagent_trace.jsonl` (raw) | `53cf2bc5eff72e9953348bda80028c93` |
| **`conv_5000.jsonl` (built)** | **`5a19d814562e650fe6e44e4234a4793c`** |
| **`toolagent_5000.jsonl` (built)** | **`c6cd3b0c8022eebb88358719a2596f1c`** |
| `data/hardware/hardware_models_h3b200.json` | `9b4cba823dd4158d9bd3efe16bf74491` |

The hardware catalogue md5 is byte-identical to the one PHASE 11 recorded, so
the hardware side is provably the same file. `build_trace.py` follows the recipe
at ~line 2277 verbatim: first N requests, timestamps ms→s, each 512-token hash
id expanded to sub-ids `32h … 32h+31` and truncated to `ceil(input_length/16)`.

**The raw FAST'25 traces carry no `cache_salt` field**, so none is emitted —
whereas the description above of the old files lists one. This cannot change any
number reported here: `cache_salt` reaches the simulator only through
`make_extra_hash` (`prefix_cache.py:16-27`), which salts every key in a run
identically, so a constant salt and an absent one give identical block, byte and
hit counts and differ only in the literal key strings. A *per-request* salt would
have destroyed all cross-request reuse and is excluded by the hit rates below.

### Reproduction gate — 12 independently recorded values, all exact

Nothing was tuned to make these match; they came out on the first run.

**Workload statistics** (counted with the store's own truncation, `//16`):

| | prompt-block instances | distinct blocks | input mean/median | output mean/median |
|---|---|---|---|---|
| Conversation | **4,080,885** ✓ | **2,695,220** ✓ | **13,066 / 7,748** ✓ | **346 / 356** ✓ |
| Tool&Agent | **2,908,548** ✓ | **1,397,479** ✓ | **9,315 / 6,397** ✓ | **185 / 30** ✓ |

**Simulator, config (b) at cap 48, 30 QPS, W = 5,000** — against the PHASE 9
STEP 3 rows (~line 4128/4130):

| | HBF write bytes | HBF write blocks | hit rate | reuse hit blocks | batch/sched | peak HBM |
|---|---|---|---|---|---|---|
| Conversation | **16,853,376,696,320** ✓ | **6,429,053** ✓ | **0.1643** ✓ | **670,585** ✓ | **23.167889** ✓ | 85.66% ✓ |
| Tool&Agent | **8,483,936,665,600** ✓ | **3,236,365** ✓ | **0.4037** ✓ | **1,174,042** ✓ | **22.885048** ✓ | 75.12% ✓ |

`batch_num_gpu_blocks_per_rank = 78,643` in both, as recorded. The four PHASE 9
STEP 4 stock rows reproduce exactly too — see the `stock` rows of the main table.

## STEP 0 — recon (line numbers current as of `4e9b9fe`)

**a. The call chain, with a site the brief did not have.** `KVCacheManager.evict`
(`kv_cache_manager.py:158`) has **two** callers, not one:

| site | trigger |
|---|---|
| `BlockAllocator.allocate` → `evict(block)` (`block_manager.py:38-42`) | capacity: a **cached** free block is repurposed |
| `KVCacheManager.register_blocks` → `evict(block)` (`kv_cache_manager.py:96-97`) | re-keying a block that already carried a different key |

Only the first is HBM pressure. They are counted separately
(`hbm_evict_blocks` vs `hbm_evict_rekey_calls`); **the re-key site fired 0 times
in all eight runs.** Release paths confirmed to write nothing to the store:
`llm_scheduler.py:319-325` and `:365-367`.

**b. What is repurposed: full prompt blocks, never decode blocks.** Three
independent reasons, all verified in code and then in the counters
(`hbm_evict_unkeyed_blocks = 0` in every run):

1. `BlockPool.pop_free_block` (`block_pool.py:27-33`) hands out fresh ids first,
   so `evict` cannot fire until all `num_blocks` ids have been materialised once.
2. `FreeBlockQueue.popleft` (`free_queue.py:72-81`) serves uncached FIFO entries
   before any cached LRU entry, so the uncached pool must be exhausted too.
3. `block.cached` is set only by `register_blocks`, reached from
   `_commit_input_cache` with `req.input_cache_keys` — the prompt range only. A
   decode block from `append_slot` has `cached = False`; on release,
   `release_block` (`kv_cache_manager.py:152-154`) wipes its metadata, and
   `allocate` then skips the evict branch entirely.

**The generated-block branch at this hook is therefore unreachable and was not
built.** The counter that would have caught it is kept.

**c. What `put` does on a key the store already has: LRU refresh, zero bytes.**
`_put_one` (`store.py:191-202`) sees `existing is not None`, calls `_touch`, and
returns an empty `StoreWriteTiming`. So an eviction-driven save produces a new
far-tier write **only** for a key the store does not currently hold. Under config
(b) (`write_through` + `always`) nothing is ever removed from `store.objects` —
`_drops_on_eviction` is False and `ssd_eviction_count = 0` — so the only way a
key can be absent is that it was never written: `_aligned` save semantics
(`mooncake_store.py:131-137`) mark a request saved and then **skip its save
entirely if that step also loaded from the store**, leaving that request's miss
blocks in HBM but not in the store. Measured, that channel is **2.0–3.0%** of
evicted blocks.

**d. Batch points** — the lowest- and highest-occupancy rows of the PHASE 9
STEP 4 sweep (~line 3645/3661), copied verbatim: Conversation `--max_parallem_sum`
**8** (peak HBM 25.0%) and **256** (99.2%); Tool&Agent **8** (29.5%) and **96**
(99.1%). W = 5,000, 30 QPS offered, `memory_capacity_blocks = 142213`,
`ssd_capacity_blocks = 10066329`, `charge_eviction_writes = ON`, `H3-B200-KVONLY`,
`LLaMa2-70B-GQA` TP2/DP4, block 2,621,440 B, FP16, `--random_seed 0`.

## The flag

`save_policy` was already taken (`config.py:87`, values `mooncake`/`every_step`/
`once`, meaning how often the *prefill* save re-runs), so the new knob is
**`hbm_evict_save_policy`**:

| value | meaning |
|---|---|
| **`prefill`** | **default, stock.** No observer installed; no counter emitted. |
| `count` | instrumentation only — no store write, no behaviour change |
| `on_evict` | prefill save **plus** an eviction-driven save |

**Byte-identity at default, verified.** Both traces, config (b), cap 48, against
stock `4e9b9fe`:

```
strip(){ python -c "import json,sys;d=json.load(open(sys.argv[1]));
  [d.pop(k,None) for k in ('simulator_wall_time','mooncake_pool_keys')];
  print(json.dumps(d,sort_keys=True,indent=1))" "$1"; }
diff <(strip head/result_30.0.json) <(strip p12_default/result_30.0.json)
```

→ **no output for either trace.** `count` is behaviourally stock as well: every
non-`hbm_evict_*` field of the four `stock` rows below matches the PHASE 9
STEP 4 figures exactly.

**Tests: 177 pass** (158 pre-existing, unchanged, plus 19 new in
`tests/test_hbm_evict_policy.py`).

One export defect had to be fixed to get any counter out at all: `LLMResult`
(`psla_config.py:36`) is a **pydantic** dataclass, so `export_result`'s
`LLMResult(**mooncake_stats)` silently **drops** keys the schema does not
declare. Declaring the new fields would have put null-valued keys in every stock
result, so they are merged into `result_dict` after `asdict()` instead
(`util/results.py`), leaving stock output untouched.

## STEP 3/4 — the runs

Config (b) only (`memory_media=dram`, `admission_write_policy=write_through`,
`demote_policy=always`). Conservation checks pass in all eight runs:
`demoted + dropped == memory evictions`, `ssd_eviction_count = 0`, `notdone = 0`,
`preemption_count = 0`, `recomputation_count = 0`.

### Conversation

| point | policy | batch/sched | peak HBM | evict blocks (cap / re-key) | key already in store | key absent | puts refresh-only / wrote bytes | **HBF B/req** | HBF wr ops/req | PCM B/req | hit rate | reuse hit blk | eff. prefill tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap 8 | stock | 3.942 | 24.96% | 3,387,012 / **0** | 3,297,468 | 89,544 | — | **3,377,115,693** | 1,288.3 | 1,725,838,131 | 0.1647 | 672,089 | 54,577,535 |
| cap 8 | **on_evict** | 3.944 | 24.96% | 3,387,012 / **0** | 3,300,607 | 86,405 | 3,300,607 / **86,405** | **3,467,550,130** | 1,322.8 | 1,771,055,350 | 0.1658 | 676,796 | 54,502,223 |
| cap 256 | stock | 43.609 | 99.19% | 3,443,404 / **0** | 3,364,388 | 79,016 | — | **3,387,654,930** | 1,292.3 | 1,731,107,750 | 0.1631 | 665,419 | 54,684,255 |
| cap 256 | **on_evict** | 43.693 | 99.19% | 3,443,404 / **0** | 3,366,372 | 77,032 | 3,366,372 / **77,032** | **3,467,556,422** | 1,322.8 | 1,771,058,495 | 0.1638 | 668,491 | 54,635,103 |

### Tool&Agent

| point | policy | batch/sched | peak HBM | evict blocks (cap / re-key) | key already in store | key absent | puts refresh-only / wrote bytes | **HBF B/req** | HBF wr ops/req | PCM B/req | hit rate | reuse hit blk | eff. prefill tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap 8 | stock | 3.901 | 29.54% | 1,536,486 / **0** | 1,506,434 | 30,052 | — | **1,696,587,055** | 647.2 | 885,573,812 | 0.4037 | 1,174,113 | 27,788,891 |
| cap 8 | **on_evict** | 3.893 | 29.54% | 1,536,486 / **0** | 1,506,562 | 29,924 | 1,506,562 / **29,924** | **1,727,931,089** | 659.2 | 901,245,829 | 0.4040 | 1,174,945 | 27,775,579 |
| cap 96 | stock | 43.031 | 99.14% | 1,558,264 / **0** | 1,512,237 | 46,027 | — | **1,678,879,752** | 640.4 | 876,720,161 | 0.4020 | 1,169,287 | 27,866,107 |
| cap 96 | **on_evict** | 42.775 | 99.14% | 1,558,264 / **0** | 1,512,621 | 45,643 | 1,512,621 / **45,643** | **1,726,739,907** | 658.7 | 900,650,238 | 0.4022 | 1,169,959 | 27,855,355 |

### The decision rule

| | HBF B/req, high batch vs low batch | verdict |
|---|---|---|
| **Conversation, `on_evict`** | 3,467,550,130 → 3,467,556,422 = **+0.00%** | < 5% |
| **Tool&Agent, `on_evict`** | 1,727,931,089 → 1,726,739,907 = **−0.07%** | < 5% |

**Both traces are under 5%. The hypothesis is dead.** Routing HBM evictions into
the store does **not** make far-tier write volume depend on HBM occupancy. It is
not a small effect that needs a longer run or a wider sweep; it is **zero to
three significant figures on one trace and negative on the other**, across an
11× batch range that drives peak HBM occupancy from 25% to 99.2%.

On_evict against stock at each point, for completeness:

| point | HBF B/req, on_evict vs stock |
|---|---|
| Conversation cap 8 | +2.68% |
| Conversation cap 256 | +2.36% |
| Tool&Agent cap 8 | +1.85% |
| Tool&Agent cap 96 | +2.85% |

### Why it is flat — the mechanism, now measured

**Eviction volume itself barely moves with batch.** Over the same 11× batch
range: Conversation 3,387,012 → 3,443,404 evicted blocks (**+1.66%**),
Tool&Agent 1,536,486 → 1,558,264 (**+1.42%**). The reason is
`BlockPool.pop_free_block`: `_next_fresh` counts blocks **ever** handed out, not
blocks currently held. Once 78,643 ids have been touched — which happens early
at every batch size — every further allocation comes from the free queue, and the
eviction rate tracks *total allocations*, i.e. the workload's block footprint,
not instantaneous occupancy. HBM occupancy changes **how long** a block sits
before eviction, not **how many** blocks are evicted.

**And 97.4–98.0% of those evictions are duplicates.** Under config (b) nothing
leaves `store.objects`, so almost every evicted block's key is still there and
its put is an LRU refresh costing zero bytes. The write increase comes entirely
from the 2.0–3.0% whose save was skipped by the `_aligned` load/save exclusion.

The accounting closes: the extra bytes are exactly the hook's own writes.

| point | Δ(HBF + PCM write bytes), on_evict − stock | `hbm_evict_write_bytes` |
|---|---|---|
| Conversation cap 8 | 678,258,278,400 | 679,516,569,600 |
| Conversation cap 256 | 599,261,184,000 | 605,804,298,240 |
| Tool&Agent cap 8 | 235,080,253,440 | 235,331,911,680 |
| Tool&Agent cap 96 | 358,951,157,760 | **358,951,157,760** (exact) |

The 0.2–1.1% residuals are second-order: the extra write stalls shift the
schedule slightly, which changes the prefill-save stream a little.

### Confounds, all flagged

1. **Achieved batch does move, slightly, and not in one direction.** Conversation
   **+0.04%** (cap 8) and **+0.19%** (cap 256); Tool&Agent **−0.21%** (cap 8) and
   **−0.59%** (cap 96). The largest is Tool&Agent cap 96. All are an order of
   magnitude smaller than the 5% threshold and could not hide an occupancy
   effect, but they are a real mismatch between the stock and `on_evict` arms.
2. **Latency and throughput deltas are contaminated and must not be read as a
   performance result.** Throughput falls 1.2–2.4% under `on_evict`
   (Conversation 19,439 → 19,202 and 25,698 → 25,348 tok/s; Tool&Agent
   26,468 → 26,177 and 36,280 → 35,409), and mean e2e rises 0.5–1.2%. That is
   queueing feedback from the extra `save_wait` the hook charges (+175.2 s,
   +156.2 s, +60.7 s, +92.5 s of simulated time), not a property of the memory
   hierarchy. The far tier still has **no shared-bandwidth model**, so no day
   count or lifetime is quoted anywhere in this phase.
3. **Duplicate-key writes dominate the hook's traffic** — by design, and reported
   rather than suppressed. The `refresh-only / wrote bytes` split is the honest
   version of "how much of this hook is real".
4. **Peak HBM occupancy is identical between the two arms** at every point
   (24.96 / 99.19 / 29.54 / 99.14), so the two arms really are being compared at
   the same occupancy.

### What this does and does not settle

- **Settled:** prompt-block HBM eviction is not a channel by which HBM pressure
  reaches the far tier, even when it is wired directly into the store. PHASE 9
  STEP 4's flat curve is **not** an artefact of the missing hook.
- **Not settled, and untouched:** generated/transient KV. It has no key space, so
  it cannot enter the store at all, and — per STEP 0(b) — it cannot reach this
  hook either. The thesis subject remains unmeasured, exactly as the SCOPE
  STATEMENT above says.
- **Follow-up not built, as instructed:** the shared-key lower-bound variant of
  the hook.

## STEP C — option 2 feasibility (trace analysis only, no simulator change)

Does generated KV actually get re-read? Measured on the **raw 512-token hash
ids**. A request `R` with ≥ 2 ids has stem `R.hash_ids[:-1]` — the last id
usually covers a partial block that grows as the request generates — and a later
request `L` *continues* `R` when `L.hash_ids[:len(stem)] == stem`. Script:
`scripts/phase12/stepc.py`. No simulator was started.

| | window | (1) requests continued by a later request | (2) gap to first continuation, p50 / p90 / mean | extra 512-tok hashes past the stem, p50 / p90 / mean | `ceil(output_length/512)`, p50 / p90 / mean | extra ÷ expected, p50 / mean |
|---|---|---|---|---|---|---|
| **Conversation** | W = 5,000 | **2,001 / 5,000 = 40.02%** | **72.0 / 366.0 / 130.4 s** | 2 / 28 / 10.46 | 1 / 2 / 1.23 | 2.00 / 9.49 |
| | full (12,031) | **5,287 / 12,031 = 43.94%** | 78.0 / 411.0 / 158.0 s | 2 / 23 / 9.18 | 1 / 2 / 1.22 | 2.00 / 8.37 |
| **Tool&Agent** | W = 5,000 | **2,012 / 5,000 = 40.24%** | **0.0 / 171.0 / 49.5 s** | 2 / 10 / 5.51 | 1 / 1 / 1.11 | 2.00 / 5.08 |
| | full (23,608) | **10,413 / 23,608 = 44.11%** | 0.0 / 234.0 / 70.7 s | 2 / 10 / 5.16 | 1 / 2 / 1.11 | 2.00 / 4.76 |

Three observations, no conclusions drawn:

1. **Roughly 40% of requests are continued by a later one**, on both traces. The
   W = 5,000 window costs about **4 percentage points** against the full trace
   (40.02% vs 43.94%; 40.24% vs 44.11%) — the truncation is a modest
   underestimate, not a structural one.
2. **The gaps are long, and are the same order as the residencies already
   recorded.** Conversation p50 **72 s**, p90 366 s. PHASE 9 STEP 3 measured mean
   PCM residency of 109.17 s (b) / 77.36 s (c) / 38.85 s (d) at this pool size,
   so a continuation-driven reuse would land inside the (b)/(c) residency window
   and outside (d)'s. Tool&Agent's p50 gap is **0.0 s** — continuations arrive in
   the same trace millisecond, which is a much easier target.
3. **The continuation carries about twice as much new prompt as the previous
   output could explain** (extra ÷ expected p50 = **2.00** on both traces, mean
   4.8–9.5). So a continuation's prompt is not just "previous prompt + previous
   output": it contains substantial fresh user/tool input. **Storing generated KV
   would therefore capture only a fraction of what the continuation re-reads**,
   and the mean ratio is heavily skewed by long-tail turns. That fraction is what
   an option-2 hook would have to justify itself on; it is not estimated here.

## Settings and provenance

| | |
|---|---|
| base commit compared against | **`4e9b9fe`** |
| this phase's commit | **`b4cc8f1`** (code, tests, harness, entry) on `falsifier-on-evict`, not merged |
| trace files on this machine | `tmp/phase12/` (gitignored), md5 sums above |
| hardware | `H3-B200-KVONLY`, `data/hardware/hardware_models_h3b200.json` (md5 `9b4cba823dd4158d9bd3efe16bf74491`), unchanged |
| cluster | `data/clusters/8_b200_h3/h8_tp2dp4.json`, 8 workers, TP2/DP4 |
| model | `LLaMa2-70B-GQA` — Nhead 64, KV heads 8 (Grouped_Num 8), Nlayer 80, Dmodel 8192, head_dim 128 |
| KV precision | **FP16**, hardcoded (`cache_config.py`), the conservative choice |
| KV bytes/token, block bytes | 163,840 B/token at TP2; **2,621,440 B** per 16-token block |
| HBM per rank | 192 GiB = **78,643 blocks** (pre-compensated `Capacity`, the documented workaround) |
| PCM / memory pool | **142,213 blocks = 347.2 GiB** (43.4 GiB/GPU × 8) — the design configuration |
| HBF / offload pool | **10,066,329 blocks = 24 TiB** (3 TiB/GPU × 8) |
| HBF timings | read 0.1 µs / 7451 GiB/s (**perfect-prefetch upper bound**, not measured); write 200 µs / 3.0 GB/s |
| workload | Mooncake FAST'25 conversational and Tool&Agent traces — **stand-ins for CAG**, not the target workload |
| W, offered load, seed | 5,000 requests, 30 QPS (`--trace_target_qps 30`), `--random_seed 0` |
| harness | `scripts/phase12/{build_trace.py,run_one.py,step3.sh,stepc.py,stepc_tokens.py,audit_export.py,llama-70b-gqa.json}` |

**No hardware number was invented or changed in this phase.** Every value above
is either read from the tracked files or carried over unchanged from PHASE 9/10.
Trace files are not committed (33 MB / 21 MB); their md5 sums are above and
`scripts/phase12/README.md` gives the source URLs.

---

# PHASE 12 (addendum) — STEP C redone in tokens, and an export audit

Appended after the PHASE 12 verdict; it does not change it. Trace analysis only
— **no simulator was started for any number in this section.**

## STEP 1a — continuations measured in tokens

Same match rule as STEP C, on the raw 512-token hash ids: request `R` with ≥ 2
ids has stem `R.hash_ids[:-1]`, and the first later `R'` whose ids start with
that stem is its continuation. Now measured as
`delta = R'.input_length − R.input_length` against `R.output_length`.
Script: `scripts/phase12/stepc_tokens.py`.

`R.output_length == 0` never occurs in either trace, so no pair is excluded and
every ratio below is defined.

### Conversation

| | continued | `delta` (tokens) p10 / p50 / p90 / mean | **`delta / R.output_length`** p10 / p50 / p90 | **`delta ≥ out`** | **`delta < out`** | of which `delta < 0` |
|---|---|---|---|---|---|---|
| W = 5,000 | 2,001 / 5,000 | 62 / 598 / 13,809 / 4,799.4 | **0.970 / 1.486 / 82.599** | **1,755 = 87.71%** | **246 = 12.29%** | 43 = 2.15% |
| full (12,031) | 5,287 / 12,031 | 64 / 563 / 11,259 / 4,150.0 | **0.977 / 1.450 / 79.870** | **4,661 = 88.16%** | **626 = 11.84%** | 103 = 1.95% |

### Tool&Agent

| | continued | `delta` (tokens) p10 / p50 / p90 / mean | **`delta / R.output_length`** p10 / p50 / p90 | **`delta ≥ out`** | **`delta < out`** | of which `delta < 0` |
|---|---|---|---|---|---|---|
| W = 5,000 | 2,012 / 5,000 | −18 / 443 / 4,416 / 2,302.4 | **−2.333 / 6.364 / 199.000** | **1,553 = 77.19%** | **459 = 22.81%** | 290 = 14.41% |
| full (23,608) | 10,413 / 23,608 | −11 / 444 / 4,647 / 2,118.0 | **−0.667 / 5.308 / 203.809** | **8,206 = 78.81%** | **2,207 = 21.19%** | 1,330 = 12.77% |

### What this says, and what it does not

**The assumption, stated plainly: the trace cannot show that `delta` begins with
`R`'s output.** It records only lengths and chunk hashes — no token content. So
"the continuation contains the previous output" is an **assumption**, not a
measurement, and every figure above is consistent with a continuation whose
extra tokens are entirely unrelated to `R`'s output.

Given that, three things are counted rather than dropped:

1. **`delta < R.output_length` in 12.29% (Conversation) and 22.81% (Tool&Agent)
   of continuations** — W = 5,000; 11.84% and 21.19% on the full traces. For
   these, the continuation's prompt grew by **less** than the previous output.
   **These directly contradict "the continuation contains the whole output."**
   They cannot be explained by a continuation that appends the output verbatim.
2. **`delta < 0` in 2.15% (Conversation) and 14.41% (Tool&Agent)** — the
   continuation's prompt is *shorter* than the one it continues. Tool&Agent's
   14.41% is large enough that it is a property of the workload, not noise:
   agent turns evidently drop or rewrite earlier context. Under the p10 column
   this shows as a **negative ratio** (−2.333 / −0.667).
3. **Where `delta ≥ out` it is usually much larger than `out`.** Conversation
   p50 ratio **1.486**, Tool&Agent p50 **6.364** — and both p90s are enormous
   (82.6, 199.0). So even on the optimistic reading, the previous output is a
   **minority of what the continuation adds**: at the median, roughly two thirds
   of the delta on Conversation and five sixths on Tool&Agent is fresh input.

Conversation's p10 ratio of **0.970** — just under 1 — is worth noting: the
distribution has a visible mass *at* `delta ≈ out`, consistent with a plain
"append the reply, ask a short follow-up" turn, which is what makes the
assumption tempting. Tool&Agent has no such mass.

**Bearing on option 2:** storing generated KV can, at best, serve the part of a
continuation that is the previous output. These numbers bound that at well under
half of the delta at the median on both traces, and at nothing at all for the
12–23% of continuations whose delta is smaller than the output. That bound is
not the same as the cost, which is in the design note.

## STEP 1b — export audit: did the dropped-key defect ever hide a recorded number?

**No. There are none.** The `LLMResult` defect fixed in PHASE 12
(`util/results.py`) was latent: before PHASE 12 no stats builder produced a key
the schema did not declare.

Checked two ways, both exhaustive:

1. **Statically**, over every stats builder `export_result` splats into
   `LLMResult`:

   | builder | keys | dropped |
   |---|---|---|
   | `ConnectorStats.as_dict` | 10 | **none** |
   | `MooncakeStats.as_dict` (stock) | 42 | **none** |
   | `MoEStats.as_dict` | 13 | **none** |
   | `MooncakeStats.as_dict` (PHASE 12 counters on) | 56 | the 14 `hbm_evict_*` keys |

2. **Dynamically**, by running a real simulation with `LLMResult` wrapped to
   capture every keyword handed to it (`scripts/phase12/audit_export.py`;
   it patches nothing in the tree). 124 keys offered, 110 declared, **14
   dropped — all 14 of them `hbm_evict_*`**, i.e. introduced by PHASE 12 itself.

So every counter quoted anywhere in PHASES 1–11 reached the result file. The
defect was introduced-and-fixed within PHASE 12 and has no retroactive reach.

## STEP 2 — option 2 design note

Scoped in **`TokenSim/docs/option2-output-kv-scope.md`** — design only, nothing
implemented, per instruction. Two measurements taken for that note belong here
because they are trace facts, not design choices:

| | Conversation | Tool&Agent |
|---|---|---|
| continuation pairs (W = 5,000) | 2,001 | 2,012 |
| **`R`/`R'` sub-id chains diverge *before* `R`'s last full prompt block** | **1,958 = 97.9%** | **1,986 = 98.7%** |
| size of that divergent tail, `p − s` (16-token blocks) — p50 / max | **22 / 32** | **14 / 32** |
| `R.prefill_len` not a multiple of 16 (boundary block is mixed) | 1,861 = 93.0% | 1,892 = 94.0% |
| `R` generates enough to fill that boundary block | 1,937 = 96.8% | **1,577 = 78.4%** |
| output blocks wanted, `ceil(out/16)` — p50 / mean | 22 / 22.0 | 2 / 11.1 |
| output blocks keyable from `R'` — p50 / mean | 38 / 300.9 | 29 / 146.2 |
| **`R`'s whole output is keyable from `R'`** | **91.4%** | **81.6%** |

The first row is the one that matters and it was not anticipated: **for ~98% of
continuation pairs the two key chains diverge up to 32 blocks before the end of
`R`'s prompt**, because the trace hashes at 512-token granularity while the
simulator blocks at 16, so a partial trailing 512-token chunk gets a fresh id
even where its leading tokens are identical. Since both `plan_reuse`
(`kv_cache_manager.py:58-64`) and `store.lookup` (`store.py:79-93`) stop at the
first miss, `R'` would never reach `R`'s stored output blocks at all unless that
divergent tail is dealt with. The design note treats this as its central
problem.

---

# PHASE 13 — Corrections, a limitation, and a tail-sharing sensitivity run

Append-only: PHASE 12 is untouched; the correction below supersedes one sentence
in it. **PHASE 9 remains the headline result.** Everything here is either a
correction, a limitation, or a sensitivity check.

## STEP 1a — CORRECTION to the PHASE 12 addendum

The PHASE 12 addendum (STEP 1a) concluded:

> "the previous output is a **minority** of what the continuation adds: at the
> median, roughly two thirds of the delta on Conversation and five sixths on
> Tool&Agent is fresh input."

**That is wrong for Conversation, and the arithmetic was inverted.** The median
ratio is `delta / R.output_length`, so the output's share of the delta is its
**reciprocal**:

| | median `delta / out` | **output's share of the added tokens** | |
|---|---|---|---|
| **Conversation** | 1.486 | **1 / 1.486 = 67.3%** | **a MAJORITY** |
| **Tool&Agent** | 6.364 | **1 / 6.364 = 15.7%** | **a minority** |

So at the median, on Conversation the previous output is **about two thirds of
what the continuation adds**, not one third. Only Tool&Agent matches the
original claim, at **~16%**.

This changes the reading of option 2's *ceiling*: on Conversation a stored
output block could serve a clear majority of a continuation's new prompt, which
is a materially better case for the idea than PHASE 12 stated. It does **not**
change anything else in PHASE 12 — no measured number moves, and the
`delta < out` counts (12.29% Conversation, 22.81% Tool&Agent) and `delta < 0`
counts (2.15%, 14.41%) stand as reported.

## STEP 1b — generated KV as a share of all KV

Every token carries the same KV bytes (`size_per_token` is a model/TP constant,
`config/cache_config.py`), so the token share **is** the byte share. Trace
analysis only; script `scripts/phase12/generated_share.py`.

| | window | input tokens | output tokens | input mean / median | output mean / median / p90 | **generated share** |
|---|---|---|---|---|---|---|
| **Conversation** | W = 5,000 | 65,330,959 | 1,730,337 | 13,066.2 / 7,757 | 346.1 / 356 / 606 | **2.58%** |
| | full (12,031) | 144,793,823 | 4,122,048 | 12,035.1 / 6,909 | 342.6 / 350 / 597 | **2.77%** |
| **Tool&Agent** | W = 5,000 | 46,574,699 | 922,544 | 9,314.9 / 6,398 | 184.5 / 30 / 513 | **1.94%** |
| | full (23,608) | 202,940,084 | 4,299,817 | 8,596.2 / 6,346 | 182.1 / 30 / 507 | **2.07%** |

> **Upper bound: storing *all* generated KV could add at most 2.58%
> (Conversation) and 1.94% (Tool&Agent) to the KV byte stream** at W = 5,000 —
> 2.77% and 2.07% on the full traces.

Two things this bound does and does not say:

- It is an **upper** bound on the *addition*, taken against total KV tokens.
  Measured instead against the store's actual write population — **distinct**
  prompt blocks, after the 1.51× / 2.08× dedup — the same output volume is
  **4.0%** and **4.1%** (108,146 output blocks against 2,695,220 distinct
  prompt blocks; 57,659 against 1,397,479). Generated blocks cannot dedup, so
  this second framing is the one that matches what the far tier would see.
- Either way the conclusion is the same and it is worth stating plainly:
  **generated KV is a small fraction of this workload.** These are prompt-heavy
  traces — a 13,066-token mean prompt against a 346-token mean output on
  Conversation, 9,315 against 185 on Tool&Agent. Option 2 cannot move the write
  stream by more than a few percent on *these* traces whatever it does, and that
  bound should be stated wherever option 2 is proposed or evaluated.

## STEP 2 — LIMITATION: the prompt tail a continuation cannot reach

**What it is.** The FAST'25 traces hash at **512 tokens**; the simulator blocks
at **16**. For a continuation pair `(R, R')` the key chains agree only on
sub-indices `[0, s)` with `s = 32·(len(R.raw_hash_ids) − 1)`, because the match
rule excludes `R`'s last raw id — a partial chunk whose content grows. But `R`
holds `p = R.input_length // 16` full prompt blocks, and `s < p` almost always.
`R`'s blocks in `[s, p)` therefore carry sub-ids `R'` never asks for; and since
both `plan_reuse` (`kv_cache_manager.py:58-64`) and `MooncakeStore.lookup`
(`store.py:79-93`) stop at the **first** miss, everything after them is
unreachable too.

**How much**, at W = 5,000 (script `scripts/phase12/tail_divergence.py`):

| | Conversation | Tool&Agent |
|---|---|---|
| continuation pairs | 2,001 | 2,012 |
| **affected** (`s < p`) | **1,958 = 97.9%** | **1,986 = 98.7%** |
| unreachable prompt blocks per affected pair — p50 / p90 / max | **22 / 28 / 32** | **14 / 28 / 32** |
| total unreachable blocks | 36,422 | 27,226 |
| as % of prompt-block instances in the window | 0.89% (of 4,080,885) | 0.94% (of 2,908,548) |
| as % of distinct prompt blocks | **1.35%** (of 2,695,220) | **1.95%** (of 1,397,479) |

**What it limits.** A real prefix cache keys 16-token blocks by token content, so
those blocks would match automatically. The modelled workload therefore has
**~1.4% / ~2.0% more distinct prompt blocks** than a content-addressed cache
would see, which means:

- **Prefix hit rate is understated** in every phase — every `prefix_cache_hit_rate`
  and `reuse_hit_blocks` figure in PHASES 1–12 is a floor, not an estimate.
- **Reuse retained (d vs b)** — 59.3% (Conversation) and 90.4% (Tool&Agent) at
  the cap-48 setting — is measured against that understated base.
- **The `if_read` hit-rate cost, −40.7% / −9.6%**, is measured on the same base
  and inherits the same bias. STEP 4 below shows it is **sensitive** to it.

**What it does not touch.** Every result that is a comparison *between configs
at a fixed workload*: both arms see the same trace, the same keys and the same
footprint, so the differences are unaffected in kind. Specifically, the PHASE 12
falsifier verdict (+0.00% / −0.07%, a within-workload high-vs-low-batch
comparison), the HBF write reductions between (b), (c) and (d), and the
conservation identities. It also does not touch the byte counts as
*measurements* — they are what that workload wrote — though STEP 4 shows the
absolute values are workload-sensitive and so should not be read as
characteristic of the architecture.

## STEP 3 — tail-sharing as a trace option

`build_trace.py` gains **`--share_partial_tail`, default OFF**. When on, each
`R'` takes its predecessor `R`'s sub-ids across `R`'s last 512-token chunk, up
to `floor(R.input_length / 16)`; positions beyond that keep `R'`'s own. The
predecessor is chosen by the PHASE 12 match rule (longest stem wins, ties to the
most recent) and only where `R'.input_length ≥ R.input_length`, so the 2.15% /
14.41% of pairs with negative delta are never rewritten.

**This is an ASSUMPTION, not a correction.** It presumes `R'`'s tokens match
`R`'s up to `R`'s length, which the trace cannot show — the traces carry lengths
and chunk hashes, no token content. The original files remain the reference
workload.

### Default OFF is byte-identical

```
 before:  5a19d814562e650fe6e44e4234a4793c  conv_5000.jsonl
          c6cd3b0c8022eebb88358719a2596f1c  toolagent_5000.jsonl
 rebuilt: 5a19d814562e650fe6e44e4234a4793c  (default off)
          c6cd3b0c8022eebb88358719a2596f1c  (default off)
```

### What it changed, and how broad it is

| | requests changed | sub-ids rewritten | new file md5 | distinct blocks (was) |
|---|---|---|---|---|
| Conversation | 4,904 / 5,000 | 109,166 | `13d05f75aaaae8669619ae88efbb4998` | **2,620,973** (2,695,220, **−2.75%**) |
| Tool&Agent | 4,914 / 5,000 | 80,839 | `edcd7c1028738dff7c0228fd175ea399` | **1,332,772** (1,397,479, **−4.63%**) |

**The rule as specified is far broader than the divergence it was meant to
address, and this must be stated before its results are read.** 4,977 / 5,000
(Conversation) and 4,959 / 5,000 (Tool&Agent) requests have *some* qualifying
predecessor, because the longest-match rule accepts a stem of a single 512-token
chunk — in practice the shared system prompt. On Conversation **70.8%** of
chosen predecessors share exactly **one** chunk, and those pairs account for
**79.6%** of all rewritten sub-ids (Tool&Agent: 32.6% and **49.2%**). For those
pairs the trace evidences agreement on the first 512 tokens only, while the rule
rewrites up to `R.input_length / 16` positions. That is a very strong assumption.

Files: `conv_5000_tailshare.jsonl`, `toolagent_5000_tailshare.jsonl`; the
originals are kept. Tests: **190 pass** (177 + 13 new in
`tests/test_build_trace_tail_sharing.py`), covering default-off identity,
negative-delta exclusion, the `input_length % 16 ≠ 0` boundary, longest-stem and
most-recent tie-breaking, single-chunk ineligibility, and the fact that sharing
**chains** through an already-rewritten predecessor.

## STEP 4 — SENSITIVITY: configs (b) and (d) on the tailshare workload

Configs (b) and (d) at the PHASE 9 STEP 3 batch-matched setting: cap 48, 30 QPS,
W = 5,000, `memory_capacity_blocks = 142213`, `ssd_capacity_blocks = 10066329`,
`charge_eviction_writes = ON`, `H3-B200-KVONLY`
(md5 `9b4cba823dd4158d9bd3efe16bf74491`), `LLaMa2-70B-GQA` TP2/DP4, seed 0.
8 runs: 4 on the tailshare files, 4 on the originals as the paired baseline.
**No latency or day-count figure is quoted.**

**The four original-trace runs reproduce PHASE 9 STEP 3 exactly**, config (d)
included — which had not been re-measured before now:

| | HBF write bytes | HBF blocks | hit rate | reuse hit blocks | batch/sched |
|---|---|---|---|---|---|
| (b) Conversation | 16,853,376,696,320 ✓ | 6,429,053 ✓ | 0.1643 ✓ | 670,585 ✓ | 23.167889 ✓ |
| (d) Conversation | 209,508,106,240 ✓ | 79,921 ✓ | 0.0974 ✓ | 397,596 ✓ | 22.378200 ✓ |
| (b) Tool&Agent | 8,483,936,665,600 ✓ | 3,236,365 ✓ | 0.4037 ✓ | 1,174,042 ✓ | 22.885048 ✓ |
| (d) Tool&Agent | 85,844,295,680 ✓ | 32,747 ✓ | 0.3650 ✓ | 1,061,727 ✓ | 22.429517 ✓ |

### Conversation

| metric | (b) original | (d) original | (b) tailshare | (d) tailshare |
|---|---|---|---|---|
| prefix hit rate (block-level) | **0.1643** | 0.0974 | **0.1202** | 0.0869 |
| reuse hit blocks | 670,585 | 397,596 | 490,322 | 354,571 |
| **HBF bytes / request** | **3,370,675,339** | 41,901,621 | **3,620,584,554** | 15,721,300 |
| PCM bytes / request | 1,722,617,954 | 1,987,125,969 | 1,847,572,562 | 1,980,261,466 |
| achieved batch / scheduler | 23.1679 | 22.3782 | 23.2212 | 22.3953 |
| peak HBM occupancy | 85.66% | 90.54% | 86.45% | 96.64% |
| workload footprint (distinct blocks) | 2,695,220 | 2,695,220 | 2,620,973 | 2,620,973 |

| derived | original | tailshare |
|---|---|---|
| reuse retained (d vs b) | 59.3% | **72.3%** |
| **`if_read` hit-rate cost** | **−40.7%** | **−27.7%** |
| HBF write reduction (d vs b) | 98.76% | 99.57% |
| HBF lifetime ratio (work-normalized, b ÷ d) | **80.4×** | **230.3×** |

### Tool&Agent

| metric | (b) original | (d) original | (b) tailshare | (d) tailshare |
|---|---|---|---|---|
| prefix hit rate (block-level) | **0.4037** | 0.3650 | **0.3769** | 0.3643 |
| reuse hit blocks | 1,174,042 | 1,061,727 | 1,096,087 | 1,059,578 |
| **HBF bytes / request** | **1,696,787,333** | 17,168,859 | **1,814,623,158** | 8,518,107 |
| PCM bytes / request | 885,673,951 | 1,001,932,194 | 944,591,864 | 990,484,365 |
| achieved batch / scheduler | 22.8850 | 22.4295 | 22.9675 | 22.6307 |
| peak HBM occupancy | 75.12% | 79.49% | 79.52% | 78.43% |
| workload footprint (distinct blocks) | 1,397,479 | 1,397,479 | 1,332,772 | 1,332,772 |

| derived | original | tailshare |
|---|---|---|
| reuse retained (d vs b) | 90.4% | **96.7%** |
| **`if_read` hit-rate cost** | **−9.6%** | **−3.3%** |
| HBF write reduction (d vs b) | 98.99% | 99.53% |
| HBF lifetime ratio (work-normalized, b ÷ d) | **98.8×** | **213.0×** |

### Batch mismatch — flagged

| comparison | Conversation | Tool&Agent |
|---|---|---|
| (d) vs (b), original | −3.41% | −1.99% |
| (d) vs (b), tailshare | −3.56% | −1.47% |
| (b) tailshare vs (b) original | **+0.23%** | **+0.36%** |
| (d) tailshare vs (d) original | **+0.08%** | **+0.90%** |

The (d)-vs-(b) gap of 2–3.6% is the same one PHASE 9 recorded and is inherent to
the configs. The **trace-variant** deltas are all under 1%, so the two workloads
are being compared at effectively the same operating point; none of the shifts
below can be attributed to batch.

### Result: tail-sharing is not a fix — it makes the workload worse

**Numbers first.** Against the original trace, on config (b):

| | prefix hit rate | HBF bytes / request |
|---|---|---|
| Conversation | 0.1643 → 0.1202 = **−26.9%** | 3,370,675,339 → 3,620,584,554 = **+7.41%** |
| Tool&Agent | 0.4037 → 0.3769 = **−6.6%** | 1,696,787,333 → 1,814,623,158 = **+6.94%** |

**Tail-sharing lowered the hit rate and raised the write volume**, which is the
opposite of what it was constructed to do, even though it *reduced* the distinct
block footprint by 2.75% / 4.63%.

The mechanism: distinct-block count is a **set** measure and does not capture
chain structure, but a prefix cache needs a **contiguous** match from position 0.
The rule picks each `R'`'s predecessor independently, so two siblings that
previously shared an identical chain can be rewritten against *different*
predecessors and diverge earlier than they did before. Fewer store hits then mean
the `_aligned` load/save exclusion (`mooncake_store.py:131-137`) fires less
often, more requests save, and write volume rises — which is exactly the +7%
seen. The 79.6% / 49.2% of rewrites resting on a single shared 512-token chunk
is where most of that damage comes from.

**Conclusion, stated as a sensitivity result and nothing more:**

1. **PHASE 9 stands as the headline.** The original traces remain the reference
   workload; nothing here supersedes a PHASE 9 number.
2. **`--share_partial_tail` as specified is not usable as a correction** for the
   STEP 2 limitation. It is committed, default off, and its files are kept, but
   its results should be read as "a different, worse workload", not "the same
   workload with an artefact removed". A narrower rule — restricted to pairs with
   a genuinely multi-chunk stem, and applied consistently across siblings — would
   be needed to test the STEP 2 limitation properly.
3. **The `if_read` hit-rate cost is the most workload-sensitive number in this
   thesis.** It moves −40.7% → −27.7% (Conversation) and −9.6% → −3.3%
   (Tool&Agent) under a trace variant that changes 2–5% of the footprint. It
   should be quoted with that sensitivity attached, never as a constant.
4. **The HBF lifetime ratio is even more sensitive** — 80.4× → 230.3× and
   98.8× → 213.0× — because (d)'s HBF writes are near zero, so small absolute
   changes move the ratio by a large factor. **Work-normalized bytes per request
   is the robust figure; the ratio is not**, and PHASE 8/9's lifetime multipliers
   should carry that caveat.

### Settings

| | |
|---|---|
| runs | `tmp/phase12/runs/p13_{b,d}_{conv,ta}_{or,ts}` (8) |
| harness | `scripts/phase12/run_one.py`, `WORK=tmp/phase12` |
| new scripts | `generated_share.py`, `tail_divergence.py`, `build_trace.py --share_partial_tail` |
| tests | **190** (177 + 13) |
| simulator change | **none** — no TokenSim source file was modified in PHASE 13 |

---

# PHASE 14 — Fixes for known simulation problems

Append-only. Nothing above is edited; corrections to older entries appear below
as new notes that point back to them. **PHASE 9 STEP 3 (cap 48) remains the
headline.** Every behaviour change this phase adds sits behind a config flag
whose default is stock.

Branch **`phase14-fixes`**, cut from `main` at **`ced3806`**.

## Provenance

| file | md5 | matches PHASE 12 |
|---|---|---|
| `conversation_trace.jsonl` (raw) | `bc2f368504912f72f59caee90b5f2e46` | ✓ |
| `toolagent_trace.jsonl` (raw) | `53cf2bc5eff72e9953348bda80028c93` | ✓ |
| `conv_5000.jsonl` (built) | `5a19d814562e650fe6e44e4234a4793c` | ✓ |
| `toolagent_5000.jsonl` (built) | `c6cd3b0c8022eebb88358719a2596f1c` | ✓ |
| `data/hardware/hardware_models_h3b200.json` | `9b4cba823dd4158d9bd3efe16bf74491` | ✓ |

The trace files were present in `tmp/phase12/` and did not need rebuilding.
PHASE 14 runs live in `tmp/phase14/runs/` (gitignored); the trace files there
are symlinks to `tmp/phase12/`.

## STEP 0 — Reproduction gate (no code change)

Configs (b) and (d), cap 48, 30 QPS, W = 5,000, both traces, on `ced3806` with
`scripts/phase12/run_one.py` unchanged. Model and platform as PHASE 12 Settings:
`LLaMa2-70B-GQA` (Nhead 64, KV heads 8, Nlayer 80, Dmodel 8192), **FP16**,
163,840 B/token at TP2, block **2,621,440 B**, TP2/PP1/DP4, `H3-B200-KVONLY`,
PCM pool 142,213 blk, HBF pool 10,066,329 blk, seed 0. Workload: Mooncake
FAST'25 traces, **a stand-in for CAG**.

| run | HBF write bytes | HBF blocks | hit rate | reuse hit blk | batch/sched |
|---|---|---|---|---|---|
| (b) Conversation | 16,853,376,696,320 ✓ | 6,429,053 ✓ | 0.1643 ✓ | 670,585 ✓ | 23.167889 ✓ |
| (d) Conversation | 209,508,106,240 ✓ | 79,921 ✓ | 0.0974 ✓ | 397,596 ✓ | 22.378200 ✓ |
| (b) Tool&Agent | 8,483,936,665,600 ✓ | 3,236,365 ✓ | 0.4037 ✓ | 1,174,042 ✓ | 22.885048 ✓ |
| (d) Tool&Agent | 85,844,295,680 ✓ | 32,747 ✓ | 0.3650 ✓ | 1,061,727 ✓ | 22.429517 ✓ |

**All 20 values match the report.md "Matched batch (cap 48)" table to the last
digit.** Beyond the gate, each of the four result files is identical — after
dropping `simulator_wall_time` and `mooncake_pool_keys` — to its PHASE 13
counterpart (`tmp/phase12/runs/p13_{b,d}_{conv,ta}_or`). Tests: 190 pass.

## STEP 1a — LIMITATION: rank-scoped keys in one shared, rank-blind pool

Re-verified in source at `ced3806`. Two facts, which pull in opposite
directions.

**1. Every pool key carries the writer's tp/pp/dp rank.**

| step | file:line |
|---|---|
| each worker gets a rank from its global index | `TokenSim/config/cluster_config.py:83` → `ParallelRankInfo.from_global_rank` (`TokenSim/config/parallel_config.py:139-164`) |
| the engine hands it to its cache config | `TokenSim/llm/llm_engine.py:176`, `:192` |
| the connector reads it | `TokenSim/kv_transfer/connectors/mooncake_store.py:43` |
| and builds every request's keys with it | `mooncake_store.py:496-511` → `pool_keys_for_request` (`TokenSim/mooncake/pool_key.py:76-106`) |
| the key string embeds `tp_rank`, `pp_rank`, `dp_rank` | `pool_key.py:13-23` (`KeyMetadata`), `:31-44` (`to_string`), `:47-73` (`build_key_metadata`) |

Consequence: a prefix saved by one DP group can never be hit by a request routed
to another DP group, even when the token content is identical. (Within a DP
group the two TP ranks each key their own head shard — that part is physically
right, since block bytes are per-rank shards.) In `s0_b_conv`, DP placement is
1,250 / 1,250 / 1,250 / 1,250 requests.

**2. But there is only one LRU and one capacity counter for all eight workers.**

| step | file:line |
|---|---|
| every worker's connector shares one `engine_id` | `TokenSim/config/kv_transfer_config.py:26-27` (one uuid per run), `:61` (`with_worker_defaults` copies it), called at `llm_engine.py:235-242` |
| so all eight resolve to the same service | `TokenSim/mooncake/service.py:33-39` (singleton keyed on `engine_id` + config fingerprint) |
| which owns one store with one `memory_capacity_bytes`, one `memory_used_bytes`, one `objects` map and one `memory_lru` | `TokenSim/mooncake/store.py:58-67` |
| admission frees room against that one counter | `_ensure_memory_capacity`, `store.py:262-281` (loop at `:277`) |
| by popping the global LRU head, whoever owns it | `_evict_memory_lru`, `store.py:283-284` (`popitem(last=False)`) |
| the offload tier has the same shape: one `capacity_blocks`, one `_lru` | `TokenSim/mooncake/ssd.py:36-40` |

Consequence: a block admitted by rank *i* can evict a block that only rank *j*
can ever read, and one rank's burst can occupy capacity the others' working sets
need. A real deployment of 8 GPUs with a PCM stack each would have per-stack
capacity — the per-GPU design values in `CLAUDE.md` are exactly that — not one
pooled LRU.

**Status: unquantified.** Neither effect has been measured. Fact 1 lowers hit
rates relative to a content-addressed cluster-wide cache (another reason, beside
PHASE 13 STEP 2, that every hit rate here is a floor). Fact 2 lets capacity
migrate between ranks, which a per-stack design would not allow; its sign on
write volume is not known. Both apply equally to every config, so between-config
comparisons at fixed workload are affected only through second-order
interactions with policy. STEP 3 of this phase measures fact 2.

## STEP 1b — NOTE: "~10^8 endurance" is unsourced

**Points back to** `findings.md` ENDURANCE/POWER RESCOPE, "Go/no-go for NVSim"
(~line 1431: *"which PCM's ~10^8 endurance absorbs"*), and the PHASE 4/5
"Go/no-go for NVSim" (~line 1892: *"PCM's ~10⁸ endurance should absorb it"*).

**Both phrases are unsourced. ~10^8 is not a value.** No document in this tree
gives a PCM endurance figure, and none was used in any computation — no number
in this file depends on it. Per `CLAUDE.md`, **PCM endurance stays OPEN.** Read
both sentences as "PCM's endurance is assumed high" and nothing more.

The LaTeX draft carries the same unsourced claim in prose (see report.md,
Pending LaTeX sync).

## STEP 1c — NOTE: which timing values are assumed

**PCM media timing is assumed, not sourced.** Config (c)/(d) set
`memory_media = pcm` and do not override the timing, so every PCM run in this
file uses the dataclass defaults at `TokenSim/mooncake/config.py:81-84`:

| knob | default | status |
|---|---|---|
| `pcm_read_latency_us` | 0.1 | **assumed** |
| `pcm_write_latency_us` | 1.0 | **assumed** |
| `pcm_read_bw_gbps` | 100.0 (GiB/s, `store.py` divides by 2^30) | **assumed** |
| `pcm_write_bw_gbps` | 20.0 (GiB/s) | **assumed** |

The 5× write/read asymmetry was already flagged as assumed at ~line 1434 and
~line 1883. This note extends that to all four values. Git history in this tree
starts at `4e9b9fe` (the submodule conversion), so where the defaults came from
cannot be traced.

**HBF write timing is TokenSim's stock SSD default, not a sourced HBF figure.**
`ssd_write_latency_us = 200` and `ssd_write_bw_gbps = 3.0` (GiB/s) are the
`MooncakeConfig` defaults (`config.py:48`, `:46`). `run_one.py` passes them
explicitly with the same values. PHASE 8 (~lines 2784-2785) and `CLAUDE.md` call
them "NAND-realistic". That describes intent; it is not a citation. H3 gives HBF
read bandwidth (8 TB/s) and the LHB argument, but no HBF write latency or write
bandwidth. **Both write values are unsourced placeholders.**

(The read side differs: `ssd_read_latency_us = 0.1` and `ssd_read_bw_gbps =
7451` are documented in PHASE 8 as a modelling assumption derived from the
paper's 8 TB/s and LHB argument. The stock SSD read defaults are 100 µs and
7.0 GiB/s, `config.py:47`, `:45`.)

## STEP 1d — report.md

`Documents/Report/report.md` "What is established" now quotes the PHASE 9
STEP 3 cap-48 batch-matched figures, with bytes/request next to every lifetime
ratio. The cap-256 table is labelled as saturated, with its batch gaps. Stale
LaTeX figures are listed under "Pending LaTeX sync" but not fixed. No number in
this file changed.

## STEP 1 — addenda (appended after review; nothing above is edited)

### 1e — CORRECTION: the LaTeX headline does not match its own table

**Points back to** `Documents/Latex/pcm-kv-micro.tex` (abstract l.59–63,
`sec:headline` l.379–382, Conclusion l.609–611). Those passages quote
**×90.3 / ×86.4**. Those are the PHASE 8 multipliers (×90.31 / ×86.39, PHASE 8
STEP 3, batch-unmatched). But `tab:headline` in the same draft is the PHASE 9
STEP 3 (revised) **cap-256** table, and its own bytes/request give different
ratios.

Recomputed from result files. The cap-256 (b) and (d) runs were rerun for this
check (`tmp/phase14/runs/s1_{b,d}256_{conv,ta}`), because the PHASE 9 files no
longer exist on this machine. All four reproduce the recorded cap-256 HBF write
bytes exactly (16,938,274,652,160 / 212,252,753,920 / 8,479,548,375,040 /
146,022,072,320).

| | (b) HBF B/req | (d) HBF B/req | **b ÷ d** | (d) vs (b) batch |
|---|---|---|---|---|
| Conversation, cap 256 | 3,387,654,930 | 42,450,551 | **×79.80** | −4.72% |
| Tool&Agent, cap 256 | 1,695,909,675 | 29,204,414 | **×58.07** | −7.68% |
| Conversation, cap 48 | 3,370,675,339 | 41,901,621 | **×80.44** | −3.41% |
| Tool&Agent, cap 48 | 1,696,787,333 | 17,168,859 | **×98.83** | −1.99% |

**Confirmed:** ×79.8 and ×58.1 at cap 256, not ×90.3 / ×86.4. The draft's prose
and its table come from different phases. **Canonical figure: PHASE 9 STEP 3
cap 48, ×80.4 / ×98.8**, always quoted with the bytes/request it is computed
from. The LaTeX is not edited here; it is listed under Pending LaTeX sync in
report.md.

### 1f — GAP: "concurrency-invariant" is shown only for (b) and (c)

**Points back to** PHASE 9 STEP 4 (~lines 3636–3700). The sweep ran configs
**(b) and (c) only**. Its conclusion, "Every lifetime number in Phases 8 and 9 —
×90.31, ×86.39, ×2.05, ×2.09 — holds unchanged from 25% to 99% HBM occupancy"
(~line 3700), therefore covers ×2.05 / ×2.09 and **not** the (b ÷ d) figures.
LaTeX `sec:batch` and the Conclusion inherit the same over-reach.

Measured, config (d), from the result files above:

| | cap 48 HBF B/req (batch) | cap 256 HBF B/req (batch) | change |
|---|---|---|---|
| Conversation (d) | 41,901,621 (22.38) | 42,450,551 (41.55) | **+1.31%** |
| Tool&Agent (d) | 17,168,859 (22.43) | 29,204,414 (44.72) | **+70.10%** |
| *for reference:* Conversation (b) | 3,370,675,339 (23.17) | 3,387,654,930 (43.61) | +0.50% |
| *for reference:* Tool&Agent (b) | 1,696,787,333 (22.89) | 1,695,909,675 (48.44) | −0.05% |

**Config (d)'s write volume is not shown to be invariant to concurrency.** On
Tool&Agent it rises by 70% between two points about 2× apart in achieved batch.
Two points cannot establish a curve, and this is recorded, not explained. Until
it is measured, every "concurrency-invariant" statement is supported for (b)
and (c) only. That includes ~line 3700, LaTeX `sec:batch` and the Conclusion.
None of that old text is edited.

### 1g — addition to 1a: the shared HBF LRU never evicts

`mooncake_ssd_eviction_count = 0` in every result file on this machine (the 8
PHASE 13 runs, `s0_*`, `s1_*`), and PHASE 9 STEP 3 and PHASE 12 recorded the
same in their conservation checks. At 10,066,329 blocks (24 TiB) the offload
tier's single LRU (`ssd.py:36-40`, `:73`) never reaches capacity on these
workloads. **So cross-rank eviction (fact 2 of 1a) can only occur in the PCM
memory tier.** STEP 3 counts it there.

## STEP 2 — Do the endurance numbers depend on timing at all?

**Pre-registered rule, fixed before any run:** if HBF bytes/request and PCM
(memory-tier) bytes/request change by < 5% (the PHASE 12 threshold) for both
(b) and (d) across every arm below, timing does not contaminate the
write-volume results.

**Result: the rule FAILS, and it fails only on (d)'s HBF stream.** The largest
change is **+57.03%** (T&A (d), `pcm_write_bw_gbps` 0.25×). (b)'s HBF stream
moves ≤ 0.18% and every memory-tier stream moves ≤ 0.54%. Numbers first; the
reading follows them.

### The flags

| flag | values | where |
|---|---|---|
| **`store_timing_mode`** | **`charged`** (default, stock) \| `free` | `mooncake/config.py:95`, validated `:174-179` |
| **`count_aligned_save_skips`** | **`false`** (default) \| `true` — instrumentation only | `mooncake/config.py:99` |

`free` returns exactly `0.0` from every store-time source, and only from those:

- offload read and write (`mooncake/ssd.py:110`, `:118`)
- PCM memory-tier media (`mooncake/store.py:382`)
- every transfer-engine transfer (`mooncake/transfer_engine.py:31-37`, set at
  `:88`), which covers save and load interconnect time

These are branches, not scale factors, so the `charged` path computes the same
floats as before. Byte, block, hit and LRU accounting is untouched. A non-stock
mode is recorded in `mooncake_offload_profiles` (`store.py:438-439`). The stock
profile is unchanged.

`count_aligned_save_skips` counts saves skipped by the aligned rule
(`kv_transfer/connectors/mooncake_store.py:265-266` → `_count_save_skip`,
`:191-201`). The rule is that a step which loads from the store does not save.
Two counters result:

- `save_skip_requests`: how many saves were skipped.
- `save_skip_missing_blocks`: the full prompt blocks the store did not hold at
  that moment, i.e. what the skipped save would have written. It uses
  `missing_indices`, which touches no hit/miss statistic and no LRU heat.

Both are merged into `result_dict` after `asdict()` through
`OPTIONAL_MOONCAKE_KEY_PREFIXES` (`util/results.py:400`), as in PHASE 12.

**Byte identity at default, verified.** All new flags are at default, compared
against `ced3806` (`s0_*`) with the PHASE 12 strip():

```
diff <(strip s0_b_conv/result_30.0.json) <(strip s2_id_b_conv/result_30.0.json)   -> no output
diff <(strip s0_b_ta/result_30.0.json)   <(strip s2_id_b_ta/result_30.0.json)     -> no output
diff <(strip s0_d_conv/result_30.0.json) <(strip s2_id_d_conv/result_30.0.json)   -> no output
diff <(strip s0_d_ta/result_30.0.json)   <(strip s2_id_d_ta/result_30.0.json)     -> no output
```

**The counter is behaviourally stock.** `s2_ref_*` (counter on) against
`s2_id_*`: the only diff in all four is the two added keys,
`save_skip_missing_blocks` and `save_skip_requests`.

**Tests: 205 pass.** That is 190 pre-existing and unchanged, plus 15 new in
`tests/test_store_timing_mode.py`. They cover parse/reject, the zero-time
property on both tiers and on transfers, byte/block counters equal between the
two modes, connector load and save time at zero, the profile key appearing only
off stock, the counter's default-off export, its counts on a constructed
load-then-skip case, and its non-perturbation of the store.

### Runs

Configs (b) and (d), cap 48, 30 QPS, W = 5,000, both traces. Everything not
named in an arm is exactly the STEP 0 configuration, so the model, FP16,
block 2,621,440 B, TP2/DP4, pools 142,213 / 10,066,329 blocks and seed 0 are as
STEP 0. All arms except `id_*` run with `count_aligned_save_skips = true`. Every
Δ is against the `ref` row of the same config and trace.

- **HBF read 100 µs / 7.0 GiB/s** is the in-tree **stock SSD default, not HBF**
  (`config.py:45,47`; used at findings ~line 990).
- **The PCM write arms are sensitivity around an assumption, not values.** The
  assumed `pcm_write_latency_us = 1.0` and `pcm_write_bw_gbps = 20.0` were each
  scaled 0.25× and 4×, **one at a time** (that is how "0.25x and 4x of the
  current assumed values" was read). Under (b) `memory_media = dram`, so these
  knobs are inert by construction (`store.py:380-383`). The (b) rows confirm
  it: identical to `ref`.
- "mem-tier B/req" is `mooncake_memory_write_bytes / 5000`. Under (d) that is
  PCM. Under (b) it is the DRAM-media memory tier.
- `notdone = 0`, `preemption_count = 0` in all 32 runs.

### (b) Conversation

| arm | HBF B/req | Δ | mem-tier B/req | Δ | hit rate | reuse hit blk | batch/sched | Δ batch | saves skipped (req / missing blk) | save_wait s |
|---|---|---|---|---|---|---|---|---|---|---|
| charged (reference) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1679 | +0.00% | 451 / 113,297 | 6678.3 |
| store_timing_mode=free | 3,376,638,591 | +0.18% | 1,725,599,580 | +0.17% | 0.1645 | 671,263 | 22.5048 | -2.86% | 424 / 108,739 | 0.0 |
| HBF read 100 us / 7.0 GiB/s (stock SSD default, not HBF) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1144 | -0.23% | 451 / 113,297 | 6678.3 |
| pcm_write_latency_us 0.25 (0.25x, sensitivity) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1679 | +0.00% | 451 / 113,297 | 6678.3 |
| pcm_write_latency_us 4.0 (4x, sensitivity) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1679 | +0.00% | 451 / 113,297 | 6678.3 |
| pcm_write_bw_gbps 5.0 (0.25x, sensitivity) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1679 | +0.00% | 451 / 113,297 | 6678.3 |
| pcm_write_bw_gbps 80.0 (4x, sensitivity) | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | 670,585 | 23.1679 | +0.00% | 451 / 113,297 | 6678.3 |

### (b) Tool&Agent

| arm | HBF B/req | Δ | mem-tier B/req | Δ | hit rate | reuse hit blk | batch/sched | Δ batch | saves skipped (req / missing blk) | save_wait s |
|---|---|---|---|---|---|---|---|---|---|---|
| charged (reference) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8850 | +0.00% | 155 / 38,882 | 3363.7 |
| store_timing_mode=free | 1,695,299,404 | -0.09% | 884,929,987 | -0.08% | 0.4041 | 1,175,233 | 22.3338 | -2.41% | 154 / 40,301 | 0.0 |
| HBF read 100 us / 7.0 GiB/s (stock SSD default, not HBF) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8756 | -0.04% | 155 / 38,882 | 3363.7 |
| pcm_write_latency_us 0.25 (0.25x, sensitivity) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8850 | +0.00% | 155 / 38,882 | 3363.7 |
| pcm_write_latency_us 4.0 (4x, sensitivity) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8850 | +0.00% | 155 / 38,882 | 3363.7 |
| pcm_write_bw_gbps 5.0 (0.25x, sensitivity) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8850 | +0.00% | 155 / 38,882 | 3363.7 |
| pcm_write_bw_gbps 80.0 (4x, sensitivity) | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | 1,174,042 | 22.8850 | +0.00% | 155 / 38,882 | 3363.7 |

### (d) Conversation

| arm | HBF B/req | Δ | mem-tier B/req | Δ | hit rate | reuse hit blk | batch/sched | Δ batch | saves skipped (req / missing blk) | save_wait s |
|---|---|---|---|---|---|---|---|---|---|---|
| charged (reference) | 41,901,621 | +0.00% | 1,987,125,969 | +0.00% | 0.0974 | 397,596 | 22.3782 | +0.00% | 23 / 4,546 | 732.6 |
| store_timing_mode=free | 38,407,242 | -8.34% | 1,993,910,256 | +0.34% | 0.0939 | 383,177 | 22.4235 | +0.20% | 21 / 3,291 | 0.0 |
| HBF read 100 us / 7.0 GiB/s (stock SSD default, not HBF) | 41,465,414 | -1.04% | 1,987,467,805 | +0.02% | 0.0974 | 397,596 | 22.3934 | +0.07% | 23 / 4,546 | 731.9 |
| pcm_write_latency_us 0.25 (0.25x, sensitivity) | 38,445,515 | -8.25% | 1,993,549,545 | +0.32% | 0.0955 | 389,852 | 22.4416 | +0.28% | 24 / 4,591 | 725.2 |
| pcm_write_latency_us 4.0 (4x, sensitivity) | 38,440,796 | -8.26% | 1,991,415,169 | +0.22% | 0.0976 | 398,298 | 22.4912 | +0.50% | 22 / 4,890 | 738.7 |
| pcm_write_bw_gbps 5.0 (0.25x, sensitivity) | 37,316,198 | -10.94% | 1,988,093,805 | +0.05% | 0.0954 | 389,354 | 22.7421 | +1.63% | 28 / 7,072 | 2112.7 |
| pcm_write_bw_gbps 80.0 (4x, sensitivity) | 40,693,662 | -2.88% | 1,986,114,093 | -0.05% | 0.0982 | 400,828 | 22.6286 | +1.12% | 22 / 4,592 | 383.1 |

### (d) Tool&Agent

| arm | HBF B/req | Δ | mem-tier B/req | Δ | hit rate | reuse hit blk | batch/sched | Δ batch | saves skipped (req / missing blk) | save_wait s |
|---|---|---|---|---|---|---|---|---|---|---|
| charged (reference) | 17,168,859 | +0.00% | 1,001,932,194 | +0.00% | 0.3650 | 1,061,727 | 22.4295 | +0.00% | 4 / 2,942 | 361.8 |
| store_timing_mode=free | 20,088,095 | +17.00% | 1,006,519,714 | +0.46% | 0.3677 | 1,069,407 | 22.1144 | -1.40% | 4 / 319 | 0.0 |
| HBF read 100 us / 7.0 GiB/s (stock SSD default, not HBF) | 17,907,057 | +4.30% | 1,001,193,996 | -0.07% | 0.3650 | 1,061,727 | 22.4782 | +0.22% | 4 / 2,942 | 363.0 |
| pcm_write_latency_us 0.25 (0.25x, sensitivity) | 24,299,176 | +41.53% | 997,192,630 | -0.47% | 0.3671 | 1,067,775 | 22.4149 | -0.07% | 5 / 3,008 | 372.6 |
| pcm_write_latency_us 4.0 (4x, sensitivity) | 23,074,439 | +34.40% | 997,790,319 | -0.41% | 0.3672 | 1,068,063 | 22.5458 | +0.52% | 5 / 3,008 | 377.6 |
| pcm_write_bw_gbps 5.0 (0.25x, sensitivity) | 26,959,938 | +57.03% | 996,674,109 | -0.52% | 0.3668 | 1,066,719 | 22.4234 | -0.03% | 4 / 319 | 1075.2 |
| pcm_write_bw_gbps 80.0 (4x, sensitivity) | 23,460,315 | +36.64% | 996,548,805 | -0.54% | 0.3676 | 1,069,151 | 22.3224 | -0.48% | 5 / 952 | 198.2 |


`save_wait` is listed only because it is the timing variable each arm changes.
It is **not** a result.

### (b) → (d) at matched arm

Each arm's (d) is compared with the same arm's (b).

| arm | Conv reduction | Conv b ÷ d | Conv (d) vs (b) batch | T&A reduction | T&A b ÷ d | T&A (d) vs (b) batch |
|---|---|---|---|---|---|---|
| charged (reference) | 98.76% | ×80.4 | −3.41% | 98.99% | ×98.8 | −1.99% |
| free | 98.86% | ×87.9 | −0.36% | 98.82% | ×84.4 | −0.98% |
| HBF read stock SSD default | 98.77% | ×81.3 | −3.12% | 98.94% | ×94.8 | −1.74% |
| PCM write lat 0.25× | 98.86% | ×87.7 | −3.14% | 98.57% | ×69.8 | −2.05% |
| PCM write lat 4× | 98.86% | ×87.7 | −2.92% | 98.64% | ×73.5 | −1.48% |
| PCM write bw 0.25× | 98.89% | ×90.3 | −1.84% | 98.41% | ×62.9 | −2.02% |
| PCM write bw 4× | 98.79% | ×82.8 | −2.33% | 98.62% | ×72.3 | −2.46% |
| **range** | **98.76–98.89%** | **×80.4–×90.3** | | **98.41–98.99%** | **×62.9–×98.8** | |

### Batch mismatch — flagged

- **Against `ref`:** every arm is within 3%. The largest gaps are (b) `free`,
  **−2.86%** (Conv) and **−2.41%** (T&A), and (d) Conv `pbw025`, +1.63%.
- **(d) vs (b), same arm:** −0.36% to −3.41% (Conv) and −0.98% to −2.46% (T&A).
  This is the structural gap PHASE 9 recorded.
- **(d) HBF shifts vs (d) batch gaps against `ref`:**
  - **T&A:** batch gaps are at most 1.40% in magnitude, against HBF shifts of
    +4.30% to +57.03%. The largest shift, `pbw025` at +57.03%, has a batch gap
    of −0.03%. Batch does not explain these.
  - **Conversation:** batch gaps run +0.07% to +1.63%, against HBF shifts of
    −1.04% to −10.94%. In `pbw4` the batch gap (+1.12%) is about a third of the
    HBF shift (−2.88%), so a batch contribution to that one arm cannot be ruled
    out. On T&A (d) the arm with the
  largest HBF shift (`pbw025`, +57.03%) has a batch gap of −0.03%.

### Reading

1. **(b)'s write volume is timing-independent.** HBF B/req moves +0.18% /
   −0.09% when all store time is removed, and 0.00% in every other arm. The
   memory-tier stream behaves the same. **Every (b) and (c)-type number — the
   write-through/`always` stream, and the ~51% write-back halving — is not
   contaminated by timing.** (c) was not run here. This rests on (b), whose
   `always` stream has the same structure.
2. **(d)'s PCM (memory-tier) write volume is timing-independent**: ≤ 0.54% in
   every arm on both traces.
3. **(d)'s HBF write volume is NOT timing-independent.** Conversation: −1.04%
   to −10.94%. Tool&Agent: +4.30% to +57.03%. That stream is the `if_read`
   demotes: blocks read at least once while resident in PCM. Whether a block is
   read before it is evicted depends on how requests interleave with evictions,
   and timing changes that interleaving.
4. **The response is non-monotone. It looks like sensitivity to any
   perturbation, not a timing law.** On T&A, PCM write latency at **0.25× and
   at 4×** both *raise* (d) HBF B/req (+41.53% and +34.40%). On Conversation the
   same two arms both lower it, by an identical −8.25% / −8.26%. Making PCM
   faster and making it slower move the number the same way. So no direction can
   be attributed to "timing" here.
5. **The PHASE 9 cap-48 reference sits at an edge of the spread on both
   traces.**
   - Conversation (d) reference: 41,901,621 B/req, the **highest** of the seven
     arms.
   - Tool&Agent (d) reference: 17,168,859 B/req, the **lowest** of the seven.
   - So the T&A headline ratio ×98.8 is the **maximum** of a ×62.9–×98.8 range.
     The Conversation ×80.4 is the **minimum** of ×80.4–×90.3.
   - This fits STEP 1f, where T&A (d) moved +70.1% between cap 48 and cap 256.
6. **What survives.** The *reduction* (b) → (d) stays within
   **98.41–98.99%** in every arm. The bytes/request (d) writes to HBF stay
   between 17.2 M (17,168,859, T&A reference) and 41.9 M (41,901,621, Conv
   reference), against 1.70–3.38 G under (b). The **ratio** is the fragile figure, as PHASE 13 already
   said.

**Consequence for quoting.** (d)'s HBF bytes/request, and anything divided by
it, should be quoted as a **range over perturbations**, not as one run's value:

- Conversation: 37.3–41.9 M B/req (×80.4–×90.3)
- Tool&Agent: 17.2–27.0 M B/req (×62.9–×98.8)

The single-run figures ×80.4 / ×98.8 are not wrong, but they carry no error bar.
A seed or arrival-jitter replicate set would separate perturbation sensitivity
from timing proper. **It is not run here, as it was not asked for.**

### Settings

| | |
|---|---|
| code commit | **`9e8b4d1`** (flags, counter, tests, driver) on `phase14-fixes`; STEP 1 docs commit **`1797110`** |
| driver / table | `scripts/phase14/step2.sh`, `scripts/phase14/step2_table.py` |
| runs | `tmp/phase14/runs/s2_{id,ref,free,ssd,plat025,plat4,pbw025,pbw4}_{b,d}_{conv,ta}` (32) |
| traces / hardware | md5s as in Provenance above, unchanged |
| hardware numbers | **none invented or changed.** The stock SSD read values are in-tree defaults. The PCM arms are multiples of in-tree assumed defaults, labelled as such |

### STEP 2 — verdict, stated plainly (appended after review)

**The pre-registered rule failed.**

| stream | max \|Δ\| vs charged reference, all arms | rule (< 5%) |
|---|---|---|
| (b) HBF B/req, both traces | 0.18% | **pass** |
| (b) memory-tier B/req, both traces | 0.17% | **pass** |
| (d) PCM B/req, both traces | 0.54% | **pass** |
| **(d) HBF B/req, Conversation** | **10.94%** | **FAIL** |
| **(d) HBF B/req, Tool&Agent** | **57.03%** | **FAIL** |

**How much is moving, in absolute terms.** (d)'s HBF stream is small, so a
large percentage is a small number of blocks:

| | (d) HBF write blocks, min – max over the 7 arms | spread | as % of (b)'s HBF stream (ref) |
|---|---|---|---|
| Tool&Agent | 32,747 (ref) – 51,422 (`pbw025`) | **~18,700 blocks** (18,675) | **0.58%** of 3,236,365 |
| Conversation | 71,175 (`pbw025`) – 79,921 (ref) | **~8,700 blocks** (8,746) | **0.14%** of 6,429,053 |

**What holds and what does not:**

- The (b) → (d) **reduction holds**: **98.41–98.99%** (T&A) and
  **98.76–98.89%** (Conv), in every arm.
- The **ratio does not hold**: ×62.9–×98.8 (T&A) and ×80.4–×90.3 (Conv).

**PHASE 9's cap-48 T&A (d) value (32,747 blocks, 17,168,859 B/req) is the
lowest-write arm measured.** So **×98.8 is the top of its range**, not a central
value. The cap-48 Conversation (d) value is the highest-write arm, so ×80.4 is
the bottom of its range.

Whether this is a timing effect or run-to-run sensitivity is STEP 2b.

## STEP 2b — Replicates

### 2b(a) — `--random_seed` is inert on these runs

(d) Tool&Agent, cap 48, seeds 0, 1 and 2 (`scripts/phase14/run_one.py`, which is
the PHASE 12 command with the seed exposed). The stripped results are
**byte-identical** across all three, and identical to `s0_d_ta`: md5
`78ba8472d06c6393192290b386eadf08` for all four.

The seed reaches only three places:

- the synthetic workload generator (`TokenSim/workload/loaders.py:58-65`)
- MoE expert routing (`TokenSim/latency/roofline.py:51`)
- the per-worker latency-backend seed (`TokenSim/llm/llm_engine.py:226`)

The qwen_jsonl loader draws no random numbers, and `LLaMa2-70B-GQA` is dense.
**The seed is inert, so it cannot provide replicates.**

### 2b(b) — `--arrival_jitter_ms` / `--jitter_seed`

| flag | default | where |
|---|---|---|
| `--arrival_jitter_ms` | **0.0 (stock)** | `benchmark.py`, applied in `apply_arrival_jitter` (`TokenSim/workload/loaders.py`) |
| `--jitter_seed` | 0 | same |

How the jitter works:

- Each timestamped arrival is offset by `uniform(-j, +j)` ms.
- The draw comes from a private `random.Random(jitter_seed)`, never the global
  generators, one draw per request in file order.
- It is applied on the **trace clock, before** `--trace_target_qps` scaling. The
  simulated offset is therefore j × scale: ×0.1046 on Conversation (native
  3.138 QPS → 30) and ×0.2020 on Tool&Agent (native 6.059 QPS → 30).
- A result below 0 is clamped to 0.
- `j = 0` returns the list untouched and draws nothing.
- These are CLI arguments. The result file records neither, so stock output
  gains no keys.

**j = 0.5 ms was chosen as half the trace's 1 ms timestamp resolution. This is a
methodology choice, not a hardware number.** The premise behind it turned out to
be wrong; see the caveat below.

**Byte identity, verified:** the new code without the flag (`s2b_id_*`) against
`ced3806` (`s0_*`), for (b) and (d) on both traces → **no diff in all four.**
**Tests: 213 pass** (205 + 8 new in `tests/test_arrival_jitter.py`).

### 2b(c) — 40 runs

Five replicates (`--jitter_seed 1..5`, j = 0.5 ms) of (b) and (d), both traces,
caps 48 and 256. Timing is charged, all values stock, and otherwise as STEP 0.
`notdone = 0` in all 40.

| cap | config | trace | HBF B/req mean [min, max] | spread | PCM B/req mean [min, max] | spread | hit rate mean [min, max] | batch/sched mean [min, max] | notdone |
|---|---|---|---|---|---|---|---|---|---|
| 48 | (b) | Conversation | 3,386,787,129 [3,372,843,794, 3,401,276,981] | 0.84% | 1,730,673,849 [1,723,702,182, 1,737,918,775] | 0.82% | 0.1587 [0.1561, 0.1610] | 22.835 [22.359, 23.578] | 0 |
| 48 | (b) | Tool&Agent | 1,737,374,564 [1,697,780,335, 1,756,610,691] | 3.39% | 905,967,567 [886,170,452, 915,585,630] | 3.25% | 0.3916 [0.3863, 0.4001] | 22.338 [21.926, 22.921] | 0 |
| 48 | (d) | Conversation | 33,517,942 [29,316,612, 37,516,476] | 24.46% | 1,994,186,451 [1,988,735,009, 2,002,482,889] | 0.69% | 0.0880 [0.0838, 0.0946] | 22.595 [22.256, 23.127] | 0 |
| 48 | (d) | Tool&Agent | 17,798,948 [13,412,336, 22,079,341] | 48.69% | 1,005,200,291 [1,002,518,348, 1,011,733,234] | 0.92% | 0.3573 [0.3531, 0.3650] | 21.737 [20.997, 22.666] | 0 |
| 256 | (b) | Conversation | 3,404,593,627 [3,393,219,723, 3,417,887,474] | 0.72% | 1,739,577,098 [1,733,890,146, 1,746,224,022] | 0.71% | 0.1584 [0.1557, 0.1605] | 43.341 [42.649, 44.837] | 0 |
| 256 | (b) | Tool&Agent | 1,737,201,340 [1,706,404,872, 1,754,431,750] | 2.76% | 905,880,954 [890,482,721, 914,496,160] | 2.65% | 0.3900 [0.3856, 0.3983] | 47.136 [45.413, 48.526] | 0 |
| 256 | (d) | Conversation | 37,046,085 [25,754,599, 44,074,271] | 49.45% | 1,985,105,258 [1,973,800,141, 2,006,616,900] | 1.65% | 0.0919 [0.0894, 0.0959] | 42.229 [41.539, 43.771] | 0 |
| 256 | (d) | Tool&Agent | 20,741,253 [17,543,725, 22,895,657] | 25.80% | 1,000,530,667 [995,065,594, 1,005,966,066] | 1.09% | 0.3602 [0.3536, 0.3671] | 43.119 [41.006, 44.593] | 0 |

"spread" is (max − min) ÷ mean over the five replicates.

**Batch mismatch — flagged**

- **(d) vs (b), replicate means:** cap 48 is −1.05% (Conv) and −2.69% (T&A).
  Cap 256 is −2.57% (Conv) and **−8.52% (T&A)**. The T&A cap-256 gap is the
  largest in this phase.
- **Jittered vs unjittered, same config and cap:**
  - cap 48: (b) −1.44% / −2.39%, (d) +0.97% / −3.09%
  - cap 256: (b) −0.61% / −2.69%, (d) +1.63% / −3.58%

### CAVEAT, discovered in this step: the jitter is not zero-mean noise

**The "1 ms timestamp resolution" premise is false.** The raw FAST'25
`timestamp` field is in milliseconds, but its values are whole seconds: every
value mod 1000 is 0, 999 or 1. They are also clustered about 3 s apart.

| | requests (W = 5,000) | distinct arrival times ("bursts") | mean requests / burst | smallest gap between bursts | continuations in the **same burst** as their predecessor |
|---|---|---|---|---|---|
| Conversation | 5,000 | **532** | 9.4 | 2.999 s | **495 / 2,001 = 24.7%** |
| Tool&Agent | 5,000 | **276** | 18.1 | 2.999 s | **1,137 / 2,012 = 56.5%** |

Every request shares its timestamp with at least one other. So ±0.5 ms never
moves a request into another burst. **It only randomly permutes the order of
requests within each burst.** Without jitter, `LLMSource` keeps file order
within a burst, because it uses a stable sort (`util/request.py:34-39`). A
continuation that shared a burst with its predecessor therefore always arrived
after it. Under jitter it arrives first about half the time, and then it cannot
hit the prefix its predecessor would have saved.

The shift is systematic, not scatter:

| | jittered runs with a **lower** hit rate than the unjittered run | jittered runs with **higher** HBF B/req than the unjittered run |
|---|---|---|
| (b), all 4 trace × cap cells | **20 / 20** | **20 / 20** |
| (d) Conv cap 48 / cap 256 | 5/5 / 4/5 | 0/5 / 1/5 |
| (d) T&A cap 48 / cap 256 | 5/5 / 5/5 | 2/5 / 0/5 |

**So the jittered replicates are a workload variant, "within-burst arrival
order randomized", and not replicates of the unjittered workload.** Their
spread *is* a valid measure of sensitivity to within-burst order. The trace does
not record that order; the stock run assumes file order. But their *location*
is shifted relative to every unjittered run, including all STEP 2 arms.

### 2b(d) — the two questions, numbers only

**Q1. Does the replicate spread of (d) HBF B/req cover the STEP 2 spread?**

| trace | STEP 2 arms [min, max] | replicates [min, max] | replicates + unjittered ref [min, max] | replicates + unjittered ref cover STEP 2? |
|---|---|---|---|---|
| Conversation | 37,316,198, 41,901,621 | 29,316,612, 37,516,476 | 29,316,612, 41,901,621 | yes |
| Tool&Agent | 17,168,859, 26,959,938 | 13,412,336, 22,079,341 | 13,412,336, 22,079,341 | no |


| | STEP 2 range width | replicate range width | width ratio |
|---|---|---|---|
| Conversation | 4,585,423 | 8,199,864 | **1.79×** |
| Tool&Agent | 9,791,079 | 8,667,005 | **0.89×** |

- **Replicates alone cover the STEP 2 range: no, on both traces.**
  - Conv: replicate max 37,516,476 < STEP 2 max 41,901,621.
  - T&A: replicate max 22,079,341 < STEP 2 max 26,959,938.
- **Widths are comparable.** The replicate spread is 1.79× the STEP 2 spread on
  Conv and 0.89× on T&A.
- The non-coverage is a location shift (the caveat above), not a narrower
  spread.

**Q2. Is the cap-48 vs cap-256 gap in (d) T&A (+70.10%) larger than the
replicate spread at either cap?**

| trace | single runs (s1 vs s0) | replicate means, cap 256 vs cap 48 | cap-48 replicate range | cap-256 replicate range | ranges overlap? | spread cap 48 / cap 256 |
|---|---|---|---|---|---|---|
| Conversation | +1.31% | +10.53% | 29,316,612 – 37,516,476 | 25,754,599 – 44,074,271 | yes | 24.46% / 49.45% |
| Tool&Agent | +70.10% | +16.53% | 13,412,336 – 22,079,341 | 17,543,725 – 22,895,657 | yes | 48.69% / 25.80% |

| (d) | cap-256 replicate ÷ cap-48 replicate, all 25 pairs: min / max |
|---|---|
| Conversation | −31.35% / +50.34% |
| Tool&Agent | −20.54% / **+70.71%** |

- **T&A replicate spread:** 48.69% of the mean at cap 48, 25.80% at cap 256.
- **The single-run gap, +70.10%, exceeds both spreads as a percentage of the
  mean.** But it lies **inside** the range of cap-256 ÷ cap-48 differences
  between individual replicates (−20.54% to +70.71%).
- **The gap between replicate means is +16.53%**, smaller than the spread at
  either cap. The two caps' replicate ranges **overlap**: 13.4–22.1 M against
  17.5–22.9 M.
- **Conversation:** single-run +1.31%, replicate means +10.53%. Ranges overlap.
- **Batch is not shown to drive the cap-48 → cap-256 change in (d).**

### Not concluded here

Whether STEP 2's failure is "run-to-run sensitivity rather than timing" is the
question the jitter was meant to settle. Because of the caveat it settles it
only partly:

- **Settled:** (d)'s HBF B/req moves by amounts comparable to the STEP 2 spread
  under a perturbation that involves **no timing change at all**, only
  within-burst order.
- **Not settled:** what the spread around the *unjittered* workload is. A
  perturbation that preserves within-burst file order is needed for that. One
  option is jitter applied to burst times only, with the same offset for every
  request in a burst. Another is jitter with j ≥ 1.5 s, which moves requests
  between bursts. Neither is chosen or run here.

### Settings

| | |
|---|---|
| code | **`5108029`** (jitter flag, tests, harness) on `phase14-fixes` |
| runs | `tmp/phase14/runs/s2b_seed{0,1,2}_d_ta`, `s2b_id_{b,d}_{conv,ta}`, `s2b_j{1..5}_c{48,256}_{b,d}_{conv,ta}` (47) |
| table | `scripts/phase14/step2b_table.py tmp/phase14` |
| hardware numbers | none invented or changed |

## STEP 2b — revised after review: request-mode jitter is a sensitivity result; clean replicates use burst-mode jitter

The decision after review: keep per-request jitter as a named workload variant,
and add burst-level jitter for replicates. Code commit and settings are at the
end of this section.

### The flag, now with a mode

| flag | values | clock | what moves |
|---|---|---|---|
| `--arrival_jitter_ms` | **0 (default, stock)**, or j > 0 | — | — |
| `--arrival_jitter_mode` | **`request`** (default when j > 0) | trace clock, before `--trace_target_qps` | each request independently. **A workload variant** |
| | `burst` | **replay clock, after rescaling** | all requests sharing an arrival time, by one shared offset. **Guarded:** after jittering, arrival order must equal file order exactly, otherwise `WorkloadValidationError` and the run does not start |

`request` is the behaviour of `5108029` under its new name. `s2c_req_d_ta`, with
`--arrival_jitter_mode request` passed explicitly, reproduces `s2b_j1_c48_d_ta`
byte for byte. **Byte identity at default, verified again on the new code:**
`s2c_id_{b,d}_{conv,ta}` against `ced3806` (`s0_*`) → **no diff in all four.**
**Tests: 221 pass** (213 + 8 new burst-mode tests in
`tests/test_arrival_jitter.py`). They cover identity at 0, one shared offset
per burst, bounds, order kept, the guard firing when bursts swap and on a trace
not in time order, determinism, and replay-clock application.

### LIMITATION — within-second request order is not recorded in the trace

The FAST'25 `timestamp` field has whole-second values, clustered at ~3 s
(STEP 2b caveat above). The order of requests within one burst is therefore not
in the data. **TokenSim replays file order within a burst** (stable sort,
`util/request.py:34-39`). Every result in this file assumes that order. This
matters because a continuation can only hit its predecessor's prefix if the
predecessor arrives first, and 24.7% (Conv) / 56.5% (T&A) of continuations
share a burst with their predecessor.

### Request-mode jitter as an intra-burst-order SENSITIVITY result (the earlier 2b runs)

The 40 `s2b_j*` runs (j = 0.5 ms, request mode, seeds 1–5) are re-labelled here
as a sensitivity result. **They are not noise.** Each seed permutes requests
within bursts.

**Continuations that arrive before their predecessor**
(`scripts/phase14/intraburst_order.py`, PHASE 12 STEP C pair rule, W = 5,000):

| | pairs | same burst | stock (file order) | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|---|---|---|---|
| Conversation | 2,001 | 495 (24.7%) | 0 (0.0%) | 233 (11.6%) | 226 (11.3%) | 260 (13.0%) | 228 (11.4%) | 255 (12.7%) |
| Tool&Agent | 2,012 | 1,137 (56.5%) | 0 (0.0%) | 551 (27.4%) | 560 (27.8%) | 575 (28.6%) | 550 (27.3%) | 564 (28.0%) |

As expected, about half of the same-burst pairs are reversed.

**Effect against the stock (file-order) run**, five seeds:

| cap | config | trace | stock hit rate | request-mode hit rate, Δ vs stock: mean [min, max] | stock HBF B/req | request-mode HBF B/req, Δ vs stock: mean [min, max] |
|---|---|---|---|---|---|---|
| 48 | (b) | Conversation | 0.1643 | -3.42% [-5.00%, -2.00%] | 3,370,675,339 | +0.48% [+0.06%, +0.91%] |
| 48 | (b) | Tool&Agent | 0.4037 | -2.99% [-4.29%, -0.89%] | 1,696,787,333 | +2.39% [+0.06%, +3.53%] |
| 48 | (d) | Conversation | 0.0974 | -9.65% [-13.99%, -2.88%] | 41,901,621 | -20.01% [-30.03%, -10.47%] |
| 48 | (d) | Tool&Agent | 0.3650 | -2.12% [-3.27%, -0.01%] | 17,168,859 | +3.67% [-21.88%, +28.60%] |
| 256 | (b) | Conversation | 0.1631 | -2.88% [-4.51%, -1.54%] | 3,387,654,930 | +0.50% [+0.16%, +0.89%] |
| 256 | (b) | Tool&Agent | 0.4014 | -2.84% [-3.93%, -0.75%] | 1,695,909,675 | +2.43% [+0.62%, +3.45%] |
| 256 | (d) | Conversation | 0.0942 | -2.42% [-5.13%, +1.86%] | 42,450,551 | -12.73% [-39.33%, +3.82%] |
| 256 | (d) | Tool&Agent | 0.3733 | -3.51% [-5.29%, -1.67%] | 29,204,414 | -28.98% [-39.93%, -21.60%] |

- **On (b):** reordering within a burst lowers the hit rate by 2.8–3.4% on
  average. HBF B/req rises by +0.5% (Conv) to +2.4% (T&A).
- **On (d):** HBF B/req moves −29.0% to +3.7% on average. Individual seeds span
  −39.9% to +28.6%.
- **Conclusion:** the stock file order is a favourable order for prefix reuse.
  That is expected: predecessors precede continuations in the file.

### Burst-mode jitter — the clean replicates

**Inter-burst gaps in replay time** (`scripts/phase14/burst_gaps.py`, the
simulator's own loader, W = 5,000, `--trace_target_qps 30`):

| | bursts | gap min | gap p50 | gap max | **j = 0.45 × min** |
|---|---|---|---|---|---|
| Conversation | 532 | 313.706 ms | 313.810 ms | 313.915 ms | **141.168 ms** |
| Tool&Agent | 276 | 605.737 ms | 605.939 ms | 606.141 ms | **272.582 ms** |

The gaps are almost constant: the trace issues one burst every ~3 s of trace
time. **j = 0.45 × the minimum gap is a methodology choice, not a hardware
number.** At 2j < min gap no two bursts can swap. The order guard confirmed this
in all 40 runs: none failed.

**40 runs:** `--arrival_jitter_mode burst`, `--jitter_seed 1..5`, (b) and (d),
both traces, caps 48 and 256. Timing is charged, all values stock, and
otherwise as STEP 0. `notdone = 0` in all 40.

| cap | config | trace | HBF B/req mean [min, max] | spread | PCM B/req mean [min, max] | spread | hit rate mean [min, max] | batch/sched mean [min, max] | notdone |
|---|---|---|---|---|---|---|---|---|---|
| 48 | (b) | Conversation | 3,373,740,746 [3,370,120,643, 3,377,824,530] | 0.23% | 1,724,150,658 [1,722,340,606, 1,726,192,550] | 0.22% | 0.1643 [0.1641, 0.1647] | 23.184 [23.167, 23.198] | 0 |
| 48 | (b) | Tool&Agent | 1,696,381,954 [1,691,782,480, 1,699,508,388] | 0.46% | 885,471,261 [883,171,525, 887,034,479] | 0.44% | 0.4033 [0.4027, 0.4037] | 22.894 [22.807, 22.939] | 0 |
| 48 | (d) | Conversation | 37,965,477 [32,048,677, 41,914,728] | 25.99% | 1,991,880,003 [1,986,138,210, 2,002,090,721] | 0.80% | 0.0951 [0.0917, 0.0979] | 22.473 [22.370, 22.583] | 0 |
| 48 | (d) | Tool&Agent | 21,749,039 [19,802,882, 26,329,219] | 30.01% | 1,001,879,555 [997,253,448, 1,004,686,279] | 0.74% | 0.3676 [0.3643, 0.3696] | 22.325 [22.151, 22.402] | 0 |
| 256 | (b) | Conversation | 3,385,905,696 [3,382,722,429, 3,390,542,709] | 0.23% | 1,730,233,133 [1,728,641,499, 1,732,551,639] | 0.23% | 0.1639 [0.1631, 0.1646] | 43.925 [43.838, 44.012] | 0 |
| 256 | (b) | Tool&Agent | 1,693,402,740 [1,687,769,580, 1,696,351,126] | 0.51% | 883,981,654 [881,165,074, 885,455,847] | 0.49% | 0.4011 [0.4000, 0.4030] | 48.347 [47.714, 48.704] | 0 |
| 256 | (d) | Conversation | 41,693,269 [38,663,094, 45,437,420] | 16.25% | 1,984,781,248 [1,982,704,648, 1,988,798,972] | 0.31% | 0.0946 [0.0923, 0.0992] | 41.934 [41.595, 42.114] | 0 |
| 256 | (d) | Tool&Agent | 25,567,953 [22,920,298, 27,843,887] | 19.26% | 995,091,599 [992,346,112, 1,000,926,609] | 0.86% | 0.3708 [0.3695, 0.3717] | 44.902 [44.687, 45.119] | 0 |

"spread" is (max − min) ÷ mean over the five replicates.

**Burst jitter is neutral on (b).** Against the stock run, the replicate-mean
hit rate moves −0.08% to +0.52% and HBF B/req −0.15% to +0.09%. The stock (b)
value lies inside the replicate range in all four cells. Compare request mode:
−2.8% to −3.4% hit rate in every cell.

**Batch mismatch — flagged.**

| | (d) vs (b), replicate means | burst vs stock, (b) | burst vs stock, (d) |
|---|---|---|---|
| cap 48 Conversation | −3.07% | +0.07% | +0.42% |
| cap 48 Tool&Agent | −2.49% | +0.04% | −0.47% |
| cap 256 Conversation | −4.53% | +0.72% | +0.92% |
| cap 256 Tool&Agent | **−7.13%** | −0.19% | +0.40% |

**Where the stock single run sits within the clean replicates**, for (d) HBF
B/req:

| | stock value | replicates [min, max] | stock position |
|---|---|---|---|
| Conv cap 48 | 41,901,621 | 32,048,677 – 41,914,728 | inside, 2nd highest of 6 |
| **T&A cap 48** | **17,168,859** | 19,802,882 – 26,329,219 | **below all five** |
| Conv cap 256 | 42,450,551 | 38,663,094 – 45,437,420 | inside, 3rd highest of 6 |
| **T&A cap 256** | **29,204,414** | 22,920,298 – 27,843,887 | **above all five** |

**(b) → (d) at the same seed, clean replicates:**

| | reduction | b ÷ d |
|---|---|---|
| Conv cap 48 | 98.76–99.05% | ×80.5–×105.2 |
| T&A cap 48 | 98.44–98.83% | ×64.3–×85.7 |
| Conv cap 256 | 98.66–98.86% | ×74.5–×87.6 |
| T&A cap 256 | 98.35–98.65% | ×60.6–×73.8 |

The PHASE 9 single-run T&A ratio **×98.8 is above every clean replicate**
(max ×85.7).

### Q1 and Q2 with the clean replicates — numbers only

**Q1. Does the replicate spread of (d) HBF B/req cover the STEP 2 spread?**

| trace | STEP 2 arms [min, max] | replicates [min, max] | replicates + unjittered ref [min, max] | replicates + unjittered ref cover STEP 2? |
|---|---|---|---|---|
| Conversation | 37,316,198, 41,901,621 | 32,048,677, 41,914,728 | 32,048,677, 41,914,728 | yes |
| Tool&Agent | 17,168,859, 26,959,938 | 19,802,882, 26,329,219 | 17,168,859, 26,329,219 | no |


| | STEP 2 width | replicate width | ratio | STEP 2 non-reference arms inside replicate range |
|---|---|---|---|---|
| Conversation | 4,585,423 | 9,866,051 | **2.15×** | 6 / 6 |
| Tool&Agent | 9,791,079 | 6,526,337 | **0.67×** | 4 / 6 (`ssd` 17,907,057 below, `pbw025` 26,959,938 above) |

- **Conversation: yes.** Replicates alone, 32.0–41.9 M, contain the whole STEP 2
  range, 37.3–41.9 M.
- **Tool&Agent: no.** The STEP 2 range runs 17.2–27.0 M. Its low end is the
  stock run itself, below every replicate. Its high end, `pbw025`, is 2.4% above
  the replicate max. Four of the six timing arms fall inside the replicate range.

**Q2. Is the cap-48 vs cap-256 gap in (d) T&A (+70.10%) larger than the
replicate spread at either cap?**

| trace | single runs (s1 vs s0) | replicate means, cap 256 vs cap 48 | cap-48 replicate range | cap-256 replicate range | ranges overlap? | spread cap 48 / cap 256 |
|---|---|---|---|---|---|---|
| Conversation | +1.31% | +9.82% | 32,048,677 – 41,914,728 | 38,663,094 – 45,437,420 | yes | 25.99% / 16.25% |
| Tool&Agent | +70.10% | +17.56% | 19,802,882 – 26,329,219 | 22,920,298 – 27,843,887 | yes | 30.01% / 19.26% |

| (d) | cap-256 ÷ cap-48, all 25 replicate pairs: min / max |
|---|---|
| Conversation | −7.76% / +41.78% |
| Tool&Agent | −12.95% / +40.61% |

- **The single-run +70.10% is larger than the spread at either cap** (30.01% at
  cap 48, 19.26% at cap 256). It is also larger than every replicate pair
  (max +40.61%).
- **The replicates do not reproduce it.**
  - The replicate-mean gap is **+17.56%**, smaller than the spread at either
    cap, and the two ranges overlap.
  - The stock cap-48 run is below all five cap-48 replicates. The stock cap-256
    run is above all five cap-256 replicates.
- **Conversation:** single-run +1.31%, replicate means +9.82%. Ranges overlap.
- Per the decision rule: **batch is not shown to drive the cap-48 → cap-256
  change in (d).**

### Rule for STEP 3

The clean replicate spread of (d) HBF B/req is **25.99% / 30.01% at cap 48** and
**16.25% / 19.26% at cap 256**. That is above 5% of the mean at both caps, on
both traces. **So STEP 3 runs 3 burst-jitter replicates per sweep point and
reports ranges.**

### Settings

| | |
|---|---|
| code | **`20695c3`** on `phase14-fixes` |
| scripts | `scripts/phase14/{step2c.sh,burst_gaps.py,intraburst_order.py,step2b_table.py}`. Table: `step2b_table.py tmp/phase14 s2c` |
| runs | `tmp/phase14/runs/s2c_id_*` (4), `s2c_req_d_ta` (1), `s2c_j{1..5}_c{48,256}_{b,d}_{conv,ta}` (40) |
| traces / hardware | md5s as in Provenance, unchanged. No hardware number invented or changed |

## STEP 3 prelude — jitter size, (c) replicates, and a note on the request-mode table

Driver: `scripts/phase14/prelude3.sh`. Runs: `tmp/phase14/runs/p3_*` (20). The
configuration is otherwise as STEP 0: cap 48, 30 QPS, W = 5,000, charged timing,
stock values. `notdone = 0` in all 20.

### Prelude (a) — does (d)'s replicate spread depend on the jitter size?

(d), cap 48, burst mode, seeds 1–5, at **j = 0.05 × the minimum inter-burst
gap**: Conversation 0.05 × 313.706 ms = **15.685 ms**, Tool&Agent
0.05 × 605.737 ms = **30.287 ms** (`burst_gaps.py ... 0.05`). This is a
methodology choice, not a hardware number. It is compared with the j = 0.45
replicates (`s2c_j*_c48_d_*`).

| (d) HBF B/req, cap 48 | j | mean [min, max] | **spread** | stock run (j = 0) |
|---|---|---|---|---|
| Conversation | 0.05 × gap (15.685 ms) | 40,021,629 [38,428,738, 41,851,290] | **8.55%** | 41,901,621 (0.12% above the max) |
| Conversation | 0.45 × gap (141.168 ms) | 37,965,477 [32,048,677, 41,914,728] | **25.99%** | inside |
| Tool&Agent | 0.05 × gap (30.287 ms) | 21,470,852 [20,660,093, 23,678,419] | **14.06%** | 17,168,859 (**below all five**) |
| Tool&Agent | 0.45 × gap (272.582 ms) | 21,749,039 [19,802,882, 26,329,219] | **30.01%** | below all five |

Spread ratio, small j ÷ large j: **0.33 (Conversation)** and **0.47
(Tool&Agent)**.

Other columns at small j:

| | PCM B/req spread | hit rate | batch/sched |
|---|---|---|---|
| Conversation | 0.26% | 0.0946–0.0977 | 22.382–22.579 |
| Tool&Agent | 0.80% | 0.3672–0.3678 | 22.239–22.457 |

**Reading, per the rule set before the runs:** on both traces the small-j spread
is below half the large-j spread. **The spread of (d) HBF B/req depends on j.**
So every (d) range quoted in report.md must state the j used.

Two further facts. Neither is a conclusion.

- Even at small j the spread is **8.55% / 14.06%, above the 5% replicate
  threshold**.
- On Tool&Agent the stock run, 17,168,859, lies below every replicate at
  **both** jitter sizes. The small-j replicates, 20.7–23.7 M, do not approach it.

### Prelude (b) — (c) replicates

(c) = +PCM, `write_back` + `always`, cap 48, burst mode, j = 0.45 × gap
(141.168 / 272.582 ms), seeds 1–5. The (b) partner of each seed is
`s2c_j<k>_c48_b_*`.

| (c), cap 48 | HBF B/req mean [min, max] | spread | PCM B/req mean [min, max] | spread | **(b) → (c) reduction, same seed** | headline (PHASE 9 STEP 3) |
|---|---|---|---|---|---|---|
| Conversation | 1,648,399,955 [1,646,584,136, 1,649,811,653] | **0.20%** | 1,722,960,524 [1,721,144,705, 1,724,372,222] | 0.19% | **51.07–51.25%** | 51.10% (inside) |
| Tool&Agent | 811,342,181 [809,852,469, 811,886,707] | **0.25%** | 885,902,751 [884,413,039, 886,447,276] | 0.23% | **52.02–52.25%** | 52.18% (inside) |

Other columns:

- Hit rate: 0.1638–0.1646 (Conv) and 0.4039–0.4041 (T&A).
- Batch/sched, mean [min, max]: 23.038 [22.943, 23.073] (Conv) and 22.743
  [22.622, 22.797] (T&A).
- **Batch gap (c) vs (b)**, replicate means: −0.63% (Conv) and −0.66% (T&A).

**(c) is replicate-stable.** The write-back halving holds to within 0.25% across
seeds, and the headline lies inside its replicate range on both traces.

### Prelude (c) — NOTE on the request-mode sensitivity table

**Points back to** "Request-mode jitter as an intra-burst-order SENSITIVITY
result" (STEP 2b revised).

Those (d) HBF deltas are not readable as effects of intra-burst order. Each
request-mode mean shift against the clean (j = 0.45 burst) replicate mean is
inside, or at the edge of, (d)'s own replicate spread:

| cap / trace | request-mode mean vs clean mean | clean spread | request-mode runs outside the clean range |
|---|---|---|---|
| 48 Conversation | −11.71% | 25.99% | 1 / 5 |
| 48 Tool&Agent | −18.16% | 30.01% | 3 / 5 |
| 256 Conversation | −11.15% | 16.25% | 3 / 5 |
| 256 Tool&Agent | **−18.88%** | **19.26%** (at the edge) | **5 / 5** |

**Readable from that table:**

- (b)'s hit-rate drop (−2.8% to −3.4%), which is far outside (b)'s ≤ 0.5%
  replicate spread.
- The share of continuations moved ahead of their predecessor (11–13% / 27–29%).

**Counter-evidence, recorded so it is not lost:** all four (d) shifts have the
same sign (−11% to −19%), and 12 of 20 request-mode runs fall outside the clean
range. That includes all five at T&A cap 256. A consistent direction is
suggestive. By the spread criterion it is not established.

### Prelude — notes after review (supersede the wording above, which stays as written)

1. **Request-mode note, revised wording.** This replaces the reading in
   "Prelude (c)". For request-mode jitter, the (d) HBF effect is as follows:
   **the direction is consistent (lower (d) HBF) and it falls outside the clean
   range in 12/20 runs; its size is not pinned down.** (b)'s hit-rate drop and
   the moved-continuation share remain the readable effects.
2. **Which reference each table uses.**

   | table | deltas are against |
   |---|---|
   | STEP 2b revised, "Effect against the stock (file-order) run": (d) +3.67% / −28.98% etc. | the **stock single run** (j = 0) |
   | Prelude (c) note: −11.71% … −18.88% | the **clean-replicate mean** (burst, j = 0.45 × gap) |

   The two sets of numbers are not comparable with each other. The stock T&A
   (d) run is itself below every replicate, so a delta against it overstates
   the shift on that trace.
3. **T&A's j-dependence is borderline.** Its spread ratio is **0.47**, against
   the 0.5 cut-off. Conversation's 0.33 is clear. The "depends on j" reading
   holds for both traces under the rule as written, but on Tool&Agent it is
   within 0.03 of the cut-off.

## STEP 3 — (d) concurrency sweep with replicates

**Pre-registered reading, fixed before any run:** "batch drives (d) HBF" only if
(i) the per-cap means move in one direction across caps **and** (ii) the change
in mean from the lowest to the highest cap exceeds the largest per-cap
replicate spread. Otherwise "not shown".

**Result: "not shown" on both traces.** The means are **non-monotone** across
caps on both traces, which fails (i). The cap 8 → cap 256 change is +15.01%
(Conv) and +2.45% (T&A), against largest per-cap spreads of 23.76% and 20.22%,
which fails (ii) as well. **But (d) HBF is not flat across caps either.** On
Tool&Agent, 18 of 28 cap pairs have non-overlapping replicate ranges (see
"Non-monotone, flagged").

### Why j = 0.05 and not 0.45 (recorded per instruction)

The sweep uses burst jitter at **j = 0.05 × the minimum inter-burst gap**
(15.685 ms Conv, 30.287 ms T&A), not 0.45. At j = 0.45 the per-cap spread is
26–30%. That would make "not shown", the answer that supports the paper's
invariance claim, the easy outcome. **j = 0.05 is the stricter test.** It is a
methodology choice, not a hardware number.

### Setup

- (d) only, caps {8, 16, 32, 48, 64, 96, 192, 256}, both traces, seeds 1–3:
  48 runs. Driver `scripts/phase14/step3.sh`, table
  `scripts/phase14/step3_table.py`.
- 30 QPS, W = 5,000, charged timing, stock values, otherwise as STEP 0.
  `notdone = 0` in all runs.
- **The cap-48 runs reproduce `p3_js{1,2,3}_d_*` byte for byte** (stripped
  diff empty, all six).
- Conversation caps 192 and 256 are byte-identical per seed, because the cap
  stops binding. PHASE 9 found the same for (b).

**(b) is not rerun, as agreed.** Burst jitter moves (b) HBF by at most 0.15%
(STEP 2b revised), so the stock PHASE 9 STEP 4 rows are reused. The exception
is **Tool&Agent caps 192 and 256, which PHASE 9 STEP 4 never ran**:

- cap 256 is `s1_b256_ta` (STEP 1e, which reproduces PHASE 9 STEP 3 revised);
- cap 192 is a new stock run, `s3_b192_ta`: 1,684,909,064 B/req, batch 47.87.

The (b) batch figures from PHASE 9 STEP 4 are recorded there to 2 dp only.

### Conversation — (d), j = 0.05 x min gap, 3 seeds per cap

| cap | (d) HBF B/req mean [min, max] | spread | (d) PCM B/req mean [min, max] | hit rate [min, max] | (d) batch/sched mean [min, max] | (b) HBF B/req | (b) batch | (b)->(d) reduction | batch gap (d) vs (b) | (b) source |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 35,062,983 [34,280,047, 35,605,447] | 3.78% | 1,993,435,251 [1,991,602,340, 1,995,988,533] | [0.0969, 0.0985] | 3.927 [3.923, 3.935] | 3,377,115,693 | 3.94 | 98.95–98.98% | -0.33% | PHASE 9 STEP 4 |
| 16 | 34,621,533 [34,212,938, 35,103,179] | 2.57% | 1,992,957,450 [1,992,391,918, 1,993,310,470] | [0.0956, 0.0960] | 7.692 [7.690, 7.693] | 3,370,987,815 | 7.83 | 98.96–98.99% | -1.77% | PHASE 9 STEP 4 |
| 32 | 39,260,433 [35,370,566, 41,213,755] | 14.88% | 1,991,402,761 [1,986,555,019, 2,000,733,340] | [0.0919, 0.0960] | 15.036 [14.978, 15.147] | 3,383,093,625 | 15.46 | 98.78–98.95% | -2.74% | PHASE 9 STEP 4 |
| 48 | 40,731,236 [39,653,474, 41,851,290] | 5.40% | 1,989,226,791 [1,987,605,168, 1,990,650,757] | [0.0948, 0.0977] | 22.491 [22.382, 22.579] | 3,370,675,339 | 23.17 | 98.76–98.82% | -2.92% | PHASE 9 STEP 4 |
| 64 | 35,491,152 [32,770,097, 38,093,193] | 15.00% | 1,995,854,665 [1,991,621,214, 1,998,590,050] | [0.0933, 0.0951] | 29.701 [29.635, 29.776] | 3,376,704,651 | 30.81 | 98.87–99.03% | -3.60% | PHASE 9 STEP 4 |
| 96 | 37,733,881 [35,269,378, 39,067,320] | 10.07% | 1,991,804,365 [1,986,854,388, 1,998,140,735] | [0.0914, 0.0963] | 40.587 [40.333, 40.947] | 3,385,003,082 | 42.16 | 98.85–98.96% | -3.73% | PHASE 9 STEP 4 |
| 192 | 40,325,612 [34,849,948, 44,431,311] | 23.76% | 1,988,299,850 [1,980,956,148, 1,997,772,161] | [0.0913, 0.0983] | 41.833 [41.606, 42.049] | 3,387,654,930 | 43.61 | 98.69–98.97% | -4.08% | PHASE 9 STEP 4 |
| 256 | 40,325,612 [34,849,948, 44,431,311] | 23.76% | 1,988,299,850 [1,980,956,148, 1,997,772,161] | [0.0913, 0.0983] | 41.833 [41.606, 42.049] | 3,387,654,930 | 43.61 | 98.69–98.97% | -4.08% | PHASE 9 STEP 4 |

means move in one direction: no (signs of cap-to-cap steps: -++-++-); cap 8 -> cap 256 change in mean: +15.01%; largest per-cap spread: 23.76% (cap 192); extreme means: min 34,621,533 (cap 16), max 40,731,236 (cap 48) -> not shown

### Tool&Agent — (d), j = 0.05 x min gap, 3 seeds per cap

| cap | (d) HBF B/req mean [min, max] | spread | (d) PCM B/req mean [min, max] | hit rate [min, max] | (d) batch/sched mean [min, max] | (b) HBF B/req | (b) batch | (b)->(d) reduction | batch gap (d) vs (b) | (b) source |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 26,966,753 [24,148,181, 28,459,926] | 15.99% | 997,260,438 [995,666,952, 1,000,145,945] | [0.3691, 0.3695] | 3.881 [3.877, 3.884] | 1,696,587,055 | 3.90 | 98.32–98.58% | -0.49% | PHASE 9 STEP 4 |
| 16 | 23,505,753 [22,996,320, 23,963,632] | 4.12% | 999,009,463 [998,725,124, 999,359,513] | [0.3684, 0.3685] | 7.614 [7.604, 7.628] | 1,697,241,367 | 7.75 | 98.59–98.65% | -1.76% | PHASE 9 STEP 4 |
| 32 | 24,399,839 [22,638,232, 27,570,733] | 20.22% | 998,791,883 [995,901,309, 1,000,833,810] | [0.3664, 0.3677] | 15.060 [15.043, 15.088] | 1,695,311,987 | 15.29 | 98.37–98.66% | -1.50% | PHASE 9 STEP 4 |
| 48 | 21,777,525 [20,660,093, 23,678,419] | 13.86% | 1,000,152,061 [998,507,020, 1,003,026,383] | [0.3672, 0.3677] | 22.374 [22.239, 22.457] | 1,696,787,333 | 22.89 | 98.60–98.78% | -2.24% | PHASE 9 STEP 4 |
| 64 | 19,657,480 [18,947,244, 20,356,530] | 7.17% | 1,007,766,121 [1,006,485,111, 1,008,767,336] | [0.3641, 0.3641] | 29.580 [29.566, 29.603] | 1,697,208,861 | 30.38 | 98.80–98.88% | -2.63% | PHASE 9 STEP 4 |
| 96 | 26,475,321 [26,033,521, 26,738,164] | 2.66% | 995,847,307 [995,584,115, 996,273,029] | [0.3700, 0.3701] | 40.905 [40.780, 41.149] | 1,678,879,752 | 43.03 | 98.41–98.45% | -4.94% | PHASE 9 STEP 4 |
| 192 | 25,144,328 [24,674,566, 25,614,090] | 3.74% | 999,081,465 [998,574,653, 999,394,640] | [0.3686, 0.3692] | 45.446 [45.380, 45.551] | 1,684,909,064 | 47.87 | 98.48–98.54% | -5.06% | s3_b192_ta |
| 256 | 27,626,657 [26,855,604, 28,020,572] | 4.22% | 994,657,348 [992,700,531, 995,653,321] | [0.3724, 0.3733] | 45.240 [45.202, 45.261] | 1,695,909,675 | 48.44 | 98.35–98.42% | -6.60% | s1_b256_ta |

means move in one direction: no (signs of cap-to-cap steps: -+--+-+); cap 8 -> cap 256 change in mean: +2.45%; largest per-cap spread: 20.22% (cap 32); extreme means: min 19,657,480 (cap 64), max 27,626,657 (cap 256) -> not shown

### Batch mismatch — flagged

The (d)-vs-(b) batch gap grows with cap:

- **Conversation:** −0.33% at cap 8 to −4.08% at caps 192/256.
- **Tool&Agent:** −0.49% at cap 8 to **−6.60%** at cap 256.

From cap 64 upward (d) also reaches 98.8–99.2% peak HBM occupancy, the same as
(b) in PHASE 9:

| cap | 8 | 16 | 32 | 48 | 64 | 96 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| Conv (d) peak HBM, max over seeds | 29.1% | 37.6% | 67.9% | 91.7% | 99.0% | 99.2% | 99.1% | 99.1% |
| T&A (d) peak HBM, max over seeds | 33.6% | 40.8% | 60.0% | 79.5% | 98.8% | 99.2% | 99.2% | 99.2% |

### Non-monotone, flagged (not smoothed)

| | range of per-cap means (max ÷ min − 1) | cap pairs whose 3-seed ranges do not overlap |
|---|---|---|
| Conversation | 17.65% (cap 16 → cap 48) | 6 / 28 |
| Tool&Agent | **40.54%** (cap 64 → cap 256) | **18 / 28** |

On Tool&Agent the mean falls from cap 8 to cap 64, then jumps at cap 96. Cap 64
(19.66 M, range 18.95–20.36 M) and cap 256 (27.63 M, range 26.86–28.02 M) do not
overlap.

Two caveats on reading that:

- With 3 seeds per cap, a non-overlapping range is weak evidence.
- The pattern is not monotone, so the rule's "batch drives" reading does not
  apply.

**What can be said:** (d)'s HBF write volume **varies with cap by more than its
replicate spread on Tool&Agent, non-monotonically.** That is not the
"concurrency-invariant" behaviour PHASE 9 STEP 4 showed for (b) and (c).

**The (b) → (d) reduction, by contrast, stays within 98.32–98.88% (T&A) and
98.69–98.99% (Conv) at every cap.**

### Second view — the j = 0.45 end points (STEP 2b revised), labelled with their j

At **j = 0.45 × gap**, 5 seeds, (d) T&A:

- cap 48: 19.8–26.3 M, spread 30.01%
- cap 256: 22.9–27.8 M, spread 19.26%
- the replicate means differ by **+17.56%**, inside both spreads, and the ranges
  overlap

At j = 0.05 the same two caps give +26.86% between means (21.78 M → 27.63 M).
That exceeds both of those caps' own spreads (13.86% and 4.22%), and their
ranges do not overlap. But it is not monotone across the intervening caps.

## Extra check — j = 0.005 × min gap: do the replicates approach the stock run?

(d), cap 48, burst mode, **j = 0.005 × min gap = 1.569 ms (Conv), 3.029 ms
(T&A)**, seeds 1–5 (`s3_jt*`). This is a methodology choice, not a hardware
number.

| (d) HBF B/req, cap 48 | j | values (seeds 1–5) | mean [min, max] | spread |
|---|---|---|---|---|
| Conversation | 0.005 × gap | 43,025,695 · 41,566,077 · 43,025,695 · 42,772,464 · **41,901,621** | 42,458,310 [41,566,077, 43,025,695] | 3.44% |
| Conversation | 0.05 × gap | — | 40,021,629 [38,428,738, 41,851,290] | 8.55% |
| Conversation | 0.45 × gap | — | 37,965,477 [32,048,677, 41,914,728] | 25.99% |
| Tool&Agent | 0.005 × gap | 17,185,636 · 20,759,183 · 23,544,201 · **17,168,859** · 17,185,636 | 19,168,703 [17,168,859, 23,544,201] | **33.26%** |
| Tool&Agent | 0.05 × gap | — | 21,470,852 [20,660,093, 23,678,419] | 14.06% |
| Tool&Agent | 0.45 × gap | — | 21,749,039 [19,802,882, 26,329,219] | 30.01% |

Stock (j = 0): Conversation 41,901,621; Tool&Agent 17,168,859.

**Answer: at j = 0.005 the replicates reach the stock run. They do not stay with
the j = 0.05 range.**

- **Tool&Agent seed 4** reproduces the stock store outcome exactly. HBF and PCM
  write blocks, hit rate, reuse hit blocks, demoted/dropped, batch and duration
  are all identical; the stripped results differ in 8 lines of
  latency/timing detail. Seeds 1 and 5 land 32 blocks away (17,185,636 B/req).
- **Conversation seed 5** reproduces the stock store counters exactly. Batch
  and duration differ in the 5th significant figure.
- **The outcomes are discrete.** Repeated values appear across seeds
  (43,025,695 twice on Conv; 17,185,636 twice on T&A). A small arrival
  perturbation either leaves the store trajectory exactly where the stock run
  put it, or flips it to another of a few states.
- **Spread against j is not monotone on Tool&Agent:** 33.26% → 14.06% → 30.01%
  for j = 0.005 → 0.05 → 0.45. That weakens the prelude's "spread depends on j"
  reading for T&A, which was already borderline (ratio 0.47). On Conversation
  the spread does grow with j: 3.44% → 8.55% → 25.99%.

**Candidate, not a finding: the stock replay is a perfectly periodic burst
train.** Replay-time gaps are 313.706–313.915 ms (Conv) and 605.737–606.141 ms
(T&A), and that periodicity could be why the stock run sits outside every
j ≥ 0.05 replicate. The j = 0.005 data bear on it without settling it:

- The stock store outcome is reproduced exactly under a **non-periodic** train
  (T&A seed 4, ±3.029 ms offsets). So exact periodicity is **not necessary** for
  it.
- What the data do show is that the stock outcome is one of several discrete
  states. It is common at very small perturbations and rare at larger ones.

### Settings

| | |
|---|---|
| runs | `tmp/phase14/runs/s3_c{8,…,256}_s{1,2,3}_d_{conv,ta}` (48), `s3_jt{1..5}_d_{conv,ta}` (10), `s3_b192_ta` (1, stock (b)) |
| scripts | `scripts/phase14/step3.sh`, `step3_table.py`; `burst_gaps.py` for the j values |
| hardware | none invented or changed |

## STEP 3 — pooled (d) bound at cap 48, and corrections (docs/analysis only, no new runs)

### Pooled (d), cap 48

Every distinct (d) cap-48 run in this phase: the stock run, and the burst-jitter
replicates at j = 0.005, 0.05 and 0.45 × min inter-burst gap. Excluded:

- byte-identical duplicates: `s3_c48_s{1,2,3}` = `p3_js{1,2,3}`, and the
  stock-identical `s2_id`, `s2_ref`, `s2b_id`, `s2c_id`;
- request-mode runs, a workload variant;
- the STEP 2 timing arms, which are listed separately below.

(b) for each reduction is the same-seed (b) where one exists (j = 0.45).
Otherwise it is the stock (b) run; burst jitter moves (b) HBF by ≤ 0.15%.

#### Conversation

| run | (d) HBF B/req | (b) HBF B/req (source) | (b) → (d) reduction | b ÷ d |
|---|---|---|---|---|
| stock (j = 0) | 41,901,621 | 3,370,675,339 (stock (b)) | 98.76% | ×80.4 |
| j = 0.005 × gap, seed 1 | 43,025,695 | 3,370,675,339 (stock (b)) | 98.72% | ×78.3 |
| j = 0.005 × gap, seed 2 | 41,566,077 | 3,370,675,339 (stock (b)) | 98.77% | ×81.1 |
| j = 0.005 × gap, seed 3 | 43,025,695 | 3,370,675,339 (stock (b)) | 98.72% | ×78.3 |
| j = 0.005 × gap, seed 4 | 42,772,464 | 3,370,675,339 (stock (b)) | 98.73% | ×78.8 |
| j = 0.005 × gap, seed 5 | 41,901,621 | 3,370,675,339 (stock (b)) | 98.76% | ×80.4 |
| j = 0.05 × gap, seed 1 | 41,851,290 | 3,370,675,339 (stock (b)) | 98.76% | ×80.5 |
| j = 0.05 × gap, seed 2 | 39,653,474 | 3,370,675,339 (stock (b)) | 98.82% | ×85.0 |
| j = 0.05 × gap, seed 3 | 40,688,943 | 3,370,675,339 (stock (b)) | 98.79% | ×82.8 |
| j = 0.05 × gap, seed 4 | 38,428,738 | 3,370,675,339 (stock (b)) | 98.86% | ×87.7 |
| j = 0.05 × gap, seed 5 | 39,485,702 | 3,370,675,339 (stock (b)) | 98.83% | ×85.4 |
| j = 0.45 × gap, seed 1 | 37,824,758 | 3,375,079,358 (same-seed (b)) | 98.88% | ×89.2 |
| j = 0.45 × gap, seed 2 | 39,766,196 | 3,377,824,530 (same-seed (b)) | 98.82% | ×84.9 |
| j = 0.45 × gap, seed 3 | 32,048,677 | 3,370,120,643 (same-seed (b)) | 99.05% | ×105.2 |
| j = 0.45 × gap, seed 4 | 41,914,728 | 3,373,242,253 (same-seed (b)) | 98.76% | ×80.5 |
| j = 0.45 × gap, seed 5 | 38,273,024 | 3,372,436,947 (same-seed (b)) | 98.87% | ×88.1 |

#### Tool&Agent

| run | (d) HBF B/req | (b) HBF B/req (source) | (b) → (d) reduction | b ÷ d |
|---|---|---|---|---|
| stock (j = 0) | 17,168,859 | 1,696,787,333 (stock (b)) | 98.99% | ×98.8 |
| j = 0.005 × gap, seed 1 | 17,185,636 | 1,696,787,333 (stock (b)) | 98.99% | ×98.7 |
| j = 0.005 × gap, seed 2 | 20,759,183 | 1,696,787,333 (stock (b)) | 98.78% | ×81.7 |
| j = 0.005 × gap, seed 3 | 23,544,201 | 1,696,787,333 (stock (b)) | 98.61% | ×72.1 |
| j = 0.005 × gap, seed 4 | 17,168,859 | 1,696,787,333 (stock (b)) | 98.99% | ×98.8 |
| j = 0.005 × gap, seed 5 | 17,185,636 | 1,696,787,333 (stock (b)) | 98.99% | ×98.7 |
| j = 0.05 × gap, seed 1 | 20,660,093 | 1,696,787,333 (stock (b)) | 98.78% | ×82.1 |
| j = 0.05 × gap, seed 2 | 20,994,064 | 1,696,787,333 (stock (b)) | 98.76% | ×80.8 |
| j = 0.05 × gap, seed 3 | 23,678,419 | 1,696,787,333 (stock (b)) | 98.60% | ×71.7 |
| j = 0.05 × gap, seed 4 | 20,859,847 | 1,696,787,333 (stock (b)) | 98.77% | ×81.3 |
| j = 0.05 × gap, seed 5 | 21,161,837 | 1,696,787,333 (stock (b)) | 98.75% | ×80.2 |
| j = 0.45 × gap, seed 1 | 26,329,219 | 1,691,782,480 (same-seed (b)) | 98.44% | ×64.3 |
| j = 0.45 × gap, seed 2 | 20,373,307 | 1,694,342,054 (same-seed (b)) | 98.80% | ×83.2 |
| j = 0.45 × gap, seed 3 | 21,279,277 | 1,698,942,157 (same-seed (b)) | 98.75% | ×79.8 |
| j = 0.45 × gap, seed 4 | 20,960,510 | 1,699,508,388 (same-seed (b)) | 98.77% | ×81.1 |
| j = 0.45 × gap, seed 5 | 19,802,882 | 1,697,334,690 (same-seed (b)) | 98.83% | ×85.7 |

#### Pooled bound

| | **N** | (d) HBF B/req min – max | **worst-case reduction** | best-case reduction | b ÷ d range |
|---|---|---|---|---|---|
| **Conversation** | **16** | 32,048,677 – 43,025,695 | **98.72%** | 99.05% | ×78.3 – ×105.2 |
| **Tool&Agent** | **16** | 17,168,859 – 26,329,219 | **98.44%** | 98.99% | ×64.3 – ×98.8 |

#### STEP 2 timing arms, listed separately

These are not part of the pooled bound: each changes a timing knob, so it is not
a replicate. Each arm uses its own same-arm (b).

| arm | Conv (d) HBF B/req | Conv reduction | Conv b ÷ d | T&A (d) HBF B/req | T&A reduction | T&A b ÷ d |
|---|---|---|---|---|---|---|
| free | 38,407,242 | 98.86% | ×87.9 | 20,088,095 | 98.82% | ×84.4 |
| ssd | 41,465,414 | 98.77% | ×81.3 | 17,907,057 | 98.94% | ×94.8 |
| plat025 | 38,445,515 | 98.86% | ×87.7 | 24,299,176 | 98.57% | ×69.8 |
| plat4 | 38,440,796 | 98.86% | ×87.7 | 23,074,439 | 98.64% | ×73.5 |
| pbw025 | 37,316,198 | 98.89% | ×90.3 | 26,959,938 | 98.41% | ×62.9 |
| pbw4 | 40,693,662 | 98.79% | ×82.8 | 23,460,315 | 98.62% | ×72.3 |

The timing arms widen T&A's worst case to 98.41% (`pbw025`). They do not widen
Conversation's.

**Candidate, not a finding: outcomes cluster on a few discrete values.**
Exact repeats among the 16 pooled runs:

- **Conversation:** 43,025,695 (twice, j = 0.005 seeds 1 and 3) and 41,901,621
  (stock, and j = 0.005 seed 5).
- **Tool&Agent:** 17,185,636 (twice, j = 0.005 seeds 1 and 5) and 17,168,859
  (stock, and j = 0.005 seed 4).

All repeats occur at the smallest j. With 16 runs per trace this is suggestive
only.

### Corrections to STEP 3 (the text above stays as written)

1. **The periodic-burst-train candidate is not supported.** A non-periodic
   train reproduces the stock store outcome exactly: T&A, j = 0.005, seed 4.
   Exact periodicity is therefore not what places the stock run outside the
   j ≥ 0.05 replicates.
2. **T&A's j-dependence call is withdrawn.** Its spread is non-monotone in j:
   **33.26% / 14.06% / 30.01%** at j = 0.005 / 0.05 / 0.45. The prelude's "the
   spread depends on j" now stands for Conversation only (3.44% / 8.55% /
   25.99%, monotone). Every (d) range is still quoted with its j.
3. **The j = 0.05 cap 48 → 256 gap on T&A (+26.86%) is part of a non-monotone
   swing, not a batch effect.** The per-cap means fall from cap 8 to cap 64
   and rise again from cap 96 (STEP 3 table). A two-point difference picked
   from that series is not evidence of a batch dependence.

### Pooled bound — addendum: the worst case over every cap tested

The pooled bound above is cap-48 only. Over all 46 (d) runs per trace at every
cap tested, the worst-case (b) → (d) reduction is lower:

- Conversation: **98.66%** (`s2c_j3_c256`).
- Tool&Agent: **98.28%** (the stock cap-256 run).

These runs are the stock runs, the j = 0.005 / 0.05 / 0.45 replicates at cap 48
and 256, and the STEP 3 sweep. Residual (d) HBF B/req across them spans
32.0–45.4 M (+41.8%) on Conversation and 17.2–29.2 M (+70.1%) on Tool&Agent.
report.md's Pending LaTeX sync uses these all-cap figures.

## STEP 3a — Cross-rank eviction in the shared store

### The flag

| flag | values | where |
|---|---|---|
| **`store_partition`** | **`global`** (default, stock) \| `count` \| `per_rank` | `mooncake/config.py` (validated in `__post_init__`) |
| **`store_partition_ranks`** | required for `per_rank`, **no default**; must equal the world size (checked in the connector) | `mooncake/config.py`; `MooncakeStoreConnector._check_partition_ranks` |

**A rank is the key's (tp_rank, pp_rank, dp_rank),** read from
`PoolKey.key_metadata`. The modes:

- **`count`** records each memory-tier object's rank. It counts memory-tier
  evictions where the victim's rank differs from the rank of the admission
  that forced them. It tracks each rank's peak memory-tier occupancy. Nothing
  else changes.
- **`per_rank`** gives each rank its own memory-tier LRU and capacity
  (`_ensure_rank_capacity`, `_rank_lru`), and its own offload tier
  (`PartitionedOffloadTier`, `mooncake/ssd.py`). Capacities are **floored**:
  - PCM: floor(142,213 / 8) = **17,776 blocks per rank**, 5 blocks unused.
  - HBF: floor(10,066,329 / 8) = **1,258,291 blocks per rank**, 1 block unused.

  Note that 17,776 here is the per-GPU design value applied per rank. It is
  **not** the "17,776-block control" of `CLAUDE.md`: the pool knob stays
  `memory_capacity_blocks = 142213`.

Counters (`partition_*`) are merged after `asdict()`. The non-stock mode is
recorded in `mooncake_offload_profiles`.

**Byte identity at default:** `s3a_id_{b,d}_{conv,ta}` against `ced3806` →
**no diff, all four.** For (c), with no cap-48 result left on disk, the
default-code run `s3a_def_c_*` reproduces PHASE 9 STEP 3's recorded values
exactly:

| | HBF write bytes | HBF blocks | hit rate | reuse hit blocks | batch |
|---|---|---|---|---|---|
| Conversation | 8,240,926,556,160 | 3,143,664 | 0.1645 | 671,266 | 23.08 |
| Tool&Agent | 4,057,074,237,440 | 1,547,651 | 0.4040 | 1,174,945 | 22.64 |

**`count` is behaviourally stock:** against stock, (b), (c) and (d) on both
traces have no differing non-counter field. **Tests: 236 pass** (221 + 15 new
in `tests/test_store_partition.py`).

### Runs

(b), (c), (d), cap 48, 30 QPS, W = 5,000, unjittered, charged timing, stock
values. Driver `scripts/phase14/step3a.sh`, table
`scripts/phase14/step3a_table.py`.

### Conversation

| config | mode | HBF B/req | Δ vs stock | PCM B/req | Δ | hit rate | Δ | reuse hit blk | memory evictions | cross-rank / same-rank | cross-rank share | per-rank peak blk min / max | ssd evictions | batch/sched | batch gap vs stock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | stock | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | +0.00% | 670,585 | 3,143,420 | — | — | — | 0 | 23.1679 | +0.00% |
| (b) | count | 3,370,675,339 | +0.00% | 1,722,617,954 | +0.00% | 0.1643 | +0.00% | 670,585 | 3,143,420 | 2,373,254 / 770,166 | 75.50% | 42,933 / 54,093 | 0 | 23.1679 | +0.00% |
| (b) | per_rank | 3,418,975,371 | +1.43% | 1,728,127,173 | +0.32% | 0.1648 | +0.29% | 672,505 | 3,225,037 | 0 / 3,225,037 | 0.00% | 17,776 / 17,776 | 0 | 23.2174 | +0.21% |
| (c) | stock | 1,648,185,311 | +0.00% | 1,722,745,881 | +0.00% | 0.1645 | +0.00% | 671,266 | 3,143,664 | — | — | — | 0 | 23.0768 | +0.00% |
| (c) | count | 1,648,185,311 | +0.00% | 1,722,745,881 | +0.00% | 0.1645 | +0.00% | 671,266 | 3,143,664 | 2,314,646 / 829,018 | 73.63% | 43,197 / 50,528 | 0 | 23.0768 | +0.00% |
| (c) | per_rank | 1,681,900,700 | +2.05% | 1,719,179,674 | -0.21% | 0.1645 | -0.01% | 671,202 | 3,207,971 | 0 / 3,207,971 | 0.00% | 17,776 / 17,776 | 0 | 23.0033 | -0.32% |
| (d) | stock | 41,901,621 | +0.00% | 1,987,125,969 | +0.00% | 0.0974 | +0.00% | 397,596 | 3,647,929 | — | — | — | 0 | 22.3782 | +0.00% |
| (d) | count | 41,901,621 | +0.00% | 1,987,125,969 | +0.00% | 0.0974 | +0.00% | 397,596 | 3,647,929 | 2,756,168 / 891,761 | 75.55% | 45,509 / 71,892 | 0 | 22.3782 | +0.00% |
| (d) | per_rank | 10,543,432 (**outside pooled range**) | -74.84% | 2,042,432,061 | +2.78% | 0.0926 | -4.95% | 377,916 | 3,824,526 | 0 / 3,824,526 | 0.00% | 17,776 / 17,776 | 0 | 22.4946 | +0.52% |


### Tool&Agent

| config | mode | HBF B/req | Δ vs stock | PCM B/req | Δ | hit rate | Δ | reuse hit blk | memory evictions | cross-rank / same-rank | cross-rank share | per-rank peak blk min / max | ssd evictions | batch/sched | batch gap vs stock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | stock | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | +0.00% | 1,174,042 | 1,547,076 | — | — | — | 0 | 22.8850 | +0.00% |
| (b) | count | 1,696,787,333 | +0.00% | 885,673,951 | +0.00% | 0.4037 | +0.00% | 1,174,042 | 1,547,076 | 1,151,459 / 395,617 | 74.43% | 40,798 / 48,424 | 0 | 22.8850 | +0.00% |
| (b) | per_rank | 1,729,667,006 | +1.94% | 883,472,990 | -0.25% | 0.4036 | -0.01% | 1,173,911 | 1,613,987 | 0 / 1,613,987 | 0.00% | 17,776 / 17,776 | 0 | 22.9836 | +0.43% |
| (c) | stock | 811,414,847 | +0.00% | 885,975,417 | +0.00% | 0.4040 | +0.00% | 1,174,945 | 1,547,651 | — | — | — | 0 | 22.6431 | +0.00% |
| (c) | count | 811,414,847 | +0.00% | 885,975,417 | +0.00% | 0.4040 | +0.00% | 1,174,945 | 1,547,651 | 1,145,753 / 401,898 | 74.03% | 39,925 / 54,241 | 0 | 22.6431 | +0.00% |
| (c) | per_rank | 847,666,217 | +4.47% | 884,945,191 | -0.12% | 0.4040 | +0.01% | 1,175,105 | 1,616,795 | 0 / 1,616,795 | 0.00% | 17,776 / 17,776 | 0 | 22.8503 | +0.92% |
| (d) | stock | 17,168,859 | +0.00% | 1,001,932,194 | +0.00% | 0.3650 | +0.00% | 1,061,727 | 1,768,821 | — | — | — | 0 | 22.4295 | +0.00% |
| (d) | count | 17,168,859 | +0.00% | 1,001,932,194 | +0.00% | 0.3650 | +0.00% | 1,061,727 | 1,768,821 | 1,343,086 / 425,735 | 75.93% | 41,694 / 54,749 | 0 | 22.4295 | +0.00% |
| (d) | per_rank | 4,082,631 (**outside pooled range**) | -76.22% | 1,024,847,249 | +2.29% | 0.3671 | +0.56% | 1,067,647 | 1,883,637 | 0 / 1,883,637 | 0.00% | 17,776 / 17,776 | 0 | 22.4478 | +0.08% |


`per_rank` capacities: 17,776 PCM blocks and 1,258,291 HBF blocks per rank.
**Ranks seen: 4** (see the constraint below). `ssd_eviction_count = 0` in all
18 runs. **Batch gap vs stock:** `count` 0.00% in all six. `per_rank` −0.32%
to +0.92%.

### CONSTRAINT found in this step: only TP-rank-0 workers are scheduled; 4 of 8 ranks ever write to the store

**What the code does.** `LLMEngine._is_schedulable_worker`
(`TokenSim/llm/llm_engine.py:577-579`) returns `tp_rank == 0 and pp_rank == 0`
whenever `world_size > 1`. Only those workers enter the prefill and decode
pools. At TP2/DP4 that is **4 of the 8 workers**, one per DP group.

The TP-rank-1 workers never receive a request, never allocate HBM and never
call the connector. Measured:

- `partition_ranks_seen = 4` in every `count` and `per_rank` run.
- Per-rank peaks are listed for 4 ranks only, e.g. 47,412 / 42,933 / 52,318 /
  54,093 blocks in (b) Conversation.
- `batch_peak_cluster = 192` = 4 × 48 at cap 48.

**Consequence 1 — every "batch/sched" figure in this file is half the
active-scheduler batch.** `get_batch_stats` (`util/results.py:359-395`)
divides by all 8 workers that own a block manager, including the 4 idle ones.
Same-config gaps and every ratio between arms are unaffected, because they
share the denominator. Absolute values are not:

| run | reported batch/sched (÷ 8) | **per active scheduler (÷ 4)** | reported HBM mean | **HBM mean over active ranks** (inferred: idle ranks hold 0) |
|---|---|---|---|---|
| (b) Conv cap 48 | 23.168 | **46.336** | 26.43% | **52.86%** |
| (d) Conv cap 48 | 22.378 | **44.756** | 25.39% | **50.78%** |
| (b) T&A cap 48 | 22.885 | **45.770** | 25.41% | **50.83%** |
| (d) T&A cap 48 | 22.430 | **44.859** | 25.12% | **50.25%** |
| (b) Conv cap 256 | 43.609 | **87.218** | 46.90% | **93.80%** |
| (b) T&A cap 256 | 48.437 | **96.874** | 45.68% | **91.36%** |

So at cap 48 the active schedulers run at **~93–97% of the cap on average**,
not ~47%. PHASE 9 STEP 1's "8 schedulers, so cluster concurrency is
8 × max_parallem_sum" should read **4 ×**. Peak HBM occupancy is a max over
workers and is unaffected.

**Consequence 2 — the PCM pool sees half the shards it was sized for.** The
block size, 2,621,440 B, is one **TP shard**: TP2 splits the KV heads. In the
real system both TP ranks of a DP group hold, and would save, their shard. In
the simulator only the TP-0 shard reaches the store. Yet the pool is sized for
**all 8 GPUs**: 142,213 shard-blocks = 43.4 GiB × 8. So:

- **The PCM capacity relative to its write stream is 2× the design.** Each
  writing rank peaks at 41,694–54,749 blocks under `global`/`count`, against a
  per-GPU design share of 17,776.
- **HBF and PCM bytes/request are one TP shard's worth.** On the TP-0-only
  model a real TP2 deployment writes 2× these bytes. Every reduction
  percentage and every within-config comparison is unaffected in kind, but
  absolute bytes are halved.

This is a capacity-bookkeeping question of the kind `CLAUDE.md` flags as
not-negotiable, so it is reported here as a **constraint, not worked around**.
No run in this file is re-interpreted on its strength yet.

### What `per_rank` actually measured

Because only 4 ranks exist, `per_rank` at 8 partitions gave each writing rank
its per-GPU 17,776 blocks and left the other 4 partitions empty. **The usable
PCM capacity was therefore 71,104 blocks, half the pool.** The HBF partitions,
at 1,258,291 blocks per rank, are never full: 0 HBF evictions. So the `per_rank`
arm changes two things at once:

1. it removes cross-rank sharing (by construction: 0 cross-rank evictions);
2. it halves the usable PCM capacity.

The two cannot be separated from these runs.

**Numbers, read per the rule set before the runs** (single runs readable for
(b), (c), all PCM and hit rate; (d) HBF only outside the pooled cap-48 range):

| | Conversation | Tool&Agent | readable? |
|---|---|---|---|
| **cross-rank share of memory-tier evictions (`count`)** | (b) 75.50%, (c) 73.63%, (d) 75.55% | (b) 74.43%, (c) 74.03%, (d) 75.93% | yes (counter) |
| per-rank peak occupancy, min / max (`count`) | 42,933–54,093 (b) | 40,798–48,424 (b) | yes |
| (b) HBF B/req, `per_rank` vs stock | +1.43% | +1.94% | yes |
| (c) HBF B/req, `per_rank` vs stock | +2.05% | +4.47% | yes |
| PCM B/req, `per_rank` vs stock | +0.32% / −0.21% / +2.78% | −0.25% / −0.12% / +2.29% | yes |
| hit rate, `per_rank` vs stock | +0.29% / −0.01% / **−4.95%** | −0.01% / +0.01% / +0.56% | yes |
| **(d) HBF B/req, `per_rank`** | **10,543,432 (−74.84%)** | **4,082,631 (−76.22%)** | **outside the pooled range** (32.0–43.0 M / 17.2–26.3 M) → an effect |

Cross-rank eviction is **the majority**: about 74–76% of all memory-tier
evictions, in every config on both traces. With 4 equally loaded ranks, a
rank-blind LRU would give 75% by chance, and the measured shares sit at that
value.

**(d)'s HBF drop under `per_rank` is an effect by the pre-registered rule.** It
cannot be attributed to cross-rank sharing, because usable capacity halved at
the same time. Fewer blocks survive in PCM until their first read, so more are
dropped by `if_read` instead of demoted. This matches (d)'s lower hit rate on
Conversation, −4.95%.

### Not concluded — needs a decision

Whether the pool should be compared at "8 partitions, 4 used" or "4 partitions"
depends on how the TP-0-only model is meant to map to the hardware. That
mapping is consequence 2 above. It is left open here.

### Settings

| | |
|---|---|
| code | **`cd62320`** on `phase14-fixes` |
| runs | `tmp/phase14/runs/s3a_{id,def,cnt,pr}_*` (18) |
| hardware | none invented or changed; per-rank capacities are floor(pool / 8) of the sourced pools |

## STEP 3a — control runs: every tier sized for the 4 store-writing ranks

**Decision (after review): option (B).** Only TP-0/PP-0 workers are scheduled
(`llm_engine.py:577-579`), so 4 of 8 ranks write to the store. **The TP-1
stream is an exact mirror of TP-0 and is not simulated.** Every tier is
therefore sized for the 4 writing ranks:

| tier | per writing GPU | × 4 writing ranks | blocks |
|---|---|---|---|
| PCM | 43.4 GiB (the design input; findings ~1997 / ~2769) | 173.6 GiB | **4 × 17,776 = 71,104** |
| HBF | 3 TiB = 1,258,291 blocks | 12 TiB | **4 × 1,258,291 = 5,033,164** |

`store_partition_ranks` counts **writing ranks only**. The connector check
(`_check_partition_ranks`) now requires it to equal `data_parallel_size`, one
TP-0 worker per DP group, not `world_size`. Tests: **237 pass** (236, with the
world-size test replaced by two writer-count tests).

**Labels.**

- `global` @ 71,104 is the **"43.4 GiB/GPU design point"**. It runs as
  `store_partition = count`, which is behaviourally global (STEP 3a), so the
  cross-rank share is available.
- `per_rank` is 4 partitions of 17,776 PCM / 1,258,291 HBF blocks.
- **The PHASE 9 headline is not replaced.** That decision is pending.

**Runs.** Cap 48, both traces, 30 QPS, W = 5,000, charged timing:

- (b) and (c): one unjittered run each.
- (d): the unjittered run plus burst jitter j = 0.05 × gap (15.685 /
  30.287 ms), seeds 1–3.

That is 24 runs (`scripts/phase14/step3a_ctrl.sh`, table
`step3a_ctrl_table.py`). The comparison arm "global @ 142,213" is the PHASE 9
pool, using the same composition: `s0_b`, `s3a_def_c`, and `s0_d` +
`p3_js{1,2,3}`.

**Labels on the columns:**

- **B/req are per TP shard** (one of 2).
- **Batch is per active scheduler (TP group)**, i.e. `batch_mean_cluster / 4`.
- `notdone = 0` and HBF evictions = 0 in all runs. Ranks seen = 4 in every
  partition-mode run.

### Conversation

| config | arm | N | HBF B/req (per TP shard) | PCM B/req (per TP shard) | hit rate | reuse hit blk | cross-rank share | mean PCM residency s | batch / active sched | HBF evictions | ranks seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | global @ 142,213 (PHASE 9 pool) | 1 | 3,370,675,339 | 1,722,617,954 | 0.1643 | 670,585 | — | 113.34 | 46.336 | 0 | None |
| (b) | global @ 71,104 — 43.4 GiB/GPU design point | 1 | 3,411,597,591 | 1,724,438,282 | 0.1648 | 672,601 | 74.49% | 56.80 | 46.323 | 0 | 4 |
| (b) | per_rank 4 × 17,776 | 1 | 3,418,975,371 | 1,728,127,173 | 0.1648 | 672,505 | — | 56.42 | 46.435 | 0 | 4 |
| (c) | global @ 142,213 (PHASE 9 pool) | 1 | 1,648,185,311 | 1,722,745,881 | 0.1645 | 671,266 | — | 81.22 | 46.154 | 0 | None |
| (c) | global @ 71,104 — 43.4 GiB/GPU design point | 1 | 1,685,218,918 | 1,722,497,892 | 0.1633 | 666,233 | 75.43% | 40.85 | 46.268 | 0 | 4 |
| (c) | per_rank 4 × 17,776 | 1 | 1,681,900,700 | 1,719,179,674 | 0.1645 | 671,202 | — | 40.67 | 46.007 | 0 | 4 |
| (d) | global @ 142,213 (PHASE 9 pool) | 4 | 41,023,832 [39,653,474, 41,901,621] | 1,988,701,585 [1,987,125,969, 1,990,650,757] | 0.0968 [0.0948, 0.0977] | 395,108 [387,068, 398,652] | — | 42.11 [42.02, 42.16] | 44.926 [44.756, 45.159] | 0 | None |
| (d) | global @ 71,104 — 43.4 GiB/GPU design point | 4 | 11,885,347 [10,942,939, 12,227,445] | 2,036,713,259 [2,034,385,814, 2,041,517,703] | 0.0927 [0.0894, 0.0946] | 378,100 [364,796, 386,140] | 75.57% [74.92%, 76.40%] | 20.44 [20.36, 20.48] | 45.073 [44.932, 45.349] | 0 | 4 |
| (d) | per_rank 4 × 17,776 | 4 | 10,251,403 [9,677,308, 10,543,432] | 2,042,366,394 [2,042,045,137, 2,042,556,318] | 0.0929 [0.0924, 0.0939] | 379,044 [376,988, 383,356] | — | 20.11 [20.11, 20.11] | 45.013 [44.986, 45.083] | 0 | 4 |

(b) Conversation: sharing (per_rank vs global@71k) HBF +0.22%, PCM +0.21%, hit -0.01%, batch +0.24% | capacity (global@71k vs global@142k) HBF +1.21%, PCM +0.11%, hit +0.30%, batch -0.03%
(c) Conversation: sharing (per_rank vs global@71k) HBF -0.20%, PCM -0.19%, hit +0.75%, batch -0.57% | capacity (global@71k vs global@142k) HBF +2.25%, PCM -0.01%, hit -0.75%, batch +0.25%
(d) Conversation: sharing (per_rank vs global@71k) HBF -13.75%, PCM +0.28%, hit +0.25%, batch -0.13% | capacity (global@71k vs global@142k) HBF -71.03%, PCM +2.41%, hit -4.30%, batch +0.33% | (d) HBF ranges: g142 39,653,474-41,901,621, g71 10,942,939-12,227,445, p71 9,677,308-10,543,432; sharing ranges overlap: NO; capacity ranges overlap: NO
global @ 142,213 (PHASE 9 pool): (b)->(c) 51.10%; (b)->(d) 98.76–98.82% (b ÷ d ×80.4–×85.0); batch gap (d) vs (b) -3.04%, (c) vs (b) -0.39%
global @ 71,104 — 43.4 GiB/GPU design point: (b)->(c) 50.60%; (b)->(d) 99.64–99.68% (b ÷ d ×279.0–×311.8); batch gap (d) vs (b) -2.70%, (c) vs (b) -0.12%
per_rank 4 × 17,776: (b)->(c) 50.81%; (b)->(d) 99.69–99.72% (b ÷ d ×324.3–×353.3); batch gap (d) vs (b) -3.06%, (c) vs (b) -0.92%

### Tool&Agent

| config | arm | N | HBF B/req (per TP shard) | PCM B/req (per TP shard) | hit rate | reuse hit blk | cross-rank share | mean PCM residency s | batch / active sched | HBF evictions | ranks seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | global @ 142,213 (PHASE 9 pool) | 1 | 1,696,787,333 | 885,673,951 | 0.4037 | 1,174,042 | — | 113.02 | 45.770 | 0 | None |
| (b) | global @ 71,104 — 43.4 GiB/GPU design point | 1 | 1,730,365,358 | 883,822,166 | 0.4037 | 1,174,071 | 74.23% | 56.97 | 46.090 | 0 | 4 |
| (b) | per_rank 4 × 17,776 | 1 | 1,729,667,006 | 883,472,990 | 0.4036 | 1,173,911 | — | 56.35 | 45.967 | 0 | 4 |
| (c) | global @ 142,213 (PHASE 9 pool) | 1 | 811,414,847 | 885,975,417 | 0.4040 | 1,174,945 | — | 80.80 | 45.286 | 0 | None |
| (c) | global @ 71,104 — 43.4 GiB/GPU design point | 1 | 848,324,723 | 885,603,697 | 0.4039 | 1,174,785 | 72.94% | 40.97 | 45.718 | 0 | 4 |
| (c) | per_rank 4 × 17,776 | 1 | 847,666,217 | 884,945,191 | 0.4040 | 1,175,105 | — | 40.37 | 45.701 | 0 | 4 |
| (d) | global @ 142,213 (PHASE 9 pool) | 4 | 20,625,359 [17,168,859, 23,678,419] | 1,000,597,094 [998,507,020, 1,003,026,383] | 0.3668 [0.3650, 0.3677] | 1,066,919 [1,061,727, 1,069,567] | — | 42.84 [42.71, 42.93] | 44.775 [44.478, 44.913] | 0 | None |
| (d) | global @ 71,104 — 43.4 GiB/GPU design point | 4 | 5,403,836 [4,854,383, 5,944,902] | 1,023,349,883 [1,022,163,419, 1,024,544,735] | 0.3665 [0.3663, 0.3666] | 1,065,887 [1,065,535, 1,066,239] | 74.73% [72.97%, 76.37%] | 20.66 [20.64, 20.69] | 45.036 [45.001, 45.088] | 0 | 4 |
| (d) | per_rank 4 × 17,776 | 4 | 4,082,631 [4,082,631, 4,082,631] | 1,024,994,181 [1,024,847,249, 1,025,434,976] | 0.3670 [0.3669, 0.3671] | 1,067,447 [1,067,199, 1,067,647] | — | 20.37 [20.36, 20.39] | 44.894 [44.882, 44.911] | 0 | 4 |

(b) Tool&Agent: sharing (per_rank vs global@71k) HBF -0.04%, PCM -0.04%, hit -0.01%, batch -0.27% | capacity (global@71k vs global@142k) HBF +1.98%, PCM -0.21%, hit +0.00%, batch +0.70%
(c) Tool&Agent: sharing (per_rank vs global@71k) HBF -0.08%, PCM -0.07%, hit +0.03%, batch -0.04% | capacity (global@71k vs global@142k) HBF +4.55%, PCM -0.04%, hit -0.01%, batch +0.95%
(d) Tool&Agent: sharing (per_rank vs global@71k) HBF -24.45%, PCM +0.16%, hit +0.15%, batch -0.31% | capacity (global@71k vs global@142k) HBF -73.80%, PCM +2.27%, hit -0.10%, batch +0.58% | (d) HBF ranges: g142 17,168,859-23,678,419, g71 4,854,383-5,944,902, p71 4,082,631-4,082,631; sharing ranges overlap: NO; capacity ranges overlap: NO
global @ 142,213 (PHASE 9 pool): (b)->(c) 52.18%; (b)->(d) 98.60–98.99% (b ÷ d ×71.7–×98.8); batch gap (d) vs (b) -2.17%, (c) vs (b) -1.06%
global @ 71,104 — 43.4 GiB/GPU design point: (b)->(c) 50.97%; (b)->(d) 99.66–99.72% (b ÷ d ×291.1–×356.5); batch gap (d) vs (b) -2.29%, (c) vs (b) -0.81%
per_rank 4 × 17,776: (b)->(c) 50.99%; (b)->(d) 99.76–99.76% (b ÷ d ×423.7–×423.7); batch gap (d) vs (b) -2.33%, (c) vs (b) -0.58%

### Reading, per the rule set before the runs

The **sharing effect** is per_rank vs global @ 71,104. The **capacity effect**
is global @ 71,104 vs global @ 142,213. For (d) HBF, a difference counts as an
effect only if the ranges do not overlap. Single runs are readable for (b), (c)
and all PCM / hit-rate values (spread < 1%).

| | sharing: (b) HBF | sharing: (c) HBF | **sharing: (d) HBF** | capacity: (b) HBF | capacity: (c) HBF | **capacity: (d) HBF** |
|---|---|---|---|---|---|---|
| Conversation | +0.22% | −0.20% | **−13.75% (ranges do not overlap → effect)** | +1.21% | +2.25% | **−71.03% (ranges do not overlap → effect)** |
| Tool&Agent | −0.04% | −0.08% | **−24.45% (ranges do not overlap → effect)** | +1.98% | +4.55% | **−73.80% (ranges do not overlap → effect)** |

The (d) ranges behind those calls:

| (d) HBF B/req range | global @ 142,213 | global @ 71,104 | per_rank |
|---|---|---|---|
| Conversation | 39.7–41.9 M | 10.9–12.2 M | 9.7–10.5 M |
| Tool&Agent | 17.2–23.7 M | 4.85–5.94 M | 4.08 M (all four runs identical) |

1. **The capacity effect is large for (d) and small for (b) and (c).** At the
   43.4 GiB/GPU design point:
   - mean PCM residency halves: (b) 113.3 → 56.8 s Conv, 113.0 → 57.0 s T&A;
     (d) 42.1 → 20.4 s and 42.8 → 20.7 s;
   - fewer blocks are read while resident, so `if_read` drops more and demotes
     fewer;
   - (d) HBF falls by ~71–74%, and (d) PCM B/req rises by 2.4% / 2.3%;
   - (d) hit rate falls 4.30% on Conversation and 0.10% on Tool&Agent;
   - (b) and (c) write 1.2–4.6% more to HBF.
2. **The sharing effect is negligible for (b) and (c) (≤ 0.22%) but is an
   effect for (d) by the rule:** −13.75% / −24.45%, a few million B/req in
   absolute terms. Partitioning lowers (d)'s HBF writes further.
   - Cross-rank evictions are **72.9–76.4%** of memory-tier evictions at
     71,104, in every config, unchanged from 142,213.
   - All four per_rank T&A (d) runs give **the same** HBF value, 4,082,631
     B/req: the jitter seeds did not move it. This fits the discrete-outcome
     candidate recorded earlier.
3. **Reductions at each pool size** (single (b)/(c); (d) range over 4 runs):

   | | (b) → (c) | (b) → (d) | b ÷ d |
   |---|---|---|---|
   | Conv, global @ 142,213 | 51.10% | 98.76–98.82% | ×80.4–×85.0 |
   | Conv, **global @ 71,104 (design point)** | **50.60%** | **99.64–99.68%** | ×279.0–×311.8 |
   | Conv, per_rank | 50.81% | 99.69–99.72% | ×324.3–×353.3 |
   | T&A, global @ 142,213 | 52.18% | 98.60–98.99% | ×71.7–×98.8 |
   | T&A, **global @ 71,104 (design point)** | **50.97%** | **99.66–99.72%** | ×291.1–×356.5 |
   | T&A, per_rank | 50.99% | 99.76% | ×423.7 |

**Batch:**

- Per active scheduler: 44.9–46.4 in every arm.
- **Gaps (d) vs (b)**, within each arm: −2.70% to −3.06% (Conv) and −2.17% to
  −2.33% (T&A).
- **Gaps (c) vs (b):** −0.12% to −1.06%.
- **Across arms:** capacity −0.03% to +0.95%, sharing −0.57% to +0.24%.

**Labelled as instructed: the 71,104 runs are the "43.4 GiB/GPU design point".
They are a control here, and the PHASE 9 headline (142,213) is unchanged until
decided.**

## CORRECTION NOTE — audit of every place that divides by 8 workers or assumes 8 writers

**Points back to** PHASE 5 (corrected), PHASE 8, PHASE 9, PHASE 10, PHASE 12
and the earlier PHASE 14 steps. None of that text is edited. The cause, in all
cases: **only the 4 TP-0 workers are scheduled and write to the store**
(STEP 3a constraint).

### Code (stock output, deliberately unchanged)

| where | what | effect |
|---|---|---|
| `util/results.py:382-395` (`get_batch_stats`) | `batch_mean_per_scheduler`, `hbm_mean_used_blocks_per_rank`, `hbm_mean_occupancy_frac` divide by `len(workers)` = 8, which includes the 4 idle TP-1 workers. `batch_scheduler_count` = 8 | means are **½** the per-active-scheduler value. `batch_peak_*` and `hbm_peak_*` are maxima and are unaffected. `batch_mean_cluster` is correct; **per active scheduler = `batch_mean_cluster / 4`** |
| `util/results.py:93,100` (`get_parallel_stats`) | `parallel_*_rank_count` = 8 | correct as a worker count, but only 4 run requests |
| `mooncake/store.py` capacity, via `run_one.py` BASE `memory_capacity_blocks = 142213`, `ssd_capacity_blocks = 10066329` | sized for 8 writers | **2× the design capacity** relative to the TP-0 write stream |

### findings.md, by kind

1. **"43.4 GiB/GPU" attached to 142,213-block results means 86.8 GiB per
   writing GPU.** This covers PHASE 5 corrected (~1944–2121), the PHASE 8
   STEP 1 block table (~2769), PHASE 9 STEP 3 "at 43.4 GiB per GPU the PCM
   tier is still too small…" (~3385, ~3633), PHASE 10 throughout (~3857–4069)
   and the PHASE 12 Settings (~4529). The **design point** (43.4 GiB per
   writing GPU) is 71,104 blocks, measured above.
2. **PHASE 10 capacity sweep.**
   - Its **GiB/GPU column is ×2** per writing GPU.
   - The knee at "~822 GiB/GPU, 19× the budget" (~3949, ~4061) is
     **~1,645 GiB per writing GPU, ~38× the design point**.
   - The residency/TTFR crossing at "1.8–2.8×, 77–120 GiB/GPU" (~4001–4016) is
     **3.6–5.6×, 154–240 GiB** per writing GPU.
   - "43.4 GiB/GPU = 5.3% / 10.2% of footprint" (~3906, ~4062) refers to
     142,213 blocks. **The design point, 71,104 blocks, is 2.6% / 5.1%** of the
     footprint (2,695,220 / 1,397,479 blocks).
   - Its "requests" columns share the error in item 3.
3. **PHASE 9 STEP 4 "requests to wear out"** (~3641–3668), and the PHASE 10
   "requests" columns, divide the **24 TiB × 3000 P/E** budget by **one TP
   shard's** bytes per request. Only the TP-0 half of the HBF (12 TiB) receives
   those bytes; the TP-1 half receives the mirror stream. **Every absolute
   requests-to-wear-out figure is 2× too high. Ratios between columns (×2.05,
   ×2.09, etc.) are unaffected.**
4. **Absolute B/req are per TP shard.** That is every HBF, PCM and memory-tier
   bytes/request, bytes/token and total write-byte figure in this file.
   Physical per-cluster write bytes are 2× these. Reductions and ratios are
   unaffected.
5. **Batch and HBM-mean figures are ÷ 8 workers.** That is every "batch/sched",
   "mean batch / scheduler", "achieved batch" and "mean HBM occupancy" in PHASE
   9 STEPs 2–4, PHASE 11, 12 and 13, and PHASE 14 up to this point. Per active
   scheduler (TP group) the value is **2×**; e.g. cap 48 (b) Conversation is
   23.168 → **46.336**. PHASE 9 STEP 1 "8 schedulers, so cluster concurrency is
   8 × `max_parallem_sum`" (~3117–3118, also ~3143, ~3206 "384 cluster-wide")
   reads **4 schedulers, 4 × `max_parallem_sum`, 192 cluster-wide**. Batch
   *gaps* between arms are unaffected.
6. **PCM residency at 142,213** (PHASE 9 STEP 3, PHASE 10) is the residency of a
   2×-design pool. At the design point it measures about half (above).

### Also assumes 8 writers, not edited here

`CLAUDE.md`, "Capacity bookkeeping" ("pool value = per-GPU design value × 8";
"`memory_capacity_blocks = 142213` is THE correct configuration"). It is
gitignored and is the project's instruction file, so it is flagged to the user
rather than changed.

## DECISION RECORD — headline configuration (after the STEP 3a controls)

**These decisions were made on hardware grounds, not on the control-run
results.** They are recorded as such, and they would stand even if the results
had gone the other way.

1. **Headline = design capacity + `per_rank`.**
   - **Capacity.** 43.4 GiB/GPU is fixed by die area: 2 PCM dies per stack,
     each the area of a 1 Tbit TLC die. The pool sized for the 4 writing
     (TP-0) ranks is therefore the design point: **71,104 blocks, or 66,220
     pending the unit check below.** The PHASE 9 pool of 142,213 blocks
     becomes the **86.8 GiB/GPU point**, which corresponds to 4 PCM dies per
     stack.
   - **Partition.** PCM sits in-package on each GPU's own HBF stacks, so the
     per-GPU pools are private. **`per_rank` is the faithful model;** `global`
     is reported as a variant.
   - **What it moves**, from the STEP 3a controls at 71,104:
     - (c) goes slightly **down**: (b)→(c) 51.10% / 52.18% → 50.81% / 50.99%
       under per_rank (50.60% / 50.97% global).
     - (d) goes **up**: (b)→(d) 98.6–99.0% → 99.69–99.76% under per_rank.
     - Both directions are stated here, so the decision cannot be read as
       chosen for the (d) number.
2. **The PHASE 9 STEP 3 cap-48 figures (142,213, global) are not deleted.**
   From now on they are quoted as "the 86.8 GiB/GPU point, global pool".

**No headline run has been made yet.** It waits on the unit check.

## OPEN — unit check before any headline run

The LaTeX turns 43.4 Gbit/stack into 5.43 GiB, i.e. it reads Gbit as **binary**
(2^30 bit). `CLAUDE.md` adopted the same reading. If the 0.62 Gb/mm² source
is **decimal** (10^9 bit), the design changes:

| reading of "Gb" | per stack | per GPU | blocks per writing rank (÷ 2,621,440 B) | pool, 4 writing ranks (`per_rank` 4 × …) |
|---|---|---|---|---|
| binary (2^30 bit) | 5.425 GiB | **43.4 GiB** | 17,776.64 → **17,776** | **71,104** |
| decimal (10^9 bit) | 5.425 GB = 5.052 GiB | **40.42 GiB** | 16,555.79 → **16,555** | **66,220** |

**Both are recorded; neither is chosen.** No headline run until the user
confirms.

**The same question runs through the rest of the chain, and should be settled
together:**

- **The 35 mm² die is "the area of a 1 Tbit TLC die" at 29 Gb/mm².**
  1 Tbit = 1024 Gbit gives 1024 / 29 = **35.31 mm²**, with 128 GiB per die (the
  LaTeX l.215–216 figures). 1 Tbit = 1000 Gbit gives **34.48 mm²**. So the PCM
  die area itself depends on the same unit reading.
- **HBF capacity, 384 GB per stack.** Read as GiB (as `CLAUDE.md` does): 3 TiB
  per GPU, **1,258,291** blocks per writing rank. Read as decimal GB:
  3.072 TB = 2.794 TiB per GPU, **1,171,875** blocks per rank (pool 4,687,500).
  HBF never evicts in any run so far, so this changes no measured number, but
  it does change the endurance budget.
- **TLC sets two things here:** PCM capacity, through die area, and HBF
  endurance, through 3000 P/E. One sourcing exercise should cover both.

## FLAG — under `if_read`, a smaller PCM gives FEWER HBF writes

Under `if_read`, a block reaches HBF only if it was read while resident in
PCM. A smaller pool shortens residency, so more blocks are dropped unread and
fewer are demoted. Part of (d)'s gain at the design point can therefore be
**lost reuse rather than better absorption**.

**From now on, every (d) result reports two numbers next to the reduction:**

- hit rate vs (b) at the same pool;
- HBF bytes written per reuse-hit block, for (b), (c) and (d).

Computed now for the STEP 3a control arms, with the 86.8 GiB/GPU point for
reference. (d) is the range over 4 runs: unjittered + j = 0.05 seeds 1–3.

#### Conversation

| pool / partition | config | N | hit rate | hit rate vs (b), same pool | reuse hit blocks | HBF write bytes per reuse-hit block | HBF B/req |
|---|---|---|---|---|---|---|---|
| global @ 142,213 (86.8 GiB/GPU point) | (b) | 1 | 0.1643 | — | 670,585 | 25,132,350 | 3,370,675,339 |
| global @ 142,213 (86.8 GiB/GPU point) | (c) | 1 | 0.1645 | +0.10% | 671,266 | 12,276,693 | 1,648,185,311 |
| global @ 142,213 (86.8 GiB/GPU point) | (d) | 4 | 0.0948 – 0.0977 | -42.28% – -40.55% | 387,068 – 398,652 | 497,344 – 526,940 | 39,653,474 – 41,901,621 |
| global @ 71,104 (design capacity, variant) | (b) | 1 | 0.1648 | — | 672,601 | 25,361,229 | 3,411,597,591 |
| global @ 71,104 (design capacity, variant) | (c) | 1 | 0.1633 | -0.95% | 666,233 | 12,647,369 | 1,685,218,918 |
| global @ 71,104 (design capacity, variant) | (d) | 4 | 0.0894 – 0.0946 | -45.76% – -42.59% | 364,796 – 386,140 | 149,987 – 161,460 | 10,942,939 – 12,227,445 |
| per_rank 4 × 17,776 (design capacity, faithful) | (b) | 1 | 0.1648 | — | 672,505 | 25,419,702 | 3,418,975,371 |
| per_rank 4 × 17,776 (design capacity, faithful) | (c) | 1 | 0.1645 | -0.19% | 671,202 | 12,529,020 | 1,681,900,700 |
| per_rank 4 × 17,776 (design capacity, faithful) | (d) | 4 | 0.0924 – 0.0939 | -43.94% – -43.00% | 376,988 – 383,356 | 126,218 – 139,494 | 9,677,308 – 10,543,432 |

#### Tool&Agent

| pool / partition | config | N | hit rate | hit rate vs (b), same pool | reuse hit blocks | HBF write bytes per reuse-hit block | HBF B/req |
|---|---|---|---|---|---|---|---|
| global @ 142,213 (86.8 GiB/GPU point) | (b) | 1 | 0.4037 | — | 1,174,042 | 7,226,263 | 1,696,787,333 |
| global @ 142,213 (86.8 GiB/GPU point) | (c) | 1 | 0.4040 | +0.08% | 1,174,945 | 3,452,991 | 811,414,847 |
| global @ 142,213 (86.8 GiB/GPU point) | (d) | 4 | 0.3650 – 0.3677 | -9.57% – -8.90% | 1,061,727 – 1,069,567 | 80,853 – 110,838 | 17,168,859 – 23,678,419 |
| global @ 71,104 (design capacity, variant) | (b) | 1 | 0.4037 | — | 1,174,071 | 7,369,083 | 1,730,365,358 |
| global @ 71,104 (design capacity, variant) | (c) | 1 | 0.4039 | +0.06% | 1,174,785 | 3,610,553 | 848,324,723 |
| global @ 71,104 (design capacity, variant) | (d) | 4 | 0.3663 – 0.3666 | -9.24% – -9.18% | 1,065,535 – 1,066,239 | 22,779 – 27,878 | 4,854,383 – 5,944,902 |
| per_rank 4 × 17,776 (design capacity, faithful) | (b) | 1 | 0.4036 | — | 1,173,911 | 7,367,113 | 1,729,667,006 |
| per_rank 4 × 17,776 (design capacity, faithful) | (c) | 1 | 0.4040 | +0.10% | 1,175,105 | 3,606,768 | 847,666,217 |
| per_rank 4 × 17,776 (design capacity, faithful) | (d) | 4 | 0.3669 – 0.3671 | -9.09% – -9.05% | 1,067,199 – 1,067,647 | 19,120 – 19,128 | 4,082,631 – 4,082,631 |

**What the numbers say about lost reuse at the design point** (per_rank @ 71,104
vs global @ 142,213):

| | (d) hit rate vs (b), same pool | (d) reuse-hit blocks | (d) HBF bytes per reuse-hit block |
|---|---|---|---|
| **Conversation** | −40.55% to −42.28% → **−43.00% to −43.94%** | 387,068–398,652 → **376,988–383,356** (about −2.5%) | 497,344–526,940 → **126,218–139,494** |
| **Tool&Agent** | −8.90% to −9.57% → **−9.05% to −9.09%** | 1,061,727–1,069,567 → **1,067,199–1,067,647** (inside the old range) | 80,853–110,838 → **19,120–19,128** |

- **Conversation:** some reuse is lost at the design point. The hit-rate cost of
  `if_read` widens by ~1.5–3 points and reuse-hit blocks fall ~2.5%. That is
  small next to a ~4× fall in HBF bytes per reuse-hit block.
- **Tool&Agent:** no reuse loss is detectable. Reuse-hit blocks at the design
  point fall inside the 142,213 range.
- So on these traces **most of (d)'s gain at the design point is not lost
  reuse.** The mechanism the flag describes is real, and the per-reuse-hit
  metric will show it wherever it dominates. This is recorded, not concluded,
  because each arm has only 4 runs.
- (b) and (c) are unaffected:
  - (b) spends ~25.1–25.4 M (Conv) and ~7.2–7.4 M (T&A) HBF bytes per
    reuse-hit block.
  - (c) spends about half that at every pool.
  - Their hit rates match (b)'s within 1%.

## NOTE — CLAUDE.md updated

The "Capacity bookkeeping" section of `CLAUDE.md` now states:

- **pool = per-GPU capacity × number of WRITING ranks** (TP-0 only = DP size =
  4);
- design pool **71,104** (or 66,220, pending the unit check);
- 142,213 = the 86.8 GiB/GPU point;
- HBF pool 5,033,164 for the writing ranks;
- `store_partition_ranks` = DP size.

`CLAUDE.md` is gitignored, so the change is on disk only.

## RESOLVED — unit check: binary throughout (user confirmation, 2026-10-11)

**Points back to** "OPEN — unit check" above. **Resolved; the binary reading
stands:**

| quantity | reading | basis (as given by the user) | consequence |
|---|---|---|---|
| 0.62 Gb/mm² (PCM density) | **binary** | TechInsights lists the 1st-gen die as "128Gb (16GB)" | **headline pool 71,104 = `per_rank` 4 × 17,776 = 43.4 GiB/GPU** |
| 1 Tbit TLC die | **binary** | → 1024 / 29 = 35.31 mm², 128 GiB | the design keeps **35 mm²**; that rounding costs **< 1% capacity** (35 / 35.31 = 0.991; 35.31 mm² would give 43.79 GiB/GPU) |
| 384 GB per HBF stack | **binary** | H3 sizes HBF at ~16× its 192 GB HBM: 16 × 192 = 3,072 = 8 × 384 | **HBF stays 5,033,164 = 4 × 1,258,291** |
| 29 Gb/mm² (TLC density) | **OPEN** | still uncited | its unit stays OPEN until sourced |

The decimal alternative (66,220 / 16,555) is **not** used.

## DECISION — PCM density: 1st-gen XPoint, 0.62 Gb/mm² is the headline density

- **Citation:** Ehrmann *et al.*, *Applied Research* 2022, e202200024,
  doi:10.1002/appl.202200024. It gives 0.62 Gb/mm² for 20 nm Optane, and takes
  that value from TechInsights (its ref. [127]). The LaTeX's "derived from
  published die parameters" now cites it, at the contribution item and at the
  density line (`\cite{ehrmann2022}`). The bib entry carries a TODO for title
  and full author list. Source details are as supplied by the user; they were
  not independently re-checked here.
- **LIMITATION, higher-density PCM.** 2nd-gen XPoint (4-deck) roughly doubles
  density. That would put the tier near the existing **86.8 GiB/GPU point**
  (PHASE 9, 142,213 blocks, global pool), which is reported as an
  **approximate higher-density point**. No new runs are made for it. It is
  approximate because that point used the global pool, not `per_rank`.
- **WORDING RULE.** Do not call 0.62 Gb/mm² "conservative" overall. A smaller
  pool **lowers** (c)'s reduction slightly (51.10 / 52.18% → ~50.8 / 51.0%) but
  **raises** (d)'s HBF reduction (→ ~99.7–99.8%) (STEP 3a controls). Its
  direction depends on the policy.

## NOTE — cost metrics: lost reuse explains little of (d)'s gain at the design point

**Recorded (user-confirmed reading of the decision-record metrics):**

- On **Tool&Agent**, (d) shows no detectable reuse loss while its HBF writes
  fall **−74%**: reuse-hit blocks are 1,067,199–1,067,647 at the design point,
  against 1,061,727–1,069,567 at 142,213.
- On **Conversation**, reuse loss is **~2.5%** while HBF writes fall **−71%**.

**Candidate mechanism, not a finding.** With a larger pool, more blocks get
exactly one read while in PCM. They pass the `if_read` gate and are written to
HBF, but are never read from HBF again. A smaller pool drops them unread
instead, saving the HBF write at no reuse cost.

**Test: new counter `count_hbf_readback`** (default off; stock output
byte-identical when off). For every block written to HBF, it records whether
that block is read back from HBF before the run ends.

- `hbf_readback_written_blocks`: successful offload writes. A re-write of the
  same key starts a new instance.
- `hbf_readback_read_blocks`: instances read from HBF at least once before the
  next write of that key or the end of the run.
- Reads served from the memory tier do not count.
- Code: `mooncake/store.py` `_record_offload_write` / `_record_offload_read`,
  merged into the result after `asdict()` (prefix `hbf_readback_`).
- Tests: **242 pass** (237 + 5 in `tests/test_hbf_readback.py`).

It runs on the headline set and on the 142,213 global stock configs (b), (c)
and (d) at cap 48, so the two pools can be compared. Results follow in the
headline-set entry.

## HEADLINE SET — design capacity (43.4 GiB/GPU), per_rank, cap 48

**Configuration.** This is the headline, decided on hardware grounds (see the
DECISION RECORD):

| setting | value |
|---|---|
| PCM | `memory_capacity_blocks = 71104`, `store_partition = per_rank`, `store_partition_ranks = 4` (4 × 17,776 = 43.4 GiB per writing GPU, binary) |
| HBF | `ssd_capacity_blocks = 5033164` (4 × 1,258,291) |
| other flags | `count_hbf_readback = true` |
| load | cap 48, 30 QPS, W = 5,000, charged timing, stock values |
| model | `LLaMa2-70B-GQA` (Nhead 64, KV heads 8, Nlayer 80, Dmodel 8192), **FP16**, 163,840 B/token at TP2, block 2,621,440 B, TP2/PP1/DP4 (4 writing TP-0 ranks; the TP-1 stream is a mirror and is not simulated) |
| hardware | `H3-B200-KVONLY` (md5 `9b4cba823dd4158d9bd3efe16bf74491`, pre-compensated `Capacity`, 78,643 HBM blocks per rank) |
| workload | Mooncake FAST'25 traces, **a stand-in for CAG** |
| runs | (b), (c): single unjittered runs. (d): pooled set of **N = 16 per trace**: unjittered + burst-jitter seeds 1–5 at j = 0.005 / 0.05 / 0.45 × min inter-burst gap (Conv 1.569 / 15.685 / 141.168 ms; T&A 3.029 / 30.287 / 272.582 ms) |

Driver: `scripts/phase14/headline.sh`. Table:
`scripts/phase14/headline_table.py`. Mechanism check:
`scripts/phase14/headline_readback.py`. The jitter j are methodology choices,
not hardware numbers.

**Column units:** B/req are **per TP shard**; batch is **per active scheduler
(TP group)**. (d) reductions are against the single unjittered (b) run;
burst jitter moved (b) HBF by ≤ 0.5% wherever it was measured.

**Checks, all passed:**

- `hl_id_{b,d}_*` (new code, flags at default) are byte-identical to
  `ced3806`, all four.
- `count_hbf_readback` changes no non-counter field: `rb142_*` against stock,
  all six.
- The headline (b), (c) and unjittered (d) runs reproduce the STEP 3a per_rank
  controls (`c48p_*`) exactly in every non-counter field.
- `notdone = 0`, HBF evictions = 0 and cross-rank evictions = 0 in all 36
  headline runs.

### Conversation

**Headline — 71,104 blocks, per_rank 4 × 17,776, 43.4 GiB/GPU** ((d): N = 16)

| config | N | HBF B/req | (b)→ reduction | PCM B/req | hit rate | hit rate vs (b) | reuse-hit blk | HBF bytes / reuse-hit blk | HBF-written blk | read back from HBF | read-back share | batch / active sched | batch gap vs (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 3,418,975,371 | — | 1,728,127,173 | 0.1648 | — | 672,505 | 25,419,702 | 6,521,178 | 254,446 | 3.90% | 46.435 | — |
| (c) | 1 | 1,681,900,700 | 50.81% | 1,719,179,674 | 0.1645 | -0.19% | 671,202 | 12,529,020 | 3,207,971 | 261,390 | 8.15% | 46.007 | -0.92% |
| (d) | 16 | 5,952,766 – 11,702,632 | 99.66% – 99.83% | 2,037,775,860 – 2,043,415,626 | 0.0892 – 0.0939 | -45.87% – -43.00% | 364,035 – 383,356 | 77,939 – 153,261 | 11,354 – 22,321 | 480 – 5,408 | 2.39% – 33.88% | 44.872 – 45.263 | -3.36% – -2.52% |

(d) pooled: N = 16, HBF B/req 5,952,766 – 11,702,632, worst-case (b)→(d) reduction 99.66%, best 99.83%, b ÷ d ×292.2 – ×574.4; (b)→(c) 50.81%
  j = 0.005 x gap: 10,543,432 – 10,543,432, spread 0.00%; values [10543432, 10543432, 10543432, 10543432, 10543432]
  j = 0.05 x gap: 9,677,308 – 11,702,632, spread 19.21%; values [10543432, 10241442, 9677308, 11702632, 10543432]
  j = 0.45 x gap: 5,952,766 – 8,368,685, spread 32.70%; values [7238320, 5952766, 7714374, 7664042, 8368685]
  unjittered: 10,543,432

**Comparison — 142,213 blocks, global, the 86.8 GiB/GPU point** (unjittered, single runs)

| config | N | HBF B/req | (b)→ reduction | PCM B/req | hit rate | hit rate vs (b) | reuse-hit blk | HBF bytes / reuse-hit blk | HBF-written blk | read back from HBF | read-back share | batch / active sched | batch gap vs (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 3,370,675,339 | — | 1,722,617,954 | 0.1643 | — | 670,585 | 25,132,350 | 6,429,053 | 261,870 | 4.07% | 46.336 | — |
| (c) | 1 | 1,648,185,311 | 51.10% | 1,722,745,881 | 0.1645 | +0.10% | 671,266 | 12,276,693 | 3,143,664 | 259,519 | 8.26% | 46.154 | -0.39% |
| (d) | 1 | 41,901,621 | 98.76% | 1,987,125,969 | 0.0974 | -40.71% | 397,596 | 526,937 | 79,921 | 18,304 | 22.90% | 44.756 | -3.41% |

### Tool&Agent

**Headline — 71,104 blocks, per_rank 4 × 17,776, 43.4 GiB/GPU** ((d): N = 16)

| config | N | HBF B/req | (b)→ reduction | PCM B/req | hit rate | hit rate vs (b) | reuse-hit blk | HBF bytes / reuse-hit blk | HBF-written blk | read back from HBF | read-back share | batch / active sched | batch gap vs (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 1,729,667,006 | — | 883,472,990 | 0.4036 | — | 1,173,911 | 7,367,113 | 3,299,078 | 98,332 | 2.98% | 45.967 | — |
| (c) | 1 | 847,666,217 | 50.99% | 884,945,191 | 0.4040 | +0.10% | 1,175,105 | 3,606,768 | 1,616,795 | 109,826 | 6.79% | 45.701 | -0.58% |
| (d) | 16 | 4,082,631 – 6,347,555 | 99.63% – 99.76% | 1,022,936,220 – 1,025,434,976 | 0.3663 – 0.3688 | -9.24% – -8.62% | 1,065,439 – 1,072,671 | 19,120 – 29,718 | 7,787 – 12,107 | 1,504 – 1,696 | 12.42% – 19.31% | 44.845 – 45.022 | -2.44% – -2.06% |

(d) pooled: N = 16, HBF B/req 4,082,631 – 6,347,555, worst-case (b)→(d) reduction 99.63%, best 99.76%, b ÷ d ×272.5 – ×423.7; (b)→(c) 50.99%
  j = 0.005 x gap: 4,082,631 – 4,082,631, spread 0.00%; values [4082631, 4082631, 4082631, 4082631, 4082631]
  j = 0.05 x gap: 4,082,631 – 4,082,631, spread 0.00%; values [4082631, 4082631, 4082631, 4082631, 4082631]
  j = 0.45 x gap: 4,854,383 – 6,347,555, spread 28.08%; values [5290590, 4904714, 6347555, 4854383, 5189927]
  unjittered: 4,082,631

**Comparison — 142,213 blocks, global, the 86.8 GiB/GPU point** (unjittered, single runs)

| config | N | HBF B/req | (b)→ reduction | PCM B/req | hit rate | hit rate vs (b) | reuse-hit blk | HBF bytes / reuse-hit blk | HBF-written blk | read back from HBF | read-back share | batch / active sched | batch gap vs (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 1,696,787,333 | — | 885,673,951 | 0.4037 | — | 1,174,042 | 7,226,263 | 3,236,365 | 98,428 | 3.04% | 45.770 | — |
| (c) | 1 | 811,414,847 | 52.18% | 885,975,417 | 0.4040 | +0.08% | 1,174,945 | 3,452,991 | 1,547,651 | 100,898 | 6.52% | 45.286 | -1.06% |
| (d) | 1 | 17,168,859 | 98.99% | 1,001,932,194 | 0.3650 | -9.57% | 1,061,727 | 80,853 | 32,747 | 2,080 | 6.35% | 44.859 | -1.99% |

### The headline, numbers only

| | **(b) → (c)** | **(b) → (d), worst case (N = 16)** | (b) → (d), best | (d) HBF B/req range | b ÷ d (range only) |
|---|---|---|---|---|---|
| **Conversation** | **50.81%** | **99.66%** | 99.83% | 5,952,766 – 11,702,632 | ×292.2 – ×574.4 |
| **Tool&Agent** | **50.99%** | **99.63%** | 99.76% | 4,082,631 – 6,347,555 | ×272.5 – ×423.7 |

**Cost of (d), reported next to the reduction (standing rule):**

| | (d) hit rate vs (b), same pool | HBF bytes per reuse-hit block: (b) / (c) / (d) | HBF read-back share: (b) / (c) / (d) |
|---|---|---|---|
| Conversation | **−43.00% to −45.87%** | 25,419,702 / 12,529,020 / 77,939–153,261 | 3.90% / 8.15% / 2.39–33.88% |
| Tool&Agent | **−8.62% to −9.24%** | 7,367,113 / 3,606,768 / 19,120–29,718 | 2.98% / 6.79% / 12.42–19.31% |

**Batch gaps vs (b):** (c) −0.92% / −0.58%; (d) −2.52% to −3.36% / −2.06% to
−2.44%. PCM B/req, (d) vs (b): +17.9–18.2% (Conv) and +15.8–16.1% (T&A).

**Spread of (d) by j:**

| j | Conversation | Tool&Agent |
|---|---|---|
| 0.005 × gap | all 5 seeds equal the unjittered value (10,543,432) | all 5 seeds equal the unjittered value (4,082,631) |
| 0.05 × gap | 19.21% | all 5 seeds equal the unjittered value |
| 0.45 × gap | 32.70% | 28.08% |

Under `per_rank` the stock outcome is far more stable to small arrival
perturbations than under the 142,213 global pool (STEP 3, j = 0.005: 3.44% /
33.26%).

### Read-back: the candidate mechanism, tested

Candidate (recorded above): with a larger pool, more blocks get one read in PCM,
pass `if_read`, are written to HBF, and are never read from HBF again.

| trace | pool | N | HBF-written blk | read back | never read back | read-back share |
|---|---|---|---|---|---|---|
| Conversation | 142,213 global (86.8 GiB/GPU) | 1 | 79,921 | 18,304 | 61,617 | 22.90% |
| Conversation | 71,104 per_rank (design), unjittered | 1 | 20,110 | 480 | 19,630 | 2.39% |
| Conversation | 71,104 per_rank (design), pooled | 16 | 11,354 – 22,321 | 480 – 5,408 | 9,114 – 21,393 | 2.39% – 33.88% |
|| Conversation: drop in HBF-written blocks 142k→71k: 57,600 – 68,567; of which never-read-back: 69.8% – 79.8% (unjittered 70.2%) |
| Tool&Agent | 142,213 global (86.8 GiB/GPU) | 1 | 32,747 | 2,080 | 30,667 | 6.35% |
| Tool&Agent | 71,104 per_rank (design), unjittered | 1 | 7,787 | 1,504 | 6,283 | 19.31% |
| Tool&Agent | 71,104 per_rank (design), pooled | 16 | 7,787 – 12,107 | 1,504 – 1,696 | 6,283 – 10,603 | 12.42% – 19.31% |
|| Tool&Agent: drop in HBF-written blocks 142k→71k: 20,640 – 24,960; of which never-read-back: 97.2% – 98.4% (unjittered 97.7%) |

- **Tool&Agent:** **97.2–98.4%** of the HBF writes (d) avoids at the design
  pool were blocks that, at 142,213, were written to HBF and **never read back
  before the run ended**. Read-back blocks barely change: 2,080 → 1,504–1,696.
- **Conversation:** **69.8–79.8%** of the avoided writes were
  never-read-back. The remainder are blocks that were read back from HBF at
  142,213 (18,304 → 480–5,408). That matches the ~2.5% reuse loss and the
  wider hit-rate cost on Conversation.
- **The data are consistent with the candidate**, completely on Tool&Agent and
  mostly on Conversation.
- **Caveat, stated plainly:** the comparison is per_rank @ 71,104 against
  global @ 142,213, so pool size and partitioning change together. That is
  what was asked for: the existing PHASE 9 configs.
- **Status:** the mechanism stays a candidate. One more point is consistent
  with it but does not establish it.

**(b) and (c) read back almost nothing from HBF:** 2.98–4.07% and 6.52–8.26% of
HBF-written blocks, at both pools. Most of the flash write stream in the stock
configurations is never read from flash during the run. This holds within the
W = 5,000 window, which censors late reads.

## Headline set — follow-ups (before STEP 3b)

### Bibliography

The Ehrmann *et al.* entry now uses the fields supplied by the user. They are
rendered as `\bibitem{ehrmann2022pcm}`, because the paper uses a hand-written
`thebibliography`, not BibTeX. The BibTeX entry is kept verbatim as a comment
beside it. Both `\cite`s use `ehrmann2022pcm`. The "title / author TODO" is
closed.

### CAVEAT — the HBF read-back share is a lower bound

`count_hbf_readback` counts only reads that happen **before the W = 5,000
window ends**. A block written to HBF late in the run has little time left to
be read, so the share is a **lower bound, biased against late writes.** The
same applies to "never read back" below: it means "not read back within the
window".

**Observation, pending STEP 4 (full traces), not a finding:**

- In-window, config (b) reads back **3.90% (Conv) / 2.98% (T&A)** of its
  HBF-written blocks at the headline pool.
- At 142,213 the figures are 4.07% / 3.04%.
- (c): 8.15% / 6.79%.

### De-confounded read-back test — capacity only

The headline comparison (per_rank @ 71,104 vs global @ 142,213) changed pool
size and partitioning together. Here the STEP 3a `c48g` controls (global @
71,104, `store_partition = count`) were re-run with `count_hbf_readback` on:
unjittered + j = 0.05 × gap seeds 1–3, both traces, 8 runs
(`scripts/phase14/readback71g.sh`, `rb71g_*`). **Every non-counter field is
identical to the `c48g` controls in all 8.** So global 142,213 vs global
71,104 differs only in capacity.

The 142,213 side has a single run (the unjittered `rb142_d`). Both 71,104 arms
use the same 4-run composition (unjittered + j = 0.05 seeds 1–3).

| trace | pool | N | HBF-written blk | read back | never read back | read-back share | never-read-back share of the drop vs 142,213 |
|---|---|---|---|---|---|---|---|
| Conversation | 142,213 global | 1 | 79,921 | 18,304 | 61,617 | 22.90% | — |
| Conversation | 71,104 global (capacity only) | 4 | 20,872 – 23,322 | 928 – 4,800 | 18,394 – 19,944 | 4.45% – 20.59% | 70.6% – 76.2% |
| Conversation | 71,104 per_rank (capacity + partition), same 4 runs | 4 | 18,458 – 20,110 | 480 – 2,432 | 16,026 – 19,630 | 2.39% – 13.18% | 70.2% – 74.2% |
| Tool&Agent | 142,213 global | 1 | 32,747 | 2,080 | 30,667 | 6.35% | — |
| Tool&Agent | 71,104 global (capacity only) | 4 | 9,259 – 11,339 | 1,696 | 7,563 – 9,643 | 14.96% – 18.32% | 98.2% – 98.4% |
| Tool&Agent | 71,104 per_rank (capacity + partition), same 4 runs | 4 | 7,787 | 1,504 | 6,283 | 19.31% | 97.7% |

**Both comparisons agree.** Of the HBF writes (d) avoids at 71,104 compared
with 142,213, the share that was never read back is:

| | capacity only (global vs global) | capacity + partition (per_rank vs global) |
|---|---|---|
| Tool&Agent | **98.2–98.4%** | 97.7% |
| Conversation | **70.6–76.2%** | 70.2–74.2% |

**Per the rule, the "candidate" label is lifted.** The finding is stated with
its window caveat:

> **Within the W = 5,000 window, most of (d)'s HBF-write reduction at the
> smaller PCM pool consists of blocks that, at the larger pool, pass the
> `if_read` gate and are written to HBF but are not read back from HBF before
> the window ends.** This is ~98% on Tool&Agent and ~70–76% on Conversation.
> The rest of the Conversation reduction is blocks that *were* read back at
> 142,213, which matches its ~2.5% reuse loss.

Because "not read back" is window-bounded (above), the share is an **upper
bound** on "never read back at all". STEP 4 (full traces) will tighten it.

### Future work (no code)

A stricter demotion gate than "read once while in PCM" may cut the remaining
HBF writes that are not read back. One example is requiring two reads, or a
read within a recency window. **Not explored.**

## STEP 3b — ROBUSTNESS CHECK: HBM evictions spill to PCM (NOT the headline)

**The headline stays the headline set above** (design capacity, `per_rank`,
prefill admission). This is a robustness check of the **PCM entry model** only.

### The flag

`hbm_evict_save_policy` gains **`spill_only`**. The values are now `prefill`
(default, stock), `count`, `on_evict` and `spill_only`.

**What `spill_only` does:**

- The prefill/admission save is switched off:
  `kv_transfer/connectors/mooncake_store.py`, `build_connector_meta` skips
  every save plan.
- A block enters the store **only** when HBM evicts it, through the PHASE 12
  hook (`KVCacheManager.evict` ← `BlockAllocator.allocate`, `_on_hbm_evict`,
  `store.put_with_timing`).
- Everything below the store is unchanged. (c)/(d) keep `write_back` +
  `always`/`if_read`. In (b) an evicted block is written through to HBF.

**New counters**, gated on the policy not being `prefill`, prefix
`hbm_evict_`, merged after `asdict()`:

- **`residue_blocks` / `residue_absent_blocks`:** keyed blocks still in the
  writing workers' GPU prefix cache at run end, and of those the ones absent
  from the store. They are counted in `finalize_run_stats`, called from
  `get_mooncake_stats` before aggregation.
- **`first_spill_count` / `_p50_s` / `_p90_s`:** time from a key's first
  registration in the GPU prefix cache to its first capacity eviction.
  - Registration is its admission to HBM. A new
    `KVCacheManager.register_observer` records it, inert at default.
  - Wired through `LLMScheduler` → `attach_kv_cache_manager`, a no-op under
    `prefill`.
  - Percentiles are nearest-rank.
- **Spill key present / absent:** the existing `hbm_evict_present_blocks` /
  `hbm_evict_absent_blocks`.
- **`count_hbf_readback`** on.

**Byte identity:** `sp_id_{b,d}_*` (new code, defaults) against `ced3806` →
no diff, all four. **Tests: 250 pass** (242 + 8 in `tests/test_spill_only.py`).

### Runs

Same configuration as the headline: 71,104 per_rank (4 × 17,776), HBF
5,033,164, cap 48, both traces. (b) and (c) single runs; (d) the 16-run pooled
set (unjittered + seeds 1–5 at j = 0.005 / 0.05 / 0.45 × gap). 36 runs plus 4
identity runs (`scripts/phase14/step3b.sh`, table `step3b_table.py`).

**Units:**

- B/req are per TP shard; batch is per active scheduler.
- Every Δ is against the headline run with the same config, jitter and seed.
- **Residue-adjusted** = measured writes + resident keyed blocks absent from
  the store, each counted once × 2,621,440 B, per request. Those are the blocks
  the stock model wrote at admission and `spill_only` never did.
- `notdone = 0` in all runs.

### Conversation

| config | N | HBF B/req raw | HBF B/req residue-adjusted | headline HBF B/req | Δ adjusted vs headline | PCM B/req raw | PCM adjusted | Δ adjusted vs headline | hit rate | Δ hit vs headline | reuse-hit blk | memory evictions | batch / active sched | batch gap vs headline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 3,225,653,608 | 3,371,880,153 | 3,418,975,371 | -1.38% | 1,631,466,291 | 1,777,692,836 | +2.87% | 0.0860 | -47.81% | 350,983 | 3,040,671 | 46.253 | -0.39% |
| (c) | 1 | 1,595,352,285 | 1,740,413,862 | 1,681,900,700 | +3.48% | 1,632,631,259 | 1,777,692,836 | +3.40% | 0.0881 | -46.43% | 359,593 | 3,042,893 | 46.047 | +0.09% |
| (d) | 16 | 0 | 163,042,034 – 163,126,444 | 5,952,766 – 11,702,632 | +1293.45% – +2639.16% | 1,779,302,400 – 1,785,772,638 | 1,942,358,065 – 1,948,881,256 | -4.89% – -4.37% | 0.0891 – 0.0922 | -3.80% – +0.02% | 363,689 – 376,131 | 3,322,646 – 3,334,987 | 44.790 – 45.220 | -0.29% – +0.77% |

| config | HBM spills (capacity evictions) | spill key already in store / absent | end-of-run HBM residue (keyed) | of which absent from store | residue share of would-be store admissions | admission→first spill p50 / p90 s | HBF read-back share |
|---|---|---|---|---|---|---|---|
| (b) | 3,418,610 | 306,835 / 3,111,775 | 311,292 | 278,905 | 8.23% | 206.3 / 240.1 | 0.00% |
| (c) | 3,410,408 | 296,411 / 3,113,997 | 310,884 | 276,683 | 8.16% | 147.9 / 168.4 | 0.00% |
| (d) | 3,393,750 – 3,406,091 | 0 / 3,393,750 – 3,406,091 | 310,978 – 311,139 | 310,978 – 311,139 | 8.37% – 8.39% | 82.8 – 83.1 / 96.0 – 99.3 | — |


### Tool&Agent

| config | N | HBF B/req raw | HBF B/req residue-adjusted | headline HBF B/req | Δ adjusted vs headline | PCM B/req raw | PCM adjusted | Δ adjusted vs headline | hit rate | Δ hit vs headline | reuse-hit blk | memory evictions | batch / active sched | batch gap vs headline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 1,471,123,816 | 1,622,209,987 | 1,729,667,006 | -6.21% | 754,201,395 | 905,287,565 | +2.47% | 0.3655 | -9.45% | 1,063,007 | 1,367,421 | 45.611 | -0.77% |
| (c) | 1 | 716,370,346 | 868,008,591 | 847,666,217 | +2.40% | 753,649,320 | 905,287,565 | +2.30% | 0.3671 | -9.14% | 1,067,711 | 1,366,368 | 45.250 | -0.99% |
| (d) | 16 | 369,099 | 162,856,960 – 163,140,600 | 4,082,631 – 6,347,555 | +2469.15% – +3895.97% | 801,347,469 – 808,345,141 | 964,071,784 – 970,833,002 | -5.98% – -5.14% | 0.3628 – 0.3674 | -1.17% – +0.10% | 1,055,199 – 1,068,703 | 1,457,345 – 1,470,692 | 44.258 – 44.661 | -1.45% – -0.79% |

| config | HBM spills (capacity evictions) | spill key already in store / absent | end-of-run HBM residue (keyed) | of which absent from store | residue share of would-be store admissions | admission→first spill p50 / p90 s | HBF read-back share |
|---|---|---|---|---|---|---|---|
| (b) | 1,534,559 | 96,034 / 1,438,525 | 310,982 | 288,174 | 16.69% | 208.6 / 237.0 | 0.00% |
| (c) | 1,529,826 | 92,354 / 1,437,472 | 311,011 | 289,227 | 16.75% | 149.0 / 171.0 | 0.00% |
| (d) | 1,528,769 – 1,542,276 | 320 – 480 / 1,528,449 – 1,541,796 | 311,047 – 311,166 | 309,921 – 310,462 | 16.74% – 16.88% | 84.5 – 85.2 / 97.1 – 99.1 | 0.00% |


**Further numbers:**

| | Conversation | Tool&Agent |
|---|---|---|
| (b)→(c), raw / residue-adjusted | 50.54% / 48.38% (headline 50.81%) | 51.30% / 46.49% (headline 50.99%) |
| (b)→(d), raw | 100.00% ((d) wrote **0** bytes to HBF in all 16 runs) | 99.97% ((d) 369,099 B/req) |
| (d) hit rate vs (b), same model | +3.62% to +7.17% | −0.73% to +0.54% |
| HBF bytes per reuse-hit block, (b) / (c) / (d) | 45,951,707 / 22,182,749 / 0 | 6,919,634 / 3,354,702 / 1,727–1,749 |
| end-of-run residue absent from store, as % of headline (b)'s HBF blocks | 4.28% (278,905 / 6,521,178) | 8.73% (288,174 / 3,299,078) |
| batch gap vs headline | −0.39% to +0.77% | −1.45% to −0.77% |

The residue sits at the upper bound the rule anticipated: 310,884–311,292
keyed blocks against 4 × 78,643 = 314,572. HBM is essentially full of keyed
blocks at the end of the run.

### Rules, as corrected before the runs

| rule | Conversation | Tool&Agent |
|---|---|---|
| (i) (b)→(d) reduction within 5 points of the headline range | PASS (100.00% vs 99.66–99.83%) | PASS (99.97% vs 99.63–99.76%) |
| (ii) **residue-adjusted** (b) HBF, (c) HBF and all PCM B/req, each within 5% of headline | **PASS** (largest \|Δ\| 4.89%: (d) PCM) | **FAIL** ((b) HBF −6.21%; (d) PCM −5.14% to −5.98%) |
| (iii) (d) HBF B/req, **raw**, inside the per-trace band | **FAIL** (0 of 16 inside 5.65–12.29 M: every run is **0**) | **FAIL** (0 of 16 inside 3.88–6.67 M: every run is ~0.37 M) |

On (iii): the residue share next to it is 8.37–8.39% (Conv) and 16.74–16.88%
(T&A) of would-be store admissions. It biases (d) HBF **downward**. It cannot
explain a drop to zero, though.

**Verdict: the headline does NOT hold under spill. The result depends on the
PCM entry model.** Both models are reported side by side (headline set and
this section).

### Why — measured, and it is a lookup interaction rather than a write effect

| | store lookups that hit (of 5,000) | memory-tier read blocks | HBF read blocks | store-served ("disk") hit tokens |
|---|---|---|---|---|
| headline (b), Conv / T&A | 4,996 / 4,978 | 0 / 0 | 325,838 / 117,719 | 5,213,408 / 1,883,504 |
| headline (d), Conv / T&A | 4,996 / 4,979 | 0 / 0 | 480 / 3,008 | 7,680 / 48,128 |
| **spill_only (b), Conv / T&A** | **0 / 769** | 0 / 0 | **0 / 64** | **0 / 1,024** |
| **spill_only (d), Conv / T&A** | **0 / 769** | 0 / 0 | **0 / 0** | **0 / 0** |

**Under `spill_only` the store serves essentially no reads.** The connector
queries the store from block 0 of the prompt (`get_num_new_matched_tokens`,
`mooncake_store.py:89`). `MooncakeStore.lookup` stops at the first absent key
(`store.py:104-107`). The prompt head, the shared system prompt, is hot in HBM.
It is never evicted, so it is never spilled, so it is never in the store. Every
lookup therefore misses at position 0, and the spilled tail behind it is
unreachable.

The knock-on effects:

- (b)'s hit rate falls to its GPU-only part (0.0860 vs 0.1648 on Conv). Its
  HBF read-back is 0.
- (d)'s `if_read` gate never passes, because nothing in PCM is ever read. So
  (d) drops every PCM victim and writes ~nothing to HBF.

**What this robustness check therefore tests:** a PCM entry model in which PCM
and HBF serve no reuse at all on Conversation and almost none on Tool&Agent. In
that model (d)'s HBF stream is trivially ~0, and rule (iii) fails *below* the
band. The headline's (d) figure is **not** contradicted by a design where
spilled blocks are reused. That design was not reached, because of lookup
semantics.

**Not built, flagged for decision:** a lookup that starts at the end of the
local GPU hit, i.e. queries `keys[num_computed_blocks:]` and does not require
the HBM-resident head in the store. That would let spilled tails be found.
Whether a real Mooncake/vLLM connector behaves that way, or like this one, is
not settled here.

**Admission → first spill**, p50 / p90:

| config | Conversation | Tool&Agent |
|---|---|---|
| (b) | 206.3 / 240.1 s | 208.6 / 237.0 s |
| (c) | 147.9 / 168.4 s | 149.0 / 171.0 s |
| (d) | 82.8–83.1 / 96.0–99.3 s | 84.5–85.2 / 97.1–99.1 s |

Faster configs run shorter and churn HBM sooner.

**Spill keys already in the store:**

- (b): 306,835 of 3,418,610 (Conv) and 96,034 of 1,534,559 (T&A). These are
  re-spills of a key whose earlier copy is still stored.
- (d): 0 (Conv) and 320–480 (T&A), because `if_read` drops unread victims from
  the store entirely.

### STEP 3b — result as recorded after review

**Spill with prefix-from-zero store lookup: the store serves almost no reuse
(lookup hits 0 / 769 of 5,000), so (d)'s gate never opens. Copy-at-admission
is required for the store to be useful under this lookup.** This is **not** a
test of a working spill hierarchy. The verdict and rules above stand as
measured, under that label.

## store_lookup_start — the vLLM-style lookup, and two run sets

### The flag

**`store_lookup_start`**:

- **`zero`** (default, stock): the store lookup walks the prompt's keys from
  block 0 and stops at the first miss.
- **`after_hbm`**: the lookup begins at the first block **not already hit in
  HBM** (`num_computed_tokens`), using that block's key, and stops at the
  first store miss after it.
  - Code: `mooncake/config.py`; `_lookup_after_hbm` in
    `kv_transfer/connectors/mooncake_store.py`.
  - The aligned full-prompt cap is applied on the absolute span.
  - When the HBM hit already covers every full prompt block, no lookup is
    made and no hit or miss is counted.

**Justification.** vLLM's
`KVConnectorBase_V1.get_num_new_matched_tokens(request, num_computed_tokens)`
asks the external store for tokens **beyond** the locally computed ones. **We
cite the interface only. We make no claim about how LMCache or Mooncake
implement it internally.**

**Checks:**

- Tests: **255 pass** (250 + 5 in `tests/test_store_lookup_start.py`). They
  show that `after_hbm` reaches a stored tail behind an HBM-only head (where
  `zero` returns 0), and returns the same span as `zero` for a fully stored
  prefix, including the full-prompt cap.
- Byte identity at default: `la_id_{b,d}_*` against `ced3806`, no diff in all
  four.

**Runs:** 71,104 per_rank, HBF 5,033,164, cap 48, both traces,
`count_hbf_readback` on (`scripts/phase14/lookup_after_hbm.sh`). B/req per TP
shard; batch per active scheduler. Every Δ is against the headline run with
the same config, jitter and seed.

### Set A — headline sensitivity: copy-at-admission + `after_hbm`

(b) and (c) single runs; (d) unjittered + j = 0.05 × gap seeds 1–3
(`scripts/phase14/setA_table.py`).

**Rule:** the headline holds under the vLLM-style lookup if (b) HBF, (c) HBF
and all PCM B/req are within 5% of the headline **and** every (d) run lies in
the per-trace band (Conv 5.65–12.29 M, T&A 3.88–6.67 M).

### Conversation

| config | N | store lookups hit / lookups | headline | store-served tokens | headline | reuse-hit blk | headline | HBF B/req | Δ vs headline | PCM B/req | Δ vs headline | hit rate | Δ | batch / active sched | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 447 / 4,990 | 4,996 / 5,000 | 5,213,408 | 5,213,408 | 672,505 | 672,505 | 3,418,975,371 | +0.00% | 1,728,127,173 | +0.00% | 0.1648 | +0.00% | 46.435 | +0.00% |
| (c) | 1 | 439 / 4,989 | 4,996 / 5,000 | 5,262,416 | 5,262,416 | 671,202 | 671,202 | 1,681,900,700 | +0.00% | 1,719,179,674 | +0.00% | 0.1645 | +0.00% | 46.007 | +0.00% |
| (d) | 4 | 0 / 4,991 – 4,992 | 4,996 / 5,000 | 0 | 7,680 – 62,976 | 370,588 – 378,268 | 376,988 – 383,356 | 0 | -100.00% | 2,047,655,018 – 2,048,890,241 | +0.27% – +0.31% | 0.0908 – 0.0927 | -3.33% – +0.34% | 44.895 – 45.185 | -0.42% – +0.44% |

(b)->(c) 50.81%; (b)->(d) 100.00–100.00%; (d) hit vs (b) -44.89% – -43.75%; HBF bytes per reuse-hit block (b) 25,419,702 / (c) 12,529,020 / (d) 0; read-back share (b) 4.09% / (c) 8.51% / (d) — (nothing written to HBF)

### Tool&Agent

| config | N | store lookups hit / lookups | headline | store-served tokens | headline | reuse-hit blk | headline | HBF B/req | Δ vs headline | PCM B/req | Δ vs headline | hit rate | Δ | batch / active sched | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 158 / 4,997 | 4,978 / 5,000 | 1,883,504 | 1,883,504 | 1,173,911 | 1,173,911 | 1,729,667,006 | +0.00% | 883,472,990 | +0.00% | 0.4036 | +0.00% | 45.967 | +0.00% |
| (c) | 1 | 158 / 4,993 | 4,979 / 5,000 | 1,934,752 | 1,934,752 | 1,175,105 | 1,175,105 | 847,666,217 | +0.00% | 884,945,191 | +0.00% | 0.4040 | +0.00% | 45.701 | +0.00% |
| (d) | 4 | 0 / 4,994 | 4,979 / 5,000 | 0 | 48,128 | 1,058,847 – 1,059,871 | 1,067,199 – 1,067,647 | 0 | -100.00% | 1,054,897,340 – 1,056,187,613 | +2.89% – +3.06% | 0.3640 – 0.3644 | -0.82% – -0.70% | 44.581 – 44.687 | -0.67% – -0.49% |

(b)->(c) 50.99%; (b)->(d) 100.00–100.00%; (d) hit vs (b) -9.80% – -9.71%; HBF bytes per reuse-hit block (b) 7,367,113 / (c) 3,606,768 / (d) 0; read-back share (b) 3.11% / (c) 7.07% / (d) — (nothing written to HBF)

| rule | Conversation | Tool&Agent |
|---|---|---|
| A1: (b), (c) HBF and all PCM within 5% | PASS (largest \|Δ\| 0.31%) | PASS (largest \|Δ\| 3.06%) |
| A2: every (d) run in band | **FAIL** (0 / 4: every (d) run writes **0** to HBF) | **FAIL** (0 / 4: **0**) |

**Set A fails.** It fails *below* the band: under the vLLM-style lookup, (d)
writes **nothing** to HBF.

- (b) and (c) are **identical** to the headline in every write, hit and reuse
  figure, including store-served tokens (5,213,408 / 1,883,504 for (b)).
  Their lookup *hit counts* fall from 4,996 to 447 (Conv) because the from-zero
  lookups counted HBM-resident heads as store hits while serving no tokens
  from them.
- **(d) store lookups hit 0 times** in every run. (d)'s hit-rate cost vs (b)
  is −43.75% to −44.89% (Conv) and −9.71% to −9.80% (T&A).

## CORRECTION NOTE — what opened (d)'s `if_read` gate in every copy-at-admission result

**Points back to** every config-(d) result in PHASES 4–14 that used the stock
lookup and copy-at-admission: PHASE 9, 10, 13, and the PHASE 14 headline,
pooled bound, sweep and read-back test. None of that text is edited.

**Measured:**

1. **Across all 332 copy-at-admission runs on disk, PCM (the memory tier)
   never served a single store hit:** `mooncake_memory_hit_tokens = 0` in
   every one. Every store-served token came from HBF. At this capacity, under
   copy-at-admission, PCM holds blocks that are still in HBM, and they have
   left PCM before HBM evicts them.
2. **In the headline and PHASE 9 (d) runs, `mooncake_memory_read_blocks = 0`,
   yet 7,787–79,921 blocks per run passed the `if_read` gate and were
   demoted**: `hl_d` 20,110 / 7,787, `s0_d` and `rb142_d` 79,921 / 32,747.
   Their `read_count > 0` came from `MooncakeStore.lookup` incrementing
   `read_count` on **every** hit key (`store.py:125`). Under the from-zero
   lookup that includes HBM-resident head blocks, whose tokens are served
   locally and never fetched from the store.
3. **So the residual (d) HBF writes in every copy-at-admission result are
   artefacts of lookup accounting, not of reuse.** With the vLLM-style lookup,
   which looks up only beyond the HBM hit, they vanish (set A: 0). The
   PHASE 14 read-back mechanism ("blocks pass `if_read` at the larger pool but
   are not read back") stands as measured. It is re-labelled: the "pass" was a
   lookup touch, not a read.
4. **Constraint, same family.** Under the stock lookup, the load plan reads
   `load_info["keys"][:blocks]` (`mooncake_store.py:543`). Those are the first
   `blocks` keys counted from position 0, i.e. the head, not the external span
   that was missing. That is why 64 of the 332 runs show
   `memory_read_blocks > 0` with `memory_hit_tokens = 0`. Under `after_hbm`
   the keys are the external span. Byte and block totals are unaffected; which
   blocks get "read" is not.

**What this means for the headline claim.** Under copy-at-admission, (d) at the
design point is, to within lookup-accounting artefacts, **"drop every PCM
victim, write ~nothing to flash, give up all store reuse"**. The reduction is
real, but its price is the full hit-rate cost (−44% / −10% vs (b)), not a
selective one.

## Set B — STEP 3b rerun: spill_only + `after_hbm`

Same runs and counters as STEP 3b (16-run (d) set). Table:
`step3b_table.py tmp/phase14 lb`.

**Store lookup hits first:**

### Conversation

| config | store lookups that hit / lookups | headline: hit / lookups | HBF read blocks | headline HBF read blocks |
|---|---|---|---|---|
| (b) | 436 / 4,990 | 4,996 / 5,000 | 300,788 | 325,838 |
| (c) | 428 / 4,989 | 4,996 / 5,000 | 298,194 | 328,901 |
| (d) | 52 – 76 / 4,988 – 4,991 | 4,996 / 5,000 | 16,633 – 24,256 | 480 – 8,448 |

| config | N | HBF B/req raw | HBF B/req residue-adjusted | headline HBF B/req | Δ adjusted vs headline | PCM B/req raw | PCM adjusted | Δ adjusted vs headline | hit rate | Δ hit vs headline | reuse-hit blk | memory evictions | batch / active sched | batch gap vs headline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 3,224,847,254 | 3,371,476,976 | 3,418,975,371 | -1.39% | 1,631,063,114 | 1,777,692,836 | +2.87% | 0.1646 | -0.14% | 671,530 | 3,039,902 | 46.327 | -0.23% |
| (c) | 1 | 1,595,182,416 | 1,740,413,862 | 1,681,900,700 | +3.48% | 1,632,461,390 | 1,777,692,836 | +3.40% | 0.1645 | +0.02% | 671,306 | 3,042,569 | 46.138 | +0.29% |
| (d) | 16 | 18,806,735 – 24,434,967 | 178,556,764 – 185,453,249 | 5,952,766 – 11,702,632 | +1465.26% – +2899.56% | 1,754,074,710 – 1,760,026,427 | 1,915,867,890 – 1,919,776,457 | -6.24% – -5.84% | 0.1023 – 0.1041 | +9.30% – +15.10% | 417,410 – 424,657 | 3,274,528 – 3,285,880 | 44.792 – 45.277 | -0.49% – +0.90% |

| config | HBM spills (capacity evictions) | spill key already in store / absent | end-of-run HBM residue (keyed) | of which absent from store | residue share of would-be store admissions | admission→first spill p50 / p90 s | HBF read-back share |
|---|---|---|---|---|---|---|---|
| (b) | 3,418,577 | 307,571 / 3,111,006 | 311,325 | 279,674 | 8.25% | 203.9 / 238.2 | 3.89% |
| (c) | 3,406,413 | 292,740 / 3,113,673 | 310,879 | 277,007 | 8.17% | 145.8 / 166.3 | 8.10% |
| (d) | 3,395,521 – 3,407,981 | 38,537 – 56,369 / 3,345,632 – 3,356,984 | 310,969 – 311,163 | 304,699 – 309,552 | 8.32% – 8.46% | 82.9 – 83.3 / 96.0 – 99.3 | 31.43% – 45.48% |


### Tool&Agent

| config | store lookups that hit / lookups | headline: hit / lookups | HBF read blocks | headline HBF read blocks |
|---|---|---|---|---|
| (b) | 150 / 4,993 | 4,978 / 5,000 | 89,576 | 117,719 |
| (c) | 152 / 4,993 | 4,979 / 5,000 | 92,024 | 120,922 |
| (d) | 23 – 30 / 4,993 | 4,979 / 5,000 | 2,952 – 13,799 | 1,504 – 3,200 |

| config | N | HBF B/req raw | HBF B/req residue-adjusted | headline HBF B/req | Δ adjusted vs headline | PCM B/req raw | PCM adjusted | Δ adjusted vs headline | hit rate | Δ hit vs headline | reuse-hit blk | memory evictions | batch / active sched | batch gap vs headline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 1,471,166,808 | 1,622,231,482 | 1,729,667,006 | -6.21% | 754,222,891 | 905,287,565 | +2.47% | 0.4013 | -0.56% | 1,167,294 | 1,367,462 | 45.775 | -0.42% |
| (c) | 1 | 716,365,103 | 868,008,591 | 847,666,217 | +2.40% | 753,644,077 | 905,287,565 | +2.30% | 0.4028 | -0.30% | 1,171,556 | 1,366,358 | 45.466 | -0.51% |
| (d) | 16 | 5,340,922 – 15,306,588 | 167,089,013 – 177,615,667 | 4,082,631 – 6,347,555 | +2675.01% – +4081.42% | 791,801,758 – 796,957,082 | 953,871,237 – 959,508,382 | -6.98% – -6.33% | 0.3704 – 0.3741 | +0.42% – +1.94% | 1,077,215 – 1,087,967 | 1,439,138 – 1,448,971 | 44.208 – 44.631 | -1.54% – -0.87% |

| config | HBM spills (capacity evictions) | spill key already in store / absent | end-of-run HBM residue (keyed) | of which absent from store | residue share of would-be store admissions | admission→first spill p50 / p90 s | HBF read-back share |
|---|---|---|---|---|---|---|---|
| (b) | 1,534,312 | 95,746 / 1,438,566 | 310,941 | 288,133 | 16.69% | 206.6 / 233.3 | 2.99% |
| (c) | 1,531,000 | 93,538 / 1,437,462 | 310,989 | 289,237 | 16.75% | 146.9 / 165.4 | 6.12% |
| (d) | 1,529,825 – 1,542,553 | 10,531 – 30,667 / 1,510,242 – 1,520,075 | 311,000 – 311,213 | 306,360 – 310,042 | 16.80% – 17.00% | 84.6 – 85.2 / 97.1 – 99.3 | 10.50% – 47.79% |


**Tier of the store-served tokens** (unjittered runs):

| | memory-tier (PCM) hit tokens | HBF hit tokens | PCM read blocks | HBF read blocks | demoted |
|---|---|---|---|---|---|
| Conv (b) | **590,976** | 4,765,520 | 33,993 | 300,788 | 3,039,902 |
| Conv (c) | **424,064** | 4,722,992 | 23,497 | 298,194 | 3,042,569 |
| Conv (d) | **635,568** | 204,544 | 34,665 | 17,842 | 39,851 |
| T&A (b) | **346,576** | 1,395,616 | 19,311 | 89,576 | 1,367,462 |
| T&A (c) | **331,824** | 1,427,360 | 17,925 | 92,024 | 1,366,358 |
| T&A (d) | **298,912** | 41,472 | 14,764 | 6,510 | 18,266 |

**This is a working spill hierarchy.**

- PCM serves real reads: it now holds blocks HBM has just evicted, a victim
  cache.
- (b) and (c) keep the headline hit rate: −0.14% / +0.02% (Conv), −0.56% /
  −0.30% (T&A).
- (d)'s gate is opened by genuine PCM reads.

**Further numbers:**

| | Conversation | Tool&Agent |
|---|---|---|
| (b)→(c), raw / residue-adjusted | 50.53% / 48.38% | 51.31% / 46.49% |
| (b)→(d), raw (16 runs) | **99.24–99.42%** | **98.96–99.64%** |
| (d) HBF B/req, raw | **18,806,735 – 24,434,967** | **5,340,922 – 15,306,588** |
| (d) hit rate vs (b), same model | −36.76% to −37.84% | −6.80% to −7.72% |
| (d) hit rate vs headline (d) | **+9.30% to +15.10%** | +0.42% to +1.94% |
| HBF bytes per reuse-hit block, (b) / (c) / (d) | 24.0 M / 11.9 M / 225–292 k | 6.30 M / 3.06 M / 25–70 k |
| HBF read-back share, (b) / (c) / (d) | 3.89% / 8.10% / 31.4–45.5% | 2.99% / 6.12% / 10.5–47.8% |
| end-of-run residue absent from store | 279,674 (b) | 288,133 (b) |
| batch gap vs headline | −0.49% to +0.90% | −1.54% to −0.42% |

**Rules (as for STEP 3b):**

| rule | Conversation | Tool&Agent |
|---|---|---|
| (i) (b)→(d) within 5 points of the headline range | PASS (99.24–99.42% vs 99.66–99.83%) | PASS (98.96–99.64% vs 99.63–99.76%) |
| (ii) residue-adjusted (b), (c) HBF and all PCM within 5% | **FAIL** ((d) PCM −5.84% to −6.24%; (b) −1.39%, (c) +3.48%) | **FAIL** ((b) HBF −6.21%; (d) PCM −6.33% to −6.98%) |
| (iii) (d) HBF raw in the per-trace band | **FAIL** (0 / 16, all **above**: 18.8–24.4 M vs 5.65–12.29 M) | **FAIL** (1 / 16; most above: 5.34–15.31 M vs 3.88–6.67 M) |

On (iii), the residue share next to it is 8.32–8.46% (Conv) and 16.80–17.00%
(T&A) of would-be store admissions. It biases (d) HBF downward, so (d)'s true
spill-model HBF stream is, if anything, larger than shown.

**Verdict: the headline numbers do not carry over to a working spill
hierarchy; the result depends on the PCM entry model.**

- Under spill + vLLM-style lookup, (d) writes **1.6–4.1× (Conv) and
  0.8–3.8× (T&A)** the headline's HBF bytes. The (b)→(d) reduction is still
  **≥ 98.96%**.
- In exchange, (d) recovers part of the reuse that copy-at-admission gives up
  entirely: +9–15% hit rate vs headline (d) on Conv.
- Rule (ii)'s failure comes mainly from the residue: (b) on T&A, and (d)'s
  PCM stream.

## CORRECTION NOTES (after sets A and B) — append-only; no earlier text edited

**a) RETRACTED as a mechanism: "one read in PCM passes `if_read`".**

- **Points back to:** "NOTE — cost metrics", "HEADLINE SET … Read-back: the
  candidate mechanism, tested", and "De-confounded read-back test — capacity
  only".
- **Why it is retracted.** No PCM-served reads occurred in any
  copy-at-admission run: `memory_hit_tokens = 0` in all 332, and
  `memory_read_blocks = 0` in every headline / PHASE 9 (d) run. The gate was
  opened by **lookup bookkeeping**: `MooncakeStore.lookup` credits every key it
  walks over (`store.py:125`).
- **What still stands.** The read-back numbers (HBF-written blocks, read back
  in-window, never read back, and the shares of the avoided writes) stand as
  **measurements of blocks credited as read by the lookup walk**. They are not
  evidence about blocks read from PCM.

**b) QUALIFIED: every (d) HBF variability analysis characterises the
bookkeeping artefact, not the design.**

- **Points back to:** STEP 2 (timing arms), STEP 2b (request- and burst-mode
  replicates), the STEP 3 prelude j-size check, the STEP 3 concurrency sweep
  and j = 0.005 check, the pooled bound, and the "outcomes cluster on a few
  discrete values" candidate.
- **Why.** Each of those measured how many blocks the from-zero lookup walk
  credited while they sat in PCM. They do not measure a reuse-driven write
  stream. Their numbers stand as measurements; their reading as properties of
  the read-gated design does not.

**c) The claim that holds in every model tested: write-back admission, config
(c).** (c) has no read dependence: `always` demotes every victim. So it is
untouched by the lookup bookkeeping:

| model | (b) → (c) HBF reduction, Conv / T&A |
|---|---|
| headline (copy-at-admission, from-zero lookup, design pool, per_rank) | **50.81% / 50.99%** |
| set A (copy-at-admission, vLLM-style lookup) | 50.81% / 50.99% (identical) |
| set B (spill_only, vLLM-style lookup) | **50.53% / 51.31%** |
| 86.8 GiB/GPU point (142,213, global) | 51.10% / 52.18% |

**Write-back admission halving the flash write stream is the result that holds
in every PCM entry model and lookup tested.**

## Bookkeeping fixes and verification runs (V1, V2)

### The fixes

Both are flag-gated, stock by default, and byte-identical when off.

- **`read_credit`**:
  - `walked` (stock): every key a lookup walks over is credited as read.
  - `served`: a key is credited only when `MooncakeStore.get` actually serves
    its block, from either tier.
  - Code: `mooncake/store.py` `lookup` / `get`.
- **`load_span`**:
  - `from_start` (stock): a load plan reads `load_info["keys"][:blocks]`.
  - `missing`: it reads exactly the span not already hit in HBM, and the load
    is attributed to that span's slowest tier.
  - Code: `kv_transfer/connectors/mooncake_store.py` `_build_load_plan`,
    `_span_tier`.
- **New check counter, `count_if_read_gate`** (instrumentation only): at every
  `if_read` demotion it records whether the block had at least one read served
  from the memory (PCM) tier. Output `if_read_gate_passes` and
  `if_read_gate_passes_pcm_served`.

**Checks:**

- **Tests: 263 pass** (255 + 8 in `tests/test_read_credit_load_span.py`).
  - Under `walked`, a lookup walk alone opens the gate, with 0 PCM-served
    reads behind it.
  - Under `served`, a walk does not open the gate and a real read does.
  - `missing` loads the span after the HBM hit; `from_start` loads the head.
- **Byte identity at default:** `v_id_{b,d}_*` against `ced3806` → no diff,
  all four.

**Runs:** 71,104 per_rank, HBF 5,033,164, cap 48, both traces, **both fixes
on**, plus `count_if_read_gate` and `count_hbf_readback`
(`scripts/phase14/verify_fixes.sh`, table `verify_table.py`). B/req per TP
shard; batch per active scheduler.

### V1 — copy-at-admission + from-zero lookup, fixes ON (expected (d) HBF = 0)

| trace | run | (d) HBF B/req | headline (fixes off) | gate passes | passes with a PCM-served read | difference | store-served tokens (PCM / HBF) |
|---|---|---|---|---|---|---|---|
| Conversation | unjittered | 0 | 10,543,432 | 0 | 0 | 0 | 0 / 0 |
| Conversation | _j05_s1 | 0 | 10,543,432 | 0 | 0 | 0 | 0 / 0 |
| Conversation | _j05_s2 | 0 | 10,241,442 | 0 | 0 | 0 | 0 / 0 |
| Conversation | _j05_s3 | 0 | 9,677,308 | 0 | 0 | 0 | 0 / 0 |
| Tool&Agent | unjittered | 0 | 4,082,631 | 0 | 0 | 0 | 0 / 0 |
| Tool&Agent | _j05_s1 | 0 | 4,082,631 | 0 | 0 | 0 | 0 / 0 |
| Tool&Agent | _j05_s2 | 0 | 4,082,631 | 0 | 0 | 0 | 0 / 0 |
| Tool&Agent | _j05_s3 | 0 | 4,082,631 | 0 | 0 | 0 | 0 / 0 |

### V2 — spill_only + vLLM-style lookup, fixes ON


#### Conversation

| config | N | HBF B/req | set B (fixes off) | PCM B/req | hit rate | store-served tokens | from PCM | HBF bytes / reuse-hit blk | gate passes | with PCM-served read | difference | end-of-run residue share | batch / active sched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 3,224,847,254 | 3,224,847,254 | 1,631,063,114 | 0.1646 | 5,356,496 | 11.03% | 24,011,193 | — | — | — | 8.25% | 46.327 |
| (c) | 1 | 1,595,182,416 | 1,595,182,416 | 1,632,461,390 | 0.1645 | 5,147,056 | 8.24% | 11,881,187 | — | — | — | 8.17% | 46.138 |
| (d) | 16 | 14,800,650 – 20,724,056 | 18,806,735 – 24,434,967 | 1,759,446,565 – 1,768,017,625 | 0.1003 – 0.1028 | 635,376 – 856,448 | 78.86% – 92.33% | 179,493 – 247,431 | 28,230 – 39,528 | 28,230 – 39,528 | 0 | 8.33% – 8.44% | 44.807 – 45.257 |

(b)->(c) 50.53%; (b)->(d) worst 99.36%, best 99.54% (N = 16); (d) HBF 14,800,650 – 20,724,056; hit rate (b) 0.1646 / (c) 0.1645 / (d) 0.1003 – 0.1028; (d) vs (b) -39.07% – -37.53%; batch gap (c) vs (b) -0.41%, (d) vs (b) -3.28% – -2.31%; vs set B: (b) HBF +0.00%, (c) HBF +0.00%, (b) hit +0.00%

#### Tool&Agent

| config | N | HBF B/req | set B (fixes off) | PCM B/req | hit rate | store-served tokens | from PCM | HBF bytes / reuse-hit blk | gate passes | with PCM-served read | difference | end-of-run residue share | batch / active sched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) | 1 | 1,471,166,808 | 1,471,166,808 | 754,222,891 | 0.4013 | 1,742,192 | 19.89% | 6,301,612 | — | — | — | 16.69% | 45.775 |
| (c) | 1 | 716,365,103 | 716,365,103 | 753,644,077 | 0.4028 | 1,759,184 | 18.86% | 3,057,323 | — | — | — | 16.75% | 45.466 |
| (d) | 16 | 4,937,744 – 12,909,543 | 5,340,922 – 15,306,588 | 794,164,724 – 799,095,652 | 0.3704 – 0.3741 | 203,824 – 490,160 | 80.16% – 95.20% | 22,805 – 59,514 | 9,418 – 24,623 | 9,418 – 24,623 | 0 | 16.86% – 16.96% | 44.167 – 44.603 |

(b)->(c) 51.31%; (b)->(d) worst 99.12%, best 99.66% (N = 16); (d) HBF 4,937,744 – 12,909,543; hit rate (b) 0.4013 / (c) 0.4028 / (d) 0.3704 – 0.3741; (d) vs (b) -7.72% – -6.80%; batch gap (c) vs (b) -0.68%, (d) vs (b) -3.51% – -2.56%; vs set B: (b) HBF +0.00%, (c) HBF +0.00%, (b) hit +0.00%

### V1 — the artefact diagnosis, confirmed

**Expected (d) HBF = 0; measured 0 in all 8 runs.** There are **0** `if_read`
gate passes and **0** store-served tokens, from PCM or HBF.

With honest read credit, copy-at-admission (d) at the design pool is exactly
"drop every PCM victim". Nothing reaches HBF, nothing is ever served from the
store, and (d)'s hit rate is its GPU-only part. The headline (d) HBF range
(5.95–11.70 M / 4.08–6.35 M) was **entirely** produced by the lookup walk.

### V2 — spill_only + vLLM-style lookup, honest bookkeeping

| | Conversation | Tool&Agent |
|---|---|---|
| **(b) → (c)** | **50.53%** | **51.31%** |
| **(b) → (d), worst case (N = 16)** | **99.36%** (best 99.54%) | **99.12%** (best 99.66%) |
| (d) HBF B/req | 14,800,650 – 20,724,056 | 4,937,744 – 12,909,543 |
| hit rate (b) / (c) / (d) | 0.1646 / 0.1645 / 0.1003–0.1028 | 0.4013 / 0.4028 / 0.3704–0.3741 |
| **(d) hit rate vs (b)** | **−37.53% to −39.07%** | **−6.80% to −7.72%** |
| share of store-served tokens from PCM: (b) / (c) / (d) | 11.03% / 8.24% / 78.86–92.33% | 19.89% / 18.86% / 80.16–95.20% |
| HBF bytes per reuse-hit block: (b) / (c) / (d) | 24.0 M / 11.9 M / 179–247 k | 6.30 M / 3.06 M / 23–60 k |
| PCM B/req: (b) / (c) / (d) | 1.631 G / 1.632 G / 1.759–1.768 G | 0.754 G / 0.754 G / 0.794–0.799 G |
| **gate passes / with a PCM-served read / difference** | 28,230–39,528 / same / **0** | 9,418–24,623 / same / **0** |
| end-of-run residue share (of would-be admissions) | 8.17–8.44% | 16.69–16.96% |
| batch per active scheduler | 44.8–46.3 | 44.2–45.8 |
| batch gap (c) / (d) vs (b) | −0.41% / −2.31% to −3.28% | −0.68% / −2.56% to −3.51% |

**What the honest fixes change in set B:**

- (b) and (c) are unchanged, to the byte. Their lookup and load spans were
  already correct under `after_hbm`, and they make no read-gated decisions.
- (d) HBF falls from 18.8–24.4 M to **14.8–20.7 M** on Conversation, and from
  5.3–15.3 M to **4.9–12.9 M** on Tool&Agent. Credits from blocks merely
  walked over, never served, are removed.
- **Every remaining gate pass is backed by a real PCM-served read (difference
  0 in all 32 (d) runs).**

**What this establishes about the design,** in the only model tested where PCM
is actually read:

- PCM serves most of (d)'s store reuse: 79–95% of its store-served tokens.
- Read-gated demotion keeps flash writes **≥ 99.12%** below the HBF-only
  baseline.
- The cost is a real hit-rate loss against (b): −37.5% to −39.1% (Conv) and
  −6.8% to −7.7% (T&A).
- (c)'s write-back halving is unchanged.

These are measurements; no headline is changed here.

## DECISION RECORD — new headline configuration (modelling grounds)

**Decided on modelling grounds, not on the results, and recorded as such.**

**HEADLINE =** `hbm_evict_save_policy = spill_only` +
`store_lookup_start = after_hbm` + `read_credit = served` +
`load_span = missing`, with PCM `memory_capacity_blocks = 71104`,
`store_partition = per_rank`, `store_partition_ranks = 4`, HBF
`ssd_capacity_blocks = 5033164`, cap 48. These are the V2 runs above.

**Reasons:**

1. It is the only configuration tested in which PCM serves reads: 79–95% of
   (d)'s store-served tokens come from PCM. Under copy-at-admission, PCM served
   none in 332 runs.
2. It matches the vLLM connector contract: the external store is asked for
   tokens beyond the locally computed ones.
3. It is the intended HBM → PCM → HBF hierarchy (`CLAUDE.md`, *Scope*).

**Stated plainly: it LOWERS (d)'s reduction against the old headline** —
worst case 99.66% / 99.63% (copy-at-admission, artefact-driven) → **99.36% /
99.12%**.

**Variants retained:**

- **Copy-at-admission (b) and (c)** stay as a reported variant: (b)→(c) 50.81% /
  50.99%.
- **Copy-at-admission (d) is recorded as equivalent to "no store".** With
  honest bookkeeping (V1) it writes 0 to HBF and is served 0 tokens from the
  store.

## FRAMING

- **The primary result is (c), write-back admission:** **~50.5–51.3% fewer HBF
  writes in every model tested** (50.53 / 51.31% headline; 50.81 / 50.99%
  copy-at-admission; 51.10 / 52.18% at the 86.8 GiB/GPU point), **with no
  hit-rate cost**. (c) keeps 99.9% / 104.3% of the store's reuse benefit
  (below).
- **(d), read-gated demotion, is the aggressive end of a write/reuse
  trade-off.** It removes ≥ 99.12% of HBF writes and gives up most of the
  store's reuse benefit (below).

## NO-STORE CONTROL — how much of the store's reuse benefit each config keeps

Same configuration as the headline (cap 48, 30 QPS, W = 5,000, unjittered, same
model, cluster and hardware), but with the KV connector set to `NoopConnector`:
**no store at all** (`scripts/phase14/run_nostore.py`, `ns_*`). The hit rate is
then the GPU-only prefix hit rate. `notdone = 0`.

Retained share of the store's reuse benefit =
**(hit_x − hit_nostore) / (hit_b − hit_nostore)**, using the headline (V2)
runs:

| trace | no-store hit rate | (b) hit | (c) hit | (d) hit range | retained (b) | retained (c) | retained (d) | no-store batch / active sched | no-store notdone |
|---|---|---|---|---|---|---|---|---|---|
| Conversation | 0.0885 | 0.1646 | 0.1645 | 0.1003 – 0.1028 | 100% | 99.9% | 15.4% – 18.8% | 44.953 | 0 |
| Tool&Agent | 0.3670 | 0.4013 | 0.4028 | 0.3704 – 0.3741 | 100% | 104.3% | 9.8% – 20.6% | 44.184 | 0 |

- **(c) keeps the store's reuse benefit in full:** 99.9% (Conv), 104.3% (T&A;
  (c)'s hit rate is slightly above (b)'s).
- **(d) keeps 15.4–18.8% (Conv) and 9.8–20.6% (T&A)** over its 16-run pool. It
  removes ≥ 99.12% of HBF writes and gives up 80–90% of what the store adds to
  the hit rate.
- In absolute terms the store adds +0.0761 to the hit rate on Conversation
  (0.0885 → 0.1646) but only +0.0343 on Tool&Agent (0.3670 → 0.4013), where
  most reuse is already served from HBM.
- Batch per active scheduler with no store: 44.953 / 44.184, against headline
  (b) 46.327 / 45.775.

## STEP 4 — full traces: plan and wall-time estimate (no STEP 4 run made yet)

**Configuration.** The headline flag set (DECISION RECORD above), cap 48,
30 QPS. Runs: (b), (c), and (d) unjittered + burst j = 0.05 × gap seeds 1–3,
both traces. That is 12 runs. Each run also gets `count_hbf_readback`,
`count_if_read_gate` and the end-of-run residue counters, so the residue share
and the read-back share can be reported without the W = 5,000 cut.

**Full traces**, built with `scripts/phase12/build_trace.py` (unchanged), all
rows:

| file | requests | md5 | first 5,000 rows |
|---|---|---|---|
| `conv_full.jsonl` | 12,031 | `8c528d805634be79ab24c85643127fbb` | byte-identical to `conv_5000.jsonl` (md5 `5a19d814…`) |
| `toolagent_full.jsonl` | 23,608 | `8896cd1cb82d57cb18f6392bb9f98042` | byte-identical to `toolagent_5000.jsonl` |

**The replay pace changes:**

- `--trace_target_qps 30` rescales by the *whole file's* native rate, so the
  full traces replay at a slightly different pace from the W = 5,000 files.
- Replay-time burst gaps (`burst_gaps.py`) are 339.8–340.5 ms (Conv) and
  666.8–668.1 ms (T&A), against 313.7 / 605.7 ms at W = 5,000.
- **j = 0.05 × min gap is therefore 16.989 ms (Conv) and 33.338 ms (T&A).** A
  methodology choice, not a hardware number.
- The first 5,000 requests of a full-trace run are therefore **not** a replay
  of the W = 5,000 run. It is a different arrival schedule, so the two are not
  digit-comparable.

**Measured basis for the estimate.** Three headline runs at W = 5,000 were
rerun under `/usr/bin/time -v` (`mem_*`); all three reproduce their V2 runs
exactly.

| run (W = 5,000) | wall time | peak RSS |
|---|---|---|
| (b) Conversation | 3:29.6 (simulator 196 s) | **4.83 GiB** |
| (b) Tool&Agent | 1:56.7 (107 s) | **2.98 GiB** |
| (d) Tool&Agent | 1:45.3 (96 s) | **2.44 GiB** |

**Estimate.** The scaling is linear in requests (×2.406 Conv, ×4.722 T&A),
with a ×1.5 margin. The margin is there because offered load is fixed at
30 QPS against a system that already runs ~6–15× longer than the arrival span,
so the waiting queue, and with it per-step scheduler cost, grows with the
trace.

| run | wall, linear | wall, ×1.5 | peak RSS, linear |
|---|---|---|---|
| Conv (b) | 8.0 min | 12.0 min | ~11.6 GiB |
| Conv (c) | 7.5 min | 11.2 min | ~11.6 GiB |
| Conv (d) × 4 | 6.9 min each | 10.4 min each | ~9.5 GiB each (scaled from T&A's d/b ratio; not measured) |
| T&A (b) | 8.7 min | 13.1 min | ~14.1 GiB |
| T&A (c) | 8.2 min | 12.3 min | ~14.1 GiB |
| T&A (d) × 4 | 7.8 min each | 11.7 min each | ~11.5 GiB each |
| **all 12, serial** | **91 min** | **137 min** | — |

**Memory, not CPU, is the limit.** The machine has 62 GB, but another user's
job (`./run … lay_weight_site.yaml`) holds ~28 GB, leaving ~25–28 GB free.
That allows **2 runs at a time**, never two T&A (b)/(c) together, for
**~46–68 min** in total. With 3 at a time, if memory frees up: ~30–46 min.

**Proposal.** Run T&A (b), the largest, first and alone as a calibration. Then
schedule the rest by measured peak RSS.

### Quoting notes (after review)

- **(c)'s Tool&Agent store-benefit kept is quoted as ~100%, not 104.3%.**
  (c)'s hit rate is 0.4028 against (b)'s 0.4013. That difference is within
  noise and scheduling feedback; it is not an improvement over (b).
- **Until STEP 4 settles the residue, (c) is quoted as ~48–51%.** Raw: 50.53%
  (Conv) / 51.31% (T&A). Residue-adjusted: 48.38% (Conv). The adjustment is
  needed because end-of-run HBM residue (8.2–8.4% / 16.7–17.0% of would-be
  store admissions) is never spilled, so raw counts miss it. If it would
  eventually be demoted to HBF under (c), the raw reduction overstates.

## STEP 4 — full traces, headline configuration

**Configuration.** The headline flag set (spill_only, `after_hbm`,
`read_credit = served`, `load_span = missing`, PCM 71,104 per_rank 4 × 17,776,
HBF 5,033,164 per_rank 4 × 1,258,291), cap 48, 30 QPS, **full traces**:
`conv_full.jsonl` (12,031 requests) and `toolagent_full.jsonl` (23,608; md5s
in the STEP 4 plan above).

**Runs, 14:**

- (b), (c), (d) unjittered;
- (d) burst jitter j = 0.05 × gap, **16.989 ms (Conv), 33.338 ms (T&A)**,
  seeds 1–3;
- one **no-store** run (`NoopConnector`) per trace.

**Model and settings:** `LLaMa2-70B-GQA` (Nhead 64, KV heads 8, Nlayer 80,
Dmodel 8192), FP16, 163,840 B/token at TP2, block 2,621,440 B, TP2/DP4, 4
writing TP-0 ranks, `H3-B200-KVONLY` (pre-compensated `Capacity`). The workload
is a **stand-in for CAG**. B/req are per TP shard; batch is per active
scheduler.

**Calibration (pre-registered stop: wall or RSS > 1.5 × estimate).** T&A (b)
ran alone first: **8:42.4 wall, 11.13 GiB peak**, against an estimate of
8.7 min and ~14.1 GiB. Within bounds, so the rest went 2 at a time under
`scripts/phase14/step4_schedule.py`: memory-gated, never two T&A (b)/(c)
together.

| run | wall | peak RSS |
|---|---|---|
| T&A (b) / (c) | 8:42 / 8:09 | 11.13 / 11.14 GiB |
| T&A (d) × 4 | 7:31–7:35 | 8.98–9.00 GiB |
| Conv (b) / (c) | 8:03 / 7:25 | 9.63 / 9.62 GiB |
| Conv (d) × 4 | 6:47–6:51 | 7.46–7.48 GiB |
| no-store Conv / T&A | 2:55 / 3:32 | 4.66 / 6.30 GiB |

- `notdone = 0` in all 14 runs.
- Total wall time 54 min (09:51–10:45).

### Numbers — full trace against W = 5,000, by metric

The W = 5,000 column uses the same 4-run (d) composition (`v2_d` +
`v2_d_j05_s{1,2,3}`). **The replay pace and the jitter values differ** (STEP 4
plan), so compare **by metric, not by digits.**

#### Conversation — full trace (12,031 requests) vs W = 5,000 (by metric)

| metric | full trace | W = 5,000 (same 4-run (d) composition) |
|---|---|---|
| (b) → (c), raw | 50.23% | 50.53% |
| (b) → (c), residue-adjusted | 49.29% | 48.38% |
| (b) → (d), worst / best (N = 4) | 99.45% / 99.52% | 99.36% / 99.46% |
| (b) HBF B/req | 3,036,078,553 | 3,224,847,254 |
| (c) HBF B/req | 1,511,160,257 | 1,595,182,416 |
| (d) HBF B/req range | 14,628,947 – 16,642,690 | 17,453,023 – 20,724,056 |
| hit rate (b) / (c) / (d) | 0.1898 / 0.1898 / 0.1006 – 0.1019 | 0.1646 / 0.1645 / 0.1018 – 0.1026 |
| no-store hit rate | 0.0916 | 0.0885 |
| (d) hit rate vs (b) | -47.00% – -46.33% | -38.12% – -37.64% |
| store benefit kept (c) / (d) | 100.1% / 9.2% – 10.5% | 99.9% / 17.5% – 18.5% |
| PCM-served share of store-served tokens (b) / (c) / (d) | 6.92% / 6.28% / 72.59% – 76.45% | 11.03% / 8.24% / 83.88% – 84.34% |
| HBF bytes per reuse-hit block (b) / (c) / (d) | 21,281,639 / 10,589,577 / 193,464 – 217,357 | 24,011,193 / 11,881,187 / 209,267 – 247,431 |
| PCM B/req (b) / (c) / (d) | 1,525,785,718 / 1,526,653,140 / 1,702,763,967 – 1,705,810,947 | 1,631,063,114 / 1,632,461,390 / 1,759,446,565 – 1,760,523,452 |
| end-of-run residue share (b) / (c) / (d) | 3.76% / 3.70% / 3.78% – 3.82% | 8.25% / 8.17% / 8.42% – 8.44% |
| HBF read-back share (b) / (c) / (d) | 4.61% / 9.24% / 23.66% – 25.47% | 3.89% / 8.10% / 20.32% – 20.91% |
| (d) gate passes / with PCM-served read | 67,139 – 76,381 / 67,139 – 76,381 (diff 0) | 33,289 – 39,528 / 33,289 – 39,528 (diff 0) |
| batch / active sched (b) / (c) / (d) / no-store | 47.089 / 47.105 / 46.778 – 46.814 / 46.699 | 46.327 / 46.138 / 44.963 – 45.011 / 44.953 |
| batch gap (c) vs (b) / (d) vs (b) | +0.03% / -0.66% – -0.59% | -0.41% / -2.94% – -2.84% |

residue-share expectation (W = 5,000 share x 5,000 / 12,031): (b) 3.43%, (c) 3.40%

#### Tool&Agent — full trace (23,608 requests) vs W = 5,000 (by metric)

| metric | full trace | W = 5,000 (same 4-run (d) composition) |
|---|---|---|
| (b) → (c), raw | 50.25% | 51.31% |
| (b) → (c), residue-adjusted | 49.26% | 46.49% |
| (b) → (d), worst / best (N = 4) | 99.33% / 99.36% | 99.47% / 99.64% |
| (b) HBF B/req | 1,466,116,938 | 1,471,166,808 |
| (c) HBF B/req | 729,379,925 | 716,365,103 |
| (d) HBF B/req range | 9,319,615 – 9,769,217 | 5,271,716 – 7,816,610 |
| hit rate (b) / (c) / (d) | 0.4510 / 0.4511 / 0.3959 – 0.3966 | 0.4013 / 0.4028 / 0.3722 – 0.3741 |
| no-store hit rate | 0.3870 | 0.3670 |
| (d) hit rate vs (b) | -12.21% – -12.05% | -7.26% – -6.80% |
| store benefit kept (c) / (d) | 100.2% / 13.9% – 15.1% | 104.3% / 15.2% – 20.6% |
| PCM-served share of store-served tokens (b) / (c) / (d) | 7.22% / 6.58% / 71.31% – 73.93% | 19.89% / 18.86% / 84.61% – 87.84% |
| HBF bytes per reuse-hit block (b) / (c) / (d) | 6,056,370 / 3,012,341 / 43,835 – 45,883 | 6,301,612 / 3,057,323 / 24,348 – 35,923 |
| PCM B/req (b) / (c) / (d) | 737,006,175 / 737,275,336 / 815,238,081 – 816,065,665 | 754,222,891 / 753,644,077 / 794,164,724 – 796,696,510 |
| end-of-run residue share (b) / (c) / (d) | 3.91% / 3.87% / 3.95% – 3.97% | 16.69% / 16.75% / 16.90% – 16.92% |
| HBF read-back share (b) / (c) / (d) | 4.18% / 8.56% / 22.32% – 25.69% | 2.99% / 6.12% / 9.87% – 14.32% |
| (d) gate passes / with PCM-served read | 83,930 – 87,979 / 83,930 – 87,979 (diff 0) | 10,055 – 14,909 / 10,055 – 14,909 (diff 0) |
| batch / active sched (b) / (c) / (d) / no-store | 45.626 / 45.443 / 44.685 – 44.776 / 44.591 | 45.775 / 45.466 / 44.167 – 44.189 / 44.184 |
| batch gap (c) vs (b) / (d) vs (b) | -0.40% / -2.06% – -1.86% | -0.68% / -3.51% – -3.47% |

residue-share expectation (W = 5,000 share x 5,000 / 23,608): (b) 3.53%, (c) 3.55%

### NEW REGIME on full traces: HBF fills in (b) and (c)

At W = 5,000 `ssd_eviction_count = 0` everywhere. On the full traces the HBF
partitions (1,258,291 blocks per writing rank, 3 TiB per writing GPU) **fill**:

| | HBF blocks written | written ÷ pool (5,033,164) | **HBF evictions** | evicted ÷ written |
|---|---|---|---|---|
| (b) Conv | 13,933,968 | 2.77 | **1,969,372** | 14.1% |
| (c) Conv | 6,935,413 | 1.38 | **1,902,249** | 27.4% |
| (d) Conv | 67,139–72,771 | 0.01 | **0** | 0% |
| (b) T&A | 13,203,464 | 2.62 | **1,604,120** | 12.1% |
| (c) T&A | 6,568,604 | 1.31 | **1,535,440** | 23.4% |
| (d) T&A | 86,701–87,979 | 0.02 | **0** | 0% |

What this means for reading the numbers:

- **(b) and (c) now run with a capacity-bound flash tier.** Blocks evicted
  from HBF can no longer be served. And a key evicted from HBF leaves the
  store (`store.py:555` `_drop_offload_victims`), so its next HBM spill re-admits
  it to PCM and, under (b)/(c), writes it to HBF again.
- Every (b)/(c) HBF byte count on the full traces therefore includes
  **re-writes forced by HBF capacity**.
- (d) is untouched.
- This is recorded, not corrected. Capacity is the sourced design value.
- `CLAUDE.md`'s "HBF never evicts" applies to W = 5,000 only.

### Pre-registered expectation — residue share

**Expected:** it falls roughly in proportion to request count, to ~3.5%:
W = 5,000 share × 5,000 / N gives 3.40–3.43% (Conv) and 3.53–3.55% (T&A).

**Measured:** **3.70–3.82% (Conv), 3.87–3.97% (T&A).**

**It does fall roughly in proportion:** 8.2–8.4% → 3.7–3.8%, and 16.7–16.9% →
3.9–4.0%. The measured shares sit about 9–12% above the proportional
expectation. That fits a residue that is fixed in blocks (HBM is full of keyed
blocks at the end: ~311 k, 4 × 78,643 = 314,572) against a store-admission
count that grows slightly less than linearly.

**(c), raw and residue-adjusted, side by side. The gap shrinks as expected:**

| | W = 5,000 raw / adjusted / gap | **full trace raw / adjusted / gap** |
|---|---|---|
| Conversation | 50.53% / 48.38% / 2.15 pt | **50.23% / 49.29% / 0.94 pt** |
| Tool&Agent | 51.31% / 46.49% / 4.82 pt | **50.25% / 49.26% / 0.99 pt** |

**(c) on full traces is ~49.3–50.3%** on both traces.

### Read-back share without the window cut

These shares are still cut at the end of the trace, so they remain lower
bounds; they no longer have the W = 5,000 cut.

| | W = 5,000 | **full trace** |
|---|---|---|
| (b) | 3.89% (Conv) / 2.99% (T&A) | **4.61% / 4.18%** |
| (c) | 8.10% / 6.12% | **9.24% / 8.56%** |
| (d) | 20.32–20.91% / 9.87–14.32% | **23.66–25.47% / 22.32–25.69%** |

All rise when the window is removed. This is consistent with W = 5,000 cutting
off late reads. Even uncut, (b) reads back < 5% of what it writes to HBF.

### What moved between W = 5,000 and the full trace (flagged, not smoothed)

**Movements:**

- **The store's reuse benefit is larger on full traces**, because reuse beyond
  request 5,000 is no longer censored.
  - (b) hit rate 0.1646 → **0.1898** (Conv) and 0.4013 → **0.4510** (T&A).
  - No-store hit rate: 0.0885 → 0.0916 and 0.3670 → 0.3870.
- **(d) keeps less of that benefit:** **9.2–10.5% (Conv), 13.9–15.1% (T&A)**,
  against 17.5–18.5% / 15.2–20.6% at W = 5,000.
- **(d)'s hit-rate cost vs (b) is larger:** **−46.3% to −47.0%** (Conv) and
  **−12.1% to −12.2%** (T&A), against −37.6% to −38.1% and −6.8% to −7.3%.
- **(c) keeps ~100% (100.1% / 100.2%).**
- **(d) HBF B/req moves in opposite directions on the two traces.**
  - Conv: 17.5–20.7 M → **14.6–16.6 M**, down.
  - T&A: 5.3–7.8 M → **9.3–9.8 M**, up.
- The worst-case (b)→(d) reduction stays **≥ 99.33%**: 99.45% Conv, 99.33% T&A.
- The PCM-served share of store-served tokens falls for every config:
  - (d): 84% → 73–76% (Conv) and 85–88% → 71–74% (T&A).
  - (b): 11.0% → 6.9% and 19.9% → 7.2%.
  - More reuse now reaches back to blocks that have already left PCM.
- **Batch gaps shrink.** (d) vs (b): −2.84% to −2.94% → −0.59% to −0.66% (Conv),
  and −3.47% to −3.51% → −1.86% to −2.06% (T&A).
- **The (d) gate check stays clean:** 67–76 k and 84–88 k passes, every one
  with a PCM-served read (difference 0).

## Code reading — is a block served from HBF re-admitted to PCM on its way back to HBM?

**No. In no configuration does the HBF → HBM read path write PCM.**

| step | code | effect |
|---|---|---|
| load | `MooncakeStoreConnector.start_load_kv` → `store.get` (`kv_transfer/connectors/mooncake_store.py:407-415`) | — |
| HBF serve | `MooncakeStore.get` (`mooncake/store.py:161-186`): an HBF-tier object is read with `self.offload.read(obj.key)` (`:179-182`) | `obj.tier` stays `"ssd"`. No memory-tier admission, no `_record_memory_write`, no PCM bytes |
| LRU touch | `_touch` (`store.py:392-396`) | moves the PCM LRU only for memory-tier objects (`:395`); `OffloadTier.read` refreshes the HBF LRU (`mooncake/ssd.py:58-59`) |
| block later evicted from HBM (spill_only) | `_on_hbm_evict` → `missing_indices` (`mooncake_store.py:243`) → `put_with_timing` (`:259`) → `_put_one`: if the key is still in the store, `_touch` and return zero bytes (`store.py:234-237`) | no PCM write while the key remains stored |
| copy-at-admission variant | prefill save prefiltered by `missing_indices` (`mooncake_store.py:601`) | present keys never rewritten |

**The read path in the paper should therefore be: HBF → HBM directly, PCM
bypassed.** PCM is written only by keys **absent** from the store, which
happens in three ways:

1. First-time spills.
2. **(d) only:** re-spills of keys `if_read` dropped (`store.py:354-363`, popped
   at `:363` when no HBF copy exists).
3. **Full traces, (b)/(c):** re-spills of keys **evicted from HBF**
   (`_drop_offload_victims`, `store.py:555`). Not possible at W = 5,000, where HBF never
   evicts.

**How much of (d)'s PCM writes are re-admissions?** This is an estimate from
existing counters, not a dedicated counter.

- **At W = 5,000** (V2; no HBF evictions anywhere, so (c) never loses a key and
  its absent-spill count is the first-time-spill count): (d) absent spills −
  (c) absent spills = **242,205–258,553 blocks (Conv) = 7.2–7.7% of (d)'s PCM
  writes**, and **77,287–86,692 (T&A) = 5.1–5.7%**. Caveat: (c) and (d) run
  slightly different schedules.
- **On full traces the same subtraction is not valid.** (c) loses 1.90 M /
  1.54 M keys to HBF evictions and re-admits them, so its absent count is no
  longer first-time-only. The raw difference, 0.81 M / 0.70 M = 10.3–10.5% /
  9.6–9.7%, is **not** an estimate of (d)'s re-admissions.
- **A dedicated counter would settle it:** an admission of a key the store has
  admitted before. Not built.

## CORRECTION NOTE — where "HBF never evicts" is written (refers to STEP 4, "NEW REGIME")

The STEP 4 entry above says "`CLAUDE.md`'s 'HBF never evicts' applies to
W = 5,000 only". **CLAUDE.md never said that.** The claim is in this file:
PHASE 14 STEP 1 "1g — addition to 1a: the shared HBF LRU never evicts", and
the STEP 3b note ("HBF never evicts in any run so far"). Both hold at
W = 5,000 only. On the full traces (b) and (c) evict from HBF (1.5–2.0 M
evictions) and (d) does not. CLAUDE.md now states this (2026-10-12).

## AFTER STEP 4 — decisions (2026-10-12)

1. **Headline window = full traces** (STEP 4). W = 5,000 is kept as the earlier
   reference. Quote:
   - (c) as **~49–50%**. Raw and residue-adjusted values have converged:
     50.23 / 49.29 (Conv), 50.25 / 49.26 (T&A).
   - (d) worst case **99.45 / 99.33%**, with hit rate **−47% / −12%** vs (b)
     and store benefit kept **9–15%**.
2. **(b) is described exactly, not as "HBF only".** It is a **71,104-block
   write-through DRAM buffer in front of HBF**, and on the full traces that
   buffer serves **6.9% / 7.2%** of (b)'s store-served tokens.
3. **Read path:** PCM or HBF serve directly into HBM. An HBF hit is not
   re-admitted to PCM (`store.py:179-182`, `:234-237`).
4. Remaining order:
   - the HBF capacity check below (STOP);
   - the single LaTeX rewrite (STOP before commit);
   - STEP 5 (content-aligned tail);
   - STEP 6 (shared far-tier bandwidth: ASSUMED 30 / 48 GB/s per stack =
     27.94 / 44.70 GiB/s).
   STEP 5 and STEP 6 are sensitivity sections.

## HBF CAPACITY CHECK — is (c)'s halving driven by HBF capacity?

### Flags (both stock by default)

| flag | values (stock first) | effect |
|---|---|---|
| `hbf_capacity_mode` | `finite` \| `unlimited` | `unlimited`: the offload tier never evicts and never refuses a write (`mooncake/ssd.py` `OffloadTier.write`). **A SENSITIVITY setting, not a hardware value.** HBF dies per stack is OPEN; capacity per stack (384 GB) is sourced. The capacity knob is still read, and the peak the tier actually held is reported: `hbf_capacity_configured_blocks`, `hbf_capacity_peak_used_blocks` (pool), `hbf_capacity_rank_peak_used_blocks_max` |
| `count_pcm_admit_source` | `false` \| `true` | Instrumentation only. Every memory-tier (PCM) admission is tagged by why its key was absent from the store: `pcm_admit_first_blocks` (never stored), `pcm_admit_after_if_read_drop_blocks` (dropped by `if_read`, `store.py` eviction path), `pcm_admit_after_hbf_evict_blocks` (only copy evicted from HBF, `_drop_offload_victims`), `pcm_admit_after_other_blocks` (expected 0). The four sum to `mooncake_memory_write_blocks` |

Both are merged into `result_dict` after `asdict()` through
`OPTIONAL_MOONCAKE_KEY_PREFIXES` (`pcm_admit_`, `hbf_capacity_`), so stock output
gains no keys. `hbf_capacity_mode` is added to the offload profile only when it
is not `finite`.

**Tests: 277 pass.** That is 263 pre-existing and unchanged, plus 14 new:

- `tests/test_hbf_capacity_mode.py` (6):
  - default and reject;
  - finite evicts while unlimited does not, with the same bytes written;
  - identical behaviour when capacity never binds;
  - per-rank peaks;
  - export prefix.
- `tests/test_pcm_admit_source.py` (8):
  - default off;
  - first admissions;
  - a refresh is not an admission;
  - re-admission after an `if_read` drop;
  - re-admission after an HBF eviction;
  - an HBF-served block is not re-admitted;
  - the four counters sum to the memory writes, and the counter changes
    nothing;
  - export prefix.

### Runs (`scripts/phase14/hbfcap_schedule.py`, log `tmp/phase14/hbfcap.log`)

All runs use the full traces, the headline configuration and cap 48, at
30 QPS. `count_hbf_readback`, `count_if_read_gate` and `count_pcm_admit_source`
are on.

| tags | what | purpose |
|---|---|---|
| `hc_u_{b,c}_{conv,ta}` | `hbf_capacity_mode = unlimited` | **the check** (4 runs) |
| `hc_f_{b,c,d}_{conv,ta}` | finite (headline) | The admission counter on the headline runs. They must reproduce `s4_{b,c,d}_*` exactly on every other key |
| `hc_id_{b,d}_{conv,ta}` | new code, every flag at default, W = 5,000 | byte identity against `ced3806` (`s0_*`) |

(d) is not run with unlimited HBF. On the full traces (d) evicts nothing from
HBF, so the capacity knob never binds and the mode cannot change it. That is
the property tested by `test_identical_when_capacity_never_binds`.

### Pre-registered rule (set before any result)

If (b)→(c) under unlimited HBF is **within 5 points** of the finite-HBF value,
then **"(c)'s halving is not driven by HBF capacity."** Otherwise both values
are reported, together with the dependence on HBF capacity (dies per stack is
OPEN).

### Analytic expectation, recorded before the results (an identity, not a fit)

The STEP 4 counters satisfy two identities exactly, on all four full-trace
(b)/(c) runs:

- **(b):** HBF write blocks = PCM admissions + PCM demotions.
  - Conv: 13,933,968 = 7,002,536 + 6,931,432.
  - T&A: 13,203,464 = 6,637,284 + 6,566,180.
- **(c):** HBF write blocks = PCM demotions.
  - Conv: 6,935,413.
  - T&A: 6,568,604.

Why:

- (b) writes every admission through to HBF and writes the block again when
  the DRAM buffer demotes it (`store.py` `_put_one` write-through branch and
  `_evict_memory_lru`). `OffloadTier.write` counts the second write in full,
  even when the key is still on HBF.
- (c) writes HBF only at demotion.
- Demotions = admissions − end-of-run buffer occupancy (≤ 71,104).

So **(b)→(c) = 1 − dem_c / (adm_b + dem_b) ≈ 50% − (a term ≤ 71,104 /
(adm + dem))**, provided (b) and (c) admit about the same number of blocks.
Raising HBF capacity changes the admission count, since the re-admissions that
follow HBF evictions disappear. It does not change the structure.
**Expected: the rule passes by a wide margin.**

This also means **(b)→(c) is a write-policy effect, write-through vs
write-back**, not a property of PCM media. Under these counters, a write-back
DRAM buffer of the same size would write the same HBF bytes, up to the small
schedule feedback from media timing: `memory_media` changes timing only
(`store.py` `_memory_media_latency`). This
is recorded here as an observation; the framing decision is the user's.

### Results

Code: the commit containing this entry, on branch `phase14-fixes`, parent
`6590989`. Trace md5s:

| trace | md5 |
|---|---|
| conv_full | `8c528d805634be79ab24c85643127fbb` |
| toolagent_full | `8896cd1cb82d57cb18f6392bb9f98042` |
| conv_5000 | `5a19d814562e650fe6e44e4234a4793c` |
| toolagent_5000 | `c6cd3b0c8022eebb88358719a2596f1c` |

Hardware json md5: `9b4cba823dd4158d9bd3efe16bf74491`. Model LLaMa2-70B-GQA
(Nhead 64, KV heads 8, Nlayer 80, Dmodel 8192), FP16, 163,840 B/token at TP2,
block 2,621,440 B, TP2/DP4. Workload: Mooncake FAST'25 traces, a stand-in for
CAG.

**Gates.**

- **Byte identity at default:** `hc_id_{b,d}_{conv,ta}` against `ced3806`
  (`s0_*`), using the PHASE 12 strip() → **no diff in all 4**.
- **The counter changes nothing:** `hc_f_{b,c,d}_{conv,ta}` against STEP 4's
  `s4_*` → **no diff in all 6** on every key except the four added
  `pcm_admit_*` keys.
- All 14 runs finished with rc 0 and `notdone = 0`. The 10 full-trace runs
  took 6:50–9:05 each, with peak RSS 8.5–12.4 GiB (unlimited (b) T&A was
  12.36 GiB). Total time
  46 min, at most 2 runs at a time.

**Batch:** the arms are matched. Per active scheduler, unlimited vs finite
differs by ≤ 0.07%. Within each mode, (c) vs (b) differs by +0.03% / −0.40%,
as in STEP 4.

#### Conversation — finite vs unlimited HBF (full trace, headline configuration)

| metric | finite (b) | finite (c) | unlimited (b) | unlimited (c) |
|---|---|---|---|---|
| HBF B/req | 3,036,078,553 | 1,511,160,257 | 3,035,241,854 | 1,510,832,550 |
| HBF write blocks | 13,933,968 | 6,935,413 | 13,930,128 | 6,933,909 |
| PCM admissions (blocks) | 7,002,536 | 7,006,517 | 7,000,616 | 7,005,013 |
| PCM demotions (blocks) | 6,931,432 | 6,935,413 | 6,929,512 | 6,933,909 |
| PCM B/req | 1,525,785,718 | 1,526,653,140 | 1,525,367,368 | 1,526,325,433 |
| hit rate | 0.1898 | 0.1898 | 0.1900 | 0.1900 |
| store-served tokens | 15,588,992 | 15,317,840 | 15,619,712 | 15,341,904 |
| PCM-served share of store-served tokens | 6.92% | 6.28% | 6.90% | 6.27% |
| HBF evictions | 1,969,372 | 1,902,249 | 0 | 0 |
| batch / active sched | 47.089 | 47.105 | 47.103 | 47.118 |
| HBF peak held, pool / largest rank (unlimited only) | — | — | 7,000,616 (1.39× pool) / 1,775,224 (1.41× rank) | 6,933,909 (1.38× pool) / 1,752,372 (1.39× rank) |

(b)->(c) raw: finite 50.23%, unlimited 50.22% (diff -0.00 pt); residue-adjusted: finite 49.29%, unlimited 49.29% (diff -0.00 pt). Rule (|diff| <= 5 pt): PASS
unlimited vs finite: (b) HBF B/req -0.03%, (c) -0.02%; hit (b) +0.10%, (c) +0.08%; identities (b) HBF = adm + dem, (c) HBF = dem: hold in all 4

##### Conversation — PCM admissions by source (count_pcm_admit_source)

| run | PCM admissions | first | after if_read drop | after HBF eviction | other | sum check |
|---|---|---|---|---|---|---|
| finite (b) | 7,002,536 | 7,000,616 (99.97%) | 0 (0.00%) | 1,920 (0.03%) | 0 (0.00%) | ok |
| finite (c) | 7,006,517 | 7,005,013 (99.98%) | 0 (0.00%) | 1,504 (0.02%) | 0 (0.00%) | ok |
| finite (d) | 7,828,755 | 7,009,652 (89.54%) | 819,103 (10.46%) | 0 (0.00%) | 0 (0.00%) | ok |
| unlimited (b) | 7,000,616 | 7,000,616 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | ok |
| unlimited (c) | 7,005,013 | 7,005,013 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | ok |

#### Tool&Agent — finite vs unlimited HBF (full trace, headline configuration)

| metric | finite (b) | finite (c) | unlimited (b) | unlimited (c) |
|---|---|---|---|---|
| HBF B/req | 1,466,116,938 | 729,379,925 | 1,464,795,114 | 728,719,013 |
| HBF write blocks | 13,203,464 | 6,568,604 | 13,191,560 | 6,562,652 |
| PCM admissions (blocks) | 6,637,284 | 6,639,708 | 6,631,332 | 6,633,756 |
| PCM demotions (blocks) | 6,566,180 | 6,568,604 | 6,560,228 | 6,562,652 |
| PCM B/req | 737,006,175 | 737,275,336 | 736,345,263 | 736,614,424 |
| hit rate | 0.4510 | 0.4511 | 0.4515 | 0.4516 |
| store-served tokens | 13,708,000 | 13,822,448 | 13,815,520 | 13,929,968 |
| PCM-served share of store-served tokens | 7.22% | 6.58% | 7.16% | 6.52% |
| HBF evictions | 1,604,120 | 1,535,440 | 0 | 0 |
| batch / active sched | 45.626 | 45.443 | 45.595 | 45.416 |
| HBF peak held, pool / largest rank (unlimited only) | — | — | 6,631,332 (1.32× pool) / 1,729,870 (1.37× rank) | 6,562,652 (1.30× pool) / 1,713,083 (1.36× rank) |

(b)->(c) raw: finite 50.25%, unlimited 50.25% (diff +0.00 pt); residue-adjusted: finite 49.26%, unlimited 49.27% (diff +0.00 pt). Rule (|diff| <= 5 pt): PASS
unlimited vs finite: (b) HBF B/req -0.09%, (c) -0.09%; hit (b) +0.11%, (c) +0.11%; identities (b) HBF = adm + dem, (c) HBF = dem: hold in all 4

##### Tool&Agent — PCM admissions by source (count_pcm_admit_source)

| run | PCM admissions | first | after if_read drop | after HBF eviction | other | sum check |
|---|---|---|---|---|---|---|
| finite (b) | 6,637,284 | 6,631,332 (99.91%) | 0 (0.00%) | 5,952 (0.09%) | 0 (0.00%) | ok |
| finite (c) | 6,639,708 | 6,633,756 (99.91%) | 0 (0.00%) | 5,952 (0.09%) | 0 (0.00%) | ok |
| finite (d) | 7,349,273 | 6,630,899 (90.23%) | 718,374 (9.77%) | 0 (0.00%) | 0 (0.00%) | ok |
| unlimited (b) | 6,631,332 | 6,631,332 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | ok |
| unlimited (c) | 6,633,756 | 6,633,756 (100.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | ok |

### Verdict against the pre-registered rule

| | finite HBF | unlimited HBF | difference |
|---|---|---|---|
| Conv (b)→(c), raw | 50.23% | 50.22% | −0.00 pt |
| Conv (b)→(c), residue-adjusted | 49.29% | 49.29% | −0.00 pt |
| T&A (b)→(c), raw | 50.25% | 50.25% | +0.00 pt |
| T&A (b)→(c), residue-adjusted | 49.26% | 49.27% | +0.00 pt |

**Rule passes, on both traces, raw and residue-adjusted: "(c)'s halving is not
driven by HBF capacity."** As expected from the identity recorded above, which
holds exactly in all 4 unlimited runs as well.

### What else the check shows (numbers first)

1. **HBF evictions barely change the write stream.** Making HBF unlimited:
   - removes all 1.97 M / 1.60 M ((b)) and 1.90 M / 1.54 M ((c)) HBF
     evictions;
   - changes HBF B/req by only **−0.03% / −0.09%** ((b)) and **−0.02% /
     −0.09%** ((c));
   - raises hit rate by **+0.08% to +0.11%**.

   The admission counter shows why. Re-admissions after an HBF eviction are
   only **1,920 / 5,952** blocks in (b) and **1,504 / 5,952** in (c), i.e.
   **0.02–0.09%** of PCM admissions. Almost every key HBF evicts is never
   spilled again.
2. **How much HBF the full traces would need.** With no capacity limit, HBF
   holds up to:
   - **7,000,616 blocks** ((b) Conv) = **1.39× the 5,033,164-block pool**, and
     1.41× a rank's 1,258,291 at the largest rank;
   - **6,631,332** ((b) T&A) = **1.32×**, and 1.37× at the largest rank.

   This is the distinct-key footprint of the trace. In both unlimited runs it
   equals the first-admission count exactly. It is an occupancy, not a
   capacity recommendation; HBF dies per stack stays OPEN.
3. **PCM admissions by source, headline (finite) runs:**

   | | first | after an `if_read` drop | after an HBF eviction | other |
   |---|---|---|---|---|
   | (b) | 99.97% / 99.91% | 0 | 0.03% / 0.09% | 0 |
   | (c) | 99.98% / 99.91% | 0 | 0.02% / 0.09% | 0 |
   | (d) (unjittered run, N = 1) | **89.54% / 90.23%** | **10.46% / 9.77%** | 0 | 0 |

   - In (d), 819,103 / 718,374 PCM admissions are re-spills of keys `if_read`
     dropped. That is **10.7% / 10.0%** of (d)'s 7,690,512 / 7,191,468 dropped
     blocks.
   - The "other" category is 0 in every run, so the three named sources
     account for every PCM write.
   - Jittered (d) runs did not carry the counter, so (d)'s shares are N = 1.

## CORRECTION NOTES — refer to STEP 4 ("NEW REGIME" and "Code reading"); no earlier text edited

- **"Every (b)/(c) HBF byte count on the full traces therefore includes
  re-writes forced by HBF capacity"** (STEP 4, NEW REGIME) overstates the
  effect.
  - The re-writes exist, but they are **0.02–0.09%** of PCM admissions and
    move HBF B/req by **≤ 0.09%** (measured above).
  - HBF capacity changes **what HBF holds**: 1.5–2.0 M cold keys are evicted
    and lost, which costs ~0.1% hit rate. It does **not** change **how much
    is written** to any material extent.
- **"On full traces the same subtraction is not valid"** (Code reading). In
  practice (c)'s re-admissions after HBF eviction are only 1,504 / 5,952
  blocks.
  - So the raw difference quoted there, 10.3–10.5% / 9.6–9.7% of (d)'s PCM
    writes, was close to the direct measurement.
  - The direct measurement is **10.46% / 9.77%** (unjittered (d)).
  - The dedicated counter is now the number to quote.

## HBF capacity — per rank or pool? (settled 2026-10-17, before any endurance figure)

**5,033,164 is the POOL across the 4 writing ranks. Under `per_rank` the code
splits it 4 ways, exactly as it splits PCM.**

| | value | source |
|---|---|---|
| configured | `ssd_capacity_blocks = 5,033,164` | headline run config: `scripts/phase14/step4_schedule.py:23`, `scripts/phase14/hbfcap_schedule.py` (`HL`) |
| read | `capacity_blocks = int(config.ssd_capacity_blocks)` | `TokenSim/mooncake/ssd.py:45` |
| **split (`per_rank`)** | `pool = OffloadTier(config, block_bytes).capacity_blocks`; **`partition_capacity_blocks = pool // partitions`** = 5,033,164 // 4 = **1,258,291**, remainder 0. Each rank gets its own `OffloadTier` and LRU with that capacity | `ssd.py:157-158`; built with `partitions = store_partition_ranks = 4` at `store.py:98`. Counterpart of PCM's `memory_capacity_blocks // ranks` (`store.py:446`) |
| no split (`global`, a variant) | one shared `OffloadTier` of 5,033,164 | `store.py` `else: self.offload = OffloadTier(config, ...)` |
| **recorded in results** | **`partition_rank_ssd_capacity_blocks = 1,258,291`** in every `per_rank` run, next to `partition_rank_memory_capacity_blocks = 17,776` and `partition_ranks_seen = 4`. The offload profile records the pool knob (`ssd_capacity_blocks 5,033,164`, `store_partition per_rank`, `store_partition_ranks 4`). Unlimited-HBF runs record `hbf_capacity_configured_blocks = 5,033,164` (the pool) and the per-rank peak separately | `store.py:104-105`; e.g. `s4_d_ta` |

Capacity in bytes, per writing rank (one TP-0 shard on one GPU), at
2,621,440 B per block:

- 1,258,291 × 2,621,440 = **3,298,534,359,040 B = 3 TiB − 524,288 B**. The
  0.2-block shortfall comes from flooring 1,258,291.2.
- Pool: 5,033,164 × 2,621,440 = **13,194,137,436,160 B = 11.999998 TiB**.

**The 24 TiB figure.** Neither number is wrong; they measure different
things, and no die count is involved.

- **24 TiB = 8 GPUs × 3 TiB.** That is the physical HBF of the whole 8-GPU
  cluster, including the 4 TP-1 GPUs. Their write stream mirrors TP-0 and is
  not simulated (STEP 3a). This is the meaning of CLAUDE.md "the far tier is
  24 TiB".
- **12 TiB = 4 writing GPUs × 3 TiB.** That is the HBF that receives the
  counted per-shard bytes, and it is the endurance budget for them (CLAUDE.md
  "Endurance budget").
- **3 TiB per writing GPU** = 1,258,291 blocks. This is the per-rank partition
  the code actually enforces.

**Where the 4× comes from.** The requests-to-wear-out intermediate must
match the basis of B/req. B/req is pooled over every request served by all 4
ranks, so it pairs with the 12 TiB pool. Pairing it with 3 TiB undercounts
requests 4× (the "budget" factor). E_PCM below depends only on the capacity
ratio. It is the same with per-rank capacities (1,258,291 : 17,776) or pool
capacities (5,033,164 : 71,104), provided both sides use the same basis. Mixing
the bases, e.g. 12 TiB HBF against 43.4 GiB PCM, gives a 4× error.

## Break-even PCM endurance — per TP shard, full traces, headline configuration

```
E_PCM = (HBF_cap x 3000 / HBF_B_per_req) x PCM_B_per_req / PCM_cap
HBF_cap = 1,258,291 x 2,621,440 = 3,298,534,359,040 B   (per writing rank)
PCM_cap =    17,776 x 2,621,440 =    46,598,717,440 B   (per writing rank)
HBF endurance 3000 P/E (TLC, CLAUDE.md NAND row)
```

E_PCM is the PCM endurance, in cycles, at which PCM and HBF exhaust their
write budgets at the same time. **It is a threshold, not a lifetime and not a
verdict.** No literature PCM endurance is compared here. Script:
`scripts/phase14/breakeven_pcm.py`. Inputs: STEP 4 runs `s4_*`. The (d)
counter reruns `hc_f_d*` reproduce them on every key except the added
counters.

**Assumptions (all optimistic; relaxing any of the first three raises the (d)
threshold):**

1. Perfect wear levelling in both tiers.
2. HBF write amplification = 1.
3. Every write is a full-block write at both tiers.
4. Each writing rank's PCM:HBF write ratio equals the pooled ratio. Per-rank
   write counts are not recorded; this is not measured.

Notes:

- Published chip-level PCM endurance already absorbs some cell variability, so
  any comparison must not apply a further levelling penalty on top of it. That
  would double-count.
- HBF also holds the model weights (60.38 GiB per rank, the pre-compensation in
  `Capacity`), while the simulator gives KV the whole 3 TiB. Under assumption 1
  static data rotates too, so the write budget is the full capacity. Without
  static wear levelling the HBF budget would be ~2% smaller.

### Results

#### Conversation

| config | run | HBF B/req | PCM B/req | requests to HBF exhaustion (pool) | **E_PCM (cycles)** | E_PCM, residue spilled, never demoted | E_PCM, residue spilled, demoted at the config's own rate | demotion rate |
|---|---|---|---|---|---|---|---|---|
| (b) | unjittered | 3,036,078,553 | 1,525,785,718 | 13,037,348 | **106,720.7** | 110,889.5 | 106,700.3 | 1.0000 |
| (c) | unjittered | 1,511,160,257 | 1,526,653,140 | 26,193,392 | **214,535.0** | 222,775.9 | 214,453.7 | 1.0000 |
| (d) | unjittered | 14,628,947 | 1,705,810,947 | 2,705,759,518 | **24,762,023.8** | 25,734,060.9 | 24,753,454.0 | 0.0087 |
| (d) | _j05_s1 | 15,856,106 | 1,704,020,759 | 2,496,351,407 | **22,821,629.3** | 23,719,799.9 | 22,813,703.1 | 0.0094 |
| (d) | _j05_s2 | 16,642,690 | 1,702,763,967 | 2,378,366,194 | **21,726,972.7** | 22,588,831.8 | 21,719,363.6 | 0.0099 |
| (d) | _j05_s3 | 14,981,929 | 1,705,248,790 | 2,642,010,330 | **24,170,648.7** | 25,120,015.7 | 24,162,276.1 | 0.0089 |

(d) E_PCM range N = 4: 21,726,973 – 24,762,024; (d)/(c) threshold ratio 101.27 – 115.42; sanity check (E_d/E_c == HBF_c/HBF_d x PCM_d/PCM_c): holds in all 4 to 1e-12

#### Tool&Agent

| config | run | HBF B/req | PCM B/req | requests to HBF exhaustion (pool) | **E_PCM (cycles)** | E_PCM, residue spilled, never demoted | E_PCM, residue spilled, demoted at the config's own rate | demotion rate |
|---|---|---|---|---|---|---|---|---|
| (b) | unjittered | 1,466,116,938 | 737,006,175 | 26,998,128 | **106,750.7** | 111,092.3 | 106,728.3 | 1.0000 |
| (c) | unjittered | 729,379,925 | 737,275,336 | 54,268,579 | **214,656.6** | 223,305.2 | 214,566.6 | 1.0000 |
| (d) | unjittered | 9,627,307 | 816,065,665 | 4,111,472,845 | **18,000,666.1** | 18,741,274.6 | 17,993,719.4 | 0.0119 |
| (d) | _j05_s1 | 9,769,217 | 815,437,621 | 4,051,748,794 | **17,725,532.9** | 18,458,178.2 | 17,718,656.7 | 0.0121 |
| (d) | _j05_s2 | 9,413,444 | 815,238,081 | 4,204,881,240 | **18,390,953.1** | 19,152,174.9 | 18,383,807.4 | 0.0117 |
| (d) | _j05_s3 | 9,319,615 | 815,592,078 | 4,247,215,622 | **18,584,177.9** | 19,351,234.2 | 18,576,979.7 | 0.0115 |

(d) E_PCM range N = 4: 17,725,533 – 18,584,178; (d)/(c) threshold ratio 82.58 – 86.58; sanity check (E_d/E_c == HBF_c/HBF_d x PCM_d/PCM_c): holds in all 4 to 1e-12

| | Conv | T&A |
|---|---|---|
| **(b)** (DRAM buffer, read as PCM) | **106,721** | **106,751** |
| **(c)** | **214,535** | **214,657** |
| **(d), range over N = 4** | **21.73 M – 24.76 M** | **17.73 M – 18.58 M** |
| (d) ÷ (c) | 101.3 – 115.4 | 82.6 – 86.6 |

(b)'s figure treats its write-through DRAM buffer as if it were PCM. It is the
threshold PCM would need in (b)'s position. (b) has no PCM.

**Sanity check: passes in all 8 (d) runs, to 1e-12.** The check is that
E_d / E_c = (HBF_c / HBF_d) × (PCM_d / PCM_c). The capacities cancel, so the
check is an algebraic identity: it verifies the arithmetic, not the inputs.
For example, Conv unjittered: (1,511,160,257 / 14,628,947) ×
(1,705,810,947 / 1,526,653,140) = 103.30 × 1.1174 = 115.42, which equals
24,762,024 / 214,535.

**End-of-run HBM residue.** On the full traces the residue is 3.70–3.97% of
would-be admissions. The 8.2–17% figure is the W = 5,000 residue. It does
**not** simply nudge both thresholds up; the direction depends on where the
residue would go:

- **If it were spilled and then demoted at each config's own measured rate**
  (1 under `always`, 0.009–0.012 under `if_read`), every threshold moves by
  **−0.02% to −0.04%**, i.e. essentially unchanged. The extra HBF writes scale
  with the extra PCM writes.
- **If it were spilled and never demoted** (upper bound), every threshold rises
  by **+3.8% to +4.1%**.
