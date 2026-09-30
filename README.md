# A Hybrid HBM + HBF + PCM Memory Hierarchy for LLM KV-Cache Offloading

Masters thesis (MTP). Extends the **H3** HBM+HBF architecture with a
bit-alterable NVM (PCM) tier and tests one argument:

> **PCM absorbs the transient write stream that NAND flash handles badly.**

HBF (High-Bandwidth Flash) gives a GPU ~16× the capacity of HBM at comparable
bandwidth, but NAND has limited write endurance. H3 sidesteps that by targeting
**read-only** data — model weights and shared pre-computed KV. This work asks
what happens on the same hardware under **general write-heavy serving**, where
the flash tier does receive a transient write stream and endurance is at risk,
and whether a small PCM tier between HBM and HBF absorbs enough of it to matter.

The claim is scoped to **endurance and lifetime**, not serving speed.

---

## The architecture

Per GPU, B200-class. The HBM / HBF / topology figures are from **H3 (SK hynix,
IEEE Computer Architecture Letters, Jan–Jun 2026)**; **PCM is this work's
addition and is not in the paper.**

| tier | capacity | bandwidth | role here |
|---|---|---|---|
| **HBM3e** | 192 GB (8 cubes × 24 GB) | 8 TB/s | memory tier, **entirely KV cache** |
| **PCM** | **43.4 GiB** (8 stacks × 2 dies × 35 mm² @ 0.62 Gb/mm²) | — | **the write absorber** |
| **HBF** | 3 TB (8 stacks × 384 GB) | 8 TB/s | offload tier receiving spill |

Two structural points that a lot of naive models get wrong:

- **No HBM is given up to make room for flash.** HBM and HBF are
  **daisy-chained** — the HBM cubes stay on the GPU shoreline and an address
  decoder and router in the HBM base die splits traffic between the two paths.
  There is no per-stack bandwidth derate.
- **PCM stacks vertically on top of each HBF stack**, inside the existing
  footprint. It does not displace NAND dies laterally and does not reduce HBF
  capacity.

A **40 MB SRAM Latency Hiding Buffer** in the HBM base die prefetches from HBF so
its microsecond read latency is hidden. The mechanism is *predictability*, not
speed: it works because LLM inference has a deterministic, sequential access
pattern. That makes it a good fit for H3's read-only data and a **poor** fit for
demand-driven hits on transient KV — a caveat carried explicitly through every
result here.

`CLAUDE.md` holds the authoritative parameter list, including the values that
are still **OPEN** and must not be guessed: PCM endurance, HBF dies per stack,
and the CAG workload parameters.

---

## What is in this repository

```
TokenSim/                 SimPy LLM-serving simulator with a Mooncake KV store
  TokenSim/mooncake/        memory tier + offload tier, LRU, admission/demote policies
  TokenSim/block/           paged KV allocator and GPU prefix cache
  docs/findings.md          THE DELIVERABLE — every run, append-only
  docs/option2-output-kv-scope.md   design note for storing generated KV (not built)
  scripts/phase12/          workload rebuild + run harness for PHASES 12-13
  tests/                    190 tests
README.md                 this file
CLAUDE.md                 authoritative design parameters and working rules
```

**Not in this repository:**

- **MQSim** (CMU-SAFARI SSD simulator, commit `51f0f2d`) is part of the project
  per `CLAUDE.md` — it is the source of NAND-level WAF and GC results — but it is
  **not currently in this tree**.
- **The workload traces.** The Mooncake FAST'25 traces and the built 5,000-request
  workload files are 21–33 MB each and are not committed. `scripts/phase12/README.md`
  gives the source URLs, the rebuild command and the md5 sums; the rebuild
  reproduces twelve independently recorded simulator values exactly, so it is
  verifiable rather than merely repeatable.

---

## Results so far

`TokenSim/docs/findings.md` is the deliverable and is **append-only** — including
its corrections and its void results. Headlines:

**PCM as a write absorber works, and the size of the effect is known.** At the
buildable 43.4 GiB/GPU, moving the admission policy to write-back cuts HBF write
volume by **~51%** on the conversational trace and **~53%** on Tool&Agent; adding
an `if_read` demote policy takes it to **~98.8%** and **~98.3%**. The benefit is
bought by the *policy*, not by the capacity.

**HBF wear is set by pool capacity against workload footprint — not by batch size
or concurrency.** Over a 12× concurrency range driving peak HBM occupancy from
25% to 99.2%, HBF bytes per request move by **0.50%** and **1.10%**. There is no
"read-only below batch N" regime; the deployment rule of that shape does not
exist in this model.

**43.4 GiB/GPU is the buildable point, not a capacity optimum.** It sits at
5.3% / 10.2% of the workload's prompt footprint, on the flat part of the curve.
HBF only goes fully read-only at ~100% of footprint — about **19×** the buildable
budget. Separately, mean PCM residency crosses the median time-to-first-read at
**1.8–2.8×** the design point. The tier is nearly big enough to *hold* a block
until its reuse; nowhere near big enough to *keep it off HBF*.

**PCM capacity buys write reduction and latency, never hit rate** — flat at
0.163 / 0.402 across a 150× capacity range.

**A falsifier was run against the central mechanism and it held.** If HBM
eviction pressure drove far-tier writes, HBF bytes per request would depend on
HBM occupancy. Wiring HBM evictions directly into the store changes it by
**+0.00%** and **−0.07%** across an 11× batch range — the hypothesis is dead, and
the flat curve above is not an artefact of a missing code path.

---

## What this does *not* establish

Stated up front because it bounds every number above:

- **The store has only ever held prompt blocks.** Generated/transient KV has no
  key space in the simulator, so it never enters the store. The defensible claim
  is narrower than the thesis title: *PCM as a write-back buffer in front of HBF
  reduces **prefix-cache offload** writes by 51% (`always`) or 99% (`if_read`).*
  Closing this gap is scoped in `TokenSim/docs/option2-output-kv-scope.md` and is
  **not built**. Generated KV is only **1.9–2.8%** of tokens on these traces, so
  its ceiling is known to be small *on this workload*.
- **The workload is a stand-in.** The target workload is CAG
  (cache-augmented generation); the runs use the Mooncake FAST'25 conversational
  and Tool&Agent traces instead. The CAG parameters remain OPEN.
- **KV precision is FP16 and hardcoded.** FP8 would halve KV bytes per token and
  improve every write-volume and endurance number, so reporting at FP16
  **understates** the architecture's benefit. It is a harness limitation, kept
  deliberately as the conservative choice.
- **Two figures are far more workload-sensitive than they look.** The `if_read`
  hit-rate cost and the HBF lifetime *ratio* both move by large factors under a
  trace variant that changes only 2–5% of the footprint. **Work-normalized bytes
  per request is the robust figure; the ratio is not.**
- **No absolute lifetimes or day counts are quoted anywhere**, because the far
  tier has no shared-bandwidth model.

---

## Reproducing

Requires Linux x86_64 and Python 3.11 (the bundled roofline extension is
precompiled). See `TokenSim/README.md` for the simulator itself.

```bash
cd TokenSim
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest                  # not in requirements.txt
.venv/bin/python -m pytest tests/ -q          # 190 tests

# hardware entry (derived from the tracked catalogue; never mutates it)
.venv/bin/python data/hardware/make_h3b200.py

# workload: fetch the FAST'25 traces, then
WORK=tmp/phase12
.venv/bin/python scripts/phase12/build_trace.py $WORK/conversation_trace.jsonl \
    $WORK/conv_5000.jsonl 5000

# the reproduction gate — must match findings.md to every digit before
# any new number is trusted
WORK=$WORK scripts/phase12/run_one.py gate_conv \
  '{"memory_media":"dram","admission_write_policy":"write_through","demote_policy":"always"}' \
  conv_5000.jsonl 5000 48 30
```

`scripts/phase12/README.md` has the md5 sums and the expected gate values.

### Working rules

These are enforced, not aspirational — `CLAUDE.md` has the full list:

- `findings.md` is **append-only**. Corrections are appended; earlier phases are
  never edited, and wrong results are flagged in place rather than deleted.
- Tracked catalogue files are never mutated; new hardware entries go in a copy
  passed via `--hardware_models`.
- Report the occupancy and capacity actually *achieved*, not the knob value set.
- Report throughput alongside every latency number, and flag non-monotonic series
  rather than smoothing them.
- Capacity knobs are **cluster pool totals across 8 GPUs**, not per-GPU values.
  Getting this wrong is an 8× error and has already cost this project one full
  round of void results.
- When a simulator constraint forces a deviation, name the file and line and
  report it as a constraint — never work around it silently.

---

## Status

Active. `main` holds the baseline; `falsifier-on-evict` carries PHASES 12–13 and
the option-2 design note. Nothing in the option-2 path is implemented.
