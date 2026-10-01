# Running report — PCM write-absorbing tier for HBM + HBF KV cache

**Working document.** Updated as work happens. The LaTeX paper
(`../Latex/pcm-kv-micro.tex`) is synced from this **once a week** — see
[Pending LaTeX sync](#pending-latex-sync) for what has not made it across yet.

| | |
|---|---|
| Last updated | **2026-10-01** |
| Simulator | `TokenSim/` — see `TokenSim/docs/findings.md` (append-only, the full record) |
| Latest phase | **PHASE 13** |
| Tests | **190** passing |
| Headline status | Write-absorption result stands; scope limited to prompt blocks |

---

## How to use this file

- **Append to the weekly log**, newest first. One entry per working session is
  fine; they get merged at sync time.
- **Keep `findings.md` as the source of truth for numbers.** This file
  summarises; it does not replace. If the two disagree, `findings.md` wins.
- **Move things from [Pending LaTeX sync](#pending-latex-sync) into the paper
  once a week**, then clear the entry here.
- Anything marked **OPEN** is an unsourced parameter. Do not guess one to make a
  run go through; record what was used and that it is unsourced.

---

## Status at a glance

### What is established

| claim | numbers | where |
|---|---|---|
| Write-back admission halves flash writes | **51.24%** / **52.55%** reduction (conv / T&A) | PHASE 9 |
| Read-gated demotion removes almost all of the rest | **98.75%** / **98.28%** | PHASE 9 |
| …as work-normalized lifetime | **×90.31** / **×86.39** | PHASE 8 |
| Wear is invariant to concurrency | **0.50%** / **1.10%** spread over a 12× batch range | PHASE 9 STEP 4 |
| Design point is on the flat foot, not a knee | 43.4 GiB/GPU = **5.3%** / **10.2%** of footprint; knee at **19×** | PHASE 10 |
| PCM capacity buys writes + latency, never hit rate | hit rate flat over a **150×** capacity sweep | PHASE 10 |
| HBM eviction does **not** drive far-tier writes | **+0.00%** / **−0.07%** (5% rule, pre-registered) | PHASE 12 |

### What is not established

- **The store holds prompt blocks only.** Generated/transient KV has no key
  space and never enters it. Defensible claim is the narrower one: *PCM as a
  write-back buffer in front of HBF reduces **prefix-cache offload** writes by
  51% (`always`) or 99% (`if_read`).*
- **Ceiling on closing that gap, on these traces:** generated tokens are
  **2.58%** / **1.94%** of all tokens (**4.0%** / **4.1%** against the
  deduplicated write population). Prompt-heavy traces; a long-generation
  workload would move this.
- **Workload is a stand-in.** Target is CAG; Mooncake FAST'25 traces substitute.
- **FP16 is hardcoded** in the harness. FP8 would halve KV bytes/token and
  improve every write-volume number — so FP16 understates the benefit.
- **PCM's own endurance is unpriced.** Largest open parameter.
- **No day counts anywhere** — the far tier has no shared-bandwidth model.

### Open parameters

| parameter | status |
|---|---|
| PCM endurance (P/E cycles) | **OPEN** — to be sourced |
| HBF dies per stack | **OPEN** — not needed for capacity (384 GB/stack is sourced) |
| CAG corpus size, requests sharing it, transient KV per request | **OPEN** |

---

## Weekly log

> Newest first. Keep entries short: what was done, what moved, what is next.

### 2026-10-01 — PHASE 13 and paper draft

- **PHASE 13** appended to `findings.md`: a correction, a limitation, a trace
  option, and sensitivity runs. No simulator source changed.
- **Correction to PHASE 12.** "The previous output is a minority of what a
  continuation adds" was wrong for Conversation and the arithmetic was inverted:
  median `delta/out` = 1.486 makes the output **67.3%** of added tokens, a
  *majority*. Tool&Agent is **15.7%**, a minority. Appended, not edited in place.
- **Tail-divergence limitation recorded.** The traces hash at 512 tokens while
  the simulator blocks at 16, so a continuation cannot reuse its predecessor's
  prompt tail in **97.9% / 98.7%** of pairs — median 22 / 14 blocks, **1.35% /
  1.95%** of distinct blocks. All reported hit rates are therefore **floors**.
  Within-workload comparisons are unaffected.
- **`build_trace.py --share_partial_tail`** added, default off and byte-identical
  when off. **It does not work as a fix:** on config (b) it *lowers* the hit rate
  by 26.9% / 6.6% and *raises* HBF bytes/request by 7.4% / 6.9%, because the rule
  picks each continuation's predecessor independently and siblings that
  previously shared a chain diverge earlier. 70.8% of Conversation's chosen
  predecessors share only one 512-token chunk.
- **Two numbers shown to be far more workload-sensitive than reported:** the
  `if_read` hit-rate cost (−40.7% → −27.7%, −9.6% → −3.3%) and the HBF lifetime
  *ratio* (80.4× → 230.3×, 98.8× → 213.0×). **Bytes per request is the robust
  figure; the ratio is not.**
- All four original-trace runs reproduce PHASE 9 STEP 3 exactly, **config (d)
  included** — not previously re-measured.
- **MICRO paper draft** written to `../Latex/pcm-kv-micro.tex`. Not compiled —
  no LaTeX toolchain on this machine.
- Tests: 177 → **190**.

### 2026-09-28 — PHASE 12, the falsifier

- Pre-registered falsification test of the mechanism: if HBM eviction drove
  far-tier writes, HBF bytes/request would track HBM occupancy. Wired HBM
  evictions into the store behind `hbm_evict_save_policy`. Result **+0.00% /
  −0.07%** across an 11× batch range — **hypothesis dead**.
- Verdict covers **prompt-block eviction only**; no decode block can reach the
  hook (`hbm_evict_unkeyed_blocks = 0` in every run).
- Mechanism measured: eviction *volume* barely moves with batch (+1.66% /
  +1.42%), and **97.4–98.0%** of evicted keys are already in the store, so their
  put is a zero-byte LRU refresh.
- Fixed a silent export defect — `LLMResult` is a pydantic dataclass and was
  dropping undeclared stats keys. Audited: **no pre-PHASE-12 counter was ever
  affected**.
- Workload and harness rebuilt from scratch (nothing survived on the machine);
  reproduces **twelve** independently recorded PHASE 9 values exactly, first try.
- Tests: 158 → 177.

---

## Current numbers

> Config (b) = HBF only, write-through + always. (c) = +PCM, write-back +
> always. (d) = +PCM, write-back + if\_read. All at `H3-B200-KVONLY`,
> LLaMa2-70B-GQA TP2/DP4, FP16, block 2,621,440 B, PCM pool 142,213 blk
> (347.2 GiB), HBF pool 10,066,329 blk (24 TiB), W = 5,000, 30 QPS.

### HBM-saturated (cap 256 / 96), PHASE 9 STEP 3

| | (b) | (c) | (d) |
|---|---|---|---|
| **Conversation** HBF write blocks | 6,461,439 | 3,150,432 | **80,968** |
| HBF bytes/request | 3,387,654,930 | 1,651,733,692 | **42,450,551** |
| prefix hit rate | 0.1631 | 0.1641 | 0.0942 |
| achieved batch/sched | 43.61 | 42.90 | 41.55 |
| **reduction vs (b)** | — | **51.24%** | **98.75%** |
| **Tool&Agent** HBF write blocks | 3,234,691 | 1,534,945 | **55,703** |
| HBF bytes/request | 1,695,909,675 | 804,753,244 | **29,204,414** |
| prefix hit rate | 0.4014 | 0.4033 | 0.3733 |
| achieved batch/sched | 48.44 | 47.32 | 44.72 |
| **reduction vs (b)** | — | **52.55%** | **98.28%** |

### Matched batch (cap 48), PHASE 9 STEP 3 — the reproduction gate

| | HBF write bytes | HBF blocks | hit rate | reuse hit blk | batch/sched |
|---|---|---|---|---|---|
| (b) Conversation | 16,853,376,696,320 | 6,429,053 | 0.1643 | 670,585 | 23.167889 |
| (d) Conversation | 209,508,106,240 | 79,921 | 0.0974 | 397,596 | 22.378200 |
| (b) Tool&Agent | 8,483,936,665,600 | 3,236,365 | 0.4037 | 1,174,042 | 22.885048 |
| (d) Tool&Agent | 85,844,295,680 | 32,747 | 0.3650 | 1,061,727 | 22.429517 |

**Any new campaign must reproduce this table to every digit before its own
numbers are trusted.**

### Workload footprint

| | prompt-block instances | distinct blocks | bytes | dedup |
|---|---|---|---|---|
| Conversation | 4,080,885 | 2,695,220 | 6,580 GiB | 1.51× |
| Tool&Agent | 2,908,548 | 1,397,479 | 3,412 GiB | 2.08× |

---

## Next steps

| | item | status |
|---|---|---|
| 1 | **Option 2 — store generated KV.** Design note at `TokenSim/docs/option2-output-kv-scope.md`; three decisions open at its §7. | scoped, **not built** |
| 2 | **Source PCM endurance.** Blocks any claim about the PCM tier's own wear. | OPEN |
| 3 | **MQSim WAF/GC run.** Would replace the WAF = 1 assumption. MQSim is not currently in the tree. | not started |
| 4 | **CAG workload definition.** Corpus size, sharing factor, transient KV per request. | OPEN |
| 5 | **Narrower tail-sharing rule** — multi-chunk stems only, applied consistently across siblings — to test the PHASE 13 limitation properly. | proposed |
| 6 | Figures for the paper: capacity sweep and concurrency sweep as `pgfplots`. | not started |

---

## Pending LaTeX sync

> What is in this report / `findings.md` but **not yet** in
> `../Latex/pcm-kv-micro.tex`. Clear entries as they are moved across.

- [ ] Bibliography is incomplete — five entries marked `% TODO` (author lists,
      volumes, page ranges, DOIs). **Must be filled before submission.**
- [ ] Paper has **not been compiled** — no LaTeX toolchain on this machine.
      `sudo apt install texlive-publishers texlive-latex-extra texlive-science`
- [ ] Confirm the MICRO template for the target year. The draft targets
      `IEEEtran[conference]`; the switch to `acmart/sigconf` is a commented
      one-liner in the header.
- [ ] Author block is a placeholder guess.
- [ ] No figures yet (next steps #6).
- [ ] PHASE 13's tail-sharing negative result is summarised in the Limitations
      section but not given its own subsection — decide whether it earns one.
