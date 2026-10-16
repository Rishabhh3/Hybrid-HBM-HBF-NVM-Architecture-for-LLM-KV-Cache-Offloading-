# Running report — PCM write-absorbing tier for HBM + HBF KV cache

**Working document.** Updated as work happens. The LaTeX paper
(`../Latex/pcm-kv-micro.tex`) is synced from this **once a week** — see
[Pending LaTeX sync](#pending-latex-sync) for what has not made it across yet.

| | |
|---|---|
| Last updated | **2026-10-12** |
| Simulator | `TokenSim/` — see `TokenSim/docs/findings.md` (append-only, the full record) |
| Latest phase | **PHASE 14** (STEP 4 full traces; HBF capacity check) |
| Tests | **277** passing |
| Headline status | **HEADLINE = spill-only + vLLM-style lookup + honest read bookkeeping, 43.4 GiB/GPU per_rank, cap 48, FULL TRACES** (configuration decided 2026-10-12 on modelling grounds; window decided after STEP 4). **Contribution: (d) read-gated demotion** (drops 98.7–99.0% of (c)'s HBF writes, keeps 9–15% of the store's reuse benefit) **plus the argued case for a non-volatile in-package tier** (not simulator-supported). (c)'s ~49–50% is a write-policy identity that characterises the baseline. Scope: prompt blocks only |

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

### Decision record (2026-10-11) — read before the tables below

- **Headline = design capacity + `per_rank`.** The decision was made on
  hardware grounds, not on the control-run results.
  - **Capacity.** 43.4 GiB/GPU is fixed by die area: 2 PCM dies per stack,
    each the area of a 1 Tbit TLC die. So the pool sized for the 4 writing
    ranks is the design point. 142,213 blocks becomes the **86.8 GiB/GPU
    point**, i.e. 4 PCM dies per stack.
  - **Partition.** PCM is in-package on each GPU's own HBF stacks, so the
    pools are private: `per_rank` is faithful, and `global` is a variant.
- **What it moves (from the controls).** (c) goes slightly **down**,
  51.10 / 52.18% → ~50.8 / 51.0%. (d) goes **up**, 98.6–99.0% → ~99.7–99.8%.
- **Unit check RESOLVED (2026-10-11): binary throughout.** The headline pool is
  **71,104 = `per_rank` 4 × 17,776 = 43.4 GiB/GPU**. HBF stays 5,033,164. The
  35 mm² die is kept, a < 1% rounding of 35.3 mm². 29 Gb/mm² (TLC) is still
  uncited, so its unit stays OPEN.
- **Density decision.** 1st-gen XPoint, 0.62 Gb/mm² (Ehrmann *et al.*,
  *Applied Research* 2022, e202200024, via TechInsights) is the headline
  density.
  - 2nd-gen (4-deck) XPoint would roughly double it, which is approximately the
    86.8 GiB/GPU point. That point is reported as such; there are no new runs.
  - **Do not call 0.62 "conservative" overall:** it lowers (c)'s reduction
    slightly but raises (d)'s.
- **New reporting rule for (d).** Every (d) result also reports its hit rate vs
  (b) at the same pool, and HBF bytes per reuse-hit block for (b), (c) and (d).
  Under `if_read` a smaller PCM writes less to HBF partly by dropping reuse.
  - At the design point that loss is small: Conv reuse-hit blocks −2.5%, T&A
    none detectable.
  - HBF bytes per reuse-hit block (d) at the design point: **126–139 k**
    (Conv) / **19.1 k** (T&A).
  - (b) at the design point: **25.4 M** / **7.4 M**.

**The headline rows below are the design point** (71,104 blocks, per_rank,
cap 48; PHASE 14 headline set). The 86.8 GiB/GPU point (142,213, global) is
given alongside, labelled.

### What is established

**HEADLINE (2026-10-12): spill-only entry (HBM → PCM on eviction) +
vLLM-style lookup (`after_hbm`) + honest read bookkeeping (`read_credit =
served`, `load_span = missing`), on the FULL TRACES** (12,031 Conv / 23,608
T&A requests, PHASE 14 STEP 4). Design capacity 43.4 GiB/GPU: PCM 71,104
blocks as `per_rank` 4 × 17,776; HBF 5,033,164. Cap 48, 30 QPS. FP16; **B/req
are per TP shard**; batch per active scheduler. Workload: Mooncake FAST'25
traces, a stand-in for CAG. (d) is N = 4: the unjittered run plus burst
jitter j = 0.05 × gap, seeds 1–3.

**The configs, exactly.**
- **(b) is not "HBF only".** It is a **71,104-block write-through DRAM buffer
  in front of HBF** (`dram`, `write_through`, `always`). The buffer serves
  **6.9% / 7.2%** of (b)'s store-served tokens.
- (c) = PCM, write-back, `always`.
- (d) = PCM, write-back, `if_read`.

**Framing (decided 2026-10-17).**

- **(c) characterises the baseline; it is not a design contribution.** Its
  halving is a write-policy identity.
- **The contribution is (d)**, a measured selection policy, plus the
  architectural case for a non-volatile in-package staging tier, which is
  argued, not measured.

> **A same-capacity volatile buffer would write identical HBF bytes.** The case
> for PCM rests on durability and in-package density, not on write volume.

#### 1. MEASURED, an identity: the baseline's structural write cost

We quantify a structural cost in the baseline and confirm that it does not
depend on HBF capacity. Write-through costs exactly one extra HBF write per
admitted block. In every (b)/(c) run:

- (b) HBF writes = PCM admissions + demotions;
- (c) HBF writes = demotions.

Removing the write-through copy therefore halves HBF writes, whatever the
medium: `memory_media` changes timing only.

| | numbers (conv / T&A) | where |
|---|---|---|
| (b)→(c), 8 (b)/(c) runs (finite + unlimited HBF) | raw **50.22–50.25%**; residue-adjusted **49.26–49.29%**. Finite vs unlimited HBF: **≤ 0.01 pt** | STEP 4; HBF capacity check |
| HBF B/req (b) → (c) | 3,036,078,553 → 1,511,160,257 / 1,466,116,938 → 729,379,925 | STEP 4 |
| Hit rate (c) vs (b) | 0.1898 vs 0.1898 / 0.4511 vs 0.4510; keeps **~100%** of the store's reuse benefit | STEP 4 |
| HBF evictions (finite HBF) | 1.97 M / 1.60 M ((b)), 1.90 M / 1.54 M ((c)). **Immaterial:** removing them moves HBF bytes by **≤ 0.09%** and hit rate by **≤ +0.11%**. Unlimited HBF peaks at 1.39× / 1.32× the pool (the distinct-key footprint) | HBF capacity check |

#### 2. ARGUED, NOT MEASURED: why the staging tier should be non-volatile and in-package

**TokenSim does not support this part.** It rests on the durability argument
and on capacity-per-area figures, not on byte counts.

- **Durability.**
  - Under write-back, an acknowledged block lives only in the staging tier
    until it is demoted, so a volatile tier loses it on power loss.
  - TokenSim cannot measure durability.
  - **Caveat:** KV blocks can be recomputed. For a cache, losing them costs
    recomputation (time and energy), not correctness. The argument is weaker
    here than for primary storage, and the paper must say so.
- **Why not DRAM.**
  - 43.4 GiB/GPU of in-package buffer is a cost and area argument.
  - PCM's capacity per area is sourced: 0.62 Gb/mm² (Ehrmann *et al.* 2022),
    fitted on top of the HBF stacks without displacing NAND.
  - **A comparable in-package DRAM density is OPEN.** Nothing in the tracked
    files gives one, so no area comparison is made until it is sourced.

#### 3. MEASURED, non-trivial: (d) read-gated demotion — the contribution

(d) is a **selection policy, not an identity**. It demotes only blocks that
were served from PCM. N = 4 per trace: the unjittered run plus burst jitter
j = 0.05 × gap, seeds 1–3.

| claim | numbers (conv / T&A) | where |
|---|---|---|
| (d) drops almost all of what (c) writes | (d) writes **0.97–1.10% / 1.28–1.34%** of (c)'s HBF bytes, i.e. drops **98.7–99.0%**. Against (b): worst-case reduction **99.45% / 99.33%**. (d) HBF B/req **14.6–16.6 M / 9.3–9.8 M** | STEP 4 |
| …and keeps part of the reuse | (d) keeps **53.0–53.7% / 87.8–88.0%** of (b)'s reuse-hit blocks (hit rate 0.1006–0.1019 / 0.3959–0.3966 vs 0.1898 / 0.4510) | STEP 4 |
| **…but most of that reuse needs no store** | **No store at all keeps 48.3% / 85.8%** (hit rate 0.0916 / 0.3870). (d) therefore keeps only **9.2–10.5% / 13.9–15.1%** of the store's reuse benefit. Its store-served tokens are 10.7–12.4% / 13.9–15.7% of (b)'s. Over no store, (d)'s hit rate is +9.8% to +11.2% / +2.3% to +2.5% | STEP 4; no-store control |
| PCM is read: it serves (d)'s reuse | Share of store-served tokens from PCM: (b) 6.9% / 7.2%; (c) 6.3% / 6.6%; **(d) 72.6–76.5% / 71.3–73.9%**. Every (d) `if_read` pass is backed by a PCM-served read (difference 0, 8 runs) | STEP 4 |
| PCM admissions by source | @@ADMIT@@ | HBF capacity check |
| HBF bytes per reuse-hit block | (b) 21.3 M / 6.06 M; (c) 10.6 M / 3.01 M; (d) **193–217 k / 44–46 k** | STEP 4 |
| PCM write traffic | PCM B/req: (b) 1.526 G / 0.737 G; (c) 1.527 G / 0.737 G; (d) 1.703–1.706 G / 0.815–0.816 G. PCM endurance is still OPEN | STEP 4 |
| Lifetime ratio (b ÷ d) | **Only as a range:** ×182–×208 / ×150–×157. Quote the reductions and B/req instead | STEP 4 |

#### Measurement conditions (all configs)

| | numbers (conv / T&A) | where |
|---|---|---|
| End-of-run HBM residue (bias) | **3.70–3.82% / 3.87–3.97%** of would-be store admissions (8.2–8.4% / 16.7–17.0% at W = 5,000). It fell roughly in proportion to request count, as pre-registered. Undercounts absolute B/req in every config | STEP 4 |
| HBF read-back share (end-of-trace cut only) | (b) 4.61% / 4.18%; (c) 9.24% / 8.56%; (d) 23.7–25.5% / 22.3–25.7% | STEP 4 |
| HBF fills in (b)/(c) | (d) never evicts. At W = 5,000 nothing evicts | STEP 4 |
| Batch | Per active scheduler: (b) 47.09 / 45.63; (c) +0.03% / −0.40% vs (b); (d) −0.59% to −0.66% / −1.86% to −2.06% vs (b) | STEP 4 |

**Earlier reference: W = 5,000** (same configuration, first 5,000 requests;
the headline window until 2026-10-12). Compare with the rows above by metric,
not by digit: the replay pace and the jitter values differ. (d) here is
pooled over N = 16.

| claim | numbers (conv / T&A) | where |
|---|---|---|
| **(c) halves flash writes, at no hit-rate cost — in every model tested** | raw **50.53% / 51.31%** (headline), residue-adjusted 48.38% (conv). HBF B/req (b) → (c): 3,224,847,254 → **1,595,182,416** / 1,471,166,808 → **716,365,103**. Hit rate (c) vs (b): 0.1645 vs 0.1646 / 0.4028 vs 0.4013. Keeps **~100%** of the store's reuse benefit (99.9% / 104.3% measured; T&A's > 100% is noise or scheduling feedback, not an improvement). Other models: copy-at-admission 50.81 / 50.99%; 86.8 GiB/GPU point 51.10 / 52.18% | PHASE 14 V2; no-store control |
| (d) read-gated demotion removes almost all flash writes | **Pooled over N = 16 per trace** (stock + burst jitter, j = 0.005 / 0.05 / 0.45 × gap): **worst-case reduction 99.36% / 99.12%** (best 99.54% / 99.66%). (d) HBF B/req **14.8–20.7 M / 4.9–12.9 M** | PHASE 14 V2 |
| …at a hit-rate cost | (d) hit rate **0.1003–0.1028 / 0.3704–0.3741**, against (b) 0.1646 / 0.4013: **−37.5% to −39.1% / −6.8% to −7.7%**. (d) keeps only **15.4–18.8% / 9.8–20.6%** of the store's reuse benefit (no-store hit rate 0.0885 / 0.3670) | PHASE 14 V2; no-store control |
| PCM is read: it serves (d)'s reuse | Share of store-served tokens from PCM: (b) 11.0% / 19.9%; (c) 8.2% / 18.9%; **(d) 78.9–92.3% / 80.2–95.2%**. Every (d) `if_read` pass is backed by a PCM-served read (check: difference 0, 32 runs) | PHASE 14 V2 |
| HBF bytes per reuse-hit block | (b) 24.0 M / 6.30 M; (c) 11.9 M / 3.06 M; (d) **179–247 k / 23–60 k** | PHASE 14 V2 |
| PCM write traffic | PCM B/req: (b) 1.631 G / 0.754 G; (c) 1.632 G / 0.754 G; (d) 1.759–1.768 G / 0.794–0.799 G. PCM endurance is still OPEN | PHASE 14 V2 |
| Lifetime ratio (b ÷ d) | **Only as a range:** ×155.6–×217.9 / ×114.0–×297.9. Quote the reductions and B/req instead | PHASE 14 V2 |
| **End-of-run HBM residue** (bias) | Keyed blocks still in HBM, never spilled: **8.2–8.4% / 16.7–17.0%** of would-be store admissions. **Direction:** undercounts absolute B/req in every config. Understates (d)'s reduction. For (c) it depends on where those blocks would land: if they would eventually be demoted to HBF, raw (b)→(c) overstates by up to ~2 points (48.38% / 46.49% residue-adjusted); if they would stay in PCM within the window, raw understates. STEP 4 (full traces) shrinks the end effect | PHASE 14 STEP 3b, V2 |
| Batch | Per active scheduler: (b) 46.33 / 45.78; (c) −0.41% / −0.68% vs (b); (d) −2.31% to −3.28% / −2.56% to −3.51% vs (b) | PHASE 14 V2 |

**Read path (code):** HBF → HBM directly; an HBF-served block is **not**
re-admitted to PCM in any config (`store.py:161-186`, `:234-237`). PCM is
written only for keys absent from the store: first spills, (d)'s re-spills
after an `if_read` drop, and (on full traces) re-spills after an HBF eviction.
At W = 5,000, re-admissions are ~7.2–7.7% / 5.1–5.7% of (d)'s PCM writes.

**Variants (labelled, not the headline):**

| variant | numbers (conv / T&A) | where |
|---|---|---|
| Copy-at-admission, from-zero lookup (the old headline): (b)/(c) | (b)→(c) 50.81% / 50.99% | PHASE 14 headline set |
| Copy-at-admission (d) | **Equivalent to "no store"**: with honest bookkeeping, 0 HBF writes and 0 store-served tokens (V1). Its old 99.6%+ reduction was a lookup-walk artefact | PHASE 14 V1, set A |
| Spill + from-zero lookup | Degenerate: the store serves 0 / 769 of 5,000 lookups | PHASE 14 STEP 3b |
| 86.8 GiB/GPU point (142,213 blocks, global) | (b)→(c) 51.10% / 52.18% (≈ 2nd-gen XPoint density) | PHASE 9 STEP 3 |
| Wear vs concurrency, (b) and (c) | 0.50% / 1.10% spread over a 12× batch range (copy-at-admission, 86.8 GiB/GPU point) | PHASE 9 STEP 4 |
| PCM capacity buys writes, not hit rate | hit rate flat over a 150× capacity sweep (copy-at-admission, global); knee ~38× the design point | PHASE 10; PHASE 14 STEP 3a audit |
| HBM eviction does not drive far-tier writes under copy-at-admission | +0.00% / −0.07% (5% rule) | PHASE 12 |

### What is not established

- **VERIFIED (V1/V2, honest bookkeeping: `read_credit = served`,
  `load_span = missing`).**
  - **V1, copy-at-admission:** (d) HBF = **0** in all 8 runs, with 0 gate
    passes and 0 store-served tokens. The headline (d) range was entirely the
    lookup-walk artefact.
  - **V2, spill + vLLM-style lookup:** every gate pass has a real PCM-served
    read (difference 0 in all 32 runs). Reductions: (b)→(c) **50.53% /
    51.31%**; (b)→(d) worst case **99.36% / 99.12%**.
  - V2 detail: (d) HBF B/req 14.8–20.7 M / 4.9–12.9 M. (d) hit rate vs (b)
    **−37.5% to −39.1% / −6.8% to −7.7%**. PCM serves 79–95% of (d)'s
    store-served tokens.
- **(c) is the claim that holds in every model tested.** Write-back admission
  halves the flash write stream: **50.81% / 50.99%** in the headline (and
  identically under the vLLM-style lookup), **50.53% / 51.31%** under spill +
  vLLM-style lookup. (c) has no read dependence.
- **RETRACTED as a mechanism:** "one read in PCM passes `if_read`". No
  PCM-served reads occurred under copy-at-admission; the gate was opened by
  lookup bookkeeping (`store.py:125`). The read-back numbers stand only as
  measurements of blocks credited by the lookup walk.
- **QUALIFIED:** every (d) HBF variability analysis characterises the
  bookkeeping artefact, not the design. That covers STEP 2 timing arms, 2b
  replicates, the j-size study, the concurrency sweep, the pooled bound and the
  discrete-outcome pattern.

- **CORRECTION (PHASE 14, set A): under copy-at-admission, PCM never serves a
  read.** `memory_hit_tokens = 0` in all 332 such runs. (d)'s `if_read` gate
  has been opened by **lookup touches**, not reads:
  - `store.py:125` credits every hit key, including HBM-resident heads.
  - `memory_read_blocks = 0` in the headline (d) runs, yet 7.8k–80k blocks were
    demoted.
  - **Set A fails rule A2:** under the vLLM-style lookup, headline (d) writes
    **0** bytes to HBF in every run. (b) and (c) are unchanged.
  - So the headline (d) HBF range (5.95–11.70 M / 4.08–6.35 M) is a
    lookup-accounting artefact. (d) is in effect "drop every PCM victim". Its
    real cost is the full hit-rate loss (−44% / −10% vs (b)).
  - **The headline tables are not rewritten here.** Which lookup the headline
    uses is a decision pending.

- **CONSTRAINT (PHASE 14 STEP 3a): only TP-rank-0 workers are scheduled**
  (`llm_engine.py:577-579`), so only 4 of 8 ranks ever write to the store.
  - **Every "batch/sched" figure is half the active-scheduler batch.** Cap 48
    is really ~45–46 per active scheduler, not ~23. Gaps between arms are
    unaffected.
  - **The PCM pool (142,213 blocks, sized for 8 GPUs) receives only TP-0
    shards.** Its capacity relative to its write stream is **2× the design**,
    and absolute bytes are one TP shard's worth.
  - Undecided; no number has been re-interpreted yet.

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
- **(d)'s HBF write volume is perturbation-sensitive (PHASE 14 STEP 2).**
  Removing all store time, or scaling the assumed PCM write timing 0.25×/4×,
  moves it by **−10.9% to +57.0%**, non-monotonically. The (b) stream (≤ 0.18%)
  and every PCM stream (≤ 0.54%) do not move. The (b)→(d) *reduction* stays at
  98.41–98.99%.
- **(d)'s HBF write volume has a large run-to-run spread (PHASE 14 STEP 2b).**
  Over 5 clean replicates (burst-level jitter, arrival order unchanged) the
  spread is **26% / 30%** of the mean at cap 48 (conv / T&A) at j = 0.45 × gap
  and 16% / 19% at cap 256. **The spread depends on j:** at j = 0.05 × gap it is
  8.6% / 14.1%, still above 5%. Every (d) range must state its j. The T&A
  ratio of 0.47 is borderline against the 0.5 cut-off. The stock T&A
  (d) run is below every replicate at both j. (b) spreads by ≤ 0.5% and PCM by ≤ 0.9%. Any (d) HBF figure needs
  replicates.
- **Within-second request order is not in the trace.** File order is used.
  Randomizing it (request-level jitter) lowers the (b) hit rate by 2.8–3.4%: the
  stock order favours reuse. For (d) HBF the effect has a consistent direction
  (lower) and falls outside the clean range in 12/20 runs, but its size is not
  pinned down. Against the clean-replicate mean it is −11% to −19%. Against the
  stock run it is −40% to +29% per seed.
- **(d)'s HBF write volume against concurrency is unresolved (PHASE 14
  STEP 3).** It is not shown to be batch-driven, since the means are
  non-monotone across caps. It is not invariant either: on T&A the means span
  40.5% across caps and 18/28 cap pairs have non-overlapping 3-seed ranges. Do
  not call (d) concurrency-invariant.
- **(d)'s outcome is discrete under small perturbations (PHASE 14 STEP 3 extra
  check).** At j = 0.005 × gap some seeds reproduce the stock store outcome
  exactly, and others jump to a few other values. The stock run is one of
  several states, common only at very small perturbation.
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

### 2026-10-12 — PHASE 14: HBF capacity check, admission counter, full-trace headline

- **Decisions:**
  - **Full traces are the headline window.** W = 5,000 is the earlier
    reference.
  - (b) is described exactly: a 71,104-block write-through DRAM buffer in
    front of HBF.
  - CLAUDE.md now says HBF eviction depends on the window.
- **New flags:**
  - `hbf_capacity_mode` (`finite` | `unlimited`): a sensitivity setting, not
    a hardware value.
  - `count_pcm_admit_source`: tags each PCM write as first, after an
    `if_read` drop, or after an HBF eviction.
  - Both are stock by default. 277 tests pass. Byte identity against
    `ced3806` holds, and the counter reproduces STEP 4 exactly.
- **Rule passes.** (b)→(c) changes by ≤ 0.01 pt with unlimited HBF, so
  **(c)'s halving is not driven by HBF capacity**. It is an identity:
  (b) HBF = admissions + demotions, and (c) HBF = demotions. That makes it a
  write-policy effect.
- **HBF evictions are of cold keys.** They cause 0.02–0.09% of PCM admissions
  and move HBF bytes by ≤ 0.09%. This corrects the STEP 4 note that called
  them material re-writes.
- **(d) re-admissions after an `if_read` drop:** 10.46% / 9.77% of (d)'s PCM
  writes.

### 2026-10-12 — PHASE 14 STEP 4 (full traces, headline configuration)

- **14 runs.** Calibration on T&A (b): 8:42 and 11.13 GiB, within the
  estimate. The rest ran 2 at a time; 54 min in total.
- **(c) ≈ 49.3–50.3%** on full traces: raw 50.23 / 50.25%, residue-adjusted
  49.29 / 49.26%. The raw-adjusted gap shrank from 2.2 / 4.8 points to 0.9 /
  1.0. Store benefit kept ~100%.
- **(d):** worst case 99.45 / 99.33%. Its hit-rate cost grows on full traces
  (−46 to −47% / −12%), and it keeps only 9–15% of the store's reuse benefit.
- **Residue share** 3.7–3.8% / 3.9–4.0%: falls roughly in proportion to
  request count, as pre-registered.
- **New regime:** HBF fills in (b)/(c) on full traces (1.5–2.0 M evictions);
  (d) none.
- **Code reading:** the HBF → HBM read path bypasses PCM. There is no
  re-admission of HBF-served blocks.
- **Quoting notes recorded:**
  - (c)'s T&A benefit kept is quoted as ~100%, not 104.3%.
  - (c) was ~48–51% until this step; on full traces it is ~49.3–50.3%.

### 2026-10-12 — PHASE 14: bookkeeping fixes verified (V1, V2)

- **Correction notes:**
  - The read-back mechanism is **retracted** as a mechanism.
  - All (d) HBF variability analyses are **qualified** as characterising the
    artefact.
  - **(c)** is the claim that holds in every model: 50.81 / 50.99%, and
    50.53 / 51.31% under spill.
- **New flags** `read_credit` (`walked` | `served`), `load_span` (`from_start`
  | `missing`) and the `count_if_read_gate` check. Byte-identical at default;
  tests **263**.
- **V1:** (d) HBF = 0, 0 gate passes. The artefact diagnosis is confirmed.
- **V2:** gate passes backed by PCM-served reads, difference 0. (b)→(c) 50.53 /
  51.31%; (b)→(d) worst 99.36 / 99.12%. (d) HBF 14.8–20.7 M / 4.9–12.9 M. (d)
  hit rate vs (b) −37.5 to −39.1% / −6.8 to −7.7%.
- **No headline change**; decision pending.

### 2026-10-12 — PHASE 14: vLLM-style lookup, sets A and B

- **New `store_lookup_start`** (`zero` stock | `after_hbm`, the vLLM connector
  contract). Byte-identical at default; tests **255**.
- **Set A (headline + after_hbm) fails rule A2.**
  - (b) and (c) are identical to the headline.
  - (d) writes **0** to HBF in every run, and its store lookups never hit.
  - The cause is a **correction**: under copy-at-admission PCM never serves a
    read (all 332 runs). The headline (d)'s gate was opened by lookup touches
    of HBM-resident heads (`store.py:125`), so its HBF writes are an
    accounting artefact.
  - Headline tables not rewritten; decision pending.
- **Set B (spill + after_hbm)** is a working spill hierarchy: PCM serves real
  reads, and (b)/(c) keep the headline hit rate.
  - (d) HBF is 18.8–24.4 M / 5.3–15.3 M, above the bands. (b)→(d) ≥ 98.96%.
  - Rules: (i) passes; (ii) and (iii) fail.
  - The headline numbers depend on the PCM entry model.
- **Recorded:** the from-zero spill result ("copy-at-admission required under
  this lookup"), and a stock load-plan constraint: the head keys are read, not
  the external span (`mooncake_store.py:543`).

### 2026-10-12 — PHASE 14 STEP 3b (robustness: HBM spill)

- **New `hbm_evict_save_policy = spill_only`**, with end-of-run residue and
  admission-to-first-spill counters. Byte-identical at default; tests **250**.
- **Verdict (pre-registered rules): the headline does not hold under spill;
  the result depends on the PCM entry model.**
  - Rule (iii) fails on both traces: (d) HBF is 0 / 0.37 M B/req, below the
    bands.
  - Rule (ii) (residue-adjusted) passes on conv (≤ 4.89%) and fails on T&A
    ((b) HBF −6.21%).
  - Rule (i) passes.
- **Cause, measured:** under spill_only the store serves almost no reads. It
  hits 0 / 5,000 (conv) and 769 / 5,000 (T&A) lookups, against ~4,990 in the
  headline. The prompt head stays in HBM and is never spilled, and
  `store.lookup` stops at the first absent key. So `if_read` never passes.
- **Open:** a lookup that starts after the local GPU hit. Not built; needs a
  decision.
- **Future-work line added:** a stricter demotion gate than "read once in PCM".

### 2026-10-11 — PHASE 14 headline follow-ups

- **Bib entry** `ehrmann2022pcm` uses the supplied fields: Ehrmann,
  Blachowicz, Ehrmann, Grethe, "Recent developments in phase-change memory",
  *Applied Research* 1(4) e202200024, 2022.
- **Read-back share is a lower bound** (in-window only). The (b) figure, 3.90%
  / 2.98%, is an observation pending STEP 4.
- **De-confounded read-back test.** At global @ 71,104 vs global @ 142,213
  (capacity only, 8 new runs), **98.2–98.4% (T&A) / 70.6–76.2% (Conv)** of
  (d)'s avoided HBF writes were not read back in-window. That agrees with the
  per_rank comparison, so the mechanism is **lifted from "candidate"**. It
  stands as a within-window result: those blocks pass `if_read` at the larger
  pool and then go unread from HBF.

### 2026-10-11 — PHASE 14 headline set (design capacity, per_rank)

- **Unit resolved: binary.** Pool 71,104 = 4 × 17,776 = 43.4 GiB/GPU. 0.62
  Gb/mm² is now cited (Ehrmann *et al.* 2022) in the LaTeX.
- **New `count_hbf_readback` counter.** Byte-identical when off and
  behaviourally neutral when on. Tests **242**.
- **Headline set** (36 runs; (d) N = 16 per trace):
  - (b)→(c) **50.81% / 50.99%**.
  - (b)→(d) worst case **99.66% / 99.63%**.
  - (d) HBF B/req 5.95–11.70 M / 4.08–6.35 M.
  - (d) hit rate vs (b) −43.0% to −45.9% / −8.6% to −9.2%.
- **Read-back test of the candidate mechanism.** Of the HBF writes (d) avoids
  at the design pool vs the 86.8 GiB/GPU point, **97–98% (T&A) / 70–80%
  (Conv)** were blocks never read back from HBF. This is consistent with the
  candidate, but pool size and partition change together.
- **(b) and (c) read back only 3–4% / 7–8% of what they write to HBF,
  in-window.** That is a **lower bound**: reads after the W = 5,000 window are
  not counted, which biases it against late writes. It is an **observation
  pending STEP 4**, not a finding.
- **Under `per_rank`, (d) is stable to small jitter:** all 5 seeds at
  j = 0.005 reproduce the stock run on both traces.

### 2026-10-11 — PHASE 14, decision record (docs only)

- **Headline = design capacity + `per_rank`.** Decided on hardware grounds:
  die area fixes 43.4 GiB/GPU, and PCM is private per GPU. 142,213 is now the
  86.8 GiB/GPU point. Recorded that this moves (c) down slightly and (d) up.
- **OPEN unit check:** binary 71,104 vs decimal 66,220. It also bears on the
  TLC die area and the HBF stack size. No headline runs until confirmed.
- **(d) cost metrics added** (hit rate vs (b); HBF bytes per reuse-hit block).
  At the design point, (d)'s gain is mostly not lost reuse: Conv reuse −2.5%,
  T&A within its range.
- **CLAUDE.md capacity section rewritten** for writing ranks (on disk;
  gitignored). Pending LaTeX sync gains the citation and unit items.

### 2026-10-11 — PHASE 14, STEP 3a controls (pool sized for 4 writing ranks)

- **Option (B):** every tier is sized for the 4 TP-0 writing ranks. PCM
  71,104 blocks is the 43.4 GiB/GPU design point; HBF is 5,033,164 blocks. The
  TP-1 stream is a mirror and is not simulated. `store_partition_ranks` must
  equal the number of writing ranks (= DP size). Tests **237**.
- **Capacity effect (71,104 vs 142,213):**
  - (d) HBF **−71% / −74%**, ranges do not overlap.
  - (b)/(c) HBF +1.2% to +4.6%.
  - (b)→(c) 50.60% / 50.97%; (b)→(d) **99.64–99.68% / 99.66–99.72%**.
  - PCM residency halves.
- **Sharing effect (per_rank vs global at 71,104):** (d) HBF −13.8% / −24.5%,
  ranges do not overlap. (b)/(c) ≤ 0.22%.
- **Audit:** every "43.4 GiB/GPU" attached to 142,213 means 86.8 GiB per
  writing GPU. The PHASE 10 knee is ~38×, not 19×. Requests-to-wear-out are 2×
  too high. B/req are per TP shard. Batch is restated per active scheduler.
- The headline is **not** replaced; decision pending.

### 2026-10-11 — PHASE 14, STEP 3a (cross-rank eviction)

- **New `store_partition`** (`global` | `count` | `per_rank`) and
  `store_partition_ranks`. Default byte-identical; `count` is behaviourally
  stock; tests 221 → **236**; code `cd62320`.
- **Cross-rank evictions are 73.6–75.9% of all memory-tier evictions** in every
  config, at the 75% expected for 4 equally loaded ranks under a rank-blind LRU.
- **Found: only 4 of 8 ranks write to the store** (TP-0 only; see the
  constraint above). So `per_rank` at 8 partitions left half the PCM pool
  unused (71,104 usable blocks).
- **Under `per_rank`:**
  - (b) HBF +1.4% / +1.9%; (c) +2.1% / +4.5%.
  - (d) HBF **−75% / −76%**, outside the pooled range, so an effect by the
    rule. But it is confounded with the capacity halving.
- **Open:** how the TP-0-only model maps to hardware. That decides both the
  per_rank comparison and the PCM capacity bookkeeping.

### 2026-10-11 — PHASE 14, STEP 3 follow-up (pooled bound)

- **Pooled cap-48 (d) bound,** N = 16 per trace: stock plus burst replicates at
  three j. Worst-case reduction **98.72% / 98.44%**. (d) HBF B/req 32.0–43.0 M /
  17.2–26.3 M. Ratios ×78.3–×105.2 / ×64.3–×98.8.
- **Over every cap tested** (46 runs per trace) the worst case is 98.66% /
  98.28%.
- **Candidate:** outcomes cluster on a few discrete values.
- **Withdrawn:**
  - the periodic-burst-train candidate (an aperiodic train reproduces the
    stock outcome);
  - T&A's j-dependence call (its spread is non-monotone in j).
- **The T&A cap 48→256 +26.86% at j = 0.05** is part of a non-monotone swing,
  not a batch effect.

### 2026-10-11 — PHASE 14, STEP 3 (concurrency sweep for (d))

- **Sweep:** (d) at caps 8–256, both traces, 3 seeds per cap, burst jitter at
  j = 0.05 × gap. That j is the stricter test; at 0.45 the threshold would be
  26–30%. 48 runs. The cap-48 runs reproduce the prelude runs byte for byte.
  (b) reuses the PHASE 9 STEP 4 rows, plus new stock T&A cap-192 and cap-256
  runs, which PHASE 9 never had.
- **Verdict: "not shown" on both traces.** The means are non-monotone. The cap
  8→256 change is +15.01% / +2.45%, against largest spreads of 23.76% / 20.22%.
- **Flagged:** T&A (d) means vary 40.5% non-monotonically, with 18/28 cap pairs
  non-overlapping. So (d) is not invariant either.
- **Batch gaps:** (d) vs (b) grows with cap, to −4.08% (Conv) and −6.60% (T&A).
- **The (b)→(d) reduction holds at every cap:** 98.69–98.99% / 98.32–98.88%.
- **Extra check at j = 0.005:**
  - T&A seed 4 and Conv seed 5 reproduce the stock store outcome exactly.
  - Outcomes repeat in discrete values.
  - T&A spread against j is non-monotone (33% → 14% → 30%).
  - The "periodic burst train" explanation is a candidate only: an aperiodic
    train also reproduces the stock outcome.
- Notes after the prelude review: the request-mode wording is revised, the
  reference of every delta is stated, and T&A's j-dependence is marked
  borderline.

### 2026-10-11 — PHASE 14, STEP 3 prelude

- **Jitter size.** (d) cap 48 at j = 0.05 × gap (15.685 / 30.287 ms) spreads
  **8.55% / 14.06%**, against 25.99% / 30.01% at j = 0.45. That is a ratio of
  0.33 / 0.47, below half on both traces. **The spread depends on j**, so (d)
  ranges now carry their j. It is still above 5% at small j.
- **(c) replicates.** HBF spread 0.20% / 0.25%. The same-seed (b)→(c)
  reduction is **51.07–51.25% / 52.02–52.25%**, which contains the headline
  51.10% / 52.18%.
- **Request-mode table note.** Its (d) HBF shifts are within (d)'s spread, or at
  its edge (T&A cap 256: −18.88% against 19.26%), so they are not readable as
  effects. Counter-evidence recorded: all four shifts are negative, and 12/20
  runs fall outside the clean range.

### 2026-10-07 — PHASE 14, STEP 2b revised (clean replicates)

- **New `--arrival_jitter_mode`.** `request` is the earlier per-request jitter,
  now named and treated as a workload variant. `burst` gives one offset per
  burst on the replay clock and **fails the run unless arrival order equals file
  order**. Byte-identical at default; tests 213 → **221**; code `20695c3`.
- **Burst gaps in replay time are almost constant:** 313.7 ms (conv) and
  605.7 ms (T&A). That gives j = 0.45 × min = **141.168 / 272.582 ms**, a
  methodology choice. The order guard never fired in 40 runs.
- **Request-mode runs recorded as an intra-burst-order sensitivity result.**
  11–13% (conv) and 27–29% (T&A) of continuations end up before their
  predecessor. (b) hit rate −2.8% to −3.4%.
- **Clean replicates:**
  - (b) is neutral against stock: HBF ±0.15%, hit rate within +0.52%.
  - (d) HBF spread is **26% / 30%** at cap 48 and 16% / 19% at cap 256.
  - The stock T&A (d) run is **below all five** replicates at cap 48 and
    **above all five** at cap 256.
- **Q1:** conv yes (the replicates cover the STEP 2 range). T&A no: the STEP 2
  low end is the stock run itself, and its high end is 2.4% above the replicate
  max.
- **Q2:** the single-run +70.10% exceeds both spreads but is not reproduced.
  Replicate means differ by +17.56% and the ranges overlap, so batch is not
  shown to drive it.
- **STEP 3 rule:** the spread is > 5%, so STEP 3 runs 3 replicates per point.

### 2026-10-07 — PHASE 14, STEP 2b (replicates)

- **`--random_seed` is inert.** (d) T&A seeds 0/1/2 are byte-identical. The
  seed reaches only synthetic workloads and MoE routing.
- **New flags `--arrival_jitter_ms` / `--jitter_seed`.** Default 0, and stock
  output stays byte-identical. Tests 205 → **213**. Code `5108029`.
- **The 1 ms resolution premise is false.**
  - The trace's timestamps are whole seconds, clustered ~3 s apart: 532 / 276
    bursts in W = 5,000.
  - ±0.5 ms jitter only permutes order within a burst, and 24.7% / 56.5% of
    continuations share a burst with their predecessor.
  - Result: (b) hit rate is lower in 20/20 jittered runs and HBF B/req higher
    in 20/20. The replicates are a workload variant, not noise around the stock
    run.
- **Q1:** replicates alone do not cover the STEP 2 (d) range on either trace.
  The widths are comparable (1.79× / 0.89×); the location is shifted.
- **Q2:** (d) T&A cap-256 vs cap-48 replicate means differ by +16.53%, and the
  ranges overlap. The single-run +70.10% lies inside the pairwise range
  (−20.5% to +70.7%). Batch is not shown to drive it.
- Open: a perturbation that keeps within-burst order. Awaiting a decision.

### 2026-10-07 — PHASE 14, STEP 2

- New flags, both default-stock:
  - `store_timing_mode` (`charged` | `free`).
  - `count_aligned_save_skips`, instrumentation only.
- Stock output is byte-identical to `ced3806` on (b) and (d), both traces.
  Tests: 190 → **205**. Code commit `9e8b4d1`.
- **The pre-registered 5% rule failed, on (d)'s HBF stream only.**
  - (b) HBF ≤ 0.18%, and every memory-tier/PCM stream ≤ 0.54%: not
    contaminated by timing.
  - (d) HBF: Conv −1.0% to −10.9%; T&A +4.3% to +57.0%.
  - The response is non-monotone: PCM write latency 0.25× and 4× both raise
    T&A (d) HBF (+41.5%, +34.4%).
- The PHASE 9 cap-48 (d) value sits at an edge of the spread on both traces, so
  ×98.8 (T&A) is the top of ×62.9–×98.8. The reduction stays 98.41–98.99%.
- Not run: seed/jitter replicates. They would separate perturbation
  sensitivity from timing.

### 2026-10-07 — PHASE 14, STEPs 0–1

- Branch `phase14-fixes` cut from `main` at `ced3806`.
- **STEP 0 gate passed.** Configs (b) and (d), cap 48, both traces: all 20 values
  in the "Matched batch (cap 48)" table reproduce to the last digit. The result
  files are also byte-identical (stripped) to the PHASE 13 runs.
- **STEP 1 is paperwork only:**
  - Recorded as an unquantified limitation: pool keys embed the tp/pp/dp rank,
    yet one LRU and one capacity counter serve all 8 workers.
  - Flagged "~10^8 PCM endurance" (findings.md ~1431, ~1892) as unsourced. PCM
    endurance stays OPEN.
  - Recorded as assumed: the `pcm_*` timing defaults, and the HBF write timing
    (200 µs, 3.0 GiB/s), which is TokenSim's stock SSD default.
- **This report's headline now uses the PHASE 9 STEP 3 cap-48 figures:**
  51.10/52.18%, 98.76/98.99%, ×80.4/×98.8. These replace the cap-256 reductions
  and the PHASE 8 multipliers ×90.31/×86.39. Every lifetime ratio is now paired
  with bytes/request.
- The cap-256 table was labelled "cap 256 / 96". All its rows are cap 256, so
  the label is corrected.

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

> Config (b) = a write-through DRAM buffer (same pool size as the PCM tier) in
> front of HBF, write-through + always — not "HBF only". (c) = +PCM, write-back +
> always. (d) = +PCM, write-back + if\_read. All at `H3-B200-KVONLY`,
> LLaMa2-70B-GQA TP2/DP4, FP16, block 2,621,440 B, PCM pool 142,213 blk
> (347.2 GiB), HBF pool 10,066,329 blk (24 TiB), W = 5,000, 30 QPS.

### HBM-saturated (cap 256, both traces), PHASE 9 STEP 3 (revised) — saturated, NOT the headline

> **Saturated operating point.** Not matched to the cap-48 standard. Batch gaps
> are larger here: **(d) vs (b) −4.72% (Conversation), −7.68% (Tool&Agent)**,
> and (c) vs (b) −1.63% / −2.31%. Use the cap-48 table below for headline
> figures.
>
> **Label correction (2026-10-07).** This heading used to say "cap 256 / 96".
> All six rows are from `--max_parallem_sum 256` (findings.md ~line 3510). The
> T&A rows match the cap-256 runs, e.g. (b) 1,695,909,675 B/req, and not the
> cap-96 STEP 4 point (1,678,879,752 B/req).

| | (b) | (c) | (d) |
|---|---|---|---|
| **Conversation** HBF write blocks | 6,461,439 | 3,150,432 | **80,968** |
| HBF bytes/request | 3,387,654,930 | 1,651,733,692 | **42,450,551** |
| prefix hit rate | 0.1631 | 0.1641 | 0.0942 |
| achieved batch, raw `batch_mean_per_scheduler` (÷ 8 workers) | 43.61 | 42.90 | 41.55 |
| **batch per active scheduler (TP group)** = 2 × raw | **87.22** | **85.80** | **83.10** |
| **reduction vs (b)** | — | **51.24%** | **98.75%** |
| **Tool&Agent** HBF write blocks | 3,234,691 | 1,534,945 | **55,703** |
| HBF bytes/request | 1,695,909,675 | 804,753,244 | **29,204,414** |
| prefix hit rate | 0.4014 | 0.4033 | 0.3733 |
| achieved batch, raw `batch_mean_per_scheduler` (÷ 8 workers) | 48.44 | 47.32 | 44.72 |
| **batch per active scheduler (TP group)** = 2 × raw | **96.88** | **94.64** | **89.44** |
| **reduction vs (b)** | — | **52.55%** | **98.28%** |

### Matched batch (cap 48), PHASE 9 STEP 3 — the reproduction gate

| | HBF write bytes | HBF blocks | hit rate | reuse hit blk | raw `batch_mean_per_scheduler` (÷ 8 workers; the gate value) | **batch per active scheduler (TP group)** |
|---|---|---|---|---|---|---|
| (b) Conversation | 16,853,376,696,320 | 6,429,053 | 0.1643 | 670,585 | 23.167889 | **46.335778** |
| (d) Conversation | 209,508,106,240 | 79,921 | 0.0974 | 397,596 | 22.378200 | **44.756400** |
| (b) Tool&Agent | 8,483,936,665,600 | 3,236,365 | 0.4037 | 1,174,042 | 22.885048 | **45.770096** |
| (d) Tool&Agent | 85,844,295,680 | 32,747 | 0.3650 | 1,061,727 | 22.429517 | **44.859034** |

**Any new campaign must reproduce this table to every digit before its own
numbers are trusted.**

> **Units, from PHASE 14 STEP 3a:**
> - **Batch.** Only the 4 TP-0 workers are scheduled. Raw
>   `batch_mean_per_scheduler` divides by all 8 workers, so batch is now quoted
>   **per active scheduler (TP group) = 2 × raw**. The raw column stays because
>   it is the reproduction-gate value.
> - **Bytes.** All B/req and write-byte figures are **per TP shard** (one of
>   2).
> - **Capacity.** 142,213 blocks is **86.8 GiB per writing GPU**, 2× the
>   design.

### 43.4 GiB/GPU design point — CONTROL, not the headline (PHASE 14 STEP 3a)

PCM 71,104 blocks (4 writing ranks × 17,776) and HBF 5,033,164 blocks, cap 48.
(b) and (c) are single runs; (d) is a range over 4 runs (unjittered + burst
j = 0.05 × gap, seeds 1–3). B/req per TP shard. **The headline above
(142,213) is unchanged pending a decision.**

| | (b) HBF B/req | (c) HBF B/req | (d) HBF B/req | (b)→(c) | (b)→(d) | (d) hit rate | batch / active sched |
|---|---|---|---|---|---|---|---|
| Conversation | 3,411,597,591 | 1,685,218,918 | 10.9–12.2 M | 50.60% | 99.64–99.68% | 0.0894–0.0946 | 44.9–46.3 |
| Tool&Agent | 1,730,365,358 | 848,324,723 | 4.85–5.94 M | 50.97% | 99.66–99.72% | 0.3663–0.3666 | 45.0–46.1 |

Against 142,213, the capacity effect on (d) HBF is −71% / −74% (ranges do not
overlap). On (b) and (c) it is +1.2% to +4.6%. Per-rank partitioning at the same
capacity moves (d) HBF a further −13.8% / −24.5% (ranges do not overlap), and
(b)/(c) by ≤ 0.22%.

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

> **One LaTeX rewrite, after STEP 4.** Every item below is listed, not applied.

- [ ] **NEW HEADLINE (2026-10-12).** Spill-only + vLLM-style lookup + honest
      read bookkeeping, 43.4 GiB/GPU per_rank, cap 48.
  - Rewrite the abstract, contributions, `tab:headline`, `sec:headline` and
    the Conclusion around **(c) as the primary result** (~50.5–51.3% in every
    model, no hit-rate cost).
  - Present (d) as the trade-off end: worst-case 99.36% / 99.12%; hit rate
    −37.5 to −39.1% / −6.8 to −7.7%; keeps 15–19% / 10–21% of the store's
    reuse benefit.
  - Add the no-store control, the PCM-served share, the residue bias, and the
    copy-at-admission variant ((d) ≡ no store).
  - **State that the headline was chosen on modelling grounds and lowers (d)'s
    reduction.** STEP 4 full-trace numbers replace the W = 5,000 ones if they
    differ.
- [ ] **STEP 4 (full traces) — DECIDED: full traces are the headline window**
      (2026-10-12). W = 5,000 can appear as an earlier reference only.
  - (c) ~49–50%: raw 50.23 / 50.25%, residue-adjusted 49.29 / 49.26%.
  - (d) worst case 99.45 / 99.33%; (d) hit rate −47% / −12% vs (b); store
    benefit kept 9–15%.
  - Note the HBF-capacity regime on full traces: (b)/(c) evict, (d) does not.
- [ ] **Describe (b) exactly.** It is a 71,104-block write-through DRAM buffer
      in front of HBF, serving 6.9% / 7.2% of (b)'s store-served tokens on full
      traces. Not "HBF only": retire that label wherever it appears.
- [ ] **(c)'s halving is a write-policy identity** ((b) HBF = admissions +
      demotions, (c) HBF = demotions). State the HBF capacity check result and
      how (c) is framed (decision pending; see findings.md "HBF CAPACITY
      CHECK").
- [ ] **Read path:** PCM or HBF serve directly into HBM. HBF hits are not
      re-admitted to PCM (`store.py:179-182`, `:234-237`).
- [ ] **Architecture / method:** describe spill-only entry (HBM evicts → PCM),
      the vLLM-style lookup beyond the HBM hit, and read credit on actual
      serve. Retire "each prompt block is copied to PCM at admission" as the
      model description; it is the variant.
- [ ] **Retract** any read-back mechanism text ("one read in PCM passes
      `if_read`"), and any (d) variability, concurrency or j-size discussion
      presented as a property of the design.

- [ ] **Architecture/Methodology wording (PHASE 14 STEP 3b).** State plainly
      that **"each prompt block is copied to PCM at admission"** (the prefill
      save). Nothing should imply that HBM evictions spill into the store.
      Lines that currently imply it: l.102 ("a KV store that absorbs spill
      from HBM"), l.170 ("HBF receives the spill stream from a KV store") and
      l.379 ("The residual is the genuine spill"). Reword them to "the store's
      overflow/demotion stream".
- [ ] **From-zero result, one sentence:** *"If blocks enter the store only on
      HBM eviction and the store is looked up from the first prompt block, the
      HBM-resident prompt head is never stored and the store serves almost no
      reuse (0 and 769 of 5,000 lookups); copy-at-admission is required for
      the store to be useful under that lookup."*
- [ ] **Robustness paragraph for Evaluation, rewritten from set B** (replaces
      the draft below). Draft: *"Robustness to the PCM entry model. We also
      let blocks enter the store only when HBM evicts them, with the store
      queried, as in vLLM's connector interface, only for tokens beyond those
      already in HBM. PCM then behaves as a victim cache and serves real
      reads. Write-back halving is unchanged (50.5% / 51.3%), and read-gated
      demotion still cuts flash writes by at least 98.96% against the
      flash-only baseline. But (d)'s residual flash traffic is 1.6–4.1× the
      copy-at-admission figure (18.8–24.4 MB vs 6.0–11.7 MB per request,
      conversational). By our pre-registered rules the headline numbers
      therefore depend on how blocks enter PCM."*
- [ ] **Read-gate accounting (PHASE 14 correction).** Any (d) mechanism text
      must say that under copy-at-admission PCM serves no reads, and that the
      gate was opened by lookup touches (`store.py:125`). Wait for the
      headline-lookup decision before rewriting.
- [ ] *(superseded by the set-B paragraph above)* **Robustness paragraph for
      Evaluation (PHASE 14 STEP 3b)** — one short paragraph; headline tables
      unchanged. Draft:
      *"Robustness to the PCM entry model. If blocks enter the store only when
      HBM evicts them, rather than at admission, the result does not carry
      over. The HBM-resident prompt head is then never stored, and
      prefix-ordered lookups stop at it: the store serves 0 of 5,000 lookups
      (conversational) and 769 of 5,000 (tool/agent). Read-gated demotion
      therefore never passes, and (d) writes essentially nothing to flash
      (0 and 0.37 MB per request). By our pre-registered rules the headline
      does not hold under spill-only entry (rule iii fails on both traces,
      rule ii on tool/agent), so the result depends on how blocks enter PCM."*

- [x] **0.62 Gb/mm² cited (2026-10-11).** `\cite{ehrmann2022pcm}` is at the
      contribution item and the density line, replacing "derived from
      published die parameters". The bib entry is complete (supplied fields).
- [ ] **Mechanism paragraph for (d).** Within the window, ~98% (T&A) /
      ~70–76% (Conv) of the HBF writes (d) avoids at the design pool, against
      2× capacity, are blocks that pass `if_read` at the larger pool but are
      not read back from HBF. Read-back figures are in-window lower bounds.
- [ ] **Still uncited: 35 mm² (PCM die) and 29 Gb/mm² (TLC density).** The
      29 Gb/mm² unit stays OPEN until sourced. Note the 35 vs 35.3 mm²
      rounding (< 1% capacity).
- [ ] **Add a limitation line:** 2nd-gen (4-deck) XPoint roughly doubles
      density, which is approximately the 86.8 GiB/GPU point (142,213, global
      pool). No runs for it.
- [ ] **Wording rule:** nowhere call 0.62 Gb/mm² "conservative" overall. A
      smaller pool lowers (c)'s reduction and raises (d)'s.
- [ ] *(superseded by the items above)* **Uncited die figures (PHASE 14
      decision record).** l.207–216 give 35 mm² per PCM die, 0.62 Gb/mm² PCM
      density and 29 Gb/mm² TLC density with **no `\cite`**.
  - **TLC sets two things in this paper:** PCM capacity (the PCM die is the
    area of a 1 Tbit TLC die) and HBF endurance (3000 P/E). Both need a
    source.
  - **Unit.** l.210–212 turn 43.4 Gbit/stack into 5.43 GiB, i.e. binary Gbit.
    If the density source is decimal, PCM is 40.42 GiB/GPU. The "1 Tbit die
    ≈ 35.3 mm², 128 GiB" (l.215–216) also assumes binary Tbit; decimal gives
    34.5 mm². Fix all three together once confirmed.
- [ ] **Headline configuration change (PHASE 14 decision record) — numbers now
      available (headline set).** The headline becomes design capacity +
      `per_rank`:
      - (b)→(c) 50.81% / 50.99%;
      - (b)→(d) worst case 99.66% / 99.63% (N = 16), with (d) HBF B/req
        5.95–11.70 M / 4.08–6.35 M;
      - (d) hit rate vs (b) −43.0 to −45.9% / −8.6 to −9.2%;
      - HBF bytes per reuse-hit block and HBF read-back share as in findings.

      Earlier wording: Every
      `tab:headline` number, the abstract and the Conclusion change. The
      current 142,213-block figures are the **86.8 GiB/GPU point**. Add the
      (d) cost metrics: hit rate vs (b), and HBF bytes per reuse-hit block.

- [ ] **8-writer assumptions (PHASE 14 STEP 3a audit).** List only, not fixed.
      Only 4 TP-0 workers are scheduled and write.
  - **Capacity.** "43.4 GiB per GPU" is attached to 142,213-block results: the
    abstract l.56 and l.69, contributions l.112, l.288 (pool = 43.4 × 8), the
    Conclusion l.608. At 142,213 the pool is **86.8 GiB per writing GPU**. The
    design point is 71,104 blocks.
  - **Endurance budget.** l.172 gives `24 × 2^40 × 3000 = 7.92e16 B`, but only
    the TP-0 half (12 TiB) receives the counted per-shard bytes. The
    requests-to-wear-out figures (l.381–382, `tab:capacity` l.442–456) are
    **2× too high**. Their ratios are unaffected.
  - **Capacity sweep.** `tab:capacity` "GiB/GPU" column ×2. l.466: 5.3% of
    footprint is at 86.8 GiB per writing GPU; the design point is 2.6%. l.476:
    "822 GiB per GPU, 19× the budget" → ~1,645 GiB, ~38×. l.481–483:
    "1.8×–2.8×, 77–120 GiB" → 3.6×–5.6×, 154–240 GiB. l.71, l.121 and l.613:
    "19×" → ~38×.
  - **Batch.** l.325, l.356, l.365 ("achieved batch/sched"), l.385, l.394 and
    `tab:batch` l.404 must be quoted **per active scheduler (TP group)**, which
    is 2× the old figures. Those were divided by 8 workers including 4 idle
    TP-1 workers. The cap is per active scheduler; cluster concurrency is
    4 × cap.
  - **Platform.** l.278 "eight B200-class workers" and l.285 "single
    cluster-wide pool": add that the TP-1 stream is an exact mirror of TP-0
    and is not simulated, and that bytes are per TP shard.
  - **Possible headline change.** If the 43.4 GiB/GPU design point (71,104
    blocks) becomes the headline, `tab:headline` and every (d) figure change:
    (b)→(d) 99.64–99.72%, (b)→(c) 50.60% / 50.97%. Decision pending.

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
- [ ] **Stale headline figures (PHASE 14 STEP 1d), listed only, not fixed.**
      Line numbers are in `pcm-kv-micro.tex` at `ced3806`:
  - Abstract, l.59–63: "51.2% / 52.6%", "98.8% / 98.3%" (cap-256, saturated)
    and "90.3× / 86.4×" (PHASE 8, batch-unmatched). Cap-48 values:
    51.10% / 52.18%, 98.76% / 98.99%, ×80.4 / ×98.8, each paired with
    bytes/request.
  - `tab:headline`, l.340–370: the cap-256 saturated table is presented as the
    headline. It does not state the (d)-vs-(b) batch gaps (−4.72% / −7.68%).
  - Evaluation text, l.379–382: "98.75% and 98.28%" and "90.3× and 86.4×". No
    bytes/request is paired with the ratio.
  - l.384–385: "matched to within 4.7% and 7.7%" is the saturated spread. It
    should be labelled as such.
  - Conclusion, l.609–611: "51–53%", "98–99%", "90×".
- [ ] **Unsourced statements in the LaTeX (PHASE 14 STEP 1b/1c), listed only:**
  - l.177: PCM "write endurance is orders of magnitude higher". This is
    unsourced, and PCM endurance is OPEN.
  - l.273–274: flash writes "NAND-realistic at 200 µs and 3 GB/s". These are
    TokenSim's stock SSD defaults, not sourced HBF figures, and the knob is in
    GiB/s.
  - The PCM media timing (0.1/1.0 µs, 100/20 GiB/s, `config.py:81-84`) is never
    stated in the paper. It is assumed and should be disclosed as such.
- [ ] **HBF write bandwidth unit (PHASE 14 STEP 1).** The LaTeX (l.274,
      `\SI{3}{\giga\byte\per\second}`) and the project record (findings.md
      ~2802, ~3829, ~4531) say HBF writes run at "3 GB/s". The knob
      `ssd_write_bw_gbps = 3.0` is divided by 2^30 (`ssd.py:116-121`), so it is
      **3.0 GiB/s** (3.22 GB/s).
- [ ] **Headline ratio mismatch (PHASE 14 STEP 1e).** The prose says ×90.3 / ×86.4
      (PHASE 8), but `tab:headline`'s own B/req give ×79.8 / ×58.1 (cap 256). The
      canonical figure is ×80.4 / ×98.8 (cap 48), quoted with its bytes/request.
- [ ] **(d) is now quoted as a pooled bound (PHASE 14 STEP 3).** This
      supersedes the item below. Use the worst-case reduction 98.72% (conv) /
      98.44% (T&A), with N = 16 and the (d) HBF B/req range 32.0–43.0 M /
      17.2–26.3 M at cap 48. Ratios only as ranges: ×78.3–×105.2 / ×64.3–×98.8.
- [ ] **(d) must be quoted as ranges (PHASE 14 STEP 2).** The abstract,
      `sec:headline` and the Conclusion must quote config (d) as a reduction
      range plus an HBF B/req range, not a single ratio:
      - reduction 98.76–98.89% (conv) / 98.41–98.99% (T&A)
      - (d) HBF B/req 37.3–41.9 M / 17.2–27.0 M
      - if a ratio is given at all: ×80.4–×90.3 / ×62.9–×98.8

      Superseded by the STEP 2b clean replicates. Use reduction 98.76–99.05% /
      98.44–98.83%, (d) HBF B/req 32.0–41.9 M / 19.8–26.3 M and, if a ratio is
      given, ×80.5–×105.2 / ×64.3–×85.7. State that the PHASE 9 T&A ×98.8 lies
      above every replicate.
- [ ] **Concurrency claim over-reaches (PHASE 14 STEP 1f, STEP 3).** Replace
      any "concurrency-invariant" wording for (d) in `sec:batch` and the
      Conclusion with:
      *"reduction ≥ 98.66% (conversational) / ≥ 98.28% (tool/agent) at every
      cap tested; residual HBF writes vary by up to 42% / 70% across caps and
      runs, with no trend."*
      - Those numbers are the worst case over all 46 (d) runs per trace at
        caps 8–256 (stock + burst-jitter replicates). The cap-48 pooled bound
        alone gives 98.72% / 98.44%, but caps other than 48 go lower. The
        T&A worst case is the stock cap-256 run.
      - Residual (d) HBF B/req spans 32.0–45.4 M (conv) and 17.2–29.2 M (T&A).
      - Keep "concurrency-invariant" for (b) and (c) only.
