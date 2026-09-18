# Scope — making HBF wear depend on HBM pressure

Status: **scope only, nothing implemented.** Written after Phase 9, which found
that HBF write volume is flat in batch size (0.5–1.1% across a 12× concurrency
range) because HBM eviction never reaches the Mooncake store.

The one claim this unlocks: **"below batch N, HBF stays effectively read-only;
PCM moves N from X to Y."** Phase 9 STEP 4 shows that claim is not derivable
from TokenSim as it stands — not because the measurement was wrong, but because
the causal link does not exist in the model.

---

## 1. What the simulator does today

Two structures that the architecture treats as one hierarchy are in fact
parallel and unconnected.

**The store's only writer is the prefill save path:**

| step | location |
|---|---|
| scheduler admits a request for prefill | `llm_scheduler.py:194-241` |
| `saves` built from `scheduler_output.scheduled`, once per request | `mooncake_store.py:119-153` |
| whole prompt written (`prefill_len // block_size` blocks) | `mooncake_store.py:329-331` |
| existence prefilter under `save_policy="mooncake"` | `mooncake_store.py:333-340` |
| `put_with_timing` | `mooncake_store.py:186-193` → `store.py:168` |

**HBM eviction writes nothing.** There are three ways a GPU block leaves a
request, and none of them touches the store:

| event | location | what it does |
|---|---|---|
| request preempted | `llm_scheduler.py:319-325` | `release_request_blocks`, then requeue |
| request finished | `llm_scheduler.py:365-367` | `release_request_blocks` |
| **cached block repurposed under pressure** | `block_manager.py:38-42` → `kv_cache_manager.py:158-161` | drops the prefix-cache entry, clears metadata |

**Store keys exist only for prompt blocks.** `pool_keys_for_request` truncates
to `full_input_blocks = req.prefill_len // req.block_size`
(`pool_key.py:59-62`). Decode-generated blocks have **no keys**, so they cannot
be put even in principle. `output_hash_ids` exists as an optional trace field
(`loaders.py:128`, `:215`; `kv_cache_manager.py:104-135`) but **our traces do
not carry it** — `conv_5000.jsonl` / `toolagent_5000.jsonl` have only
`timestamp, input_length, output_length, hash_ids, cache_salt` — so
`register_output_blocks` short-circuits at `kv_cache_manager.py:105`.

### Consequence

Admissions to the store ≈ the workload's distinct prompt-block footprint
(~3.29 M blocks conversation, ~1.69 M Tool&Agent), fixed by the trace. Against
a 142,213-block PCM pool that is a **23× oversubscription present at every batch
size**, so the pool overflows into HBF at the same rate whatever HBM is doing.

---

## 2. The correct hook — and why the obvious one is wrong

The obvious change ("route `release_request_blocks` into `store.put`") **would
not produce batch dependence.** Every request finishes eventually, so every
request's blocks are released regardless of concurrency; the write stream would
stay flat, and we would have rewritten the model for nothing.

**The capacity-driven event is `KVCacheManager.evict`**
(`kv_cache_manager.py:158-161`), called from exactly one place —
`BlockAllocator.allocate` (`block_manager.py:38-42`) — when a block on the free
queue is still `cached` and has to be repurposed because HBM has run out.

That is the real HBM eviction, and it is genuinely batch-dependent:

- **Small batch:** free blocks are plentiful, finished requests' blocks stay
  cached and reusable, `evict` rarely fires → little or nothing reaches the
  store → **HBF read-only regime.**
- **Large batch:** HBM is under pressure, cached blocks are repurposed
  constantly, `evict` fires on every allocation → full spill → **HBF write
  stream.**

This is the mechanism the thesis argues about, and it is one function.

---

## 3. Workstreams

### A — demote on HBM eviction (small, the core of the change)

Give `KVCacheManager` an optional demote callback, invoked from `evict` with the
block's `block_hash`; wire it through `BlockManager` → `LLMPagedAttnScheduler`
→ the connector → `MooncakeStore.put`. The connector already owns a block
releaser hook (`set_block_releaser`, `mooncake_store.py:89-90`,
`llm_scheduler.py:180-181`), so the plumbing pattern exists and can be copied.

Needs a `PrefixCacheKey → PoolKey` mapping. Today `PoolKey`s are built
per-request from `req.hash_ids` (`pool_key.py:47-80`) while the prefix cache
keys blocks by `PrefixCacheKey` (`prefix_cache.py`). Both derive from the same
hash ids, so the mapping is mechanical, but it must be written and tested —
an evicted block is identified by its cache key, not by a request.

**Estimate: ~120 LOC across 4 files, plus ~10 unit tests.**

### B — retire the prefill-time whole-prompt save (small code, large blast radius)

Delete or gate the `saves` construction at `mooncake_store.py:128-146`. The code
change is trivial; the consequences are not (§4).

Recommend adding a third `save_policy` value — `"on_evict"` — alongside the
existing `"mooncake"` / `"every_step"` / `"once"`
(`config.py:SUPPORTED_SAVE_POLICIES`) rather than changing the default. That
keeps every prior phase reproducible and makes the two models a config switch,
which is also the only honest way to present the comparison.

**Estimate: ~40 LOC, but see §4 — this is where the cost is.**

### C — keys for generated blocks (the real work)

Under (A) the evicted set includes decode blocks, which have no keys today.
Options:

| option | what it means | cost |
|---|---|---|
| **C1 — demote prompt blocks only** | decode blocks evicted from HBM are discarded, as now | none; but it **understates** the transient write stream, which is precisely the thesis subject |
| **C2 — synthesize `output_hash_ids` in the trace builder** | extend `build_trace.py` to emit `"<block>.<i>"` ids for output blocks | ~30 LOC in the trace builder + trace regeneration; `register_output_blocks` already consumes the field |
| **C3 — request-private keys for decode blocks** | generated blocks get a key unique to their request | ~50 LOC; guarantees they are never reused |

**C2 vs C3 is a modelling decision, not an implementation detail**, and it
changes the answer:

- Under **C3**, generated KV can never produce a hit — it is pure write traffic.
  That maximises the measured PCM benefit and is the most favourable reading for
  the thesis, which is a reason to be careful with it.
- Under **C2**, generated KV can be hit by a later turn — but **the Mooncake
  trace already encodes that reuse on the prompt side.** Its `hash_ids` are
  content-addressed 512-token block ids (`build_trace.py`), so when turn N's
  output reappears inside turn N+1's prompt it already carries a block id that
  the store can match. Adding output blocks under C2 therefore risks
  **double-counting the same reuse**, once as an output block and once as the
  next turn's prompt block.

**Recommendation: C3, reported explicitly as an upper bound on generated-KV
write traffic**, with C1 run as the lower-bound control. C2 should not be used
without first establishing, on the source trace, how turn boundaries map to
block ids — which the current trace files do not record.

**Estimate: C3 ~50 LOC + ~6 tests; C1 free; C2 ~30 LOC + a trace-structure
study of unknown size.**

### D — metrics reconciliation (medium, and easy to underestimate)

Several headline metrics change meaning and must be re-derived, not just
re-measured:

| metric | why it moves |
|---|---|
| `prefix_cache_hit_rate`, `reuse_hit_blocks` | today a store lookup can hit a block saved at prefill by a request that is *still running*; under (B) it can only hit something already evicted from HBM |
| `mooncake_admission_count` | stops being ≈ the workload footprint, becomes a function of HBM pressure — the whole point, but it invalidates every admission-derived figure |
| `mooncake_mean_residency_s` | Phase 9 established residency = pool ÷ admission rate; admission rate becomes batch-dependent, so residency finally becomes batch-dependent too |
| `effective_prefill_tokens` | the store and the GPU prefix cache stop being independent lookups and become two levels of one hierarchy; double-counting must be checked |
| `mooncake_memory_bypass_blocks` | `store.py:228-232` path is reached far more often when admissions are bursty |

**Estimate: ~80 LOC of metric plumbing + a careful audit. Budget more time for
the audit than the code.**

---

## 4. What breaks

**Tests.** 128 tests across the affected files (`test_mooncake_simulator.py` 67,
`test_architecture_boundaries.py` 34, `test_prefix_cache.py` 14,
`test_kv_transfer_connector.py` 11, `test_kv_transfer_integration.py` 2). Six
assert prefill-save semantics directly and would need rewriting or gating —
`test_save_happens_once_during_prefill_and_never_on_decode` (`:536`),
`test_load_step_precludes_save` (`:552`),
`test_save_transfers_only_missing_blocks` (`:569`),
`test_every_step_legacy_policy_resaves_full_prompt` (`:589`), plus two in
`test_kv_transfer_connector.py`. Gating behind a new `save_policy` (B) keeps all
of them passing unchanged, which is the main argument for that approach.

**Prior findings.** Every number in Phases 5–9 was measured under the current
model. Under `save_policy="on_evict"` none of them is directly comparable:
admissions, residency, hit rates and write volumes all shift. **Phases 8 and 9
would need rerunning to produce a matched pair**, and `findings.md` would need
both, clearly labelled — the old model is not *wrong* for the write-policy
comparison (see below), so it should not simply be replaced.

**What does NOT break.** The (b)/(c)/(d) comparison is internal to the store —
`write_through` vs `write_back`, `always` vs `if_read` operate on blocks after
admission and do not care how they arrived. The 51% / 99% write reductions and
the ×90 / ×86 lifetime multipliers should survive qualitatively. If they do not,
that is itself a finding and should be reported as one.

---

## 5. Validation plan

The change is only worth making if it produces the batch dependence. Check that
first and cheaply:

1. **Falsifier, before any metrics work.** Run caps 8 and 96 under
   `save_policy="on_evict"` with (b) only. If HBF bytes per request still differ
   by <5%, the hypothesis is wrong and the remaining workstreams should be
   abandoned. **This is one day's work and gates the other five.**
2. **Read-only regime exists.** At the smallest batch, HBF write blocks should
   approach 0 — the anchor Phase 9 could not produce.
3. **Conservation, as in every prior phase.** `demoted + dropped == memory
   evictions`; `ssd_write_bytes == ssd_write_blocks × block_bytes`;
   `notdone = 0`.
4. **New invariant.** Store admissions ≤ HBM eviction events; every admitted key
   corresponds to a block previously resident in HBM.
5. **Regression.** With `save_policy="mooncake"` all Phase 8/9 numbers must
   reproduce to the digit — the reproduction check in Phase 9 STEP 2 is the
   template.
6. **Re-baseline.** Phases 8 and 9 rerun under the new policy, reported
   alongside the old, with the model difference stated at the point of use.

---

## 6. Effort and sequencing

| # | workstream | code | tests | risk |
|---|---|---|---|---|
| 1 | **Falsifier spike** (A minimal + B gated, no metrics) | ~150 LOC throwaway | none | **low — do this first** |
| 2 | A — demote hook, productionised | ~120 LOC / 4 files | ~10 | low |
| 3 | B — `save_policy="on_evict"` | ~40 LOC | ~6 | low (gated) |
| 4 | C3 — request-private decode keys | ~50 LOC | ~6 | **medium — modelling decision** |
| 5 | D — metrics reconciliation | ~80 LOC | ~10 | **high — semantics, not code** |
| 6 | Re-baseline Phases 8–9 | — | — | medium (~30 runs, ~1 h wall) |

**Roughly 450 LOC and ~35 tests**, dominated by (5) and by the re-baselining
rather than by the demote hook itself. The sequencing matters more than the
total: **step 1 is a day and can kill the other five.**

---

## 7. The honest case against doing it

Three reasons to weigh before starting:

1. **It does not change the thesis result.** The PCM argument rests on the
   write-policy comparison, which is internal to the store and survives this
   change. What the change buys is a *deployment rule*, not the core claim.
2. **The batch-dependent write stream may be small.** Under (A), only blocks
   that HBM evicts *under pressure* reach the store. Phase 9 measured
   `preemption_count = 0` and `recomputation_count = 0` even at 99.2% HBM
   occupancy — the scheduler runs right up to the ceiling without thrashing. If
   cached-block turnover is likewise modest, the resulting write stream may be
   far smaller than the current 16.9 TB, and the interesting regime may sit
   above what the trace can drive. **Step 1 of §5 measures exactly this.**
3. **There is a cheaper experiment that answers most of the question.** Phase 9
   established that the write stream is set by *pool capacity against workload
   footprint*. Sweeping `memory_capacity_blocks` — a knob, no code change —
   produces a curve of HBF lifetime versus PCM capacity, which is arguably the
   more useful deployment rule for this thesis anyway ("how much PCM do you need
   before HBF wear becomes acceptable"). **That sweep is ~10 runs and half a
   day.**

**Recommendation: run the `memory_capacity_blocks` sweep first, then the §5
step-1 falsifier, and only commit to workstreams 2–6 if the falsifier shows real
batch dependence.**
