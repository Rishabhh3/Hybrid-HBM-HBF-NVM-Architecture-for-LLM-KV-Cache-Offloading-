# Option 2 — storing generated KV: design note

**Status: scope only. Nothing here is implemented.** Written for review after
PHASE 12; the numbers it leans on are in `findings.md` PHASE 12 and its addendum.

Line numbers are current as of `0244f13` on `falsifier-on-evict`.

---

## 0. What this has to solve, and the one measurement that reshapes it

PHASE 12 established that HBM eviction is not a channel to the far tier, and
that the store has only ever held **prompt** blocks. Option 2 is the remaining
way to put transient KV into the write stream: **store a finished request's
generated blocks**, so that a later continuation can read them back.

The brief's key scheme is: for `R` with continuation `R'`, give `R`'s output
blocks the 16-token sub-ids of `R'` at the positions after `R`'s prompt. That is
the right idea, but it does not work as stated, for a reason measured in
`findings.md` PHASE 12 addendum STEP 2:

> **For 97.9% (Conversation) and 98.7% (Tool&Agent) of continuation pairs, the
> two key chains diverge *before* `R`'s last full prompt block — by a median of
> 22 and 14 blocks, up to 32.**

Why: the FAST'25 trace hashes at **512-token** granularity and the simulator
blocks at **16**. Writing `s = 32·(len(R.raw_hash_ids) − 1)` for the sub-index
where `R` and `R'` first differ, and `p = R.prefill_len // 16` for `R`'s full
prompt-block count, the match rule (which excludes `R`'s last raw id, because it
covers a partial chunk that grows) gives agreement only on `[0, s)`. And
`s < p` almost always, because `R`'s final 512-token chunk is partial.

Both lookups stop at the **first** miss — `plan_reuse` breaks out of its loop
(`TokenSim/block/kv_cache_manager.py:58-64`) and `MooncakeStore.lookup` breaks
too (`TokenSim/mooncake/store.py:79-93`). So `R'` would stall at index `s` and
**never reach the output blocks at `p` and beyond**, however perfectly they are
keyed. Option 2 as specified would measure a write cost and return zero reads.

This is a **trace-granularity artefact, not a property of real serving.** A real
prefix cache keys 16-token blocks by token content, so `R`'s prompt blocks in
`[s, p)` — whose tokens are identical to `R'`'s at those positions, under the
assumption in §1 — would match automatically. §3 gives two ways to handle it.

---

## 1. The assumption, stated once

Everything below rests on: **`R'`'s prompt begins with `R`'s prompt, and (where
`delta ≥ R.output_length`) continues with `R`'s output.**

The traces cannot show this. They carry lengths and chunk hashes, no token
content. The addendum measures what the traces *can* say, and it is not
uniformly supportive: `delta < R.output_length` for **12.29%** (Conversation)
and **22.81%** (Tool&Agent) of continuations, and `delta < 0` for **2.15%** and
**14.41%**. Any run built on this design inherits that assumption and must say
so at the point the results are reported.

---

## 2. (a) The hook point

### Where

**`BlockManager.free` (`TokenSim/block/block_manager.py:155-157`)**, between
`commit_finished_cache` and `release_request_blocks`:

```
155    def free(self, req: Request):
156        self.commit_finished_cache(req)
157 +      self.save_output_blocks(req)          # new, gated
158        self.release_request_blocks(req)
```

### Why not the other two candidates

- **Not `KVCacheManager.release_block` (`kv_cache_manager.py:143-156`).** It
  fires per block on every release, including preemption
  (`llm_scheduler.py:~325` → `release_request_blocks`). A preempted request is
  recomputed and released again, so its output blocks would be written twice.
  It also cannot see the request, so it cannot tell a prompt block from an
  output one.
- **Not `BlockManager.release_request_blocks` (`block_manager.py:166-171`)
  directly.** It has three callers and only two of them mean "finished":
  `BlockManager.free` and `LLMPagedAttnScheduler._release_delayed_blocks`
  (`llm_scheduler.py:371-373`). The third is `_preempt`. Hooking `free` gets the
  finish/preempt distinction for free.

`_release_delayed_blocks` bypasses `BlockManager.free`, so it needs the same
call. That path only runs when `load_async` is set; configs (b)/(c)/(d) all have
`load_async: false`, so it is dead in the run matrix — wire it anyway rather
than leave a silent hole.

### Order matters

The hook must run **after** `commit_finished_cache` (so the prompt blocks are
registered and their keys settled) and **before** `release_request_blocks` (which
pops the block table at `:167` and clears `req._physical_token_blocks` at `:170`).
After `:168` the blocks are gone.

### Which blocks

`self.block_table.get_blocks(req.id)`, indices `p .. (prefill_len+decode_len)//16 − 1`,
where `p = req.prefill_len // self.block_size`. This mirrors the truncation
`register_output_blocks` already uses (`kv_cache_manager.py:114-122`) and the one
in `_build_save_plan` (`mooncake_store.py:~341`).

Note the brief's cap, `ceil(R.output_length/16)`, and the count of blocks that
actually become full, `(prefill+decode)//16 − prefill//16`, **differ by one**
whenever `prefill % 16 + decode % 16 ≥ 16`. Use the full-block count; it is the
physically real one. Flag the discrepancy where results are reported.

### Wiring, mirroring PHASE 12

`BlockManager` gets `set_output_save_observer(observer)`; the scheduler installs
`getattr(self.connector, "output_save_observer", None)` beside the existing
`hbm_evict_observer` line (`llm_scheduler.py:182-187`). The connector returns
`None` under the stock policy, so nothing is installed and no counter moves.

### The flag

`output_save_policy` on `MooncakeConfig` (`TokenSim/mooncake/config.py`), beside
`hbm_evict_save_policy`:

| value | meaning |
|---|---|
| **`none`** | **default, stock.** No observer; no counter; `register_output_blocks` stays gated off (see §4). |
| `on_release` | a finished request's full output blocks are put into the store. |

---

## 3. (b) The key scheme

### 3.1 Continued requests — the divergent tail

Two options. **Option A is recommended.**

**Option A — fix the sub-id expansion in `build_trace.py`.** Make the trailing
partial chunk's sub-ids shared as far as the common token count goes, instead of
deriving them all from the chunk id. For request `R` with raw ids `h` and
`c = len(h) − 1` (index of the trailing, partial chunk), the tokens in that chunk
that `R` and any continuation share are exactly the first
`R.prefill_len − 512·c`. So emit, for sub-index `i` in that chunk,
a sub-id derived from **`(h[:c], i)`** rather than from `h[c]`, for
`i < (R.prefill_len − 512·c) // 16`, and from `h[c]` beyond that.

This makes the two chains agree on `[0, p)` and the lookup reaches `p`. It is a
**build-time** change; the simulator is untouched. It is also the change that
makes the trace behave the way real content-addressed 16-token blocks would, so
it removes an artefact rather than adding an oracle. Cost: the existing
`conv_5000.jsonl` / `toolagent_5000.jsonl` are superseded, so the PHASE 9
reproduction gate must be re-run against the new files and **is expected to
move** — prompt-side prefix hits will rise. That is a real change of baseline and
must be measured and reported before any option-2 number is quoted.

**Option B — re-key the tail at save time.** Leave the trace alone and have the
hook also write `R`'s prompt blocks in `[s, p)` a second time, under `R'`-chain
keys. It keeps the PHASE 9 baseline intact, but it writes a **median 22
(Conversation) / 14 (Tool&Agent) extra blocks per continued request** — against
a median output of 22 / 2 blocks. On Tool&Agent that is **~7× the output itself**,
so the measured write cost of option 2 would be dominated by an artefact. If
Option B is taken anyway, those blocks must be counted as a **third population**
(§5), never folded into the output-block totals.

### 3.2 Continued requests — the keys themselves

With the chains agreeing on `[0, p)`, `R`'s output block at index `p + j` takes
**`R'`'s key at the same index**:

```
build_prefix_keys(R'.hash_ids[: p + j + 1], model, cache_salt, reuse_group)[p + j]
```

then wrapped by `prefix_key_digest` and `build_key_metadata`
(`TokenSim/mooncake/pool_key.py:109`, `:47`) exactly as `pool_keys_for_request`
does (`:76-107`). Because `R'.hash_ids[:p] == R.hash_ids[:p]` under Option A,
this is the key `R'` will look up — so the hit is by construction, not by luck.

Count: `min(ceil(R.output_length/16), len(R'.hash_ids) − p)`, which covers `R`'s
**whole** output for **91.4%** (Conversation) and **81.6%** (Tool&Agent) of
continued requests; the mean coverage is 0.946 and 0.844.

### 3.3 The partial boundary block

`R.prefill_len % 16 ≠ 0` for **93.0% / 94.0%** of continued requests, so block
`p` is mixed: prompt tail plus the first generated tokens. Three facts settle
how to treat it:

1. In `R'` those same token positions are **all prompt**, so `R'`'s key at index
   `p` is the correct key for that content. Including it is exact, not
   approximate.
2. It must be included. Skipping it leaves a hole at index `p`, and since both
   lookups stop at the first miss, everything after it becomes unreachable — the
   scheme would yield zero hits.
3. It is only valid once **full**, which needs `R` to generate at least
   `(−R.prefill_len) mod 16` tokens. True for **96.8%** (Conversation) but only
   **78.4%** (Tool&Agent), whose outputs are short (median 30 tokens).

So: include block `p` when it is full; when it is not, store nothing for that
request and record it. Tool&Agent's **21.6%** shortfall is not a rounding
detail — it is a fifth of its continued requests contributing write traffic that
can never be read, and it should be its own counter.

### 3.4 Requests with no continuation

Request-private keys, so the block is pure write traffic that nothing can match:

- rank half: `build_key_metadata(model_name=…, rank_info=…, engine_id=…, …)`
  (`pool_key.py:47-74`). `KeyMetadata.to_string` (`:31-44`) already emits
  `tp_rank`, `pp_rank`, `dp_rank`, `engine`, `kv_cache_group`, `pcp`, `dcp`, so
  the rank requirement is met by reusing it — no new spelling.
- digest half: a private string over `(engine_id, req.id, block_index)`, hashed
  with `_stable_hash` (`pool_key.py:124`) for shape consistency with real digests.

This is the upper bound the brief asks for: every such block is a write that buys
nothing, and `demote_policy=if_read` should drop all of it.

---

## 4. (c) Precomputing keys at trace build time

**The rule: the simulator must never learn at runtime whether a request has a
continuation.** If it did, `if_read` would be handed the answer it is supposed to
discover, and the PHASE 8/9 `if_read` results would stop meaning anything.

The mechanism already exists. `build_trace.py` emits **`output_hash_ids`** per
request — a field the loader already reads (`TokenSim/workload/loaders.py:215`,
and `:128` for the json_pairs path) and `Request` already carries
(`TokenSim/llm/llm_request.py:206`). The continuation search runs **offline**, in
`build_trace.py`, exactly where the prompt-side `hash_ids` are already decided.

Why this is not cheating: `hash_ids` is already an oracle in precisely the same
sense — it tells the simulator which prompt blocks will be shared, before they
are. `output_hash_ids` is the same kind of information about the same workload.
What it must not become is an oracle about *policy*: the store sees only keys,
`_drops_on_eviction` (`store.py:327-329`) still sees only `obj.read_count`, and
nothing anywhere branches on "will this be read".

Two consequences to handle:

1. **`register_output_blocks` (`kv_cache_manager.py:108-137`) fires as soon as
   `output_hash_ids` is present** — it is currently dormant only because the
   field is absent (`:109`). That path registers output blocks in the **GPU**
   prefix cache, which would raise GPU-side hits independently of the store and
   silently move the baseline. **Gate it on `output_save_policy` too**, so
   `none` on the new trace file reproduces PHASE 9 exactly and the three configs
   differ only in store policy.
2. **`pool_keys_for_request` truncates to the prompt**
   (`pool_key.py:88`, `full_input_blocks = req.prefill_len // req.block_size`).
   Under `on_release` it must be able to return the output keys too — as a
   separate call, not by widening the existing one, so the prefill save path is
   untouched.

A third, smaller point: `build_trace.py` must write `output_hash_ids` for **every**
request — real sub-ids for continued ones, private placeholders for the rest — or
the presence of the field would itself leak which requests have continuations.

---

## 5. (d) Metrics that separate the two populations

The population is a build-time property of the key, so it travels with it.

**Mechanism.** Add `origin` to `StoreObject` (`store.py:14-28`) and an `origin`
argument to `put_with_timing` / `_put_one` (`store.py:168`, `:191`), defaulting
to `"prompt"` so every existing call site and every stock number is unchanged.
Values: `prompt`, `output_continued`, `output_private` — and `prompt_rekey` if
§3.1 Option B is taken.

**Counters**, per population, gated so stock output stays byte-identical (the
PHASE 12 pattern in `metrics.py` — a `reporting` field plus a field-name tuple
omitted from `as_dict`):

| question | counter | site |
|---|---|---|
| **bytes written to PCM** | `output_memory_write_bytes_{continued,private}` | `_record_memory_write` (`store.py:352-362`) |
| **bytes reaching HBF** | `output_ssd_write_bytes_{continued,private}` | `_record_offload_write` (`store.py:393-400`) |
| **bytes dropped by `if_read`** | `output_dropped_bytes_{continued,private}` | the `_drops_on_eviction` branch of `_evict_memory_lru` (`store.py:296-307`) |
| bytes demoted (the complement) | `output_demoted_bytes_{continued,private}` | same method, `:308-320` |
| did it ever buy a read | `output_read_blocks_{continued,private}` | `lookup`'s `read_count` bump (`store.py:94-100`) |
| residency before eviction | `output_residency_sum_s_{continued,private}` | `record_memory_eviction` (`metrics.py:155`) |
| boundary block never filled | `output_skipped_unfilled_requests` | the hook |

**The numbers to report**, all work-normalized, never as day counts (the far tier
still has no shared-bandwidth model):

- PCM bytes/request and HBF bytes/request, split `continued` vs `private`;
- **`output_read_blocks / output_write_blocks` per population** — the useful-write
  fraction. The whole thesis claim is that `private` is ≈ 0 and `continued` is
  not, and that `if_read` therefore removes the first and keeps the second;
- bytes dropped by `if_read` per population, as a fraction of bytes written;
- mean residency per population against the continuation gaps already measured
  (Conversation p50 **72 s**, Tool&Agent p50 **0.0 s**, `findings.md` STEP C) —
  this is what decides whether a stored output block is still resident when its
  continuation arrives.

---

## 6. (e) Run matrix and test count

### Runs

The PHASE 9 batch-matched setting, unchanged: **cap 48, 30 QPS, W = 5,000**,
`H3-B200-KVONLY`, `LLaMa2-70B-GQA` TP2/DP4, `memory_capacity_blocks = 142213`,
`ssd_capacity_blocks = 10066329`, `charge_eviction_writes = ON`, `--random_seed 0`.

| config | memory media | admission write | demote |
|---|---|---|---|
| (b) | dram | write_through | always |
| (c) | pcm | write_back | always |
| (d) | pcm | write_back | if_read |

3 configs × 2 traces × 2 policies (`none`, `on_release`) = **12 runs**.

The 6 `none` runs are the **reproduction gate** and must be run first. Recorded
oracles at cap 48 exist for **(b) and (d) on both traces** (`findings.md`
~4128/4130); (c) at cap 48 is *not* recorded, so it gets a fresh baseline rather
than a check. If §3.1 Option A is taken the gate is expected to **move**, and
the move must be quantified and reported before any option-2 number is quoted —
if it is taken and the gate does *not* move, that is a bug, because prompt-side
hits should rise.

Add 2 runs if §3.1 Option B is chosen instead, to separate the `prompt_rekey`
population's cost from the output population's.

### Tests

Current: **177**. Estimated new: **~22**, giving **~199**.

| area | ~n | what |
|---|---|---|
| config flag | 3 | default `none`; all values parse; unknown rejected |
| hook placement | 4 | fires on finish; **not** on preempt; fires on the delayed-release path; runs before the block table is popped |
| block selection | 3 | output range only; full-block count vs `ceil(out/16)`; unfilled boundary block skipped and counted |
| key scheme | 5 | continued key equals `R'`'s key at the same index; private key is unique per (req, index) and carries tp/pp/dp; boundary block included when full |
| build-time keys | 3 | `output_hash_ids` emitted for every request; continued sub-ids align at `p + j`; Option A's tail expansion makes the chains agree on `[0, p)` |
| store origin | 3 | `origin` defaults to `prompt`; demote/drop counted per population; `if_read` drops `private` and keeps read `continued` |
| gating | 1 | under `none`: no observer, `register_output_blocks` dormant, no `output_*` key in `as_dict` |

Plus the standing rule: the stock arm must be **byte-identical** to the
pre-change commit on both traces, excluding `simulator_wall_time` and
`mooncake_pool_keys`.

---

## 7. What to decide before building

1. **§3.1 Option A or B.** A is cleaner and measures the real mechanism, but
   supersedes the trace files and moves the PHASE 9 baseline. B preserves the
   baseline but lets a 512-vs-16 granularity artefact dominate the write cost it
   is trying to measure — badly on Tool&Agent (~7× the output).
2. **Whether to gate `register_output_blocks`** (§4). Recommended yes; without
   it the GPU-side and store-side effects are confounded from the first run.
3. **Whether Tool&Agent's 21.6% unfillable boundary blocks** should be excluded
   from the continued population or reported inside it. Recommended: report
   inside, with its own counter, since excluding them would flatter the result.
