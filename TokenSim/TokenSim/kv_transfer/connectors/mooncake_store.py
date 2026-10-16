from __future__ import annotations

import math
from typing import Any

from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import (
    PoolKey,
    Segment,
    build_key_metadata,
    pool_keys_for_request,
    prefix_key_digest,
)
from TokenSim.mooncake.service import get_mooncake_service

from .base import BaseKVConnector
from .metadata import (
    ConnectorTransferPlan,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


class MooncakeStoreConnector(BaseKVConnector):
    name = "MooncakeStoreConnector"

    def __init__(
        self,
        config: KVTransferConfig,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__(config, cache_config)
        self.block_bytes = 1
        self.block_size = 1
        self.model_name = "unknown"
        self.rank_info = None
        if cache_config is not None:
            self.block_size = int(cache_config.block_size)
            self.block_bytes = int(
                cache_config.block_size * cache_config.size_per_token
            )
            self.model_name = cache_config.model
            self.rank_info = getattr(cache_config, "rank_info", None)
        self.service = get_mooncake_service(config, block_bytes=self.block_bytes)
        self._check_partition_ranks(cache_config)
        self.mooncake_stats = self.service.store.stats
        # "mooncake" mirrors the reference vLLM MooncakeStoreConnector
        # (ref/vllm .../kv_connector/v1/mooncake/store/); "every_step" keeps
        # the historical TokenSim behavior.
        self._aligned = self.service.config.save_policy == "mooncake"
        self._pending_loads: dict[int, dict[str, Any]] = {}
        self._request_keys: dict[int, list[PoolKey]] = {}
        # Per-request key chain and its string form, both computed exactly once.
        self._keys_cache: dict[int, tuple[list[PoolKey], list[str]]] = {}
        self._saved_request_ids: set[int] = set()
        self._pending_async_puts: set[int] = set()
        self._delayed_releases: dict[int, tuple[Request, list[int]]] = {}
        self._block_releaser: Any | None = None
        # HBM-eviction hook. "prefill" (stock) leaves the observer None so the
        # prefix cache never calls in and no counter moves.
        self._hbm_evict_policy = self.service.config.hbm_evict_save_policy
        self._hbm_evict_metadata: Any | None = None
        self._hbm_evict_seq = 0
        self._pending_evict_latency = 0.0
        if self._hbm_evict_policy != "prefill":
            self.mooncake_stats.hbm_evict_reporting = 1
        self._kv_cache_manager = None
        self._hbm_first_registered: dict[Any, float] = {}
        self._hbm_spilled: set[Any] = set()
        self._run_stats_finalized = False
        self._count_save_skips = bool(self.service.config.count_aligned_save_skips)
        if self._count_save_skips:
            self.mooncake_stats.save_skip_reporting = 1

    def get_num_new_matched_tokens(
        self,
        req: Request,
        num_computed_tokens: int,
    ) -> int:
        keys = self._keys_for_request(req)
        self._request_keys[req.id] = keys
        if not keys:
            return 0
        local_gpu_tokens = max(0, int(num_computed_tokens))
        if local_gpu_tokens:
            self.mooncake_stats.local_gpu_hit_tokens += local_gpu_tokens
        if self.service.config.store_lookup_start == "after_hbm":
            looked_up = self._lookup_after_hbm(req, keys, local_gpu_tokens)
            if looked_up is None:
                return 0
            hit, hit_keys, external_tokens = looked_up
            span_start = 0  # hit_keys already begins at the first missing block
        else:
            hit = self.service.store.lookup(keys, now=self.simulation_time)
            if hit.hit_blocks <= 0:
                return 0
            hit_keys = list(hit.keys)
            hit_tokens = hit.hit_blocks * self.block_size
            if self._aligned and hit_tokens >= req.prefill_len:
                # Reference behavior: on a full-prompt hit, leave the trailing
                # block uncomputed-from-cache so prefill still computes tokens.
                capped_blocks = max(0, (req.prefill_len - 1) // self.block_size)
                hit_keys = hit_keys[:capped_blocks]
                hit_tokens = capped_blocks * self.block_size
            external_tokens = max(0, hit_tokens - local_gpu_tokens)
            span_start = local_gpu_tokens // self.block_size
        if external_tokens <= 0:
            return 0
        tier = hit.tier
        if self.service.config.load_span == "missing":
            # Attribute the load to the slowest tier of the span actually loaded.
            tier = self._span_tier(
                hit_keys[span_start : span_start + external_tokens // self.block_size]
            )
        if tier == "ssd":
            self.mooncake_stats.mooncake_disk_hit_tokens += external_tokens
        else:
            self.mooncake_stats.mooncake_memory_hit_tokens += external_tokens
        self._pending_loads[req.id] = {
            "keys": hit_keys,
            "tier": tier,
            "tokens": external_tokens,
            "blocks": external_tokens // self.block_size,
            "span_start": span_start,
        }
        return external_tokens

    def _span_tier(self, keys: list[PoolKey]) -> str | None:
        tier = None
        for key in keys:
            obj = self.service.store.objects.get(key.to_string())
            if obj is None:
                continue
            if obj.tier == "ssd":
                return "ssd"
            tier = "memory"
        return tier

    def _lookup_after_hbm(self, req: Request, keys: list[PoolKey], local_gpu_tokens: int):
        """store_lookup_start = "after_hbm": look up only beyond the HBM hit.

        Starts at the first block not already hit in HBM, with that block's key,
        and stops at the first store miss after it. Returns (hit, keys to load,
        external tokens), or None when nothing beyond the HBM hit is stored. No
        lookup -- and no hit/miss count -- when the HBM hit already covers every
        full prompt block.
        """
        start = local_gpu_tokens // self.block_size
        tail = keys[start:]
        if not tail:
            return None
        hit = self.service.store.lookup(tail, now=self.simulation_time)
        if hit.hit_blocks <= 0:
            return None
        hit_keys = list(hit.keys)
        if self._aligned and (start + len(hit_keys)) * self.block_size >= req.prefill_len:
            # Same reference cap as the "zero" path, on the absolute span.
            capped_blocks = max(0, (req.prefill_len - 1) // self.block_size)
            hit_keys = hit_keys[: max(0, capped_blocks - start)]
        return hit, hit_keys, len(hit_keys) * self.block_size

    def attach_kv_cache_manager(self, manager) -> None:
        """Called by the scheduler with its GPU prefix cache.

        Kept only when the HBM-eviction instrumentation is on; under the stock
        policy nothing is stored and no hook is installed.
        """
        if self._hbm_evict_policy == "prefill":
            return
        self._kv_cache_manager = manager
        manager.set_register_observer(self._on_hbm_register)

    def _on_hbm_register(self, block: Any, key: Any) -> None:
        self._hbm_first_registered.setdefault(key, self.simulation_time)

    def finalize_run_stats(self) -> None:
        """End-of-run counters; called once per connector before export."""
        if self._run_stats_finalized or self._kv_cache_manager is None:
            return
        self._run_stats_finalized = True
        stats = self.mooncake_stats
        keys = self._kv_cache_manager.cached_keys()
        stats.hbm_evict_residue_blocks += len(keys)
        if keys:
            metadata = self._hbm_evict_key_metadata()
            pool_keys = [PoolKey(metadata, prefix_key_digest(key)) for key in keys]
            stats.hbm_evict_residue_absent_blocks += len(
                self.service.store.missing_indices(pool_keys)
            )
        delays = sorted(self.service.hbm_first_spill_delays)
        stats.hbm_evict_first_spill_count = len(delays)
        if delays:
            stats.hbm_evict_first_spill_p50_s = _percentile(delays, 0.5)
            stats.hbm_evict_first_spill_p90_s = _percentile(delays, 0.9)

    def set_block_releaser(self, releaser) -> None:
        self._block_releaser = releaser

    @property
    def hbm_evict_observer(self):
        """Hook for KVCacheManager.evict, or None under the stock policy."""
        if self._hbm_evict_policy == "prefill":
            return None
        return self._on_hbm_evict

    def _on_hbm_evict(self, block: Any, reason: str) -> None:
        """One block is leaving the GPU prefix cache.

        Counts it under every policy but "prefill"; additionally writes it back
        into the store under "on_evict", through the same
        ``store.put_with_timing`` the prefill save uses, so the eviction cascade
        and ``charge_eviction_writes`` behave identically to any other write.

        Every block reaching the capacity site carries a key: ``block.cached``
        is set only by register_blocks, and BlockAllocator.allocate consults the
        prefix cache only for cached blocks. Decode blocks are never cached, so
        this hook sees prompt blocks and nothing else.
        """
        stats = self.mooncake_stats
        stats.hbm_evict_calls += 1
        if reason != "capacity":
            stats.hbm_evict_rekey_calls += 1
            return
        stats.hbm_evict_blocks += 1
        stats.hbm_evict_bytes += self.block_bytes

        prefix_key = getattr(block, "block_hash", None)
        if prefix_key is None:
            # Unreachable for the reason in the docstring; counted rather than
            # handled, so that a future scheduler change that breaks the
            # invariant shows up as a non-zero number instead of a wrong key.
            stats.hbm_evict_unkeyed_blocks += 1
            return
        stats.hbm_evict_prompt_blocks += 1
        stats.hbm_evict_keyed_blocks += 1
        key = PoolKey(self._hbm_evict_key_metadata(), prefix_key_digest(prefix_key))

        # missing_indices is the save-side existence check and touches no
        # hit/miss statistics or LRU heat, so counting here does not perturb
        # the run the way a lookup() would.
        present = not self.service.store.missing_indices([key])
        if present:
            stats.hbm_evict_present_blocks += 1
        else:
            stats.hbm_evict_absent_blocks += 1

        if prefix_key not in self._hbm_spilled and prefix_key in self._hbm_first_registered:
            self._hbm_spilled.add(prefix_key)
            self.service.hbm_first_spill_delays.append(
                self.simulation_time - self._hbm_first_registered[prefix_key]
            )
        if self._hbm_evict_policy not in ("on_evict", "spill_only"):
            return
        # Bytes are read off the store's own counters rather than assumed, so
        # the figure includes any cascade eviction this put forced.
        before = stats.memory_write_bytes + stats.ssd_write_bytes
        timing = self.service.store.put_with_timing(
            [key],
            blocks=1,
            now=self.simulation_time,
        )
        written = stats.memory_write_bytes + stats.ssd_write_bytes - before
        stats.hbm_evict_puts += 1
        stats.hbm_evict_write_bytes += written
        if written:
            stats.hbm_evict_puts_wrote_bytes += 1
        else:
            stats.hbm_evict_puts_refresh_only += 1
        stats.hbm_evict_save_latency += timing.blocking_latency
        self._pending_evict_latency += timing.blocking_latency

    def _hbm_evict_key_metadata(self):
        if self._hbm_evict_metadata is None:
            self._hbm_evict_metadata = build_key_metadata(
                model_name=self.model_name,
                rank_info=self.rank_info,
                engine_id=self.config.engine_id or "default",
                group_id=int(self.service.config.group_id),
                pcp_rank=int(self.service.config.pcp_rank),
                dcp_rank=int(self.service.config.dcp_rank),
            )
        return self._hbm_evict_metadata

    def _check_partition_ranks(self, cache_config: CacheConfig | None) -> None:
        """per_rank partitions must match the number of ranks that write.

        Only TP-rank-0 / PP-rank-0 workers are scheduled when world_size > 1
        (LLMEngine._is_schedulable_worker), so the store is written by one rank
        per data-parallel group: data_parallel_size writers, not world_size.
        """
        mooncake_config = self.service.config
        if mooncake_config.store_partition != "per_rank":
            return
        parallel_config = getattr(cache_config, "parallel_config", None)
        if parallel_config is None:
            return
        world_size = getattr(parallel_config, "world_size", 1)
        writers = 1 if world_size == 1 else getattr(parallel_config, "data_parallel_size", None)
        if writers is not None and writers != mooncake_config.store_partition_ranks:
            raise ConfigurationError(
                "store_partition_ranks="
                + f"{mooncake_config.store_partition_ranks} does not match the "
                + f"{writers} store-writing ranks (one TP-0/PP-0 worker per "
                + "data-parallel group)"
            )

    def _count_save_skip(self, req: Request) -> None:
        """Record one save skipped because this step loaded from the store.

        missing_indices touches no hit/miss statistics and no LRU heat, so
        counting does not perturb the run.
        """
        keys = self._keys_for_request(req)
        full_blocks = min(len(keys), req.prefill_len // self.block_size)
        missing = self.service.store.missing_indices(keys[:full_blocks])
        self.mooncake_stats.save_skip_requests += 1
        self.mooncake_stats.save_skip_missing_blocks += len(missing)

    def _drain_evict_latency(self) -> float:
        """Media time owed for this step's eviction saves.

        Charged as a store_save so it lands in save_wait_time with the rest of
        the write stalls. No bytes are added to transferred_bytes: an eviction
        save moves no data across the interconnect -- the block is already on
        this worker -- and its media bytes are counted by the store itself.
        """
        pending = self._pending_evict_latency
        self._pending_evict_latency = 0.0
        if pending:
            self.mooncake_stats.record_transfer(
                bytes_=0,
                latency=pending,
                kind="store_save",
                blocking_latency=pending,
            )
        return pending

    def update_state_after_alloc(
        self,
        req: Request,
        blocks: list[Any],
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens:
            req.cached_prefill_tokens = max(
                req.cached_prefill_tokens,
                num_external_tokens,
            )
            req.cached_prefill_blocks = max(
                req.cached_prefill_blocks,
                num_external_tokens // self.block_size,
            )
            req.effective_prefill_tokens = max(
                0,
                req.prefill_len - req.cached_prefill_tokens,
            )
            req.reuse_hit_blocks = max(req.reuse_hit_blocks, req.cached_prefill_blocks)
            req.reuse_miss_blocks = max(
                0,
                req.prefill_len // self.block_size - req.reuse_hit_blocks,
            )

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        loads: list[ConnectorTransferPlan] = []
        saves: list[ConnectorTransferPlan] = []
        can_save = self.config.kv_role in {"kv_producer", "kv_both", None}
        for req in getattr(scheduler_output, "scheduled", []) or []:
            load_info = self._pending_loads.pop(req.id, None)
            if load_info is not None:
                loads.append(self._build_load_plan(req, load_info))
            if not can_save:
                continue
            if self._hbm_evict_policy == "spill_only":
                # No prefill/admission save: blocks reach the store only
                # through the HBM-eviction hook (_on_hbm_evict).
                continue
            if self._aligned:
                # Reference semantics: saves happen only within the prefill
                # range (decode steps never save), each request saves at most
                # once, and a step that loads from the store skips its save.
                if not req.is_prefill or req.id in self._saved_request_ids:
                    continue
                self._saved_request_ids.add(req.id)
                if load_info is not None:
                    if self._count_save_skips:
                        self._count_save_skip(req)
                    continue
            save_keys = self._keys_for_request(req)
            if not save_keys:
                continue
            plan = self._build_save_plan(req, save_keys)
            if plan is None:
                continue
            saves.append(plan)
            if plan.async_transfer:
                self._pending_async_puts.update(plan.request_ids)
        preempted = [req.id for req in getattr(scheduler_output, "preempted", []) or []]
        return KVConnectorMetadata(
            connector_name=self.name,
            loads=loads,
            saves=saves,
            preempted_request_ids=preempted,
        )

    def start_load_kv(self) -> float:
        latency = 0.0
        for plan in self._own_plans(self._metadata.loads):
            keys = self._pool_keys_from_plan(plan)
            disk_latency = self.service.store.get(
                keys,
                plan.tier,
                now=self.simulation_time,
            )
            latency += plan.latency + disk_latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=plan.latency + disk_latency,
                kind="load",
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=plan.latency + disk_latency,
                kind="store_load",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.service.config.transfer_overlap
                    else plan.latency + disk_latency
                ),
            )
            self.mooncake_stats.record_pool_keys(plan.keys)
        if self.service.config.load_async and self.service.config.transfer_overlap:
            self.mooncake_stats.pending_async_jobs += len(self._metadata.loads)
            return 0.0
        return latency

    def wait_for_save(self) -> float:
        evict_latency = self._drain_evict_latency()
        latency = evict_latency
        for plan in self._own_plans(self._metadata.saves):
            keys = self._pool_keys_from_plan(plan)
            store_timing = self.service.store.put_with_timing(
                keys,
                blocks=plan.blocks,
                now=self.simulation_time,
                initial_delay=plan.latency,
            )
            total_latency = plan.latency + store_timing.accounting_latency
            blocking_latency = plan.latency + store_timing.blocking_latency
            latency += blocking_latency
            self.stats.record_transfer(
                blocks=plan.blocks,
                bytes_=plan.bytes,
                latency=total_latency,
                kind="save",
                blocking_latency=blocking_latency,
            )
            self.mooncake_stats.record_transfer(
                bytes_=plan.bytes,
                latency=total_latency,
                kind="store_save",
                blocking_latency=(
                    0.0
                    if plan.async_transfer and self.service.config.transfer_overlap
                    else blocking_latency
                ),
            )
            self.mooncake_stats.record_pool_keys(plan.keys)
        if self.service.config.load_async and self.service.config.transfer_overlap:
            self.mooncake_stats.pending_async_jobs += len(self._metadata.saves)
            # Eviction saves are not part of the async plan, so their time is
            # still owed even when the plan saves overlap.
            return evict_latency
        return latency

    def request_finished(
        self,
        req: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if req.id in self._pending_async_puts:
            self._delayed_releases[req.id] = (req, list(block_ids))
            return True, {"pending_async_put": True}
        self._forget_request(req.id)
        return False, None

    def get_finished(
        self,
        finished_req_ids: set[int],
    ) -> tuple[set[int], set[int]]:
        finished_sending = set(finished_req_ids) & self._pending_async_puts
        finished_recving = set(finished_req_ids) & self._pending_async_puts
        return finished_sending, finished_recving

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        meta = super().build_connector_worker_meta()
        if self._pending_async_puts:
            meta.finished_sending.update(self._pending_async_puts)
            meta.finished_recving.update(self._pending_async_puts)
        return meta

    def update_connector_output(
        self,
        worker_output: KVConnectorWorkerMetadata,
    ) -> None:
        releasable = set(worker_output.finished_sending)
        self._pending_async_puts.difference_update(worker_output.finished_sending)
        self.mooncake_stats.pending_async_jobs = max(
            0,
            self.mooncake_stats.pending_async_jobs
            - len(worker_output.finished_sending),
        )
        for req_id in releasable:
            delayed = self._delayed_releases.pop(req_id, None)
            if delayed is None or self._block_releaser is None:
                self._forget_request(req_id)
                continue
            req, block_ids = delayed
            self._block_releaser(req, block_ids)
            self._forget_request(req_id)
        super().update_connector_output(worker_output)

    def handle_preemptions(self, meta: KVConnectorMetadata) -> None:
        super().handle_preemptions(meta)
        # Recomputed requests rebuild connector state when they are admitted again.
        for req_id in meta.preempted_request_ids:
            self._forget_request(req_id)

    def on_request_released(self, req: Request) -> None:
        if req.id in self._pending_async_puts:
            # An async save is still in flight; the delayed-release path in
            # update_connector_output() cleans up once it completes.
            return
        self._forget_request(req.id)

    def _forget_request(self, req_id: int) -> None:
        self._keys_cache.pop(req_id, None)
        self._request_keys.pop(req_id, None)
        self._pending_loads.pop(req_id, None)
        self._saved_request_ids.discard(req_id)

    def _build_load_plan(
        self,
        req: Request,
        load_info: dict[str, Any],
    ) -> ConnectorTransferPlan:
        blocks = int(load_info["blocks"])
        bytes_ = blocks * self.block_bytes
        tier = load_info.get("tier") or "memory"
        transfer = self.service.transfer_engine.create_transfer(
            source_segment=Segment(
                name=f"store-{tier}", tier="ssd" if tier == "ssd" else "dram"
            ),
            target_segment=Segment(name=f"worker-{self.config.kv_rank}", tier="vram"),
            blocks=blocks,
            bytes_=bytes_,
            protocol="nvmeof" if tier == "ssd" else self.service.config.protocol,
            kind="store",
            topology="cross_node",
        )
        if self.service.config.load_span == "missing":
            # Only the span not already hit in HBM.
            start = int(load_info.get("span_start", 0))
            pool_keys = list(load_info["keys"][start : start + blocks])
            self._request_keys[req.id] = pool_keys
            keys = [key.to_string() for key in pool_keys]
        else:
            pool_keys = list(load_info["keys"][:blocks])
            self._request_keys[req.id] = pool_keys
            keys = self._key_strings_for_request(req)[: len(pool_keys)]
        return ConnectorTransferPlan(
            request_ids=[req.id],
            request_block_counts={req.id: blocks},
            target_worker_id=self.config.kv_rank,
            blocks=blocks,
            bytes=bytes_,
            latency=transfer.latency,
            kind="load",
            keys=keys,
            tier=tier,
            async_transfer=self.service.config.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
            },
        )

    def _build_save_plan(
        self, req: Request, keys: list[PoolKey]
    ) -> ConnectorTransferPlan | None:
        full_blocks = min(len(keys), req.prefill_len // self.block_size)
        pool_keys = keys[:full_blocks]
        key_strings = self._key_strings_for_request(req)[:full_blocks]
        if self._aligned:
            # Reference semantics: an existence prefilter runs before the put
            # (worker-side batch_is_exist) so only missing keys are written
            # and transferred.
            missing = self.service.store.missing_indices(pool_keys)
            if not missing:
                self._request_keys[req.id] = pool_keys
                return None
            pool_keys = [pool_keys[i] for i in missing]
            key_strings = [key_strings[i] for i in missing]
        self._request_keys[req.id] = pool_keys
        blocks = len(pool_keys)
        bytes_ = blocks * self.block_bytes
        transfer = self.service.transfer_engine.create_transfer(
            source_segment=Segment(name=f"worker-{self.config.kv_rank}", tier="vram"),
            target_segment=Segment(name="store-memory", tier="dram"),
            blocks=blocks,
            bytes_=bytes_,
            kind="store",
            topology="cross_node",
        )
        return ConnectorTransferPlan(
            request_ids=[req.id],
            request_block_counts={req.id: blocks},
            source_worker_id=self.config.kv_rank,
            blocks=blocks,
            bytes=bytes_,
            latency=transfer.latency,
            kind="save",
            keys=key_strings,
            tier="memory",
            async_transfer=self.service.config.load_async,
            extra={
                "protocol": transfer.protocol,
                "connector": self.name,
                "pool_keys": pool_keys,
            },
        )

    def _keys_for_request(self, req: Request) -> list[PoolKey]:
        return self._request_key_bundle(req)[0]

    def _key_strings_for_request(self, req: Request) -> list[str]:
        return self._request_key_bundle(req)[1]

    def _request_key_bundle(self, req: Request) -> tuple[list[PoolKey], list[str]]:
        bundle = self._keys_cache.get(req.id)
        if bundle is None:
            keys = pool_keys_for_request(
                req,
                model_name=self.model_name,
                rank_info=self.rank_info,
                engine_id=self.config.engine_id or "default",
                group_id=int(self.service.config.group_id),
                pcp_rank=int(self.service.config.pcp_rank),
                dcp_rank=int(self.service.config.dcp_rank),
            )
            bundle = (keys, [key.to_string() for key in keys])
            self._keys_cache[req.id] = bundle
        return bundle

    def _pool_keys_from_plan(self, plan: ConnectorTransferPlan) -> list[PoolKey]:
        pool_keys = plan.extra.get("pool_keys")
        if pool_keys is not None:
            return list(pool_keys)
        # Fallback for plans built elsewhere; only live requests are indexed.
        key_lookup = {
            key.to_string(): key for keys in self._request_keys.values() for key in keys
        }
        result = []
        for key_string in plan.keys:
            key = key_lookup.get(key_string)
            if key is not None:
                result.append(key)
        return result

    def _own_plans(
        self, plans: list[ConnectorTransferPlan]
    ) -> list[ConnectorTransferPlan]:
        return [
            plan
            for plan in plans
            if plan.extra.get("connector") == self.name
            or plan.tier in {"memory", "ssd"}
        ]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[index])
