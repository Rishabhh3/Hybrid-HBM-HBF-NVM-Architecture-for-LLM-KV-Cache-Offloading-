from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from TokenSim.mooncake.config import MooncakeConfig
from TokenSim.mooncake.metrics import MooncakeStats
from TokenSim.mooncake.pool_key import PoolKey
from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.ssd import OffloadTier, PartitionedOffloadTier

_GB = 1 << 30


@dataclass
class StoreObject:
    key: str
    blocks: int
    bytes: int
    tier: str = "memory"
    segment: str = "global"
    replicas: int = 1
    persisted: bool = False
    last_access: int = 0
    # Lookups that hit this object since admission. Zero at memory-tier
    # eviction means every byte written for it was a wasted write.
    read_count: int = 0
    # Simulated time of the put that admitted this object, for residency.
    write_time: float = 0.0
    # Owning rank index (store_partition != "global" only).
    rank: int | None = None
    # Reads actually served from the memory tier (MooncakeStore.get), whatever
    # read_credit says; feeds the count_if_read_gate check only.
    pcm_served_reads: int = 0


@dataclass
class StoreHit:
    keys: list[PoolKey]
    hit_blocks: int
    tier: str | None

    @property
    def hit_tokens(self) -> int:
        if not self.keys:
            return 0
        block_size = getattr(self.keys[0], "_block_size", 0)
        return self.hit_blocks * block_size


@dataclass(frozen=True)
class StoreWriteTiming:
    accounting_latency: float = 0.0
    blocking_latency: float = 0.0

    def __add__(self, other: StoreWriteTiming) -> StoreWriteTiming:
        return StoreWriteTiming(
            accounting_latency=self.accounting_latency + other.accounting_latency,
            blocking_latency=self.blocking_latency + other.blocking_latency,
        )


class MooncakeStore:
    def __init__(self, config: MooncakeConfig, block_bytes: int):
        self.config = config
        self.block_bytes = max(1, int(block_bytes))
        if config.memory_capacity_blocks is not None:
            self.memory_capacity_bytes = config.memory_capacity_blocks * self.block_bytes
        else:
            self.memory_capacity_bytes = config.global_segment_size
        self.memory_used_bytes = 0
        self.objects: dict[str, StoreObject] = {}
        self.memory_lru: OrderedDict[str, None] = OrderedDict()
        self.stats = MooncakeStats()
        self._count_readback = bool(config.count_hbf_readback)
        self._credit_on_serve = config.read_credit == "served"
        if config.count_if_read_gate:
            self.stats.if_read_gate_reporting = 1
        # key -> True while its latest offload copy has not been read back.
        self._hbf_unread: dict[str, bool] = {}
        if self._count_readback:
            self.stats.hbf_readback_reporting = 1
        # count_pcm_admit_source: keys ever stored, and for keys that left the
        # store, why they left (the latest departure wins).
        self._count_admit_source = bool(config.count_pcm_admit_source)
        self._seen_keys: set[str] = set()
        self._departed: dict[str, str] = {}
        if self._count_admit_source:
            self.stats.pcm_admit_reporting = 1
        self._hbf_unlimited = config.hbf_capacity_mode == "unlimited"
        self._partitioned = config.store_partition != "global"
        self._per_rank = config.store_partition == "per_rank"
        if self._partitioned:
            self._init_partitions()
        if not config.enable_offload:
            self.offload = None
        elif self._per_rank:
            self.offload = PartitionedOffloadTier(
                config,
                self.block_bytes,
                config.store_partition_ranks,
                self._key_rank.get,
            )
            self.stats.partition_rank_ssd_capacity_blocks = (
                self.offload.partition_capacity_blocks
            )
        else:
            self.offload = OffloadTier(config, self.block_bytes)
        self.ssd = self.offload
        if self.offload is not None:
            self.stats.record_offload_tier(self.offload.tier)
            self.stats.record_offload_profile(self._offload_profile())
            if self._hbf_unlimited:
                self.stats.hbf_capacity_reporting = 1
                self.stats.hbf_capacity_configured_blocks = self.offload.capacity_blocks
        self._access_counter = 0

    def lookup(self, keys: list[PoolKey], *, now: float = 0.0) -> StoreHit:
        hit_keys: list[PoolKey] = []
        slowest_tier: str | None = None
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None:
                break
            if (
                self._is_offload_tier(obj.tier)
                and self.offload is not None
                and not self.offload.contains(obj.key)
            ):
                break
            hit_keys.append(key)
            if self._is_offload_tier(obj.tier):
                slowest_tier = obj.tier
            elif slowest_tier is None:
                slowest_tier = "memory"
        if hit_keys:
            # Per-object, not a global counter: the wasted-write classification
            # needs to know which objects were read, not how many reads ran.
            if not self._credit_on_serve:
                for key in hit_keys:
                    obj = self.objects.get(key.to_string())
                    if obj is not None:
                        obj.read_count += 1
            self.stats.store_hit_count += 1
            if self._is_offload_tier(slowest_tier):
                self.stats.disk_tier_hit_count += 1
            else:
                self.stats.memory_tier_hit_count += 1
        else:
            self.stats.store_miss_count += 1
        return StoreHit(keys=hit_keys, hit_blocks=len(hit_keys), tier=slowest_tier)

    def missing_indices(self, keys: list[PoolKey]) -> list[int]:
        """Indices of keys absent from the store.

        Save-side existence prefilter, mirroring the reference connector's
        ``batch_is_exist`` before ``batch_put``: checked per key (no prefix
        requirement) and without touching hit/miss statistics or LRU heat.
        """
        missing: list[int] = []
        for index, key in enumerate(keys):
            obj = self.objects.get(key.to_string())
            present = obj is not None and (
                not self._is_offload_tier(obj.tier)
                or self.offload is None
                or self.offload.contains(obj.key)
            )
            if not present:
                missing.append(index)
        return missing

    def get(
        self,
        keys: list[PoolKey],
        tier: str | None,
        *,
        now: float = 0.0,
    ) -> float:
        if not keys:
            return 0.0
        self.stats.get_count += 1
        latency = 0.0
        for key in keys:
            obj = self.objects.get(key.to_string())
            if obj is None:
                continue
            self._touch(obj)
            if self._credit_on_serve:
                obj.read_count += 1
            if self._is_offload_tier(obj.tier) and self.offload is not None:
                op = self.offload.read(obj.key)
                self._record_offload_read(op, key=obj.key)
                latency += op.latency
            else:
                obj.pcm_served_reads += 1
                latency += self._record_memory_read(obj)
        return latency

    def put(
        self,
        keys: list[PoolKey],
        blocks: int | None = None,
        *,
        now: float = 0.0,
        initial_delay: float = 0.0,
    ) -> float:
        return self.put_with_timing(
            keys,
            blocks,
            now=now,
            initial_delay=initial_delay,
        ).blocking_latency

    def put_with_timing(
        self,
        keys: list[PoolKey],
        blocks: int | None = None,
        *,
        now: float = 0.0,
        initial_delay: float = 0.0,
    ) -> StoreWriteTiming:
        if self.config.admission_policy == "never" or not keys:
            self.stats.admission_rejection_count += len(keys)
            return StoreWriteTiming()
        self.stats.put_count += 1
        timing = StoreWriteTiming()
        blocks = blocks if blocks is not None else len(keys)
        for key in keys[:blocks]:
            key_timing = self._put_one(
                key,
                now=now,
                initial_delay=initial_delay + timing.accounting_latency,
            )
            timing += key_timing
        return timing

    def _put_one(
        self,
        key: PoolKey,
        *,
        now: float,
        initial_delay: float,
    ) -> StoreWriteTiming:
        key_string = key.to_string()
        existing = self.objects.get(key_string)
        if existing is not None:
            self._touch(existing)
            return StoreWriteTiming()
        obj = StoreObject(
            key=key_string,
            blocks=1,
            bytes=self.block_bytes,
            replicas=self.config.replica_num,
            segment=self.config.preferred_segment or "global",
            write_time=now,
        )
        if self._partitioned:
            obj.rank = self._rank_of(key)
            if self._per_rank:
                self._key_rank[obj.key] = obj.rank
        admitted, evict_timing = self._ensure_memory_capacity(
            obj.bytes, now=now, rank=obj.rank
        )
        if not admitted:
            if self.config.enable_offload and self.offload is not None:
                obj.tier = self.offload.tier
                op = self.offload.write(obj.key, obj.blocks)
                if obj.key in op.evicted_keys:
                    self.stats.admission_rejection_count += 1
                    return evict_timing
                timing = self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay + evict_timing.accounting_latency,
                )
                self._record_offload_write(op, owner_key=obj.key)
                self._drop_offload_victims(op, owner_key=obj.key)
                self.objects[obj.key] = obj
                if self._count_admit_source:
                    self._seen_keys.add(obj.key)
                self.stats.admission_count += 1
                # Not a memory-tier admission: memory could not hold the block,
                # so it goes straight to the offload tier under either write
                # policy. Counted separately so the write identity stays exact.
                self.stats.memory_bypass_blocks += obj.blocks
                return evict_timing + timing
            self.stats.admission_rejection_count += 1
            return evict_timing
        self.objects[obj.key] = obj
        self.memory_used_bytes += obj.bytes
        self.memory_lru[obj.key] = None
        if self._partitioned:
            self._admit_to_rank(obj)
        if self._count_admit_source:
            self._record_admit_source(obj)
        self.stats.admission_count += 1
        media_latency = self._record_memory_write(obj)
        timing = StoreWriteTiming(
            accounting_latency=media_latency,
            blocking_latency=media_latency,
        )
        if (
            self.config.enable_offload
            and self.offload is not None
            and self.config.admission_write_policy == "write_through"
        ):
            op = self.offload.write(obj.key, obj.blocks)
            if obj.key not in op.evicted_keys:
                timing += self._write_timing(
                    obj,
                    op.latency,
                    now=now,
                    initial_delay=initial_delay + evict_timing.accounting_latency,
                )
                self._record_offload_write(op, owner_key=obj.key)
                self._drop_offload_victims(op, owner_key=obj.key)
        return evict_timing + timing

    def _ensure_memory_capacity(
        self,
        bytes_: int,
        *,
        now: float,
        rank: int | None = None,
    ) -> tuple[bool, StoreWriteTiming]:
        """Free room for ``bytes_``, returning admission plus eviction cost.

        The timing is non-zero only when ``charge_eviction_writes`` is set; it
        accumulates across every eviction this call had to perform, including
        the ones that ran before an unsuccessful admission.
        """
        timing = StoreWriteTiming()
        if self._per_rank:
            return self._ensure_rank_capacity(bytes_, now=now, rank=rank)
        if bytes_ > self.memory_capacity_bytes:
            return False, timing
        while self.memory_used_bytes + bytes_ > self.memory_capacity_bytes:
            if self.config.eviction_policy != "lru" or not self.memory_lru:
                return False, timing
            timing += self._evict_memory_lru(now=now, admitter=rank)
        return True, timing

    def _evict_memory_lru(
        self,
        *,
        now: float,
        admitter: int | None = None,
    ) -> StoreWriteTiming:
        if self._per_rank:
            key, _ = self._rank_lru[admitter].popitem(last=False)
            self.memory_lru.pop(key, None)
        else:
            key, _ = self.memory_lru.popitem(last=False)
        obj = self.objects.get(key)
        if obj is None or obj.tier != "memory":
            return StoreWriteTiming()
        self.memory_used_bytes -= obj.bytes
        if self._partitioned:
            self._release_from_rank(obj, admitter)
        self.stats.memory_eviction_count += 1
        self.stats.record_memory_eviction(
            blocks=obj.blocks,
            read_count=obj.read_count,
            residency_s=now - obj.write_time,
        )
        timing = StoreWriteTiming()
        if self.config.enable_offload and self.offload is not None:
            if self._drops_on_eviction(obj):
                # No demote write. The object already has a write-through copy
                # on the offload tier from admission (_put_one), so its bytes
                # stay readable there; this only declines to spend a second,
                # byte-identical write on a block nothing read.
                self.stats.blocks_dropped += obj.blocks
                if self.offload.contains(obj.key):
                    obj.tier = self.offload.tier
                else:
                    self.objects.pop(key, None)
                    self._record_departure(key, "if_read_drop")
                return timing
            self.stats.blocks_demoted += obj.blocks
            if self.config.count_if_read_gate and self.config.demote_policy == "if_read":
                self.stats.if_read_gate_passes += obj.blocks
                if obj.pcm_served_reads > 0:
                    self.stats.if_read_gate_passes_pcm_served += obj.blocks
            obj.tier = self.offload.tier
            op = self.offload.write(obj.key, obj.blocks)
            write_timing = self._write_timing(
                obj,
                op.latency,
                now=now,
                initial_delay=0.0,
            )
            self._record_offload_write(op, owner_key=obj.key)
            self._drop_offload_victims(op, owner_key=obj.key)
            if self.config.charge_eviction_writes:
                timing = write_timing
        else:
            # No offload tier to demote to; the victim leaves the store.
            self.stats.blocks_dropped += obj.blocks
            self.objects.pop(key, None)
            self._record_departure(key, "other")
        return timing

    def _record_departure(self, key: str, reason: str) -> None:
        if self._count_admit_source:
            self._departed[key] = reason

    def _record_admit_source(self, obj: StoreObject) -> None:
        """Classify a memory-tier admission by why its key was absent."""
        if obj.key not in self._seen_keys:
            self._seen_keys.add(obj.key)
            self.stats.pcm_admit_first_blocks += obj.blocks
            return
        reason = self._departed.pop(obj.key, "other")
        if reason == "if_read_drop":
            self.stats.pcm_admit_after_if_read_drop_blocks += obj.blocks
        elif reason == "hbf_evict":
            self.stats.pcm_admit_after_hbf_evict_blocks += obj.blocks
        else:
            self.stats.pcm_admit_after_other_blocks += obj.blocks

    def _drops_on_eviction(self, obj: StoreObject) -> bool:
        """Whether ``demote_policy`` declines the demote write for ``obj``."""
        return self.config.demote_policy == "if_read" and obj.read_count == 0

    def _touch(self, obj: StoreObject) -> None:
        self._access_counter += 1
        obj.last_access = self._access_counter
        if obj.tier == "memory" and obj.key in self.memory_lru:
            self.memory_lru.move_to_end(obj.key)
            if self._per_rank:
                self._rank_lru[obj.rank].move_to_end(obj.key)

    # --- store_partition != "global" -------------------------------------

    def _init_partitions(self) -> None:
        self._rank_index: dict[tuple[int, int, int], int] = {}
        self._rank_used_bytes: dict[int, int] = {}
        self._rank_peak_bytes: dict[int, int] = {}
        self._key_rank: dict[str, int] = {}
        self.stats.partition_reporting = 1
        if self._per_rank:
            ranks = self.config.store_partition_ranks
            if self.config.memory_capacity_blocks is not None:
                rank_blocks = self.config.memory_capacity_blocks // ranks
                self._rank_capacity_bytes = rank_blocks * self.block_bytes
            else:
                self._rank_capacity_bytes = self.config.global_segment_size // ranks
                rank_blocks = self._rank_capacity_bytes // self.block_bytes
            self._rank_lru: list[OrderedDict[str, None]] = [
                OrderedDict() for _ in range(ranks)
            ]
            self.stats.partition_rank_memory_capacity_blocks = rank_blocks

    def _rank_of(self, key: PoolKey) -> int:
        md = key.key_metadata
        identity = (md.tp_rank, md.pp_rank, md.dp_rank)
        index = self._rank_index.get(identity)
        if index is None:
            index = len(self._rank_index)
            if self._per_rank and index >= self.config.store_partition_ranks:
                raise ConfigurationError(
                    "store_partition='per_rank': saw more distinct ranks than "
                    + f"store_partition_ranks={self.config.store_partition_ranks}"
                )
            self._rank_index[identity] = index
            self._rank_used_bytes[index] = 0
            self._rank_peak_bytes[index] = 0
            self.stats.partition_ranks_seen = len(self._rank_index)
        return index

    def _ensure_rank_capacity(
        self,
        bytes_: int,
        *,
        now: float,
        rank: int,
    ) -> tuple[bool, StoreWriteTiming]:
        timing = StoreWriteTiming()
        if bytes_ > self._rank_capacity_bytes:
            return False, timing
        while self._rank_used_bytes[rank] + bytes_ > self._rank_capacity_bytes:
            if self.config.eviction_policy != "lru" or not self._rank_lru[rank]:
                return False, timing
            timing += self._evict_memory_lru(now=now, admitter=rank)
        return True, timing

    def _admit_to_rank(self, obj: StoreObject) -> None:
        rank = obj.rank
        used = self._rank_used_bytes[rank] + obj.bytes
        self._rank_used_bytes[rank] = used
        if self._per_rank:
            self._rank_lru[rank][obj.key] = None
        if used > self._rank_peak_bytes[rank]:
            self._rank_peak_bytes[rank] = used
            peaks = [
                self._rank_peak_bytes[index] // self.block_bytes
                for index in range(len(self._rank_index))
            ]
            self.stats.partition_rank_peak_blocks = peaks
            self.stats.partition_rank_peak_blocks_min = min(peaks)
            self.stats.partition_rank_peak_blocks_max = max(peaks)

    def _release_from_rank(self, obj: StoreObject, admitter: int | None) -> None:
        self._rank_used_bytes[obj.rank] -= obj.bytes
        if obj.rank == admitter:
            self.stats.partition_evictions_same_rank += 1
        else:
            self.stats.partition_evictions_cross_rank += 1

    def _write_timing(
        self,
        obj: StoreObject,
        media_latency: float,
        *,
        now: float,
        initial_delay: float,
    ) -> StoreWriteTiming:
        del now, initial_delay
        obj.persisted = True
        return StoreWriteTiming(
            accounting_latency=media_latency,
            blocking_latency=media_latency,
        )

    def _record_memory_write(self, obj: StoreObject) -> float:
        """Count one memory-tier admission and return its media latency.

        Under the stock ``dram`` media the latency is zero and only the counters
        move, so memory-tier traffic stays visible without changing timing.
        """
        self.stats.memory_write_blocks += obj.blocks
        self.stats.memory_write_bytes += obj.bytes
        latency = self._memory_media_latency(obj.bytes, write=True)
        self.stats.memory_write_latency += latency
        return latency

    def _record_memory_read(self, obj: StoreObject) -> float:
        self.stats.memory_read_blocks += obj.blocks
        self.stats.memory_read_bytes += obj.bytes
        latency = self._memory_media_latency(obj.bytes, write=False)
        self.stats.memory_read_latency += latency
        return latency

    def _memory_media_latency(self, bytes_: int, *, write: bool) -> float:
        """Memory-tier media time, asymmetric by direction.

        ``dram`` keeps the historical model where the memory tier costs only
        interconnect time (charged by the transfer engine, identical in both
        directions). ``pcm`` charges a real media cost whose write side is
        deliberately far more expensive than its read side.
        """
        if (
            self.config.memory_media != "pcm"
            or bytes_ <= 0
            or self.config.store_timing_mode == "free"
        ):
            return 0.0
        if write:
            fixed_us = self.config.pcm_write_latency_us
            bandwidth = self.config.pcm_write_bw_gbps
        else:
            fixed_us = self.config.pcm_read_latency_us
            bandwidth = self.config.pcm_read_bw_gbps
        return fixed_us / 1e6 + bytes_ / _GB / max(1e-9, bandwidth)

    def _record_offload_read(self, op, *, key: str | None = None) -> None:
        self.stats.ssd_read_blocks += op.blocks
        self.stats.ssd_read_bytes += op.bytes
        if self._count_readback and op.blocks and self._hbf_unread.get(key):
            self._hbf_unread[key] = False
            self.stats.hbf_readback_read_blocks += op.blocks

    def _record_offload_write(self, op, *, owner_key: str | None = None) -> None:
        self.stats.ssd_write_blocks += op.blocks
        self.stats.ssd_write_bytes += op.bytes
        if self._count_readback and op.blocks and owner_key is not None:
            self._hbf_unread[owner_key] = True
            self.stats.hbf_readback_written_blocks += op.blocks
        if self._hbf_unlimited and op.blocks:
            used = self.offload.used_blocks
            if used > self.stats.hbf_capacity_peak_used_blocks:
                self.stats.hbf_capacity_peak_used_blocks = used
            self.stats.hbf_capacity_rank_peak_used_blocks_max = self.offload.peak_used_blocks
        # A refused write echoes the key back in evicted_keys; that is an
        # admission rejection for owner_key, not an eviction of it.
        self.stats.ssd_eviction_count += len(
            [key for key in op.evicted_keys if key != owner_key and key in self.objects]
        )

    def _record_ssd_write(self, op, *, owner_key: str | None = None) -> None:
        self._record_offload_write(op, owner_key=owner_key)

    def _drop_offload_victims(self, op, *, owner_key: str | None = None) -> None:
        """Forget objects whose only copy the offload tier just evicted.

        Without this the store keeps entries pointing at offload data that is
        gone: ``lookup`` stops at them, and ``_put_one`` treats them as already
        stored, so the key can never be written back.

        A memory-tier object that merely held a backup copy on the offload tier
        is still valid and is kept; only entries living solely on that tier are
        dropped. ``owner_key`` is skipped because a refused write echoes it back
        in ``evicted_keys`` as an admission rejection, not an eviction.

        Must run after :meth:`_record_offload_write`, which counts victims by
        their presence in ``self.objects``.
        """
        for evicted_key in op.evicted_keys:
            if evicted_key == owner_key:
                continue
            victim = self.objects.get(evicted_key)
            if victim is not None and self._is_offload_tier(victim.tier):
                self.objects.pop(evicted_key, None)
                self._record_departure(evicted_key, "hbf_evict")

    @staticmethod
    def _is_offload_tier(tier: str | None) -> bool:
        return tier == "ssd"

    def _offload_profile(self) -> dict[str, object]:
        profile = self._stock_offload_profile()
        # Recorded only off the stock value, so stock profiles are unchanged.
        if self.config.store_timing_mode != "charged":
            profile["store_timing_mode"] = self.config.store_timing_mode
        if self._hbf_unlimited:
            profile["hbf_capacity_mode"] = self.config.hbf_capacity_mode
        if self._partitioned:
            profile["store_partition"] = self.config.store_partition
            profile["store_partition_ranks"] = self.config.store_partition_ranks
        return profile

    def _stock_offload_profile(self) -> dict[str, object]:
        return {
            "tier": self.offload.tier if self.offload is not None else None,
            "memory_capacity_blocks": self.config.memory_capacity_blocks,
            "ssd_capacity_blocks": self.config.ssd_capacity_blocks,
            "ssd_capacity_gb": self.config.ssd_capacity_gb,
            "ssd_read_bw_gbps": self.config.ssd_read_bw_gbps,
            "ssd_write_bw_gbps": self.config.ssd_write_bw_gbps,
            "ssd_read_latency_us": self.config.ssd_read_latency_us,
            "ssd_write_latency_us": self.config.ssd_write_latency_us,
            "admission_write_policy": self.config.admission_write_policy,
            "memory_media": self.config.memory_media,
            "pcm_read_latency_us": self.config.pcm_read_latency_us,
            "pcm_write_latency_us": self.config.pcm_write_latency_us,
            "pcm_read_bw_gbps": self.config.pcm_read_bw_gbps,
            "pcm_write_bw_gbps": self.config.pcm_write_bw_gbps,
            "demote_policy": self.config.demote_policy,
            "write_blocking": True,
            "write_persistence": "blocking",
        }
