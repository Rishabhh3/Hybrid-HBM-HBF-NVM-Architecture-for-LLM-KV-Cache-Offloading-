from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_GB = 1 << 30

# Result export only surfaces a small sample of pool keys; keeping more than
# this in memory only grows RSS with simulation length.
POOL_KEY_SAMPLE_LIMIT = 64

# Upper edges (seconds) of the memory-tier residency histogram; the final
# bucket is open-ended. Sized for trace runs of a few hundred seconds.
RESIDENCY_BUCKET_EDGES_S: tuple[float, ...] = (
    0.1,
    1.0,
    10.0,
    30.0,
    60.0,
    120.0,
    180.0,
    240.0,
    300.0,
)


def _residency_bucket_labels() -> list[str]:
    labels = []
    low = 0.0
    for edge in RESIDENCY_BUCKET_EDGES_S:
        labels.append(f"{low:g}-{edge:g}s")
        low = edge
    labels.append(f"{low:g}s+")
    return labels


RESIDENCY_BUCKET_LABELS = _residency_bucket_labels()


@dataclass
class MooncakeStats:
    get_count: int = 0
    put_count: int = 0
    store_hit_count: int = 0
    store_miss_count: int = 0
    memory_tier_hit_count: int = 0
    disk_tier_hit_count: int = 0
    ssd_read_blocks: int = 0
    ssd_read_bytes: int = 0
    ssd_write_blocks: int = 0
    ssd_write_bytes: int = 0
    transferred_bytes: int = 0
    transfer_latency: float = 0.0
    p2p_latency: float = 0.0
    store_latency: float = 0.0
    admission_count: int = 0
    admission_rejection_count: int = 0
    memory_eviction_count: int = 0
    ssd_eviction_count: int = 0
    pending_async_jobs: int = 0
    local_gpu_hit_tokens: int = 0
    mooncake_memory_hit_tokens: int = 0
    mooncake_disk_hit_tokens: int = 0
    load_wait_time: float = 0.0
    save_wait_time: float = 0.0
    # Memory-tier eviction classification. A block evicted from memory having
    # never been read back (read_count == 0) is a wasted write: the bytes it
    # cost the offload tier bought no cache hit.
    wasted_write_blocks: int = 0
    useful_write_blocks: int = 0
    residency_sum_s: float = 0.0
    residency_count: int = 0
    # demote_policy accounting; every memory eviction is exactly one of these.
    blocks_dropped: int = 0
    blocks_demoted: int = 0
    # Blocks that went straight to the offload tier because the memory tier
    # could not hold them; independent of the admission write policy.
    memory_bypass_blocks: int = 0
    # Memory-tier traffic. Counted for every media type; only "pcm" charges
    # media time for it.
    memory_write_blocks: int = 0
    memory_write_bytes: int = 0
    memory_read_blocks: int = 0
    memory_read_bytes: int = 0
    memory_write_latency: float = 0.0
    memory_read_latency: float = 0.0
    pool_keys: list[str] = field(default_factory=list)
    offload_tiers: list[str] = field(default_factory=list)
    offload_profiles: list[dict[str, Any]] = field(default_factory=list)
    residency_histogram: list[int] = field(
        default_factory=lambda: [0] * (len(RESIDENCY_BUCKET_EDGES_S) + 1)
    )

    @property
    def eviction_count(self) -> int:
        """Total evictions across tiers.

        Derived rather than stored so it cannot drift from its parts; the two
        tier counters are the ones incremented at the eviction sites.
        """
        return self.memory_eviction_count + self.ssd_eviction_count

    # Lists of per-bucket counters are summed element-wise; every other list
    # field is a sample that is concatenated and truncated.
    _COUNTER_LIST_FIELDS = ("residency_histogram",)

    def aggregate(self, other: "MooncakeStats") -> "MooncakeStats":
        result = MooncakeStats()
        for field_name in self.__dataclass_fields__:
            left = getattr(self, field_name)
            right = getattr(other, field_name)
            if field_name in self._COUNTER_LIST_FIELDS:
                setattr(result, field_name, [a + b for a, b in zip(left, right)])
            elif isinstance(left, list):
                setattr(result, field_name, [*left, *right][:POOL_KEY_SAMPLE_LIMIT])
            else:
                setattr(result, field_name, left + right)
        return result

    @property
    def mean_residency_s(self) -> float:
        if self.residency_count <= 0:
            return 0.0
        return self.residency_sum_s / self.residency_count

    def record_memory_eviction(self, *, blocks: int, read_count: int, residency_s: float) -> None:
        """Classify one memory-tier eviction and record its residency.

        ``read_count`` is the number of lookups that hit this object while it
        was resident; zero means the offload write it is about to trigger (or
        the write-through already spent on it) bought nothing.
        """
        if read_count > 0:
            self.useful_write_blocks += blocks
        else:
            self.wasted_write_blocks += blocks
        residency_s = max(0.0, float(residency_s))
        self.residency_sum_s += residency_s
        self.residency_count += 1
        self.residency_histogram[self._residency_bucket(residency_s)] += 1

    @staticmethod
    def _residency_bucket(residency_s: float) -> int:
        for index, edge in enumerate(RESIDENCY_BUCKET_EDGES_S):
            if residency_s < edge:
                return index
        return len(RESIDENCY_BUCKET_EDGES_S)

    def record_transfer(
        self,
        *,
        bytes_: int,
        latency: float,
        kind: str,
        blocking_latency: float | None = None,
    ) -> None:
        self.transferred_bytes += bytes_
        self.transfer_latency += latency
        if kind == "p2p":
            self.p2p_latency += latency
        elif kind.startswith("store"):
            self.store_latency += latency
        if blocking_latency is None:
            blocking_latency = latency
        if kind in {"store", "store_load", "load"}:
            self.load_wait_time += blocking_latency
        elif kind in {"store_save", "save"}:
            self.save_wait_time += blocking_latency

    def record_pool_keys(self, keys: list[str]) -> None:
        remaining = POOL_KEY_SAMPLE_LIMIT - len(self.pool_keys)
        if remaining > 0:
            self.pool_keys.extend(keys[:remaining])

    def record_offload_tier(self, tier: str) -> None:
        if tier not in self.offload_tiers:
            self.offload_tiers.append(tier)

    def record_offload_profile(self, profile: dict[str, Any]) -> None:
        self.offload_profiles.append(dict(profile))

    def as_dict(self) -> dict[str, Any]:
        bandwidth = 0.0
        if self.transfer_latency > 0:
            bandwidth = self.transferred_bytes / _GB / self.transfer_latency
        offload_tiers = sorted(set(self.offload_tiers))
        offload_profiles = []
        seen_profiles = set()
        for profile in self.offload_profiles:
            fingerprint = json.dumps(profile, sort_keys=True, default=str)
            if fingerprint in seen_profiles:
                continue
            seen_profiles.add(fingerprint)
            offload_profiles.append(profile)
        return {
            "mooncake_get_count": self.get_count,
            "mooncake_put_count": self.put_count,
            "mooncake_store_hit_count": self.store_hit_count,
            "mooncake_store_miss_count": self.store_miss_count,
            "mooncake_memory_tier_hit_count": self.memory_tier_hit_count,
            "mooncake_disk_tier_hit_count": self.disk_tier_hit_count,
            "mooncake_ssd_read_blocks": self.ssd_read_blocks,
            "mooncake_ssd_read_bytes": self.ssd_read_bytes,
            "mooncake_ssd_write_blocks": self.ssd_write_blocks,
            "mooncake_ssd_write_bytes": self.ssd_write_bytes,
            "mooncake_transferred_bytes": self.transferred_bytes,
            "mooncake_effective_transfer_bandwidth_gbps": bandwidth,
            "mooncake_transfer_latency": self.transfer_latency,
            "mooncake_p2p_latency": self.p2p_latency,
            "mooncake_store_latency": self.store_latency,
            "mooncake_admission_count": self.admission_count,
            "mooncake_admission_rejection_count": self.admission_rejection_count,
            "mooncake_eviction_count": self.eviction_count,
            "mooncake_memory_eviction_count": self.memory_eviction_count,
            "mooncake_ssd_eviction_count": self.ssd_eviction_count,
            "mooncake_pending_async_jobs": self.pending_async_jobs,
            "mooncake_local_gpu_hit_tokens": self.local_gpu_hit_tokens,
            "mooncake_memory_hit_tokens": self.mooncake_memory_hit_tokens,
            "mooncake_disk_hit_tokens": self.mooncake_disk_hit_tokens,
            "mooncake_load_wait_time": self.load_wait_time,
            "mooncake_save_wait_time": self.save_wait_time,
            "mooncake_wasted_write_blocks": self.wasted_write_blocks,
            "mooncake_useful_write_blocks": self.useful_write_blocks,
            "mooncake_mean_residency_s": self.mean_residency_s,
            "mooncake_residency_histogram": dict(
                zip(RESIDENCY_BUCKET_LABELS, self.residency_histogram)
            ),
            "mooncake_blocks_dropped": self.blocks_dropped,
            "mooncake_blocks_demoted": self.blocks_demoted,
            "mooncake_memory_bypass_blocks": self.memory_bypass_blocks,
            "mooncake_memory_write_blocks": self.memory_write_blocks,
            "mooncake_memory_write_bytes": self.memory_write_bytes,
            "mooncake_memory_read_blocks": self.memory_read_blocks,
            "mooncake_memory_read_bytes": self.memory_read_bytes,
            "mooncake_memory_write_latency": self.memory_write_latency,
            "mooncake_memory_read_latency": self.memory_read_latency,
            "mooncake_pool_keys": self.pool_keys[:64],
            "mooncake_offload_tiers": offload_tiers,
            "mooncake_offload_profiles": offload_profiles,
        }
