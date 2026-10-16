from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from TokenSim.errors import ConfigurationError


SUPPORTED_CONNECTORS = {
    "MooncakeConnector",
    "MooncakeStoreConnector",
    "MultiConnector",
}
SUPPORTED_MODES = {"embedded", "standalone-store"}
SUPPORTED_PROTOCOLS = {"tcp", "rdma", "nvlink", "nvmeof"}
SUPPORTED_OFFLOAD_TIERS = {"ssd"}
SUPPORTED_ADMISSION_POLICIES = {"always", "never"}
SUPPORTED_EVICTION_POLICIES = {"lru", "none"}
SUPPORTED_DEMOTE_POLICIES = {"always", "if_read"}
SUPPORTED_MEMORY_MEDIA = {"dram", "pcm"}
SUPPORTED_ADMISSION_WRITE_POLICIES = {"write_through", "write_back"}
SUPPORTED_SAVE_POLICIES = {"mooncake", "every_step", "once"}
# What an HBM (GPU prefix-cache) eviction does. "prefill" is stock: HBM eviction
# is invisible to the store, which is written only by the prefill save path.
# "count" adds instrumentation and nothing else. "on_evict" additionally sends
# the evicted block to the store through the normal put path, i.e. the prefill
# save PLUS an eviction-driven save. "spill_only" switches the prefill save off:
# a block enters the store ONLY when HBM evicts it (PHASE 14 STEP 3b).
#
# Spelled hbm_evict_save_policy, not save_policy: save_policy already exists
# above and means how often the *prefill* save re-runs (mooncake/every_step/once).
SUPPORTED_HBM_EVICT_SAVE_POLICIES = {"prefill", "count", "on_evict", "spill_only"}
# Whether the store charges simulated time. "charged" is stock: media time on
# both tiers and interconnect transfer time are computed from the knobs below.
# "free" makes every store media and transfer latency exactly zero, leaving
# byte, block and hit accounting untouched -- a diagnostic for whether timing
# feeds back into write volume.
SUPPORTED_STORE_TIMING_MODES = {"charged", "free"}
# How the store's capacity is shared between the ranks whose keys it holds.
# "global" is stock: one LRU and one capacity counter per tier for every rank.
# "count" keeps that and only counts -- memory-tier evictions whose victim
# belongs to another rank than the admission that forced them, and each rank's
# peak memory-tier occupancy. "per_rank" gives each rank its own LRU and
# capacity on both tiers, floor(pool / store_partition_ranks) blocks each.
SUPPORTED_STORE_PARTITIONS = {"global", "count", "per_rank"}
# Where the store lookup for a new request begins. "zero" is stock: the lookup
# walks the prompt's keys from block 0 and stops at the first store miss, so a
# prompt head held only in HBM hides any stored tail. "after_hbm" begins at the
# first block NOT already hit in HBM (num_computed_tokens), with that block's
# key, and stops at the first store miss after it -- the vLLM KV-connector
# contract, KVConnectorBase_V1.get_num_new_matched_tokens(request,
# num_computed_tokens), asks the external store for tokens beyond the locally
# computed ones. (Interface only; no claim about how LMCache or Mooncake
# implement it internally.)
SUPPORTED_STORE_LOOKUP_STARTS = {"zero", "after_hbm"}
# What counts as a "read" for demote_policy = if_read. "walked" is stock: every
# key a lookup walks over is credited (store.py lookup), including blocks whose
# tokens are then served from HBM. "served": a key is credited only when its
# block is actually served from the store (MooncakeStore.get).
SUPPORTED_READ_CREDITS = {"walked", "served"}
# Which keys a load plan reads. "from_start" is stock: the first `blocks` keys of
# the lookup result, counted from block 0. "missing": exactly the span not
# already hit in HBM.
SUPPORTED_LOAD_SPANS = {"from_start", "missing"}
# Whether the offload (HBF) tier's capacity binds. "finite" is stock: a write
# past the tier's capacity evicts its LRU entries. "unlimited" never evicts and
# never refuses a write -- a SENSITIVITY setting for whether a result depends on
# HBF capacity (dies per stack is OPEN), not a hardware value. The capacity
# knobs are still read; the peak occupancy the tier reached is reported
# (hbf_capacity_*).
SUPPORTED_HBF_CAPACITY_MODES = {"finite", "unlimited"}

_GB = 1 << 30


@dataclass
class MooncakeConfig:
    mode: str = "embedded"
    protocol: str = "rdma"
    global_segment_size: int = 8 * _GB
    local_buffer_size: int = 1 * _GB
    enable_offload: bool = False
    offload_tier: str = "ssd"
    ssd_capacity_gb: float = 0.0
    ssd_read_bw_gbps: float = 7.0
    ssd_write_bw_gbps: float = 3.0
    ssd_read_latency_us: float = 100.0
    ssd_write_latency_us: float = 200.0
    replica_num: int = 1
    admission_policy: str = "always"
    eviction_policy: str = "lru"
    # When true, the offload write triggered by a memory-tier eviction is
    # charged to the put that forced the eviction. Default false keeps the
    # historical behavior where eviction-driven writes cost no simulated time.
    charge_eviction_writes: bool = False
    # What a memory-tier eviction does with the victim. "always" demotes it to
    # the offload tier (stock behavior). "if_read" demotes only victims that
    # were read back at least once while resident; never-read victims are
    # dropped from the write stream instead, to spend no further NAND writes on
    # bytes that bought no cache hit.
    # Whether a memory-tier admission also writes the block through to the
    # offload tier. "write_through" is stock: every admission spends an offload
    # write immediately, so the later demote write is a byte-identical duplicate
    # and no eviction-time policy can remove more than that duplicate half.
    # "write_back" spends no offload write at admission; the block reaches the
    # offload tier only if demote_policy later decides to demote it.
    #
    # Spelled ``admission_write_policy`` because ``admission_policy`` already
    # exists and means something else (whether to admit at all, always/never).
    # parse_mooncake_config also accepts these two values under the name
    # ``admission_policy``; the value sets are disjoint so routing is unambiguous.
    admission_write_policy: str = "write_through"
    demote_policy: str = "always"
    # Media backing the Mooncake memory tier. "dram" is stock: the memory tier
    # costs no media time in either direction, only interconnect time. "pcm"
    # charges asymmetric read/write media latency on the memory tier, which is
    # the whole point of a PCM tier -- its writes are far more expensive than
    # its reads, and a DRAM model with one protocol latency for both directions
    # cannot express that.
    memory_media: str = "dram"
    pcm_read_latency_us: float = 0.1
    pcm_write_latency_us: float = 1.0
    pcm_read_bw_gbps: float = 100.0
    pcm_write_bw_gbps: float = 20.0
    # "every_step" re-saves a request's prompt blocks on every scheduled step
    # (historical behavior); "once" saves each request a single time.
    save_policy: str = "mooncake"
    hbm_evict_save_policy: str = "prefill"
    store_timing_mode: str = "charged"
    store_partition: str = "global"
    store_lookup_start: str = "zero"
    read_credit: str = "walked"
    load_span: str = "from_start"
    # Instrumentation only: at every if_read demotion, whether the block had at
    # least one read actually served from the memory (PCM) tier.
    count_if_read_gate: bool = False
    hbf_capacity_mode: str = "finite"
    # Instrumentation only: classify every memory-tier (PCM) admission by why
    # the key was absent from the store -- never stored before, dropped by
    # demote_policy = if_read, or dropped when the offload tier evicted its only
    # copy. Off by default so stock output gains no keys.
    count_pcm_admit_source: bool = False
    # Required for store_partition = "per_rank", with no default: the number of
    # ranks the pool is split between. Must equal the number of ranks that write
    # to the store -- one TP-0/PP-0 worker per data-parallel group, since only
    # those are scheduled (LLMEngine._is_schedulable_worker).
    store_partition_ranks: int | None = None
    # Instrumentation only: count prefill saves skipped by the "mooncake"
    # save_policy rule that a step which loads from the store does not save.
    # Off by default so stock output gains no keys.
    count_aligned_save_skips: bool = False
    # Instrumentation only: for every block written to the offload (HBF) tier,
    # record whether it is read back from that tier before the run ends.
    # Off by default so stock output gains no keys.
    count_hbf_readback: bool = False
    load_async: bool = False
    transfer_overlap: bool = False
    num_nics: int = 1
    parallel_paths: int = 1
    fixed_latency_us: float | None = None
    bandwidth_gbps: float | None = None
    memory_capacity_blocks: int | None = None
    ssd_capacity_blocks: int | None = None
    staging_buffer_size: int = 256 * 1024 * 1024
    preferred_segment: str | None = None
    group_id: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0
    protocol_bandwidth_gbps: dict[str, float] = field(default_factory=dict)
    protocol_latency_us: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"unsupported Mooncake mode {self.mode!r}; expected one of "
                + f"{sorted(SUPPORTED_MODES)}"
            )
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ConfigurationError(
                f"unsupported Mooncake protocol {self.protocol!r}; expected one of "
                + f"{sorted(SUPPORTED_PROTOCOLS)}"
            )
        if self.offload_tier not in SUPPORTED_OFFLOAD_TIERS:
            raise ConfigurationError(
                f"unsupported Mooncake offload_tier {self.offload_tier!r}; "
                + f"expected one of {sorted(SUPPORTED_OFFLOAD_TIERS)}"
            )
        if self.admission_policy not in SUPPORTED_ADMISSION_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake admission_policy "
                + f"{self.admission_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_ADMISSION_POLICIES)}"
            )
        if self.eviction_policy not in SUPPORTED_EVICTION_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake eviction_policy "
                + f"{self.eviction_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_EVICTION_POLICIES)}"
            )
        if self.admission_write_policy not in SUPPORTED_ADMISSION_WRITE_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake admission_write_policy "
                + f"{self.admission_write_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_ADMISSION_WRITE_POLICIES)}"
            )
        if self.memory_media not in SUPPORTED_MEMORY_MEDIA:
            raise ConfigurationError(
                "unsupported Mooncake memory_media "
                + f"{self.memory_media!r}; expected one of "
                + f"{sorted(SUPPORTED_MEMORY_MEDIA)}"
            )
        if self.demote_policy not in SUPPORTED_DEMOTE_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake demote_policy "
                + f"{self.demote_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_DEMOTE_POLICIES)}"
            )
        if self.save_policy not in SUPPORTED_SAVE_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake save_policy "
                + f"{self.save_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_SAVE_POLICIES)}"
            )
        if self.hbm_evict_save_policy not in SUPPORTED_HBM_EVICT_SAVE_POLICIES:
            raise ConfigurationError(
                "unsupported Mooncake hbm_evict_save_policy "
                + f"{self.hbm_evict_save_policy!r}; expected one of "
                + f"{sorted(SUPPORTED_HBM_EVICT_SAVE_POLICIES)}"
            )
        if self.store_timing_mode not in SUPPORTED_STORE_TIMING_MODES:
            raise ConfigurationError(
                "unsupported Mooncake store_timing_mode "
                + f"{self.store_timing_mode!r}; expected one of "
                + f"{sorted(SUPPORTED_STORE_TIMING_MODES)}"
            )
        for name, allowed in (
            ("read_credit", SUPPORTED_READ_CREDITS),
            ("load_span", SUPPORTED_LOAD_SPANS),
            ("hbf_capacity_mode", SUPPORTED_HBF_CAPACITY_MODES),
        ):
            if getattr(self, name) not in allowed:
                raise ConfigurationError(
                    f"unsupported Mooncake {name} {getattr(self, name)!r}; "
                    + f"expected one of {sorted(allowed)}"
                )
        if self.store_lookup_start not in SUPPORTED_STORE_LOOKUP_STARTS:
            raise ConfigurationError(
                "unsupported Mooncake store_lookup_start "
                + f"{self.store_lookup_start!r}; expected one of "
                + f"{sorted(SUPPORTED_STORE_LOOKUP_STARTS)}"
            )
        if self.store_partition not in SUPPORTED_STORE_PARTITIONS:
            raise ConfigurationError(
                "unsupported Mooncake store_partition "
                + f"{self.store_partition!r}; expected one of "
                + f"{sorted(SUPPORTED_STORE_PARTITIONS)}"
            )
        if self.store_partition == "per_rank":
            if self.store_partition_ranks is None:
                raise ConfigurationError(
                    "store_partition='per_rank' requires store_partition_ranks"
                )
            if int(self.store_partition_ranks) < 1:
                raise ConfigurationError("store_partition_ranks must be at least 1")
            self.store_partition_ranks = int(self.store_partition_ranks)
        for field_name in (
            "global_segment_size",
            "local_buffer_size",
            "staging_buffer_size",
            "replica_num",
            "num_nics",
            "parallel_paths",
        ):
            value = getattr(self, field_name)
            if int(value) < 0:
                raise ConfigurationError(f"{field_name} must be non-negative")
            setattr(self, field_name, int(value))
        if self.replica_num < 1:
            raise ConfigurationError("replica_num must be at least 1")
        if self.num_nics < 1:
            raise ConfigurationError("num_nics must be at least 1")
        if self.parallel_paths < 1:
            raise ConfigurationError("parallel_paths must be at least 1")
        for field_name in (
            "ssd_read_bw_gbps",
            "ssd_write_bw_gbps",
            "ssd_read_latency_us",
            "ssd_write_latency_us",
            "pcm_read_bw_gbps",
            "pcm_write_bw_gbps",
            "pcm_read_latency_us",
            "pcm_write_latency_us",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ConfigurationError(f"{field_name} must be non-negative")
        if self.ssd_capacity_gb < 0:
            raise ConfigurationError("ssd_capacity_gb must be non-negative")
        if self.memory_capacity_blocks is not None and self.memory_capacity_blocks < 0:
            raise ConfigurationError("memory_capacity_blocks must be non-negative")
        if self.ssd_capacity_blocks is not None and self.ssd_capacity_blocks < 0:
            raise ConfigurationError("ssd_capacity_blocks must be non-negative")


def parse_mooncake_config(value: dict[str, Any] | None = None) -> MooncakeConfig:
    raw = dict(value or {})
    raw.pop("connectors", None)
    _route_admission_write_policy(raw)
    _normalize_size(raw, "global_segment_size", "global_segment_size_gb")
    _normalize_size(raw, "local_buffer_size", "local_buffer_size_gb")
    _normalize_size(raw, "staging_buffer_size", "staging_buffer_size_gb")
    return MooncakeConfig(**raw)


def validate_mooncake_connector_config(
    connector_name: str | None,
    extra_config: dict[str, Any] | None,
) -> None:
    if connector_name not in SUPPORTED_CONNECTORS:
        return
    extra_config = dict(extra_config or {})
    if connector_name == "MultiConnector":
        children = extra_config.get("connectors", [])
        if not isinstance(children, list):
            raise ConfigurationError("MultiConnector connectors must be a list")
        for child in children:
            if not isinstance(child, dict):
                raise ConfigurationError("MultiConnector child config must be an object")
            validate_mooncake_connector_config(
                child.get("kv_connector"),
                child.get("kv_connector_extra_config", {}),
            )
        return
    parse_mooncake_config(extra_config)


def _route_admission_write_policy(raw: dict[str, Any]) -> None:
    """Accept the write policy spelled as ``admission_policy``.

    ``admission_policy`` (always/never) and ``admission_write_policy``
    (write_through/write_back) are different knobs with disjoint value sets, so a
    write-policy value appearing under the older name can be routed without
    ambiguity rather than failing validation.
    """
    value = raw.get("admission_policy")
    if value not in SUPPORTED_ADMISSION_WRITE_POLICIES:
        return
    if "admission_write_policy" in raw and raw["admission_write_policy"] != value:
        raise ConfigurationError(
            "conflicting admission write policy: admission_policy="
            + f"{value!r} but admission_write_policy="
            + f"{raw['admission_write_policy']!r}"
        )
    raw.pop("admission_policy")
    raw["admission_write_policy"] = value


def _normalize_size(raw: dict[str, Any], bytes_key: str, gb_key: str) -> None:
    if bytes_key in raw:
        return
    if gb_key in raw:
        raw[bytes_key] = int(float(raw.pop(gb_key)) * _GB)
