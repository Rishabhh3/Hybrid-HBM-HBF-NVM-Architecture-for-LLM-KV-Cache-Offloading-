"""store_partition (PHASE 14 STEP 3a).

"global" is stock: one LRU and one capacity counter per tier for every rank.
"count" changes nothing and counts cross-rank memory-tier evictions and per-rank
peak occupancy. "per_rank" gives each rank its own LRU and capacity on both
tiers, floor(pool / store_partition_ranks) blocks each.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import MooncakeStoreConnector
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.service import reset_mooncake_services
from TokenSim.mooncake.store import MooncakeStore
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES


def _key(name: str, dp: int = 0, tp: int = 0) -> PoolKey:
    return PoolKey(KeyMetadata(model_name="M", tp_rank=tp, dp_rank=dp), name)


def _store(mode: str, **extra) -> MooncakeStore:
    config = {
        "memory_capacity_blocks": 2,
        "enable_offload": True,
        "ssd_capacity_blocks": 64,
        "admission_write_policy": "write_back",
        "store_partition": mode,
    }
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


# Two ranks interleaving admissions and re-reads.
SEQUENCE = [
    ("a", 0), ("b", 0), ("c", 1), ("a", 0), ("d", 1), ("e", 0), ("b", 0), ("f", 1), ("c", 1),
]


def _drive(store: MooncakeStore) -> None:
    for name, dp in SEQUENCE:
        store.put([_key(name, dp)])
        store.lookup([_key(name, dp)])


class StorePartitionConfigTest(unittest.TestCase):
    def test_default_is_stock_global(self):
        self.assertEqual(parse_mooncake_config({}).store_partition, "global")

    def test_values_parse_and_unknown_is_rejected(self):
        for value in ("global", "count"):
            self.assertEqual(parse_mooncake_config({"store_partition": value}).store_partition, value)
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"store_partition": "sharded"})

    def test_per_rank_requires_rank_count(self):
        with self.assertRaises(ConfigurationError) as ctx:
            parse_mooncake_config({"store_partition": "per_rank"})
        self.assertIn("store_partition_ranks", str(ctx.exception))
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"store_partition": "per_rank", "store_partition_ranks": 0})


class GlobalAndCountTest(unittest.TestCase):
    def test_global_emits_no_partition_keys(self):
        store = _store("global")
        _drive(store)
        self.assertEqual([k for k in store.stats.as_dict() if k.startswith("partition_")], [])

    def test_count_changes_no_behaviour(self):
        stock, counted = _store("global"), _store("count")
        _drive(stock)
        _drive(counted)
        skip = {"offload_profiles"} | {f for f in stock.stats.__dataclass_fields__ if f.startswith("partition_")}
        for name in stock.stats.__dataclass_fields__:
            if name not in skip:
                self.assertEqual(getattr(stock.stats, name), getattr(counted.stats, name), name)
        self.assertEqual(
            {k: (o.tier, o.read_count) for k, o in stock.objects.items()},
            {k: (o.tier, o.read_count) for k, o in counted.objects.items()},
        )
        self.assertEqual(list(stock.memory_lru), list(counted.memory_lru))

    def test_count_classifies_cross_rank_evictions(self):
        store = _store("count")
        store.put([_key("a", 0)])
        store.put([_key("b", 0)])
        store.put([_key("c", 1)])  # evicts a (rank 0) for a rank-1 admission
        store.put([_key("d", 1)])  # evicts b (rank 0) for a rank-1 admission
        store.put([_key("e", 1)])  # evicts c (rank 1) for a rank-1 admission
        self.assertEqual(store.stats.partition_evictions_cross_rank, 2)
        self.assertEqual(store.stats.partition_evictions_same_rank, 1)
        self.assertEqual(store.stats.memory_eviction_count, 3)
        self.assertEqual(store.stats.partition_ranks_seen, 2)
        self.assertEqual(store.stats.partition_rank_peak_blocks, [2, 2])
        exported = store.stats.as_dict()
        self.assertEqual(exported["partition_evictions_cross_rank"], 2)
        self.assertEqual(exported["partition_rank_peak_blocks_min"], 2)

    def test_tp_rank_is_part_of_the_rank_identity(self):
        store = _store("count")
        store.put([_key("a", dp=0, tp=0)])
        store.put([_key("a", dp=0, tp=1)])
        self.assertEqual(store.stats.partition_ranks_seen, 2)


class PerRankTest(unittest.TestCase):
    def test_capacities_are_floored_per_rank(self):
        store = _store("per_rank", store_partition_ranks=2, memory_capacity_blocks=5, ssd_capacity_blocks=7)
        self.assertEqual(store.stats.partition_rank_memory_capacity_blocks, 2)
        self.assertEqual(store.stats.partition_rank_ssd_capacity_blocks, 3)
        self.assertEqual(store.offload.capacity_blocks, 6)

    def test_a_rank_only_evicts_its_own_blocks(self):
        store = _store("per_rank", store_partition_ranks=2, memory_capacity_blocks=4)
        store.put([_key("a", 0)])
        store.put([_key("b", 0)])
        for name in ("c", "d", "e", "f"):
            store.put([_key(name, 1)])
        self.assertEqual(store.objects[_key("a", 0).to_string()].tier, "memory")
        self.assertEqual(store.objects[_key("b", 0).to_string()].tier, "memory")
        self.assertEqual(store.stats.partition_evictions_cross_rank, 0)
        self.assertEqual(store.stats.partition_evictions_same_rank, 2)

    def test_lru_order_is_per_rank(self):
        store = _store("per_rank", store_partition_ranks=2, memory_capacity_blocks=4)
        store.put([_key("a", 1)])
        store.put([_key("b", 1)])
        store.lookup([_key("a", 1)])
        store.get([_key("a", 1)], "memory")  # touches a: b is now LRU
        store.put([_key("c", 1)])
        self.assertEqual(store.objects[_key("a", 1).to_string()].tier, "memory")
        self.assertEqual(store.objects[_key("b", 1).to_string()].tier, "ssd")

    def test_offload_capacity_is_per_rank(self):
        store = _store(
            "per_rank", store_partition_ranks=2, memory_capacity_blocks=2, ssd_capacity_blocks=4
        )
        # Rank 0: memory holds 1, offload partition holds 2. Five puts overflow it.
        for name in "abcde":
            store.put([_key(name, 0)])
        store.put([_key("z", 1)])
        self.assertEqual(store.offload.parts[0].used_blocks, 2)
        self.assertEqual(store.offload.parts[1].used_blocks, 0)
        self.assertGreater(store.stats.ssd_eviction_count, 0)
        self.assertTrue(store.offload.contains(store.objects[_key("d", 0).to_string()].key))

    def test_more_ranks_than_partitions_is_an_error(self):
        store = _store("per_rank", store_partition_ranks=2, memory_capacity_blocks=4)
        store.put([_key("a", 0)])
        store.put([_key("b", 1)])
        with self.assertRaises(ConfigurationError):
            store.put([_key("c", 2)])

    def test_profile_records_partition_only_off_stock(self):
        self.assertNotIn("store_partition", _store("global")._offload_profile())
        self.assertEqual(_store("count")._offload_profile()["store_partition"], "count")


class PerRankConnectorTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def _connector(self, world_size: int, ranks: int, dp: int = 4):
        cache_config = SimpleNamespace(
            block_size=16, size_per_token=8, model="M", rank_info=ParallelRankInfo(),
            parallel_config=SimpleNamespace(world_size=world_size, data_parallel_size=dp),
        )
        return MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 16,
                    "enable_offload": True,
                    "ssd_capacity_blocks": 64,
                    "store_partition": "per_rank",
                    "store_partition_ranks": ranks,
                },
            ),
            cache_config,
        )

    def test_rank_count_must_match_the_writing_ranks(self):
        # TP2/DP4: 8 workers, but only the 4 TP-0 workers write to the store.
        self._connector(world_size=8, ranks=4, dp=4)
        reset_mooncake_services()
        with self.assertRaises(ConfigurationError):
            self._connector(world_size=8, ranks=8, dp=4)

    def test_single_worker_has_one_writer(self):
        self._connector(world_size=1, ranks=1, dp=1)

    def test_export_merges_partition_keys(self):
        self.assertIn("partition_", OPTIONAL_MOONCAKE_KEY_PREFIXES)


if __name__ == "__main__":
    unittest.main()
