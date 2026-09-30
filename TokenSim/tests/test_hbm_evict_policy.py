"""hbm_evict_save_policy: the HBM-eviction hook and its instrumentation.

Stock behaviour is `"prefill"`: the Mooncake store is written only by the
prefill save path and an HBM eviction is invisible to it. `"count"` adds
counters and nothing else; `"on_evict"` additionally writes the evicted block
back into the store through the normal put path.
"""

from __future__ import annotations

import unittest

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.block_manager import BlockAllocator, BlockManager
from TokenSim.block.kv_cache_manager import KVCacheManager
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import MooncakeStoreConnector
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.service import reset_mooncake_services


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()
    num_gpu_blocks = 8
    num_cpu_blocks = 4


class HbmEvictPolicyConfigTest(unittest.TestCase):
    def test_default_is_stock_prefill(self):
        self.assertEqual(parse_mooncake_config({}).hbm_evict_save_policy, "prefill")

    def test_supported_values_parse(self):
        for value in ("prefill", "count", "on_evict"):
            config = parse_mooncake_config({"hbm_evict_save_policy": value})
            self.assertEqual(config.hbm_evict_save_policy, value)

    def test_unknown_value_is_rejected(self):
        with self.assertRaises(ConfigurationError) as ctx:
            parse_mooncake_config({"hbm_evict_save_policy": "on_release"})
        self.assertIn("hbm_evict_save_policy", str(ctx.exception))


class KVCacheManagerEvictObserverTest(unittest.TestCase):
    """The hook itself, independent of any connector."""

    @staticmethod
    def _key(name: str):
        return build_prefix_keys([name], model="TestModel")[0]

    def test_no_observer_by_default(self):
        manager = KVCacheManager(block_size=16, model="TestModel")
        self.assertIsNone(manager.evict_observer)

    def test_capacity_eviction_from_allocate_reaches_the_observer(self):
        manager = KVCacheManager(block_size=16, model="TestModel")
        seen: list[tuple[int, str, object]] = []
        # The block must still carry its key when the observer runs, otherwise
        # the hook has nothing to build a pool key from.
        manager.set_evict_observer(
            lambda block, reason: seen.append(
                (block.block_number, reason, block.block_hash)
            )
        )
        allocator = BlockAllocator(Device.GPU, 16, 1, kv_cache_manager=manager)

        block = allocator.allocate()
        manager.register_blocks([block], [self._key("a")])
        allocator.free(block)
        self.assertEqual(seen, [])

        allocator.allocate()
        self.assertEqual(len(seen), 1)
        number, reason, block_hash = seen[0]
        self.assertEqual(number, block.block_number)
        self.assertEqual(reason, "capacity")
        self.assertEqual(block_hash, self._key("a"))

    def test_rekey_eviction_is_reported_with_its_own_reason(self):
        manager = KVCacheManager(block_size=16, model="TestModel")
        reasons: list[str] = []
        manager.set_evict_observer(lambda block, reason: reasons.append(reason))
        block = PhysicalTokenBlock(Device.GPU, 0, 16)

        manager.register_blocks([block], [self._key("a")])
        self.assertEqual(reasons, [])
        manager.register_blocks([block], [self._key("b")])
        self.assertEqual(reasons, ["rekey"])

    def test_uncached_block_never_reaches_the_observer(self):
        """Decode blocks are uncached, so allocate() hands them back silently."""
        manager = KVCacheManager(block_size=16, model="TestModel")
        seen: list[str] = []
        manager.set_evict_observer(lambda block, reason: seen.append(reason))
        allocator = BlockAllocator(Device.GPU, 16, 1, kv_cache_manager=manager)

        block = allocator.allocate()
        allocator.free(block)
        allocator.allocate()
        self.assertEqual(seen, [])


class HbmEvictConnectorTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def _connector(self, policy: str, **extra):
        config = {
            "memory_capacity_blocks": 64,
            "enable_offload": True,
            "ssd_capacity_blocks": 256,
            "hbm_evict_save_policy": policy,
        }
        config.update(extra)
        return MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config=config,
            ),
            _CacheConfig(),
        )

    @staticmethod
    def _save(connector, req):
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req], "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        connector.wait_for_save()

    @staticmethod
    def _block(name: str | None, number: int = 0):
        block = PhysicalTokenBlock(Device.GPU, number, 16)
        if name is not None:
            block.block_hash = build_prefix_keys([name], model="TestModel")[0]
        return block

    def test_stock_policy_installs_no_observer_and_emits_no_counters(self):
        connector = self._connector("prefill")
        self.assertIsNone(connector.hbm_evict_observer)
        exported = connector.mooncake_stats.as_dict()
        self.assertEqual([k for k in exported if k.startswith("hbm_evict_")], [])

    def test_enabled_policy_emits_counters(self):
        connector = self._connector("count")
        self.assertIsNotNone(connector.hbm_evict_observer)
        exported = connector.mooncake_stats.as_dict()
        self.assertIn("hbm_evict_calls", exported)
        self.assertIn("hbm_evict_absent_blocks", exported)

    def test_count_classifies_a_key_already_in_the_store_and_writes_nothing(self):
        connector = self._connector("count")
        req = Request(0, 32, 4, 16, hash_ids=["a", "b"])
        self._save(connector, req)
        stats = connector.mooncake_stats
        before = (stats.ssd_write_blocks, stats.memory_write_blocks, stats.put_count)

        connector.hbm_evict_observer(self._block("a"), "capacity")

        self.assertEqual(stats.hbm_evict_calls, 1)
        self.assertEqual(stats.hbm_evict_blocks, 1)
        self.assertEqual(stats.hbm_evict_bytes, connector.block_bytes)
        self.assertEqual(stats.hbm_evict_prompt_blocks, 1)
        self.assertEqual(stats.hbm_evict_keyed_blocks, 1)
        self.assertEqual(stats.hbm_evict_present_blocks, 1)
        self.assertEqual(stats.hbm_evict_absent_blocks, 0)
        self.assertEqual(stats.hbm_evict_puts, 0)
        self.assertEqual(
            (stats.ssd_write_blocks, stats.memory_write_blocks, stats.put_count),
            before,
        )

    def test_count_classifies_a_key_absent_from_the_store(self):
        connector = self._connector("count")
        connector.hbm_evict_observer(self._block("never-saved"), "capacity")
        stats = connector.mooncake_stats
        self.assertEqual(stats.hbm_evict_absent_blocks, 1)
        self.assertEqual(stats.hbm_evict_present_blocks, 0)
        self.assertEqual(stats.memory_write_blocks, 0)

    def test_rekey_calls_are_counted_apart_from_capacity_evictions(self):
        connector = self._connector("count")
        connector.hbm_evict_observer(self._block("a"), "rekey")
        stats = connector.mooncake_stats
        self.assertEqual(stats.hbm_evict_calls, 1)
        self.assertEqual(stats.hbm_evict_rekey_calls, 1)
        self.assertEqual(stats.hbm_evict_blocks, 0)
        self.assertEqual(stats.hbm_evict_bytes, 0)

    def test_on_evict_writes_a_key_the_store_does_not_have(self):
        connector = self._connector("on_evict")
        stats = connector.mooncake_stats
        connector.hbm_evict_observer(self._block("never-saved"), "capacity")
        self.assertEqual(stats.hbm_evict_absent_blocks, 1)
        self.assertEqual(stats.hbm_evict_puts, 1)
        self.assertEqual(stats.hbm_evict_puts_wrote_bytes, 1)
        self.assertEqual(stats.hbm_evict_puts_refresh_only, 0)
        self.assertEqual(stats.hbm_evict_write_bytes, 2 * connector.block_bytes)
        self.assertEqual(stats.memory_write_blocks, 1)
        self.assertEqual(stats.admission_count, 1)

    def test_on_evict_spends_no_bytes_on_a_key_the_store_already_has(self):
        connector = self._connector("on_evict")
        req = Request(0, 32, 4, 16, hash_ids=["a", "b"])
        self._save(connector, req)
        stats = connector.mooncake_stats
        before = (
            stats.memory_write_bytes,
            stats.ssd_write_bytes,
            stats.admission_count,
        )

        connector.hbm_evict_observer(self._block("a"), "capacity")

        self.assertEqual(stats.hbm_evict_present_blocks, 1)
        self.assertEqual(stats.hbm_evict_puts, 1)
        self.assertEqual(stats.hbm_evict_puts_refresh_only, 1)
        self.assertEqual(stats.hbm_evict_puts_wrote_bytes, 0)
        self.assertEqual(stats.hbm_evict_write_bytes, 0)
        self.assertEqual(stats.hbm_evict_save_latency, 0.0)
        self.assertEqual(
            (stats.memory_write_bytes, stats.ssd_write_bytes, stats.admission_count),
            before,
        )

    def test_unkeyed_block_is_counted_and_never_put(self):
        """A keyless block cannot reach the capacity site; pin that it is not
        silently given a key if some future change lets one through."""
        connector = self._connector("on_evict")
        stats = connector.mooncake_stats
        connector.hbm_evict_observer(self._block(None, number=3), "capacity")

        self.assertEqual(stats.hbm_evict_blocks, 1)
        self.assertEqual(stats.hbm_evict_unkeyed_blocks, 1)
        self.assertEqual(stats.hbm_evict_prompt_blocks, 0)
        self.assertEqual(stats.hbm_evict_puts, 0)
        self.assertEqual(stats.admission_count, 0)

    def test_eviction_save_latency_is_charged_to_wait_for_save_exactly_once(self):
        connector = self._connector(
            "on_evict",
            memory_capacity_blocks=1,
            admission_write_policy="write_through",
        )
        connector.bind_connector_metadata(
            type("Meta", (), {"saves": [], "loads": [], "preempted_request_ids": []})()
        )
        connector.hbm_evict_observer(self._block("never-saved"), "capacity")

        charged = connector.mooncake_stats.hbm_evict_save_latency
        self.assertGreater(charged, 0.0)
        self.assertAlmostEqual(connector.wait_for_save(), charged)
        self.assertEqual(connector.wait_for_save(), 0.0)


class HbmEvictSchedulerWiringTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def _scheduler(self, policy: str):
        connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 64,
                    "hbm_evict_save_policy": policy,
                },
            ),
            _CacheConfig(),
        )
        return connector, LLMPagedAttnScheduler(
            0,
            cache_config=_CacheConfig(),
            connector=connector,
        )

    def test_stock_policy_leaves_the_prefix_cache_unhooked(self):
        _, scheduler = self._scheduler("prefill")
        self.assertIsNone(scheduler.block_manager.kv_cache_manager.evict_observer)

    def test_enabled_policy_hooks_the_prefix_cache(self):
        connector, scheduler = self._scheduler("count")
        self.assertEqual(
            scheduler.block_manager.kv_cache_manager.evict_observer,
            connector.hbm_evict_observer,
        )

    def test_block_manager_without_a_connector_hook_stays_unhooked(self):
        manager = BlockManager(block_size=16, num_gpu_blocks=8, num_cpu_blocks=4)
        self.assertIsNone(manager.kv_cache_manager.evict_observer)


if __name__ == "__main__":
    unittest.main()
