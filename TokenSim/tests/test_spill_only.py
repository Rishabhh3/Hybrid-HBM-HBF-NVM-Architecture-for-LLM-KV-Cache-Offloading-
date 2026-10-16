"""hbm_evict_save_policy = "spill_only" (PHASE 14 STEP 3b) and the end-of-run
HBM residue / admission-to-first-spill counters."""

from __future__ import annotations

import unittest

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.kv_cache_manager import KVCacheManager
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.kv_transfer import MooncakeStoreConnector
from TokenSim.kv_transfer.connectors.mooncake_store import _percentile
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.service import reset_mooncake_services


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()
    num_gpu_blocks = 8
    num_cpu_blocks = 4


def _connector(policy: str, **extra):
    config = {
        "memory_capacity_blocks": 64,
        "enable_offload": True,
        "ssd_capacity_blocks": 256,
        "hbm_evict_save_policy": policy,
    }
    config.update(extra)
    return MooncakeStoreConnector(
        KVTransferConfig(kv_connector="MooncakeStoreConnector", kv_connector_extra_config=config),
        _CacheConfig(),
    )


def _step(connector, req):
    connector.get_num_new_matched_tokens(req, 0)
    meta = connector.build_connector_meta(type("Output", (), {"scheduled": [req], "preempted": []})())
    connector.bind_connector_metadata(meta)
    return meta


def _key(name):
    return build_prefix_keys([name], model="TestModel")[0]


def _block(name, number=0):
    block = PhysicalTokenBlock(Device.GPU, number, 16)
    block.block_hash = _key(name)
    return block


class SpillOnlyTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_policy_parses(self):
        self.assertEqual(
            parse_mooncake_config({"hbm_evict_save_policy": "spill_only"}).hbm_evict_save_policy,
            "spill_only",
        )

    def test_prefill_save_is_off(self):
        connector = _connector("spill_only")
        meta = _step(connector, Request(0, 32, 4, 16, hash_ids=["a", "b"]))
        connector.wait_for_save()
        self.assertEqual(list(meta.saves), [])
        self.assertEqual(connector.mooncake_stats.admission_count, 0)
        self.assertEqual(connector.mooncake_stats.put_count, 0)

    def test_stock_prefill_still_saves(self):
        connector = _connector("prefill")
        meta = _step(connector, Request(0, 32, 4, 16, hash_ids=["a", "b"]))
        connector.wait_for_save()
        self.assertEqual(len(meta.saves), 1)
        self.assertEqual(connector.mooncake_stats.admission_count, 2)

    def test_eviction_writes_the_block(self):
        connector = _connector("spill_only")
        connector.hbm_evict_observer(_block("a"), "capacity")
        s = connector.mooncake_stats
        self.assertEqual(s.hbm_evict_absent_blocks, 1)
        self.assertEqual(s.hbm_evict_puts_wrote_bytes, 1)
        self.assertEqual(s.admission_count, 1)

    def test_rekey_is_not_a_spill(self):
        connector = _connector("spill_only")
        connector.hbm_evict_observer(_block("a"), "rekey")
        self.assertEqual(connector.mooncake_stats.admission_count, 0)


class ResidueAndFirstSpillTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_stock_attaches_nothing(self):
        connector = _connector("prefill")
        manager = KVCacheManager(block_size=16, model="TestModel")
        connector.attach_kv_cache_manager(manager)
        self.assertIsNone(manager.register_observer)
        connector.finalize_run_stats()
        self.assertEqual([k for k in connector.mooncake_stats.as_dict() if k.startswith("hbm_evict_")], [])

    def test_residue_and_first_spill_delay(self):
        connector = _connector("spill_only")
        manager = KVCacheManager(block_size=16, model="TestModel")
        connector.attach_kv_cache_manager(manager)
        manager.set_evict_observer(connector.hbm_evict_observer)
        blocks = [PhysicalTokenBlock(Device.GPU, i, 16) for i in range(3)]
        connector.set_simulation_time(1.0)
        manager.register_blocks(blocks, [_key("a"), _key("b"), _key("c")])
        connector.set_simulation_time(4.0)
        manager.evict(blocks[0])            # a: first spill after 3.0 s
        connector.set_simulation_time(9.0)
        manager.register_blocks([blocks[0]], [_key("a")])   # re-admitted
        connector.set_simulation_time(20.0)
        manager.evict(blocks[0])            # second spill of a: not counted again
        manager.evict(blocks[1])            # b: first spill after 19.0 s
        connector.finalize_run_stats()
        connector.finalize_run_stats()      # idempotent
        s = connector.mooncake_stats
        self.assertEqual(s.hbm_evict_first_spill_count, 2)
        self.assertEqual(s.hbm_evict_first_spill_p50_s, 3.0)
        self.assertEqual(s.hbm_evict_first_spill_p90_s, 19.0)
        # c is still resident and was never spilled, so the store lacks it.
        self.assertEqual(s.hbm_evict_residue_blocks, 1)
        self.assertEqual(s.hbm_evict_residue_absent_blocks, 1)
        exported = s.as_dict()
        self.assertEqual(exported["hbm_evict_residue_blocks"], 1)
        self.assertEqual(exported["hbm_evict_first_spill_p90_s"], 19.0)

    def test_percentile_is_nearest_rank(self):
        values = list(range(1, 11))
        self.assertEqual(_percentile(values, 0.5), 5)
        self.assertEqual(_percentile(values, 0.9), 9)
        self.assertEqual(_percentile([7.0], 0.9), 7.0)


if __name__ == "__main__":
    unittest.main()
