"""store_timing_mode and count_aligned_save_skips (PHASE 14 STEP 2).

`store_timing_mode = "charged"` is stock: store media time and interconnect
transfer time are computed from the configured knobs. `"free"` makes every one
of them exactly zero while leaving byte, block and hit accounting untouched.

`count_aligned_save_skips` is instrumentation only: it counts prefill saves the
aligned save rule skips because the same step loaded from the store.
"""

from __future__ import annotations

import unittest

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import MooncakeStoreConnector
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.service import reset_mooncake_services
from TokenSim.mooncake.store import MooncakeStore
from TokenSim.mooncake.pool_key import PoolKey, build_key_metadata
from TokenSim.mooncake.transfer_engine import Segment, TransferEngineSimulator
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()
    num_gpu_blocks = 8
    num_cpu_blocks = 4


def _keys(*names: str) -> list[PoolKey]:
    metadata = build_key_metadata(model_name="TestModel")
    return [PoolKey(metadata, name) for name in names]


def _store(mode: str, **extra) -> MooncakeStore:
    config = {
        "memory_capacity_blocks": 2,
        "enable_offload": True,
        "ssd_capacity_blocks": 64,
        "memory_media": "pcm",
        "admission_write_policy": "write_through",
        "charge_eviction_writes": True,
        "store_timing_mode": mode,
    }
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


class StoreTimingModeConfigTest(unittest.TestCase):
    def test_default_is_stock_charged(self):
        self.assertEqual(parse_mooncake_config({}).store_timing_mode, "charged")

    def test_supported_values_parse(self):
        for value in ("charged", "free"):
            config = parse_mooncake_config({"store_timing_mode": value})
            self.assertEqual(config.store_timing_mode, value)

    def test_unknown_value_is_rejected(self):
        with self.assertRaises(ConfigurationError) as ctx:
            parse_mooncake_config({"store_timing_mode": "scaled"})
        self.assertIn("store_timing_mode", str(ctx.exception))


class StoreTimingModeStoreTest(unittest.TestCase):
    def _exercise(self, store: MooncakeStore):
        """Writes that overflow memory (demotes), then a read of a demoted key."""
        put = store.put_with_timing(_keys("a", "b", "c", "d"), now=1.0)
        hit = store.lookup(_keys("a"), now=2.0)
        get = store.get(_keys("a"), hit.tier, now=2.0)
        return put, get

    def test_charged_mode_charges_every_media(self):
        put, get = self._exercise(_store("charged"))
        self.assertGreater(put.blocking_latency, 0.0)
        self.assertGreater(get, 0.0)

    def test_free_mode_charges_nothing_on_either_tier(self):
        store = _store("free")
        put, get = self._exercise(store)
        self.assertEqual(put.blocking_latency, 0.0)
        self.assertEqual(put.accounting_latency, 0.0)
        self.assertEqual(get, 0.0)
        self.assertEqual(store.stats.memory_write_latency, 0.0)
        self.assertEqual(store.stats.memory_read_latency, 0.0)

    def test_free_mode_leaves_every_byte_and_block_counter_unchanged(self):
        charged = _store("charged")
        free = _store("free")
        self._exercise(charged)
        self._exercise(free)
        # offload_profiles differs by design: it records the non-stock mode.
        timing_fields = {
            "memory_write_latency",
            "memory_read_latency",
            "offload_profiles",
        }
        for name in charged.stats.__dataclass_fields__:
            if name in timing_fields:
                continue
            self.assertEqual(
                getattr(charged.stats, name), getattr(free.stats, name), name
            )

    def test_free_mode_zeroes_offload_read_and_write(self):
        store = _store("free")
        self.assertEqual(store.offload.write("k", 3).latency, 0.0)
        self.assertEqual(store.offload.read("k").latency, 0.0)
        self.assertEqual(store.offload.read("k").bytes, 3 * 4096)

    def test_offload_profile_records_mode_only_when_not_stock(self):
        self.assertNotIn("store_timing_mode", _store("charged")._offload_profile())
        self.assertEqual(
            _store("free")._offload_profile()["store_timing_mode"], "free"
        )


class StoreTimingModeTransferTest(unittest.TestCase):
    def _transfer(self, mode: str):
        engine = TransferEngineSimulator(
            parse_mooncake_config({"store_timing_mode": mode})
        )
        return engine.create_transfer(
            source_segment=Segment(name="a", tier="vram"),
            target_segment=Segment(name="b", tier="dram"),
            blocks=4,
            bytes_=1 << 20,
            kind="store",
            topology="cross_node",
        )

    def test_charged_transfer_has_latency(self):
        self.assertGreater(self._transfer("charged").latency, 0.0)

    def test_free_transfer_has_zero_latency_and_same_bytes(self):
        charged = self._transfer("charged")
        free = self._transfer("free")
        self.assertEqual(free.latency, 0.0)
        self.assertEqual(free.bytes, charged.bytes)
        self.assertFalse(charged.free)


class StoreTimingModeConnectorTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    @staticmethod
    def _connector(**extra):
        config = {
            "memory_capacity_blocks": 64,
            "enable_offload": True,
            "ssd_capacity_blocks": 256,
            "memory_media": "pcm",
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
    def _step(connector, req):
        connector.get_num_new_matched_tokens(req, 0)
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req], "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        return connector.start_load_kv(), connector.wait_for_save()

    def test_free_mode_charges_no_load_or_save_time(self):
        charged = self._connector()
        self._step(charged, Request(0, 48, 4, 16, hash_ids=["a", "b", "c"]))
        charged_load, charged_save = self._step(
            charged, Request(1, 64, 4, 16, hash_ids=["a", "b", "x", "y"])
        )
        reset_mooncake_services()
        free = self._connector(store_timing_mode="free")
        _, free_first_save = self._step(
            free, Request(0, 48, 4, 16, hash_ids=["a", "b", "c"])
        )
        free_load, _ = self._step(
            free, Request(1, 64, 4, 16, hash_ids=["a", "b", "x", "y"])
        )
        self.assertGreater(charged_load, 0.0)
        self.assertEqual(free_load, 0.0)
        self.assertEqual(free_first_save, 0.0)
        self.assertEqual(free.mooncake_stats.save_wait_time, 0.0)
        self.assertEqual(free.mooncake_stats.load_wait_time, 0.0)
        self.assertEqual(
            free.mooncake_stats.transferred_bytes,
            charged.mooncake_stats.transferred_bytes,
        )


class AlignedSaveSkipCounterTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    @staticmethod
    def _connector(**extra):
        config = {
            "memory_capacity_blocks": 64,
            "enable_offload": True,
            "ssd_capacity_blocks": 256,
        }
        config.update(extra)
        return MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config=config,
            ),
            _CacheConfig(),
        )

    _step = staticmethod(StoreTimingModeConnectorTest._step)

    def test_default_is_off_and_emits_no_keys(self):
        connector = self._connector()
        self.assertFalse(connector.service.config.count_aligned_save_skips)
        self._step(connector, Request(0, 48, 4, 16, hash_ids=["a", "b", "c"]))
        self._step(connector, Request(1, 64, 4, 16, hash_ids=["a", "b", "x", "y"]))
        exported = connector.mooncake_stats.as_dict()
        self.assertEqual([k for k in exported if k.startswith("save_skip_")], [])
        self.assertEqual(connector.mooncake_stats.save_skip_requests, 0)

    def test_counts_a_skipped_save_and_the_blocks_it_would_have_written(self):
        connector = self._connector(count_aligned_save_skips=True)
        self._step(connector, Request(0, 48, 4, 16, hash_ids=["a", "b", "c"]))
        stats = connector.mooncake_stats
        self.assertEqual(stats.save_skip_requests, 0)
        before = (stats.memory_write_blocks, stats.ssd_write_blocks, stats.store_hit_count)

        # Loads a, b from the store, so its save -- x, y -- is skipped.
        self._step(connector, Request(1, 64, 4, 16, hash_ids=["a", "b", "x", "y"]))

        self.assertEqual(stats.save_skip_requests, 1)
        self.assertEqual(stats.save_skip_missing_blocks, 2)
        # Counting wrote nothing and added no hit beyond the load's own lookup.
        self.assertEqual(stats.memory_write_blocks, before[0])
        self.assertEqual(stats.ssd_write_blocks, before[1])
        self.assertEqual(stats.store_hit_count, before[2] + 1)
        exported = stats.as_dict()
        self.assertEqual(exported["save_skip_requests"], 1)
        self.assertEqual(exported["save_skip_missing_blocks"], 2)

    def test_counter_does_not_change_store_behaviour(self):
        runs = []
        for flag in (False, True):
            reset_mooncake_services()
            connector = self._connector(count_aligned_save_skips=flag)
            self._step(connector, Request(0, 48, 4, 16, hash_ids=["a", "b", "c"]))
            self._step(connector, Request(1, 64, 4, 16, hash_ids=["a", "b", "x", "y"]))
            self._step(connector, Request(2, 64, 4, 16, hash_ids=["x", "y", "z", "w"]))
            exported = connector.mooncake_stats.as_dict()
            runs.append(
                {
                    k: v
                    for k, v in exported.items()
                    if not k.startswith("save_skip_") and k != "mooncake_pool_keys"
                }
            )
        self.assertEqual(runs[0], runs[1])

    def test_export_merges_save_skip_keys(self):
        self.assertIn("save_skip_", OPTIONAL_MOONCAKE_KEY_PREFIXES)
        self.assertIn("hbm_evict_", OPTIONAL_MOONCAKE_KEY_PREFIXES)


if __name__ == "__main__":
    unittest.main()
