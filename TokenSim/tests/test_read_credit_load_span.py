"""read_credit, load_span and count_if_read_gate (PHASE 14 verification).

read_credit = "walked" (stock) credits every key a lookup walks over; "served"
credits a key only when its block is actually served by MooncakeStore.get.
load_span = "from_start" (stock) loads the first `blocks` keys of the lookup
result; "missing" loads exactly the span not already hit in HBM."""

from __future__ import annotations

import unittest

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import MooncakeStoreConnector
from TokenSim.llm.llm_request import Request
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.service import reset_mooncake_services
from TokenSim.mooncake.store import MooncakeStore
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()
    num_gpu_blocks = 8
    num_cpu_blocks = 4


def _key(name):
    return PoolKey(KeyMetadata(model_name="M"), name)


def _store(**extra):
    config = {
        "memory_capacity_blocks": 1,
        "enable_offload": True,
        "ssd_capacity_blocks": 64,
        "memory_media": "pcm",
        "admission_write_policy": "write_back",
        "demote_policy": "if_read",
    }
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


class ConfigTest(unittest.TestCase):
    def test_defaults_are_stock(self):
        config = parse_mooncake_config({})
        self.assertEqual(config.read_credit, "walked")
        self.assertEqual(config.load_span, "from_start")
        self.assertFalse(config.count_if_read_gate)

    def test_unknown_values_rejected(self):
        for extra in ({"read_credit": "touched"}, {"load_span": "all"}):
            with self.assertRaises(ConfigurationError):
                parse_mooncake_config(extra)


class ReadCreditTest(unittest.TestCase):
    def test_walked_credits_a_lookup_and_the_gate_opens(self):
        store = _store(count_if_read_gate=True)
        store.put([_key("a")])
        store.lookup([_key("a")])          # walked over, never served
        store.put([_key("b")])             # evicts a: if_read gate
        self.assertEqual(store.stats.blocks_demoted, 1)
        self.assertEqual(store.stats.if_read_gate_passes, 1)
        self.assertEqual(store.stats.if_read_gate_passes_pcm_served, 0)

    def test_served_does_not_credit_a_lookup_walk(self):
        store = _store(read_credit="served", count_if_read_gate=True)
        store.put([_key("a")])
        store.lookup([_key("a")])
        store.put([_key("b")])
        self.assertEqual(store.stats.blocks_demoted, 0)
        self.assertEqual(store.stats.blocks_dropped, 1)
        self.assertEqual(store.stats.if_read_gate_passes, 0)

    def test_served_credits_a_real_read(self):
        store = _store(read_credit="served", count_if_read_gate=True)
        store.put([_key("a")])
        hit = store.lookup([_key("a")])
        store.get(hit.keys, hit.tier)      # actually served from PCM
        store.put([_key("b")])
        self.assertEqual(store.stats.blocks_demoted, 1)
        self.assertEqual(store.stats.if_read_gate_passes, 1)
        self.assertEqual(store.stats.if_read_gate_passes_pcm_served, 1)

    def test_gate_counter_off_by_default_and_exported_when_on(self):
        store = _store()
        self.assertEqual([k for k in store.stats.as_dict() if k.startswith("if_read_gate_")], [])
        self.assertIn("if_read_gate_", OPTIONAL_MOONCAKE_KEY_PREFIXES)
        on = _store(count_if_read_gate=True)
        self.assertIn("if_read_gate_passes", on.stats.as_dict())


class LoadSpanTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def _plan_keys(self, span):
        reset_mooncake_services()
        connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 64,
                    "enable_offload": True,
                    "ssd_capacity_blocks": 256,
                    "load_span": span,
                },
            ),
            _CacheConfig(),
        )
        req = Request(1, 64, 4, 16, hash_ids=["a", "b", "c", "d"])
        keys = connector._keys_for_request(req)
        connector.service.store.put(keys[:3])           # a, b, c stored
        self.assertEqual(connector.get_num_new_matched_tokens(req, 16), 32)  # a in HBM
        plan = connector._build_load_plan(req, connector._pending_loads[req.id])
        return plan.extra["pool_keys"], plan.keys, keys

    def test_from_start_loads_the_head(self):
        pool_keys, key_strings, keys = self._plan_keys("from_start")
        self.assertEqual(pool_keys, [keys[0], keys[1]])

    def test_missing_loads_the_span_after_the_hbm_hit(self):
        pool_keys, key_strings, keys = self._plan_keys("missing")
        self.assertEqual(pool_keys, [keys[1], keys[2]])
        self.assertEqual(key_strings, [keys[1].to_string(), keys[2].to_string()])


if __name__ == "__main__":
    unittest.main()
