"""store_lookup_start (PHASE 14): "zero" (stock) looks the prompt up in the
store from block 0; "after_hbm" starts at the first block not already hit in
HBM, per KVConnectorBase_V1.get_num_new_matched_tokens(request,
num_computed_tokens)."""

from __future__ import annotations

import unittest

from TokenSim.config.config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import MooncakeStoreConnector
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


def _connector(start: str):
    return MooncakeStoreConnector(
        KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "memory_capacity_blocks": 64,
                "enable_offload": True,
                "ssd_capacity_blocks": 256,
                "store_lookup_start": start,
            },
        ),
        _CacheConfig(),
    )


def _store_keys(connector, req, indices):
    keys = connector._keys_for_request(req)
    connector.service.store.put([keys[i] for i in indices])


# 64-token prompt = 4 full blocks a, b, c, d.
def _req(rid=1):
    return Request(rid, 64, 4, 16, hash_ids=["a", "b", "c", "d"])


class StoreLookupStartTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_default_is_zero_and_unknown_rejected(self):
        self.assertEqual(parse_mooncake_config({}).store_lookup_start, "zero")
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"store_lookup_start": "middle"})

    def test_after_hbm_reaches_a_stored_tail_behind_an_hbm_only_head(self):
        for start, expected in (("zero", 0), ("after_hbm", 32)):
            reset_mooncake_services()
            connector = _connector(start)
            req = _req()
            _store_keys(connector, req, [1, 2])          # b, c stored; a only in HBM
            got = connector.get_num_new_matched_tokens(req, 16)  # a hit in HBM
            self.assertEqual(got, expected, start)
        pending = connector._pending_loads[req.id]
        self.assertEqual(pending["blocks"], 2)
        keys = connector._keys_for_request(req)
        self.assertEqual(pending["keys"], [keys[1], keys[2]])

    def test_fully_stored_prefix_gives_the_same_span(self):
        results = {}
        for start in ("zero", "after_hbm"):
            reset_mooncake_services()
            connector = _connector(start)
            req = _req()
            _store_keys(connector, req, [0, 1, 2])       # a, b, c stored
            results[start] = (
                connector.get_num_new_matched_tokens(req, 16),
                connector._pending_loads[req.id]["blocks"],
            )
        self.assertEqual(results["zero"], results["after_hbm"])
        self.assertEqual(results["zero"], (32, 2))

    def test_full_prompt_hit_keeps_the_reference_cap(self):
        results = {}
        for start in ("zero", "after_hbm"):
            reset_mooncake_services()
            connector = _connector(start)
            req = _req()
            _store_keys(connector, req, [0, 1, 2, 3])
            results[start] = connector.get_num_new_matched_tokens(req, 16)
        self.assertEqual(results["zero"], results["after_hbm"])
        self.assertEqual(results["zero"], 32)  # (64 - 1) // 16 = 3 blocks, minus 1 local

    def test_after_hbm_skips_the_lookup_when_hbm_covers_the_prompt(self):
        connector = _connector("after_hbm")
        req = _req()
        self.assertEqual(connector.get_num_new_matched_tokens(req, 64), 0)
        self.assertEqual(connector.mooncake_stats.store_miss_count, 0)
        self.assertEqual(connector.mooncake_stats.store_hit_count, 0)


if __name__ == "__main__":
    unittest.main()
