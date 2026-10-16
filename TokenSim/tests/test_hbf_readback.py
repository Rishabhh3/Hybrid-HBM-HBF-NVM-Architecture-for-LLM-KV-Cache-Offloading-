"""count_hbf_readback (PHASE 14): for every block written to the offload (HBF)
tier, whether it is read back from that tier before the run ends.
Instrumentation only; default off emits nothing and changes nothing."""

from __future__ import annotations

import unittest

from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.store import MooncakeStore
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES


def _key(name: str) -> PoolKey:
    return PoolKey(KeyMetadata(model_name="M"), name)


def _store(flag: bool, **extra) -> MooncakeStore:
    config = {
        "memory_capacity_blocks": 1,
        "enable_offload": True,
        "ssd_capacity_blocks": 64,
        "admission_write_policy": "write_back",
        "count_hbf_readback": flag,
    }
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


def _read(store, name):
    hit = store.lookup([_key(name)])
    store.get(hit.keys, hit.tier)


class HbfReadbackTest(unittest.TestCase):
    def test_default_off_emits_nothing(self):
        store = _store(False)
        store.put([_key("a")])
        store.put([_key("b")])
        self.assertEqual([k for k in store.stats.as_dict() if k.startswith("hbf_readback_")], [])
        self.assertFalse(parse_mooncake_config({}).count_hbf_readback)

    def test_counts_written_and_read_back(self):
        store = _store(True)
        store.put([_key("a")])
        store.put([_key("b")])  # demotes a to HBF
        store.put([_key("c")])  # demotes b to HBF
        _read(store, "a")       # a read back from HBF
        _read(store, "a")       # a second read is not a second instance
        s = store.stats
        self.assertEqual(s.hbf_readback_written_blocks, 2)
        self.assertEqual(s.hbf_readback_read_blocks, 1)
        self.assertEqual(s.as_dict()["hbf_readback_read_blocks"], 1)

    def test_memory_tier_reads_are_not_hbf_reads(self):
        store = _store(True, admission_write_policy="write_through", memory_capacity_blocks=4)
        store.put([_key("a")])  # write-through: HBF copy, memory copy
        _read(store, "a")       # served from memory
        self.assertEqual(store.stats.hbf_readback_written_blocks, 1)
        self.assertEqual(store.stats.hbf_readback_read_blocks, 0)

    def test_counter_changes_no_behaviour(self):
        runs = []
        for flag in (False, True):
            store = _store(flag)
            for name in "abcdab":
                store.put([_key(name)])
                _read(store, name)
            runs.append({k: v for k, v in store.stats.as_dict().items() if not k.startswith("hbf_readback_")})
        self.assertEqual(runs[0], runs[1])

    def test_export_prefix(self):
        self.assertIn("hbf_readback_", OPTIONAL_MOONCAKE_KEY_PREFIXES)


if __name__ == "__main__":
    unittest.main()
