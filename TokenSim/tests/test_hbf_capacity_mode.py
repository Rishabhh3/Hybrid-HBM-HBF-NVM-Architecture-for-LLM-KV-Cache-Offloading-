"""hbf_capacity_mode (PHASE 14): "finite" is stock; "unlimited" makes the
offload (HBF) tier never evict, as a sensitivity check on HBF capacity. Not a
hardware value."""

from __future__ import annotations

import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.store import MooncakeStore
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES


def _key(name: str, dp: int = 0) -> PoolKey:
    return PoolKey(KeyMetadata(model_name="M", dp_rank=dp), name)


def _store(mode: str | None, **extra) -> MooncakeStore:
    config = {
        "memory_capacity_blocks": 1,
        "enable_offload": True,
        "ssd_capacity_blocks": 2,
        "admission_write_policy": "write_back",
    }
    if mode is not None:
        config["hbf_capacity_mode"] = mode
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


def _strip(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if not k.startswith("hbf_capacity_") and k != "mooncake_offload_profiles"}


class HbfCapacityModeTest(unittest.TestCase):
    def test_default_is_finite_and_emits_nothing(self):
        self.assertEqual(parse_mooncake_config({}).hbf_capacity_mode, "finite")
        store = _store(None)
        for name in "abcd":
            store.put([_key(name)])
        stats = store.stats.as_dict()
        self.assertEqual([k for k in stats if k.startswith("hbf_capacity_")], [])
        self.assertNotIn("hbf_capacity_mode", stats["mooncake_offload_profiles"][0])

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"hbf_capacity_mode": "infinite"})

    def test_finite_evicts_unlimited_does_not(self):
        finite, unlimited = _store("finite"), _store("unlimited")
        for store in (finite, unlimited):
            for name in "abcde":  # a..d demoted to HBF; HBF holds 2
                store.put([_key(name)])
        self.assertEqual(finite.stats.ssd_eviction_count, 2)
        self.assertEqual(unlimited.stats.ssd_eviction_count, 0)
        self.assertEqual(unlimited.offload.used_blocks, 4)
        for name in "abcd":
            self.assertTrue(unlimited.offload.contains(_key(name).to_string()))
        stats = unlimited.stats.as_dict()
        self.assertEqual(stats["hbf_capacity_configured_blocks"], 2)
        self.assertEqual(stats["hbf_capacity_peak_used_blocks"], 4)
        self.assertEqual(stats["hbf_capacity_rank_peak_used_blocks_max"], 4)
        self.assertEqual(stats["mooncake_offload_profiles"][0]["hbf_capacity_mode"], "unlimited")
        # Same bytes written either way: eviction changes what is held, not the
        # write that caused it.
        self.assertEqual(finite.stats.ssd_write_blocks, unlimited.stats.ssd_write_blocks)

    def test_identical_when_capacity_never_binds(self):
        runs = []
        for mode in ("finite", "unlimited"):
            store = _store(mode, ssd_capacity_blocks=64)
            for name in "abcdab":
                store.put([_key(name)])
                hit = store.lookup([_key(name)])
                store.get(hit.keys, hit.tier)
            runs.append(_strip(store.stats.as_dict()))
        self.assertEqual(runs[0], runs[1])

    def test_per_rank_peaks(self):
        store = _store(
            "unlimited",
            memory_capacity_blocks=2,
            ssd_capacity_blocks=2,
            store_partition="per_rank",
            store_partition_ranks=2,
        )
        for name in "abcd":  # rank 0: 4 puts into 1 memory block -> 3 demoted
            store.put([_key(name, dp=0)])
        for name in "xy":    # rank 1: 2 puts -> 1 demoted
            store.put([_key(name, dp=1)])
        stats = store.stats.as_dict()
        self.assertEqual(stats["hbf_capacity_configured_blocks"], 2)
        self.assertEqual(stats["hbf_capacity_rank_peak_used_blocks_max"], 3)
        self.assertEqual(stats["hbf_capacity_peak_used_blocks"], 4)
        self.assertEqual(store.stats.ssd_eviction_count, 0)

    def test_export_prefix(self):
        self.assertIn("hbf_capacity_", OPTIONAL_MOONCAKE_KEY_PREFIXES)


if __name__ == "__main__":
    unittest.main()
