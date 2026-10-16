"""count_pcm_admit_source (PHASE 14): every memory-tier (PCM) admission
classified by why its key was absent from the store. Instrumentation only;
default off emits nothing and changes nothing."""

from __future__ import annotations

import unittest

from TokenSim.mooncake import parse_mooncake_config
from TokenSim.mooncake.pool_key import KeyMetadata, PoolKey
from TokenSim.mooncake.store import MooncakeStore
from util.results import OPTIONAL_MOONCAKE_KEY_PREFIXES

FIELDS = (
    "pcm_admit_first_blocks",
    "pcm_admit_after_if_read_drop_blocks",
    "pcm_admit_after_hbf_evict_blocks",
    "pcm_admit_after_other_blocks",
)


def _key(name: str) -> PoolKey:
    return PoolKey(KeyMetadata(model_name="M"), name)


def _store(flag: bool, **extra) -> MooncakeStore:
    config = {
        "memory_capacity_blocks": 1,
        "enable_offload": True,
        "ssd_capacity_blocks": 64,
        "admission_write_policy": "write_back",
        "count_pcm_admit_source": flag,
    }
    config.update(extra)
    return MooncakeStore(parse_mooncake_config(config), block_bytes=4096)


def _counts(store) -> tuple[int, ...]:
    return tuple(getattr(store.stats, name) for name in FIELDS)


class PcmAdmitSourceTest(unittest.TestCase):
    def test_default_off_emits_nothing(self):
        self.assertFalse(parse_mooncake_config({}).count_pcm_admit_source)
        store = _store(False)
        store.put([_key("a")])
        self.assertEqual([k for k in store.stats.as_dict() if k.startswith("pcm_admit_")], [])

    def test_first_admissions(self):
        store = _store(True)
        for name in "abc":
            store.put([_key(name)])
        self.assertEqual(_counts(store), (3, 0, 0, 0))

    def test_refresh_is_not_an_admission(self):
        store = _store(True, memory_capacity_blocks=4)
        store.put([_key("a")])
        store.put([_key("a")])  # key present: LRU refresh, no write
        self.assertEqual(_counts(store), (1, 0, 0, 0))
        self.assertEqual(store.stats.memory_write_blocks, 1)

    def test_readmission_after_if_read_drop(self):
        store = _store(True, demote_policy="if_read")
        store.put([_key("a")])
        store.put([_key("b")])  # a never read, write-back: dropped, no HBF copy
        store.put([_key("a")])  # a absent again
        self.assertEqual(_counts(store), (2, 1, 0, 0))

    def test_readmission_after_hbf_eviction(self):
        store = _store(True, ssd_capacity_blocks=1)
        store.put([_key("a")])
        store.put([_key("b")])  # a demoted to HBF
        store.put([_key("c")])  # b demoted; HBF evicts a, its only copy
        store.put([_key("a")])
        self.assertEqual(_counts(store), (3, 0, 1, 0))

    def test_hbf_served_block_is_not_readmitted(self):
        store = _store(True)
        store.put([_key("a")])
        store.put([_key("b")])  # a demoted to HBF
        hit = store.lookup([_key("a")])
        store.get(hit.keys, hit.tier)  # served from HBF, stays there
        store.put([_key("a")])  # key present: refresh only
        self.assertEqual(_counts(store), (2, 0, 0, 0))
        self.assertEqual(store.objects[_key("a").to_string()].tier, "ssd")

    def test_sums_to_memory_writes_and_changes_nothing(self):
        runs = []
        for flag in (False, True):
            store = _store(flag, demote_policy="if_read", ssd_capacity_blocks=2)
            for name in "abcabdacbe":
                store.put([_key(name)])
                if name in "ab":
                    hit = store.lookup([_key(name)])
                    store.get(hit.keys, hit.tier)
            if flag:
                self.assertEqual(sum(_counts(store)), store.stats.memory_write_blocks)
            runs.append({k: v for k, v in store.stats.as_dict().items() if not k.startswith("pcm_admit_")})
        self.assertEqual(runs[0], runs[1])

    def test_export_prefix(self):
        self.assertIn("pcm_admit_", OPTIONAL_MOONCAKE_KEY_PREFIXES)


if __name__ == "__main__":
    unittest.main()
