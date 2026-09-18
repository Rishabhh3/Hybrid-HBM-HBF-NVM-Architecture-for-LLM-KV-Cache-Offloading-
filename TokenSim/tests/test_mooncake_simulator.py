from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import (
    KVConnectorFactory,
    KVConnectorMetadata,
    MooncakeConnector,
    MooncakeStoreConnector,
    MultiConnector,
)
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.mooncake import (
    KeyMetadata,
    MooncakeStats,
    MooncakeStore,
    PoolKey,
    Segment,
    TransferEngineSimulator,
    parse_mooncake_config,
    pool_keys_for_request,
)
from TokenSim.mooncake.service import reset_mooncake_services
from TokenSim.workload.loaders import load_json_pairs_workload
from util.request import LLMSource
from util.results import export_result, get_mooncake_stats


class _Roofline:
    def __init__(self):
        self.links = {
            "nvlink": type("Link", (), {"Latency": 1e-6, "UniBW": 300.0})(),
            "ethernet-test": type("Link", (), {"Latency": 1e-5, "UniBW": 12.5})(),
        }
        self.hardwares = {
            "TestGPU": type(
                "Hardware",
                (),
                {
                    "Name": "TestGPU",
                    "MM_Card_Num": 1,
                    "Capacity": 80,
                    "Nvlink": "nvlink",
                    "Pcie": "ethernet-test",
                    "MM_TFLOPS": 100,
                },
            )()
        }
        self.models = {
            "TestModel": type(
                "Model",
                (),
                {
                    "Name": "TestModel",
                    "Nlayer": 2,
                    "Dmodel": 128,
                    "Nhead": 8,
                    "FFN_Hidden": 256,
                },
            )()
        }

    def Compute_Timebreakdown_Iteration(self, *args, **kwargs):
        return 0.001, 0.0005


class _CacheConfig:
    block_size = 16
    size_per_token = 8
    model = "TestModel"
    rank_info = ParallelRankInfo()


def _psla() -> PSLAConfig:
    return PSLAConfig(
        name="test",
        model="TestModel",
        distribution="burst",
        prefill_mean_len=32,
        prefill_range_len=0,
        decode_mean_len=2,
        decode_range_len=0,
        decode_len_distribution="uniform",
        first_token_latency=MetricData(0, 0, 0),
        decode_token_latency=MetricData(0, 0, 0),
        qps=1,
    )


class MooncakeConfigTransferTest(unittest.TestCase):
    def test_config_files_parse_and_invalid_protocol_fails(self):
        for path in (
            "./data/kv_transfer/mooncake_p2p.json",
            "./data/kv_transfer/mooncake_store_embedded.json",
            "./data/kv_transfer/mooncake_store_standalone_ssd.json",
            "./data/kv_transfer/mooncake_multi_connector.json",
        ):
            config = KVTransferConfig.from_file(path)
            self.assertIn(config.kv_connector, {"MooncakeConnector", "MooncakeStoreConnector", "MultiConnector"})

        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"protocol": "bad"},
            )
        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"memory_capacity_blocks": -1},
            )
        with self.assertRaises(ConfigurationError):
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={"offload_tier": "bad"},
            )
    def test_transfer_engine_protocol_latency_and_overlap(self):
        config = parse_mooncake_config(
            {
                "protocol": "rdma",
                "num_nics": 2,
                "load_async": True,
                "transfer_overlap": True,
            }
        )
        engine = TransferEngineSimulator(config)
        rdma = engine.create_transfer(
            source_segment=Segment("a", "dram"),
            target_segment=Segment("b", "vram"),
            blocks=1,
            bytes_=1 << 20,
            protocol="rdma",
        )
        tcp = engine.create_transfer(
            source_segment=Segment("a", "dram"),
            target_segment=Segment("b", "vram"),
            blocks=1,
            bytes_=1 << 20,
            protocol="tcp",
        )
        job = engine.submit(rdma)

        self.assertLess(rdma.latency, tcp.latency)
        self.assertGreater(
            engine.create_transfer(
                source_segment=Segment("a", "dram"),
                target_segment=Segment("b", "vram"),
                blocks=1,
                bytes_=1 << 20,
                protocol="rdma",
                topology="cross_node",
            ).latency,
            rdma.latency,
        )
        self.assertEqual(job.blocking_latency, 0)
        self.assertEqual(engine.pending_jobs(0), 1)

    def test_json_pairs_loader_preserves_hash_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reuse.json"
            path.write_text(
                json.dumps(
                    [[32, 2, {"hash_ids": ["a", "b"], "reuse_group": "tenant"}]]
                )
            )
            workload = load_json_pairs_workload(str(path), request_count=1)

        self.assertEqual(workload[0].hash_ids, ["a", "b"])
        self.assertEqual(workload[0].reuse_group, "tenant")


class MooncakeStoreModelTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_pool_key_includes_rank_context(self):
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"], reuse_group="tenant")
        rank0 = ParallelRankInfo(tp_rank=0, pp_rank=0, dp_rank=0, kv_cache_group_id="dp0-pp0")
        rank1 = ParallelRankInfo(tp_rank=1, pp_rank=0, dp_rank=0, kv_cache_group_id="dp0-pp0")

        key0 = pool_keys_for_request(req, model_name="m", rank_info=rank0, engine_id="e")[0]
        key1 = pool_keys_for_request(req, model_name="m", rank_info=rank1, engine_id="e")[0]

        self.assertNotEqual(key0, key1)
        self.assertIn("@tp_rank:0", key0.to_string())
        self.assertIn("@tp_rank:1", key1.to_string())

    def test_store_memory_hit_eviction_and_ssd_hit(self):
        config = parse_mooncake_config(
            {
                "memory_capacity_blocks": 1,
                "enable_offload": True,
                "ssd_capacity_blocks": 4,
            }
        )
        store = MooncakeStore(config, block_bytes=128)
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put([key0])
        self.assertEqual(store.lookup([key0]).tier, "memory")
        store.put([key1])
        hit = store.lookup([key0])

        self.assertEqual(hit.tier, "ssd")
        self.assertGreater(store.get(hit.keys, hit.tier), 0)
        self.assertGreater(store.stats.ssd_read_blocks, 0)
        self.assertGreater(store.stats.ssd_write_blocks, 0)
        self.assertEqual(store.objects[key0.to_string()].replicas, 1)
        self.assertTrue(store.objects[key0.to_string()].persisted)

    def test_ssd_media_writes_are_blocking(self):
        block_bytes = 10 * (1 << 20)
        ssd_store = MooncakeStore(
            parse_mooncake_config(
                {
                    "memory_capacity_blocks": 0,
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_blocks": 4,
                }
            ),
            block_bytes=block_bytes,
        )
        key = PoolKey(KeyMetadata("m"), "ssd-blocking")
        timing = ssd_store.put_with_timing([key], now=1.0)
        expected = 200e-6 + block_bytes / (1 << 30) / 3.0
        self.assertAlmostEqual(timing.accounting_latency, expected)
        self.assertAlmostEqual(timing.blocking_latency, expected)

    def test_ssd_offload_profile_exports_media_parameters(self):
        ssd_store = MooncakeStore(
            parse_mooncake_config(
                {
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_gb": 2048,
                }
            ),
            block_bytes=4096,
        )
        ssd_profile = ssd_store.stats.as_dict()["mooncake_offload_profiles"][0]
        self.assertEqual(ssd_profile["tier"], "ssd")
        self.assertEqual(ssd_profile["ssd_capacity_gb"], 2048)
        self.assertEqual(ssd_profile["ssd_read_bw_gbps"], 7.0)
        self.assertEqual(ssd_profile["ssd_write_bw_gbps"], 3.0)
        self.assertTrue(ssd_profile["write_blocking"])
        self.assertEqual(ssd_profile["write_persistence"], "blocking")

    def test_store_rejects_when_no_eviction_or_admission_disabled(self):
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")
        no_evict = MooncakeStore(
            parse_mooncake_config(
                {
                    "memory_capacity_blocks": 1,
                    "eviction_policy": "none",
                }
            ),
            block_bytes=128,
        )

        no_evict.put([key0])
        no_evict.put([key1])

        self.assertEqual(no_evict.stats.admission_count, 1)
        self.assertEqual(no_evict.stats.admission_rejection_count, 1)

        never = MooncakeStore(
            parse_mooncake_config({"admission_policy": "never"}),
            block_bytes=128,
        )
        never.put([key0])
        self.assertEqual(never.stats.admission_count, 0)
        self.assertEqual(never.stats.admission_rejection_count, 1)


class MooncakeConnectorTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()

    def test_factory_creates_mooncake_connectors(self):
        p2p = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeConnector"),
            _CacheConfig(),
        )
        store = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeStoreConnector"),
            _CacheConfig(),
        )

        self.assertIsInstance(p2p, MooncakeConnector)
        self.assertIsInstance(store, MooncakeStoreConnector)

    def test_mooncake_p2p_records_transfer(self):
        connector = MooncakeConnector(
            KVTransferConfig(kv_connector="MooncakeConnector"),
            _CacheConfig(),
        )
        req = Request(0, 16, 1, 16)
        req._physical_token_blocks = [object(), object()]
        plan = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(KVConnectorMetadata(connector_name="MooncakeConnector", loads=[plan]))

        latency = connector.start_load_kv()

        self.assertGreater(latency, 0)
        self.assertGreater(connector.stats.transfer_count, 0)
        self.assertGreater(connector.mooncake_stats.p2p_latency, 0)
        self.assertEqual(plan.extra["protocol"], "rdma")
        self.assertEqual(plan.extra["topology"], "cross_node")
        self.assertTrue(plan.keys)

    def test_multi_connector_delegates_p2p_transfer_plan(self):
        connector = KVConnectorFactory.create_connector(
            KVTransferConfig.from_file("./data/kv_transfer/mooncake_multi_connector.json"),
            _CacheConfig(),
        )
        req = Request(0, 16, 1, 16)
        req._physical_token_blocks = [object()]

        plan = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(
            KVConnectorMetadata(connector_name="MultiConnector", loads=[plan])
        )
        latency = connector.start_load_kv()

        self.assertIsInstance(connector, MultiConnector)
        self.assertGreater(latency, 0)
        self.assertEqual(plan.extra["connector"], "MooncakeConnector")
        self.assertGreater(connector.children[0].stats.transfer_count, 0)

    def test_store_connector_memory_and_disk_hits(self):
        memory_connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 8,
                    "enable_offload": False,
                },
            ),
            _CacheConfig(),
        )
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        meta0 = memory_connector.build_connector_meta(
            type("Output", (), {"scheduled": [req0], "preempted": []})()
        )
        memory_connector.bind_connector_metadata(meta0)
        memory_connector.wait_for_save()
        hit_tokens = memory_connector.get_num_new_matched_tokens(req1, 0)
        memory_connector.update_state_after_alloc(
            req1,
            [
                type("Block", (), {"block_number": 1})(),
                type("Block", (), {"block_number": 2})(),
            ],
            hit_tokens,
        )
        meta1 = memory_connector.build_connector_meta(
            type("Output", (), {"scheduled": [req1], "preempted": []})()
        )

        # Aligned mooncake semantics: a full-prompt hit leaves the trailing
        # block uncomputed-from-cache, so 2 hit blocks are capped to 1.
        self.assertEqual(hit_tokens, 16)
        self.assertEqual(meta1.loads[0].tier, "memory")
        self.assertEqual(meta1.loads[0].blocks, 1)
        self.assertTrue(meta1.loads[0].keys)
        self.assertEqual(
            meta1.loads[0].keys,
            [key.to_string() for key in memory_connector._keys_for_request(req1)][
                : meta1.loads[0].blocks
            ],
        )

        reset_mooncake_services()
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "memory_capacity_blocks": 1,
                "enable_offload": True,
                "ssd_capacity_blocks": 8,
            },
        )
        connector = MooncakeStoreConnector(config, _CacheConfig())
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        req2 = Request(2, 32, 1, 16, hash_ids=["c", "d"])

        meta0 = connector.build_connector_meta(type("Output", (), {"scheduled": [req0], "preempted": []})())
        connector.bind_connector_metadata(meta0)
        connector.wait_for_save()
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)
        self.assertEqual(hit_tokens, 16)
        meta1 = connector.build_connector_meta(type("Output", (), {"scheduled": [req1], "preempted": []})())
        self.assertEqual(meta1.loads[0].tier, "ssd")
        self.assertEqual(meta1.loads[0].blocks, 1)
        self.assertTrue(meta1.loads[0].keys)
        connector.bind_connector_metadata(meta1)
        self.assertGreater(connector.start_load_kv(), 0)

        meta2 = connector.build_connector_meta(type("Output", (), {"scheduled": [req2], "preempted": []})())
        connector.bind_connector_metadata(meta2)
        connector.wait_for_save()
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)

        self.assertEqual(hit_tokens, 16)
        self.assertEqual(connector.service.store.lookup(connector._keys_for_request(req1)).tier, "ssd")
        self.assertGreater(connector.mooncake_stats.disk_tier_hit_count, 0)

    def test_ssd_saves_are_blocking(self):
        connector = MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config={
                    "memory_capacity_blocks": 0,
                    "enable_offload": True,
                    "offload_tier": "ssd",
                    "ssd_capacity_blocks": 8,
                },
            ),
            _CacheConfig(),
        )
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req], "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        connector.set_simulation_time(5.0)

        latency = connector.wait_for_save()

        self.assertGreater(latency, 0.0)
        self.assertAlmostEqual(connector.stats.save_wait_time, latency)
        self.assertAlmostEqual(connector.mooncake_stats.save_wait_time, latency)

    def test_store_connector_delays_release_until_async_feedback(self):
        config = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_connector_extra_config={
                "load_async": True,
                "transfer_overlap": True,
            },
        )
        connector = MooncakeStoreConnector(config, _CacheConfig())
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=type(
                "Cache",
                (),
                {
                    "block_size": 16,
                    "num_gpu_blocks": 8,
                    "num_cpu_blocks": 8,
                    "model": "TestModel",
                },
            )(),
            max_parallem_sum=4,
            connector=connector,
        )
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        scheduler.add_requests([req])
        running, _ = scheduler.schedule()
        meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": running, "preempted": []})()
        )
        connector.bind_connector_metadata(meta)
        req.generation_idx = req.decode_len

        scheduler.free(req)

        self.assertEqual(req.status.name, "WAITING_FOR_CONNECTOR_FREE")
        self.assertEqual(scheduler.block_manager.block_table.get_num_blocks(req.id), 2)

        connector.wait_for_save()
        worker_meta = connector.build_connector_worker_meta()
        scheduler.update_connector_output(worker_meta)

        self.assertEqual(req.status.name, "FINISHED_STOPPED")
        self.assertEqual(scheduler.block_manager.block_table.get_num_blocks(req.id), 0)


class MooncakeAlignedSemanticsTest(unittest.TestCase):
    """Pin the reference-aligned save semantics (vLLM MooncakeStoreConnector).

    Reference: ref/vllm .../kv_connector/v1/mooncake/store/{scheduler,data,worker}.py
    - each request saves at most once, during prefill; decode steps never save
    - a step that loads from the store skips its save
    - the save existence-prefilter transfers only keys missing from the store
    """

    def setUp(self):
        reset_mooncake_services()

    def _connector(self, **extra):
        config = {"memory_capacity_blocks": 64, "enable_offload": False}
        config.update(extra)
        return MooncakeStoreConnector(
            KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_connector_extra_config=config,
            ),
            _CacheConfig(),
        )

    @staticmethod
    def _meta(connector, requests):
        return connector.build_connector_meta(
            type("Output", (), {"scheduled": requests, "preempted": []})()
        )

    def test_save_happens_once_during_prefill_and_never_on_decode(self):
        connector = self._connector()
        req = Request(0, 32, 4, 16, hash_ids=["a", "b"])

        prefill_meta = self._meta(connector, [req])
        self.assertEqual(len(prefill_meta.saves), 1)
        self.assertEqual(prefill_meta.saves[0].blocks, 2)
        connector.bind_connector_metadata(prefill_meta)
        connector.wait_for_save()

        req.generation_idx = 1  # decode steps
        for _ in range(3):
            decode_meta = self._meta(connector, [req])
            self.assertEqual(decode_meta.saves, [])
        self.assertEqual(connector.mooncake_stats.put_count, 1)

    def test_load_step_precludes_save(self):
        connector = self._connector()
        req0 = Request(0, 48, 1, 16, hash_ids=["a", "b", "c"])
        connector.bind_connector_metadata(self._meta(connector, [req0]))
        connector.wait_for_save()

        req1 = Request(1, 48, 1, 16, hash_ids=["a", "b", "c"])
        hit_tokens = connector.get_num_new_matched_tokens(req1, 0)
        # Full-prompt hit capped to leave the trailing block uncomputed.
        self.assertEqual(hit_tokens, 32)
        meta = self._meta(connector, [req1])
        self.assertEqual(len(meta.loads), 1)
        self.assertEqual(meta.saves, [])
        # Decode steps do not retroactively save either.
        req1.generation_idx = 1
        self.assertEqual(self._meta(connector, [req1]).saves, [])

    def test_save_transfers_only_missing_blocks(self):
        connector = self._connector()
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        connector.bind_connector_metadata(self._meta(connector, [req0]))
        connector.wait_for_save()

        # Shares the (a, b) prefix chain; only (c, d) are missing.
        req1 = Request(1, 64, 1, 16, hash_ids=["a", "b", "c", "d"])
        meta = self._meta(connector, [req1])
        self.assertEqual(len(meta.saves), 1)
        self.assertEqual(meta.saves[0].blocks, 2)
        self.assertEqual(
            meta.saves[0].keys,
            connector._key_strings_for_request(req1)[2:4],
        )

        # A request whose prompt is fully present produces no save plan.
        req2 = Request(2, 32, 1, 16, hash_ids=["a", "b"])
        self.assertEqual(self._meta(connector, [req2]).saves, [])

    def test_every_step_legacy_policy_resaves_full_prompt(self):
        connector = self._connector(save_policy="every_step")
        req = Request(0, 32, 2, 16, hash_ids=["a", "b"])
        first = self._meta(connector, [req])
        self.assertEqual(len(first.saves), 1)
        self.assertEqual(first.saves[0].blocks, 2)
        req.generation_idx = 1
        second = self._meta(connector, [req])
        self.assertEqual(len(second.saves), 1)
        self.assertEqual(second.saves[0].blocks, 2)

    def test_on_request_released_clears_prefill_side_state(self):
        connector = self._connector()
        req = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        connector.bind_connector_metadata(self._meta(connector, [req]))
        connector.wait_for_save()
        self.assertIn(req.id, connector._keys_cache)

        connector.on_request_released(req)

        self.assertNotIn(req.id, connector._keys_cache)
        self.assertNotIn(req.id, connector._request_keys)
        self.assertNotIn(req.id, connector._saved_request_ids)


class MooncakeEndToEndTest(unittest.TestCase):
    def setUp(self):
        reset_mooncake_services()
        reset_g_time()

    def test_engine_exports_mooncake_store_metrics(self):
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=1,
                    network="net1",
                )
            ],
        )
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.from_file(
                "./data/kv_transfer/mooncake_store_embedded.json"
            ),
            psla_config=_psla(),
            cluster_config=cluster,
            roofline=_Roofline(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
        requests = [
            Request(0, 32, 2, 16, inter_arrival_time=0.05, hash_ids=["a", "b"]),
            Request(1, 32, 2, 16, hash_ids=["a", "b"]),
        ]
        env.process(LLMSource(env, engine, requests, qps=1000, distribution="uniform"))

        def stop_when_done():
            while env.now < 1:
                if all(req.is_done for req in requests):
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return env.now
                yield env.timeout(1e-6)
            raise AssertionError("Mooncake requests did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        stats = get_mooncake_stats(engine)

        self.assertGreater(stats["mooncake_put_count"], 0)
        self.assertGreater(stats["mooncake_memory_tier_hit_count"], 0)
        self.assertGreater(stats["mooncake_transferred_bytes"], 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_result(
                args=argparse.Namespace(
                    qps=1,
                    batching="paged-attn",
                    results_path=tmpdir,
                    cluster="unused.json",
                ),
                g_time=g_time,
                engine=engine,
                model_config=_psla(),
                cluster=cluster,
                request_count=2,
                prefill_lens=[32, 32],
                decode_lens=[2, 2],
                requests=requests,
                notdone=[],
                duration=monitor.value,
                simulator_wall_time=0.1,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())
        self.assertGreater(result["mooncake_memory_tier_hit_count"], 0)
        self.assertGreater(result["mooncake_transferred_bytes"], 0)

    def test_non_mooncake_export_has_zero_defaults(self):
        env = simpy.Environment()
        cluster = ClusterConfig(
            num_workers=1,
            networks={"net1": "ethernet-test"},
            worker_groups=[
                WorkerGroupConfig(
                    role="hybrid",
                    hardware="TestGPU",
                    num_workers=1,
                    network="net1",
                )
            ],
        )
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=_psla(),
            cluster_config=cluster,
            roofline=_Roofline(),
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            max_occupy_ratio=1,
        )
        request = Request(0, 32, 2, 16)
        request.arrive(env)
        request.step(env, 0.01, 1)
        request.step(env, 0.01, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_result(
                args=argparse.Namespace(
                    qps=1,
                    batching="paged-attn",
                    results_path=tmpdir,
                    cluster="unused.json",
                ),
                g_time=g_time,
                engine=engine,
                model_config=_psla(),
                cluster=cluster,
                request_count=1,
                prefill_lens=[32],
                decode_lens=[2],
                requests=[request],
                notdone=[],
                duration=0.1,
                simulator_wall_time=0.01,
            )
            result = json.loads((Path(tmpdir) / "result_1.json").read_text())

        self.assertEqual(result["mooncake_get_count"], 0)
        self.assertEqual(result["mooncake_transferred_bytes"], 0)

    def test_multi_connector_store_and_p2p_metrics_aggregate(self):
        connector = KVConnectorFactory.create_connector(
            KVTransferConfig.from_file("./data/kv_transfer/mooncake_multi_connector.json"),
            _CacheConfig(),
        )
        store = connector.children[1]
        req0 = Request(0, 32, 1, 16, hash_ids=["a", "b"])
        req1 = Request(1, 32, 1, 16, hash_ids=["a", "b"])
        req0._physical_token_blocks = [object(), object()]
        req1._physical_token_blocks = [object(), object()]

        save_meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req0], "preempted": []})()
        )
        connector.bind_connector_metadata(save_meta)
        connector.wait_for_save()
        self.assertEqual(connector.get_num_new_matched_tokens(req1, 0), 16)
        load_meta = connector.build_connector_meta(
            type("Output", (), {"scheduled": [req1], "preempted": []})()
        )
        connector.bind_connector_metadata(load_meta)
        connector.start_load_kv()
        p2p_plan = connector.build_transfer_plan(
            requests=[req1],
            source_worker_id=0,
            target_worker_id=1,
            kind="load",
            latency=None,
        )
        connector.bind_connector_metadata(
            type("Meta", (), {"loads": [p2p_plan], "saves": [], "preempted_request_ids": []})()
        )
        connector.start_load_kv()

        self.assertGreater(store.mooncake_stats.memory_tier_hit_count, 0)
        self.assertGreater(connector.children[0].mooncake_stats.p2p_latency, 0)


class MooncakeOffloadVictimTest(unittest.TestCase):
    """Offload-tier victims must not outlive their data in ``store.objects``."""

    BLOCK_BYTES = 1 << 20

    def _store(self, **overrides):
        config = {
            "enable_offload": True,
            "offload_tier": "ssd",
            "ssd_capacity_blocks": 2,
        }
        config.update(overrides)
        return MooncakeStore(parse_mooncake_config(config), block_bytes=self.BLOCK_BYTES)

    def _write_latency(self, blocks: int = 1) -> float:
        return 200e-6 + (blocks * self.BLOCK_BYTES) / (1 << 30) / 3.0

    def test_put_path_ssd_eviction_drops_victim_from_objects(self):
        # memory_capacity_blocks=0 forces every put down the direct-to-SSD
        # admission path, so the eviction below comes from a put and not from
        # a memory-tier eviction.
        store = self._store(memory_capacity_blocks=0)
        keys = [PoolKey(KeyMetadata("m"), name) for name in ("a", "b", "c")]

        store.put([keys[0]])
        store.put([keys[1]])
        self.assertEqual(store.objects[keys[0].to_string()].tier, "ssd")

        # Third put exceeds ssd_capacity_blocks=2 and evicts key "a".
        store.put([keys[2]])

        self.assertNotIn(keys[0].to_string(), store.objects)
        self.assertFalse(store.offload.contains(keys[0].to_string()))
        self.assertIn(keys[1].to_string(), store.objects)
        self.assertIn(keys[2].to_string(), store.objects)
        self.assertEqual(store.stats.ssd_eviction_count, 1)
        self.assertEqual(store.stats.memory_eviction_count, 0)

    def test_dropped_victim_can_be_stored_again(self):
        # A stale entry made _put_one treat the key as already stored, so it
        # could never be written back; dropping it restores that path.
        store = self._store(memory_capacity_blocks=0)
        keys = [PoolKey(KeyMetadata("m"), name) for name in ("a", "b", "c")]
        for key in keys:
            store.put([key])
        self.assertNotIn(keys[0].to_string(), store.objects)

        blocks_before = store.stats.ssd_write_blocks
        store.put([keys[0]])

        self.assertIn(keys[0].to_string(), store.objects)
        self.assertTrue(store.offload.contains(keys[0].to_string()))
        self.assertGreater(store.stats.ssd_write_blocks, blocks_before)
        self.assertEqual(store.lookup([keys[0]]).hit_blocks, 1)

    def test_memory_resident_object_survives_offload_victimisation(self):
        # An object written through to SSD while still memory-resident keeps
        # serving from memory after the offload tier drops its backup copy.
        store = self._store(memory_capacity_blocks=8, ssd_capacity_blocks=1)
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put([key0])
        store.put([key1])

        self.assertEqual(store.objects[key0.to_string()].tier, "memory")
        self.assertIn(key0.to_string(), store.objects)
        self.assertFalse(store.offload.contains(key0.to_string()))
        self.assertEqual(store.lookup([key0]).tier, "memory")


class MooncakeEvictionAccountingTest(unittest.TestCase):
    """charge_eviction_writes and the memory/SSD eviction counter split."""

    BLOCK_BYTES = 1 << 20

    def _store(self, **overrides):
        config = {
            "enable_offload": True,
            "offload_tier": "ssd",
            "memory_capacity_blocks": 1,
            "ssd_capacity_blocks": 8,
        }
        config.update(overrides)
        return MooncakeStore(parse_mooncake_config(config), block_bytes=self.BLOCK_BYTES)

    def _write_latency(self) -> float:
        return 200e-6 + self.BLOCK_BYTES / (1 << 30) / 3.0

    def test_charge_eviction_writes_defaults_to_false(self):
        self.assertFalse(parse_mooncake_config({}).charge_eviction_writes)

    def test_eviction_write_is_free_when_not_charged(self):
        store = self._store(charge_eviction_writes=False)
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put_with_timing([key0])
        # This put evicts key0 to SSD and then writes key1 through; only the
        # latter is charged.
        timing = store.put_with_timing([key1])

        self.assertEqual(store.stats.memory_eviction_count, 1)
        self.assertAlmostEqual(timing.accounting_latency, self._write_latency())
        self.assertAlmostEqual(timing.blocking_latency, self._write_latency())

    def test_eviction_write_is_charged_when_enabled(self):
        store = self._store(charge_eviction_writes=True)
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put_with_timing([key0])
        timing = store.put_with_timing([key1])

        self.assertEqual(store.stats.memory_eviction_count, 1)
        # Eviction write for key0 plus the write-through for key1.
        self.assertAlmostEqual(timing.accounting_latency, 2 * self._write_latency())
        self.assertAlmostEqual(timing.blocking_latency, 2 * self._write_latency())

    def test_charging_does_not_change_eviction_counts(self):
        counts = {}
        for flag in (False, True):
            store = self._store(charge_eviction_writes=flag)
            for name in ("a", "b", "c"):
                store.put([PoolKey(KeyMetadata("m"), name)])
            counts[flag] = (
                store.stats.memory_eviction_count,
                store.stats.ssd_eviction_count,
            )
        self.assertEqual(counts[False], counts[True])

    def test_eviction_counters_split_by_tier(self):
        store = self._store()
        key0 = PoolKey(KeyMetadata("m"), "a")
        key1 = PoolKey(KeyMetadata("m"), "b")

        store.put([key0])
        store.put([key1])

        # One memory eviction, no SSD pressure yet (ssd_capacity_blocks=8).
        self.assertEqual(store.stats.memory_eviction_count, 1)
        self.assertEqual(store.stats.ssd_eviction_count, 0)
        self.assertEqual(store.stats.eviction_count, 1)

    def test_memory_eviction_forcing_ssd_victim_counts_each_once(self):
        # The conflation case: a single put evicts from memory AND pushes a
        # distinct older object out of the SSD tier. Each tier counts once.
        store = self._store(ssd_capacity_blocks=2)
        keys = [PoolKey(KeyMetadata("m"), name) for name in ("a", "b", "c")]

        store.put([keys[0]])
        store.put([keys[1]])
        before = (
            store.stats.memory_eviction_count,
            store.stats.ssd_eviction_count,
        )

        store.put([keys[2]])

        memory_delta = store.stats.memory_eviction_count - before[0]
        ssd_delta = store.stats.ssd_eviction_count - before[1]
        self.assertEqual(memory_delta, 1)
        self.assertEqual(ssd_delta, 1)
        # The victim is gone from both the tier and the object table.
        self.assertNotIn(keys[0].to_string(), store.objects)
        self.assertFalse(store.offload.contains(keys[0].to_string()))

    def test_eviction_count_is_the_sum_of_its_parts(self):
        store = self._store(ssd_capacity_blocks=2)
        for name in ("a", "b", "c", "d"):
            store.put([PoolKey(KeyMetadata("m"), name)])

        stats = store.stats
        self.assertGreater(stats.memory_eviction_count, 0)
        self.assertGreater(stats.ssd_eviction_count, 0)
        self.assertEqual(
            stats.eviction_count,
            stats.memory_eviction_count + stats.ssd_eviction_count,
        )
        exported = stats.as_dict()
        self.assertEqual(
            exported["mooncake_eviction_count"],
            exported["mooncake_memory_eviction_count"]
            + exported["mooncake_ssd_eviction_count"],
        )

    def test_aggregate_sums_split_counters(self):
        left = MooncakeStats(memory_eviction_count=3, ssd_eviction_count=2)
        right = MooncakeStats(memory_eviction_count=4, ssd_eviction_count=5)

        merged = left.aggregate(right)

        self.assertEqual(merged.memory_eviction_count, 7)
        self.assertEqual(merged.ssd_eviction_count, 7)
        self.assertEqual(merged.eviction_count, 14)


class MooncakeDemotePolicyTest(unittest.TestCase):
    """read_count/write_time instrumentation and the demote_policy drop path."""

    BLOCK_BYTES = 1 << 20

    def _store(self, **overrides):
        config = {
            "enable_offload": True,
            "offload_tier": "ssd",
            "memory_capacity_blocks": 2,
            "ssd_capacity_blocks": 64,
        }
        config.update(overrides)
        return MooncakeStore(parse_mooncake_config(config), block_bytes=self.BLOCK_BYTES)

    @staticmethod
    def _key(name: str) -> PoolKey:
        return PoolKey(KeyMetadata("m"), name)

    # -- instrumentation ---------------------------------------------------

    def test_admission_stamps_write_time_and_zero_read_count(self):
        store = self._store()
        key = self._key("a")

        store.put([key], now=12.5)

        obj = store.objects[key.to_string()]
        self.assertEqual(obj.read_count, 0)
        self.assertEqual(obj.write_time, 12.5)

    def test_lookup_increments_read_count_per_object(self):
        store = self._store()
        keys = [self._key("a"), self._key("b")]
        store.put(keys, now=0.0)

        # One lookup spanning both keys must credit each object once, not
        # bump a single global counter twice.
        store.lookup(keys, now=1.0)
        store.lookup(keys[:1], now=2.0)

        self.assertEqual(store.objects[keys[0].to_string()].read_count, 2)
        self.assertEqual(store.objects[keys[1].to_string()].read_count, 1)

    def test_missed_lookup_does_not_count_as_a_read(self):
        store = self._store()
        store.put([self._key("a")], now=0.0)

        store.lookup([self._key("zzz")], now=1.0)

        self.assertEqual(store.objects[self._key("a").to_string()].read_count, 0)

    def test_wasted_and_useful_split_at_memory_eviction(self):
        store = self._store(memory_capacity_blocks=2)
        read_key, cold_key = self._key("hot"), self._key("cold")
        store.put([read_key], now=0.0)
        store.put([cold_key], now=0.0)
        store.lookup([read_key], now=1.0)

        # Two more admissions evict both originals from memory.
        store.put([self._key("x")], now=2.0)
        store.put([self._key("y")], now=3.0)

        self.assertEqual(store.stats.memory_eviction_count, 2)
        self.assertEqual(store.stats.useful_write_blocks, 1)
        self.assertEqual(store.stats.wasted_write_blocks, 1)

    def test_residency_is_now_minus_write_time(self):
        store = self._store(memory_capacity_blocks=1)
        store.put([self._key("a")], now=10.0)
        store.put([self._key("b")], now=35.0)

        self.assertEqual(store.stats.residency_count, 1)
        self.assertAlmostEqual(store.stats.mean_residency_s, 25.0)
        exported = store.stats.as_dict()
        self.assertAlmostEqual(exported["mooncake_mean_residency_s"], 25.0)
        histogram = exported["mooncake_residency_histogram"]
        self.assertEqual(sum(histogram.values()), 1)
        self.assertEqual(histogram["10-30s"], 1)

    # -- invariants --------------------------------------------------------

    def _drive(self, store, *, names, reads=()):
        for index, name in enumerate(names):
            store.put([self._key(name)], now=float(index))
            for read_name in reads:
                if store.objects.get(self._key(read_name).to_string()) is not None:
                    store.lookup([self._key(read_name)], now=float(index) + 0.5)

    def test_wasted_plus_useful_equals_memory_evictions(self):
        for policy in ("always", "if_read"):
            with self.subTest(policy=policy):
                store = self._store(demote_policy=policy)
                self._drive(store, names=[f"k{i}" for i in range(40)], reads=("k3", "k7"))
                stats = store.stats
                self.assertGreater(stats.memory_eviction_count, 0)
                self.assertEqual(
                    stats.wasted_write_blocks + stats.useful_write_blocks,
                    stats.memory_eviction_count,
                )

    def test_dropped_plus_demoted_equals_memory_evictions(self):
        for policy in ("always", "if_read"):
            with self.subTest(policy=policy):
                store = self._store(demote_policy=policy)
                self._drive(store, names=[f"k{i}" for i in range(40)], reads=("k3", "k7"))
                stats = store.stats
                self.assertGreater(stats.memory_eviction_count, 0)
                self.assertEqual(
                    stats.blocks_dropped + stats.blocks_demoted,
                    stats.memory_eviction_count,
                )

    def test_always_policy_never_drops_and_if_read_does(self):
        always = self._store(demote_policy="always")
        self._drive(always, names=[f"k{i}" for i in range(40)], reads=("k3", "k7"))
        if_read = self._store(demote_policy="if_read")
        self._drive(if_read, names=[f"k{i}" for i in range(40)], reads=("k3", "k7"))

        self.assertEqual(always.stats.blocks_dropped, 0)
        self.assertGreater(if_read.stats.blocks_dropped, 0)
        self.assertLess(
            if_read.stats.ssd_write_blocks,
            always.stats.ssd_write_blocks,
        )

    def test_byte_conservation_with_drop_path_active(self):
        store = self._store(demote_policy="if_read")
        self._drive(store, names=[f"k{i}" for i in range(40)], reads=("k3", "k7"))

        stats = store.stats
        self.assertGreater(stats.blocks_dropped, 0)
        # Every byte charged to the offload tier corresponds to a whole block
        # that was actually written; dropped blocks cost none.
        self.assertEqual(
            stats.ssd_write_bytes,
            stats.ssd_write_blocks * self.BLOCK_BYTES,
        )
        self.assertEqual(
            stats.ssd_read_bytes,
            stats.ssd_read_blocks * self.BLOCK_BYTES,
        )
        # The tier's own occupancy accounting stays consistent with what it was
        # told to hold: no dropped block leaves a phantom block behind.
        self.assertEqual(
            store.offload.used_blocks,
            sum(store.offload._blocks.values()),
        )
        self.assertLessEqual(store.offload.used_blocks, store.offload.capacity_blocks)

    def test_dropped_block_stays_readable_from_the_write_through_copy(self):
        # The store writes through to the offload tier on admission, so
        # declining the demote write must not lose the block.
        store = self._store(demote_policy="if_read", memory_capacity_blocks=1)
        key = self._key("a")
        store.put([key], now=0.0)
        store.put([self._key("b")], now=1.0)

        self.assertEqual(store.stats.blocks_dropped, 1)
        hit = store.lookup([key], now=2.0)
        self.assertEqual(hit.hit_blocks, 1)
        self.assertEqual(hit.tier, "ssd")

    def test_read_victim_is_still_demoted_under_if_read(self):
        store = self._store(demote_policy="if_read", memory_capacity_blocks=1)
        key = self._key("a")
        store.put([key], now=0.0)
        store.lookup([key], now=0.5)
        before = store.stats.ssd_write_blocks

        store.put([self._key("b")], now=1.0)

        self.assertEqual(store.stats.blocks_demoted, 1)
        self.assertEqual(store.stats.blocks_dropped, 0)
        # The demote write is a real, charged offload write.
        self.assertEqual(store.stats.ssd_write_blocks, before + 2)

    def test_unsupported_demote_policy_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"demote_policy": "sometimes"})

    def test_demote_policy_defaults_to_always(self):
        self.assertEqual(parse_mooncake_config({}).demote_policy, "always")

    def test_histogram_aggregates_element_wise(self):
        left = MooncakeStats()
        left.record_memory_eviction(blocks=1, read_count=0, residency_s=5.0)
        right = MooncakeStats()
        right.record_memory_eviction(blocks=1, read_count=2, residency_s=5.0)

        merged = left.aggregate(right)

        self.assertEqual(sum(merged.residency_histogram), 2)
        self.assertEqual(merged.wasted_write_blocks, 1)
        self.assertEqual(merged.useful_write_blocks, 1)
        self.assertAlmostEqual(merged.mean_residency_s, 5.0)


class MooncakePCMMemoryTierTest(unittest.TestCase):
    """Asymmetric read/write media timing on the memory tier."""

    BLOCK_BYTES = 1 << 20
    GB = 1 << 30

    def _store(self, **overrides):
        config = {
            "enable_offload": True,
            "offload_tier": "ssd",
            "memory_capacity_blocks": 4,
            "ssd_capacity_blocks": 64,
        }
        config.update(overrides)
        return MooncakeStore(parse_mooncake_config(config), block_bytes=self.BLOCK_BYTES)

    @staticmethod
    def _key(name: str) -> PoolKey:
        return PoolKey(KeyMetadata("m"), name)

    def _pcm(self, *, write: bool, cfg):
        fixed = cfg.pcm_write_latency_us if write else cfg.pcm_read_latency_us
        bandwidth = cfg.pcm_write_bw_gbps if write else cfg.pcm_read_bw_gbps
        return fixed / 1e6 + self.BLOCK_BYTES / self.GB / bandwidth

    def test_dram_is_the_default_and_costs_no_media_time(self):
        config = parse_mooncake_config({})
        self.assertEqual(config.memory_media, "dram")
        store = self._store()
        store.put([self._key("a")], now=0.0)
        self.assertEqual(store.stats.memory_write_latency, 0.0)
        self.assertEqual(store.stats.memory_read_latency, 0.0)

    def test_memory_traffic_is_counted_even_under_dram(self):
        store = self._store()
        store.put([self._key("a")], now=0.0)
        store.get([self._key("a")], "memory", now=1.0)
        self.assertEqual(store.stats.memory_write_blocks, 1)
        self.assertEqual(store.stats.memory_write_bytes, self.BLOCK_BYTES)
        self.assertEqual(store.stats.memory_read_blocks, 1)
        self.assertEqual(store.stats.memory_read_bytes, self.BLOCK_BYTES)

    def test_pcm_charges_the_admission_write(self):
        store = self._store(memory_media="pcm")
        timing = store.put_with_timing([self._key("a")], now=0.0)
        expected_media = self._pcm(write=True, cfg=store.config)
        # Admission pays PCM write media plus the offload write-through.
        self.assertGreater(store.stats.memory_write_latency, 0.0)
        self.assertAlmostEqual(store.stats.memory_write_latency, expected_media)
        self.assertGreater(timing.blocking_latency, expected_media)

    def test_pcm_write_is_more_expensive_than_pcm_read(self):
        store = self._store(memory_media="pcm")
        store.put([self._key("a")], now=0.0)
        store.get([self._key("a")], "memory", now=1.0)

        self.assertAlmostEqual(
            store.stats.memory_write_latency,
            self._pcm(write=True, cfg=store.config),
        )
        self.assertAlmostEqual(
            store.stats.memory_read_latency,
            self._pcm(write=False, cfg=store.config),
        )
        # The asymmetry is the point: a symmetric model cannot express it.
        self.assertGreater(
            store.stats.memory_write_latency,
            store.stats.memory_read_latency,
        )

    def test_offload_tier_reads_do_not_charge_memory_media(self):
        store = self._store(memory_media="pcm", memory_capacity_blocks=1)
        key = self._key("a")
        store.put([key], now=0.0)
        store.put([self._key("b")], now=1.0)
        before = store.stats.memory_read_latency

        hit = store.lookup([key], now=2.0)
        self.assertEqual(hit.tier, "ssd")
        store.get(hit.keys, hit.tier, now=2.0)

        self.assertEqual(store.stats.memory_read_latency, before)
        self.assertEqual(store.stats.memory_read_blocks, 0)

    def test_unsupported_memory_media_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"memory_media": "mram"})

    def test_pcm_composes_with_if_read_demotion(self):
        store = self._store(memory_media="pcm", demote_policy="if_read",
                            memory_capacity_blocks=1)
        store.put([self._key("a")], now=0.0)
        store.put([self._key("b")], now=1.0)

        self.assertEqual(store.stats.blocks_dropped, 1)
        self.assertEqual(store.stats.memory_write_blocks, 2)
        self.assertAlmostEqual(
            store.stats.memory_write_latency,
            2 * self._pcm(write=True, cfg=store.config),
        )


class MooncakeWriteBackAdmissionTest(unittest.TestCase):
    """write_back admission: no offload write until a read-back victim demotes."""

    BLOCK_BYTES = 1 << 20

    def _store(self, **overrides):
        config = {
            "enable_offload": True,
            "offload_tier": "ssd",
            "memory_capacity_blocks": 2,
            "ssd_capacity_blocks": 256,
        }
        config.update(overrides)
        return MooncakeStore(parse_mooncake_config(config), block_bytes=self.BLOCK_BYTES)

    @staticmethod
    def _key(name: str) -> PoolKey:
        return PoolKey(KeyMetadata("m"), name)

    def _drive(self, store, *, n=40, reads=("k3", "k7")):
        """Admit n blocks, reading a couple of them back while still resident."""
        for index in range(n):
            store.put([self._key(f"k{index}")], now=float(index))
            for name in reads:
                if store.objects.get(self._key(name).to_string()) is not None:
                    store.lookup([self._key(name)], now=float(index) + 0.5)

    # -- policy plumbing ---------------------------------------------------

    def test_defaults_to_write_through(self):
        self.assertEqual(parse_mooncake_config({}).admission_write_policy, "write_through")

    def test_unsupported_write_policy_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({"admission_write_policy": "write_around"})

    def test_accepts_the_policy_spelled_as_admission_policy(self):
        # admission_policy (always/never) and the write policy have disjoint
        # value sets, so the older name can carry a write-policy value.
        config = parse_mooncake_config({"admission_policy": "write_back"})
        self.assertEqual(config.admission_write_policy, "write_back")
        self.assertEqual(config.admission_policy, "always")

    def test_admission_policy_never_still_works(self):
        self.assertEqual(parse_mooncake_config({"admission_policy": "never"}).admission_policy,
                         "never")

    def test_conflicting_write_policy_spellings_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            parse_mooncake_config({
                "admission_policy": "write_back",
                "admission_write_policy": "write_through",
            })

    # -- behaviour ---------------------------------------------------------

    def test_write_back_spends_no_offload_write_at_admission(self):
        store = self._store(admission_write_policy="write_back")
        store.put([self._key("a")], now=0.0)

        self.assertEqual(store.stats.ssd_write_blocks, 0)
        self.assertEqual(store.stats.ssd_write_bytes, 0)
        self.assertFalse(store.offload.contains(self._key("a").to_string()))

    def test_write_through_spends_one_offload_write_at_admission(self):
        store = self._store(admission_write_policy="write_through")
        store.put([self._key("a")], now=0.0)

        self.assertEqual(store.stats.ssd_write_blocks, 1)
        self.assertTrue(store.offload.contains(self._key("a").to_string()))

    def test_write_back_unread_victim_is_dropped_and_never_reaches_far_tier(self):
        store = self._store(admission_write_policy="write_back",
                            demote_policy="if_read", memory_capacity_blocks=1)
        key = self._key("a")
        store.put([key], now=0.0)
        store.put([self._key("b")], now=1.0)

        self.assertEqual(store.stats.blocks_dropped, 1)
        self.assertEqual(store.stats.ssd_write_blocks, 0)
        self.assertFalse(store.offload.contains(key.to_string()))
        # Dropped with no copy anywhere: the block is gone from the store.
        self.assertNotIn(key.to_string(), store.objects)
        self.assertEqual(store.lookup([key], now=2.0).hit_blocks, 0)

    def test_write_back_read_victim_is_demoted_on_eviction(self):
        store = self._store(admission_write_policy="write_back",
                            demote_policy="if_read", memory_capacity_blocks=1)
        key = self._key("a")
        store.put([key], now=0.0)
        store.lookup([key], now=0.5)
        store.put([self._key("b")], now=1.0)

        self.assertEqual(store.stats.blocks_demoted, 1)
        self.assertEqual(store.stats.ssd_write_blocks, 1)
        self.assertTrue(store.offload.contains(key.to_string()))
        hit = store.lookup([key], now=2.0)
        self.assertEqual(hit.hit_blocks, 1)
        self.assertEqual(hit.tier, "ssd")

    # -- the invariants Step 1 asked for -----------------------------------

    def test_far_tier_writes_equal_read_back_evictions_under_write_back(self):
        store = self._store(admission_write_policy="write_back",
                            demote_policy="if_read")
        self._drive(store)
        stats = store.stats

        # Nothing bypassed memory in this drive, so the far tier received
        # exactly the victims that had been read while resident.
        self.assertEqual(stats.memory_bypass_blocks, 0)
        self.assertGreater(stats.useful_write_blocks, 0)
        self.assertEqual(stats.ssd_write_blocks, stats.useful_write_blocks)
        self.assertEqual(stats.blocks_demoted, stats.useful_write_blocks)

    def test_write_identity_holds_under_both_write_policies(self):
        for policy, expected_admission_writes in (
            ("write_through", True),
            ("write_back", False),
        ):
            with self.subTest(policy=policy):
                store = self._store(admission_write_policy=policy,
                                    demote_policy="if_read")
                self._drive(store)
                stats = store.stats
                if expected_admission_writes:
                    identity = stats.admission_count + stats.blocks_demoted
                else:
                    identity = stats.memory_bypass_blocks + stats.blocks_demoted
                self.assertEqual(stats.ssd_write_blocks, identity)

    def test_byte_conservation_under_write_back(self):
        store = self._store(admission_write_policy="write_back",
                            demote_policy="if_read")
        self._drive(store)
        stats = store.stats

        self.assertEqual(stats.ssd_write_bytes,
                         stats.ssd_write_blocks * self.BLOCK_BYTES)
        self.assertEqual(stats.ssd_read_bytes,
                         stats.ssd_read_blocks * self.BLOCK_BYTES)
        self.assertEqual(stats.memory_write_bytes,
                         stats.memory_write_blocks * self.BLOCK_BYTES)
        # Tier occupancy stays consistent and never exceeds capacity.
        self.assertEqual(store.offload.used_blocks,
                         sum(store.offload._blocks.values()))
        self.assertLessEqual(store.offload.used_blocks, store.offload.capacity_blocks)
        # Every eviction is accounted for exactly once, both ways.
        self.assertEqual(stats.wasted_write_blocks + stats.useful_write_blocks,
                         stats.memory_eviction_count)
        self.assertEqual(stats.blocks_dropped + stats.blocks_demoted,
                         stats.memory_eviction_count)

    def test_write_back_writes_strictly_less_than_write_through(self):
        through = self._store(admission_write_policy="write_through",
                              demote_policy="if_read")
        self._drive(through)
        back = self._store(admission_write_policy="write_back",
                           demote_policy="if_read")
        self._drive(back)

        self.assertLess(back.stats.ssd_write_blocks, through.stats.ssd_write_blocks)


if __name__ == "__main__":
    unittest.main()
