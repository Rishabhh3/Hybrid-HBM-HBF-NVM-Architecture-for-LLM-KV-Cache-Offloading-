"""--arrival_jitter_ms / --jitter_seed (PHASE 14 STEP 2b).

Default 0 is stock: arrivals untouched and no generator is drawn from. A
non-zero j offsets each timestamped arrival by a uniform draw in [-j, +j] ms on
the trace clock, from a private generator seeded by jitter_seed.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.loaders import (
    apply_arrival_jitter,
    apply_burst_jitter,
    burst_gaps_s,
    load_workload,
)
from TokenSim.workload.user_request import UserRequest, UserRequestList


def _requests(times):
    return UserRequestList(
        UserRequest(request_id=i, prefill_len=16, decode_len=4, arrival_time=t)
        for i, t in enumerate(times)
    )


TIMES = [0.0, 0.0, 1.0, 1.0, 1.001, 5.0, 9.0]


class ArrivalJitterTest(unittest.TestCase):
    def test_zero_is_identity_and_draws_nothing(self):
        random.seed(123)
        expected_next = random.random()
        random.seed(123)
        reqs = apply_arrival_jitter(_requests(TIMES), jitter_ms=0.0, jitter_seed=7)
        self.assertEqual([r.arrival_time for r in reqs], TIMES)
        self.assertEqual(random.random(), expected_next)

    def test_offsets_stay_within_bounds(self):
        reqs = apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=1)
        for original, req in zip(TIMES, reqs):
            self.assertLessEqual(abs(req.arrival_time - original), 0.0005 + 1e-12)
            self.assertGreaterEqual(req.arrival_time, 0.0)

    def test_moves_arrivals(self):
        reqs = apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=1)
        self.assertNotEqual([r.arrival_time for r in reqs][2:], TIMES[2:])

    def test_same_seed_is_deterministic_and_seeds_differ(self):
        a = [r.arrival_time for r in apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=3)]
        b = [r.arrival_time for r in apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=3)]
        c = [r.arrival_time for r in apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_private_generator_leaves_global_state_alone(self):
        random.seed(99)
        expected_next = random.random()
        random.seed(99)
        apply_arrival_jitter(_requests(TIMES), jitter_ms=0.5, jitter_seed=1)
        self.assertEqual(random.random(), expected_next)

    def test_untimestamped_requests_are_untouched(self):
        reqs = UserRequestList(
            [UserRequest(request_id=0, prefill_len=16, decode_len=4, inter_arrival_time=0.25)]
        )
        apply_arrival_jitter(reqs, jitter_ms=0.5, jitter_seed=1)
        self.assertIsNone(reqs[0].arrival_time)
        self.assertEqual(reqs[0].inter_arrival_time, 0.25)

    def test_negative_or_non_finite_is_rejected(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(WorkloadValidationError):
                apply_arrival_jitter(_requests(TIMES), jitter_ms=bad, jitter_seed=1)

    def test_jitter_is_applied_before_timestamp_scaling(self):
        """j is on the trace clock: under trace_timestamp_scale s the
        simulated offset is at most j * s."""
        import json, os, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            with open(path, "w") as handle:
                for i, t in enumerate([0.0, 10.0, 20.0, 30.0]):
                    handle.write(json.dumps({"timestamp": t, "input_length": 16, "output_length": 4}) + "\n")
            base = dict(
                workload_type="qwen_jsonl", dataset_path=path, request_count=None,
                dataset_skip_count=0, trace_timestamp_scale=0.1, trace_target_qps=None,
                random_seed=0,
            )
            model = SimpleNamespace(moe_config=SimpleNamespace(num_experts=0))
            stock = load_workload(SimpleNamespace(**base), model)
            jittered = load_workload(
                SimpleNamespace(**base, arrival_jitter_ms=0.5, jitter_seed=2), model
            )
            for s, j in zip(stock, jittered):
                self.assertLessEqual(abs(j.arrival_time - s.arrival_time), 0.0005 * 0.1 + 1e-12)


BURSTS = [0.0, 0.0, 0.0, 3.0, 3.0, 6.0, 6.0, 6.0, 9.0]


class BurstJitterTest(unittest.TestCase):
    def test_zero_is_identity_and_draws_nothing(self):
        random.seed(5)
        expected_next = random.random()
        random.seed(5)
        reqs = apply_burst_jitter(_requests(BURSTS), jitter_ms=0.0, jitter_seed=1)
        self.assertEqual([r.arrival_time for r in reqs], BURSTS)
        self.assertEqual(random.random(), expected_next)

    def test_every_request_in_a_burst_gets_the_same_offset(self):
        reqs = apply_burst_jitter(_requests(BURSTS), jitter_ms=1000.0, jitter_seed=2)
        times = [r.arrival_time for r in reqs]
        self.assertEqual(len(set(times[0:3])), 1)
        self.assertEqual(len(set(times[3:5])), 1)
        self.assertEqual(len(set(times[5:8])), 1)
        self.assertNotEqual(times[3:], BURSTS[3:])

    def test_offsets_within_bounds_and_order_kept(self):
        for seed in range(1, 20):
            reqs = apply_burst_jitter(_requests(BURSTS), jitter_ms=1400.0, jitter_seed=seed)
            times = [r.arrival_time for r in reqs]
            for original, t in zip(BURSTS, times):
                self.assertLessEqual(abs(t - original), 1.4 + 1e-12)
                self.assertGreaterEqual(t, 0.0)
            self.assertEqual(sorted(range(len(times)), key=times.__getitem__), list(range(len(times))))

    def test_order_guard_fails_the_run_when_bursts_swap(self):
        # j = 2.9 s against 3 s gaps: some seed swaps two bursts.
        raised = 0
        for seed in range(1, 40):
            try:
                apply_burst_jitter(_requests(BURSTS), jitter_ms=2900.0, jitter_seed=seed)
            except WorkloadValidationError as exc:
                self.assertIn("file order", str(exc))
                raised += 1
        self.assertGreater(raised, 0)

    def test_order_guard_fires_on_a_trace_not_in_time_order(self):
        with self.assertRaises(WorkloadValidationError):
            apply_burst_jitter(_requests([3.0, 0.0]), jitter_ms=1.0, jitter_seed=1)

    def test_deterministic_per_seed(self):
        a = [r.arrival_time for r in apply_burst_jitter(_requests(BURSTS), jitter_ms=500.0, jitter_seed=3)]
        b = [r.arrival_time for r in apply_burst_jitter(_requests(BURSTS), jitter_ms=500.0, jitter_seed=3)]
        c = [r.arrival_time for r in apply_burst_jitter(_requests(BURSTS), jitter_ms=500.0, jitter_seed=4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_burst_gaps(self):
        self.assertEqual(burst_gaps_s(_requests(BURSTS)), [3.0, 3.0, 3.0])

    def test_burst_mode_applies_after_scaling_and_request_mode_before(self):
        import json, os, tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            with open(path, "w") as handle:
                for t in [0.0, 0.0, 10.0, 10.0, 20.0, 30.0]:
                    handle.write(json.dumps({"timestamp": t, "input_length": 16, "output_length": 4}) + "\n")
            base = dict(
                workload_type="qwen_jsonl", dataset_path=path, request_count=None,
                dataset_skip_count=0, trace_timestamp_scale=0.1, trace_target_qps=None,
                random_seed=0, arrival_jitter_ms=400.0, jitter_seed=1,
            )
            model = SimpleNamespace(moe_config=SimpleNamespace(num_experts=0))
            stock = load_workload(SimpleNamespace(**{**base, "arrival_jitter_ms": 0.0}), model)
            burst = load_workload(SimpleNamespace(**base, arrival_jitter_mode="burst"), model)
            # Replay clock: offsets up to the full 0.4 s, not 0.4 s x 0.1.
            deltas = [abs(b.arrival_time - s.arrival_time) for b, s in zip(burst, stock)]
            self.assertTrue(all(d <= 0.4 + 1e-12 for d in deltas))
            self.assertGreater(max(deltas), 0.04)
            self.assertEqual(burst[0].arrival_time, burst[1].arrival_time)
            with self.assertRaises(WorkloadValidationError):
                load_workload(SimpleNamespace(**base, arrival_jitter_mode="bogus"), model)


if __name__ == "__main__":
    unittest.main()
