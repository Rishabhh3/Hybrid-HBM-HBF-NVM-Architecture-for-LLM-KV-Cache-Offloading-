"""build_trace.py --share_partial_tail.

Tail-sharing is a workload variant, not a correction: it assumes a
continuation's tokens match its predecessor's up to the predecessor's length,
which the trace cannot show. The tests pin the switch being off by default and
the three boundary conditions that decide which pairs it touches.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "phase12_build_trace",
    Path(__file__).resolve().parents[1] / "scripts" / "phase12" / "build_trace.py",
)
build_trace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_trace)


def _raw(input_length: int, hash_ids: list[int], output_length: int = 100, ts: int = 0):
    return {
        "timestamp": ts,
        "input_length": input_length,
        "output_length": output_length,
        "hash_ids": hash_ids,
    }


def _run(rows, share: bool):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.jsonl"
        dst = Path(tmp) / "dst.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in rows))
        n, reqs, ids = build_trace.build(str(src), str(dst), len(rows), share)
        out = [json.loads(line) for line in dst.read_text().splitlines()]
    return out, reqs, ids


class ExpandTest(unittest.TestCase):
    def test_a_512_token_chunk_becomes_32_sub_ids(self):
        self.assertEqual(build_trace.expand(_raw(512, [7])), list(range(224, 256)))

    def test_truncation_is_ceil_of_input_over_16(self):
        # 520 tokens -> 33 blocks, the last one partial
        self.assertEqual(len(build_trace.expand(_raw(520, [7, 9]))), 33)


class TailSharingOffTest(unittest.TestCase):
    def test_default_off_changes_nothing(self):
        rows = [_raw(1000, [0, 1]), _raw(2000, [0, 2, 3, 4])]
        off, reqs, ids = _run(rows, share=False)
        self.assertEqual((reqs, ids), (0, 0))
        for row, written in zip(rows, off):
            self.assertEqual(written["hash_ids"], build_trace.expand(row))

    def test_off_and_on_agree_when_no_pair_qualifies(self):
        rows = [_raw(1000, [0, 1]), _raw(1000, [5, 6])]
        off, _, _ = _run(rows, share=False)
        on, reqs, ids = _run(rows, share=True)
        self.assertEqual(off, on)
        self.assertEqual((reqs, ids), (0, 0))


class TailSharingOnTest(unittest.TestCase):
    def test_continuation_takes_the_predecessors_last_chunk(self):
        # R: 1000 tokens, raw [0, 1]; last chunk starts at sub-index 32, and R
        # holds 1000 // 16 = 62 full blocks, so positions 32..61 are shared.
        rows = [_raw(1000, [0, 1]), _raw(2000, [0, 9, 10, 11])]
        on, reqs, ids = _run(rows, share=True)
        earlier, later = on[0]["hash_ids"], on[1]["hash_ids"]
        self.assertEqual((reqs, ids), (1, 30))
        self.assertEqual(later[32:62], earlier[32:62])
        self.assertEqual(later[:32], earlier[:32])
        # R's own trailing partial block (index 62) is NOT shared, and neither
        # is anything past it: those stay R''s own sub-ids.
        self.assertEqual(later[62:], build_trace.expand(rows[1])[62:])

    def test_a_shrinking_continuation_is_never_rewritten(self):
        """Negative delta: R'.input_length < R.input_length."""
        rows = [_raw(2000, [0, 1, 2, 3]), _raw(1600, [0, 1, 2, 9])]
        off, _, _ = _run(rows, share=False)
        on, reqs, ids = _run(rows, share=True)
        self.assertEqual(off, on)
        self.assertEqual((reqs, ids), (0, 0))

    def test_equal_length_continuation_is_eligible(self):
        rows = [_raw(1000, [0, 1]), _raw(1000, [0, 9])]
        _, reqs, ids = _run(rows, share=True)
        self.assertEqual(reqs, 1)

    def test_boundary_input_length_not_a_multiple_of_16(self):
        """Copying stops at floor(R.input_length / 16), never at ceil."""
        # 1001 tokens: floor 62, ceil 63. Position 62 is R's partial block.
        rows = [_raw(1001, [0, 1]), _raw(2000, [0, 9, 10, 11])]
        on, reqs, ids = _run(rows, share=True)
        earlier, later = on[0]["hash_ids"], on[1]["hash_ids"]
        self.assertEqual(len(earlier), 63)
        self.assertEqual((reqs, ids), (1, 30))
        self.assertEqual(later[32:62], earlier[32:62])
        self.assertNotEqual(later[62], earlier[62])

    def test_single_chunk_predecessors_are_ineligible(self):
        """A one-id R has an empty stem, which every later request matches."""
        rows = [_raw(400, [0]), _raw(2000, [5, 6, 7, 8])]
        off, _, _ = _run(rows, share=False)
        on, reqs, ids = _run(rows, share=True)
        self.assertEqual(off, on)
        self.assertEqual((reqs, ids), (0, 0))

    def test_ties_on_stem_length_go_to_the_most_recent(self):
        # Both candidates have stem (0, 1, 2). long_a is longer than long_b, so
        # long_b is not itself rewritten and the two tails stay distinguishable.
        long_a = _raw(2000, [0, 1, 2, 5])
        long_b = _raw(1800, [0, 1, 2, 6])
        rows = [long_a, long_b, _raw(3000, [0, 1, 2, 9, 10, 11])]
        on, _, _ = _run(rows, share=True)
        start, stop = 32 * 3, 1800 // 16
        self.assertEqual(on[1]["hash_ids"], build_trace.expand(long_b))  # untouched
        self.assertEqual(on[2]["hash_ids"][start:stop], on[1]["hash_ids"][start:stop])
        self.assertNotEqual(on[2]["hash_ids"][start:stop], on[0]["hash_ids"][start:stop])

    def test_longest_stem_wins_over_a_shorter_one(self):
        short_stem = _raw(1000, [0, 1])            # stem (0,)   -> copies [32, 62)
        long_stem = _raw(1800, [0, 1, 2, 5])       # stem (0,1,2) -> copies [96, 112)
        rows = [short_stem, long_stem, _raw(3000, [0, 1, 2, 9, 10, 11])]
        on, _, _ = _run(rows, share=True)
        # the long stem won: the copied range is its last chunk, not the short
        # one's, so position 96 carries long_stem's sub-id and not R''s own.
        self.assertEqual(on[2]["hash_ids"][96:112], on[1]["hash_ids"][96:112])
        self.assertNotEqual(
            on[2]["hash_ids"][96:112],
            build_trace.expand(rows[2])[96:112],
        )

    def test_sharing_chains_through_an_already_rewritten_predecessor(self):
        """R1 -> R2 -> R3 propagates: R2 is rewritten before R3 reads it.

        Not a special case in the code -- rows are processed in trace order and
        each reads the current sub-ids -- but it is the behaviour that makes a
        multi-turn chain mutually matchable, so it is pinned here.
        """
        r1 = _raw(1000, [0, 1])
        r2 = _raw(1800, [0, 1, 2, 5])
        r3 = _raw(3000, [0, 1, 2, 5, 10, 11])
        on, _, _ = _run([r1, r2, r3], share=True)
        # r2 took r1's tail at [32, 62); r3 then took r2's, so all three agree
        self.assertEqual(on[1]["hash_ids"][32:62], on[0]["hash_ids"][32:62])
        self.assertEqual(on[2]["hash_ids"][32:62], on[0]["hash_ids"][32:62])

    def test_predecessor_must_come_earlier_in_the_trace(self):
        rows = [_raw(2000, [0, 1, 2, 9]), _raw(1000, [0, 1])]
        off, _, _ = _run(rows, share=False)
        on, reqs, _ = _run(rows, share=True)
        self.assertEqual(off, on)
        self.assertEqual(reqs, 0)


if __name__ == "__main__":
    unittest.main()
