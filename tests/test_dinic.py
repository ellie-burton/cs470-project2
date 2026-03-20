"""Unit checks for Dinic `build_frames` and `validate_result`."""

from __future__ import annotations

import unittest

from algorithms.dinic import PRESETS, build_frames, validate_result


class TestDinic(unittest.TestCase):
    def test_small_max_flow_value(self) -> None:
        data = PRESETS["dinic_small"]
        frames = build_frames(data, {})
        self.assertGreater(len(frames), 2)
        final = frames[-1].meta["final_state"]
        self.assertEqual(final["max_flow"], 20)
        rep = validate_result(data, final)
        self.assertTrue(rep.ok, rep.message)

    def test_layered_max_flow_value(self) -> None:
        data = PRESETS["dinic_layered"]
        frames = build_frames(data, {})
        final = frames[-1].meta["final_state"]
        self.assertEqual(final["max_flow"], 9)
        self.assertTrue(validate_result(data, final).ok)

    def test_parallel_edges_merged(self) -> None:
        data = PRESETS["dinic_parallel"]
        frames = build_frames(data, {})
        final = frames[-1].meta["final_state"]
        self.assertEqual(final["max_flow"], 8)
        self.assertTrue(validate_result(data, final).ok)

    def test_validate_detects_overflow(self) -> None:
        data = PRESETS["dinic_small"]
        bad = {"max_flow": 999, "edge_flows": {("s", "a"): 1000}}
        rep = validate_result(data, bad)
        self.assertFalse(rep.ok)


if __name__ == "__main__":
    unittest.main()
