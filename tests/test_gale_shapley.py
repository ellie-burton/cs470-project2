from __future__ import annotations

import unittest

from algorithms import gale_shapley as gs
from src.demo_loader import load_gale_shapley_preset


class TestGaleShapley(unittest.TestCase):
    def test_presets_validate(self) -> None:
        for preset in ("gs_no_swap", "gs_swaps", "gs_asymmetric"):
            data = load_gale_shapley_preset(preset)
            frames = gs.build_frames(data, {"proposer_side": "proposers"})
            self.assertGreaterEqual(len(frames), 2)
            final = frames[-1].meta.get("final_state", {})
            report = gs.validate_result(data, final)
            self.assertTrue(report.ok, msg=f"{preset}: {report.message} | {report.details}")

    def test_receiver_proposer_side_still_stable(self) -> None:
        payload = dict(load_gale_shapley_preset("gs_women_propose"))
        cfg = dict(payload.pop("__config__", {}) or {})
        frames = gs.build_frames(payload, cfg)
        final = frames[-1].meta.get("final_state", {})
        report = gs.validate_result(payload, final)
        self.assertTrue(report.ok, msg=report.message)

    def test_swap_preset_contains_swap_event(self) -> None:
        data = load_gale_shapley_preset("gs_swaps")
        frames = gs.build_frames(data, {"proposer_side": "proposers"})
        events = [f.meta.get("event") for f in frames]
        self.assertIn("swap", events)

    def test_gs_swaps_receivers_propose_stable(self) -> None:
        """Same as UI path: receivers as proposers (matches gs_women_propose data intent)."""
        data = load_gale_shapley_preset("gs_swaps")
        frames = gs.build_frames(data, {"proposer_side": "receivers"})
        final = frames[-1].meta.get("final_state", {})
        report = gs.validate_result(data, final)
        self.assertTrue(report.ok, msg=report.message)

    def test_swap_event_when_receivers_propose(self) -> None:
        data = load_gale_shapley_preset("gs_swaps")
        frames = gs.build_frames(data, {"proposer_side": "receivers"})
        events = [f.meta.get("event") for f in frames]
        self.assertIn("swap", events)

    def test_state_panel_meta_present(self) -> None:
        data = load_gale_shapley_preset("gs_no_swap")
        frames = gs.build_frames(data, {"proposer_side": "proposers"})
        mid = frames[len(frames) // 2].meta
        for key in (
            "proposer_ids",
            "receiver_ids",
            "free_proposer_names",
            "engagement_pairs",
            "event_color_hint",
        ):
            self.assertIn(key, mid, msg=f"missing meta[{key!r}]")


if __name__ == "__main__":
    unittest.main()
