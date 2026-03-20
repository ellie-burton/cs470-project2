from __future__ import annotations

import itertools
import random
import unittest

from algorithms.hungarian import PRESETS, build_frames, validate_result


def _best_assignment_objective(values: list[list[float]], *, mode: str) -> float:
    n = len(values)
    cols = range(n)
    best: float | None = None
    for perm in itertools.permutations(cols, n):
        total = float(sum(values[i][perm[i]] for i in range(n)))
        if best is None:
            best = total
            continue
        if mode == "min" and total < best:
            best = total
        if mode == "max" and total > best:
            best = total
    assert best is not None
    return float(best)


class HungarianTests(unittest.TestCase):
    def _assert_preset_optimal(self, preset_name: str) -> None:
        data = PRESETS[preset_name]
        frames = build_frames(data, {})
        self.assertGreater(len(frames), 0)
        final_state = frames[-1].meta.get("final_state", {})
        rep = validate_result(data, final_state)
        self.assertTrue(rep.ok, msg=rep.message)

        mode = str(data.get("mode", "min"))
        values = data["matrix"]["values"]
        brute = _best_assignment_objective(values, mode=mode)
        got = float(final_state["objective"])
        self.assertAlmostEqual(got, brute, places=9)

    def test_presets_are_optimal(self) -> None:
        for name in ("hungarian_easy", "hungarian_ties", "hungarian_max"):
            with self.subTest(preset=name):
                self._assert_preset_optimal(name)

    def test_validate_catches_bad_objective(self) -> None:
        data = PRESETS["hungarian_easy"]
        frames = build_frames(data, {})
        final_state = dict(frames[-1].meta["final_state"])
        final_state["objective"] = float(final_state["objective"]) + 1.0
        rep = validate_result(data, final_state)
        self.assertFalse(rep.ok)

    def test_random_small_matrices_optimal_min(self) -> None:
        """A few deterministic random instances; brute force n! is cheap for n<=4."""
        rng = random.Random(20250319)
        for n in (3, 4):
            for _ in range(6):
                values = [[rng.randint(0, 12) for _c in range(n)] for _r in range(n)]
                data = {"mode": "min", "matrix": {"values": values}}
                frames = build_frames(data, {})
                fs = frames[-1].meta["final_state"]
                self.assertTrue(validate_result(data, fs).ok)
                brute = _best_assignment_objective(values, mode="min")
                self.assertAlmostEqual(float(fs["objective"]), brute, places=9)

    def test_random_small_matrices_optimal_max(self) -> None:
        rng = random.Random(20250320)
        for n in (3, 4):
            for _ in range(4):
                values = [[rng.randint(0, 15) for _c in range(n)] for _r in range(n)]
                data = {"mode": "max", "matrix": {"values": values}}
                frames = build_frames(data, {})
                fs = frames[-1].meta["final_state"]
                self.assertTrue(validate_result(data, fs).ok)
                brute = _best_assignment_objective(values, mode="max")
                self.assertAlmostEqual(float(fs["objective"]), brute, places=9)


if __name__ == "__main__":
    unittest.main()
