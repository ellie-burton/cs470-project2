#!/usr/bin/env python3
"""Non-interactive smoke run: build frames + validation for each algorithm preset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import dinic, dummy_algorithm, gale_shapley, hungarian


def main() -> int:
    sections = [
        ("Dummy (smoke)", dummy_algorithm.PRESETS, dummy_algorithm.build_frames, dummy_algorithm.validate_result),
        ("Hungarian", hungarian.PRESETS, hungarian.build_frames, hungarian.validate_result),
        ("Gale–Shapley", gale_shapley.PRESETS, gale_shapley.build_frames, gale_shapley.validate_result),
        ("Dinic", dinic.PRESETS, dinic.build_frames, dinic.validate_result),
    ]
    for title, presets, build, validate in sections:
        print(f"=== {title} ===")
        for name, payload in presets.items():
            data = dict(payload)
            cfg = dict(data.pop("__config__", {}) or {})
            frames = build(data, cfg)
            print(f"  {name}: {len(frames)} frames")
            if frames:
                fs = frames[-1].meta.get("final_state", {})
                rep = validate(data, fs)
                print(f"    validate: {'OK' if rep.ok else 'FAIL'} — {rep.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
