from __future__ import annotations

from typing import Any


def load_demo_input() -> dict:
    return {
        "graph": {
            "nodes": [
                {"id": "A", "x": 80, "y": 90, "label": "A"},
                {"id": "B", "x": 220, "y": 90, "label": "B"},
                {"id": "C", "x": 150, "y": 200, "label": "C"},
            ],
            "edges": [
                {"source": "A", "target": "B", "label": "2"},
                {"source": "B", "target": "C", "label": "1"},
                {"source": "A", "target": "C", "label": "4"},
            ],
        },
        "matrix": {
            "values": [
                [0, 2, 4],
                [2, 0, 1],
                [4, 1, 0],
            ]
        },
    }


def load_gale_shapley_preset(name: str) -> dict[str, Any]:
    """
    Curated Gale-Shapley inputs for demos/tests.

    Presets are defined in `algorithms/gale_shapley.py` alongside the algorithm module.
    """
    # Lazy import keeps `src.demo_loader` lightweight and avoids import cycles during app startup.
    from algorithms.gale_shapley import PRESETS as gale_shapley_presets

    try:
        return dict(gale_shapley_presets[name])
    except KeyError as e:
        available = ", ".join(sorted(gale_shapley_presets.keys()))
        raise KeyError(f"Unknown Gale-Shapley preset {name!r}. Available: {available}") from e
