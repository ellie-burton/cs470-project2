from __future__ import annotations

from typing import Any

from src.models import Frame, ValidationReport


PRESETS: dict[str, dict[str, Any]] = {
    "dummy_triangle": {
        "graph": {
            "directed": True,
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
    },
}


def build_frames(input_data: dict[str, Any], config: dict[str, Any]) -> list[Frame]:
    """
    Smoke-demo algorithm: emits a tiny frame sequence using the shared `Frame` schema.

    Other algorithms should follow the same function signature and return shape.
    """
    _ = config

    base_graph = dict(input_data.get("graph") or {})
    base_matrix = dict(input_data.get("matrix") or {})

    values = base_matrix.get("values") or []

    return [
        Frame(
            explanation="Start: highlight node A and the first row of the adjacency matrix.",
            graph={
                **base_graph,
                "highlighted_nodes": ["A"],
                "highlighted_edges": [],
            },
            matrix={
                **base_matrix,
                "highlighted_cells": [[0, c] for c in range(len(values[0]))] if values else [],
            },
            legend={"status": "Demo", "items": ["Active node: A"]},
            meta={"phase": "step", "step": 0},
        ),
        Frame(
            explanation="Step: move highlight to edge A->B and matrix cell (0, 1).",
            graph={
                **base_graph,
                "highlighted_nodes": ["A", "B"],
                "highlighted_edges": [["A", "B"]],
            },
            matrix={**base_matrix, "highlighted_cells": [[0, 1]]},
            legend={"status": "Demo", "items": ["Relaxing edge A->B"]},
            meta={"phase": "step", "step": 1},
        ),
        Frame(
            explanation="Done: highlight node C and the diagonal (for visibility).",
            graph={
                **base_graph,
                "highlighted_nodes": ["C"],
                "highlighted_edges": [["B", "C"]],
            },
            matrix={
                **base_matrix,
                "highlighted_cells": [[i, i] for i in range(len(values))] if values else [],
            },
            legend={"status": "Demo", "items": ["Finished demo traversal"]},
            meta={
                "phase": "done",
                "step": 2,
                "algorithm": "dummy",
                "final_state": {"step": 2},
            },
        ),
    ]


def validate_result(input_data: dict[str, Any], final_state: dict[str, Any]) -> ValidationReport:
    _ = input_data
    step = final_state.get("step")
    if step == 2:
        return ValidationReport(ok=True, message="Dummy algorithm reached terminal demo state.")
    return ValidationReport(ok=False, message="Dummy validation expects final_state.step == 2.", details={"step": step})
