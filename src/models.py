from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    """
    Shared snapshot schema consumed by visualization frontends.

    graph:
        {
            "nodes": [{"id": "A", "x": 80, "y": 80, "label": "A"}],
            "edges": [{"source": "A", "target": "B", "label": "3"}],
            "highlighted_nodes": ["A"],
            "highlighted_edges": [["A", "B"]]
        }
    matrix:
        {
            "values": [[0, 1], [1, 0]],
            "highlighted_cells": [[0, 1]]
        }
    legend:
        {"status": "Running", "items": ["Open set = {A, B}"]}
    """

    explanation: str
    graph: dict[str, Any] = field(default_factory=dict)
    matrix: dict[str, Any] = field(default_factory=dict)
    legend: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
