"""
Hungarian algorithm (assignment problem) with step-by-step animation frames.

Contract (shared integration surface):
    build_frames(input_data, config) -> list[Frame]
    validate_result(input_data, final_state) -> ValidationReport

Input schema (input_data):
    mode: "min" | "max" (optional, default "min")
    matrix: {"values": [[...], ...]}   (preferred)
    cost_matrix: [[...], ...]          (legacy preset shape used by earlier scaffolding)

final_state (from last frame meta["final_state"]) for validate_result:
    mode: "min" | "max"
    objective: float
    assignment: list[list[int, int]]  # (row, col) pairs in original matrix coordinates
    working_matrix: list[list[float]] # matrix after reductions/adjustments (for debugging/UI)
    transform: {"kind": "none"} | {"kind": "max_to_min", "max_entry": float}

Max mode conversion (documented):
    For a square profit matrix P, define cost C_ij = max(P) - P_ij.
    Minimizing total cost on C is equivalent to maximizing total profit on P because for a perfect matching:
        sum_ij C_ij x_ij = n*max(P) - sum_ij P_ij x_ij

Visualization handoff (`Frame.meta["hungarian"]`, plus `Frame.meta["algorithm"] == "hungarian"`):
    - phase: start | row_reduction | col_reduction | matching_intro | matching | line_cover_intro | line_cover |
      adjust_uncovered | adjust_double | assignment_read | done
    - iteration, row, col, subtracted, zero_cells, matching_size, matching_edges, starred
    - row_covered, col_covered, lines{rows,cols}, line_count, h
    Terminal frames also set `Frame.meta["final_state"]` for validate_result. See README Hungarian section.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from src.models import Frame, ValidationReport


PRESETS: dict[str, dict[str, Any]] = {
    # Small easy case (3x3).
    "hungarian_easy": {
        "mode": "min",
        "matrix": {
            "values": [
                [4, 1, 3],
                [2, 0, 5],
                [3, 2, 2],
            ]
        },
    },
    # Multiple optimal assignments with the same minimum total cost.
    "hungarian_ties": {
        "mode": "min",
        "matrix": {
            "values": [
                [1, 1, 9, 9],
                [1, 1, 9, 9],
                [9, 9, 1, 1],
                [9, 9, 1, 1],
            ]
        },
    },
    # Max-profit mode (converted internally to min-cost).
    "hungarian_max": {
        "mode": "max",
        "matrix": {
            "values": [
                [7, 3, 1],
                [2, 9, 4],
                [8, 1, 5],
            ]
        },
    },
}


def _extract_matrix_values(input_data: dict[str, Any]) -> list[list[Any]]:
    if "matrix" in input_data:
        m = input_data.get("matrix") or {}
        return list(m.get("values") or [])
    if "cost_matrix" in input_data:
        return list(input_data.get("cost_matrix") or [])
    raise KeyError("Hungarian input must include either input_data['matrix']['values'] or input_data['cost_matrix'].")


def _as_float_matrix(values: list[list[Any]]) -> list[list[float]]:
    return [[float(x) for x in row] for row in values]


def _require_square_matrix(values: list[list[float]]) -> int:
    n = len(values)
    if n == 0:
        raise ValueError("Hungarian input requires a non-empty square matrix.")
    for r, row in enumerate(values):
        if len(row) != n:
            raise ValueError(f"Expected a square matrix; row {r} has length {len(row)} but n={n}.")
    return n


def _max_entry(values: list[list[float]]) -> float:
    m = -math.inf
    for row in values:
        for x in row:
            if x > m:
                m = x
    if m == -math.inf:
        raise ValueError("Matrix appears empty.")
    return float(m)


def _objective_original(matrix_orig: list[list[float]], assignment: list[tuple[int, int]], mode: str) -> float:
    if mode == "min":
        return float(sum(matrix_orig[r][c] for r, c in assignment))
    if mode == "max":
        return float(sum(matrix_orig[r][c] for r, c in assignment))
    raise ValueError("mode must be 'min' or 'max'")


def _bipartite_max_matching_zero_graph(a: list[list[float]]) -> tuple[list[int], int]:
    """
    Maximum matching in the bipartite graph where left i connects to right j if a[i][j] == 0.

    Returns (match_r, size) where match_r[j] = matched left index or -1.
    """

    n = len(a)
    match_r = [-1] * n

    def dfs(i: int, seen: list[bool]) -> bool:
        for j in range(n):
            if a[i][j] != 0:
                continue
            if seen[j]:
                continue
            seen[j] = True
            if match_r[j] == -1 or dfs(match_r[j], seen):
                match_r[j] = i
                return True
        return False

    size = 0
    for i in range(n):
        seen = [False] * n
        if dfs(i, seen):
            size += 1
    return match_r, size


def _assignment_from_matching(match_r: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for j, i in enumerate(match_r):
        if i != -1:
            pairs.append((i, j))
    pairs.sort()
    return pairs


def _min_vertex_cover_from_matching(a: list[list[float]], match_r: list[int]) -> tuple[list[bool], list[bool]]:
    """
    Minimum vertex cover in a bipartite graph defined by zero entries, given a *maximum* matching.

    Standard alternating reachability from unmatched left vertices:
      left --non-matching zero edge--> right --matching edge--> left ...
    Then:
      cover_left = left \\ visited_left
      cover_right = visited_right
    """

    n = len(a)
    match_l = [-1] * n
    for j, i in enumerate(match_r):
        if i != -1:
            match_l[i] = j

    unmatched_left = [i for i in range(n) if match_l[i] == -1]

    visited_l = [False] * n
    visited_r = [False] * n
    stack = list(unmatched_left)
    for i in unmatched_left:
        visited_l[i] = True

    while stack:
        i = stack.pop()
        for j in range(n):
            if a[i][j] != 0:
                continue
            if match_l[i] == j:
                continue
            if visited_r[j]:
                continue
            visited_r[j] = True
            mi = match_r[j]
            if mi == -1:
                continue
            if not visited_l[mi]:
                visited_l[mi] = True
                stack.append(mi)

    cover_l = [not visited_l[i] for i in range(n)]
    cover_r = [visited_r[j] for j in range(n)]
    return cover_l, cover_r


def _min_cover_num_lines(cover_l: list[bool], cover_r: list[bool]) -> int:
    return int(sum(1 for x in cover_l if x) + sum(1 for x in cover_r if x))


def _frame(
    *,
    explanation: str,
    matrix_values: list[list[float]],
    highlighted_cells: list[list[int]],
    meta_extra: dict[str, Any],
    legend_items: list[str],
) -> Frame:
    hungarian = dict(meta_extra)
    return Frame(
        explanation=explanation,
        matrix={"values": [list(map(float, row)) for row in matrix_values], "highlighted_cells": highlighted_cells},
        legend={"status": "Hungarian", "items": legend_items},
        meta={"algorithm": "hungarian", "hungarian": hungarian},
    )


def build_frames(input_data: dict[str, Any], config: dict[str, Any]) -> list[Frame]:
    _ = config

    mode = str(input_data.get("mode", "min"))
    if mode not in {"min", "max"}:
        raise ValueError("Hungarian config expects input_data['mode'] in {'min','max'}.")

    values_raw = _extract_matrix_values(input_data)
    matrix_orig = _as_float_matrix(values_raw)
    n = _require_square_matrix(matrix_orig)

    matrix_work = copy.deepcopy(matrix_orig)
    max_profit = _max_entry(matrix_orig)
    if mode == "max":
        for i in range(n):
            for j in range(n):
                matrix_work[i][j] = max_profit - matrix_orig[i][j]

    frames: list[Frame] = []

    def legend_base(phase: str) -> list[str]:
        items = [
            f"Phase: {phase}",
            f"n={n}",
            f"mode={mode}",
        ]
        if mode == "max":
            items.append(f"Max-to-min transform: C_ij = max(P)-P_ij (here max(P)={max_profit:g})")
        return items

    frames.append(
        _frame(
            explanation=(
                "What: set up the n-by-n assignment problem - pick exactly one entry per row and per column. "
                "Why: the Hungarian method rewrites costs so cheap choices appear as zeros, then repeats "
                "match-on-zeros, cover zeros with lines, and adjust until a full matching appears."
                + (
                    " What (max mode): replace profit P by cost C with C_ij = max(P) - P_ij. "
                    "Why: minimizing total C on a perfect matching maximizes total P, since n*max(P) is constant."
                    if mode == "max"
                    else ""
                )
            ),
            matrix_values=matrix_work,
            highlighted_cells=[],
            meta_extra={
                "phase": "start",
                "row_covered": [False] * n,
                "col_covered": [False] * n,
                "starred": [],
                "lines": {"rows": [], "cols": []},
            },
            legend_items=legend_base("start"),
        )
    )

    # --- Row reduction (split into identify-min, then subtract) ---
    for i in range(n):
        row = matrix_work[i]
        mn = min(row)
        frames.append(
            _frame(
                explanation=(
                    f"What: identify the minimum value in row {i}: {mn:g}. "
                    "Why: this is the constant we can remove from the whole row without changing "
                    "which column is best for that row."
                ),
                matrix_values=matrix_work,
                highlighted_cells=[[i, j] for j in range(n)],
                meta_extra={
                    "phase": "row_min_identify",
                    "row": i,
                    "subtracted": float(mn),
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": [],
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("row_min_identify") + [f"Row {i} minimum: {mn:g}"],
            )
        )

        if mn != 0:
            for j in range(n):
                matrix_work[i][j] -= mn
        frames.append(
            _frame(
                explanation=(
                    f"What: subtract {mn:g} from every entry in row {i}. "
                    "Why: this creates at least one zero in the row (or keeps zeros), which helps build "
                    "a zero-based matching while preserving optimal assignments."
                ),
                matrix_values=matrix_work,
                highlighted_cells=[[i, j] for j in range(n)],
                meta_extra={
                    "phase": "row_reduction",
                    "row": i,
                    "subtracted": float(mn),
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": [],
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("row_reduction") + [f"Updated row {i}"],
            )
        )

    # --- Column reduction (split into identify-min, then subtract) ---
    for j in range(n):
        col = [matrix_work[i][j] for i in range(n)]
        mn = min(col)
        frames.append(
            _frame(
                explanation=(
                    f"What: identify the minimum value in column {j}: {mn:g}. "
                    "Why: this is the constant we can remove from the whole column without changing "
                    "relative assignment quality."
                ),
                matrix_values=matrix_work,
                highlighted_cells=[[i, j] for i in range(n)],
                meta_extra={
                    "phase": "col_min_identify",
                    "col": j,
                    "subtracted": float(mn),
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": [],
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("col_min_identify") + [f"Column {j} minimum: {mn:g}"],
            )
        )

        if mn != 0:
            for i in range(n):
                matrix_work[i][j] -= mn
        frames.append(
            _frame(
                explanation=(
                    f"What: subtract {mn:g} from every entry in column {j}. "
                    "Why: column reduction adds/keeps zeros, expanding candidate zero edges for a full matching."
                ),
                matrix_values=matrix_work,
                highlighted_cells=[[i, j] for i in range(n)],
                meta_extra={
                    "phase": "col_reduction",
                    "col": j,
                    "subtracted": float(mn),
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": [],
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("col_reduction") + [f"Updated col {j}"],
            )
        )

    iteration = 0
    while True:
        iteration += 1
        match_r, match_size = _bipartite_max_matching_zero_graph(matrix_work)
        assignment = _assignment_from_matching(match_r)
        starred = [[r, c] for r, c in assignment]

        zero_cells = [[i, j] for i in range(n) for j in range(n) if matrix_work[i][j] == 0]
        frames.append(
            _frame(
                explanation=(
                    f"What (iteration {iteration}): treat each zero as an edge between row i and column j. "
                    "Why: after reductions, picking a zero is picking a minimum-cost option for that row/column pair "
                    "in the transformed matrix; we ask how many disjoint zeros we can take."
                ),
                matrix_values=matrix_work,
                highlighted_cells=zero_cells,
                meta_extra={
                    "phase": "matching_intro",
                    "iteration": iteration,
                    "matching_size": None,
                    "matching_edges": [],
                    "zero_cells": zero_cells,
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": [],
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("matching_intro")
                + [f"Zeros in matrix: {len(zero_cells)}", "Highlighted: all current zero entries"],
            )
        )

        frames.append(
            _frame(
                explanation=(
                    f"What: compute a maximum matching in the zero graph (iteration {iteration}). "
                    f"Result: matching size = {match_size}. "
                    + (
                        "Why: if size = n we can assign each row to a distinct column using only zeros—done for this phase. "
                        if match_size == n
                        else "Why: if size < n, zeros are too constrained; we need a line cover and a matrix adjustment "
                        "to create new zeros without breaking optimality."
                    )
                ),
                matrix_values=matrix_work,
                highlighted_cells=[[r, c] for r, c in assignment],
                meta_extra={
                    "phase": "matching",
                    "iteration": iteration,
                    "matching_size": match_size,
                    "matching_edges": [[r, c] for r, c in assignment],
                    "zero_cells": zero_cells,
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": starred,
                    "lines": {"rows": [], "cols": []},
                },
                legend_items=legend_base("matching")
                + [f"Matching size: {match_size}/{n}", "Highlighted: one maximum matching on zeros"],
            )
        )

        if match_size == n:
            obj = _objective_original(matrix_orig, assignment, mode)
            final_state = {
                "mode": mode,
                "objective": obj,
                "assignment": [[r, c] for r, c in assignment],
                "working_matrix": [list(map(float, row)) for row in matrix_work],
                "transform": (
                    {"kind": "max_to_min", "max_entry": max_profit}
                    if mode == "max"
                    else {"kind": "none"}
                ),
            }
            frames.append(
                _frame(
                    explanation=(
                        "What: read the assignment from the perfect zero matching - each highlighted cell is row-to-column. "
                        "Why: in the transformed cost matrix, a perfect zero matching is an optimal assignment; "
                        "we will next translate the cost/profit back to your original matrix."
                    ),
                    matrix_values=matrix_work,
                    highlighted_cells=[[r, c] for r, c in assignment],
                    meta_extra={
                        "phase": "assignment_read",
                        "iteration": iteration,
                        "matching_size": match_size,
                        "matching_edges": [[r, c] for r, c in assignment],
                        "zero_cells": zero_cells,
                        "row_covered": [False] * n,
                        "col_covered": [False] * n,
                        "starred": starred,
                        "lines": {"rows": [], "cols": []},
                        "final_state": final_state,
                    },
                    legend_items=legend_base("assignment_read")
                    + [f"Assignment pairs (row, col): {sorted(assignment)}"],
                )
            )
            frames.append(
                _frame(
                    explanation=(
                        f"What: report the objective on the original input matrix ({mode} mode): total = {obj:g}. "
                        "Why: reductions and max-to-min transforms are equivalent rewrites; summing original entries "
                        "for the chosen pairs is the number you care about for grading or applications."
                    ),
                    matrix_values=matrix_work,
                    highlighted_cells=[[r, c] for r, c in assignment],
                    meta_extra={
                        "phase": "done",
                        "iteration": iteration,
                        "row_covered": [False] * n,
                        "col_covered": [False] * n,
                        "starred": starred,
                        "lines": {"rows": [], "cols": []},
                        "final_state": final_state,
                    },
                    legend_items=legend_base("done") + [f"Objective ({mode}): {obj:g}"],
                )
            )
            for fr in frames:
                fr.meta["final_state"] = final_state
            return frames

        cover_l, cover_r = _min_vertex_cover_from_matching(matrix_work, match_r)
        lines = {"rows": [i for i in range(n) if cover_l[i]], "cols": [j for j in range(n) if cover_r[j]]}
        num_lines = _min_cover_num_lines(cover_l, cover_r)

        frames.append(
            _frame(
                explanation=(
                    f"What (iteration {iteration}): the zero matching has size {match_size} < {n}. "
                    "Why: before we can adjust the matrix, we need the smallest set of rows/columns that touches "
                    "every zero - that tells us where new zeros must appear next (outside those lines)."
                ),
                matrix_values=matrix_work,
                highlighted_cells=[],
                meta_extra={
                    "phase": "line_cover_intro",
                    "iteration": iteration,
                    "matching_size": match_size,
                    "row_covered": [False] * n,
                    "col_covered": [False] * n,
                    "starred": starred,
                    "matching_edges": [[r, c] for r, c in assignment],
                    "lines": {"rows": [], "cols": []},
                    "line_count": None,
                },
                legend_items=legend_base("line_cover_intro")
                + [f"Current matching size: {match_size}/{n}", "Next frame: draw minimum line cover"],
            )
        )

        frames.append(
            _frame(
                explanation=(
                    f"What: draw a minimum line cover - {num_lines} row(s)/column(s) - so every zero lies on a chosen line. "
                    "Why: Konig's theorem (for this bipartite zero graph) links maximum matching and minimum cover; "
                    f"with only {num_lines} lines and a matching of size {match_size}, the cover explains which entries "
                    'stay "blocked" versus which uncovered region will shift next.'
                ),
                matrix_values=matrix_work,
                highlighted_cells=[],
                meta_extra={
                    "phase": "line_cover",
                    "iteration": iteration,
                    "matching_size": match_size,
                    "row_covered": list(cover_l),
                    "col_covered": list(cover_r),
                    "starred": starred,
                    "matching_edges": [[r, c] for r, c in assignment],
                    "lines": lines,
                    "line_count": num_lines,
                },
                legend_items=legend_base("line_cover")
                + [f"Covered rows: {lines['rows']}", f"Covered cols: {lines['cols']}", f"Total lines: {num_lines}"],
            )
        )

        uncovered_vals: list[float] = []
        for i in range(n):
            if cover_l[i]:
                continue
            for j in range(n):
                if cover_r[j]:
                    continue
                uncovered_vals.append(matrix_work[i][j])
        if not uncovered_vals:
            raise RuntimeError("Hungarian adjustment failed: no uncovered entries while matching < n.")

        h = min(uncovered_vals)
        uncovered_cells: list[list[int]] = []
        matrix_after_uncovered = copy.deepcopy(matrix_work)
        for i in range(n):
            for j in range(n):
                if (not cover_l[i]) and (not cover_r[j]):
                    matrix_after_uncovered[i][j] -= h
                    uncovered_cells.append([i, j])

        frames.append(
            _frame(
                explanation=(
                    f"What (iteration {iteration}): let h = {h:g} (smallest entry in the uncovered region). "
                    "Subtract h from every uncovered cell. "
                    "Why: this creates at least one new zero among entries not touched by the cover, which is where "
                    "a larger zero matching must grow."
                ),
                matrix_values=matrix_after_uncovered,
                highlighted_cells=uncovered_cells,
                meta_extra={
                    "phase": "adjust_uncovered",
                    "iteration": iteration,
                    "h": float(h),
                    "row_covered": list(cover_l),
                    "col_covered": list(cover_r),
                    "starred": starred,
                    "lines": lines,
                    "line_count": num_lines,
                    "matching_edges": [[r, c] for r, c in assignment],
                },
                legend_items=legend_base("adjust_uncovered") + [f"h = {h:g}", f"Cells updated: {len(uncovered_cells)}"],
            )
        )

        doubly_cells: list[list[int]] = []
        for i in range(n):
            for j in range(n):
                if cover_l[i] and cover_r[j]:
                    matrix_after_uncovered[i][j] += h
                    doubly_cells.append([i, j])

        matrix_work = matrix_after_uncovered

        frames.append(
            _frame(
                explanation=(
                    f"What (iteration {iteration}): add h = {h:g} to each doubly-covered cell "
                    "(covered row AND covered column). "
                    "Why: this keeps the dual bookkeeping balanced so the transformation preserves optimality of the "
                    "true assignment while nudging the zero pattern so the next matching can include more edges."
                ),
                matrix_values=matrix_work,
                highlighted_cells=doubly_cells,
                meta_extra={
                    "phase": "adjust_double",
                    "iteration": iteration,
                    "h": float(h),
                    "row_covered": list(cover_l),
                    "col_covered": list(cover_r),
                    "starred": starred,
                    "lines": lines,
                    "line_count": num_lines,
                    "matching_edges": [[r, c] for r, c in assignment],
                },
                legend_items=legend_base("adjust_double")
                + [f"h = {h:g}", f"Doubly-covered cells updated: {len(doubly_cells)}"],
            )
        )


def validate_result(input_data: dict[str, Any], final_state: dict[str, Any]) -> ValidationReport:
    """
    Validate a terminal state.

    Supports:
    - Direct payload: {"assignment": ..., "objective": ...}
    - Wrapped payload: {"meta": {"final_state": {...}}}
    """

    mode = str(input_data.get("mode", "min"))
    if mode not in {"min", "max"}:
        return ValidationReport(ok=False, message="Invalid mode; expected 'min' or 'max'.")

    payload: dict[str, Any]
    if isinstance(final_state, dict) and "meta" in final_state and isinstance(final_state["meta"], dict):
        payload = final_state["meta"].get("final_state", final_state)
    else:
        payload = final_state

    values_raw = _extract_matrix_values(input_data)
    matrix_orig = _as_float_matrix(values_raw)
    n = _require_square_matrix(matrix_orig)

    assignment_raw = payload.get("assignment", [])
    assignment: list[tuple[int, int]] = []
    for pair in assignment_raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        assignment.append((int(pair[0]), int(pair[1])))

    details: dict[str, Any] = {"issues": []}

    if len(assignment) != n:
        details["issues"].append(f"Expected {n} assignments, got {len(assignment)}.")

    rows = [r for r, _ in assignment]
    cols = [c for _, c in assignment]
    if len(set(rows)) != len(rows):
        details["issues"].append("Duplicate row in assignment (not one-to-one).")
    if len(set(cols)) != len(cols):
        details["issues"].append("Duplicate column in assignment (not one-to-one).")
    for r, c in assignment:
        if r < 0 or r >= n or c < 0 or c >= n:
            details["issues"].append(f"Assignment out of bounds: ({r},{c}) with n={n}.")

    obj_reported = payload.get("objective", None)
    try:
        obj_computed = _objective_original(matrix_orig, assignment, mode)
    except Exception as exc:  # pragma: no cover
        details["issues"].append(f"Failed to compute objective: {exc}")
        obj_computed = None

    details["objective_reported"] = obj_reported
    details["objective_computed"] = obj_computed

    if obj_computed is not None and obj_reported is not None:
        if not math.isclose(float(obj_reported), float(obj_computed), rel_tol=0.0, abs_tol=1e-9):
            details["issues"].append(
                f"Objective mismatch: reported {float(obj_reported):g}, computed {float(obj_computed):g}."
            )

    ok = len(details["issues"]) == 0
    msg = "All checks passed." if ok else "; ".join(details["issues"])
    return ValidationReport(ok=ok, message=msg, details=details)
