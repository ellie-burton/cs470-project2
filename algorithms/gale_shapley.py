"""
Gale-Shapley stable matching with explicit per-event animation frames.

Input schema (input_data):
    proposers: list[str]
    receivers: list[str]
    proposer_prefs: list[list[str]] — each inner list is a total order over receivers
    receiver_prefs: list[list[str]] — each inner list is a total order over proposers

Digits-only preference rows are accepted as a convenience; they map to names in sorted order of the counterpart set.

Config (config):
    proposer_side: "proposers" | "receivers" (default "proposers")
        If "receivers", receivers propose to proposers (swap sides).

final_state (from last frame meta["final_state"]) for validate_result:
    matching: dict[str, str] — proposer_name -> receiver_name for the chosen proposer side
    proposer_side: "proposers" | "receivers"

UI hints (coordinate with Agent 1):
    meta["event"] in {"init","proposal","tentative_accept","reject","swap","done"}
    meta["event_color_hint"] in {"info","proposal","accept","reject","swap"}
    — consumed by UI rendering for event colors/highlights.

State panel (main app reads these for Gale–Shapley):
    meta["proposer_ids"], meta["receiver_ids"], meta["free_proposer_names"],
    meta["engagement_pairs"] (list of "P–R" strings), meta["engagements_by_index"] (dict)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from src.models import Frame, ValidationReport


PRESETS: dict[str, dict[str, Any]] = {
    "gs_no_swap": {
        "proposers": ["M1", "M2"],
        "receivers": ["W1", "W2"],
        "proposer_prefs": [
            ["W1", "W2"],
            ["W1", "W2"],
        ],
        "receiver_prefs": [
            ["M1", "M2"],
            ["M1", "M2"],
        ],
    },
    "gs_swaps": {
        # Classic 3x3 instance with multiple rejections and a partner swap.
        "proposers": ["M1", "M2", "M3"],
        "receivers": ["W1", "W2", "W3"],
        "proposer_prefs": [
            ["W1", "W2", "W3"],
            ["W2", "W1", "W3"],
            ["W2", "W3", "W1"],
        ],
        "receiver_prefs": [
            ["M3", "M1", "M2"],
            # Ensure W2 prefers M3 over M2 so the men-proposing run
            # triggers at least one real engagement replacement ("swap").
            ["M3", "M2", "M1"],
            ["M3", "M2", "M1"],
        ],
    },
    "gs_women_propose": {
        # Same data as gs_swaps. Prefer the app “Who proposes?” control (receivers) or this __config__ for scripts.
        "__config__": {"proposer_side": "receivers"},
        "proposers": ["M1", "M2", "M3"],
        "receivers": ["W1", "W2", "W3"],
        "proposer_prefs": [
            ["W1", "W2", "W3"],
            ["W2", "W1", "W3"],
            ["W2", "W3", "W1"],
        ],
        "receiver_prefs": [
            ["M3", "M1", "M2"],
            ["M2", "M3", "M1"],
            ["M3", "M2", "M1"],
        ],
    },
    "gs_asymmetric": {
        "proposers": ["A", "B", "C", "D"],
        "receivers": ["R1", "R2", "R3", "R4"],
        "proposer_prefs": [
            ["R2", "R1", "R3", "R4"],
            ["R1", "R2", "R4", "R3"],
            ["R1", "R3", "R2", "R4"],
            ["R4", "R3", "R2", "R1"],
        ],
        "receiver_prefs": [
            ["C", "A", "D", "B"],
            ["A", "B", "C", "D"],
            ["D", "C", "B", "A"],
            ["B", "D", "A", "C"],
        ],
    },
}


def _as_str_list(xs: Iterable[Any]) -> list[str]:
    return [str(x) for x in xs]


def _validate_total_order(prefs: Sequence[str], universe: set[str], *, who: str) -> None:
    if len(prefs) != len(universe):
        raise ValueError(f"{who}: preference list length {len(prefs)} != {len(universe)}")
    seen: set[str] = set()
    for x in prefs:
        if x not in universe:
            raise ValueError(f"{who}: unknown name {x!r} (not in universe)")
        if x in seen:
            raise ValueError(f"{who}: duplicate entry {x!r} in preference list")
        seen.add(x)
    if seen != universe:
        missing = sorted(universe - seen)
        raise ValueError(f"{who}: preference list missing {missing}")


def _normalize_prefs_named(
    agents: Sequence[str],
    prefs_in: Sequence[Sequence[Any]],
    universe: set[str],
    *,
    role: str,
) -> list[list[str]]:
    if len(prefs_in) != len(agents):
        raise ValueError(f"{role}: expected {len(agents)} preference rows, got {len(prefs_in)}")
    out: list[list[str]] = []
    ordered = sorted(universe)
    for i, row in enumerate(prefs_in):
        row_s = _as_str_list(row)
        if row_s and all(x.isdigit() for x in row_s):
            idxs = [int(x) for x in row_s]
            mapped: list[str] = []
            for j in idxs:
                if j < 0 or j >= len(ordered):
                    raise ValueError(f"{role}: bad index {j} for {agents[i]!r}")
                mapped.append(ordered[j])
            _validate_total_order(mapped, universe, who=f"{role}[{agents[i]!r}] prefs")
            out.append(mapped)
            continue
        _validate_total_order(row_s, universe, who=f"{role}[{agents[i]!r}] prefs")
        out.append(row_s)
    return out


def _invert_rank(prefs: Sequence[str]) -> dict[str, int]:
    return {name: rank for rank, name in enumerate(prefs)}


def _layout_bipartite(proposer_ids: Sequence[str], receiver_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    y0 = 60
    gap = 70
    for i, pid in enumerate(proposer_ids):
        nodes.append({"id": pid, "x": 80, "y": y0 + i * gap, "label": pid, "side": "proposer"})
    for j, rid in enumerate(receiver_ids):
        nodes.append({"id": rid, "x": 420, "y": y0 + j * gap, "label": rid, "side": "receiver"})
    return {"nodes": nodes}


def _prefs_matrix_rows(
    agents: Sequence[str], prefs: Sequence[Sequence[str]], counterpart_ids: Sequence[str]
) -> list[list[str]]:
    idx = {c: i for i, c in enumerate(counterpart_ids)}
    rows: list[list[str]] = []
    # Python 3.9 compatibility: `zip(strict=True)` is a Python 3.10+ feature.
    # We still validate that the shapes match to keep debugging straightforward.
    if len(agents) != len(prefs):
        raise ValueError(f"expected {len(agents)} preference rows, got {len(prefs)}")
    for a, row in zip(agents, prefs):
        ranks = ["—"] * len(counterpart_ids)
        for rnk, name in enumerate(row, start=1):
            ranks[idx[name]] = str(rnk)
        rows.append([a, *ranks])
    return rows


def _event_color_hint(event: str) -> str:
    return {
        "init": "info",
        "proposal": "proposal",
        "tentative_accept": "accept",
        "reject": "reject",
        "swap": "swap",
        "done": "info",
    }.get(event, "info")


def _engagement_edges(engaged: dict[int, int], proposer_ids: Sequence[str], receiver_ids: Sequence[str]) -> list[list[str]]:
    edges: list[list[str]] = []
    for pi, ri in engaged.items():
        edges.append([proposer_ids[pi], receiver_ids[ri]])
    return edges


def _pretty_matching(engaged: dict[int, int], proposer_ids: Sequence[str], receiver_ids: Sequence[str]) -> str:
    parts = [f"{proposer_ids[pi]}–{receiver_ids[ri]}" for pi, ri in sorted(engaged.items(), key=lambda kv: kv[0])]
    return ", ".join(parts) if parts else "(none)"


def build_frames(input_data: dict[str, Any], config: dict[str, Any]) -> list[Frame]:
    proposer_side = str(config.get("proposer_side", "proposers"))
    if proposer_side not in {"proposers", "receivers"}:
        raise ValueError('config["proposer_side"] must be "proposers" or "receivers"')

    men = _as_str_list(input_data["proposers"])
    women = _as_str_list(input_data["receivers"])
    if len(men) != len(women):
        raise ValueError("Gale-Shapley animation expects |proposers| == |receivers| (one-to-one matching).")
    n = len(men)

    women_set = set(women)
    men_set = set(men)
    proposer_prefs_named = _normalize_prefs_named(men, input_data["proposer_prefs"], women_set, role="proposer_prefs")
    receiver_prefs_named = _normalize_prefs_named(women, input_data["receiver_prefs"], men_set, role="receiver_prefs")

    if proposer_side == "proposers":
        proposer_ids = men
        receiver_ids = women
        proposer_prefs = proposer_prefs_named
        receiver_prefs = receiver_prefs_named
        legend_note = "Proposer side: proposers (standard Gale-Shapley)."
    else:
        proposer_ids = women
        receiver_ids = men
        proposer_prefs = receiver_prefs_named
        receiver_prefs = proposer_prefs_named
        legend_note = "Proposer side: receivers (roles swapped)."

    receiver_rank_of_proposer = [_invert_rank(row) for row in receiver_prefs]

    layout = _layout_bipartite(proposer_ids, receiver_ids)
    matrix_values = _prefs_matrix_rows(proposer_ids, proposer_prefs, receiver_ids)

    def matrix_frame(h_row: int | None, h_col: int | None) -> dict[str, Any]:
        highlighted: list[list[int]] = []
        if h_row is not None and h_col is not None:
            highlighted.append([h_row, h_col])
        return {
            "title": "Proposer preference ranks (row=proposer, cols=receivers)",
            "row_labels": False,
            "column_labels": ["proposer", *receiver_ids],
            "values": matrix_values,
            "highlighted_cells": highlighted,
        }

    frames: list[Frame] = []

    def push(
        explanation: str,
        *,
        event: str,
        engaged: dict[int, int],
        free: deque[int],
        next_idx: list[int],
        h_nodes: list[str],
        h_edges: list[list[str]],
        proposal_edge: list[str] | None,
        matrix_highlight: tuple[int | None, int | None],
        extra_legend: list[str] | None = None,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        engaged_edges = _engagement_edges(engaged, proposer_ids, receiver_ids)
        legend_items = [
            legend_note,
            f"Event: {event}",
            f"Engagements: {_pretty_matching(engaged, proposer_ids, receiver_ids)}",
            f"Free proposers queue: [{', '.join(proposer_ids[i] for i in free)}]",
        ]
        if extra_legend:
            legend_items.extend(extra_legend)

        graph = {
            **layout,
            "edges": [{"source": u, "target": v, "kind": "engaged"} for u, v in engaged_edges],
            "highlighted_nodes": list(h_nodes),
            "highlighted_edges": [list(e) for e in h_edges],
            "directed": True,
        }
        if proposal_edge is not None:
            graph["edges"].append({"source": proposal_edge[0], "target": proposal_edge[1], "kind": "proposal"})

        free_names = [proposer_ids[i] for i in free]
        engagement_pairs = [f"{proposer_ids[pi]}–{receiver_ids[ri]}" for pi, ri in sorted(engaged.items(), key=lambda kv: kv[0])]
        meta: dict[str, Any] = {
            "algorithm": "gale_shapley",
            "event": event,
            "event_color_hint": _event_color_hint(event),
            "proposer_side": proposer_side,
            "proposer_ids": list(proposer_ids),
            "receiver_ids": list(receiver_ids),
            "free_proposer_indices": list(free),
            "free_proposer_names": free_names,
            "next_proposal_index": list(next_idx),
            "engagement_pairs": engagement_pairs,
            "engagements_by_index": {proposer_ids[pi]: receiver_ids[ri] for pi, ri in engaged.items()},
        }
        if meta_extra:
            meta.update(meta_extra)

        frames.append(
            Frame(
                explanation=explanation,
                graph=graph,
                matrix=matrix_frame(*matrix_highlight),
                legend={"status": event, "items": legend_items},
                meta=meta,
            )
        )

    free = deque(range(n))
    next_idx = [0] * n
    receiver_fiance: list[int | None] = [None] * n
    engaged: dict[int, int] = {}

    push(
        "Initialize Gale-Shapley: every proposer is free, engagements are empty, and each proposer will propose "
        "down their preference list until everyone is matched.",
        event="init",
        engaged={},
        free=deque(free),
        next_idx=list(next_idx),
        h_nodes=[],
        h_edges=[],
        proposal_edge=None,
        matrix_highlight=(None, None),
        extra_legend=[
            f"n={n}",
            "Watch the free-proposer queue, proposal edges, and how engagements change after rejections/swaps.",
            "Graph edge colors: green = tentative engagement; orange = proposal arrow; yellow bold = this step highlight.",
        ],
    )

    while free:
        p = free.popleft()
        if next_idx[p] >= n:
            raise RuntimeError("Internal error: free proposer exhausted preferences (should not happen for complete prefs).")
        r_name = proposer_prefs[p][next_idx[p]]
        r = receiver_ids.index(r_name)
        next_idx[p] += 1

        prop_edge = [proposer_ids[p], receiver_ids[r]]
        push(
            f"Proposal: {proposer_ids[p]} offers to {receiver_ids[r]} — the next name on {proposer_ids[p]}'s preference list.",
            event="proposal",
            engaged=dict(engaged),
            free=deque(free),
            next_idx=list(next_idx),
            h_nodes=[proposer_ids[p], receiver_ids[r]],
            h_edges=[],
            proposal_edge=prop_edge,
            matrix_highlight=(p, r + 1),
            meta_extra={"proposer": proposer_ids[p], "receiver": receiver_ids[r]},
        )

        cur = receiver_fiance[r]
        if cur is None:
            receiver_fiance[r] = p
            engaged[p] = r
            push(
                f"Accept (free receiver): {receiver_ids[r]} had no fiancé, so the proposal is accepted tentatively. "
                f"New engagement: {proposer_ids[p]}–{receiver_ids[r]}.",
                event="tentative_accept",
                engaged=dict(engaged),
                free=deque(free),
                next_idx=list(next_idx),
                h_nodes=[proposer_ids[p], receiver_ids[r]],
                h_edges=[prop_edge],
                proposal_edge=None,
                matrix_highlight=(p, r + 1),
                meta_extra={"proposer": proposer_ids[p], "receiver": receiver_ids[r], "reason": "receiver_was_free"},
            )
            continue

        if receiver_rank_of_proposer[r][proposer_ids[p]] < receiver_rank_of_proposer[r][proposer_ids[cur]]:
            push(
                f"Swap decision: {receiver_ids[r]} compares {proposer_ids[p]} (new) vs {proposer_ids[cur]} (current). "
                f"By {receiver_ids[r]}'s preferences, {proposer_ids[p]} ranks higher — {proposer_ids[cur]} will be dropped.",
                event="swap",
                engaged=dict(engaged),
                free=deque(free),
                next_idx=list(next_idx),
                h_nodes=[proposer_ids[p], proposer_ids[cur], receiver_ids[r]],
                h_edges=[prop_edge, [proposer_ids[cur], receiver_ids[r]]],
                proposal_edge=prop_edge,
                matrix_highlight=(p, r + 1),
                meta_extra={
                    "proposer": proposer_ids[p],
                    "receiver": receiver_ids[r],
                    "rejected_proposer": proposer_ids[cur],
                    "reason": "receiver_prefers_new_proposer",
                },
            )

            del engaged[cur]
            free.append(cur)

            receiver_fiance[r] = p
            engaged[p] = r

            push(
                f"Rejection (after swap): {receiver_ids[r]} ends the engagement with {proposer_ids[cur]} in favor of {proposer_ids[p]}. "
                f"{proposer_ids[cur]} is free and will move to their next choice.",
                event="reject",
                engaged=dict(engaged),
                free=deque(free),
                next_idx=list(next_idx),
                h_nodes=[proposer_ids[cur], receiver_ids[r]],
                h_edges=[[proposer_ids[cur], receiver_ids[r]]],
                proposal_edge=None,
                matrix_highlight=(cur, r + 1),
                meta_extra={
                    "proposer": proposer_ids[cur],
                    "receiver": receiver_ids[r],
                    "reason": "dumped_during_swap",
                },
            )

            push(
                f"Accept (after swap): {receiver_ids[r]} is now tentatively engaged to {proposer_ids[p]} "
                f"({proposer_ids[p]}–{receiver_ids[r]}). All engagements: {_pretty_matching(engaged, proposer_ids, receiver_ids)}.",
                event="tentative_accept",
                engaged=dict(engaged),
                free=deque(free),
                next_idx=list(next_idx),
                h_nodes=[proposer_ids[p], receiver_ids[r]],
                h_edges=[[proposer_ids[p], receiver_ids[r]]],
                proposal_edge=None,
                matrix_highlight=(p, r + 1),
                meta_extra={"proposer": proposer_ids[p], "receiver": receiver_ids[r], "reason": "new_engagement_after_swap"},
            )
            continue

        push(
            f"{receiver_ids[r]} compares suitors {proposer_ids[p]} vs incumbent {proposer_ids[cur]}: "
            f"{proposer_ids[cur]} ranks higher in {receiver_ids[r]}'s list, so {proposer_ids[p]} is rejected.",
            event="reject",
            engaged=dict(engaged),
            free=deque(free),
            next_idx=list(next_idx),
            h_nodes=[proposer_ids[p], proposer_ids[cur], receiver_ids[r]],
            h_edges=[[proposer_ids[cur], receiver_ids[r]]],
            proposal_edge=prop_edge,
            matrix_highlight=(p, r + 1),
            meta_extra={
                "proposer": proposer_ids[p],
                "receiver": receiver_ids[r],
                "incumbent": proposer_ids[cur],
                "reason": "receiver_prefers_incumbent",
            },
        )
        free.append(p)

    final_matching = {proposer_ids[pi]: receiver_ids[ri] for pi, ri in engaged.items()}
    report = validate_result(input_data, {"matching": final_matching, "proposer_side": proposer_side})

    push(
        "Done: every proposer is matched; no one remains free. "
        f"Final tentative matching: {_pretty_matching(engaged, proposer_ids, receiver_ids)}. "
        "Stability is checked in the Validation line (no blocking pairs).",
        event="done",
        engaged=dict(engaged),
        free=deque(),
        next_idx=list(next_idx),
        h_nodes=list(proposer_ids) + list(receiver_ids),
        h_edges=_engagement_edges(engaged, proposer_ids, receiver_ids),
        proposal_edge=None,
        matrix_highlight=(None, None),
        extra_legend=[
            f"Validation: {'PASS' if report.ok else 'FAIL'} — {report.message}",
        ],
        meta_extra={
            "final_state": {"matching": final_matching, "proposer_side": proposer_side},
            "validation": {"ok": report.ok, "message": report.message, "details": report.details},
        },
    )

    return frames


def validate_result(input_data: dict[str, Any], final_state: dict[str, Any]) -> ValidationReport:
    men = _as_str_list(input_data["proposers"])
    women = _as_str_list(input_data["receivers"])
    if len(men) != len(women):
        return ValidationReport(ok=False, message="|proposers| must equal |receivers| for one-to-one stability check.")

    women_set = set(women)
    men_set = set(men)
    proposer_prefs_named = _normalize_prefs_named(men, input_data["proposer_prefs"], women_set, role="proposer_prefs")
    receiver_prefs_named = _normalize_prefs_named(women, input_data["receiver_prefs"], men_set, role="receiver_prefs")

    matching_in = final_state.get("matching", {})
    if not isinstance(matching_in, dict) or not matching_in:
        return ValidationReport(ok=False, message='final_state["matching"] must be a non-empty dict.')

    side = str(final_state.get("proposer_side", "proposers"))
    if side not in {"proposers", "receivers"}:
        return ValidationReport(ok=False, message='final_state["proposer_side"] must be "proposers" or "receivers".')

    if side == "proposers":
        proposer_ids = men
        receiver_ids = women
        proposer_prefs = proposer_prefs_named
        receiver_prefs = receiver_prefs_named
    else:
        proposer_ids = women
        receiver_ids = men
        proposer_prefs = receiver_prefs_named
        receiver_prefs = proposer_prefs_named

    n = len(proposer_ids)
    pi = {name: i for i, name in enumerate(proposer_ids)}
    ri = {name: i for i, name in enumerate(receiver_ids)}

    engaged: dict[int, int] = {}
    issues: list[str] = []

    for p_name, r_name in matching_in.items():
        p_name = str(p_name)
        r_name = str(r_name)
        if p_name not in pi:
            issues.append(f"Unknown proposer {p_name!r} in matching")
            continue
        if r_name not in ri:
            issues.append(f"Unknown receiver {r_name!r} in matching")
            continue
        engaged[pi[p_name]] = ri[r_name]

    if len(engaged) != n:
        issues.append(f"Matching size {len(engaged)} != n={n}")
    if len(set(engaged.keys())) != len(engaged):
        issues.append("A proposer is matched more than once.")
    if len(set(engaged.values())) != len(engaged.values()):
        issues.append("A receiver is matched more than once.")

    proposer_rank_of_receiver = [_invert_rank(row) for row in proposer_prefs]
    receiver_rank_of_proposer = [_invert_rank(row) for row in receiver_prefs]

    inv_receiver_for_proposer = {p: r for p, r in engaged.items()}
    inv_proposer_for_receiver = {r: p for p, r in engaged.items()}

    blocking: list[str] = []
    for m in range(n):
        m_partner = inv_receiver_for_proposer.get(m, None)
        if m_partner is None:
            continue
        for w in range(n):
            p_w = inv_proposer_for_receiver.get(w, None)
            if p_w is None:
                continue
            if proposer_rank_of_receiver[m][receiver_ids[w]] >= proposer_rank_of_receiver[m][receiver_ids[m_partner]]:
                continue
            if receiver_rank_of_proposer[w][proposer_ids[p_w]] <= receiver_rank_of_proposer[w][proposer_ids[m]]:
                continue
            blocking.append(f"{proposer_ids[m]} and {receiver_ids[w]} form a blocking pair")

    details: dict[str, Any] = {
        "proposer_side": side,
        "n": n,
        "blocking_pairs": blocking,
        "issues": issues,
    }
    ok = (len(issues) == 0) and (len(blocking) == 0)
    msg = "Stable and complete matching." if ok else ("; ".join(issues) if issues else "Unstable: blocking pairs exist.")
    return ValidationReport(ok=ok, message=msg, details=details)
