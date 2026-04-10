"""
Dinic's max-flow algorithm with step frames for BFS level graph and DFS blocking flow.

Input schema (input_data):
    source: str  — source node id
    sink: str    — sink node id
    graph: {
        "nodes": [{"id": str, "x": float, "y": float, "label": str?}, ...],
        "edges": [{"source": str, "target": str, "capacity": int}, ...],
    }

final_state (from last frame meta["final_state"]) for validate_result:
    max_flow: int
    edge_flows: dict[tuple[str, str], int]  — flow on each original forward edge
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from src.models import Frame, ValidationReport

PRESETS: dict[str, dict[str, Any]] = {
    "Small (s \u2192 t, 4 nodes)": {
        "source": "s",
        "sink": "t",
        "graph": {
            "nodes": [
                {"id": "s", "x": 60, "y": 120, "label": "s"},
                {"id": "a", "x": 180, "y": 60, "label": "a"},
                {"id": "b", "x": 180, "y": 180, "label": "b"},
                {"id": "t", "x": 300, "y": 120, "label": "t"},
            ],
            "edges": [
                {"source": "s", "target": "a", "capacity": 10},
                {"source": "s", "target": "b", "capacity": 10},
                {"source": "a", "target": "b", "capacity": 2},
                {"source": "a", "target": "t", "capacity": 10},
                {"source": "b", "target": "t", "capacity": 10},
            ],
        },
    },
    "Extended (s \u2192 f, 5 nodes)": {
        "source": "s",
        "sink": "f",
        "graph": {
            "nodes": [
                {"id": "s", "x": 60, "y": 120, "label": "s"},
                {"id": "a", "x": 180, "y": 60, "label": "a"},
                {"id": "b", "x": 180, "y": 180, "label": "b"},
                {"id": "t", "x": 300, "y": 120, "label": "t"},
                {"id": "f", "x": 400, "y": 120, "label": "f"},
            ],
            "edges": [
                {"source": "s", "target": "a", "capacity": 12},
                {"source": "s", "target": "b", "capacity": 8},
                {"source": "a", "target": "b", "capacity": 2},
                {"source": "a", "target": "t", "capacity": 10},
                {"source": "b", "target": "t", "capacity": 10},
                {"source": "t", "target": "f", "capacity": 30},
            ],
        },
    },
    "CLRS Textbook (s \u2192 t, 6 nodes)": {
        "source": "s",
        "sink": "t",
        "graph": {
            "nodes": [
                {"id": "s", "x": 60, "y": 150, "label": "s"},
                {"id": "v1", "x": 200, "y": 60, "label": "v1"},
                {"id": "v2", "x": 200, "y": 240, "label": "v2"},
                {"id": "v3", "x": 340, "y": 60, "label": "v3"},
                {"id": "v4", "x": 340, "y": 240, "label": "v4"},
                {"id": "t", "x": 480, "y": 150, "label": "t"},
            ],
            "edges": [
                {"source": "s", "target": "v1", "capacity": 16},
                {"source": "s", "target": "v2", "capacity": 13},
                {"source": "v1", "target": "v2", "capacity": 4},
                {"source": "v1", "target": "v3", "capacity": 12},
                {"source": "v2", "target": "v1", "capacity": 4},
                {"source": "v2", "target": "v4", "capacity": 14},
                {"source": "v3", "target": "v2", "capacity": 9},
                {"source": "v3", "target": "t", "capacity": 20},
                {"source": "v4", "target": "v3", "capacity": 7},
                {"source": "v4", "target": "t", "capacity": 4},
            ],
        },
    },
}


def _edge_flows_from_residual(
    resid: dict[str, dict[str, int]], originals: list[tuple[str, str, int]]
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for u, v, c in originals:
        out[(u, v)] = c - resid[u][v]
    return out


def _max_flow_value(resid: dict[str, dict[str, int]], source: str, originals: list[tuple[str, str, int]]) -> int:
    total = 0
    for u, v, c in originals:
        if u == source:
            total += c - resid[u][v]
    return total


def _build_adj_residual(resid: dict[str, dict[str, int]], nodes: list[str]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    seen: set[tuple[str, str]] = set()
    for u in nodes:
        for v, cap in resid[u].items():
            if cap > 0 and (u, v) not in seen:
                adj[u].append(v)
                seen.add((u, v))
    for u in nodes:
        adj[u].sort()
    return adj


def _snapshot_graph(
    layout: dict[str, Any],
    originals: list[tuple[str, str, int]],
    resid: dict[str, dict[str, int]],
    *,
    highlighted_nodes: list[str],
    highlighted_edges: list[list[str]],
    phase: str,
    running_flow: int,
    extra_legend: list[str] | None = None,
    level: dict[str, int] | None = None,
    show_admissible: bool = False,
) -> dict[str, Any]:
    nodes_spec = layout["nodes"]
    edges_out: list[dict[str, Any]] = []
    for u, v, c in originals:
        f = c - resid[u][v]
        edge_info: dict[str, Any] = {
            "source": u,
            "target": v,
            "label": f"{f}/{c}",
        }
        if show_admissible and level and u in level and v in level:
            if level[v] == level[u] + 1 and resid[u][v] > 0:
                edge_info["admissible"] = True
        edges_out.append(edge_info)
    legend_items = [
        f"Phase: {phase}",
        f"Running max flow (value): {running_flow}",
    ]
    if extra_legend:
        legend_items.extend(extra_legend)
    return {
        "nodes": list(nodes_spec),
        "edges": edges_out,
        "highlighted_nodes": list(highlighted_nodes),
        "highlighted_edges": [list(e) for e in highlighted_edges],
        "directed": True,
    }


def build_frames(input_data: dict[str, Any], config: dict[str, Any]) -> list[Frame]:
    """Run Dinic and emit frames for BFS level construction and each blocking-flow augment."""
    _ = config
    source = str(input_data["source"])
    sink = str(input_data["sink"])
    g = input_data["graph"]
    nodes_spec = list(g["nodes"])
    node_ids = [str(n["id"]) for n in nodes_spec]
    id_set = set(node_ids)

    raw_edges: list[tuple[str, str, int]] = []
    for e in g["edges"]:
        u, v = str(e["source"]), str(e["target"])
        c = int(e["capacity"])
        if u not in id_set or v not in id_set:
            raise ValueError(f"Edge ({u}->{v}) references unknown node")
        raw_edges.append((u, v, c))

    merged_cap: dict[tuple[str, str], int] = defaultdict(int)
    for u, v, c in raw_edges:
        merged_cap[(u, v)] += c
    originals = [(u, v, c) for (u, v), c in sorted(merged_cap.items())]

    resid: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for u, v, c in originals:
        resid[u][v] += c

    frames: list[Frame] = []
    round_idx = 0
    running = 0

    def push_frame(
        explanation: str,
        *,
        h_nodes: list[str],
        h_edges: list[list[str]],
        phase: str,
        legend_extra: list[str] | None = None,
        level: dict[str, int] | None = None,
        show_admissible: bool = False,
        bottleneck_edge: list[str] | None = None,
    ) -> None:
        nonlocal running
        running = _max_flow_value(resid, source, originals)
        leg = list(legend_extra or [])
        if level:
            lvl_str = ", ".join(f"{nid}={level[nid]}" for nid in sorted(level, key=lambda x: (level[x], x)))
            leg.append(f"Levels (BFS dist from {source}): {lvl_str}")
        graph = _snapshot_graph(
            g,
            originals,
            resid,
            highlighted_nodes=h_nodes,
            highlighted_edges=h_edges,
            phase=phase,
            running_flow=running,
            extra_legend=leg,
            level=level,
            show_admissible=show_admissible,
        )
        edge_flows = _edge_flows_from_residual(resid, originals)
        node_levels_meta = dict(level) if level else {}
        meta_dict: dict[str, Any] = {
            "algorithm": "dinic",
            "phase": phase,
            "round": round_idx,
            "running_max_flow": running,
            "current_flow": running,
            "bfs_round": round_idx,
            "node_levels": node_levels_meta,
            "final_state": {"max_flow": running, "edge_flows": edge_flows},
        }
        if bottleneck_edge:
            meta_dict["bottleneck_edge"] = bottleneck_edge
        frames.append(
            Frame(
                explanation=explanation,
                graph=graph,
                legend={"status": phase, "items": leg},
                meta=meta_dict,
            )
        )

    push_frame(
        f"Initialize residual network. Source={source}, sink={sink}. "
        f"Forward edge labels show flow/capacity; algorithm is Dinic (level graph + blocking flow).",
        h_nodes=[source],
        h_edges=[],
        phase="init",
        legend_extra=[f"Nodes: {len(node_ids)}, original edges: {len(originals)}"],
    )

    while True:
        round_idx += 1

        push_frame(
            f"Starting BFS round {round_idx}: build level graph from source {source} "
            f"using edges with remaining residual capacity.",
            h_nodes=[source],
            h_edges=[],
            phase="bfs_start",
            legend_extra=[f"Round {round_idx}: BFS from {source}"],
        )

        # --- BFS level graph ---
        level: dict[str, int] = {source: 0}
        q: deque[str] = deque([source])

        while q:
            u = q.popleft()
            for v in sorted(resid[u].keys()):
                cap = resid[u][v]
                if cap <= 0 or v in level:
                    continue
                level[v] = level[u] + 1
                q.append(v)
                push_frame(
                    f"BFS (round {round_idx}): discovered {v} at level {level[v]} via edge {u}→{v} "
                    f"(residual capacity {cap}). Admissible edges go to the next level only.",
                    h_nodes=[u, v],
                    h_edges=[[u, v]],
                    phase="bfs",
                    level=level,
                    legend_extra=[f"Target level for {sink}: {level.get(sink, '—')}"],
                )

        if sink not in level:
            final_flow = _max_flow_value(resid, source, originals)
            ef = _edge_flows_from_residual(resid, originals)
            push_frame(
                f"No augmenting level path to {sink}. Max flow = {final_flow}. "
                f"Capacity constraints and conservation are checked in validate_result.",
                h_nodes=[source, sink],
                h_edges=[],
                phase="done",
                legend_extra=["BFS: sink unreachable in residual graph"],
            )
            if frames:
                frames[-1].meta["final_state"] = {"max_flow": final_flow, "edge_flows": ef}
            break

        # Admissible-edge frame: highlight all edges in the level graph
        admissible_edges: list[list[str]] = []
        for u2, v2, _ in originals:
            if u2 in level and v2 in level and level[v2] == level[u2] + 1 and resid[u2][v2] > 0:
                admissible_edges.append([u2, v2])
        push_frame(
            f"BFS complete (round {round_idx}): level graph reaches {sink} at level {level[sink]}. "
            f"Admissible edges (level[v]=level[u]+1 with residual>0) are highlighted in blue. "
            f"Next: DFS pushes blocking flow along these edges.",
            h_nodes=[source, sink],
            h_edges=admissible_edges,
            phase="bfs_done",
            level=level,
            show_admissible=True,
            legend_extra=[f"Admissible edges: {len(admissible_edges)}"],
        )

        # --- Blocking flow DFS ---
        adj = _build_adj_residual(resid, node_ids)
        ptr: dict[str, int] = defaultdict(int)

        def dfs_augment(u: str, flow_limit: int, path: list[str]) -> int:
            if u == sink:
                return flow_limit
            while ptr[u] < len(adj[u]):
                v = adj[u][ptr[u]]
                if level.get(v, -1) != level[u] + 1:
                    ptr[u] += 1
                    continue
                cap = resid[u][v]
                if cap <= 0:
                    ptr[u] += 1
                    continue
                path.append(v)
                pushed = dfs_augment(v, min(flow_limit, cap), path)
                if pushed > 0:
                    resid[u][v] -= pushed
                    resid[v][u] += pushed
                    return pushed
                path.pop()
                ptr[u] += 1
            return 0

        aug_idx = 0
        while True:
            path = [source]
            add = dfs_augment(source, 10**18, path)
            if add == 0:
                break
            aug_idx += 1
            path_edges = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
            # Identify the bottleneck (minimum residual) edge on the augmenting path
            bn_edge: list[str] | None = None
            bn_cap = 10**18
            for i in range(len(path) - 1):
                pu, pv = path[i], path[i + 1]
                orig_resid = resid[pu][pv] + add
                if orig_resid < bn_cap:
                    bn_cap = orig_resid
                    bn_edge = [pu, pv]
            bn_label = ""
            if bn_edge:
                bn_label = f" Bottleneck edge: {bn_edge[0]}→{bn_edge[1]} (capacity limited push to {add})."
            push_frame(
                f"DFS blocking flow (round {round_idx}, augment #{aug_idx}): pushed {add} units along "
                f"path {' → '.join(path)}. Residual capacities updated on this path.{bn_label}",
                h_nodes=list(dict.fromkeys(path)),
                h_edges=path_edges,
                phase="dfs_push",
                level=level,
                show_admissible=True,
                legend_extra=[f"Augment amount: {add}"],
                bottleneck_edge=bn_edge,
            )

        push_frame(
            f"Blocking flow round {round_idx} finished (no more admissible paths in this level graph). "
            f"Current total flow value: {_max_flow_value(resid, source, originals)}.",
            h_nodes=[source, sink],
            h_edges=[],
            phase="blocking_done",
            level=level,
        )

    return frames


def validate_result(input_data: dict[str, Any], final_state: dict[str, Any]) -> ValidationReport:
    """Check capacity limits, flow conservation, and report max flow."""
    source = str(input_data["source"])
    sink = str(input_data["sink"])
    g = input_data["graph"]
    node_ids = {str(n["id"]) for n in g["nodes"]}
    originals: list[tuple[str, str, int]] = []
    for e in g["edges"]:
        u, v = str(e["source"]), str(e["target"])
        originals.append((u, v, int(e["capacity"])))

    edge_flows_raw = final_state.get("edge_flows", {})
    edge_flows: dict[tuple[str, str], int] = {}
    for k, val in edge_flows_raw.items():
        if isinstance(k, tuple) and len(k) == 2:
            edge_flows[(str(k[0]), str(k[1]))] = int(val)
        elif isinstance(k, (list, tuple)) and len(k) == 2:
            edge_flows[(str(k[0]), str(k[1]))] = int(val)
        elif isinstance(k, str) and "->" in k:
            a, b = k.split("->", 1)
            edge_flows[(a.strip(), b.strip())] = int(val)

    merged_cap: dict[tuple[str, str], int] = defaultdict(int)
    for u, v, c in originals:
        merged_cap[(u, v)] += c
    originals_unique = [(u, v, c) for (u, v), c in sorted(merged_cap.items())]

    for u, v, _ in originals_unique:
        edge_flows.setdefault((u, v), 0)

    details: dict[str, Any] = {"issues": []}

    for u, v, c in originals_unique:
        f = edge_flows.get((u, v), 0)
        if f < 0 or f > c:
            details["issues"].append(f"capacity violated on {u}→{v}: flow {f}, cap {c}")

    excess: dict[str, int] = defaultdict(int)
    for n in node_ids:
        excess[n] = 0
    for u, v, _ in originals_unique:
        f = edge_flows.get((u, v), 0)
        excess[u] -= f
        excess[v] += f

    for n in node_ids:
        if n in (source, sink):
            continue
        if excess[n] != 0:
            details["issues"].append(f"flow conservation failed at {n}: net {excess[n]}")

    reported = int(final_state.get("max_flow", 0))
    out_source = sum(edge_flows.get((source, v), 0) for v in node_ids)
    into_sink = sum(edge_flows.get((u, sink), 0) for u in node_ids)

    details["reported_max_flow"] = reported
    details["sum_out_of_source"] = out_source
    details["sum_into_sink"] = into_sink

    if out_source != into_sink:
        details["issues"].append(f"source out ({out_source}) != sink in ({into_sink})")

    if reported != out_source or reported != into_sink:
        details["issues"].append(
            f"max_flow label mismatch: reported {reported}, source_out {out_source}, sink_in {into_sink}"
        )

    ok = len(details["issues"]) == 0
    msg = "All checks passed." if ok else "; ".join(details["issues"])
    return ValidationReport(ok=ok, message=msg, details=details)
