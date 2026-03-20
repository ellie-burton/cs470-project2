from __future__ import annotations

import copy
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms import dinic, gale_shapley, hungarian
from src.models import Frame

try:
    from streamlit_sortables import sort_items

    HAS_SORTABLES = True
except Exception:  # noqa: BLE001
    HAS_SORTABLES = False

AlgorithmSpec = tuple[str, dict[str, dict[str, Any]], Any, Any]

ALGORITHMS: dict[str, AlgorithmSpec] = {
    "Hungarian (assignment)": ("hungarian", hungarian.PRESETS, hungarian.build_frames, hungarian.validate_result),
    "Gale-Shapley (stable matching)": ("gale_shapley", gale_shapley.PRESETS, gale_shapley.build_frames, gale_shapley.validate_result),
    "Dinic (max flow)": ("dinic", dinic.PRESETS, dinic.build_frames, dinic.validate_result),
}


def _inject_modern_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.25rem; padding-bottom: 1rem; max-width: 1300px;}
          [data-testid="stMetricValue"] {font-size: 1.4rem;}
          .stButton>button, .stDownloadButton>button {border-radius: 12px;}
          [data-testid="stSidebar"] {background: linear-gradient(180deg, #101322 0%, #171b2f 100%);}
          .small-caption {color: #9AA3B2; font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "frames_by_algo": {},
        "frame_idx_by_algo": {},
        "is_playing_by_algo": {},
        "loaded_input_by_algo": {},
        "loaded_config_by_algo": {},
        "validation_by_algo": {},
        "h_data": {},
        "gs_data": {},
        "gs_rank_version": 0,
        "d_data": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _parse_csv_names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_pref_lines(text: str, expected_len: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([x.strip() for x in line.split(",") if x.strip()])
    if rows and len(rows) != expected_len:
        raise ValueError(f"Expected {expected_len} preference rows, got {len(rows)}.")
    return rows


def _normalize_pref_row(row: list[str], universe: list[str]) -> list[str]:
    cleaned = [x for x in row if x in universe]
    for item in universe:
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _rank_editor_block(
    *,
    header: str,
    agents: list[str],
    counterpart_ids: list[str],
    prefs: list[list[str]],
    key_prefix: str,
) -> list[list[str]]:
    st.markdown(f"#### {header}")
    out: list[list[str]] = []
    for i, agent in enumerate(agents):
        default_row = prefs[i] if i < len(prefs) else list(counterpart_ids)
        normalized = _normalize_pref_row(list(default_row), counterpart_ids)
        c_name, c_rank = st.columns([1, 4])
        with c_name:
            st.markdown(f"**{agent}**")
        with c_rank:
            if HAS_SORTABLES:
                ranked = sort_items(normalized, key=f"{key_prefix}_{i}", direction="horizontal")
                ranked = _normalize_pref_row([str(x) for x in ranked], counterpart_ids)
            else:
                ranked = st.multiselect(
                    "Ranking (left to right = high to low)",
                    options=counterpart_ids,
                    default=normalized,
                    key=f"{key_prefix}_fallback_{i}",
                    label_visibility="collapsed",
                )
                ranked = _normalize_pref_row(list(ranked), counterpart_ids)
            out.append([str(x) for x in ranked])
    return out


def _next_name(prefix: str, existing: list[str]) -> str:
    i = len(existing) + 1
    while f"{prefix}{i}" in existing:
        i += 1
    return f"{prefix}{i}"


def _add_gs_pair() -> None:
    data = copy.deepcopy(st.session_state.get("gs_data") or {})
    proposers = [str(x) for x in (data.get("proposers") or [])]
    receivers = [str(x) for x in (data.get("receivers") or [])]
    p_prefs = [list(map(str, row)) for row in (data.get("proposer_prefs") or [])]
    r_prefs = [list(map(str, row)) for row in (data.get("receiver_prefs") or [])]

    if not proposers and not receivers:
        proposers = ["M1"]
        receivers = ["W1"]
        p_prefs = [["W1"]]
        r_prefs = [["M1"]]
    else:
        new_p = _next_name("M", proposers)
        new_r = _next_name("W", receivers)
        proposers.append(new_p)
        receivers.append(new_r)

        for row in p_prefs:
            if new_r not in row:
                row.append(new_r)
        for row in r_prefs:
            if new_p not in row:
                row.append(new_p)

        p_prefs.append(list(receivers))
        r_prefs.append(list(proposers))

    p_prefs = [_normalize_pref_row(row, receivers) for row in p_prefs]
    r_prefs = [_normalize_pref_row(row, proposers) for row in r_prefs]
    st.session_state["gs_data"] = {
        "proposers": proposers,
        "receivers": receivers,
        "proposer_prefs": p_prefs,
        "receiver_prefs": r_prefs,
    }
    st.session_state["gs_rank_version"] = int(st.session_state.get("gs_rank_version", 0)) + 1


def _remove_gs_pair() -> None:
    data = copy.deepcopy(st.session_state.get("gs_data") or {})
    proposers = [str(x) for x in (data.get("proposers") or [])]
    receivers = [str(x) for x in (data.get("receivers") or [])]
    p_prefs = [list(map(str, row)) for row in (data.get("proposer_prefs") or [])]
    r_prefs = [list(map(str, row)) for row in (data.get("receiver_prefs") or [])]

    if len(proposers) <= 1 or len(receivers) <= 1:
        return

    rem_p = proposers.pop()
    rem_r = receivers.pop()

    p_prefs = p_prefs[: len(proposers)]
    r_prefs = r_prefs[: len(receivers)]
    for row in p_prefs:
        row[:] = [x for x in row if x != rem_r]
    for row in r_prefs:
        row[:] = [x for x in row if x != rem_p]

    p_prefs = [_normalize_pref_row(row, receivers) for row in p_prefs]
    r_prefs = [_normalize_pref_row(row, proposers) for row in r_prefs]
    st.session_state["gs_data"] = {
        "proposers": proposers,
        "receivers": receivers,
        "proposer_prefs": p_prefs,
        "receiver_prefs": r_prefs,
    }
    st.session_state["gs_rank_version"] = int(st.session_state.get("gs_rank_version", 0)) + 1


def _editable_hungarian(preset_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not st.session_state["h_data"]:
        st.session_state["h_data"] = copy.deepcopy(preset_payload)

    data = st.session_state["h_data"]
    mode = st.selectbox("Mode", options=["min", "max"], index=0 if data.get("mode", "min") == "min" else 1, key="h_mode")
    matrix_vals = (data.get("matrix") or {}).get("values") or [[0, 0], [0, 0]]
    df = pd.DataFrame(matrix_vals)
    edited = st.data_editor(df, width="stretch", num_rows="dynamic", key="h_matrix")
    mat = edited.fillna(0).astype(float).values.tolist()
    return {"mode": mode, "matrix": {"values": mat}}, {}


def _editable_gale_shapley(preset_payload: dict[str, Any], preset_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not st.session_state["gs_data"]:
        st.session_state["gs_data"] = copy.deepcopy(preset_payload)
    data = copy.deepcopy(st.session_state["gs_data"])
    ctl_l, ctl_m, ctl_r = st.columns([1, 1, 4])
    with ctl_l:
        st.button("Add pair (+)", key="gs_add_pair", on_click=_add_gs_pair)
    with ctl_m:
        st.button("Remove pair (-)", key="gs_remove_pair", on_click=_remove_gs_pair)
    with ctl_r:
        st.caption("Add/remove one proposer+receiver pair (M#, W#). Rankings are auto-expanded/compacted.")

    data = copy.deepcopy(st.session_state["gs_data"])
    proposers = [str(x) for x in data.get("proposers", [])]
    receivers = [str(x) for x in data.get("receivers", [])]
    st.markdown(f"**Proposers:** {', '.join(proposers) if proposers else '(none)'}")
    st.markdown(f"**Receivers:** {', '.join(receivers) if receivers else '(none)'}")

    side = st.selectbox(
        "Who proposes?",
        options=["proposers", "receivers"],
        index=0 if preset_cfg.get("proposer_side", "proposers") == "proposers" else 1,
        key="gs_side",
    )
    p_default = data.get("proposer_prefs", [])
    r_default = data.get("receiver_prefs", [])

    if not HAS_SORTABLES:
        st.info("Install `streamlit-sortables` for drag-and-drop ranking. Using fallback selector for now.")

    left, right = st.columns(2)
    rank_v = int(st.session_state.get("gs_rank_version", 0))
    with left:
        p_prefs = _rank_editor_block(
            header="Proposer rankings",
            agents=proposers,
            counterpart_ids=receivers,
            prefs=p_default,
            key_prefix=f"gs_p_rank_v{rank_v}",
        )
    with right:
        r_prefs = _rank_editor_block(
            header="Receiver rankings",
            agents=receivers,
            counterpart_ids=proposers,
            prefs=r_default,
            key_prefix=f"gs_r_rank_v{rank_v}",
        )

    if len(p_prefs) != len(proposers) or len(r_prefs) != len(receivers):
        raise ValueError("Each agent must have exactly one ranking row.")

    return {
        "proposers": proposers,
        "receivers": receivers,
        "proposer_prefs": p_prefs,
        "receiver_prefs": r_prefs,
    }, {"proposer_side": side}


def _editable_dinic(preset_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not st.session_state["d_data"]:
        st.session_state["d_data"] = copy.deepcopy(preset_payload)
    data = st.session_state["d_data"]
    nodes = (data.get("graph") or {}).get("nodes") or []
    edges = (data.get("graph") or {}).get("edges") or []

    st.markdown("#### Nodes")
    ndf = pd.DataFrame(nodes if nodes else [{"id": "s", "x": 60, "y": 120, "label": "s"}])
    ndf = st.data_editor(ndf, width="stretch", num_rows="dynamic", key="d_nodes")
    node_ids = [str(x) for x in ndf.get("id", []).tolist() if str(x).strip()]
    if not node_ids:
        node_ids = ["s", "t"]

    c1, c2 = st.columns(2)
    with c1:
        source = st.selectbox("Source", node_ids, index=0, key="d_source")
    with c2:
        sink = st.selectbox("Sink", node_ids, index=min(1, len(node_ids) - 1), key="d_sink")

    st.markdown("#### Edges")
    edf = pd.DataFrame(edges if edges else [{"source": "s", "target": "t", "capacity": 1}])
    edf = st.data_editor(edf, width="stretch", num_rows="dynamic", key="d_edges")
    clean_nodes: list[dict[str, Any]] = []
    for _, row in ndf.fillna("").iterrows():
        nid = str(row.get("id", "")).strip()
        if not nid:
            continue
        clean_nodes.append(
            {
                "id": nid,
                "x": float(row.get("x", 0) or 0),
                "y": float(row.get("y", 0) or 0),
                "label": str(row.get("label", nid) or nid),
            }
        )
    clean_edges: list[dict[str, Any]] = []
    for _, row in edf.fillna("").iterrows():
        u = str(row.get("source", "")).strip()
        v = str(row.get("target", "")).strip()
        if not u or not v:
            continue
        clean_edges.append({"source": u, "target": v, "capacity": int(float(row.get("capacity", 0) or 0))})
    return {"source": source, "sink": sink, "graph": {"nodes": clean_nodes, "edges": clean_edges}}, {}


def _build_graph_figure(frame: Frame) -> go.Figure | None:
    g = frame.graph or {}
    nodes = g.get("nodes") or []
    if not nodes:
        return None
    pos = {str(n["id"]): (float(n.get("x", 0)), float(n.get("y", 0)), str(n.get("label", n["id"]))) for n in nodes}
    highlighted_nodes = set(g.get("highlighted_nodes") or [])
    highlighted_edges = {tuple(e) for e in (g.get("highlighted_edges") or [])}

    fig = go.Figure()
    for e in g.get("edges") or []:
        u, v = str(e.get("source")), str(e.get("target"))
        if u not in pos or v not in pos:
            continue
        x0, y0, _ = pos[u]
        x1, y1, _ = pos[v]
        kind = str(e.get("kind", "normal"))
        base_color = "#44d17a" if kind == "engaged" else ("#f6a623" if kind == "proposal" else "#7aa2f7")
        width = 4 if (u, v) in highlighted_edges else 2
        color = "#facc15" if (u, v) in highlighted_edges else base_color
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": color, "width": width},
                hoverinfo="text",
                text=[f"{u} -> {v}: {e.get('label', '')}"],
                showlegend=False,
            )
        )
        if e.get("label"):
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=str(e["label"]), showarrow=False, font={"size": 11, "color": "#d1d5db"})

    xs, ys, labels, colors, sizes = [], [], [], [], []
    for nid, (x, y, label) in pos.items():
        xs.append(x)
        ys.append(y)
        labels.append(label)
        hl = nid in highlighted_nodes
        colors.append("#facc15" if hl else "#1f2937")
        sizes.append(24 if hl else 18)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            marker={"size": sizes, "color": colors, "line": {"color": "#9ca3af", "width": 1}},
            textfont={"color": "white", "size": 11},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        height=420,
        xaxis={"visible": False},
        yaxis={"visible": False, "autorange": "reversed"},
    )
    return fig


def _build_matrix_figure(frame: Frame) -> go.Figure | None:
    m = frame.matrix or {}
    vals = m.get("values")
    if not vals:
        return None
    z = [[float(x) if str(x).replace(".", "", 1).lstrip("-").isdigit() else 0 for x in row] for row in vals]
    text = [[str(x) for x in row] for row in vals]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale="Viridis",
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        yaxis={"autorange": "reversed"},
    )
    for rc in m.get("highlighted_cells") or []:
        if len(rc) != 2:
            continue
        r, c = int(rc[0]), int(rc[1])
        fig.add_shape(
            type="rect",
            x0=c - 0.5,
            x1=c + 0.5,
            y0=r - 0.5,
            y1=r + 0.5,
            line={"color": "#facc15", "width": 3},
        )

    # Hungarian teaching overlays from frame metadata.
    meta = frame.meta or {}
    h = meta.get("hungarian") if isinstance(meta.get("hungarian"), dict) else {}
    if h:
        n_rows = len(z)
        n_cols = len(z[0]) if z else 0
        row = h.get("row")
        col = h.get("col")
        sub = h.get("subtracted")
        if isinstance(row, int) and 0 <= row < n_rows:
            fig.add_shape(type="rect", x0=-0.5, x1=n_cols - 0.5, y0=row - 0.5, y1=row + 0.5, line={"color": "#22d3ee", "width": 3})
            if sub is not None:
                fig.add_annotation(x=n_cols - 0.5, y=row, text=f"row min={sub}", showarrow=False, xanchor="left", font={"size": 11, "color": "#22d3ee"})
        if isinstance(col, int) and 0 <= col < n_cols:
            fig.add_shape(type="rect", x0=col - 0.5, x1=col + 0.5, y0=-0.5, y1=n_rows - 0.5, line={"color": "#fb7185", "width": 3})
            if sub is not None:
                fig.add_annotation(x=col, y=-0.65, text=f"col min={sub}", showarrow=False, yanchor="bottom", font={"size": 11, "color": "#fb7185"})
        for cell in h.get("zero_cells") or []:
            if len(cell) != 2:
                continue
            zr, zc = int(cell[0]), int(cell[1])
            fig.add_shape(type="circle", x0=zc - 0.22, x1=zc + 0.22, y0=zr - 0.22, y1=zr + 0.22, line={"color": "#fde047", "width": 2})
        row_cov = h.get("row_covered") or []
        col_cov = h.get("col_covered") or []
        for r_idx, covered in enumerate(row_cov):
            if covered:
                fig.add_shape(type="line", x0=-0.5, x1=n_cols - 0.5, y0=r_idx, y1=r_idx, line={"color": "#f97316", "width": 3})
        for c_idx, covered in enumerate(col_cov):
            if covered:
                fig.add_shape(type="line", x0=c_idx, x1=c_idx, y0=-0.5, y1=n_rows - 0.5, line={"color": "#f97316", "width": 3})

    return fig


def _run_build(algo_label: str, data: dict[str, Any], cfg: dict[str, Any]) -> None:
    algo_key, _, build, validate = ALGORITHMS[algo_label]
    frames = build(copy.deepcopy(data), copy.deepcopy(cfg))
    st.session_state["frames_by_algo"][algo_key] = frames
    st.session_state["frame_idx_by_algo"][algo_key] = 0
    st.session_state["is_playing_by_algo"][algo_key] = False
    st.session_state["loaded_input_by_algo"][algo_key] = copy.deepcopy(data)
    st.session_state["loaded_config_by_algo"][algo_key] = copy.deepcopy(cfg)
    if frames:
        final_state = (frames[-1].meta or {}).get("final_state", {})
        rep = validate(data, final_state)
        st.session_state["validation_by_algo"][algo_key] = f"{'PASS' if rep.ok else 'FAIL'} - {rep.message}"
    else:
        st.session_state["validation_by_algo"][algo_key] = "No frames generated."


def _algo_state(algo_key: str) -> tuple[list[Frame], int, bool, str]:
    frames = st.session_state["frames_by_algo"].get(algo_key, [])
    idx = int(st.session_state["frame_idx_by_algo"].get(algo_key, 0))
    playing = bool(st.session_state["is_playing_by_algo"].get(algo_key, False))
    validation = st.session_state["validation_by_algo"].get(algo_key, "Run an animation to validate results.")
    if frames:
        idx = min(max(0, idx), len(frames) - 1)
    else:
        idx = 0
    return frames, idx, playing, validation


def _frame_phase_label(frame: Frame) -> str:
    meta = frame.meta or {}
    if meta.get("algorithm") == "hungarian":
        h = meta.get("hungarian") if isinstance(meta.get("hungarian"), dict) else {}
        return str(h.get("phase", ""))
    return str(meta.get("phase") or meta.get("event") or "")


def main() -> None:
    st.set_page_config(page_title="CS470 Algorithm Animator", page_icon="✨", layout="wide")
    _inject_modern_css()
    _init_state()

    st.title("CS470 Algorithm Animator")
    st.caption("Dynamic, editable algorithm visualizer with dedicated pages per algorithm.")

    tabs = st.tabs(list(ALGORITHMS.keys()))
    for tab, algo_label in zip(tabs, ALGORITHMS.keys()):
        algo_key, presets, _, _ = ALGORITHMS[algo_label]
        with tab:
            st.subheader(algo_label)
            top_l, top_r = st.columns([2, 1])
            with top_l:
                preset_name = st.selectbox("Preset", list(presets.keys()), key=f"{algo_key}_preset")
            with top_r:
                speed = st.slider("Animation speed", min_value=0.25, max_value=4.0, value=1.0, step=0.25, key=f"{algo_key}_speed")

            preset_payload = copy.deepcopy(presets[preset_name])
            preset_cfg = dict(preset_payload.pop("__config__", {}) or {})

            st.markdown("#### Input Editor")
            if algo_key == "hungarian":
                edited_data, cfg = _editable_hungarian(preset_payload)
            elif algo_key == "gale_shapley":
                edited_data, cfg = _editable_gale_shapley(preset_payload, preset_cfg)
            else:
                edited_data, cfg = _editable_dinic(preset_payload)

            c_run, c_reset = st.columns([1, 1])
            with c_run:
                if st.button("Run animation", type="primary", width="stretch", key=f"{algo_key}_run"):
                    try:
                        _run_build(algo_label, edited_data, cfg)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to build frames: {exc}")
            with c_reset:
                if st.button("Load preset defaults", width="stretch", key=f"{algo_key}_defaults"):
                    if algo_key == "hungarian":
                        st.session_state["h_data"] = {}
                    elif algo_key == "gale_shapley":
                        st.session_state["gs_data"] = {}
                    else:
                        st.session_state["d_data"] = {}
                    st.session_state["frames_by_algo"][algo_key] = []
                    st.session_state["frame_idx_by_algo"][algo_key] = 0
                    st.session_state["is_playing_by_algo"][algo_key] = False
                    st.session_state["validation_by_algo"][algo_key] = "Run an animation to validate results."
                    st.rerun()

            frames, frame_idx, is_playing, validation_msg = _algo_state(algo_key)
            if not frames:
                st.info("No animation loaded yet. Edit inputs and click **Run animation**.")
                continue

            p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 2])
            with p1:
                if st.button("Reset", width="stretch", key=f"{algo_key}_reset"):
                    st.session_state["frame_idx_by_algo"][algo_key] = 0
                    st.session_state["is_playing_by_algo"][algo_key] = False
                    st.rerun()
            with p2:
                if st.button("Prev", width="stretch", key=f"{algo_key}_prev"):
                    st.session_state["frame_idx_by_algo"][algo_key] = max(0, frame_idx - 1)
                    st.session_state["is_playing_by_algo"][algo_key] = False
                    st.rerun()
            with p3:
                if st.button("Next", width="stretch", key=f"{algo_key}_next"):
                    st.session_state["frame_idx_by_algo"][algo_key] = min(len(frames) - 1, frame_idx + 1)
                    st.session_state["is_playing_by_algo"][algo_key] = False
                    st.rerun()
            with p4:
                if st.button("Play/Pause", width="stretch", key=f"{algo_key}_play"):
                    st.session_state["is_playing_by_algo"][algo_key] = not is_playing
                    st.rerun()
            with p5:
                slider_key = f"{algo_key}_frame"
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = frame_idx
                elif st.session_state["is_playing_by_algo"].get(algo_key, False):
                    # During autoplay, force slider to follow frame index so it cannot snap playback back.
                    st.session_state[slider_key] = frame_idx
                elif st.session_state[slider_key] != frame_idx:
                    # Keep slider synced when frame changes via buttons/autoplay.
                    st.session_state[slider_key] = frame_idx
                idx = st.slider("Frame", min_value=0, max_value=len(frames) - 1, key=slider_key)
                if idx != frame_idx and not st.session_state["is_playing_by_algo"].get(algo_key, False):
                    st.session_state["frame_idx_by_algo"][algo_key] = idx
                    st.session_state["is_playing_by_algo"][algo_key] = False
                    frame_idx = idx

            frame = frames[frame_idx]

            if algo_key == "hungarian":
                left, right = st.columns([1.1, 2.2])
                with left:
                    st.markdown("### Step Explanation")
                    phase = _frame_phase_label(frame)
                    if phase:
                        st.markdown(f"**Phase:** `{phase}`")
                    st.write(frame.explanation)
                    st.markdown("### Legend / Status")
                    for item in (frame.legend or {}).get("items") or []:
                        st.markdown(f"- {item}")
                with right:
                    st.markdown("### Matrix Animation")
                    mfig = _build_matrix_figure(frame)
                    if mfig is not None:
                        st.plotly_chart(mfig, width="stretch")
                    else:
                        st.info("No matrix for this frame.")
            elif algo_key == "gale_shapley":
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("### Matching Graph")
                    gfig = _build_graph_figure(frame)
                    if gfig is not None:
                        st.plotly_chart(gfig, width="stretch")
                    else:
                        st.info("No graph for this frame.")
                with right:
                    st.markdown("### Preference Matrix")
                    mfig = _build_matrix_figure(frame)
                    if mfig is not None:
                        st.plotly_chart(mfig, width="stretch")
                    else:
                        st.info("No matrix for this frame.")
            else:
                st.markdown("### Flow Graph")
                gfig = _build_graph_figure(frame)
                if gfig is not None:
                    st.plotly_chart(gfig, width="stretch")
                else:
                    st.info("No graph for this frame.")
            if algo_key != "hungarian":
                st.markdown("### Explanation")
                phase = _frame_phase_label(frame)
                if phase:
                    st.markdown(f"**Phase/Event:** `{phase}`")
                st.write(frame.explanation)
                st.markdown("### Legend / Status")
                for item in (frame.legend or {}).get("items") or []:
                    st.markdown(f"- {item}")
            st.success(f"Validation: {validation_msg}")

            if st.session_state["is_playing_by_algo"].get(algo_key, False):
                cur = int(st.session_state["frame_idx_by_algo"].get(algo_key, 0))
                if cur < len(frames) - 1:
                    time.sleep(max(0.05, 0.4 / float(speed)))
                    st.session_state["frame_idx_by_algo"][algo_key] = cur + 1
                    st.rerun()
                else:
                    st.session_state["is_playing_by_algo"][algo_key] = False


if __name__ == "__main__":
    main()
