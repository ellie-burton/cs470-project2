# CS470 Project 2 — Algorithm Animations

Ellie Burton & William Mulhern

Streamlit web interface with interactive animations of 3 algorithms covered in CS 470.

## Algorithms

- **Hungarian** (min/max assignment) — editable cost matrix, step-by-step row/column reduction, zero identification, line covering, and optimal assignment.
- **Gale-Shapley** (stable matching) — drag-and-drop preference rankings, dynamic add/remove participant pairs, animated proposal rounds with engagements and swaps.
- **Dinic's** (max flow) — editable directed graph with nodes/edges/capacities, animated BFS level construction, admissible-edge highlighting, DFS blocking-flow augmentation with bottleneck identification, and running flow metric.

## Quick Start

Requires **Python 3.10+** and pip.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens in your browser with three tabs, one per algorithm. Each tab has preset inputs you can select from, edit, then click **Run animation** to step through the visualization.

## Project Layout

| Path | Description |
|---|---|
| `streamlit_app.py` | Main Streamlit UI — tabbed interface, input editors, graph/matrix rendering, playback controls |
| `algorithms/hungarian.py` | Hungarian algorithm logic, frame generation, validation, presets |
| `algorithms/gale_shapley.py` | Gale-Shapley algorithm logic, frame generation, validation, presets |
| `algorithms/dinic.py` | Dinic's algorithm logic, frame generation, validation, presets |
| `src/models.py` | Shared `Frame` and `ValidationReport` data classes |
| `requirements.txt` | Python dependencies |
| `Assignment.md` | Original assignment specification |
