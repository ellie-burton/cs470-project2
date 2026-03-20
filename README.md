# CS470 Project 2 - Algorithm Animations

Interactive visualizations for:

- Hungarian algorithm (assignment, min/max)
- Gale-Shapley algorithm (stable matching)
- Dinic's algorithm (max flow)

The project uses shared algorithm modules (`build_frames`, `validate_result`) and a Streamlit UI.

## Requirements

- Python 3.10+ recommended
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

UI:

```bash
streamlit run streamlit_app.py
```

## Test

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/demo_all.py
```

## Project Layout


| Path                            | Role                                       |
| ------------------------------- | ------------------------------------------ |
| `streamlit_app.py`              | Main Streamlit UI (tabbed per algorithm)   |
| `algorithms/hungarian.py`       | Hungarian frames + validation + presets    |
| `algorithms/gale_shapley.py`    | Gale-Shapley frames + validation + presets |
| `algorithms/dinic.py`           | Dinic frames + validation + presets        |
| `algorithms/dummy_algorithm.py` | Small smoke/demo algorithm                 |
| `src/models.py`                 | Shared `Frame` and `ValidationReport`      |
| `tests/`                        | Unit tests                                 |
| `scripts/demo_all.py`           | CLI smoke runner across all presets        |


## Notes

- Gale-Shapley tab supports dynamic participant sizing with add/remove pair controls.
- Hungarian tab includes step-by-step matrix visuals (identify min, subtract, zero cover/lines, adjustments).
- Dinic tab supports editable nodes/edges/capacities and animated flow phases.

