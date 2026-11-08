# AIOps Module 4 assignment: SwiftParcel ETA service (DA24B039)

The written answers, tables and machine description are in `REPORT.pdf`. Its LaTeX source is `report/REPORT.tex`;
the figure is drawn from `partA/*.json` by `python report/make_figures.py`, and
`cd report && pdflatex REPORT.tex && pdflatex REPORT.tex` rebuilds the PDF. AI use is declared in `AI_DISCLOSURE.md`.

## Setup (run once, from this folder)

Every command below runs from **this folder**, the submission root, where `data/` gets created.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eta_data.py                                   # data/*.csv + data/eta_v1.joblib, data/eta_v2.joblib
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
```

Tested with Python 3.11. If your shell exports `PYTHONPATH` (a ROS install does, for example), run
`unset PYTHONPATH` first, because a foreign pytest plugin can break test collection.

## Running each part

| Part | Command | Output |
|---|---|---|
| A1 | `ETA_MODEL=data/eta_v1.joblib uvicorn --app-dir partA eta_app:app --port 8000` | service on :8000 |
| A2 | `pytest partA partB -q` (22 + 17 tests) · `python partA/mutation_check.py` | `partA/pytest_output.txt`, `partA/mutation_output.txt` |
| A3 | `bash partA/run_loadtest.sh` (starts and stops the server itself, about 2 min) · `bash partA/server_cpu.sh` | `partA/loadtest_*.json`, `partA/loadtest_console.txt`, `partA/server_cpu.txt` |
| B1 | `python partB/eta_mlflow.py register` | `mlflow.db`, `mlruns/`, registry table |
| B2 | `python partB/eta_mlflow.py gate` · `python partB/eta_mlflow.py gate --min-rel-gain 0.60` | `partB/gate_output.txt` |
| B3 | `python partB/shadow_eta.py --alias-demo` (starts `mlflow models serve` on 5101 and 5102, then stops both) | `partB/shadow_output.txt` |
| C  | `python partC/verify_ray.py --autoscale-demo` (starts its own 2-CPU Ray on :8100) | `partC/verify_output.txt` |
| C (CLI) | `serve run --app-dir partC --working-dir partC eta_ray:app_builder weight=0.1` | router on :8000 |
| D  | `python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1` (for the Ray router, add `--customer-id x --beta`) | exit 0 or 1, `partD/smoke_output.txt` |

`partA/loadtest_results.json` is the required A3.1 sweep (concurrency 1, 2, 4, 8, 16). The other
`partA/loadtest_*.json` files hold the extended sweep, the batch comparison and the 2-worker runs.

## Notes

- **Tracking store.** `MLFLOW_TRACKING_URI` defaults to `sqlite:///mlflow.db`. To get a clean registry, delete
  `mlflow.db` and `mlruns/` before `register`. Otherwise versions 3, 4, … are added and the aliases move to them.
- **Quieter MLflow output.** Set `MLFLOW_DISABLE_AGENT_HINT=1` to hide an informational MLflow hint.
- **Excluded from the zip.** Model files, databases, `mlruns/` and logs are left out, as the assignment requires.
  `python eta_data.py` regenerates the models deterministically.
