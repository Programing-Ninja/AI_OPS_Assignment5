"""MLflow pyfunc, registry, promotion gate (Part B). Run: python partB/eta_mlflow.py register|table|gate"""
import os, sys, json, time, argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_NAME = "eta-model"
EXPERIMENT = "eta-serving"
ALIASES = {"v1": "prod", "v2": "candidate"}
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)

def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

def _eta_data():
    """Imports eta_data.py from the submission root."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import eta_data
    return eta_data

def _bundle_path(version: str) -> Path:
    for cand in (Path("data") / f"eta_{version}.joblib", ROOT / "data" / f"eta_{version}.joblib"):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"eta_{version}.joblib not found -- run `python eta_data.py` first")

class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib bundle; returns eta_minutes and model_version."""

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), str(b["version"])

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        y = np.round(self.model.predict(X).astype(float), 1)
        return pd.DataFrame({"eta_minutes": y, "model_version": [self.version] * len(y)})

def _pinned_requirements():
    import sklearn
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}"]

def slice_metrics(y_true, y_pred, frame) -> dict:
    """Overall MAE plus the two slices the gate cares about."""
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    rush = frame["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (frame["rain"] == 1).to_numpy()
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()), "mae_rain": float(err[rain].mean())}

def register() -> pd.DataFrame:
    """Logs v1 and v2 as pyfunc models, registers them and sets the aliases."""
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    ed = _eta_data()
    fresh = ed.make_data(2000, seed=99)
    X_example = fresh[ed.FEATURES].head(3).astype("float64")
    client = MlflowClient()
    registered = {}
    for ver in ("v1", "v2"):
        path = _bundle_path(ver)
        b = joblib.load(path)
        out_example = pd.DataFrame({"eta_minutes": [30.5], "model_version": [ver]})
        sig = infer_signature(X_example, out_example)
        m = slice_metrics(fresh["eta_minutes"], b["model"].predict(fresh[ed.FEATURES]), fresh)
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_params({"bundle_version": ver, "algorithm": type(b["model"]).__name__,
                               "features": ",".join(b["feature_columns"])})
            mlflow.log_metrics({f"fresh_{k}": v for k, v in m.items()})
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": str(path)},
                signature=sig, input_example=X_example, pip_requirements=_pinned_requirements(),
                code_paths=[__file__], registered_model_name=MODEL_NAME)
        registered[ver] = str(info.registered_model_version)
    for ver, alias in ALIASES.items():
        client.set_registered_model_alias(MODEL_NAME, alias, registered[ver])
    return registry_table()

def registry_table() -> pd.DataFrame:
    client = MlflowClient(tracking_uri=tracking_uri())
    aliases = client.get_registered_model(MODEL_NAME).aliases
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
        tag = mlflow.get_run(mv.run_id).data.params.get("bundle_version", "?") if mv.run_id else "?"
        rows.append({"version": int(mv.version), "bundle": tag,
                     "aliases": ",".join(a for a, v in aliases.items() if str(v) == str(mv.version)) or "-",
                     "run_id": mv.run_id[:8]})
    return pd.DataFrame(rows)

_EPS = 1e-9

def decide(prod: dict, cand: dict, min_rel_gain: float = 0.03, max_p95_ms: float = 10.0,
           max_slice_regression: float = 0.05) -> dict:
    """Pure promotion decision."""
    for name, d in (("prod", prod), ("cand", cand)):
        missing = {"mae", "mae_rush", "mae_rain", "p95_ms"} - set(d)
        if missing:
            raise KeyError(f"{name} metrics missing {sorted(missing)}")
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"] if prod["mae"] > 0 else float("-inf")
    criteria = {
        "mae_gain": {"value": rel_gain, "threshold": min_rel_gain,
                     "pass": bool(rel_gain >= min_rel_gain - _EPS)},
        "latency_p95": {"value": cand["p95_ms"], "threshold": max_p95_ms,
                        "pass": bool(cand["p95_ms"] <= max_p95_ms + _EPS)},
    }
    for sl in ("mae_rush", "mae_rain"):
        limit = prod[sl] * (1 + max_slice_regression)
        rel = (cand[sl] - prod[sl]) / prod[sl] if prod[sl] > 0 else float("inf")
        criteria[f"slice_{sl[4:]}"] = {"value": rel, "threshold": max_slice_regression,
                                       "pass": bool(cand[sl] <= limit + _EPS)}
    return {"criteria": criteria, "promote": all(c["pass"] for c in criteria.values())}

def _single_row_p95_ms(model, X: pd.DataFrame, n: int = 200, warmup: int = 20) -> float:
    for i in range(warmup):
        model.predict(X.iloc[[i]])
    ts = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        t0 = time.perf_counter()
        model.predict(row)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 95))

def evaluate_alias(alias: str, fresh: pd.DataFrame, features) -> dict:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    X = fresh[features].astype("float64")
    out = model.predict(X)
    m = slice_metrics(fresh["eta_minutes"], out["eta_minutes"], fresh)
    m["p95_ms"] = _single_row_p95_ms(model, X)
    m["model_version"] = str(out["model_version"].iloc[0])
    return m

def promotion_gate(**thresholds) -> dict:
    """Evaluates both aliases on fresh data and calls decide()."""
    mlflow.set_tracking_uri(tracking_uri())
    ed = _eta_data()
    fresh = ed.make_data(2000, seed=99)
    metrics = {alias: evaluate_alias(alias, fresh, ed.FEATURES) for alias in ("prod", "candidate")}
    result = decide(metrics["prod"], metrics["candidate"], **thresholds)
    return {**result, "thresholds": thresholds or "defaults", "metrics": metrics}

def _round(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    return o

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "table", "gate"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        print(register().to_markdown(index=False))
    elif a.cmd == "table":
        print(registry_table().to_markdown(index=False))
    else:
        th = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms,
                                 "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        print(json.dumps(_round(promotion_gate(**th)), indent=2))
