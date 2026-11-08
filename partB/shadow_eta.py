"""Shadow comparison of prod and candidate MLflow servers (Part B3)."""
import os, sys, time, signal, argparse, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eta_data import FEATURES  # noqa: E402

PORTS = {"prod": 5101, "candidate": 5102}

def start(alias, port, log):
    cmd = ["mlflow", "models", "serve", "-m", f"models:/eta-model@{alias}", "-p", str(port),
           "--env-manager", "local"]
    env = {**os.environ, "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")}
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)

def wait_ready(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"http://127.0.0.1:{port}/ping", timeout=2).status_code == 200:
                return round(time.time() - t0, 1)
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"port {port} not ready -- see the serve log")

def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=15)
    except Exception:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception: pass

def call(sess, port, rec):
    t0 = time.perf_counter()
    r = sess.post(f"http://127.0.0.1:{port}/invocations", json={"dataframe_records": [rec]}, timeout=10)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    p = r.json()["predictions"][0]
    return p["eta_minutes"], p["model_version"], ms

def alias_demo(sess, rec):
    """Moves the prod alias and shows the running prod server does not change."""
    from mlflow import MlflowClient
    c = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    old = c.get_model_version_by_alias("eta-model", "prod").version
    new = c.get_model_version_by_alias("eta-model", "candidate").version
    before = call(sess, PORTS["prod"], rec)[1]
    c.set_registered_model_alias("eta-model", "prod", new)
    try:
        after = call(sess, PORTS["prod"], rec)[1]
        print(f"alias demo: prod alias moved {old} -> {new}; running :5101 answered {before!r} before "
              f"and {after!r} after the move (alias resolved once, at start-up)")
    finally:
        c.set_registered_model_alias("eta-model", "prod", old)
        print(f"alias demo: prod alias restored to version {old}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alias-demo", action="store_true",
                    help="after the comparison, point 'prod' at the candidate version and show :5101 is unchanged")
    a = ap.parse_args()
    fresh = pd.read_csv(ROOT / "data" / "eta_fresh.csv").head(a.n)
    recs = fresh[FEATURES].to_dict(orient="records")

    procs = []
    if not a.no_start:
        for alias, port in PORTS.items():
            procs.append(start(alias, port, ROOT / "partB" / f"serve_{alias}.log"))
    try:
        for alias, port in PORTS.items():
            print(f"{alias:9s} ready on :{port} after {wait_ready(port)} s")
        sess = requests.Session()
        for alias, port in PORTS.items():
            for rec in recs[:10]:
                call(sess, port, rec)
        out = {alias: {"pred": [], "ver": set(), "ms": []} for alias in PORTS}
        for i, rec in enumerate(recs):
            order = list(PORTS.items()) if i % 2 == 0 else list(PORTS.items())[::-1]
            for alias, port in order:
                y, v, ms = call(sess, port, rec)
                out[alias]["pred"].append(y); out[alias]["ver"].add(v); out[alias]["ms"].append(ms)
        if a.alias_demo:
            alias_demo(sess, recs[0])
    finally:
        for p in procs:
            stop(p)

    p, c = np.array(out["prod"]["pred"]), np.array(out["candidate"]["pred"])
    diff = np.abs(p - c)
    y = fresh["eta_minutes"].to_numpy()
    lines = [
        f"requests per server        : {len(recs)} identical single-row requests (data/eta_fresh.csv rows 0..{len(recs)-1})",
        f"versions answering         : prod={sorted(out['prod']['ver'])}  candidate={sorted(out['candidate']['ver'])}",
        f"(i)   mean |prod - cand|   : {diff.mean():.2f} min   (median {np.median(diff):.2f}, max {diff.max():.1f})",
        f"(ii)  share |diff| > 10 min: {100 * (diff > 10).mean():.1f} %  ({int((diff > 10).sum())} of {len(diff)})",
        f"(iii) p95 latency          : prod {np.percentile(out['prod']['ms'], 95):.2f} ms   "
        f"candidate {np.percentile(out['candidate']['ms'], 95):.2f} ms",
        f"      p50 latency          : prod {np.percentile(out['prod']['ms'], 50):.2f} ms   "
        f"candidate {np.percentile(out['candidate']['ms'], 50):.2f} ms",
        f"extra (labels are known here, not in a real shadow): MAE prod {np.abs(p - y).mean():.2f} min, "
        f"candidate {np.abs(c - y).mean():.2f} min",
    ]
    print("\n".join(lines))

if __name__ == "__main__":
    main()
