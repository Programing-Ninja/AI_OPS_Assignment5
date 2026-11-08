#!/usr/bin/env bash
# Runs every A3 load test. Usage (from the submission root): bash partA/run_loadtest.sh
set -euo pipefail
PY=${PY:-python}
PORT=${PORT:-8000}
URL="http://127.0.0.1:${PORT}/v1/models/eta:predict"

start() {
  ETA_MODEL=data/eta_v1.joblib setsid $PY -m uvicorn --app-dir partA eta_app:app \
      --port "$PORT" --workers "$1" --log-level warning > "partA/uvicorn_w$1.log" 2>&1 &
  SRV=$!
  for _ in $(seq 60); do curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null && return; sleep 0.5; done
  echo "server did not start"; exit 1
}
stop() { kill -TERM -- -"$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; sleep 1; }
warm() { $PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 4 --duration 2 >/dev/null; }

echo "== A3.1  1 worker, batch 1, concurrency sweep"
start 1; warm
/usr/bin/time -f "load generator: %e s wall, %U s user, %S s sys, %P CPU" \
  $PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 \
    --out partA/loadtest_results.json

echo "== A3.2  extended sweep past the knee: where does p95 cross the 80 ms SLO?"
$PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 24 32 48 64 --duration 4 \
    --out partA/loadtest_extended.json

echo "== A3.3  batch 1 vs batch 100 at concurrency 1"
$PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1 \
    --out partA/loadtest_batch1.json
$PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100 \
    --out partA/loadtest_batch100.json
stop

echo "== A3.4a  change: 2 worker processes (uvicorn --workers 2)"
start 2; warm
/usr/bin/time -f "load generator: %e s wall, %U s user, %S s sys, %P CPU" \
  $PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 \
    --out partA/loadtest_2workers_uvicorn.json
stop

echo "== A3.4b  same change via gunicorn -w 2 (sets TCP_NODELAY itself; immune to the Nagle stall, see partA/nagle_note.txt)"
ETA_MODEL=data/eta_v1.joblib setsid gunicorn --chdir partA -w 2 -k uvicorn.workers.UvicornWorker \
    -b "127.0.0.1:${PORT}" --log-level warning eta_app:app > partA/gunicorn_w2.log 2>&1 &
SRV=$!
for _ in $(seq 60); do curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null && break; sleep 0.5; done
warm
/usr/bin/time -f "load generator: %e s wall, %U s user, %S s sys, %P CPU" \
  $PY load_test.py --url "$URL" --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 \
    --out partA/loadtest_2workers_gunicorn.json
stop
