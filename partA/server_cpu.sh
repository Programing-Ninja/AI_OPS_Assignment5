#!/usr/bin/env bash
# Measures server CPU at concurrency 16. Usage: bash partA/server_cpu.sh | tee partA/server_cpu.txt
set -euo pipefail
ETA_MODEL=data/eta_v1.joblib python -m uvicorn --app-dir partA eta_app:app --port 8000 --log-level warning >/dev/null 2>&1 &
P=$!; trap 'kill $P 2>/dev/null' EXIT
until curl -sf localhost:8000/healthz >/dev/null; do sleep 0.5; done
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 4 --duration 2 >/dev/null
t0=$(awk '{print $14+$15}' /proc/$P/stat); s0=$(date +%s.%N)
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 16 --duration 6
t1=$(awk '{print $14+$15}' /proc/$P/stat); s1=$(date +%s.%N)
python -c "print(f'server process CPU during the c=16 run: {($t1-$t0)/$(getconf CLK_TCK)/($s1-$s0)*100:.0f}% of one core')"
