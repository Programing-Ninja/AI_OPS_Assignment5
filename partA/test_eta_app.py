"""Tests for eta_app (Part A2). Run: pytest partA -q"""
import re, sys, uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"
GOOD = dict(distance_km=12.5, n_stops=4, hour=10, weekday=2, traffic_index=0.3, rain=0, weight_kg=5.0)

@pytest.fixture
def client():
    return TestClient(eta_app.app, raise_server_exceptions=False)

def row(**over):
    return {**GOOD, **over}

def post(client, instances, **kw):
    return client.post(URL, json={"instances": instances}, **kw)

def test_valid_request_returns_prediction(client):
    r = post(client, [GOOD])
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"predictions", "model_version", "request_id"}
    assert j["model_version"] == eta_app.MODEL_VERSION
    assert len(j["predictions"]) == 1 and isinstance(j["predictions"][0], float)
    assert j["predictions"][0] == round(j["predictions"][0], 1)

def test_batch_order_preserved(client):
    rows = [row(distance_km=d) for d in (2, 150, 40, 5, 90)]
    r = post(client, rows)
    assert r.status_code == 200
    batch = r.json()["predictions"]
    singles = [post(client, [x]).json()["predictions"][0] for x in rows]
    assert batch == singles
    assert batch != sorted(batch)

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}

@pytest.mark.parametrize("bad,field", [
    (row(distance_km=0.05), "distance_km"),
    (row(hour=24), "hour"),
    (row(rain=2), "rain"),
    (row(weight_kg=0), "weight_kg"),
    (row(traffic_index=1.01), "traffic_index"),
])
def test_out_of_range_rejected(client, bad, field):
    r = post(client, [bad])
    assert r.status_code == 422
    j = r.json()
    assert j["error"] == "validation_failed"
    assert any(field in d["field"] for d in j["details"])

def test_missing_field_rejected(client):
    bad = {k: v for k, v in GOOD.items() if k != "n_stops"}
    r = post(client, [bad])
    assert r.status_code == 422 and any("n_stops" in d["field"] for d in r.json()["details"])

def test_wrong_type_rejected(client):
    r = post(client, [row(distance_km="far")])
    assert r.status_code == 422 and r.json()["error"] == "validation_failed"

def test_malformed_body_rejected(client):
    r = client.post(URL, json={"rows": [GOOD]})
    assert r.status_code == 422 and r.json()["error"] == "validation_failed"

def test_boundary_values_accepted(client):
    edge = [row(distance_km=0.1, n_stops=1, hour=0, weekday=0, traffic_index=0, rain=0, weight_kg=0.01),
            row(distance_km=500, n_stops=20, hour=23, weekday=6, traffic_index=1, rain=1, weight_kg=1000)]
    assert post(client, edge).status_code == 200

def test_infeasible_delivery_rejected(client):
    r = post(client, [row(distance_km=151, n_stops=21)])
    assert r.status_code == 422
    assert "infeasible" in r.text

@pytest.mark.parametrize("d,s", [(150, 21), (151, 20)])
def test_feasibility_rule_needs_both_conditions(client, d, s):
    assert post(client, [row(distance_km=d, n_stops=s)]).status_code == 200

def test_empty_batch_rejected(client):
    assert post(client, []).status_code == 422

def test_batch_of_500_accepted_501_rejected(client):
    assert post(client, [GOOD] * 500).status_code == 200
    assert post(client, [GOOD] * 501).status_code == 422

def test_request_id_echoed(client):
    rid = "test-" + uuid.uuid4().hex
    r = post(client, [GOOD], headers={"X-Request-ID": rid})
    assert r.headers["X-Request-ID"] == rid and r.json()["request_id"] == rid

def test_request_id_generated_when_absent(client):
    r = post(client, [GOOD])
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)
    assert r.json()["request_id"] == rid

def test_request_id_on_validation_error(client):
    r = post(client, [row(hour=99)], headers={"X-Request-ID": "err-123"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-123" and r.json()["request_id"] == "err-123"
    r2 = post(client, [row(hour=99)])
    uuid.UUID(r2.headers["X-Request-ID"])
    assert r2.json()["request_id"] == r2.headers["X-Request-ID"]

def _counter(text, path, status):
    m = re.search(rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\d+)', text)
    return int(m.group(1)) if m else 0

def test_metrics_exposition(client):
    before = client.get("/metrics").text
    post(client, [GOOD]); post(client, [row(hour=-1)])
    text = client.get("/metrics").text
    again = client.get("/metrics").text
    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    total = lambda t: sum(int(n) for n in re.findall(r"^eta_requests_total\{.*\} (\d+)$", t, re.M))
    assert total(again) == total(text)
    assert re.search(r"_count (\d+)", again).group(1) == re.search(r"_count (\d+)", text).group(1)
    assert "# TYPE eta_request_latency_seconds histogram" in text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert len(re.findall(r"eta_request_latency_seconds_bucket\{le=", text)) >= 5
    count = int(re.search(r"eta_request_latency_seconds_count (\d+)", text).group(1))
    inf = int(re.search(r'_bucket\{le="\+Inf"\} (\d+)', text).group(1))
    assert count == inf and re.search(r"eta_request_latency_seconds_sum [0-9.]+", text)

class Boom:
    def predict(self, X):
        raise RuntimeError("secret internal detail: /opt/model/weights.bin corrupted")

def test_inference_error_returns_generic_500(client, monkeypatch):
    monkeypatch.setattr(eta_app, "model", Boom())
    r = post(client, [GOOD], headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "weights.bin" not in r.text and "RuntimeError" not in r.text
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert r.headers["X-Request-ID"] == "boom-1"
