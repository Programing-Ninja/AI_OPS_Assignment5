"""SwiftParcel ETA model as a FastAPI service (Part A)."""
import os, time, uuid, logging, threading
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger("eta_app")
HERE = Path(__file__).resolve().parent

def _resolve(path: str) -> str:
    """Finds the model file relative to the cwd or the submission root."""
    for cand in (Path(path), HERE / path, HERE.parent / path):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"model bundle not found: {path} (run `python eta_data.py` first)")

MODEL_PATH = _resolve(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
_bundle = joblib.load(MODEL_PATH)
model = _bundle["model"]
FEATURES: List[str] = list(_bundle["feature_columns"])
MODEL_VERSION: str = str(_bundle["version"])

class Delivery(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self

class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str
    request_id: str

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5)
KNOWN_PATHS = {"/v1/models/eta:predict", "/healthz"}
_lock = threading.Lock()
_requests = {}
_bucket_counts = [0] * (len(BUCKETS) + 1)
_lat_sum, _lat_count = 0.0, 0

def _observe(path: str, status: int, seconds: float) -> None:
    global _lat_sum, _lat_count
    label = path if path in KNOWN_PATHS else "other"
    with _lock:
        _requests[(label, str(status))] = _requests.get((label, str(status)), 0) + 1
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                _bucket_counts[i] += 1
                break
        else:
            _bucket_counts[-1] += 1
        _lat_sum += seconds
        _lat_count += 1

def render_metrics() -> str:
    with _lock:
        lines = ["# HELP eta_requests_total Requests handled, by path and HTTP status.",
                 "# TYPE eta_requests_total counter"]
        for (p, s), n in sorted(_requests.items()):
            lines.append(f'eta_requests_total{{path="{p}",status="{s}"}} {n}')
        lines += ["# HELP eta_request_latency_seconds Server-side request latency.",
                  "# TYPE eta_request_latency_seconds histogram"]
        cum = 0
        for b, n in zip(BUCKETS, _bucket_counts):
            cum += n
            lines.append(f'eta_request_latency_seconds_bucket{{le="{b}"}} {cum}')
        cum += _bucket_counts[-1]
        lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {cum}')
        lines.append(f"eta_request_latency_seconds_sum {_lat_sum:.6f}")
        lines.append(f"eta_request_latency_seconds_count {_lat_count}")
    return "\n".join(lines) + "\n"

app = FastAPI(title="SwiftParcel ETA service", version=MODEL_VERSION)

def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())

@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "").strip()
    rid = incoming if 0 < len(incoming) <= 128 else str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        _observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = [str(x) for x in e.get("loc", ()) if x != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": e.get("msg", "invalid")})
    rid = _rid(request)
    return JSONResponse(status_code=422, headers={"X-Request-ID": rid},
                        content={"error": "validation_failed", "details": details, "request_id": rid})

@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = _rid(request)
    log.exception("unhandled error request_id=%s", rid)
    return JSONResponse(status_code=500, headers={"X-Request-ID": rid},
                        content={"error": "internal_error", "request_id": rid})

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")

@app.post("/v1/models/eta:predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    rid = _rid(request)
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances], columns=FEATURES)
        y = model.predict(X)
        preds = [round(float(v), 1) for v in y]
    except Exception:
        log.exception("inference failed request_id=%s", rid)
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}
