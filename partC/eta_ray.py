"""Ray Serve app with a sticky per-customer canary (Part C)."""
import os, time, hashlib
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve
from ray.serve.handle import DeploymentHandle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ROUTER_CPUS = 0.25
MODEL_CPUS = 0.25
PROD_REPLICAS = 2
CANDIDATE_AUTOSCALING = dict(
    min_replicas=1,
    initial_replicas=1,
    max_replicas=3,
    target_ongoing_requests=2,
    upscale_delay_s=5,
    downscale_delay_s=60,
    look_back_period_s=10,
    metrics_interval_s=2,
)
MAX_ONGOING = 8

SALT = "swiftparcel-eta-canary-v2"

def bucket(customer_id: str, salt: str = SALT) -> float:
    """Stable, uniform value in [0, 1) for a customer."""
    h = hashlib.sha256(f"{salt}:{customer_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64

def choose_arm(customer_id: str, weight: float, beta: bool = False) -> str:
    """Returns 'v2' for about `weight` of customers, else 'v1'."""
    if beta:
        return "v2"
    return "v2" if bucket(customer_id) < weight else "v1"

def _resolve(path: str) -> str:
    for cand in (Path(path), ROOT / path, HERE / path):
        if cand.exists():
            return str(cand.resolve())
    raise FileNotFoundError(f"{path} not found -- run `python eta_data.py` in the submission root")

@serve.deployment(ray_actor_options={"num_cpus": MODEL_CPUS}, max_ongoing_requests=MAX_ONGOING,
                  health_check_period_s=5, logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), str(b["version"])
        self.extra = extra_work_ms / 1000.0
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[dict]) -> dict:
        X = pd.DataFrame(rows, columns=self.cols)
        y = self.model.predict(X)
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": [round(float(v), 1) for v in y], "model_version": self.version,
                "served_by": self.replica}

    def check_health(self):
        if self.model is None:
            raise RuntimeError("model not loaded")

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

class CanaryRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    instances: List[Delivery] = Field(min_length=1, max_length=500)

api = FastAPI(title="SwiftParcel ETA (Ray Serve, sticky canary)")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": ROUTER_CPUS},
                  logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod: DeploymentHandle, candidate: DeploymentHandle, weight: float):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.arms = {"v1": prod, "v2": candidate}
        self.weight = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "canary_weight": self.weight, "salt": SALT}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: CanaryRequest, x_beta: Optional[str] = Header(default=None)):
        arm = choose_arm(req.customer_id, self.weight, beta=(x_beta or "").strip() == "1")
        out = await self.arms[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm, "customer_id": req.customer_id, "canary_weight": self.weight}

def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Builds the app: router -> prod v1 (2 replicas) + candidate v2 (1-3 replicas)."""
    v1 = _resolve(os.getenv("ETA_V1", "data/eta_v1.joblib"))
    v2 = _resolve(os.getenv("ETA_V2", "data/eta_v2.joblib"))
    prod = EtaModel.options(name="EtaModelProd", num_replicas=PROD_REPLICAS)
    cand = EtaModel.options(name="EtaModelCandidate", num_replicas=None,
                            autoscaling_config=CANDIDATE_AUTOSCALING)
    return EtaRouter.bind(prod.bind(v1, extra_work_ms), cand.bind(v2, extra_work_ms), weight)

def app_builder(args: dict):
    """Entry point for `serve run`."""
    return build(float(args.get("weight", 0.1)), float(args.get("extra_work_ms", 0.0)))
