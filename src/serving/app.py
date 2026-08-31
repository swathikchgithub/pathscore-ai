"""
FastAPI serving layer. Generic across every use case registered in
src/config/*.yaml -- loads the right model + config by name rather than
hardcoding per-use-case routes.

Run:
  uvicorn src.serving.app:app --reload

Env vars (all optional, all default to open/permissive for local dev):
  CORS_ORIGINS          comma-separated allowlist; unset -> "*"
  API_KEY               require X-API-Key on /score/* routes; unset -> no auth
  RATE_LIMIT_PER_MINUTE per-client cap on /score/* routes; default 60
"""

import glob
import os
import secrets
import time
from collections import defaultdict
from threading import Lock

import yaml
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

app = FastAPI(title="PathScore AI")

# Permissive by default since this demo API carries no auth/session state;
# restrict via CORS_ORIGINS (comma-separated) for a real deployment.
_cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else _cors_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
CONFIG_DIR = "src/config"

_model_cache = {}

# --- Auth + rate limiting -------------------------------------------------
# Same env-driven, off-by-default-in-local-dev pattern as CortexExtractor /
# LoRAExtractor's mock-vs-live clients: unset API_KEY -> demo mode, matching
# the README quickstart with no setup required; set it to require a
# X-API-Key header on every scoring route.

API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_rate_limit_lock = Lock()
_rate_limit_requests = defaultdict(list)  # client key -> request timestamps in the current window


def require_api_key(provided: str = Depends(_api_key_header)):
    if API_KEY is None:
        return
    if provided is None or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def enforce_rate_limit(request: Request, provided: str = Depends(_api_key_header)):
    """Fixed-window limiter keyed by API key (if presented) or client IP.
    In-process only -- fine for the single-worker demo this ships as; a
    multi-worker/production deployment needs a shared store (Redis) instead
    of this in-memory dict."""
    key = provided or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window_start = now - 60
    with _rate_limit_lock:
        timestamps = _rate_limit_requests[key]
        timestamps[:] = [t for t in timestamps if t > window_start]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")
        timestamps.append(now)


def load_use_case(use_case: str):
    if use_case in _model_cache:
        return _model_cache[use_case]

    config_path = f"{CONFIG_DIR}/{use_case}.yaml"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Unknown use case: {use_case}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = f"{MODELS_DIR}/{use_case}/model.joblib"
    calibrator_path = f"{MODELS_DIR}/{use_case}/calibrator.joblib"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not trained yet for: {use_case}")

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    explainer = shap.TreeExplainer(model)

    bundle = {"config": config, "model": model, "calibrator": calibrator, "explainer": explainer}
    _model_cache[use_case] = bundle
    return bundle


def _build_feature_frame(config: dict, records: list) -> pd.DataFrame:
    raw = pd.DataFrame(records)
    missing = [c for c in config["feature_columns"] if c not in raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required feature(s): {missing}")

    df = raw[config["feature_columns"]]
    for c in config.get("categorical_columns", []):
        df[c] = df[c].astype("category")
    return df


def _score_batch(bundle: dict, X: pd.DataFrame) -> list:
    """Scores every row of X in one vectorized pass -- a single batched
    predict_proba + a single batched SHAP call, not one call per row, so
    this stays O(n) with a small constant instead of O(n) separate model
    invocations. Used for both the single-row /score endpoint and the
    leaderboard endpoint."""
    config = bundle["config"]
    calibrator = bundle["calibrator"]
    classes = list(calibrator.classes_)
    proba_matrix = calibrator.predict_proba(X)

    # The class reported as score_pct / top SHAP factors -- config-driven so
    # binary and multiclass use cases share this code path. Defaults to the
    # highest-value class, which is class 1 for every existing binary config.
    positive_class = config.get("positive_class", max(classes))
    pos_idx = classes.index(positive_class)

    shap_values = bundle["explainer"].shap_values(X)
    model_classes = list(bundle["model"].classes_)
    shap_idx = model_classes.index(positive_class)
    if isinstance(shap_values, list):
        sv_matrix = shap_values[shap_idx]
    elif shap_values.ndim == 3:
        sv_matrix = shap_values[:, :, shap_idx]
    else:
        sv_matrix = shap_values

    results = []
    for i in range(len(X)):
        proba = proba_matrix[i]
        factors = sorted(
            zip(config["feature_columns"], sv_matrix[i]), key=lambda x: abs(x[1]), reverse=True
        )[:5]
        results.append(
            {
                "score_pct": round(float(proba[pos_idx]) * 100, 2),
                "predicted_class": str(classes[int(np.argmax(proba))]),
                "class_probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
                "top_factors": [{"feature": f, "impact": round(float(v), 4)} for f, v in factors],
            }
        )
    return results


class ScoreRequest(BaseModel):
    features: dict


class ScoreResponse(BaseModel):
    use_case: str
    score_pct: float
    predicted_class: str
    class_probabilities: dict
    top_factors: list


@app.post(
    "/score/{use_case}",
    response_model=ScoreResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def score(use_case: str, req: ScoreRequest):
    bundle = load_use_case(use_case)
    X = _build_feature_frame(bundle["config"], [req.features])
    result = _score_batch(bundle, X)[0]
    return ScoreResponse(use_case=use_case, **result)


class LeaderboardEntry(BaseModel):
    id: str
    score_pct: float
    predicted_class: str
    class_probabilities: dict
    top_factors: list


class LeaderboardResponse(BaseModel):
    use_case: str
    count: int
    results: list


@app.get(
    "/score/{use_case}/leaderboard",
    response_model=LeaderboardResponse,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def leaderboard(use_case: str, limit: int = 25):
    """Scores a sample of entities for this use case (from the config's
    sample_data CSV -- synthetic data standing in for a warehouse pull of
    entities due for scoring) and returns them ranked by score, highest
    first. Powers the dashboard's ranked list."""
    bundle = load_use_case(use_case)
    config = bundle["config"]

    sample_path = config.get("sample_data")
    if not sample_path or not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"No sample_data configured for: {use_case}")

    limit = max(1, min(limit, 500))
    df = pd.read_csv(sample_path).head(limit)
    id_col = config["id_column"]

    X = _build_feature_frame(config, df.to_dict("records"))
    scored = _score_batch(bundle, X)
    entries = [
        LeaderboardEntry(id=str(row_id), **result)
        for row_id, result in zip(df[id_col], scored)
    ]
    entries.sort(key=lambda e: e.score_pct, reverse=True)

    return LeaderboardResponse(use_case=use_case, count=len(entries), results=entries)


class UseCaseInfo(BaseModel):
    name: str
    display_name: str
    description: str
    entity: str
    label_column: str
    class_labels: dict[str, str] | None = None


@app.get("/use-cases", response_model=list[UseCaseInfo])
def list_use_cases():
    """Every registered use case with the metadata its own config already
    declares -- what it predicts and what kind of entity it scores -- so the
    dashboard (or any other client) can explain itself instead of just
    listing config filenames or raw class indices."""
    infos = []
    for path in sorted(glob.glob(f"{CONFIG_DIR}/*.yaml")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            config = yaml.safe_load(f)

        stage_names = config.get("stage_names")
        class_labels = {str(i): label for i, label in enumerate(stage_names)} if stage_names else None

        infos.append(
            UseCaseInfo(
                name=name,
                display_name=config.get("display_name", name),
                description=" ".join(config.get("description", "").split()),
                entity=config.get("entity", "unknown"),
                label_column=config.get("label_column", "unknown"),
                class_labels=class_labels,
            )
        )
    return infos


@app.get("/health")
def health():
    return {"status": "ok"}
