"""
FastAPI serving layer. Generic across every use case registered in
src/config/*.yaml -- loads the right model + config by name rather than
hardcoding per-use-case routes.

Run:
  uvicorn src.serving.app:app --reload
"""

import glob
import os
import yaml
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    df = pd.DataFrame(records)[config["feature_columns"]]
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


@app.post("/score/{use_case}", response_model=ScoreResponse)
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


@app.get("/score/{use_case}/leaderboard", response_model=LeaderboardResponse)
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


@app.get("/use-cases")
def list_use_cases():
    files = glob.glob(f"{CONFIG_DIR}/*.yaml")
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


@app.get("/health")
def health():
    return {"status": "ok"}
