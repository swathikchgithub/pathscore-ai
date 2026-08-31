import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "serving"))

import app as app_module  # noqa: E402
from app import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_model_cache():
    # _model_cache is a module-level dict shared across requests in the real
    # app (that's the point -- avoid reloading a model per request); in
    # tests it would otherwise leak fake bundles between cases that reuse a
    # use-case name.
    app_module._model_cache.clear()
    yield
    app_module._model_cache.clear()


class _FakeModel:
    def __init__(self, classes):
        self.classes_ = classes


class _FakeCalibrator:
    def __init__(self, classes, proba):
        self.classes_ = classes
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return self._proba


class _FakeExplainer:
    def __init__(self, shap_values):
        self._shap_values = shap_values

    def shap_values(self, X):
        return self._shap_values


def _binary_bundle(proba, shap_values, extra_config=None):
    classes = [0, 1]
    config = {
        "feature_columns": ["a", "b"],
        "categorical_columns": [],
        **(extra_config or {}),
    }
    return {
        "config": config,
        "model": _FakeModel(classes),
        "calibrator": _FakeCalibrator(classes, proba),
        "explainer": _FakeExplainer(shap_values),
    }


# --- /health, /use-cases -----------------------------------------------


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_use_cases_lists_configs_found_in_config_dir(tmp_path, monkeypatch):
    (tmp_path / "alpha.yaml").write_text("use_case: alpha\n")
    (tmp_path / "beta.yaml").write_text("use_case: beta\n")
    monkeypatch.setattr(app_module, "CONFIG_DIR", str(tmp_path))

    response = client.get("/use-cases")

    assert response.status_code == 200
    assert sorted(response.json()) == ["alpha", "beta"]


# --- /score/{use_case} ---------------------------------------------------


def test_score_unknown_use_case_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CONFIG_DIR", str(tmp_path))

    response = client.post("/score/nope", json={"features": {"a": 1, "b": 2}})

    assert response.status_code == 404
    assert "Unknown use case" in response.json()["detail"]


def test_score_returns_404_when_model_not_trained(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "untrained.yaml").write_text("use_case: untrained\nfeature_columns: [a]\n")
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(app_module, "CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(app_module, "MODELS_DIR", str(tmp_path / "models"))

    response = client.post("/score/untrained", json={"features": {"a": 1}})

    assert response.status_code == 404
    assert "not trained yet" in response.json()["detail"]


def test_score_rejects_a_request_missing_features():
    response = client.post("/score/anything", json={})
    assert response.status_code == 422


def test_score_happy_path_binary_list_shap(monkeypatch):
    # Older-SHAP-style output: a list of one 2D array per class.
    shap_for_class_0 = np.array([[0.1, -0.2]])
    shap_for_class_1 = np.array([[-0.1, 0.2]])
    bundle = _binary_bundle(proba=[[0.3, 0.7]], shap_values=[shap_for_class_0, shap_for_class_1])
    app_module._model_cache["binary_list"] = bundle

    response = client.post("/score/binary_list", json={"features": {"a": 1, "b": 2}})

    assert response.status_code == 200
    body = response.json()
    assert body["use_case"] == "binary_list"
    assert body["score_pct"] == 70.0
    assert body["predicted_class"] == "1"
    assert body["class_probabilities"] == {"0": 0.3, "1": 0.7}
    assert body["top_factors"][0]["feature"] == "b"  # larger |impact| (0.2) ranks first
    assert len(body["top_factors"]) == 2


def test_score_happy_path_binary_ndarray_shap_sorts_factors_by_abs_impact(monkeypatch):
    shap_values = np.array([[0.05, -0.4]])  # |b| > |a|
    bundle = _binary_bundle(proba=[[0.6, 0.4]], shap_values=shap_values)
    app_module._model_cache["binary_ndarray"] = bundle

    response = client.post("/score/binary_ndarray", json={"features": {"a": 10, "b": 20}})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "0"
    assert [f["feature"] for f in body["top_factors"]] == ["b", "a"]


def test_score_happy_path_multiclass_explicit_positive_class(monkeypatch):
    classes = [0, 1, 2, 3]
    config = {
        "feature_columns": ["a", "b"],
        "categorical_columns": [],
        "positive_class": 3,
    }
    proba = np.array([[0.1, 0.1, 0.1, 0.7]])
    # ndim==3 SHAP layout: (n_samples, n_features, n_classes)
    shap_values = np.zeros((1, 2, 4))
    shap_values[0, :, 3] = [0.9, -0.05]  # only class-3's slice should be read
    bundle = {
        "config": config,
        "model": _FakeModel(classes),
        "calibrator": _FakeCalibrator(classes, proba),
        "explainer": _FakeExplainer(shap_values),
    }
    app_module._model_cache["multiclass"] = bundle

    response = client.post("/score/multiclass", json={"features": {"a": 1, "b": 2}})

    assert response.status_code == 200
    body = response.json()
    assert body["score_pct"] == 70.0
    assert body["predicted_class"] == "3"
    assert body["top_factors"][0] == {"feature": "a", "impact": 0.9}


# --- /score/{use_case}/leaderboard ---------------------------------------


def test_leaderboard_returns_results_sorted_highest_score_first(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame(
        {"id": ["X1", "X2", "X3"], "a": [1, 3, 5], "b": [2, 4, 6]}
    ).to_csv(sample, index=False)

    bundle = _binary_bundle(
        proba=[[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]],
        shap_values=np.zeros((3, 2)),
        extra_config={"sample_data": str(sample), "id_column": "id"},
    )
    app_module._model_cache["leaderboard_uc"] = bundle

    response = client.get("/score/leaderboard_uc/leaderboard")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [r["id"] for r in body["results"]] == ["X2", "X3", "X1"]
    assert body["results"][0]["score_pct"] == 90.0


def test_leaderboard_returns_404_without_sample_data_configured():
    bundle = _binary_bundle(proba=[[0.5, 0.5]], shap_values=np.zeros((1, 2)))
    app_module._model_cache["no_sample"] = bundle

    response = client.get("/score/no_sample/leaderboard")

    assert response.status_code == 404
    assert "No sample_data configured" in response.json()["detail"]


def test_leaderboard_clamps_limit_to_the_1_to_500_range(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": ["X1", "X2", "X3"], "a": [1, 2, 3], "b": [1, 2, 3]}).to_csv(
        sample, index=False
    )
    bundle = _binary_bundle(
        proba=[[0.5, 0.5]] * 3,
        shap_values=np.zeros((3, 2)),
        extra_config={"sample_data": str(sample), "id_column": "id"},
    )
    app_module._model_cache["clamped_uc"] = bundle

    zero_limit = client.get("/score/clamped_uc/leaderboard?limit=0")
    huge_limit = client.get("/score/clamped_uc/leaderboard?limit=10000")

    assert zero_limit.json()["count"] == 1  # clamped up to at least 1
    assert huge_limit.json()["count"] == 3  # clamped down to 500, but only 3 rows exist
