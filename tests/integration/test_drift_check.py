import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "monitoring"))
import drift_check as drift_module  # noqa: E402

import train as train_module  # noqa: E402 (already on sys.path via conftest)


def _train_baseline_model(tmp_path, config, df):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    data_path = tmp_path / "train_data.csv"
    df.to_csv(data_path, index=False)
    models_dir = tmp_path / "models"
    train_module.train(str(config_path), str(data_path), str(models_dir / config["use_case"]))
    return str(config_path), str(models_dir)


def test_check_drift_reports_ok_on_same_distribution(tmp_path, binary_config, binary_dataframe):
    config = dict(binary_config)
    config["retrain"] = {"drift_check_metric": "auc_roc", "drift_threshold": 0.5, "min_new_labels": 10}
    config_path, models_dir = _train_baseline_model(tmp_path, config, binary_dataframe)

    recent_data_path = tmp_path / "recent.csv"
    binary_dataframe.to_csv(recent_data_path, index=False)

    log_path = tmp_path / "log.jsonl"
    dashboard_path = tmp_path / "dashboard.md"
    result = drift_module.check_drift(
        config_path, str(recent_data_path), models_dir=models_dir,
        log_path=str(log_path), dashboard_path=str(dashboard_path),
    )

    assert result["status"] == "ok"
    assert result["drifted"] is False
    assert log_path.exists()
    assert dashboard_path.exists()
    logged = json.loads(log_path.read_text().splitlines()[-1])
    assert logged["use_case"] == config["use_case"]
    assert "PathScore AI" in dashboard_path.read_text()


def test_check_drift_flags_drift_when_metric_drops(tmp_path, binary_config, binary_dataframe):
    config = dict(binary_config)
    config["retrain"] = {"drift_check_metric": "auc_roc", "drift_threshold": 0.01, "min_new_labels": 10}
    config_path, models_dir = _train_baseline_model(tmp_path, config, binary_dataframe)

    # Shuffle labels in the "recent" window to destroy the model's real
    # signal -> AUC should drop well below the training baseline.
    degraded = binary_dataframe.copy()
    degraded["label"] = degraded["label"].sample(frac=1, random_state=1).reset_index(drop=True)
    recent_data_path = tmp_path / "recent.csv"
    degraded.to_csv(recent_data_path, index=False)

    result = drift_module.check_drift(
        config_path, str(recent_data_path), models_dir=models_dir,
        log_path=str(tmp_path / "log.jsonl"), dashboard_path=str(tmp_path / "dashboard.md"),
    )

    assert result["status"] == "drift_detected"
    assert result["drifted"] is True
    assert result["delta"] < 0


def test_check_drift_skips_when_below_min_new_labels(tmp_path, binary_config, binary_dataframe):
    config = dict(binary_config)
    config["retrain"] = {"drift_check_metric": "auc_roc", "drift_threshold": 0.03, "min_new_labels": 10_000}
    config_path, models_dir = _train_baseline_model(tmp_path, config, binary_dataframe)

    recent_data_path = tmp_path / "recent.csv"
    binary_dataframe.to_csv(recent_data_path, index=False)

    result = drift_module.check_drift(
        config_path, str(recent_data_path), models_dir=models_dir,
        log_path=str(tmp_path / "log.jsonl"), dashboard_path=str(tmp_path / "dashboard.md"),
    )

    assert result["status"] == "insufficient_data"
    assert result["drifted"] is False
    assert result["current_value"] is None


def test_check_drift_precision_at_k_metric(tmp_path, binary_config, binary_dataframe):
    config = dict(binary_config)
    config["retrain"] = {"drift_check_metric": "precision_at_k", "drift_threshold": 0.5, "min_new_labels": 10}
    config_path, models_dir = _train_baseline_model(tmp_path, config, binary_dataframe)

    recent_data_path = tmp_path / "recent.csv"
    binary_dataframe.to_csv(recent_data_path, index=False)

    result = drift_module.check_drift(
        config_path, str(recent_data_path), models_dir=models_dir,
        log_path=str(tmp_path / "log.jsonl"), dashboard_path=str(tmp_path / "dashboard.md"),
    )

    assert result["metric"] == "precision_at_k"
    assert result["status"] == "ok"


def test_check_drift_multiclass(tmp_path, multiclass_config, multiclass_dataframe):
    config = dict(multiclass_config)
    config["retrain"] = {"drift_check_metric": "auc_roc", "drift_threshold": 0.5, "min_new_labels": 10}
    config_path, models_dir = _train_baseline_model(tmp_path, config, multiclass_dataframe)

    recent_data_path = tmp_path / "recent.csv"
    multiclass_dataframe.to_csv(recent_data_path, index=False)

    result = drift_module.check_drift(
        config_path, str(recent_data_path), models_dir=models_dir,
        log_path=str(tmp_path / "log.jsonl"), dashboard_path=str(tmp_path / "dashboard.md"),
    )

    assert result["status"] == "ok"
    assert 0.0 <= result["current_value"] <= 1.0
