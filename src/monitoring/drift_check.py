"""
Drift monitoring for trained scoring models.

Recomputes each use case's tracked metric (AUC-ROC or precision@10%, as
declared in the config's `retrain` block) on a rolling window of newly
labeled data, compares it against the metric recorded at training time
(models/<use_case>/metrics.yaml), and flags drift when it drops by more
than the configured threshold.

One shared script, N config files -- same principle as train.py. Reuses
train.py's metric functions directly instead of redefining them, so the
drift check and the training-time baseline can't quietly disagree on what
"AUC-ROC" or "precision@10%" means.

The "rolling window" here is a CSV of recently-labeled records supplied via
--data -- standing in for a query against a feature store / warehouse for
labels accumulated since the last check. Wiring that up is future work, same
as the Cortex/LoRA extractors are stubbed pending live endpoints.

Usage:
  python src/monitoring/drift_check.py \
      --config src/config/contact_score.yaml \
      --data data/synthetic/contacts.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scoring"))
from train import precision_at_k, precision_at_k_multiclass, prepare_data  # noqa: E402

LOG_PATH_DEFAULT = "monitoring/drift_log.jsonl"
DASHBOARD_PATH_DEFAULT = "monitoring/dashboard.md"

# metrics.yaml (written by train.py) stores precision@10% under this key --
# "precision_at_k" in a config's drift_check_metric means the same thing;
# the k is fixed at 10% inside train.py, not separately configurable.
BASELINE_METRIC_KEY = {"auc_roc": "auc_roc", "precision_at_k": "precision_at_10pct"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_bundle(use_case: str, models_dir: str):
    model = joblib.load(f"{models_dir}/{use_case}/model.joblib")
    calibrator = joblib.load(f"{models_dir}/{use_case}/calibrator.joblib")
    with open(f"{models_dir}/{use_case}/metrics.yaml") as f:
        baseline = yaml.safe_load(f)
    return model, calibrator, baseline


def compute_metric(metric_name: str, y_true, proba, classes):
    n_classes = len(classes)
    pos_idx = list(classes).index(max(classes))

    if metric_name == "auc_roc":
        if n_classes == 2:
            return roc_auc_score(y_true, proba[:, pos_idx])
        return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    elif metric_name == "precision_at_k":
        if n_classes == 2:
            return precision_at_k(y_true.values, proba[:, pos_idx], k_frac=0.10)
        return precision_at_k_multiclass(y_true.values, proba, classes, k_frac=0.10)
    raise ValueError(f"Unknown drift_check_metric: {metric_name}")


def check_drift(
    config_path: str,
    data_path: str,
    models_dir: str = "models",
    log_path: str = LOG_PATH_DEFAULT,
    dashboard_path: str = DASHBOARD_PATH_DEFAULT,
) -> dict:
    config = load_config(config_path)
    use_case = config["use_case"]
    retrain_cfg = config.get("retrain", {})
    metric_name = retrain_cfg.get("drift_check_metric", "auc_roc")
    threshold = retrain_cfg.get("drift_threshold", 0.03)
    min_new_labels = retrain_cfg.get("min_new_labels", 0)

    df = pd.read_csv(data_path)
    n_samples = len(df)

    model, calibrator, baseline = load_bundle(use_case, str(models_dir))
    baseline_value = baseline[BASELINE_METRIC_KEY[metric_name]]

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "use_case": use_case,
        "metric": metric_name,
        "baseline_value": baseline_value,
        "n_samples": n_samples,
        "min_new_labels": min_new_labels,
        "sufficient_data": n_samples >= min_new_labels,
    }

    if not result["sufficient_data"]:
        result.update(
            {"current_value": None, "delta": None, "threshold": threshold,
             "drifted": False, "status": "insufficient_data"}
        )
    else:
        X, y = prepare_data(df, config)
        proba = calibrator.predict_proba(X)
        current_value = float(compute_metric(metric_name, y, proba, calibrator.classes_))
        delta = current_value - baseline_value
        drifted = bool(delta < -threshold)
        result.update(
            {"current_value": current_value, "delta": delta, "threshold": threshold,
             "drifted": drifted, "status": "drift_detected" if drifted else "ok"}
        )

    _append_log(result, str(log_path))
    _write_dashboard(str(log_path), str(dashboard_path))
    _print_summary(result)
    return result


def _append_log(result: dict, log_path: str):
    dirname = os.path.dirname(log_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def _write_dashboard(log_path: str, dashboard_path: str):
    if not os.path.exists(log_path):
        return
    with open(log_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    latest_by_use_case = {}
    for row in rows:
        latest_by_use_case[row["use_case"]] = row  # log is append-only chronological -> last write wins

    status_label = {"ok": "OK", "drift_detected": "DRIFT", "insufficient_data": "INSUFFICIENT DATA"}
    lines = [
        "# PathScore AI -- Drift Monitoring Dashboard",
        "",
        f"_Last updated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "| Use Case | Status | Metric | Baseline | Current | Delta | Threshold | Checked At |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for use_case, row in sorted(latest_by_use_case.items()):
        current = f"{row['current_value']:.4f}" if row["current_value"] is not None else "-"
        delta = f"{row['delta']:+.4f}" if row["delta"] is not None else "-"
        lines.append(
            f"| {use_case} | {status_label[row['status']]} | {row['metric']} | "
            f"{row['baseline_value']:.4f} | {current} | {delta} | {row['threshold']} | {row['timestamp']} |"
        )

    dirname = os.path.dirname(dashboard_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(dashboard_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(result: dict):
    if result["status"] == "insufficient_data":
        print(
            f"[{result['use_case']}] Only {result['n_samples']} labeled rows "
            f"(< min_new_labels={result['min_new_labels']}) -- skipping drift check."
        )
    else:
        verdict = "DRIFT DETECTED" if result["drifted"] else "OK"
        print(
            f"[{result['use_case']}] {result['metric']}: baseline={result['baseline_value']:.4f} "
            f"current={result['current_value']:.4f} delta={result['delta']:+.4f} "
            f"(flags if drop > {result['threshold']}) -> {verdict}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--log", default=LOG_PATH_DEFAULT)
    parser.add_argument("--dashboard", default=DASHBOARD_PATH_DEFAULT)
    args = parser.parse_args()
    check_drift(args.config, args.data, args.models_dir, args.log, args.dashboard)
