"""
Shared training pipeline for all GTM scoring use cases.

The same code trains the Contact Score model, the PTB Prospect model, or any
future use case -- only the YAML config changes (label column, feature
columns, model hyperparams). This is the concrete answer to "build a model
for every use case": one pipeline, N config files, not N codebases.

Usage:
  python src/scoring/train.py --config src/config/contact_score.yaml \
      --data data/synthetic/contacts.csv --out models/contact_score
"""

import argparse
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import roc_auc_score, precision_score

from leakage_checks import assert_no_leakage


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_data(df: pd.DataFrame, config: dict):
    feature_cols = config["feature_columns"]
    cat_cols = config.get("categorical_columns", [])
    label_col = config["label_column"]

    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    y = df[label_col].astype(int)
    return X, y


def precision_at_k(y_true, y_score, k_frac=0.1):
    n = len(y_true)
    k = max(1, int(n * k_frac))
    order = np.argsort(-y_score)[:k]
    return precision_score(np.array(y_true)[order], np.ones(k))


def precision_at_k_multiclass(y_true, proba, classes, k_frac=0.1):
    """Generic multiclass extension of precision_at_k: for each class, rank
    samples by that class's own predicted probability and measure precision
    in the top-k fraction, then macro-average across classes."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = max(1, int(n * k_frac))
    precisions = []
    for i, c in enumerate(classes):
        order = np.argsort(-proba[:, i])[:k]
        precisions.append(np.mean(y_true[order] == c))
    return float(np.mean(precisions))


def train(config_path: str, data_path: str, out_dir: str):
    config = load_config(config_path)
    df = pd.read_csv(data_path)

    # Point-in-time / leakage guard before anything else touches the data
    assert_no_leakage(df, config)

    X, y = prepare_data(df, config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = config["model"]["params"]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, categorical_feature=config.get("categorical_columns", []))

    raw_proba = model.predict_proba(X_test)
    n_classes = len(model.classes_)

    if n_classes == 2:
        raw_scores = raw_proba[:, 1]
        auc = roc_auc_score(y_test, raw_scores)
        p_at_10 = precision_at_k(y_test.values, raw_scores, k_frac=0.10)
    else:
        # Generic multiclass extension: macro-averaged one-vs-rest AUC, and
        # the multiclass precision@k defined above. No use-case-specific
        # branching -- this path runs for any config with >2 classes.
        auc = roc_auc_score(y_test, raw_proba, multi_class="ovr", average="macro")
        p_at_10 = precision_at_k_multiclass(y_test.values, raw_proba, model.classes_, k_frac=0.10)

    print(f"[{config['use_case']}] AUC-ROC: {auc:.4f} | Precision@10%: {p_at_10:.4f} | classes: {n_classes}")

    # Calibration layer -- separate, lightweight fit on top of raw scores
    calib_method = config.get("calibration", {}).get("method", "platt")
    sigmoid_or_isotonic = "sigmoid" if calib_method == "platt" else "isotonic"
    if FrozenEstimator is not None:
        # scikit-learn >= 1.6: wrap an already-fitted model with FrozenEstimator
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=sigmoid_or_isotonic)
    else:
        # older scikit-learn: cv="prefit" does the same thing
        calibrator = CalibratedClassifierCV(model, method=sigmoid_or_isotonic, cv="prefit")
    calibrator.fit(X_test, y_test)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, f"{out_dir}/model.joblib")
    joblib.dump(calibrator, f"{out_dir}/calibrator.joblib")

    with open(f"{out_dir}/metrics.yaml", "w") as f:
        yaml.dump(
            {"auc_roc": float(auc), "precision_at_10pct": float(p_at_10), "n_classes": n_classes},
            f,
        )

    print(f"Saved model + calibrator to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args.config, args.data, args.out)
