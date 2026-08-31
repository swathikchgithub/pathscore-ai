import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# train.py and leakage_checks.py import each other with bare module names
# (`from leakage_checks import ...`), i.e. they're run as scripts, not as a
# package. Match that here instead of restructuring src/ into a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "scoring"))


@pytest.fixture
def binary_config():
    return {
        "use_case": "test_binary",
        "label_column": "label",
        "id_column": "id",
        "feature_columns": ["feat_a", "feat_b", "cat_feat"],
        "categorical_columns": ["cat_feat"],
        "model": {
            "type": "lightgbm",
            "params": {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "objective": "binary",
            },
        },
        "calibration": {"method": "platt"},
    }


@pytest.fixture
def binary_dataframe():
    rng = np.random.default_rng(0)
    n = 200
    feat_a = rng.normal(size=n)
    feat_b = rng.normal(size=n)
    cat_feat = rng.choice(["x", "y", "z"], size=n)
    label = (feat_a + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "id": [f"ID-{i}" for i in range(n)],
            "feat_a": feat_a,
            "feat_b": feat_b,
            "cat_feat": cat_feat,
            "label": label,
        }
    )


@pytest.fixture
def multiclass_config():
    return {
        "use_case": "test_multiclass",
        "label_column": "label",
        "id_column": "id",
        "feature_columns": ["feat_a", "feat_b", "cat_feat"],
        "categorical_columns": ["cat_feat"],
        "model": {
            "type": "lightgbm",
            "params": {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "objective": "multiclass",
                "num_class": 4,
            },
        },
        "calibration": {"method": "platt"},
    }


@pytest.fixture
def multiclass_dataframe():
    rng = np.random.default_rng(1)
    n = 400
    feat_a = rng.normal(size=n)
    feat_b = rng.normal(size=n)
    cat_feat = rng.choice(["x", "y", "z"], size=n)
    score = feat_a + rng.normal(scale=0.3, size=n)
    label = pd.cut(
        score, bins=[-np.inf, -0.5, 0, 0.5, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)
    return pd.DataFrame(
        {
            "id": [f"ID-{i}" for i in range(n)],
            "feat_a": feat_a,
            "feat_b": feat_b,
            "cat_feat": cat_feat,
            "label": label,
        }
    )
