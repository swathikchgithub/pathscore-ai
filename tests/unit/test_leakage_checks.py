import numpy as np
import pandas as pd
import pytest

from leakage_checks import assert_no_leakage


def test_denylisted_feature_name_raises():
    config = {
        "label_column": "label",
        "feature_columns": ["email_open_rate", "contract_signed_date"],
    }
    df = pd.DataFrame(
        {"label": [0, 1], "email_open_rate": [0.1, 0.9], "contract_signed_date": [None, "2024-01-01"]}
    )
    with pytest.raises(ValueError, match="post-outcome"):
        assert_no_leakage(df, config)


def test_high_correlation_feature_raises():
    label = np.array([0, 1] * 50)
    config = {
        "label_column": "label",
        "feature_columns": ["leaky_feat", "clean_feat"],
        "categorical_columns": [],
    }
    df = pd.DataFrame(
        {
            "label": label,
            "leaky_feat": label.astype(float) + 1e-6,  # near-perfect proxy for the label
            "clean_feat": np.random.default_rng(0).normal(size=len(label)),
        }
    )
    with pytest.raises(ValueError, match="correlation"):
        assert_no_leakage(df, config)


def test_clean_features_pass(capsys):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "label": rng.integers(0, 2, size=n),
            "clean_feat_a": rng.normal(size=n),
            "clean_feat_b": rng.normal(size=n),
        }
    )
    config = {
        "label_column": "label",
        "feature_columns": ["clean_feat_a", "clean_feat_b"],
        "categorical_columns": [],
    }
    assert_no_leakage(df, config)  # should not raise
    assert "Leakage check passed" in capsys.readouterr().out
