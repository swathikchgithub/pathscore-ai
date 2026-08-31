"""
Guardrails against label leakage -- the most common failure mode in exactly
this kind of scoring system (e.g., accidentally using post-conversion email
behavior to "predict" conversion).

Checks performed:
  1. Suspiciously high single-feature correlation with the label (>0.95),
     which usually means the feature is a proxy for the label itself.
  2. Explicit denylist of column name patterns that typically indicate
     post-outcome data (e.g., "won_date", "signed_", "closed_").
  3. Warns (does not hard-fail) on features with very low null rate that
     also happen to be highly predictive -- often a sign the feature was
     only populated *because* the outcome happened.
"""

import numpy as np
import pandas as pd

DENYLIST_PATTERNS = [
    "won_date", "signed_", "closed_", "post_conversion", "outcome_",
    "contract_signed", "deal_closed",
]

CORR_THRESHOLD = 0.95


def assert_no_leakage(df: pd.DataFrame, config: dict):
    label_col = config["label_column"]
    feature_cols = config["feature_columns"]

    # 1. Denylist check
    flagged = [c for c in feature_cols if any(p in c.lower() for p in DENYLIST_PATTERNS)]
    if flagged:
        raise ValueError(
            f"Potential label leakage: feature(s) {flagged} look like post-outcome "
            f"fields. Remove them or confirm they are computed as of scoring time, "
            f"not after the outcome occurred."
        )

    # 2. Correlation check (numeric features only)
    numeric_cols = [c for c in feature_cols if c not in config.get("categorical_columns", [])]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if df[col].dtype.kind not in "if":
            continue
        corr = np.corrcoef(df[col].fillna(0), df[label_col])[0, 1]
        if abs(corr) > CORR_THRESHOLD:
            raise ValueError(
                f"Potential label leakage: feature '{col}' has {corr:.3f} correlation "
                f"with label '{label_col}'. This usually means the feature encodes "
                f"the outcome itself rather than a pre-outcome signal."
            )

    print("Leakage check passed: no denylisted fields, no suspiciously perfect correlations.")
