"""
Feature-building pipeline: joins raw entity tables with extracted signals
into a single feature table per use case.

For contact-entity use cases (contact_score, funnel_stage), this is where
the extraction layer actually runs: get_extractor(config) resolves to
CortexExtractor -- Snowflake Cortex live, or MockCortexClient locally, see
src/extraction/intent_extractor.py -- and scores each contact's engagement
history to produce avg_intent_score, replacing the value baked into the
synthetic label-generation step. For account-entity use cases (ptb_prospect),
there's no text to extract from -- accounts.csv already carries its rolled-up
features -- so this is a pass-through.

One script, N config files -- same principle as train.py: entity type (not
use case) decides which code path runs.

Usage:
  python src/scoring/build_features.py --use-case contact_score
  python src/scoring/build_features.py --use-case ptb_prospect \
      --data-dir data/synthetic --out data/features
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extraction"))
from intent_extractor import get_extractor  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_contact_intent(events: pd.DataFrame, config: dict) -> pd.Series:
    """Runs the configured extractor over every event's note text and
    averages the resulting intent_score per contact -- the actual call site
    for CortexExtractor/LoRAExtractor that train.py used to skip entirely.

    Time:  O(n) -- one extractor call per event, plus one groupby-mean.
    Space: O(k) for the per-contact aggregate, k = distinct contacts.
    """
    extractor = get_extractor(config)
    scores = [
        extractor.extract(
            row.contact_id, str(row.notes) if pd.notna(row.notes) else ""
        ).intent_score
        for row in events.itertuples(index=False)
    ]
    return (
        events.assign(_intent_score=scores)
        .groupby("contact_id")["_intent_score"]
        .mean()
        .rename("avg_intent_score")
    )


def build_contact_features(data_dir: str, config: dict) -> pd.DataFrame:
    contacts = pd.read_csv(f"{data_dir}/contacts.csv")
    events = pd.read_csv(f"{data_dir}/events.csv")

    extracted = _extract_contact_intent(events, config)
    features = contacts.drop(columns=["avg_intent_score"]).merge(
        extracted, on="contact_id", how="left"
    )
    # Contacts with no logged events get the same neutral default the
    # synthetic generator uses for the same case.
    features["avg_intent_score"] = features["avg_intent_score"].fillna(0.2)
    return features


def build_account_features(data_dir: str, config: dict) -> pd.DataFrame:
    return pd.read_csv(f"{data_dir}/accounts.csv")


def _select_builder(entity: str):
    if entity == "contact":
        return build_contact_features
    if entity == "account":
        return build_account_features
    raise ValueError(f"Unknown entity type for build_features: {entity}")


def build_features(
    use_case: str, data_dir: str = "data/synthetic", out_dir: str = "data/features"
) -> str:
    config = load_config(f"src/config/{use_case}.yaml")
    builder = _select_builder(config["entity"])
    features = builder(data_dir, config)

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{use_case}.csv"
    features.to_csv(out_path, index=False)
    print(f"[{use_case}] Wrote {len(features)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--out", default="data/features")
    args = parser.parse_args()
    build_features(args.use_case, args.data_dir, args.out)
