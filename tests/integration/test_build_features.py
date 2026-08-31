import pandas as pd
import pytest

import build_features as bf

CONTACT_CONFIG = {
    "use_case": "test_contact",
    "entity": "contact",
    "label_column": "converted",
    "id_column": "contact_id",
    "feature_columns": ["avg_intent_score", "email_open_rate"],
    "extraction": {"backend": "cortex"},
}

ACCOUNT_CONFIG = {
    "use_case": "test_account",
    "entity": "account",
    "label_column": "ptb",
    "id_column": "account_id",
    "feature_columns": ["icp_fit_score"],
}


def test_build_contact_features_recomputes_avg_intent_score_from_notes(tmp_path):
    contacts = pd.DataFrame(
        {
            "contact_id": ["C1", "C2", "C3"],
            "email_open_rate": [0.5, 0.2, 0.9],
            "avg_intent_score": [0.99, 0.99, 0.99],  # stale value that must be overwritten
            "converted": [1, 0, 1],
        }
    )
    events = pd.DataFrame(
        {
            "contact_id": ["C1", "C1", "C2"],
            "notes": [
                "Very interested, ready to sign this week, urgent.",
                "Loved the demo, excited to move forward.",
                "Not interested, please unsubscribe, no budget.",
            ],
        }
    )
    contacts.to_csv(tmp_path / "contacts.csv", index=False)
    events.to_csv(tmp_path / "events.csv", index=False)

    features = bf.build_contact_features(str(tmp_path), CONTACT_CONFIG)
    by_id = features.set_index("contact_id")

    assert set(features["contact_id"]) == {"C1", "C2", "C3"}
    assert (features["avg_intent_score"] != 0.99).all()
    assert features["avg_intent_score"].between(0, 1).all()
    assert by_id.loc["C1", "avg_intent_score"] > by_id.loc["C2", "avg_intent_score"]
    # Non-extracted columns pass through untouched.
    assert by_id.loc["C1", "email_open_rate"] == 0.5
    assert by_id.loc["C1", "converted"] == 1


def test_build_contact_features_defaults_missing_events_to_neutral(tmp_path):
    contacts = pd.DataFrame(
        {
            "contact_id": ["C1", "C2"],
            "avg_intent_score": [0.5, 0.5],
            "converted": [0, 0],
        }
    )
    events = pd.DataFrame(
        {"contact_id": ["C1"], "notes": ["Interested, sounds great."]}
    )
    contacts.to_csv(tmp_path / "contacts.csv", index=False)
    events.to_csv(tmp_path / "events.csv", index=False)

    features = bf.build_contact_features(str(tmp_path), CONTACT_CONFIG)

    c2_score = features.set_index("contact_id").loc["C2", "avg_intent_score"]
    assert c2_score == 0.2  # same neutral default the synthetic generator uses


def test_build_account_features_is_a_passthrough(tmp_path):
    accounts = pd.DataFrame(
        {"account_id": ["A1", "A2"], "icp_fit_score": [0.8, 0.3], "ptb": [1, 0]}
    )
    accounts.to_csv(tmp_path / "accounts.csv", index=False)

    features = bf.build_account_features(str(tmp_path), ACCOUNT_CONFIG)

    pd.testing.assert_frame_equal(features, accounts)


def test_select_builder_dispatches_on_entity():
    assert bf._select_builder("contact") is bf.build_contact_features
    assert bf._select_builder("account") is bf.build_account_features


def test_select_builder_rejects_unknown_entity():
    with pytest.raises(ValueError, match="Unknown entity type"):
        bf._select_builder("campaign")
