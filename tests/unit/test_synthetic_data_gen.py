import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import synthetic_data_gen as sdg  # noqa: E402


def _accounts(icp_scores):
    return pd.DataFrame(
        {
            "account_id": [f"ACC-{i}" for i in range(len(icp_scores))],
            "icp_fit_score": icp_scores,
            "trigger_event": "none",
            "existing_customer": False,
        }
    )


def _contacts(account_ids):
    return pd.DataFrame(
        {
            "contact_id": [f"CON-{i}" for i in range(len(account_ids))],
            "account_id": account_ids,
            "buying_role": "user_buyer",
            "email_open_rate": 0.3,
            "email_reply_count_90d": 0,
        }
    )


def _events(contact_ids):
    return pd.DataFrame({"contact_id": contact_ids, "intent_label": "neutral"})


def test_gtm_fit_label_is_a_threshold_on_icp_fit_score():
    icp_scores = [0.10, 0.64, 0.65, 0.66, 0.99]
    accounts = _accounts(icp_scores)
    contacts = _contacts(accounts["account_id"].tolist())
    events = _events(contacts["contact_id"].tolist())

    accounts_out, _ = sdg.compute_ground_truth_labels(accounts, contacts, events)

    expected = [int(s >= sdg.GTM_FIT_THRESHOLD) for s in icp_scores]
    assert accounts_out["gtm_fit_label"].tolist() == expected
