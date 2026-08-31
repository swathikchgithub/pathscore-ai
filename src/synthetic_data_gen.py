"""
Synthetic GTM data generator.

Generates three linked tables:
  accounts.csv   - company-level firmographic + fit data
  contacts.csv   - person-level records tied to an account
  events.csv     - engagement events (email opens, replies, demo requests, etc.)

Ground truth (conversion) is generated from a documented, weighted function of
features + noise, so the "labels" here are transparent and defensible in an
interview setting -- not just random.

Usage:
  python src/synthetic_data_gen.py --n-accounts 5000 --n-contacts 15000 --out data/synthetic
"""

import argparse
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(42)

INDUSTRIES = [
    "Financial Services", "Healthcare", "Legal", "Real Estate",
    "Technology", "Manufacturing", "Retail", "Government", "Insurance",
]
COMPANY_SIZES = ["1-50", "51-200", "201-1000", "1001-5000", "5000+"]
SIZE_WEIGHTS = [0.30, 0.30, 0.20, 0.12, 0.08]
JOB_TITLES_BY_SENIORITY = {
    "economic_buyer": ["CFO", "VP Finance", "Chief Legal Officer", "VP Procurement"],
    "champion": ["Director of Operations", "Head of Sales Ops", "Legal Ops Manager"],
    "technical_validator": ["IT Security Manager", "VP Engineering", "CISO"],
    "user_buyer": ["Contracts Manager", "Sales Rep", "HR Coordinator", "Paralegal"],
}

# Industries with naturally higher document-signing intensity -> higher base fit for an
# e-signature / agreement platform. This encodes domain assumption transparently.
INDUSTRY_FIT_WEIGHT = {
    "Financial Services": 0.85, "Legal": 0.90, "Real Estate": 0.80,
    "Healthcare": 0.65, "Insurance": 0.75, "Technology": 0.55,
    "Manufacturing": 0.45, "Retail": 0.40, "Government": 0.60,
}

# Funnel stage buckets, ordered by the same composite score z used for
# contact_converted. Cutpoints are documented here, not tuned to hit a target
# distribution -- they're the same kind of explicit assumption as
# INDUSTRY_FIT_WEIGHT above.
FUNNEL_STAGES = ["MQL", "SQL", "Opportunity", "Closed-Won"]
FUNNEL_CUTPOINTS = [0.35, 0.55, 0.75]

# GTM Fit label: a static screen on firmographics alone -- "is this account
# worth pursuing at all" -- independent of any trigger event or contact-level
# engagement, unlike PTB Prospect. Threshold picked from the observed
# icp_fit_score distribution to land near an even split (~53% positive),
# not tuned against any downstream metric.
GTM_FIT_THRESHOLD = 0.65

# Short synthetic email/call snippets per intent_label, standing in for the
# raw text a real extraction layer would see. Vocabulary is deliberately
# aligned with MockCortexClient's word lists (src/extraction/intent_extractor.py)
# so build_features.py's extraction step can actually recover signal from it
# in local/dev mode. Picked deterministically by event_id, not rng, so this
# doesn't perturb the rng stream that drives label generation below.
EVENT_NOTES = {
    "hot": [
        "This is exactly what we need -- ready to sign this week, can you send a contract asap?",
        "Loved the demo, very excited to move forward -- what's the urgent next step?",
        "Yes, let's do this. Please send pricing today, we need this asap.",
    ],
    "warm": [
        "Interested in learning more, the demo looked great. Can we set up a follow-up?",
        "This seems like a good fit, keen to see a proposal when you have time.",
        "Thanks for the info, I'm interested -- let me loop in our team and get back to you.",
    ],
    "neutral": [
        "Thanks for reaching out, will take a look and get back to you.",
        "Received your message, still gathering requirements internally.",
        "Noted -- we're not actively evaluating vendors right now but will keep this on file.",
    ],
    "cold": [
        "Not interested, please remove me from this list.",
        "We have no budget for this right now, please stop reaching out.",
        "Please unsubscribe me, this isn't relevant to us.",
    ],
}


def gen_accounts(n_accounts: int) -> pd.DataFrame:
    rows = []
    for i in range(n_accounts):
        industry = rng.choice(INDUSTRIES)
        size = rng.choice(COMPANY_SIZES, p=SIZE_WEIGHTS)
        icp_fit = np.clip(
            INDUSTRY_FIT_WEIGHT[industry] + rng.normal(0, 0.12), 0, 1
        )
        trigger_event = rng.choice(
            ["none", "funding_round", "leadership_change", "compliance_mandate", "tech_refresh"],
            p=[0.55, 0.12, 0.13, 0.10, 0.10],
        )
        days_since_last_touch = int(rng.exponential(30))
        rows.append({
            "account_id": f"ACC-{i:06d}",
            "company_name": fake.company(),
            "industry": industry,
            "company_size": size,
            "icp_fit_score": round(icp_fit, 3),
            "trigger_event": trigger_event,
            "days_since_last_touch": days_since_last_touch,
            "existing_customer": bool(rng.random() < 0.20),
        })
    return pd.DataFrame(rows)


def gen_contacts(accounts: pd.DataFrame, n_contacts: int) -> pd.DataFrame:
    rows = []
    account_ids = accounts["account_id"].values
    for i in range(n_contacts):
        account_id = rng.choice(account_ids)
        role = rng.choice(
            list(JOB_TITLES_BY_SENIORITY.keys()), p=[0.15, 0.25, 0.20, 0.40]
        )
        title = rng.choice(JOB_TITLES_BY_SENIORITY[role])
        rows.append({
            "contact_id": f"CON-{i:07d}",
            "account_id": account_id,
            "full_name": fake.name(),
            "title": title,
            "buying_role": role,
            "email_open_rate": round(np.clip(rng.beta(2, 3), 0, 1), 3),
            "email_reply_count_90d": int(rng.poisson(1.2)),
            "demo_requested": bool(rng.random() < 0.15),
            "linkedin_engaged": bool(rng.random() < 0.25),
        })
    return pd.DataFrame(rows)


def gen_events(contacts: pd.DataFrame, avg_events_per_contact: float = 4.0) -> pd.DataFrame:
    """Synthetic email/engagement events with an intent label and a short
    notes snippet consistent with it. intent_label is the ground truth used
    to build the conversion labels below; notes is the raw text a real
    extraction layer would see -- build_features.py recovers an intent score
    from notes independently, so features never get to see the ground truth
    directly."""
    intents = ["cold", "neutral", "warm", "hot"]
    rows = []
    event_id = 0
    for _, c in contacts.iterrows():
        n_events = rng.poisson(avg_events_per_contact)
        # Higher reply count / open rate -> skew intent distribution warmer
        warmth_bias = c["email_open_rate"] * 0.5 + min(c["email_reply_count_90d"], 5) * 0.08
        probs = np.array([0.4, 0.3, 0.2, 0.1]) 
        probs = probs + np.array([-1, -0.3, 0.4, 0.9]) * warmth_bias
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum()
        for _ in range(n_events):
            # Built as a variable (not passed inline to append) so "notes"
            # can be filled in afterward from the already-drawn intent_label
            # without adding another rng call -- that would shift every rng
            # draw after it and quietly change the downstream labels below.
            row = {
                "event_id": f"EVT-{event_id:08d}",
                "contact_id": c["contact_id"],
                "event_type": rng.choice(["email_open", "email_reply", "meeting", "content_download"]),
                "intent_label": rng.choice(intents, p=probs),
                "notes": None,
                "days_ago": int(rng.exponential(20)),
            }
            templates = EVENT_NOTES[row["intent_label"]]
            row["notes"] = templates[event_id % len(templates)]
            rows.append(row)
            event_id += 1
    return pd.DataFrame(rows)


def compute_ground_truth_labels(accounts: pd.DataFrame, contacts: pd.DataFrame, events: pd.DataFrame):
    """
    Documented ground-truth generation function -- this IS the label definition,
    kept explicit so it can be defended/explained rather than treated as a black box.

    contact_converted: contact-level positive class (person engages to a qualified meeting+)
    account_ptb: account-level positive class (account reaches signed deal)
    """
    intent_score = (
        events.groupby("contact_id")["intent_label"]
        .apply(lambda s: s.map({"cold": 0, "neutral": 0.33, "warm": 0.66, "hot": 1.0}).mean())
        .rename("avg_intent_score")
        .fillna(0.2)
    )
    contacts = contacts.merge(intent_score, on="contact_id", how="left")
    contacts["avg_intent_score"] = contacts["avg_intent_score"].fillna(0.2)

    contacts = contacts.merge(
        accounts[["account_id", "icp_fit_score", "trigger_event", "existing_customer"]],
        on="account_id", how="left",
    )

    role_weight = contacts["buying_role"].map(
        {"economic_buyer": 1.0, "champion": 0.9, "technical_validator": 0.6, "user_buyer": 0.4}
    )
    trigger_weight = contacts["trigger_event"].map(
        {"none": 0.0, "tech_refresh": 0.15, "leadership_change": 0.15,
         "compliance_mandate": 0.25, "funding_round": 0.20}
    )

    z = (
        0.30 * contacts["avg_intent_score"]
        + 0.20 * contacts["email_open_rate"]
        + 0.15 * np.minimum(contacts["email_reply_count_90d"] / 5, 1)
        + 0.15 * contacts["icp_fit_score"]
        + 0.10 * role_weight
        + 0.10 * trigger_weight
        + 0.05 * contacts["existing_customer"].astype(float)
        + rng.normal(0, 0.08, size=len(contacts))  # irreducible noise
    )
    conversion_prob = 1 / (1 + np.exp(-8 * (z - 0.5)))  # logistic squashing centered at 0.5
    contacts["contact_converted"] = (rng.random(len(contacts)) < conversion_prob).astype(int)
    contacts["true_conversion_prob"] = conversion_prob.round(4)  # kept for calibration eval only

    # Funnel stage: same composite score z as contact_converted, bucketed into
    # ordered stages instead of a single binary cutoff -- a higher score means
    # both "more likely to convert" and "further along the funnel", which is
    # the same underlying propensity viewed two ways.
    funnel_stage_idx = np.digitize(z, FUNNEL_CUTPOINTS)
    contacts["funnel_stage"] = funnel_stage_idx
    contacts["funnel_stage_label"] = [FUNNEL_STAGES[i] for i in funnel_stage_idx]

    account_ptb = (
        contacts.groupby("account_id")
        .agg(any_converted=("contact_converted", "max"), max_prob=("true_conversion_prob", "max"))
        .reset_index()
    )
    accounts = accounts.merge(account_ptb, on="account_id", how="left")
    accounts["any_converted"] = accounts["any_converted"].fillna(0)
    accounts["account_ptb_label"] = (
        (accounts["any_converted"] == 1) & (accounts["icp_fit_score"] > 0.5)
    ).astype(int)

    # GTM Fit: pure function of an already-computed column, no new rng draw,
    # so it doesn't perturb any other label above.
    accounts["gtm_fit_label"] = (accounts["icp_fit_score"] >= GTM_FIT_THRESHOLD).astype(int)

    return accounts, contacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-accounts", type=int, default=5000)
    parser.add_argument("--n-contacts", type=int, default=15000)
    parser.add_argument("--out", type=str, default="data/synthetic")
    args = parser.parse_args()

    accounts = gen_accounts(args.n_accounts)
    contacts = gen_contacts(accounts, args.n_contacts)
    events = gen_events(contacts)
    accounts, contacts = compute_ground_truth_labels(accounts, contacts, events)

    import os
    os.makedirs(args.out, exist_ok=True)
    accounts.to_csv(f"{args.out}/accounts.csv", index=False)
    contacts.to_csv(f"{args.out}/contacts.csv", index=False)
    events.to_csv(f"{args.out}/events.csv", index=False)

    print(f"Wrote {len(accounts)} accounts, {len(contacts)} contacts, {len(events)} events to {args.out}")
    print(f"Contact conversion rate: {contacts['contact_converted'].mean():.3f}")
    print(f"Account PTB rate: {accounts['account_ptb_label'].mean():.3f}")
    print(f"Funnel stage distribution:\n{contacts['funnel_stage_label'].value_counts()}")


if __name__ == "__main__":
    main()
