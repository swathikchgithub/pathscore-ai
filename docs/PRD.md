# Product Requirements Document — PathScore AI

## Problem

Revenue teams (Sales, RevOps) need to prioritize where to spend outreach effort
across several distinct GTM (go-to-market) questions — which *contact* is worth
calling, which *account* has a realistic near-term path to a signed deal, where
in the funnel a contact currently sits, which accounts fit the ideal customer
profile at all. Each of these is typically built as its own bespoke model and
pipeline, which means building (and maintaining, and retraining, and explaining)
the same kind of system N separate times for N GTM motions.

## Goal

One shared scoring platform where adding a new GTM use case is a config
change — a label definition and a feature subset — not a new codebase. See
[ADR-0004](adr/0004-config-driven-single-pipeline.md).

## Users

- **RevOps / Sales Ops analyst** — consumes the ranked leaderboard and the
  per-score explanation (top SHAP factors) to prioritize outreach; doesn't
  touch config or code.
- **ML/platform engineer** — adds a new use case via a YAML file, monitors
  drift, owns the retrain cadence.

This is a demo/portfolio system simulating both roles' workflows end to end,
not a system with real production users today (see
[DEPLOYMENT.md](DEPLOYMENT.md) for current vs. target state).

## Use cases (implemented)

| Use case | Entity | Label | Question it answers |
|---|---|---|---|
| Contact Score | contact | binary | Will this contact convert to a qualified conversation in the current outreach window? |
| PTB Prospect | account | binary | Does this account have a realistic, near-term path to signed revenue? |
| Funnel Stage | contact | 4-class (MQL/SQL/Opportunity/Closed-Won) | Where in the funnel does this contact currently sit? |
| GTM Fit | account | binary | Does this account fit the ideal customer profile at all, independent of any trigger event or engagement? |

## Functional requirements

- **FR1 — Score:** For any registered use case, score a single entity given its
  feature values, returning a calibrated probability, the predicted class, and
  per-class probabilities. (`POST /score/{use_case}`)
- **FR2 — Rank:** For any registered use case, rank a sample of entities
  highest-to-lowest score. (`GET /score/{use_case}/leaderboard`)
- **FR3 — Explain:** Every score ships with the top-5 features that drove that
  *specific* prediction (SHAP), not just the number — a non-ML stakeholder must
  be able to see why an entity scored the way it did.
- **FR4 — Extend:** Adding a new use case requires only a new YAML config file
  under `src/config/` — no changes to training, serving, or dashboard code.
- **FR5 — Refresh:** Models retrain and get drift-checked on a defined cadence
  without manual intervention.
- **FR6 — Protect:** The scoring API must not be usable at unlimited,
  unauthenticated volume — both a cost control and a model-extraction defense.

## Non-functional requirements

- **NFR1 — Explainability:** Every prediction is interpretable via its SHAP
  top factors, not a black-box number.
- **NFR2 — Leakage safety:** No feature is a disguised proxy for the label
  being predicted, enforced automatically before training
  (`leakage_checks.assert_no_leakage`).
- **NFR3 — Reproducibility:** The full pipeline runs from a fresh clone with
  `pip install -r requirements.txt` and no external account required for local
  dev — Cortex and LoRA both degrade to a local mock automatically
  ([ADR-0006](adr/0006-env-driven-mock-live-clients.md)).
- **NFR4 — Cheap retrain:** CPU-only training completes in low single-digit
  minutes per use case, so a weekly retrain cadence costs almost nothing to run
  ([ADR-0001](adr/0001-lightgbm-over-neural-net.md)).
- **NFR5 — Tested at every layer:** Unit, integration, and one deliberately
  narrow end-to-end test of the real system
  ([ADR-0010](adr/0010-single-e2e-golden-path.md)).

## Out of scope (today)

- Real Snowflake/HuggingFace credentials and live inference — mocked by design
  for local/demo use.
- Multi-tenant auth or per-user RBAC — a single shared API key at most
  ([ADR-0008](adr/0008-api-key-rate-limit-over-full-auth.md)).
- A real, versioned point-in-time feature store
  ([ADR-0005](adr/0005-flat-csv-feature-snapshots.md)).
- An executed production deployment — see [DEPLOYMENT.md](DEPLOYMENT.md) for
  the recommended, not-yet-executed path.

## Success criteria (as a demo system)

- All four use cases trainable and servable end-to-end from a clean clone,
  verified live (curl against every route, dashboard interaction, E2E test).
- AUC-ROC materially above the 0.5 random baseline for every use case —
  current range 0.68–0.92 across `models/*/metrics.yaml`.
- Full test suite green: 55 Python tests, 16 dashboard tests, 2 E2E tests.
- The README's architecture claims match what's actually implemented —
  verified and corrected during this project rather than left to drift
  (see [TDD.md](TDD.md), Known Limitations).
