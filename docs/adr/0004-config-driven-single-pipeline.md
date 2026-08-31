# ADR-0004: One config-driven pipeline over N per-use-case codebases

**Status:** Accepted

## Context

Four GTM scoring use cases exist today (Contact Score, PTB Prospect, Funnel
Stage, GTM Fit), and the README states the explicit goal: adding a fifth should
be "a label definition + feature subset, not a separate codebase." The use
cases differ in entity (contact vs. account), label type (binary vs. 4-class),
feature set, and model hyperparameters — but not in the *shape* of the pipeline
that produces them.

## Decision

One `train.py`, one `build_features.py`, one `app.py`, one `drift_check.py` —
each reads a YAML config (`src/config/<use_case>.yaml`) that declares
`entity`, `label_column`, `id_column`, `feature_columns`, `categorical_columns`,
`model.params`, `calibration.method`, and `retrain.*`. Adding GTM Fit
(this repo's own proof point) required exactly one new YAML file and zero
changes to `build_features.py`, `train.py`, `app.py`, or the dashboard —
verified live via `/use-cases` and `/score/gtm_fit/leaderboard` immediately
picking it up.

## Alternatives considered

| Option | Why rejected |
|---|---|
| A separate module/package per use case | Four (soon more) near-duplicate copies of training/serving/leakage-check logic — every bug fix or improvement (e.g. the `_score_batch` vectorization, the leakage guard) would need to be ported N times instead of landing once. |
| A single hardcoded script with `if use_case == "contact_score": ...` branches | Same duplication problem in a different shape — the branch count grows linearly with use cases, and it's easy for one branch to silently drift from the others' behavior (e.g. one forgetting the leakage check). |
| A plugin/registry system (each use case registers a Python class) | More machinery than four declarative differences (label, features, hyperparameters, cadence) actually need — YAML already expresses all of it without any code, and a plugin system would require code changes to add a use case, defeating the goal. |

## Consequences

- `entity` (`contact` vs. `account`) is the one place behavior genuinely
  branches (`build_features._select_builder`), and it's a two-way dispatch, not
  a per-use-case branch — a fifth account-entity use case reuses the same
  `build_account_features` pass-through GTM Fit and PTB Prospect already share.
- Config validation is currently implicit (a config missing a required key
  fails wherever that key is first read, e.g. `KeyError` in `train.py`) rather
  than validated up front against a schema — acceptable at 4 configs, a real
  gap if this grew to dozens (see TDD.md, Known Limitations).
- Every use case inherits fixes and hardening automatically: the API-key/rate-
  limit dependencies (ADR-0008) and the missing-feature 422 validation apply to
  every `/score/{use_case}` route the moment they landed in `app.py`, with zero
  per-use-case wiring.
