# ADR-0001: LightGBM over a neural net for the core scoring model

**Status:** Accepted

## Context

Every scoring use case (Contact Score, PTB Prospect, Funnel Stage, GTM Fit) predicts
a label from tabular, mostly-structured features (engagement counts, firmographic
categories, one extracted intent score) — a few thousand to tens of thousands of
rows, 3-9 feature columns. The model needs to ship with a defensible explanation
for each individual prediction (SHAP top factors), be retrainable weekly on
commodity CPU, and be easy for a new use case to adopt without new modeling code.

## Decision

Use LightGBM (gradient-boosted trees) as the one shared model type across every
use case, via a single `train.py` reading each use case's hyperparameters from its
config.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Neural net (MLP/tabular transformer) | Needs materially more data than a few thousand rows to beat a well-tuned tree ensemble on tabular features; needs a GPU to retrain quickly; SHAP support exists but `TreeExplainer` on GBTs is exact and fast, while neural nets need the slower, approximate `KernelExplainer`/`DeepExplainer`. |
| Logistic regression / linear model | Cheapest to explain (coefficients), but can't capture the non-linear interactions the label-generation formula in `synthetic_data_gen.py` actually encodes (e.g. industry × trigger-event interactions) without manual feature engineering per use case — defeats the "one shared pipeline" goal. |
| Random Forest | Similar tabular strengths to LightGBM, but typically needs more trees/memory for comparable accuracy and calibrates less cleanly out of the box. |
| XGBoost / CatBoost | Comparable to LightGBM on this problem shape; LightGBM's histogram-based training is simply faster on CPU at this data size, which matters directly for the "cheap weekly retrain" requirement. |

## Consequences

- Retraining all four use cases from scratch is a low-single-digit-minutes CPU
  operation (verified: `python src/scoring/train.py ...` per use case), which is
  what makes the weekly GitHub Actions retrain cadence (ADR-0007) cheap enough to
  actually run on every push to a public runner.
- `shap.TreeExplainer` gives exact, fast per-prediction explanations — the
  `top_factors` field in every `/score` response — with no approximation error to
  caveat.
- Ceiling: if a use case ever needs a genuinely unstructured signal (raw text,
  images) as a *primary* feature rather than a pre-extracted score, LightGBM can't
  consume it directly — that's exactly the boundary the extraction layer
  (ADR-0002, ADR-0003) exists to handle instead of stretching the core model.
