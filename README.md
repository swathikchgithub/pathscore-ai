# PathScore AI

A config-driven scoring platform for GTM use cases (Contact Score, PTB Prospect,
Funnel Stage, GTM Fit). One shared pipeline; each use case is just a label
definition + feature subset, not a separate codebase.

## Architecture

```
Data Sources (Snowflake / synthetic)
        |
Feature Extraction
  - structured passthrough (SQL / Snowpark)
  - Snowflake Cortex (SENTIMENT, EXTRACT_ANSWER, COMPLETE) -- default for text
  - Custom LLM + LoRA adapter -- only when Cortex is too generic
        |
Feature Store (data/features/*.csv, built by build_features.py)
        |
Core Scoring Model (LightGBM, one training pipeline, per-use-case config)
        |
Calibration (Platt scaling / isotonic)
        |
Serving (FastAPI) + Monitoring + Scheduled retrain
        |
Dashboard (Next.js) -- ranked scores, SHAP explanations
```

**On "feature store":** what's actually implemented is `build_features.py` writing a
flat CSV snapshot per use case to `data/features/`, refreshed by
`src/monitoring/run_scheduled_jobs.py` on the cadence in `.github/workflows/retrain.yml`.
That's not the point-in-time-correct store the diagram above originally implied
(retrieving feature values as they existed *at label time*, not their current
value) -- this repo's synthetic data has no time-varying dimension to make that
meaningful, and a real one would need a versioned store (Postgres/Supabase table
keyed by entity + as-of timestamp) that's out of scope for a demo. The practical
substitute doing that job here is `leakage_checks.py`'s heuristics (denylisted
post-outcome column patterns, suspiciously perfect label correlation), which
catch the specific failure mode -- a feature that couldn't have existed yet at
label time -- that a point-in-time store exists to prevent architecturally.

## Why these choices

- **LightGBM, not a neural net**, for the core scoring model: tabular data,
  needs explainability (SHAP), fast to retrain on CPU.
- **LoRA, not full fine-tuning**, for the text-extraction layer: adapts a
  frozen 7-8B base model to a narrow task (~0.1-1% of params trained),
  cheap on a single rented GPU, swappable per use case.
- **Cortex first, custom model second**: if Snowflake Cortex's built-in
  functions (sentiment, entity extraction) are specific enough, use them —
  no GPU, no data egress. Only train a custom LoRA adapter for signals
  Cortex can't capture (e.g., a company-specific "contract urgency" signal).
- **No H100s**: LoRA fine-tuning an 8B model on a few thousand examples runs
  fine on a single rented A10/A6000. H100/multi-GPU is unjustified cost here.

## Repo layout

```
data/
  synthetic/        synthetic data generator output
  features/         feature snapshots per use case, built by build_features.py
src/
  config/           per-use-case YAML label + feature definitions
  extraction/       text feature extraction (Cortex client + LoRA client)
  scoring/          build_features.py + LightGBM training/inference, shared across use cases
  serving/          FastAPI app exposing /score
  monitoring/       drift_check.py + run_scheduled_jobs.py (retrain/drift-check cadence)
notebooks/           exploration / synthetic data QA
tests/               unit + integration tests for every module above
.github/workflows/   scheduled retrain + drift-check (see run_scheduled_jobs.py)
```

Calibration (Platt scaling / isotonic) is inline in `train.py`'s `train()`, not a
separate module -- it's a few lines wrapping the fitted model, not enough to
justify its own package.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic data
python src/synthetic_data_gen.py --n-accounts 5000 --n-contacts 15000

# 2. Build features
python src/scoring/build_features.py --use-case contact_score

# 3. Train a model for a given use case
python src/scoring/train.py --config src/config/contact_score.yaml

# 4. Serve
uvicorn src.serving.app:app --reload

# 5. Dashboard (in a separate terminal)
cd dashboard
npm install
cp .env.local.example .env.local  # points at the API above by default
npm run dev
```

## Adding a new use case

Add a new YAML file under `src/config/` defining the label window, positive
class definition, and feature columns. No new training code needed — the
same `train.py` and `app.py` read the config and produce a new model.
