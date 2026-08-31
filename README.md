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
Feature Store (Postgres/Supabase, point-in-time correct)
        |
Core Scoring Model (LightGBM, one training pipeline, per-use-case config)
        |
Calibration (Platt scaling / isotonic)
        |
Serving (FastAPI) + Monitoring + Scheduled retrain
        |
Dashboard (Next.js) -- ranked scores, SHAP explanations
```

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
  features/         computed feature tables (parquet/csv for local dev)
src/
  config/           per-use-case YAML label + feature definitions
  extraction/        text feature extraction (Cortex client + LoRA client)
  scoring/           LightGBM training + inference, shared across use cases
  calibration/       Platt scaling / isotonic calibration
  serving/           FastAPI app exposing /score
notebooks/           exploration / synthetic data QA
tests/               unit tests for leakage checks, calibration, scoring
```

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
