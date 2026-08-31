# Code Walkthrough Script

A presenter's script for walking someone — an interviewer, a new teammate —
through this codebase in about 15-20 minutes, ending with a short live demo.
Each stop names the file to have open, what to say, and (where useful) a
command to actually run.

## Before you start

```bash
git clone <repo> && cd pathscore-ai
pip install -r requirements.txt
cd dashboard && npm install && cd ..
```

## Stop 1 — README architecture diagram (2 min)

Open `README.md`. Walk the diagram top to bottom. Say: *"One shared pipeline,
four config files, not four codebases — adding a fifth use case is a YAML
file, not new code. That claim is the thing worth proving, not just stating,
so most of this walkthrough is proving it."*

## Stop 2 — a use case is a config file (2 min)

Open `src/config/gtm_fit.yaml` — the newest of the four. Point at
`entity: account`, `label_column: gtm_fit_label`, `feature_columns`. Say:
*"This is the entire definition of a scoring use case. No Python file
mentions `gtm_fit` by name anywhere in this repo — `app.py`, `train.py`,
`build_features.py` are all generic over whatever's in `src/config/`."*

Optional proof: `grep -rn "gtm_fit" src/*.py src/**/*.py` returns nothing.

## Stop 3 — labels are a documented formula, not a black box (3 min)

Open `src/synthetic_data_gen.py`, jump to `compute_ground_truth_labels`. Point
at the composite score `z` and its weights, then `GTM_FIT_THRESHOLD = 0.65`
with its comment explaining where the number came from (the observed
`icp_fit_score` distribution, not tuned to hit a target). Say: *"If someone
asks 'why did this contact get labeled converted,' there's a real, inspectable
answer — not 'the model said so.'"*

## Stop 4 — the extraction layer's live/mock seam (3 min)

Open `src/extraction/intent_extractor.py`. Show, in order: the
`IntentExtractor`/`CortexClient` interfaces, `MockCortexClient`'s word-list
heuristic, and `_build_cortex_client`'s env-var-driven factory. Say: *"Same
code path whether this is a live Snowflake account or a laptop with zero
credentials — that's what let me build and actually test the real extraction
logic (sentiment thresholds, prompt construction, urgency classification)
without paying for a Snowflake account."* Point out `LoRAExtractor` right
below it — same pattern, applied a second time, for the fallback path.

## Stop 5 — build_features.py, live (2 min)

Open `src/scoring/build_features.py`, show `_select_builder`'s two-way
dispatch. Run it:

```bash
python src/scoring/build_features.py --use-case gtm_fit
```

Say: *"Account-entity use cases are a pass-through; contact-entity use cases
actually call the extractor above, once per event, and average the result
per contact — that's the real call site the mock/live split from the last
stop feeds into."*

## Stop 6 — training + the leakage guard (3 min)

Open `src/scoring/train.py`. Point at `assert_no_leakage(df, config)` running
*before* anything else touches the data, then the calibration step
(`CalibratedClassifierCV` wrapping the already-fitted LightGBM model). Run
it:

```bash
python src/scoring/train.py --config src/config/gtm_fit.yaml --out /tmp/demo_model
```

Say: *"The leakage check isn't a lint rule someone can skip — it's the first
thing that happens inside `train()` itself, so there's no code path that
trains on a leaky feature by accident."*

## Stop 7 — serving + SHAP, live (3 min)

Open `src/serving/app.py`, find `_score_batch` and its comment: *"a single
batched predict_proba + a single batched SHAP call, not one call per row."*
Start the API and hit it:

```bash
uvicorn src.serving.app:app --reload &
curl -s http://localhost:8000/score/gtm_fit/leaderboard?limit=3 | python -m json.tool
```

Say: *"Every score ships with `top_factors` — the actual SHAP values behind
that specific prediction, not a global feature-importance ranking."*

## Stop 8 — the dashboard (2 min)

```bash
cd dashboard && npm run dev
```

Open `localhost:3000`, pick a use case, click a row, point at the SHAP bar
chart. Say: *"Same JSON the curl call above returned — this is just a
renderer for it."*

## Stop 9 — tests, live (2 min)

```bash
pytest tests/ -q                                # 55 tests, every layer
cd dashboard && npm test                         # 16 tests, components + api client
npm run test:e2e                                 # 2 tests, real API + real browser
```

Say: *"Testing pyramid in practice: heavy unit/integration coverage, and
exactly two E2E tests covering the one real user journey — not a duplicate of
everything above it. The E2E layer's job is 'does the real system actually
wire together,' and unit/integration tests can't answer that."*

## Stop 10 (if there's time) — the scheduling + hardening story (3 min)

Open `.github/workflows/retrain.yml` — weekly retrain, daily drift-check,
dormant until pushed to GitHub with Actions enabled. Open
`src/serving/app.py`'s `API_KEY`/`RATE_LIMIT_PER_MINUTE` block — off by
default (zero setup for the Quickstart), on by one environment variable for
anything beyond local dev.

## Closing line

*"The throughline across almost every decision in this repo is the same
shape: state the assumption explicitly, make the demo path require zero
external credentials, and leave a clearly-marked seam for the real thing to
slot in later — Cortex mock/live, LoRA mock/live, `API_KEY` off/on, a CSV
feature store standing in for a real one. See `docs/adr/` for the full list
of where that pattern shows up and what else was considered each time."*
