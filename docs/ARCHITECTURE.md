# Architecture — PathScore AI

Companion to [PRD.md](PRD.md) (what/why), [TDD.md](TDD.md) (implementation
detail), and the [ADRs](adr/README.md) (why each major choice, and what else
was considered). This document is the system view: components, data flow, and
the two request paths that matter most.

## 1. System context

Who and what this system talks to — the dashboard is the only real client;
Snowflake and HuggingFace are optional, credential-gated live dependencies that
degrade to local mocks when unconfigured
([ADR-0006](adr/0006-env-driven-mock-live-clients.md)).

```mermaid
graph TD
    User["RevOps / Sales Ops user"] -->|browses ranked scores| Dashboard["Dashboard (Next.js)"]
    Dashboard -->|"GET /use-cases, /score/*/leaderboard"| API["Serving API (FastAPI)"]
    API -->|loads| Models[("models/*.joblib")]
    API -->|reads| Configs[("src/config/*.yaml")]

    Scheduler["GitHub Actions (cron)"] -->|"retrain (weekly) / drift-check (daily)"| Jobs["run_scheduled_jobs.py"]
    Jobs --> BuildFeatures["build_features.py"]
    Jobs --> Train["train.py"]
    Jobs --> Drift["drift_check.py"]

    BuildFeatures -->|contact entity only| Extraction["Extraction layer"]
    Extraction -.->|live, optional: SNOWFLAKE_* set| Cortex[("Snowflake Cortex")]
    Extraction -.->|live, optional: HF_API_TOKEN set| HF[("HuggingFace Inference Endpoint")]
    BuildFeatures --> SynthData[("data/synthetic/*.csv")]
    BuildFeatures --> Features[("data/features/*.csv")]

    Train --> Models
    Drift --> Models
    Drift --> DriftLog[("monitoring/ (gitignored, uploaded as CI artifact)")]

    style Cortex stroke-dasharray: 5 5
    style HF stroke-dasharray: 5 5
```

Dashed edges are the live paths that require credentials this repo doesn't
ship; solid edges run unconditionally in local dev.

## 2. Component map

```mermaid
graph LR
    subgraph config["src/config/"]
        YAML["*.yaml — one file per use case"]
    end

    subgraph extraction["src/extraction/"]
        IE["intent_extractor.py<br/>CortexExtractor, LoRAExtractor"]
    end

    subgraph scoring["src/scoring/"]
        BF["build_features.py"]
        TR["train.py"]
        LC["leakage_checks.py"]
    end

    subgraph serving["src/serving/"]
        APP["app.py — FastAPI"]
    end

    subgraph monitoring["src/monitoring/"]
        DC["drift_check.py"]
        RSJ["run_scheduled_jobs.py"]
    end

    subgraph dash["dashboard/"]
        PAGE["app/page.tsx"]
        LB["components/Leaderboard.tsx"]
        SHAP["components/ShapBarChart.tsx"]
        APICLIENT["lib/api.ts"]
    end

    BF --> IE
    BF --> config
    TR --> LC
    TR --> config
    APP --> config
    APP --> TR
    RSJ --> BF
    RSJ --> TR
    RSJ --> DC
    DC --> TR

    APICLIENT --> APP
    PAGE --> APICLIENT
    PAGE --> LB
    PAGE --> SHAP
```

`train.py` and `leakage_checks.py` import each other with bare module names
(run as scripts, not a package) — `sys.path` insertion in `drift_check.py`,
`build_features.py`, and `run_scheduled_jobs.py` is what lets them share
`train.py`'s functions without restructuring `src/` into a formal package.

## 3. Data flow: synthetic data to a servable model

```mermaid
flowchart LR
    A["synthetic_data_gen.py<br/>seeded rng"] --> B["data/synthetic/<br/>accounts, contacts, events .csv"]
    B --> C{"build_features.py<br/>entity?"}
    C -->|contact| D["extraction layer<br/>per-event intent_score"]
    C -->|account| E["pass-through"]
    D --> F["data/features/&lt;use_case&gt;.csv"]
    E --> F
    F --> G["train.py<br/>leakage check -&gt; fit -&gt; calibrate"]
    G --> H["models/&lt;use_case&gt;/<br/>model.joblib, calibrator.joblib, metrics.yaml"]
    H --> I["app.py<br/>/score, /leaderboard"]
```

## 4. Sequence: `POST /score/{use_case}`

```mermaid
sequenceDiagram
    participant D as Dashboard
    participant A as app.py
    participant M as Model files

    D->>A: POST score request with feature values
    A->>A: check API key, unless unset
    A->>A: check rate limit
    alt use case unknown, or model not yet trained
        A-->>D: 404
    end
    A->>M: load model and calibrator, cached after first call
    A->>A: build feature frame, 422 if a column is missing
    A->>M: predict probability
    A->>M: compute SHAP values
    A-->>D: 200, score plus predicted class plus top factors
```

## 5. Sequence: scheduled retrain

```mermaid
sequenceDiagram
    participant GH as GitHub Actions
    participant RSJ as run_scheduled_jobs.py
    participant BF as build_features.py
    participant TR as train.py
    participant M as models directory

    GH->>RSJ: retrain, weekly cron trigger
    loop for every config in src/config
        RSJ->>BF: build features for this use case
        BF-->>RSJ: feature snapshot written
        RSJ->>TR: train using config plus features
        TR-->>M: model, calibrator, and metrics saved
    end
    RSJ-->>GH: raises, naming any use case that failed
    GH->>GH: upload models and features as a build artifact
```

Models upload as a build artifact rather than being auto-committed — a human
reviews and commits deliberately, same as any other repo change
([ADR-0007](adr/0007-github-actions-cron-scheduling.md)).

## 6. Deployment view

Current state: **local only**. No component in this diagram is deployed
anywhere; `.github/workflows/retrain.yml` is dormant until this repo is pushed
to GitHub with Actions enabled. See [DEPLOYMENT.md](DEPLOYMENT.md) for the
recommended target topology and why it hasn't been executed.

## Where each ADR fits

| Diagram element | Decision record |
|---|---|
| LightGBM inside `train.py` | [ADR-0001](adr/0001-lightgbm-over-neural-net.md) |
| Extraction layer's live/mock split | [ADR-0002](adr/0002-lora-over-full-finetuning.md), [ADR-0003](adr/0003-cortex-first-lora-fallback.md), [ADR-0006](adr/0006-env-driven-mock-live-clients.md) |
| One `app.py`/`train.py` for every use case | [ADR-0004](adr/0004-config-driven-single-pipeline.md) |
| `data/features/*.csv` | [ADR-0005](adr/0005-flat-csv-feature-snapshots.md) |
| GitHub Actions scheduler | [ADR-0007](adr/0007-github-actions-cron-scheduling.md) |
| `API_KEY`/rate limit in `app.py` | [ADR-0008](adr/0008-api-key-rate-limit-over-full-auth.md) |
| Dashboard test tooling | [ADR-0009](adr/0009-vitest-over-jest.md) |
| E2E scope | [ADR-0010](adr/0010-single-e2e-golden-path.md) |
