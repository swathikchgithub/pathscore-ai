# Deployment Strategy

## Current state: deployed

Both services are live on Railway, project `pathscore-ai`:

- **API**: https://api-production-c80d.up.railway.app
- **Dashboard**: https://dashboard-production-6423.up.railway.app

`.github/workflows/retrain.yml` is live too — this repo is on GitHub with
Actions enabled, so its weekly retrain / daily drift-check cron actually
fires ([ADR-0007](adr/0007-github-actions-cron-scheduling.md)).

The target topology below (Vercel for the dashboard) was the original
recommendation; the actual deployment put both services on Railway instead,
since that was the platform with tooling available at deploy time — Railway
hosts a Next.js app via the same Railpack auto-detection as the API, so
nothing about the app needed to change to make that substitution. The
target-topology section is left as-is below since the reasoning (why
Railway/Fly for the API, the CI/CD shape) still holds; only the dashboard's
host differs from what's written there.

### What it took to get a clean deploy

Two real issues came up, both fixed via Railway config rather than app code:

1. **`libgomp.so.1: cannot open shared object file`** — LightGBM's native
   library needs the GNU OpenMP runtime, absent from Railway's default
   Railpack Python image. Fixed with a service variable:
   `RAILPACK_DEPLOY_APT_PACKAGES=libgomp1`.
2. **Monorepo root-directory double-scoping** — deploying the dashboard via
   `railway up` from within `dashboard/` uploads that directory *as* the
   build root; a service-level `source.rootDirectory: /dashboard` on top of
   that made Railway look for a nonexistent nested `dashboard/dashboard/`.
   Fixed by leaving `rootDirectory` unset for a `railway up`-deployed service
   (it's only needed when Railway builds from a *whole-repo* source, e.g. a
   connected GitHub repo, not a scoped local upload).

Full working configuration, for reproducing this from a fresh Railway
project:

| | `api` service | `dashboard` service |
|---|---|---|
| Deploy source | `railway up` from repo root | `railway up` from `dashboard/` |
| `deploy.startCommand` | `uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT` | auto-detected (`next start`) |
| `source.rootDirectory` | unset | unset |
| Variables | `API_KEY`, `RAILPACK_DEPLOY_APT_PACKAGES=libgomp1`, `CORS_ORIGINS=<dashboard domain>` | `NEXT_PUBLIC_API_BASE_URL=<api domain>`, `NEXT_PUBLIC_API_KEY=<same as API_KEY>` |

`NEXT_PUBLIC_*` vars are baked in at Next.js build time, so the API must get
its public domain (`railway domain --service api`) *before* the dashboard's
first build for the URL to be correct.

## Target topology

```mermaid
graph TD
    subgraph "Vercel"
        Dash["Next.js dashboard"]
    end
    subgraph "Railway or Fly.io"
        API["FastAPI serving container"]
    end
    subgraph "GitHub"
        Actions[".github/workflows/retrain.yml"]
        Repo[("models/, data/features/<br/>committed to git")]
    end

    User --> Dash
    Dash -->|NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_API_KEY| API
    Actions -->|build + drift-check on cadence| Artifacts[("build artifacts")]
    Artifacts -.->|human reviews, commits| Repo
    Repo -->|deploy on merge| API
    Repo -->|deploy on merge| Dash
```

### Dashboard → Vercel

Next.js App Router deploys to Vercel with effectively zero configuration
(it's the reference platform for the framework); free tier is sufficient for
a demo, and PR preview deployments come for free — useful for reviewing a
dashboard change before it merges.

### API → Railway or Fly.io

Both handle a single lightweight, mostly-CPU Python process well and cheaply.
Neither is chosen over the other here; either is a reasonable default. This
needs a `Dockerfile` (none exists yet) — a small `python:3.11-slim` base,
`pip install -r requirements.txt`, `uvicorn src.serving.app:app --host 0.0.0.0
--port $PORT`. Railway specifically has first-class managed cron jobs, which
is the natural place to eventually move the retrain/drift-check schedule if
this ever runs on Railway anyway, rather than keeping it on GitHub Actions —
not required for launch, worth revisiting once (if) that platform is chosen.

### Models → stay in the container image

`models/*.joblib` are small (a few hundred KB to low single-digit MB each
today). Shipping them inside the API's container image is simpler than
external object storage (S3/GCS) and keeps model version = code version =
one deployable artifact. Revisit only if a use case's model grows large
enough that image size or build time becomes a real problem — not the case
today.

### Secrets

`SNOWFLAKE_*`, `HF_API_TOKEN`, `API_KEY`, `RATE_LIMIT_PER_MINUTE`,
`CORS_ORIGINS` as platform-native secret/environment variables on whichever
host runs the API; `NEXT_PUBLIC_API_KEY`/`NEXT_PUBLIC_API_BASE_URL` as
Vercel environment variables for the dashboard. The local `.gitignore`/
`.env.example` split already establishes the discipline this carries
forward: nothing secret is ever committed, `.env.example` documents every
variable with no real values.

## CI/CD

`.github/workflows/retrain.yml` already exists for the retrain/drift cadence.
A deploy pipeline is a natural sibling, not yet added:

1. On every PR: run `pytest tests/`, `npm test`, `npm run build` (dashboard)
   — the same checks this repo already runs locally before every commit in
   this project's history, just automated.
2. On merge to `main`: build and push the API container, trigger a Vercel
   deploy for the dashboard (Vercel does this automatically once the repo is
   connected — no extra workflow step needed for that half).
3. Scheduled retrain results (currently uploaded as a build artifact) could
   gain a manual "promote" step — a human downloads/reviews the artifact,
   commits it, which then flows through step 2 like any other change. This
   keeps model updates behind the same review gate as code, deliberately not
   an automatic model rollout ([ADR-0007](adr/0007-github-actions-cron-scheduling.md)).

## Environments

No staging/prod split exists or is currently needed — a single demo
deployment is sufficient at this scale. If this became a real internal tool,
the natural split is a permanent prod pair (dashboard + API, reading
committed prod models) plus per-PR preview environments — Vercel gives this
for the dashboard automatically; the API would need an equivalently cheap
ephemeral environment (Railway and Fly both support PR/branch environments).

## Rollback

- **Models:** committed to git, so rollback is `git revert` on the commit
  that updated them — no separate model registry needed at this scale.
- **API/dashboard:** standard platform rollback (redeploy the previous
  successful build) on both Vercel and Railway/Fly — no custom tooling
  needed.

## Observability (not yet built)

- **API:** currently only uvicorn's default access logs. A real deployment
  should add structured request logs (use case, status, latency) shipped
  somewhere queryable (even just the hosting platform's own log search is a
  reasonable start).
- **Drift:** `monitoring/dashboard.md` (gitignored locally, uploaded as a CI
  artifact per scheduled run) is the only current monitoring surface. A real
  deployment would want this posted somewhere durable and visible without
  downloading an artifact — a Slack post, a status page, or committing it to
  a docs site are all reasonable next steps, none implemented yet.
- **Alerting:** none exists. The `drifted: true` field `drift_check.py`
  already computes is the natural trigger for one, whenever this is wired to
  a real notification channel.
