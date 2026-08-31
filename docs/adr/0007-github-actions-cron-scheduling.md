# ADR-0007: GitHub Actions cron over an always-on scheduler service

**Status:** Accepted

## Context

Every use case config declares `retrain.cadence: weekly`, but nothing invoked
`train.py` or `drift_check.py` on any schedule — they were CLI scripts a human
had to remember to run. This repo has no deployed, always-on compute of its own
(see DEPLOYMENT.md — current state is local-only).

## Decision

Add `src/monitoring/run_scheduled_jobs.py` (enumerates every
`src/config/*.yaml` and runs it through `train.py`/`drift_check.py`) plus
`.github/workflows/retrain.yml` as the actual scheduler: weekly retrain, daily
drift-check, triggered by GitHub Actions' `schedule:` cron, with a manual
`workflow_dispatch` escape hatch. Retrained models and the drift dashboard
upload as build artifacts rather than being auto-committed to `main`.

## Alternatives considered

| Option | Why rejected |
|---|---|
| A dedicated always-on scheduler (e.g. a Railway cron service, Airflow) | Requires a deployed, billed, always-on process for a job that runs once a day at most — disproportionate infrastructure for the current single-repo, no-deployment state of this project (see DEPLOYMENT.md). GitHub Actions' scheduled workflows are free for a public repo and need nothing running between triggers. |
| A `cron` entry on someone's own machine | Not reproducible, not visible to anyone else, stops working the moment that machine is off — the opposite of "automated." |
| Auto-commit retrained models back to `main` on every scheduled run | Considered and rejected: an unattended process silently rewriting `main`'s model binaries on a cron schedule is a stronger, less reversible action than "run the job" — it changes what every subsequent clone/deploy gets without a human in the loop. Uploading as a build artifact lets a human review and commit deliberately, the same way any other change to this repo happens. |
| Have `run_scheduled_jobs.py` track "time since last retrain" internally | Would need persistent state (a last-run timestamp store) this script has no reason to own — "when" is cleanly the scheduler's job (the cron expression *is* the cadence), "what to run for every registered use case" is the script's job. Splitting it this way means the script's behavior is identical whether it's triggered by cron, `workflow_dispatch`, or a future different scheduler. |

## Consequences

- The workflow is genuinely dormant until this repo is pushed to GitHub with
  Actions enabled — adding the YAML file itself executes nothing, matching
  every other "no external account required locally" decision in this repo
  (ADR-0006).
- `run_scheduled_jobs.py` continues past one use case's failure and raises at
  the end (`SystemExit` naming every failed use case) — a broken `gtm_fit.yaml`
  doesn't silently prevent `ptb_prospect` from retraining, and the CI run still
  fails visibly.
- Running this for real during development caught a genuine bug: a stale
  `data/features/ptb_prospect.csv` left over from an earlier commit that hadn't
  been rebuilt after `accounts.csv` gained a new column. The scheduler doing
  its job (rebuilding features before every retrain/drift-check) is what
  surfaced it.
- If this project ever gets a real always-on deployment (DEPLOYMENT.md),
  migrating the schedule to that platform's native cron (e.g. Railway) is a
  natural next step — `run_scheduled_jobs.py` itself doesn't change, only what
  invokes it.
