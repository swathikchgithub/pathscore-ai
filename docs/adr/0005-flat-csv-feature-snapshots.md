# ADR-0005: Flat CSV feature snapshots over a managed point-in-time feature store

**Status:** Accepted (revises an earlier, unimplemented design)

## Context

The original architecture diagram in the README described a "Feature Store
(Postgres/Supabase, point-in-time correct)" sitting between extraction and
training — meaning: for any historical label, retrieve the feature values as
they existed *at that label's timestamp*, not their current value, which is
what a real feature store's point-in-time-correct join guarantees and what
prevents a specific class of training-time leakage (a feature that couldn't
have existed yet at label time). That component was never built; `train.py`
originally read `data/synthetic/contacts.csv` directly.

## Decision

Implement `build_features.py` writing one flat CSV snapshot per use case to
`data/features/`, refreshed on demand or on the scheduled cadence
(ADR-0007) — not a versioned, point-in-time-queryable store.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Build the originally-diagrammed Postgres/Supabase point-in-time store | Real infrastructure and schema design (entity + as-of timestamp keyed tables, a query layer that resolves "feature value as of T") disproportionate to a demo whose synthetic data has no time-varying dimension to make point-in-time retrieval meaningful in the first place — there's nothing to be point-in-time *about* yet. |
| Do nothing, leave `train.py` reading raw synthetic CSVs directly | Leaves `CortexExtractor`/`LoRAExtractor` completely unwired — the exact gap this decision closes. Also means every use case re-derives its own feature logic inline instead of sharing one `build_features.py`. |
| A lightweight local file-based store with versioning (e.g. a new snapshot per run, timestamped) | Considered and rejected as *cosmetic*: adding a timestamp column without underlying time-varying source data doesn't deliver real point-in-time correctness, it just makes the CSV look like it does. Building something that looks solved but isn't is worse than documenting the gap honestly. |

## Consequences

- `data/features/<use_case>.csv` is the real, working "feature store" this repo
  ships today — `train.py --data` defaults to it, `run_scheduled_jobs.py`
  rebuilds it, and it's what's actually used for every current model.
- The specific leakage failure mode a point-in-time store exists to prevent
  architecturally is instead caught by `leakage_checks.py`'s heuristics
  (denylisted post-outcome column-name patterns, suspiciously-perfect label
  correlation) — a practical, working substitute, not a complete one: it
  catches leakage that shows up as a *statistical signature*, not leakage from
  a feature that's individually plausible but simply wasn't available yet.
- If this system ever ingests real, time-varying warehouse data, this is the
  component that would need to become a real versioned store — the README's
  architecture diagram and this ADR both flag it explicitly as future work
  rather than silently pretending it's already handled.
