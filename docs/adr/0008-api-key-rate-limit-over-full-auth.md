# ADR-0008: API key + in-process rate limiting over full auth/session infrastructure

**Status:** Accepted

## Context

`src/serving/app.py` had no authentication and no abuse protection on
`/score/{use_case}` or `/score/{use_case}/leaderboard` — the two routes that
return real scored business data and cost real compute (model inference + SHAP)
per call. There are no user accounts, no login flow, and no session concept
anywhere in this system — it's a stateless scoring API consumed by one known
client (the dashboard).

## Decision

A single shared `API_KEY` (env var, optional — unset means open/demo mode,
matching every other mock/live decision in this repo, ADR-0006), checked via
constant-time comparison against an `X-API-Key` header on the two data-bearing
routes only (`/health` and `/use-cases` stay open). Paired with an in-process,
per-key/per-IP fixed-window rate limiter (`RATE_LIMIT_PER_MINUTE`, default 60).

## Alternatives considered

| Option | Why rejected |
|---|---|
| Full OAuth2 / session-based auth with user accounts | There are no users to authenticate — no login flow, no roles, nothing that differentiates one caller from another beyond "has the key or doesn't." Building a user/session system for a single-client scoring API is solving a problem this system doesn't have. |
| JWT bearer tokens | Same mismatch: JWTs earn their complexity when you need claims (roles, expiry, revocation) beyond "is this the right shared secret" — nothing here needs that yet, and a JWT issuance flow would need its own auth endpoint this system also doesn't have a reason to own. |
| No auth at all, rely on network-level protection (VPN/firewall) | Leaves the API fully open to anyone who can reach it over the network, and doesn't protect against a legitimate-looking caller hammering the model-inference path — the actual risk being closed here (unlimited-volume scoring calls, which is also a model-extraction vector). |
| A hosted rate-limiting service (Redis-backed, e.g. via `slowapi` + Redis) | Correct choice for a real multi-worker production deployment, explicitly noted as the natural next step in `enforce_rate_limit`'s docstring — but adds a Redis dependency this single-process demo doesn't need yet. The in-memory limiter is honestly scoped to what's true today (one worker), not dressed up as more than it is. |

## Consequences

- Zero changes needed to the existing Quickstart: `API_KEY` unset is the
  default, so `uvicorn src.serving.app:app --reload` behaves exactly as before
  this change for anyone following the README.
- `dashboard/lib/api.ts` reads an optional `NEXT_PUBLIC_API_KEY` and attaches it
  when set — the hardening is usable end-to-end by the one real client, not
  just enforced server-side with no way for the dashboard to satisfy it.
- The rate limiter is explicitly documented as in-process/single-worker only;
  scaling to multiple uvicorn workers or a real deployment would silently give
  each worker its own independent counter (effectively multiplying the limit)
  unless replaced with a shared store first — a known, stated limitation, not
  a hidden one.
- `secrets.compare_digest` for the key comparison closes the timing-attack
  side channel a naive `==` string comparison would leave open.
