# Architecture Decision Records

Lightweight ADRs: each records a real decision made in this codebase, the
alternatives actually considered, and why they were rejected — not a
retroactive justification, a record of the trade-off as it was made.

| # | Decision | Status |
|---|---|---|
| [0001](0001-lightgbm-over-neural-net.md) | LightGBM over a neural net for the core scoring model | Accepted |
| [0002](0002-lora-over-full-finetuning.md) | LoRA-adapted frozen base model over full fine-tuning | Accepted |
| [0003](0003-cortex-first-lora-fallback.md) | Snowflake Cortex first, custom LoRA adapter only as fallback | Accepted |
| [0004](0004-config-driven-single-pipeline.md) | One config-driven pipeline over N per-use-case codebases | Accepted |
| [0005](0005-flat-csv-feature-snapshots.md) | Flat CSV feature snapshots over a managed point-in-time feature store | Accepted |
| [0006](0006-env-driven-mock-live-clients.md) | Env-driven mock/live client injection for external dependencies | Accepted |
| [0007](0007-github-actions-cron-scheduling.md) | GitHub Actions cron over an always-on scheduler service | Accepted |
| [0008](0008-api-key-rate-limit-over-full-auth.md) | API key + in-process rate limiting over full auth/session infrastructure | Accepted |
| [0009](0009-vitest-over-jest.md) | Vitest + React Testing Library over Jest for the dashboard | Accepted |
| [0010](0010-single-e2e-golden-path.md) | One Playwright golden-path E2E test, not a broad E2E suite | Accepted |

## Format

Each ADR follows the same shape: **Context** (what forced a decision),
**Decision** (what was chosen), **Alternatives considered** (a table — each
option and why it was rejected, not just the winner), **Consequences** (what
this costs, not only what it buys). See [docs/README.md](../README.md) for how
this fits with the other documents in `docs/`.
