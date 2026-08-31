# ADR-0009: Vitest + React Testing Library over Jest for the dashboard

**Status:** Accepted

## Context

`dashboard/package.json` had no test runner configured at all — `Leaderboard`,
`ShapBarChart`, and `lib/api.ts` were untested. The dashboard runs Next.js
16.3.3 (App Router, Turbopack) on React 19.2.8.

## Decision

Vitest + `@testing-library/react` + `@testing-library/jest-dom`, run via the
`vitest` CLI directly (no `next/jest` wrapper).

## Alternatives considered

| Option | Why rejected |
|---|---|
| Jest (via `next/jest`) | Historically the default Next.js recommendation, but needs a Babel-based transform config (`next/jest`) to handle the App Router/TS/JSX pipeline, and React 19 + Jest has had rougher edges around peer dependency resolution during this transition period. Nothing about this project's testing needs (rendering two presentational components, mocking `fetch`) benefits from Jest's snapshot-testing-era feature set. |
| Vitest with `globals: true` | Considered for convenience (auto-injects `describe`/`it`/`expect`), but explicit imports (`import { describe, it, expect } from "vitest"`) keep every test file's dependencies visible at the top of the file and avoid a global-namespace assumption leaking into `tsconfig.json`'s types. Trade-off made explicit: `@testing-library/react`'s automatic cleanup relies on detecting a global `afterEach`, so going explicit meant wiring `afterEach(cleanup)` by hand in `vitest.setup.ts` — hit and fixed exactly this during implementation (tests were leaking rendered DOM across cases in the same file until that was added). |
| No component tests, only `lib/api.ts` unit tests | Would leave `Leaderboard`'s selection/click behavior and `ShapBarChart`'s width-scaling math (the one genuinely non-trivial piece of logic in either component) completely unverified — exactly the kind of UI logic that silently regresses without a render-and-interact test. |

## Consequences

- `vitest.config.mts` uses the `.mts` extension and `import.meta.dirname` (not
  `__dirname`) specifically to satisfy Vite's native (non-Node-compat) config
  loader without a deprecation warning — a small but deliberate compatibility
  choice for a fast-moving toolchain (Vitest 4, Vite's native config loader).
- `npm test` runs in under a second locally (16 tests, jsdom environment) —
  fast enough to run on every save during development, which is the actual
  point of unit-level component tests.
- This is a different tool from the E2E layer (Playwright, ADR-0010)
  deliberately — Vitest/RTL tests render components in isolation with a mocked
  `fetch`; nothing here spins up a real browser or a real API.
