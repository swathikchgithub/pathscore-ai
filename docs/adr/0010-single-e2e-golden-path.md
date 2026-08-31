# ADR-0010: One Playwright golden-path E2E test, not a broad E2E suite

**Status:** Accepted

## Context

By the time this decision was made, every layer had unit or integration
coverage — 55 Python tests, 16 dashboard component tests — but nothing
exercised the real running system: the actual FastAPI process serving the
actual trained models, and the actual Next.js dashboard in a real browser
talking to it. The dashboard has exactly one user journey: pick a use case,
see it ranked, click a row, see why it scored the way it did.

## Decision

Add Playwright with two tests covering that one journey (`__tests__/e2e/dashboard.spec.ts`):
the golden path (pick a use case → leaderboard renders from the live API →
click a row → its SHAP factors render and the row shows selected) and one
adjacent check (switching use cases reloads cleanly with no error state).
`playwright.config.ts` boots two real `webServer` processes — `uvicorn` on
:8010, `next dev` on :3010 — rather than mocking either side.

## Alternatives considered

| Option | Why rejected |
|---|---|
| A broad E2E suite covering every route/component combination | Duplicates what the unit (Vitest) and integration (pytest `TestClient`) layers already cover faster and more precisely — a 404, a 422, a rate-limit response are all better proven with a fake bundle in milliseconds than by driving a real browser through a real server for the same assertion. E2E's value here is specifically "does the real system wire together," not "re-verify every branch." |
| Mock the API in the E2E test (e.g. Playwright route interception) | Would stop being an end-to-end test in any meaningful sense — the whole point is proving the real FastAPI process, the real trained `models/*.joblib`, and the real dashboard build all actually work together, which is exactly the class of bug that unit/integration tests (which mock at the boundary) structurally can't catch. |
| Multiple browsers (Chromium + Firefox + WebKit) | Two focused tests don't need cross-browser coverage to be useful; Chromium alone is the standard single-browser default for CI-speed E2E, and nothing in this dashboard is browser-engine-sensitive (no complex CSS layout, no browser-specific APIs). |
| Skip E2E entirely, call unit + integration sufficient | Leaves exactly the "does it actually run" question unanswered — and it wasn't rhetorical: implementing this caught nothing new in application code, but did prove `playwright.config.ts`'s two-webServer wiring (Next reading `PORT`/`NEXT_PUBLIC_API_BASE_URL` correctly, `uvicorn` starting from the repo root with the right cwd) actually works, which no other test layer could have verified. |

## Consequences

- `npx playwright install chromium` is a one-time additional local setup step
  beyond `npm install`, and `python3` must already have `requirements.txt`
  installed for the API `webServer` to start — both documented in the README's
  new Testing section rather than assumed.
- The two tests together run in under 10 seconds once both servers are warm
  (verified: 2 passed in 5.9s) — cheap enough to run before a merge, matching
  "E2E: few, slow(er), high confidence" rather than becoming the primary
  feedback loop during development (that's still Vitest/pytest, both
  sub-second to low-single-digit-seconds).
- This test is data-dependent on `gtm_fit` and `ptb_prospect` existing as
  configured, trained use cases — if either is ever renamed/removed, the test
  needs a corresponding update, same as any test asserting against real
  fixtures rather than fully synthetic data.
