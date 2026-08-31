# ADR-0006: Env-driven mock/live client injection for external dependencies

**Status:** Accepted

## Context

Three integrations in this repo need live credentials this demo doesn't have:
Snowflake Cortex (`CortexExtractor`), a HuggingFace Inference Endpoint
(`LoRAExtractor`), and — implicitly — any real point-in-time data source
(ADR-0005). The README's Quickstart promises the whole pipeline runs from a
fresh clone with `pip install -r requirements.txt` and no external account.

## Decision

For each live dependency, define a small client interface (`CortexClient`,
`LoRAClient`) with two implementations — a live one (`SnowparkCortexClient`,
`HFInferenceEndpointClient`) and a local mock (`MockCortexClient`,
`MockLoRAClient`) — and a factory (`_build_cortex_client`, `_build_lora_client`)
that picks live vs. mock based purely on whether the relevant env vars
(`SNOWFLAKE_*`, `HF_API_TOKEN`) are set. `CortexExtractor`/`LoRAExtractor`'s own
code never branches on live-vs-mock; they only ever call the injected client.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Require real credentials to run anything | Breaks the "no external account required for local dev/demo" Quickstart promise outright; makes the extraction layer, and everything downstream of it, untestable without paying for Snowflake/HF access. |
| A `--mock` CLI flag instead of env-var detection | Extra flag to remember and pass through every entrypoint (`build_features.py`, tests, `uvicorn`); env-var detection instead means the *same* command (`python src/scoring/build_features.py --use-case contact_score`) behaves correctly in both a laptop and a fully-configured production environment with zero flag changes. |
| Mock at the `IntentExtractor` level (fake the whole extractor) instead of the client level | Would leave `CortexExtractor`'s own logic (sentiment-to-label thresholds, prompt construction, urgency classification) completely unexercised by any test run without real credentials — mocking one level lower (the client) means the *real* extractor logic runs in every test and every local demo, only the actual network call is faked. |

## Consequences

- The identical pattern applies twice (Cortex, LoRA) with no shared abstraction
  forcing them together — each was independently the right size for its own
  interface (`sentiment`/`complete`/`classify_text` vs. a single `predict`),
  and forcing a shared base class across two two-method and one-method
  interfaces would have been abstraction for its own sake.
- `MockCortexClient`'s heuristic (word-list matching) is deliberately
  documented as "not a substitute for validating against the real Cortex
  functions before shipping" in its own docstring — the mock exists to
  exercise the pipeline, not to claim equivalence with the real thing.
- `MockLoRAClient` explicitly reuses `MockCortexClient`'s heuristic rather than
  faking adapter-specific behavior, because there's no real trained adapter to
  approximate in the first place (ADR-0002) — faking specificity that doesn't
  exist would be more misleading than reusing the honest, general-purpose mock.
- This is the same principle applied a third time in `src/serving/app.py`:
  `API_KEY` unset → open/demo mode, set → enforced auth (ADR-0008) — one
  consistent shape across the whole repo for "off by default locally, on by
  configuration."
