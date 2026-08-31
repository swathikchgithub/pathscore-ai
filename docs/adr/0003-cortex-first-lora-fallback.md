# ADR-0003: Snowflake Cortex first, custom LoRA adapter only as fallback

**Status:** Accepted

## Context

Every contact-entity use case needs a numeric intent signal (`avg_intent_score`)
derived from unstructured engagement text. Two implementations exist:
`CortexExtractor` (Snowflake's built-in SENTIMENT/COMPLETE/CLASSIFY_TEXT
functions) and `LoRAExtractor` (a custom fine-tuned adapter, ADR-0002). Both
implement the same `IntentExtractor` interface (`extract(contact_id, text) ->
IntentSignal`), selected per use case by `get_extractor()`.

## Decision

Default every use case to Cortex (`extraction: {backend: cortex}` is the
implicit default when a config omits the key entirely — see `contact_score.yaml`
and `funnel_stage.yaml`, neither of which sets `extraction` at all). Only reach
for a custom LoRA adapter when a specific use case needs a signal Cortex's
generic functions genuinely can't produce.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Always use a custom LoRA adapter | Every use case would need its own labeled training set before it could ship at all — directly opposed to "one shared pipeline, N config files, not N codebases." Also means running/maintaining a GPU inference endpoint for every use case instead of zero. |
| Always use Cortex, never build LoRA | Leaves no answer for a genuinely company-specific signal (e.g. "contract urgency" phrasing particular to this business) that a generic sentiment/classification function can't capture — the whole reason `LoRAExtractor` exists. |
| Call an external LLM API (OpenAI/Anthropic) directly | Data egress: customer email/call text would leave the warehouse boundary. Cortex runs natively where Snowflake-resident data already lives — no separate data pipeline, no additional vendor in the trust boundary for the default path. |

## Consequences

- Zero GPU cost, zero data egress, and zero adapter-training data requirement for
  every use case that ships today — all three configured use cases needing text
  extraction (`contact_score`, `funnel_stage`) use the Cortex path.
- The decision of *which* backend a use case uses is a one-line config change
  (`extraction.backend: cortex|lora`), not a code branch — `get_extractor()` is
  the only place that reads it.
- `CortexExtractor` degrades gracefully to `MockCortexClient` with no Snowflake
  account configured (ADR-0006), so this default costs nothing to exercise
  locally either.
