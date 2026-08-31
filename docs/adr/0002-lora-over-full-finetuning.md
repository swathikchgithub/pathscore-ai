# ADR-0002: LoRA-adapted frozen base model over full fine-tuning for text extraction

**Status:** Accepted

## Context

`LoRAExtractor` (`src/extraction/intent_extractor.py`) exists as the fallback path
for a text-derived signal that Snowflake Cortex's generic functions (SENTIMENT,
COMPLETE, CLASSIFY_TEXT) can't capture — e.g. a company-specific "renewal risk" or
"contract urgency" phrasing pattern. It needs to adapt an LLM to a narrow,
domain-specific classification/scoring task on a few thousand labeled examples.

## Decision

If a custom model is needed at all, fine-tune a small LoRA adapter (~0.1-1% of
parameters) on top of a frozen 7-8B base model, rather than fully fine-tuning the
base model.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Full fine-tuning | Updates every parameter of a 7-8B model — needs far more labeled data than "a few thousand examples" to avoid overfitting/catastrophic forgetting, and needs multi-GPU or a large single GPU to fine-tune in reasonable time (see ADR: no H100s, below). |
| Prompt engineering only (no fine-tuning) | This is what `CortexExtractor`'s `COMPLETE` call already does — it's the *first* fallback, tried before LoRA. LoRA only gets reached for signals prompt engineering against a generic base model genuinely can't capture reliably; if prompting sufficed, Cortex would already handle it (ADR-0003) and a trained adapter would be unjustified complexity. |
| A separate small model trained from scratch per signal | No transfer learning from the base model's language understanding; needs much more labeled data per signal than a few thousand examples to reach usable accuracy. |

## Consequences

- One frozen base model can be shared across every LoRA adapter/use case that
  ever needs one — adapters are cheap to store and swap (`adapter_name` in a use
  case's `extraction` config), unlike N separately fine-tuned full models.
- Fine-tuning a LoRA adapter on a few thousand examples runs on a single rented
  A10/A6000 — no multi-GPU cluster needed, which keeps the cost proportionate to
  a narrow, single-signal task.
- No use case in this repo currently sets `extraction.backend: lora` — none of
  the three demo use cases needed a signal Cortex's generic functions couldn't
  already produce, which is exactly the intended state: LoRA is the fallback,
  not the default. See ADR-0003.
- `LoRAExtractor` is implemented against an injectable `LoRAClient` interface
  (live `HFInferenceEndpointClient` vs. `MockLoRAClient`) rather than an actually
  trained adapter, since training one requires labeled data this demo doesn't
  have. See ADR-0006 for why that's the right boundary to mock at.
