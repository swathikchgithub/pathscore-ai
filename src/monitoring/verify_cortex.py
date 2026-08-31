"""
One-shot live Cortex connectivity check. Exercises CortexExtractor's real
SnowparkCortexClient against a single short text -- not the full synthetic
dataset. build_features.py calls Cortex once per event (tens of thousands
of live calls at full scale); this is a single call to each of SENTIMENT,
COMPLETE, and CLASSIFY_TEXT, meant for confirming credentials/connectivity
are correctly wired without the cost or runtime of a real extraction pass.

Fails loudly if SNOWFLAKE_* env vars are unset, since the point is
confirming the live path specifically -- silently falling back to
MockCortexClient here would defeat the purpose of running this at all.

Usage:
  python src/monitoring/verify_cortex.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extraction"))
from intent_extractor import REQUIRED_SNOWFLAKE_ENV_VARS, CortexExtractor  # noqa: E402


def main():
    missing = [v for v in REQUIRED_SNOWFLAKE_ENV_VARS if not os.getenv(v)]
    if missing:
        print(
            f"SNOWFLAKE_* env vars not set ({', '.join(missing)}) -- CortexExtractor "
            f"would silently fall back to MockCortexClient, which defeats the point "
            f"of this check. Set them and retry."
        )
        raise SystemExit(1)

    extractor = CortexExtractor()
    signal = extractor.extract(
        "VERIFY-TEST",
        "This is great news, very interested, ready to sign this week, it's urgent!",
    )
    print(
        f"intent_score={signal.intent_score:.3f} sentiment={signal.sentiment} "
        f"urgency_flag={signal.urgency_flag}"
    )
    print("Live Cortex connection verified.")


if __name__ == "__main__":
    main()
