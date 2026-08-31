import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "extraction"))

from intent_extractor import (  # noqa: E402
    REQUIRED_SNOWFLAKE_ENV_VARS,
    CortexExtractor,
    IntentSignal,
    MockCortexClient,
    _build_cortex_client,
    _parse_intent_score,
    get_extractor,
)


@pytest.fixture(autouse=True)
def clear_snowflake_env(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS + ["SNOWFLAKE_PASSWORD", "SNOWFLAKE_PRIVATE_KEY_PATH"]:
        monkeypatch.delenv(var, raising=False)


def test_mock_client_used_when_snowflake_env_vars_absent():
    client = _build_cortex_client()
    assert isinstance(client, MockCortexClient)


def test_live_client_requires_password_or_key(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS:
        monkeypatch.setenv(var, "placeholder")
    with pytest.raises(RuntimeError, match="SNOWFLAKE_PASSWORD"):
        _build_cortex_client()


def test_cortex_extractor_extract_returns_intent_signal_via_mock():
    extractor = CortexExtractor(cortex_client=MockCortexClient())
    signal = extractor.extract(
        "CON-0000001", "We are very interested and ready to sign this week, it's urgent."
    )
    assert isinstance(signal, IntentSignal)
    assert signal.contact_id == "CON-0000001"
    assert 0.0 <= signal.intent_score <= 1.0
    assert signal.sentiment in {"positive", "neutral", "negative"}
    assert signal.urgency_flag is True


def test_cortex_extractor_flags_negative_email_as_not_urgent():
    extractor = CortexExtractor(cortex_client=MockCortexClient())
    signal = extractor.extract("CON-0000002", "Not interested, please unsubscribe.")
    assert signal.sentiment == "negative"
    assert signal.urgency_flag is False


def test_parse_intent_score_handles_unparseable_output():
    assert _parse_intent_score("I cannot determine a score") == 0.5


def test_parse_intent_score_clamps_out_of_range_values():
    assert _parse_intent_score("1.7") == 1.0
    assert _parse_intent_score("-0.3") == 0.0


def test_get_extractor_returns_cortex_extractor_by_default():
    extractor = get_extractor({})
    assert isinstance(extractor, CortexExtractor)
