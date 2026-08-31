import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "extraction"))

from intent_extractor import (  # noqa: E402
    REQUIRED_SNOWFLAKE_ENV_VARS,
    CortexExtractor,
    HFInferenceEndpointClient,
    IntentSignal,
    LoRAExtractor,
    MockCortexClient,
    MockLoRAClient,
    SnowparkCortexClient,
    _build_connection_params,
    _build_cortex_client,
    _build_lora_client,
    _parse_intent_score,
    get_extractor,
)

_SNOWFLAKE_SECRET_VARS = ["SNOWFLAKE_PASSWORD", "SNOWFLAKE_PRIVATE_KEY", "SNOWFLAKE_PRIVATE_KEY_PATH"]


@pytest.fixture(autouse=True)
def clear_snowflake_env(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS + _SNOWFLAKE_SECRET_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def clear_hf_env(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)


def test_mock_client_used_when_snowflake_env_vars_absent():
    client = _build_cortex_client()
    assert isinstance(client, MockCortexClient)


def test_live_client_requires_password_or_key(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS:
        monkeypatch.setenv(var, "placeholder")
    with pytest.raises(RuntimeError, match="SNOWFLAKE_PRIVATE_KEY, SNOWFLAKE_PRIVATE_KEY_PATH, or SNOWFLAKE_PASSWORD"):
        _build_cortex_client()


def _set_required_snowflake_env(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS:
        monkeypatch.setenv(var, "placeholder")


def test_connection_params_use_password_when_only_password_is_set(monkeypatch):
    _set_required_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")

    params = _build_connection_params()

    assert params["password"] == "hunter2"
    assert "private_key" not in params
    assert "private_key_file" not in params


def test_connection_params_use_private_key_file_when_set(monkeypatch):
    _set_required_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_PATH", "/run/secrets/snowflake_key.p8")

    params = _build_connection_params()

    assert params["private_key_file"] == "/run/secrets/snowflake_key.p8"
    assert "password" not in params


def test_connection_params_prefer_private_key_content_over_password(monkeypatch):
    # Password auth on this project's Snowflake account hit a platform-wide
    # MFA-for-password requirement a headless CI job can't satisfy; key-pair
    # is exempt, so it must win whenever both are configured.
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    _set_required_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY", pem)
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")  # present, but must be ignored

    params = _build_connection_params()

    expected_der = key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    assert params["private_key"] == expected_der
    assert "password" not in params


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


def test_get_extractor_returns_lora_extractor_when_configured():
    extractor = get_extractor(
        {"extraction": {"backend": "lora", "hf_endpoint_url": "https://example/endpoint", "adapter_name": "renewal-risk"}}
    )
    assert isinstance(extractor, LoRAExtractor)
    assert extractor.adapter_name == "renewal-risk"


def test_mock_lora_client_used_when_hf_token_absent():
    client = _build_lora_client("https://example/endpoint")
    assert isinstance(client, MockLoRAClient)


def test_live_lora_client_used_when_hf_token_set(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "placeholder")
    client = _build_lora_client("https://example/endpoint")
    assert isinstance(client, HFInferenceEndpointClient)
    assert client.endpoint_url == "https://example/endpoint"
    assert client.api_token == "placeholder"


def test_mock_lora_client_predict_returns_structured_response():
    result = MockLoRAClient().predict("renewal-risk", "Very interested, ready to sign this week, urgent.")
    assert 0.0 <= result["intent_score"] <= 1.0
    assert result["sentiment"] == "positive"
    assert result["urgency_flag"] is True


def test_lora_extractor_extract_returns_intent_signal_via_mock():
    extractor = LoRAExtractor(
        hf_endpoint_url="https://example/endpoint",
        adapter_name="renewal-risk",
        lora_client=MockLoRAClient(),
    )
    signal = extractor.extract("CON-0000003", "Not interested, no budget, please stop.")
    assert isinstance(signal, IntentSignal)
    assert signal.contact_id == "CON-0000003"
    assert signal.sentiment == "negative"
    assert 0.0 <= signal.intent_score <= 1.0


def test_lora_extractor_defaults_and_clamps_a_malformed_client_response():
    class _MalformedClient:
        def predict(self, adapter_name, text):
            return {"intent_score": 5.0}  # missing sentiment/urgency_flag, out-of-range score

    extractor = LoRAExtractor(
        hf_endpoint_url="https://example/endpoint",
        adapter_name="renewal-risk",
        lora_client=_MalformedClient(),
    )
    signal = extractor.extract("CON-0000004", "irrelevant")
    assert signal.intent_score == 1.0
    assert signal.sentiment == "neutral"
    assert signal.urgency_flag is False


class _FakeSnowparkSession:
    """Records the last SQL text + bind params passed to .sql(), standing
    in for a real Snowpark session -- no live account or the optional
    snowpark dependency needed to test bind-parameter shape."""

    def __init__(self, row: dict):
        self._row = row
        self.query = None
        self.params = None

    def sql(self, query, params=None):
        self.query = query
        self.params = params
        return self

    def collect(self):
        return [self._row]


def test_snowpark_client_classify_text_binds_categories_as_individual_scalars():
    # A raw Python list passed as one bind value is read by the Snowflake
    # connector as an executemany batch-size signal, which collides with
    # `text`'s scalar binding ("batch size of 1 ... not the same as
    # previous size of N") -- verified against the live account. Each
    # category must be its own bind param, reassembled via ARRAY_CONSTRUCT.
    session = _FakeSnowparkSession({"LABEL": "urgent"})
    client = SnowparkCortexClient(session)

    result = client.classify_text("this is urgent", ["urgent", "not urgent"])

    assert result == "urgent"
    assert session.params == ["this is urgent", "urgent", "not urgent"]
    assert "ARRAY_CONSTRUCT(?, ?)" in session.query


def test_snowpark_client_sentiment_and_complete_bind_scalars_directly():
    session = _FakeSnowparkSession({"SCORE": 0.5, "RESPONSE": "0.5"})
    client = SnowparkCortexClient(session)

    assert client.sentiment("hello") == 0.5
    assert session.params == ["hello"]

    assert client.complete("llama3.1-8b", "rate this") == "0.5"
    assert session.params == ["llama3.1-8b", "rate this"]
