import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "monitoring"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "extraction"))

from intent_extractor import REQUIRED_SNOWFLAKE_ENV_VARS  # noqa: E402
import verify_cortex  # noqa: E402


@pytest.fixture(autouse=True)
def clear_snowflake_env(monkeypatch):
    for var in REQUIRED_SNOWFLAKE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_main_exits_loudly_when_snowflake_env_vars_are_unset(capsys):
    # The whole point of this script is confirming the live path -- silently
    # exercising MockCortexClient instead would defeat that, so it must fail
    # rather than fall back.
    with pytest.raises(SystemExit) as exc_info:
        verify_cortex.main()

    assert exc_info.value.code == 1
    assert "SNOWFLAKE_ACCOUNT" in capsys.readouterr().out
