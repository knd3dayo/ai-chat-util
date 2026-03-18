from __future__ import annotations

import logging

import pytest

from ai_chat_util_base.config.ai_chat_util_runtime import AutonomousAgentUtilConfig
from ai_chat_util_base.config import ai_chat_util_runtime as runtime_mod


def test_logging_redacts_common_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = AutonomousAgentUtilConfig.model_validate({"logging": {"level": "INFO"}})
    runtime_mod._configure_autonomous_python_logging(cfg)

    secret_key = "sk-THIS_SHOULD_NOT_APPEAR_1234567890"

    logging.getLogger("test").info("Authorization: Bearer %s", secret_key)
    logging.getLogger("test").info("api_key='%s'", secret_key)
    logging.getLogger("test").info("api_key: '%s'", secret_key)
    logging.getLogger("test").info("raw=%s", secret_key)

    out = capsys.readouterr().err

    # Original secret must not be present
    assert secret_key not in out

    # Best-effort redactions should appear
    assert "Authorization: Bearer ***" in out
    assert "sk-***" in out


def test_llm_api_key_literal_is_rejected() -> None:
    with pytest.raises(Exception):
        AutonomousAgentUtilConfig.model_validate({"llm": {"api_key": "literal-secret"}})
