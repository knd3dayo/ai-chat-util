from __future__ import annotations

import logging

import pytest

from ai_chat_util_base.config.runtime import CodingAgentUtilConfig
from ai_chat_util_base.config.runtime import AiChatUtilConfig
from ai_chat_util_base.config import runtime as runtime_mod


def test_logging_redacts_common_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = CodingAgentUtilConfig.model_validate({"logging": {"level": "INFO"}})
    runtime_mod._configure_python_logging(cfg)

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


def test_logging_redacts_common_secrets_ai_chat_util_config(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = AiChatUtilConfig.model_validate({"llm": {"provider": "dummy"}, "logging": {"level": "INFO"}})
    runtime_mod._configure_python_logging(cfg)

    secret_key = "sk-THIS_SHOULD_NOT_APPEAR_1234567890"

    logging.getLogger("test").info("Authorization: Bearer %s", secret_key)
    logging.getLogger("test").info("api_key='%s'", secret_key)
    logging.getLogger("test").info("api_key: '%s'", secret_key)
    logging.getLogger("test").info("raw=%s", secret_key)

    out = capsys.readouterr().err

    assert secret_key not in out
    assert "Authorization: Bearer ***" in out
    assert "sk-***" in out


def test_llm_api_key_literal_is_rejected() -> None:
    with pytest.raises(Exception):
        CodingAgentUtilConfig.model_validate({"llm": {"api_key": "literal-secret"}})
