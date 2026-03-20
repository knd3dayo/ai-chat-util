from __future__ import annotations

from pathlib import Path

import pytest

from ai_chat_util_base.config.runtime import AutonomousAgentUtilConfig, init_autonomous_runtime


def test_endpoint_section_model_validate_accepts_and_strips() -> None:
    cfg = AutonomousAgentUtilConfig.model_validate({"endpoint": {"mcp_server_name": "  my-coding-agent  "}})
    assert cfg.endpoint.mcp_server_name == "my-coding-agent"


def test_endpoint_section_model_validate_rejects_empty() -> None:
    with pytest.raises(Exception):
        AutonomousAgentUtilConfig.model_validate({"endpoint": {"mcp_server_name": "   "}})


def test_init_autonomous_runtime_loads_endpoint_from_integrated_yaml(tmp_path: Path, monkeypatch) -> None:
    # Minimal integrated ai-chat-util-config.yml containing autonomous_agent_util only.
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "ai_chat_util_config: {}\n"
        "autonomous_agent_util:\n"
        "  endpoint:\n"
        "    mcp_server_name: my-coding-agent\n",
        encoding="utf-8",
    )

    # Ensure runtime resolves this config.
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))
    monkeypatch.delenv("AUTONOMOUS_AGENT_UTIL_CONFIG", raising=False)

    loaded = init_autonomous_runtime(None)
    assert loaded.endpoint.mcp_server_name == "my-coding-agent"
