from __future__ import annotations

from pathlib import Path

import pytest

from ai_chat_util_base.config.runtime import CodingAgentEndpointSection, init_runtime


def test_coding_agent_endpoint_section_model_validate_accepts_and_strips() -> None:
    cfg = CodingAgentEndpointSection.model_validate({"mcp_server_name": "  my-coding-agent  "})
    assert cfg.mcp_server_name == "my-coding-agent"


def test_coding_agent_endpoint_section_model_validate_rejects_empty() -> None:
    with pytest.raises(Exception):
        CodingAgentEndpointSection.model_validate({"mcp_server_name": "   "})


def test_init_runtime_loads_coding_agent_endpoint_from_integrated_yaml(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "ai_chat_util_config:\n"
        "  llm:\n"
        "    provider: openai\n"
        "    api_key: os.environ/LLM_API_KEY\n"
        "  mcp:\n"
        "    coding_agent_endpoint:\n"
        "      mcp_server_name: my-coding-agent\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))
    monkeypatch.setenv("LLM_API_KEY", "dummy")

    loaded = init_runtime(None)
    assert loaded.mcp.coding_agent_endpoint.mcp_server_name == "my-coding-agent"