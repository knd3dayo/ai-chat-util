from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ai_chat_util.common.config.runtime import CodingAgentEndpointSection, init_runtime
from ai_chat_util.ai_chat_util_agent.coding._cli_.docker_main import app as coding_cli_app
from ai_chat_util.ai_chat_util_agent.coding.core.task_manager import TaskManager


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


def test_cli_run_uses_selected_task_service(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "ai_chat_util_config: {}\n"
        "coding_agent_util:\n"
        "  backend:\n"
        "    task_backend: process\n"
        "  process:\n"
        "    command: opencode run\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    selected_service = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "ai_chat_util.agent.coding._cli_.docker_main.select_task_service",
        lambda: selected_service,
    )

    async def _fake_run_task(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(TaskManager, "run_task", classmethod(lambda cls, **kwargs: _fake_run_task(**kwargs)))

    runner = CliRunner()
    result = runner.invoke(coding_cli_app, ["--config", str(cfg_path), "run", "hello", "--no-wait"])

    assert result.exit_code == 0
    assert captured["task_service"] is selected_service
    assert captured["prompt"] == "hello"
    assert captured["wait"] is False


def test_cli_prune_rejects_non_docker_backend(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "ai_chat_util_config: {}\n"
        "coding_agent_util:\n"
        "  backend:\n"
        "    task_backend: process\n"
        "  process:\n"
        "    command: opencode run\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    monkeypatch.setattr(
        "ai_chat_util.agent.coding._cli_.docker_main.select_task_service",
        lambda: SimpleNamespace(),
    )

    runner = CliRunner()
    result = runner.invoke(coding_cli_app, ["--config", str(cfg_path), "prune", "executor-service"])

    assert result.exit_code != 0
    assert "docker backend" in result.output