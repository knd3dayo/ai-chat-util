from __future__ import annotations

from ai_chat_util.base.agent.deep_agent_support import build_deep_agent_system_prompt


def test_build_deep_agent_system_prompt_mentions_working_directory_and_directory_handling() -> None:
    prompt = build_deep_agent_system_prompt("/tmp/workspace")

    assert "/tmp/workspace" in prompt
    assert "working_directory" in prompt
    assert "directory パス" in prompt