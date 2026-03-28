from __future__ import annotations

import os

import pytest

from ai_chat_util_base.config.runtime import CodingAgentUtilConfig


def test_backend_subprocess_is_normalized_to_process() -> None:
    cfg = CodingAgentUtilConfig.model_validate(
        {"backend": {"task_backend": "subprocess"}, "process": {"command": "agent run"}}
    )
    assert cfg.backend.task_backend == "process"


def test_backend_windows_linux_constraints() -> None:
    if os.name == "nt":
        cfg = CodingAgentUtilConfig.model_validate(
            {"backend": {"task_backend": "windows_process"}, "process": {"command": "agent run"}}
        )
        assert cfg.backend.task_backend == "windows_process"
        with pytest.raises(ValueError):
            CodingAgentUtilConfig.model_validate(
                {"backend": {"task_backend": "linux_process"}, "process": {"command": "agent run"}}
            )
    else:
        cfg = CodingAgentUtilConfig.model_validate(
            {"backend": {"task_backend": "linux_process"}, "process": {"command": "agent run"}}
        )
        assert cfg.backend.task_backend == "linux_process"
        with pytest.raises(ValueError):
            CodingAgentUtilConfig.model_validate(
                {"backend": {"task_backend": "windows_process"}, "process": {"command": "agent run"}}
            )


def test_process_subprocess_command_coalesce() -> None:
    cfg = CodingAgentUtilConfig.model_validate({"subprocess": {"command": "agent run"}})
    assert cfg.process.command == "agent run"
    assert cfg.subprocess.command == "agent run"

    cfg2 = CodingAgentUtilConfig.model_validate({"process": {"command": "agent --version"}})
    assert cfg2.process.command == "agent --version"
    assert cfg2.subprocess.command == "agent --version"

    cfg3 = CodingAgentUtilConfig.model_validate(
        {"process": {"command": "agent run"}, "subprocess": {"command": "agent run"}}
    )
    assert cfg3.process.command == "agent run"

    with pytest.raises(ValueError):
        CodingAgentUtilConfig.model_validate(
            {"process": {"command": "a"}, "subprocess": {"command": "b"}}
        )