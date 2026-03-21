from __future__ import annotations

import os

import pytest

from ai_chat_util_base.config.runtime import AutonomousAgentUtilConfig


def test_backend_subprocess_is_normalized_to_process() -> None:
    cfg = AutonomousAgentUtilConfig.model_validate(
        {"backend": {"task_backend": "subprocess"}, "process": {"command": "agent run"}}
    )
    assert cfg.backend.task_backend == "process"


def test_backend_windows_linux_constraints() -> None:
    if os.name == "nt":
        cfg = AutonomousAgentUtilConfig.model_validate(
            {"backend": {"task_backend": "windows_process"}, "process": {"command": "agent run"}}
        )
        assert cfg.backend.task_backend == "windows_process"
        with pytest.raises(ValueError):
            AutonomousAgentUtilConfig.model_validate(
                {"backend": {"task_backend": "linux_process"}, "process": {"command": "agent run"}}
            )
    else:
        cfg = AutonomousAgentUtilConfig.model_validate(
            {"backend": {"task_backend": "linux_process"}, "process": {"command": "agent run"}}
        )
        assert cfg.backend.task_backend == "linux_process"
        with pytest.raises(ValueError):
            AutonomousAgentUtilConfig.model_validate(
                {"backend": {"task_backend": "windows_process"}, "process": {"command": "agent run"}}
            )


def test_process_subprocess_command_coalesce() -> None:
    # Old config style: subprocess.command only.
    cfg = AutonomousAgentUtilConfig.model_validate({"subprocess": {"command": "agent run"}})
    assert cfg.process.command == "agent run"
    assert cfg.subprocess.command == "agent run"

    # New config style: process.command only.
    cfg2 = AutonomousAgentUtilConfig.model_validate({"process": {"command": "agent --version"}})
    assert cfg2.process.command == "agent --version"
    assert cfg2.subprocess.command == "agent --version"

    # Both present and consistent.
    cfg3 = AutonomousAgentUtilConfig.model_validate(
        {"process": {"command": "agent run"}, "subprocess": {"command": "agent run"}}
    )
    assert cfg3.process.command == "agent run"

    # Both present but inconsistent should fail.
    with pytest.raises(ValueError):
        AutonomousAgentUtilConfig.model_validate(
            {"process": {"command": "a"}, "subprocess": {"command": "b"}}
        )
