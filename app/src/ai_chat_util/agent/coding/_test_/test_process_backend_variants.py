from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ai_chat_util.agent.coding.core.task_manager import TaskManager
from ai_chat_util.common.model.agent_util_models import TaskStatus
from ai_chat_util.common.config.runtime import CodingAgentUtilConfig


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


def test_cancel_task_accepts_process_backend(monkeypatch) -> None:
    task = TaskStatus.create(task_id="t-process", workspace_path="/tmp/ws")
    task.starting_background()
    task.metadata["backend"] = "process"
    task.metadata["pid"] = 12345

    monkeypatch.setattr(TaskManager, "load_tasks", classmethod(lambda cls: None))
    monkeypatch.setattr(TaskManager, "get_task", classmethod(lambda cls, task_id: task))

    updated: dict[str, TaskStatus] = {}
    monkeypatch.setattr(TaskManager, "upsert_task", classmethod(lambda cls, status: updated.setdefault(status.task_id, status)))

    killed: list[int] = []
    monkeypatch.setattr(
        "ai_chat_util.agent.coding.core.task_manager.kill_process_tree",
        lambda pid: killed.append(pid),
    )

    result = asyncio.run(TaskManager.cancel_task("t-process"))

    assert killed == [12345]
    assert result["message"] == "cancelled"
    assert updated["t-process"].sub_status == "cancelled"


def test_get_status_preserves_cancelled_for_process_backend(monkeypatch, tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    exit_code_path = tmp_path / ".exit_code"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("cancelled", encoding="utf-8")
    exit_code_path.write_text("0", encoding="utf-8")

    task = TaskStatus.create(task_id="t-cancelled", workspace_path=str(tmp_path))
    task.cancelled()
    task.metadata.update(
        {
            "backend": "process",
            "pid": 12345,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "exit_code_path": str(exit_code_path),
            "workspace_path": str(tmp_path),
            "cancel_requested": True,
        }
    )

    monkeypatch.setattr(TaskManager, "load_tasks", classmethod(lambda cls: None))
    monkeypatch.setattr(TaskManager, "get_task", classmethod(lambda cls, task_id: task))

    updated: dict[str, TaskStatus] = {}
    monkeypatch.setattr(TaskManager, "upsert_task", classmethod(lambda cls, status: updated.setdefault(status.task_id, status)))

    result = asyncio.run(TaskManager.get_status("t-cancelled", tail=20))

    assert result.sub_status == "cancelled"
    assert updated["t-cancelled"].sub_status == "cancelled"