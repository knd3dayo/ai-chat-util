from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ai_chat_util.ai_chat_util_agent.coding.core.task_manager import TaskManager
from ai_chat_util.ai_chat_util_agent.coding.core.abstract_actions import AbstractActions
from ai_chat_util.ai_chat_util_agent.coding.core.abstract_task_service import AbstractTaskService
from ai_chat_util.ai_chat_util_agent.coding.core.subprocess import subprocess_entrypoint
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


def test_get_status_preserves_timeout_without_exit_code_for_process_backend(monkeypatch, tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("Task timed out after 3 seconds", encoding="utf-8")

    task = TaskStatus.create(task_id="t-timeout", workspace_path=str(tmp_path))
    task.timeouted(3)
    task.metadata.update(
        {
            "backend": "process",
            "pid": 12345,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "exit_code_path": str(tmp_path / ".exit_code"),
            "workspace_path": str(tmp_path),
        }
    )

    monkeypatch.setattr(TaskManager, "load_tasks", classmethod(lambda cls: None))
    monkeypatch.setattr(TaskManager, "get_task", classmethod(lambda cls, task_id: task))
    monkeypatch.setattr(
        "ai_chat_util.agent.coding.core.task_manager.pid_is_running",
        lambda pid: False,
    )

    updated: dict[str, TaskStatus] = {}
    monkeypatch.setattr(TaskManager, "upsert_task", classmethod(lambda cls, status: updated.setdefault(status.task_id, status)))

    result = asyncio.run(TaskManager.get_status("t-timeout", tail=20))

    assert result.sub_status == "timeout"
    assert updated["t-timeout"].sub_status == "timeout"


class _ImmediateReturnActions(AbstractActions):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed = 0

    def after_start_task_action(self, tid: str) -> None:
        self.started.append(tid)

    def after_start_detach_task_action(self, tid: str) -> None:
        return

    async def progress_action(self, tid: str) -> TaskStatus:
        return TaskStatus.create(task_id=tid, workspace_path="/tmp/ws")

    def after_complete_action(self, runner) -> None:
        self.completed += 1

    def after_task_not_found_action(self) -> None:
        return

    def after_list_action(self, table: list) -> None:
        return

    def after_cancel_action(self, task_id: str, result: dict[str, object] | None = None) -> None:
        return

    def after_get_status_action(self, task_id: str, status_data: TaskStatus) -> None:
        return

    def prune_progress_action(self, generator) -> None:
        return


class _FakeRunner:
    def __init__(self, status: TaskStatus) -> None:
        self._status = status

    def get_task_status(self) -> TaskStatus:
        return self._status


class _DelayedMonitorTaskService(AbstractTaskService):
    def __init__(self) -> None:
        self.status = TaskStatus.create(task_id="t-wait", workspace_path="/tmp/ws")
        self.status.starting_foregrond()
        self._runner = _FakeRunner(self.status)

    async def prepare(self, prompt, sources, task_id, workspace_path=None, extra_env=None) -> None:
        return

    def start(self, *, wait: bool, timeout: int) -> TaskStatus:
        return self.status

    def get_agent_runner(self) -> _FakeRunner:
        return self._runner

    def spawn_detached_monitor(self, task_id: str, timeout: int) -> None:
        return

    def cancel_task(self, task: TaskStatus) -> None:
        task.cancelled()

    async def monitor(self, timeout: int):
        await asyncio.sleep(0.01)
        self.status.completed()
        yield self.status


def test_run_task_wait_awaits_monitor_completion(monkeypatch) -> None:
    service = _DelayedMonitorTaskService()
    actions = _ImmediateReturnActions()

    updated: dict[str, TaskStatus] = {}
    monkeypatch.setattr(TaskManager, "upsert_task", classmethod(lambda cls, status: updated.__setitem__(status.task_id, status)))
    monkeypatch.setattr(
        TaskManager,
        "get_status",
        classmethod(lambda cls, task_id, tail=1000: asyncio.sleep(0, result=updated[task_id])),
    )

    asyncio.run(
        TaskManager.run_task(
            task_service=service,
            actions=actions,
            prompt="hello",
            sources=None,
            task_id="t-wait",
            timeout=1,
            wait=True,
        )
    )

    assert updated["t-wait"].sub_status == "completed"
    assert actions.started == ["t-wait"]
    assert actions.completed == 1


def test_subprocess_entrypoint_keeps_child_in_same_process_group(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    exit_code = tmp_path / ".exit_code"
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    captured: dict[str, object] = {}

    class _FakeProc:
        def wait(self) -> int:
            return 0

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(subprocess_entrypoint.subprocess, "Popen", _fake_popen)

    rc = subprocess_entrypoint.main(
        [
            "--workspace",
            str(workspace),
            "--exit-code-file",
            str(exit_code),
            "--stdout-file",
            str(stdout),
            "--stderr-file",
            str(stderr),
            "--",
            "bash",
            "-lc",
            "echo ok",
        ]
    )

    assert rc == 0
    assert captured["cwd"] == workspace.resolve().as_posix()
    assert "start_new_session" not in captured
    assert "creationflags" not in captured