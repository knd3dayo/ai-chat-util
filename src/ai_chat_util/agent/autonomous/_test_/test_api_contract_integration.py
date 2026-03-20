from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi.testclient import TestClient

from ai_chat_util.agent.autonomous._api_.api_server import create_app
from ai_chat_util_base.model.agent_util_models import TaskStatus


@dataclass
class _FakeRunner:
    st: TaskStatus

    def get_task_status(self) -> TaskStatus:
        return self.st


class _FakeTaskService:
    def __init__(self, *, store: dict[str, TaskStatus]):
        self._store = store
        self._runner: Optional[_FakeRunner] = None

    async def prepare(
        self,
        prompt: str,
        sources: Optional[list[Path]],
        task_id: Optional[str],
        workspace_path: Optional[Path] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        tid = task_id or "t-test"
        ws = str(workspace_path or Path("/tmp/ws"))
        st = TaskStatus.create(task_id=tid, workspace_path=ws)
        st.starting_foregrond()
        self._store[tid] = st
        self._runner = _FakeRunner(st=st)

    def start(self, *, wait: bool, timeout: int) -> TaskStatus:
        assert self._runner is not None
        return self._runner.st

    def get_agent_runner(self) -> _FakeRunner:
        assert self._runner is not None
        return self._runner

    def spawn_detached_monitor(self, task_id: str, timeout: int) -> None:
        # no-op for tests
        return

    def cancel_task(self, task: TaskStatus) -> None:
        task.cancelled()
        self._store[task.task_id] = task

    async def monitor(self, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        # converge to completion
        assert self._runner is not None
        st = self._runner.st
        st.completed()
        self._store[st.task_id] = st
        yield st


def test_http_execute_status_cancel_contract(tmp_path: Path, monkeypatch) -> None:
    # Provide a minimal ai-chat-util-config.yml so config resolution succeeds.
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text("ai_chat_util_config: {}\nautonomous_agent_util: {}\n", encoding="utf-8")
    monkeypatch.delenv("AUTONOMOUS_AGENT_UTIL_CONFIG", raising=False)
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    store: dict[str, TaskStatus] = {}
    fake_service = _FakeTaskService(store=store)

    # Patch endpoint dependencies to avoid real backends / filesystem.
    from ai_chat_util.agent.autonomous.core import endpoint as endpoint_mod
    from ai_chat_util.agent.autonomous.core import task_manager as tm_mod

    monkeypatch.setattr(endpoint_mod, "select_task_service", lambda backend=None: fake_service)

    def _upsert(status: TaskStatus) -> None:
        store[status.task_id] = status

    async def _get_status(task_id: str, tail: int | None = 200) -> TaskStatus:
        return store[task_id]

    async def _cancel_task(task_id: str) -> dict[str, Any]:
        st = store[task_id]
        st.cancelled()
        store[task_id] = st
        return {"task_id": task_id, "status": st.status, "sub_status": st.sub_status, "message": "cancelled"}

    monkeypatch.setattr(tm_mod.TaskManager, "upsert_task", classmethod(lambda cls, status: _upsert(status)))
    monkeypatch.setattr(tm_mod.TaskManager, "get_status", classmethod(lambda cls, task_id, tail=200: _get_status(task_id, tail)))
    monkeypatch.setattr(tm_mod.TaskManager, "cancel_task", classmethod(lambda cls, task_id: _cancel_task(task_id)))

    app = create_app(sync_mode=False)
    client = TestClient(app)

    ws = tmp_path / "ws"
    ws.mkdir()

    # execute (async)
    res = client.post(
        "/execute",
        json={"prompt": "hello", "workspace_path": str(ws), "timeout": 10, "task_id": "t1"},
        headers={"X-Trace-Id": "trace-123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["task_id"] == "t1"

    # status
    res2 = client.get("/status/t1")
    assert res2.status_code == 200
    st = res2.json()
    assert st["task_id"] == "t1"
    assert st.get("trace_id") == "trace-123"

    # get_result (stdout/stderr convenience)
    # Populate outputs to verify the endpoint returns them.
    store["t1"].stdout = "hello out"
    store["t1"].stderr = "hello err"
    res_r = client.get("/get_result/t1")
    assert res_r.status_code == 200
    data_r = res_r.json()
    assert data_r.get("stdout") == "hello out"
    assert data_r.get("stderr") == "hello err"

    # cancel
    res3 = client.delete("/cancel/t1")
    assert res3.status_code == 200
    cancel = res3.json()
    assert cancel["task_id"] == "t1"
