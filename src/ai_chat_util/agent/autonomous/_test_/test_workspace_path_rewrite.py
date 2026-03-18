from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi.testclient import TestClient
import yaml

from ai_chat_util.agent.autonomous._api_.api_server import create_app
from ai_chat_util.agent.autonomous.core.endpoint import EndPoint
from ai_chat_util_base.model.autonomous_agent_util_models import TaskStatus
from ai_chat_util_base.config import ai_chat_util_runtime as runtime_mod

endpoint = EndPoint() 

def _reset_runtime(monkeypatch, cfg_path: Path) -> None:
    monkeypatch.setenv("AUTONOMOUS_AGENT_UTIL_CONFIG", str(cfg_path))
    runtime_mod._autonomous_runtime_state = None  # type: ignore[attr-defined]
    runtime_mod.init_autonomous_runtime(None)


def _reset_runtime_via_ai_chat_util_config(monkeypatch, cfg_path: Path) -> None:
    # Integration mode: resolve autonomous settings from ai-chat-util-config.yml.
    monkeypatch.delenv("AUTONOMOUS_AGENT_UTIL_CONFIG", raising=False)
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))
    runtime_mod._autonomous_runtime_state = None  # type: ignore[attr-defined]
    runtime_mod.init_autonomous_runtime(None)


def test_rewrite_workspace_path_pure(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "autonomous-agent-util-config.yml"
    to_prefix = (tmp_path / "executor_workspaces").as_posix()
    cfg_path.write_text(
        """
paths:
  workspace_path_rewrites:
    - from: /srv/ai_platform/workspaces
      to: {to_prefix}
""".format(to_prefix=to_prefix),
        encoding="utf-8",
    )
    _reset_runtime(monkeypatch, cfg_path)

    raw = "/srv/ai_platform/workspaces/e2e_sv_ws_1"
    rewritten = endpoint.rewrite_workspace_path(raw)
    assert rewritten == f"{to_prefix}/e2e_sv_ws_1"

    # Non-matching path should pass through
    raw2 = "/tmp/other"
    assert endpoint.rewrite_workspace_path(raw2) == raw2


def test_rewrite_workspace_path_pure_from_ai_chat_util_config_embedded(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    to_prefix = (tmp_path / "executor_workspaces").as_posix()

    data = {
        "autonomous_agent_util": {
            "paths": {
                "workspace_path_rewrites": [
                    {"from": "/srv/ai_platform/workspaces", "to": to_prefix}
                ]
            }
        }
    }
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    _reset_runtime_via_ai_chat_util_config(monkeypatch, cfg_path)

    raw = "/srv/ai_platform/workspaces/e2e_sv_ws_1"
    rewritten = endpoint.rewrite_workspace_path(raw)
    assert rewritten == f"{to_prefix}/e2e_sv_ws_1"


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
        return

    def cancel_task(self, task: TaskStatus) -> None:
        task.cancelled()
        self._store[task.task_id] = task

    async def monitor(self, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        assert self._runner is not None
        st = self._runner.st
        st.completed()
        self._store[st.task_id] = st
        yield st


def test_http_execute_applies_rewrite_and_persists_metadata(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "autonomous-agent-util-config.yml"
    to_prefix = (tmp_path / "executor_workspaces").as_posix()
    cfg_path.write_text(
        """
paths:
  workspace_path_rewrites:
    - from: /srv/ai_platform/workspaces
      to: {to_prefix}
""".format(to_prefix=to_prefix),
        encoding="utf-8",
    )
    _reset_runtime(monkeypatch, cfg_path)

    store: dict[str, TaskStatus] = {}
    fake_service = _FakeTaskService(store=store)

    from ai_chat_util.agent.autonomous.core import endpoint as endpoint_mod
    from ai_chat_util.agent.autonomous.core import task_manager as tm_mod

    monkeypatch.setattr(endpoint_mod, "select_task_service", lambda backend=None: fake_service)

    def _upsert(status: TaskStatus) -> None:
        store[status.task_id] = status

    async def _get_status(task_id: str, tail: int | None = 200) -> TaskStatus:
        return store[task_id]

    monkeypatch.setattr(tm_mod.TaskManager, "upsert_task", classmethod(lambda cls, status: _upsert(status)))
    monkeypatch.setattr(
        tm_mod.TaskManager, "get_status", classmethod(lambda cls, task_id, tail=200: _get_status(task_id, tail))
    )

    app = create_app(sync_mode=False)
    client = TestClient(app)

    res = client.post(
        "/execute",
        json={
            "prompt": "list files",
            "workspace_path": "/srv/ai_platform/workspaces/e2e_sv_ws_1",
            "timeout": 10,
            "task_id": "t1",
        },
    )
    assert res.status_code == 200

    st = store["t1"]
    assert Path(st.workspace_path).as_posix() == f"{to_prefix}/e2e_sv_ws_1"
    assert st.metadata.get("requested_workspace_path") == "/srv/ai_platform/workspaces/e2e_sv_ws_1"
    assert st.metadata.get("rewritten_workspace_path") == f"{to_prefix}/e2e_sv_ws_1"
    assert st.metadata.get("workspace_path") == f"{to_prefix}/e2e_sv_ws_1"

    # Directory should have been created by validate_workspace_path.
    assert Path(st.workspace_path).is_dir()
