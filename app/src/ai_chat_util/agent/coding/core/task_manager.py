from __future__ import annotations

from typing import Optional, ClassVar, Generator, Callable, AsyncGenerator, Any, TYPE_CHECKING
import pathlib
import json
import pathlib
import asyncio
import json
import signal

from pathlib import Path
from fastapi import HTTPException
from python_on_whales import docker as whales
from pydantic import ValidationError
from ai_chat_util_base.model.agent_util_models import TaskStatus

from ..core.process_utils import kill_process_tree, pid_is_running

from ..core.abstract_actions import AbstractActions
from ..core.abstract_task_service import AbstractTaskService

from ai_chat_util_base.config.runtime import get_coding_runtime_config

from ..util.logging import get_application_logger


logger = get_application_logger()


def _get_host_projects_root() -> pathlib.Path:
    cfg = get_coding_runtime_config()
    return pathlib.Path(cfg.paths.host_projects_root)

class TaskManager:
    """タスク管理クラス（シングルトン）"""
    tasks: ClassVar[dict[str, TaskStatus]] = {}
    
    @classmethod
    def get_projects_root(cls) -> pathlib.Path:
        return _get_host_projects_root()
    
    @classmethod
    def get_tasks_file_path(cls) -> pathlib.Path:
        return cls.get_projects_root() / "tasks_db.json"
    
    @classmethod
    def get_task(cls, task_id: str) -> Optional[TaskStatus]:
        return cls.tasks.get(task_id)
    
    @classmethod
    def upsert_task(cls, status: TaskStatus):
        cls.tasks[status.task_id] = status
        cls.save_tasks()

    @classmethod
    def get_all_tasks(cls) -> dict[str, TaskStatus]:
        return cls.tasks
    
    @classmethod
    def remove_task(cls, task_id: str):
        if task_id in cls.tasks:
            del cls.tasks[task_id]
        cls.save_tasks()

    @classmethod
    def save_tasks(cls):
        """現在のタスク状態をファイルに保存する"""
        TaskManager.get_tasks_file_path().parent.mkdir(parents=True, exist_ok=True)
        with open(TaskManager.get_tasks_file_path(), "w") as f:
            data = {k: v.model_dump(mode='json') for k, v in TaskManager.get_all_tasks().items()}
            json.dump(data, f, indent=2)

    @classmethod
    def load_tasks(cls):
        """ファイルからタスク状態を復元する"""
        if TaskManager.get_tasks_file_path().exists():
            with open(TaskManager.get_tasks_file_path(), "r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning(
                        "tasks_db.json format is invalid (expected object/dict): %s",
                        type(data),
                    )
                    return

                for k, v in data.items():
                    # 旧フォーマット互換（任意）:
                    # 以前は trace_id を TaskStatus.metadata["trace_id"] に入れていた。
                    # DBを作り直さない運用でも壊れないよう、読み込み時だけ第一級フィールドへ移植する。
                    if isinstance(v, dict) and not v.get("trace_id"):
                        md = v.get("metadata")
                        if isinstance(md, dict) and isinstance(md.get("trace_id"), str) and md.get("trace_id"):
                            v = {**v, "trace_id": md.get("trace_id")}

                    if not isinstance(v, dict):
                        logger.warning(
                            "Skipping invalid task record (expected dict) for key=%s: %s",
                            k,
                            type(v),
                        )
                        continue

                    try:
                        st = TaskStatus(**v)
                    except ValidationError as e:
                        # TaskStatus has required fields (task_id/workspace_path).
                        # If an old/corrupt tasks_db.json is present, do not crash the whole app.
                        logger.warning(
                            "Skipping invalid TaskStatus record key=%s: %s",
                            k,
                            e,
                        )
                        continue

                    cls.tasks[st.task_id] = st


    @classmethod
    def list_tasks(cls) -> list[TaskStatus]:
        """タスクの一覧表示"""
        TaskManager.load_tasks()
        tasks = TaskManager.get_all_tasks()
        if not tasks:
            # self.actions.after_task_not_found_action()
            return []
        
        return list(tasks.values())

    @classmethod
    async def show_status(cls, task_id: str, tail: int) -> TaskStatus:
        """ステータスとログの表示（async版に統一）"""
        data = await TaskManager.get_status(task_id, tail=tail)
        return data

    # --- ステータス取得の修正 ---
    @classmethod
    async def get_status(cls, task_id: str, tail: int | None = 200) -> TaskStatus:
        # CLI はコマンドごとに別プロセスで起動されるため、永続ストアから都度復元する
        TaskManager.load_tasks()
        task = TaskManager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # デタッチ実行などで monitor が走らない場合でも、status 取得時に状態を再評価して追従する。
        if task.container_id:
            try:
                container = whales.container.inspect(task.container_id)
                # state を最新化
                container.reload()

                # logs の取得（running/exited いずれも取得可能）
                if tail is None:
                    logs = whales.container.logs(task.container_id)
                else:
                    logs = whales.container.logs(task.container_id, tail=tail)

                if isinstance(logs, str):
                    logs_str = logs
                else:
                    logs_str = "".join(
                        b.decode("utf-8", errors="replace") if isinstance(b, bytes) else str(b)
                        for _, b in logs
                    )

                status = getattr(getattr(container, "state", None), "status", None)
                exit_code = getattr(getattr(container, "state", None), "exit_code", None)

                if status == "exited":
                    # 監視プロセスがいなくても、ここで最終状態へ寄せる
                    if exit_code == 0:
                        task.completed()
                    else:
                        task.failed()
                    task.stdout = logs_str

                    # workspace_path が分かる場合は、成果物一覧をオンデマンドで再計算する
                    try:
                        ws = None
                        if isinstance(task.metadata, dict):
                            ws = task.metadata.get("workspace_path")
                        if isinstance(ws, str) and ws:
                            base = pathlib.Path(ws)
                            if base.exists() and base.is_dir():
                                task.artifacts = [
                                    str(p.relative_to(base).as_posix())
                                    for p in base.rglob("*")
                                    if p.is_file()
                                ]
                    except Exception:
                        # 成果物一覧の計算失敗で status を落とさない
                        pass

                    TaskManager.upsert_task(task)
                    return task

                # running の場合は増分ログを返す（保存はしない）
                return TaskStatus(**task.model_dump(exclude={"stdout"}), stdout=logs_str)
            except Exception as e:
                # コンテナが既に削除されている等。
                # 既に stdout が保存済みならそれを返し、無い場合のみエラーを埋める。
                if task.stdout is not None or task.status == "exited":
                    return task
                return TaskStatus(**task.model_dump(exclude={"stderr"}), stderr=f"Log/state fetch failed: {e}")

        # --- subprocess backend ---
        md = task.metadata if isinstance(task.metadata, dict) else {}
        if md.get("backend") == "subprocess" or ("pid" in md and ("stdout_path" in md or "exit_code_path" in md)):
            pid = md.get("pid")
            stdout_path = md.get("stdout_path")
            stderr_path = md.get("stderr_path")
            exit_code_path = md.get("exit_code_path")

            def _pid_running(p: int) -> bool:
                return pid_is_running(p)

            def _tail_text(path_str: str | None, tail_lines: int | None) -> str:
                if not path_str:
                    return ""
                p = pathlib.Path(path_str)
                if not p.exists() or not p.is_file():
                    return ""
                if tail_lines is None:
                    return p.read_text(encoding="utf-8", errors="replace")

                # Read from end with a bounded buffer.
                # This is a simple implementation sufficient for typical log sizes.
                data = p.read_text(encoding="utf-8", errors="replace")
                lines = data.splitlines()
                return "\n".join(lines[-tail_lines:])

            running = isinstance(pid, int) and pid > 1 and _pid_running(pid)

            # If exit code is present, treat as exited and persist final status.
            if exit_code_path and pathlib.Path(exit_code_path).exists():
                try:
                    rc = int(pathlib.Path(exit_code_path).read_text(encoding="utf-8").strip())
                except Exception:
                    rc = 1

                if rc == 0:
                    task.completed()
                else:
                    task.failed()

                task.stdout = _tail_text(stdout_path, None)
                task.stderr = _tail_text(stderr_path, None)

                # workspace_path が分かる場合は成果物一覧を算出
                try:
                    ws = md.get("workspace_path")
                    if isinstance(ws, str) and ws:
                        base = pathlib.Path(ws)
                        if base.exists() and base.is_dir():
                            task.artifacts = [
                                str(p.relative_to(base).as_posix())
                                for p in base.rglob("*")
                                if p.is_file()
                            ]
                except Exception:
                    pass

                TaskManager.upsert_task(task)
                return task

            # While process is still running, return incremental logs (do not persist).
            if running:
                stdout = _tail_text(stdout_path, tail)
                stderr = _tail_text(stderr_path, tail)
                return TaskStatus(
                    **task.model_dump(exclude={"stdout", "stderr"}),
                    stdout=stdout,
                    stderr=stderr,
                )

            # Race condition: process may have exited but `.exit_code` isn't written yet.
            # Do NOT mark the task as failed/exited here; keep it running so callers continue polling.
            stdout = _tail_text(stdout_path, tail)
            stderr = _tail_text(stderr_path, tail)
            wait_msg = "Process exited but exit code file is not yet available. Waiting..."
            if wait_msg not in (stderr or ""):
                stderr = (stderr + "\n" if stderr else "") + wait_msg
            return TaskStatus(
                **task.model_dump(exclude={"stdout", "stderr", "status", "sub_status"}),
                status="running",
                sub_status="starting",
                stdout=stdout,
                stderr=stderr,
            )

        return task

    # --- キャンセル ---
    @classmethod
    async def cancel_task(cls, task_id: str) -> dict[str, Any]:
        """task_id を指定してタスクをキャンセルする（CLI/API共通）。"""
        TaskManager.load_tasks()
        task = TaskManager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        md = task.metadata if isinstance(task.metadata, dict) else {}

        # subprocess backend
        pid_val = md.get("pid")
        if md.get("backend") == "subprocess" and isinstance(pid_val, int):
            pid = pid_val
            if task.status != "running":
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "sub_status": task.sub_status,
                    "message": "Task is not running",
                }
            try:
                kill_process_tree(pid)
                task.cancelled()
                cls.upsert_task(task)
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "sub_status": task.sub_status,
                    "message": "cancelled",
                }
            except ProcessLookupError:
                task.cancelled()
                cls.upsert_task(task)
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "sub_status": task.sub_status,
                    "message": "process not found (already exited)",
                }
            except Exception as e:
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "sub_status": task.sub_status,
                    "message": f"cancel failed: {e}",
                }

        # docker backend: 既に終了している/コンテナIDが無い場合は状態だけ返す
        if task.status != "running" or not task.container_id:
            return {
                "task_id": task.task_id,
                "status": task.status,
                "sub_status": task.sub_status,
                "message": "Task is not running or has no container",
            }

        try:
            whales.container.kill(task.container_id)
            task.cancelled()
            cls.upsert_task(task)
            return {
                "task_id": task.task_id,
                "status": task.status,
                "sub_status": task.sub_status,
                "message": "cancelled",
            }
        except Exception as e:
            return {
                "task_id": task.task_id,
                "status": task.status,
                "sub_status": task.sub_status,
                "message": f"cancel failed: {e}",
            }
    

    @classmethod
    async def run_task(
        cls,
        task_service: AbstractTaskService,
        actions: AbstractActions,
        prompt: str,
        sources: Optional[list[Path]],
        task_id: Optional[str],
        timeout: int = 300,
        wait: bool = True,
    ) -> None:
        """新しいタスクを実行します。"""
        await task_service.prepare(prompt, sources, task_id, extra_env=None)

        async for status in cls._run_(task_service, timeout, wait):
            if status.sub_status in ("running-foreground", "starting"):
                actions.after_start_task_action(status.task_id)
            elif status.sub_status == "running-background":
                actions.after_start_detach_task_action(status.task_id)
                break
            elif status.sub_status == "completed":
                actions.after_complete_action(task_service.get_agent_runner())
                break
            elif status.sub_status in ("failed", "timeout", "cancelled"):
                break

            await actions.progress_action(status.task_id)


    @classmethod
    async def _monitor_wrapper(cls, task_service: AbstractTaskService, timeout: int):
        """Wrapper to consume the async generator from monitor"""
        async for task in task_service.monitor(timeout):
            cls.upsert_task(task)

    @classmethod
    async def _run_(
        cls,
        task_service: AbstractTaskService,
        timeout: int,
        wait: bool,
    ) -> AsyncGenerator[TaskStatus, None]:
        """タスクの開始、監視、完了後の同期までを一括管理"""
        task_status = task_service.start(wait=wait, timeout=timeout)
        cls.upsert_task(task_status)

        if not wait:
            task_service.spawn_detached_monitor(task_status.task_id, timeout)
            yield task_status
            return

        # wait=True: monitor はバックグラウンドで最終状態へ収束させる
        asyncio.create_task(cls._monitor_wrapper(task_service, timeout))
        yield task_status

        async for status in cls.progress_action(task_status.task_id):
            yield status


    @classmethod
    async def progress_action(cls, tid: str) -> AsyncGenerator[TaskStatus, None]:

            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()
            def handle_interrupt(): stop_event.set()
            try:

                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, handle_interrupt)

                while not stop_event.is_set():
                    status_data = await cls.get_status(tid, tail=1000)
                    yield status_data
                    # 終了判定
                    if status_data.status not in ["running", "pending"]:
                        break
                    
                    await asyncio.sleep(1.5)
                
                # 最終状態を一度取得してから終了
                status_data = await cls.get_status(tid, tail=1000)
                yield status_data
                return

            finally:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)

