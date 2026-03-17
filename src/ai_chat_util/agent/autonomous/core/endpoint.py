import os
import pathlib
from typing import Any, Dict, Optional, Annotated

from fastapi import Body, HTTPException, Path, Query

from .task_service_factory import select_task_service
from .task_manager import TaskManager

from ai_chat_util_base.model.autonomous_agent_util_models import (
    CancelResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthzResponse,
    TaskStatus,
)

from ai_chat_util_base.model.request_headers import RequestHeaders, get_current_request_headers
from ai_chat_util_base.config.autonomous_agent_util_runtime import get_runtime_config
from ai_chat_util_base.autonomous.abstract_endpoint import AutonomousEndPointBase

from ..util.logging import get_application_logger

logger = get_application_logger()

class EndPoint(AutonomousEndPointBase):

    
    def rewrite_workspace_path(self, workspace_path: str) -> str:
        """Rewrite inbound workspace_path based on config.

        This is useful when the supervisor and executor run in different
        environments (host vs container) and see the same workspace under
        different absolute paths.
        """
        try:
            rules = get_runtime_config().paths.workspace_path_rewrites
        except Exception:
            return workspace_path

        if not isinstance(workspace_path, str) or not workspace_path:
            return workspace_path

        raw = workspace_path
        for rule in (rules or []):
            try:
                from_prefix = (rule.from_prefix or "").rstrip("/")
                to_prefix = (rule.to_prefix or "").rstrip("/")
            except Exception:
                continue

            if not from_prefix or not to_prefix:
                continue

            if raw == from_prefix:
                return to_prefix
            if raw.startswith(from_prefix + "/"):
                suffix = raw[len(from_prefix) :]
                return to_prefix + suffix

        return workspace_path

    def validate_workspace_path(self, workspace_path: str) -> pathlib.Path:
        if not isinstance(workspace_path, str) or not workspace_path.strip():
            raise HTTPException(status_code=400, detail="workspace_path is required")

        p = pathlib.Path(workspace_path).expanduser()
        if not p.is_absolute():
            raise HTTPException(status_code=400, detail="workspace_path must be an absolute path")

        # 任意だが、パス注入対策として許可ルート配下に制限できる
        allowed_root = get_runtime_config().paths.executor_allowed_workspace_root
        if allowed_root:
            root = pathlib.Path(allowed_root).expanduser().resolve()
            resolved = p.resolve()
            try:
                if not resolved.is_relative_to(root):
                    raise HTTPException(status_code=403, detail="workspace_path is outside allowed root")
            except AttributeError:
                # Python < 3.9 互換（本PJは 3.11+ 前提だが念のため）
                if str(resolved).startswith(str(root)) is False:
                    raise HTTPException(status_code=403, detail="workspace_path is outside allowed root")

        # workspace はSVが用意するが、無ければ作る
        p.mkdir(parents=True, exist_ok=True)
        return p


    async def healthz(self) -> HealthzResponse:
        """
        ヘルスチェック用エンドポイント。SVはこれを叩いてエージェントが生きているか確認する。
        """
        return HealthzResponse(status="ok")


    async def _execute_main_(
        self,
        wait_for_completion: bool,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（同期版/非同期版）の共通処理。SVはこれを叩いてエージェントにタスク実行を指示する。
        引数のwait_for_completionで、タスク完了までHTTPレスポンスを返すかどうかを制御する。
        """
        requested_workspace_path = req.workspace_path
        req.workspace_path = self.rewrite_workspace_path(req.workspace_path)
        workspace_dir = self.validate_workspace_path(req.workspace_path)

        # Inbound headers are captured by:
        # - FastAPI middleware (HTTP API)
        # - MCP tool wrapper (MCP server)
        incoming = get_current_request_headers()

        if incoming and incoming.trace_id and not req.trace_id:
            req.trace_id = incoming.trace_id

        task_service = select_task_service()
        await task_service.prepare(
            prompt=req.prompt,
            sources=None,
            task_id=req.task_id,
            workspace_path=workspace_dir,
            extra_env=(incoming.to_env() if incoming else None),
        )

        if req.trace_id:
            task_service.get_agent_runner().get_task_status().trace_id = req.trace_id

        # Persist workspace path information for status/artifact recomputation and debugging.
        try:
            st = task_service.get_agent_runner().get_task_status()
            if isinstance(getattr(st, "metadata", None), dict):
                st.metadata.setdefault("workspace_path", workspace_dir.resolve().as_posix())
                st.metadata.setdefault("requested_workspace_path", requested_workspace_path)
                if requested_workspace_path != req.workspace_path:
                    st.metadata.setdefault("rewritten_workspace_path", req.workspace_path)
        except Exception:
            pass

        task_status = task_service.start(wait=wait_for_completion, timeout=req.timeout)
        TaskManager.upsert_task(task_status)

        # For synchronous execution, block until the task converges to a final state.
        # `start(wait=True)` only marks the initial intent (foreground) for the backend;
        # actual convergence is performed by `monitor()`.
        if wait_for_completion:
            async for st in task_service.monitor(timeout=req.timeout):
                TaskManager.upsert_task(st)

        return ExecuteResponse(task_id=task_status.task_id)

    async def execute_sync(
        self,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（同期版）。ユーザーはこれを叩いてエージェントにタスク実行を指示する。
        executeとの違いは、タスク完了までHTTPレスポンを返さない点。小規模タスクやテスト用途向け。
        """
        return await self._execute_main_(wait_for_completion=True, req=req)


    async def execute_async(
        self,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（非同期版）。ユーザーはこれを叩いてエージェントにタスク実行を指示する。
        処理が完了する前にHTTPレスポンスを返す。
        ユーザーは/statusエンドポイントを叩いてタスクの進捗や結果を取得する。   
        キャンセルを行う場合は、/cancelエンドポイントを叩く。
        """
        logger.debug("Received execute request: %s", req)
        return await self._execute_main_(wait_for_completion=False, req=req)

    # タスクが使用しているワークスペースのパスを返すエンドポイント
    async def workspace_path(
        self,
        task_id: Annotated[str, Path(description="task id")],
    ) -> Dict[str, str]:
        """
        タスクのワークスペースパス取得用エンドポイント。SVはこれを叩いてタスクのワークスペースパスを取得する。
        """
        task_status = await TaskManager.get_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"workspace_path": task_status.workspace_path}

    async def status(
        self,
        task_id: Annotated[str, Path(description="task id")],
        tail: Annotated[
            Optional[int],
            Query(description="ログの末尾 n 行（省略時は 200、null で全量）", ge=0),
        ] = 200,
    ) -> TaskStatus:
        """
         タスクステータス取得用エンドポイント。SVはこれを叩いてタスクの進捗や結果を取得する。
         tailはログの末尾n行を取得するためのパラメータ。
         既存のTaskStatus(JSONをそのまま返す。SV側は status/sub_status を見て完了判定する。
        """
        return await TaskManager.get_status(task_id, tail=tail)

    async def get_result(
        self,
        task_id: Annotated[str, Path(description="task id")],
        tail: Annotated[
            Optional[int],
            Query(description="ログの末尾 n 行（省略時は 200、null で全量）", ge=0),
        ] = 200,
    ) -> Dict[str, Any]:
        """
         タスク結果取得用エンドポイント。SVはこれを叩いてタスクの結果を取得する。
         tailはログの末尾n行を取得するためのパラメータ。

         注意: task_id は必須です。execute のレスポンスで返る task_id を指定してください。
        """
        status = await TaskManager.get_status(task_id, tail=tail)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"stdout": status.stdout, "stderr": status.stderr}

    async def cancel(self, task_id: Annotated[str, Path(description="task id")]) -> CancelResponse:
        """
         タスクキャンセル用エンドポイント。SVはこれを叩いてタスクのキャンセルを指示する。
        """
        res: Any = await TaskManager.cancel_task(task_id)
        if isinstance(res, dict):
            return CancelResponse(**res)
        return CancelResponse(message="cancel requested")

